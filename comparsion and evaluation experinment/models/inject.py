from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import nn

from .lora import LoRALinear
from .loraxs import LoRAXSLinear
from .tinylora import TinyLoRALinear
from .vera import VeRALinear


AdapterMethod = Literal["lora", "vera", "loraxs", "tinylora"]


@dataclass
class AdapterRecord:
    name: str
    method: str
    rank: int
    trainable_parameters: int
    svd_time_seconds: float = 0.0


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _get_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _matches(name: str, targets: Iterable[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    for target in targets:
        if name == target or leaf == target or name.endswith("." + target):
            return True
    return False


def apply_adapter(
    model: nn.Module,
    *,
    method: AdapterMethod,
    target_modules: Iterable[str],
    rank: int,
    alpha: float | None = None,
    freeze_backbone: bool = True,
    basis: str = "svd_top",
    seed: int = 0,
    tiny_projection_dim: int = 1,
    tiny_tie_every: int = 1,
    tiny_trainable_projection: bool = False,
) -> list[AdapterRecord]:
    """Replace selected nn.Linear modules with adapter wrappers.

    target_modules can contain exact qualified names ("encoder.layer.0...query")
    or leaf names ("query", "value"). For TinyLoRA, tiny_tie_every controls how
    many adapted modules share one trainable vector v.
    """

    if freeze_backbone:
        freeze_model(model)
    targets = tuple(target_modules)
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches(name, targets)
    ]
    if not selected:
        raise ValueError(f"no nn.Linear modules matched targets={targets}")

    records: list[AdapterRecord] = []
    tiny_shared_vectors: dict[int, nn.Parameter] = {}
    tie_every = max(1, int(tiny_tie_every))

    for index, (name, linear) in enumerate(selected):
        if method == "lora":
            wrapped = LoRALinear(linear, rank=rank, alpha=alpha)
        elif method == "vera":
            wrapped = VeRALinear(linear, rank=rank, alpha=alpha, seed=seed + index)
        elif method == "loraxs":
            wrapped = LoRAXSLinear(linear, rank=rank, alpha=alpha, basis=basis, seed=seed + index)
        elif method == "tinylora":
            group = index // tie_every
            shared_vector = tiny_shared_vectors.get(group)
            if shared_vector is None:
                shared_vector = nn.Parameter(
                    torch.zeros(tiny_projection_dim, device=linear.weight.device, dtype=linear.weight.dtype)
                )
                tiny_shared_vectors[group] = shared_vector
            wrapped = TinyLoRALinear(
                linear,
                rank=rank,
                projection_dim=tiny_projection_dim,
                alpha=alpha,
                basis=basis,
                seed=seed + index,
                shared_vector=shared_vector,
                trainable_projection=tiny_trainable_projection,
            )
        else:
            raise ValueError(f"unsupported adapter method {method}")

        parent, child_name = _get_parent(model, name)
        setattr(parent, child_name, wrapped)
        records.append(
            AdapterRecord(
                name=name,
                method=method,
                rank=rank,
                trainable_parameters=wrapped.adapter_trainable_parameters(),
                svd_time_seconds=float(getattr(wrapped, "svd_time_seconds", 0.0)),
            )
        )
    return records
