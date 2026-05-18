from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class TieCase:
    name: str
    tie_every: int | None


class TiedTinyLoRAFit(nn.Module):
    """Fit synthetic per-module R targets with TinyLoRA-style tied vectors."""

    def __init__(
        self,
        *,
        target_r: torch.Tensor,
        projection: torch.Tensor,
        tie_every: int,
    ) -> None:
        super().__init__()
        self.register_buffer("target_r", target_r)
        self.register_buffer("projection", projection)
        self.module_count, self.projection_dim, self.rank, _ = projection.shape
        self.tie_every = max(1, tie_every)
        self.group_count = math.ceil(self.module_count / self.tie_every)
        self.v = nn.Parameter(torch.zeros(self.group_count, self.projection_dim))

    def predicted_r(self) -> torch.Tensor:
        group_ids = torch.arange(self.module_count, device=self.projection.device) // self.tie_every
        module_v = self.v[group_ids]
        return torch.einsum("mu,mujk->mjk", module_v, self.projection)

    def loss(self) -> torch.Tensor:
        return (self.predicted_r() - self.target_r).pow(2).mean()


class DirectRFit(nn.Module):
    """LoRA-XS upper bound in this toy setting: train one R per module."""

    def __init__(self, target_r: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target_r", target_r)
        self.R = nn.Parameter(torch.zeros_like(target_r))

    def loss(self) -> torch.Tensor:
        return (self.R - self.target_r).pow(2).mean()


def tie_cases(layers: int, modules_per_layer: int) -> list[TieCase]:
    module_count = layers * modules_per_layer
    return [
        TieCase("zero_update_baseline", None),
        TieCase("full_model_tie", module_count),
        TieCase("per_layer_tie", modules_per_layer),
        TieCase("no_tie", 1),
    ]


def train(model: nn.Module, *, steps: int, lr: float) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    final_loss = float(model.loss().detach().item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    return final_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="TinyLoRA tying sweep for final-loss changes under fewer parameters.")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--modules-per-layer", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--u", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=Path("output/tinylora_tying_sweep.csv"))
    args = parser.parse_args()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    module_count = args.layers * args.modules_per_layer
    target_r = torch.randn(module_count, args.rank, args.rank, generator=generator)
    projection = torch.randn(module_count, args.u, args.rank, args.rank, generator=generator) / math.sqrt(args.rank * args.rank)

    rows: list[dict[str, str | int | float]] = []
    baseline_loss = float(target_r.pow(2).mean().item())
    rows.append(
        {
            "variant": "TinyLoRA",
            "tie_case": "zero_update_baseline",
            "tie_every": "",
            "tie_groups": 0,
            "trainable_params": 0,
            "final_loss": baseline_loss,
        }
    )

    for case in tie_cases(args.layers, args.modules_per_layer):
        if case.tie_every is None:
            continue
        model = TiedTinyLoRAFit(target_r=target_r, projection=projection, tie_every=case.tie_every)
        final_loss = train(model, steps=args.steps, lr=args.lr)
        rows.append(
            {
                "variant": "TinyLoRA",
                "tie_case": case.name,
                "tie_every": case.tie_every,
                "tie_groups": model.group_count,
                "trainable_params": model.group_count * args.u,
                "final_loss": final_loss,
            }
        )

    direct = DirectRFit(target_r)
    direct_loss = train(direct, steps=args.steps, lr=args.lr)
    rows.append(
        {
            "variant": "LoRA-XS-direct-R",
            "tie_case": "per_module_R",
            "tie_every": 1,
            "tie_groups": module_count,
            "trainable_params": module_count * args.rank * args.rank,
            "final_loss": direct_loss,
        }
    )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("| variant | tie_case | tie_every | tie_groups | trainable_params | final_loss |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {variant} | {tie_case} | {tie_every} | {tie_groups} | {trainable_params} | {final_loss:.6e} |".format(
                **row
            )
        )
    print(f"Saved CSV to {args.csv}")
    print("Use score-vs-params plots on the CSV to detect whether fewer trainable parameters cause a sharp degradation.")


if __name__ == "__main__":
    main()
