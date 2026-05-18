from __future__ import annotations

import time

import torch
from torch import nn

from .basis import random_orthogonal_basis, svd_basis
from .common import AdapterLinearBase, BasisKind, check_rank


class LoRAXSLinear(AdapterLinearBase):
    """LoRA-XS: frozen SVD/random basis with a trainable R matrix."""

    adapter_name = "loraxs"

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float | None = None,
        basis: BasisKind = "svd_top",
        seed: int = 0,
    ) -> None:
        check_rank(rank, linear.out_features, linear.in_features)
        started = time.perf_counter()
        super().__init__(linear, alpha=alpha if alpha is not None else 1.0)
        self.rank = rank
        self.basis = basis
        if basis in {"svd_top", "svd_bottom"}:
            left, right = svd_basis(linear.weight, rank, basis=basis)
        elif basis == "random_orthogonal":
            left, right = random_orthogonal_basis(
                linear.out_features,
                linear.in_features,
                rank,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
                seed=seed,
            )
        else:
            raise ValueError(f"unsupported basis {basis}")
        self.register_buffer("left_basis", left)
        self.register_buffer("right_basis", right)
        self.R = nn.Parameter(torch.zeros(rank, rank, device=linear.weight.device, dtype=linear.weight.dtype))
        self.svd_time_seconds = time.perf_counter() - started if basis.startswith("svd") else 0.0

    def delta_weight(self) -> torch.Tensor:
        return self.left_basis @ self.R @ self.right_basis
