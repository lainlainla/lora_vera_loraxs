from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


BasisKind = Literal["svd_top", "svd_bottom", "random_orthogonal"]


def freeze_linear(linear: nn.Linear) -> None:
    for parameter in linear.parameters():
        parameter.requires_grad = False


def check_rank(rank: int, out_features: int, in_features: int) -> None:
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    max_rank = min(out_features, in_features)
    if rank > max_rank:
        raise ValueError(f"rank={rank} exceeds min(out_features, in_features)={max_rank}")


class AdapterLinearBase(nn.Module):
    """Base wrapper that keeps the original linear layer frozen."""

    adapter_name = "base"

    def __init__(self, linear: nn.Linear, alpha: float | None = None) -> None:
        super().__init__()
        self.base = linear
        freeze_linear(self.base)
        self.alpha = alpha
        self._merged = False

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def scaling(self) -> float:
        return 1.0 if self.alpha is None else float(self.alpha)

    def delta_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self._merged:
            return output
        delta = self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        return output + F.linear(x, delta * self.scaling, None)

    @torch.no_grad()
    def merge(self) -> None:
        if self._merged:
            return
        delta = self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.base.weight.add_(delta * self.scaling)
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        delta = self.delta_weight().to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.base.weight.sub_(delta * self.scaling)
        self._merged = False

    def adapter_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
