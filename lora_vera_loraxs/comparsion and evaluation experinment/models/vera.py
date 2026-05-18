from __future__ import annotations

import math

import torch
from torch import nn

from .common import AdapterLinearBase, check_rank


class VeRALinear(AdapterLinearBase):
    """VeRA-style wrapper with frozen random bases and trainable scaling."""

    adapter_name = "vera"

    def __init__(self, linear: nn.Linear, rank: int, alpha: float | None = None, seed: int = 0) -> None:
        check_rank(rank, linear.out_features, linear.in_features)
        super().__init__(linear, alpha=alpha if alpha is not None else 1.0 / rank)
        self.rank = rank
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + linear.in_features * 1009 + linear.out_features)
        vera_A = torch.randn(rank, linear.in_features, generator=generator) / math.sqrt(linear.in_features)
        vera_B = torch.randn(linear.out_features, rank, generator=generator) / math.sqrt(rank)
        self.register_buffer("vera_A", vera_A.to(device=linear.weight.device, dtype=linear.weight.dtype))
        self.register_buffer("vera_B", vera_B.to(device=linear.weight.device, dtype=linear.weight.dtype))
        self.vera_b = nn.Parameter(torch.ones(rank, device=linear.weight.device, dtype=linear.weight.dtype))
        self.vera_d = nn.Parameter(torch.zeros(linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype))

    def delta_weight(self) -> torch.Tensor:
        scaled_B = self.vera_d.unsqueeze(1) * self.vera_B
        scaled_A = self.vera_b.unsqueeze(1) * self.vera_A
        return scaled_B @ scaled_A
