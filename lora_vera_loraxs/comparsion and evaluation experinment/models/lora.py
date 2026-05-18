from __future__ import annotations

import math

import torch
from torch import nn

from .common import AdapterLinearBase, check_rank


class LoRALinear(AdapterLinearBase):
    """Classic LoRA: Delta W = B @ A for an nn.Linear weight."""

    adapter_name = "lora"

    def __init__(self, linear: nn.Linear, rank: int, alpha: float | None = None) -> None:
        check_rank(rank, linear.out_features, linear.in_features)
        super().__init__(linear, alpha=alpha if alpha is not None else 1.0 / rank)
        self.rank = rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, linear.in_features, device=linear.weight.device, dtype=linear.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(linear.out_features, rank, device=linear.weight.device, dtype=linear.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def delta_weight(self) -> torch.Tensor:
        return self.lora_B @ self.lora_A
