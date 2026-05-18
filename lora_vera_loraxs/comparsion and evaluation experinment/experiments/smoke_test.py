from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import apply_adapter, count_trainable_parameters  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(16, 16)
        self.value = nn.Linear(16, 16)
        self.fc1 = nn.Linear(16, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(torch.relu(self.query(x) + self.value(x)))


def run(method: str) -> None:
    torch.manual_seed(0)
    model = TinyBlock()
    records = apply_adapter(
        model,
        method=method,
        target_modules=["query", "value"],
        rank=2,
        tiny_projection_dim=1,
        tiny_tie_every=2,
    )
    output = model(torch.randn(4, 16))
    loss = output.pow(2).mean()
    loss.backward()
    print(method, "records=", len(records), "trainable=", count_trainable_parameters(model), "output=", tuple(output.shape))


def main() -> None:
    for method in ["lora", "vera", "loraxs", "tinylora"]:
        run(method)


if __name__ == "__main__":
    main()
