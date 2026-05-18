from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TinyProjectionModel(nn.Module):
    def __init__(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        projection_dim: int,
        *,
        trainable_projection: bool,
        seed: int,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        rank = left.shape[1]
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        projection = torch.randn(projection_dim, rank, rank, generator=generator) / (rank * rank) ** 0.5
        if trainable_projection:
            self.projection = nn.Parameter(projection)
            self.v = nn.Parameter(0.01 * torch.randn(projection_dim, generator=generator))
        else:
            self.register_buffer("projection", projection)
            self.v = nn.Parameter(torch.zeros(projection_dim))

    def delta(self) -> torch.Tensor:
        r_matrix = torch.einsum("u,uij->ij", self.v, self.projection)
        return self.left @ r_matrix @ self.right.T


class DirectRModel(nn.Module):
    def __init__(self, left: torch.Tensor, right: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("left", left)
        self.register_buffer("right", right)
        self.R = nn.Parameter(torch.zeros(left.shape[1], right.shape[1]))

    def delta(self) -> torch.Tensor:
        return self.left @ self.R @ self.right.T


def train_model(model: nn.Module, target: torch.Tensor, *, steps: int, lr: float) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.delta() - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy convergence: fixed random projection vs trainable projection.")
    parser.add_argument("--out-features", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--u", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=Path("output/toy_convergence.csv"))
    args = parser.parse_args()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    left, _ = torch.linalg.qr(torch.randn(args.out_features, args.rank, generator=generator), mode="reduced")
    right, _ = torch.linalg.qr(torch.randn(args.in_features, args.rank, generator=generator), mode="reduced")
    target_r = torch.randn(args.rank, args.rank, generator=generator)
    target = left @ target_r @ right.T

    models = {
        "fixed_random_projection": TinyProjectionModel(
            left,
            right,
            args.u,
            trainable_projection=False,
            seed=args.seed + 1,
        ),
        "trainable_random_projection": TinyProjectionModel(
            left,
            right,
            args.u,
            trainable_projection=True,
            seed=args.seed + 1,
        ),
        "direct_trainable_R": DirectRModel(left, right),
    }

    histories = {name: train_model(model, target, steps=args.steps, lr=args.lr) for name, model in models.items()}
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *histories.keys()])
        for step in range(args.steps):
            writer.writerow([step + 1, *(histories[name][step] for name in histories)])

    print("Final toy losses")
    for name, history in histories.items():
        trainable = sum(parameter.numel() for parameter in models[name].parameters() if parameter.requires_grad)
        print(f"{name:28s} params={trainable:4d} loss={history[-1]:.6e}")
    print(f"Saved curve CSV to {args.csv}")
    print("Interpretation: trainable random projection usually lowers the attainable loss because it adds capacity.")
    print("Compare it against direct_trainable_R to separate optimization speed from extra parameter count.")


if __name__ == "__main__":
    main()
