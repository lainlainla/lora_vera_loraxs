from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AdapterMethod = Literal["lora", "vera", "loraxs", "tinylora"]


@dataclass(frozen=True)
class AdapterModelSetting:
    """One method-level setting that can be reused across experiments."""

    name: str
    method: AdapterMethod
    target_modules: tuple[str, ...]
    rank: int
    alpha: float | None = None
    basis: str = "svd_top"
    seed: int = 0
    tiny_projection_dim: int = 1
    tiny_tie_every: int = 1
    tiny_trainable_projection: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def adapter_kwargs(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target_modules": self.target_modules,
            "rank": self.rank,
            "alpha": self.alpha,
            "basis": self.basis,
            "seed": self.seed,
            "tiny_projection_dim": self.tiny_projection_dim,
            "tiny_tie_every": self.tiny_tie_every,
            "tiny_trainable_projection": self.tiny_trainable_projection,
        }
