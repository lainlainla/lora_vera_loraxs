from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSetting:
    """Experiment-level setting composed from method-level model settings."""

    name: str
    task_family: str
    base_model: str
    tasks: tuple[str, ...]
    model_settings: tuple[str, ...]
    seeds: tuple[int, ...] = (1,)
    notes: str = ""
