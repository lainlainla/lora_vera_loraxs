from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_settings import ALL_EXPERIMENT_SETTINGS  # noqa: E402
from model_settings import ALL_MODEL_SETTINGS  # noqa: E402


def print_model_settings() -> None:
    print("| name | method | rank | targets | basis | tiny_u | tiny_tie_every | tags |")
    print("|---|---|---:|---|---|---:|---:|---|")
    for setting in ALL_MODEL_SETTINGS:
        print(
            "| {name} | {method} | {rank} | {targets} | {basis} | {tiny_u} | {tie_every} | {tags} |".format(
                name=setting.name,
                method=setting.method,
                rank=setting.rank,
                targets=",".join(setting.target_modules),
                basis=setting.basis,
                tiny_u=setting.tiny_projection_dim,
                tie_every=setting.tiny_tie_every,
                tags=",".join(setting.tags),
            )
        )


def print_experiment_settings() -> None:
    print("| name | task_family | base_model | tasks | model_settings | seeds |")
    print("|---|---|---|---|---|---|")
    for setting in ALL_EXPERIMENT_SETTINGS:
        print(
            "| {name} | {family} | {base} | {tasks} | {models} | {seeds} |".format(
                name=setting.name,
                family=setting.task_family,
                base=setting.base_model,
                tasks=",".join(setting.tasks),
                models=",".join(setting.model_settings),
                seeds=",".join(str(seed) for seed in setting.seeds),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="List model-level and experiment-level settings.")
    parser.add_argument("--kind", choices=["models", "experiments", "all"], default="all")
    args = parser.parse_args()

    if args.kind in {"models", "all"}:
        print_model_settings()
    if args.kind == "all":
        print()
    if args.kind in {"experiments", "all"}:
        print_experiment_settings()


if __name__ == "__main__":
    main()
