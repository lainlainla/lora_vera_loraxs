from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    in_features: int
    out_features: int


def roberta_specs(width: int, intermediate: int, setting: str) -> list[ModuleSpec]:
    square = [
        ModuleSpec("query", width, width),
        ModuleSpec("value", width, width),
    ]
    if setting == "same-target":
        return square
    if setting == "paper-style":
        return square + [
            ModuleSpec("attention_output", width, width),
            ModuleSpec("fc1", width, intermediate),
        ]
    if setting == "all-transformer":
        return [
            ModuleSpec("query", width, width),
            ModuleSpec("key", width, width),
            ModuleSpec("value", width, width),
            ModuleSpec("attention_output", width, width),
            ModuleSpec("fc1", width, intermediate),
            ModuleSpec("fc2", intermediate, width),
        ]
    raise ValueError(f"unknown setting {setting}")


def lora_params(modules: list[ModuleSpec], layers: int, rank: int) -> int:
    return layers * sum(rank * (module.in_features + module.out_features) for module in modules)


def vera_params(modules: list[ModuleSpec], layers: int, rank: int) -> int:
    return layers * sum(module.out_features + rank for module in modules)


def loraxs_params(modules: list[ModuleSpec], layers: int, rank: int) -> int:
    return layers * len(modules) * rank * rank


def tinylora_params(modules: list[ModuleSpec], layers: int, projection_dim: int, tie_every: int) -> int:
    adapted_modules = layers * len(modules)
    return math.ceil(adapted_modules / max(1, tie_every)) * projection_dim


def tinylora_tie_cases(modules: list[ModuleSpec], layers: int, requested_tie_every: int) -> list[tuple[str, int | None]]:
    modules_per_layer = len(modules)
    adapted_modules = layers * modules_per_layer
    cases: list[tuple[str, int | None]] = [
        ("zero_update_baseline", None),
        ("full_model_tie", adapted_modules),
        ("per_layer_tie", modules_per_layer),
        ("no_tie", 1),
    ]
    if requested_tie_every not in {1, modules_per_layer, adapted_modules}:
        cases.append((f"custom_tie_{requested_tie_every}", requested_tie_every))
    return cases


def rows(args: argparse.Namespace) -> list[dict[str, str | int]]:
    modules = roberta_specs(args.width, args.intermediate, args.setting)
    ranks = [int(item) for item in args.ranks.split(",") if item]
    tie_cases = tinylora_tie_cases(modules, args.layers, args.tie_every)

    output: list[dict[str, str | int]] = []
    for rank in ranks:
        output.append(
            {
                "method": "LoRA",
                "setting": args.setting,
                "layers": args.layers,
                "modules_per_layer": len(modules),
                "rank": rank,
                "u": "",
                "tie_case": "",
                "tie_every": "",
                "tie_groups": "",
                "trainable_params": lora_params(modules, args.layers, rank),
            }
        )
        output.append(
            {
                "method": "VeRA",
                "setting": args.setting,
                "layers": args.layers,
                "modules_per_layer": len(modules),
                "rank": rank,
                "u": "",
                "tie_case": "",
                "tie_every": "",
                "tie_groups": "",
                "trainable_params": vera_params(modules, args.layers, rank),
            }
        )
        output.append(
            {
                "method": "LoRA-XS",
                "setting": args.setting,
                "layers": args.layers,
                "modules_per_layer": len(modules),
                "rank": rank,
                "u": "",
                "tie_case": "",
                "tie_every": "",
                "tie_groups": "",
                "trainable_params": loraxs_params(modules, args.layers, rank),
            }
        )
        adapted_modules = args.layers * len(modules)
        for tie_case, tie_every in tie_cases:
            trainable_params = 0 if tie_every is None else tinylora_params(modules, args.layers, args.tiny_u, tie_every)
            output.append(
                {
                    "method": "TinyLoRA",
                    "setting": args.setting,
                    "layers": args.layers,
                    "modules_per_layer": len(modules),
                    "rank": rank,
                    "u": args.tiny_u,
                    "tie_case": tie_case,
                    "tie_every": "" if tie_every is None else tie_every,
                    "tie_groups": 0 if tie_every is None else math.ceil(adapted_modules / max(1, tie_every)),
                    "trainable_params": trainable_params,
                }
            )
    return output


def print_markdown(table: list[dict[str, str | int]]) -> None:
    headers = [
        "method",
        "setting",
        "layers",
        "modules_per_layer",
        "rank",
        "u",
        "tie_case",
        "tie_every",
        "tie_groups",
        "trainable_params",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in table:
        print("| " + " | ".join(str(row[header]) for header in headers) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute adapter parameter trade-offs.")
    parser.add_argument("--setting", choices=["same-target", "paper-style", "all-transformer"], default="paper-style")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--ranks", default="1,2,4,8,16,64,256")
    parser.add_argument("--tiny-u", type=int, default=1)
    parser.add_argument("--tie-every", type=int, default=1)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    table = rows(args)
    print_markdown(table)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)


if __name__ == "__main__":
    main()
