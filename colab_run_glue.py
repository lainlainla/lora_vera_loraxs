from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_experiment_dir(project_dir: Path | None) -> Path:
    candidates = []
    if project_dir is not None:
        candidates.append(project_dir)
    candidates.extend([Path.cwd(), Path(__file__).resolve().parent])

    for candidate in candidates:
        experiment_dir = candidate / "comparsion and evaluation experinment"
        if experiment_dir.exists():
            return experiment_dir
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Cannot find 'comparsion and evaluation experinment'. Searched: {searched}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab CLI entrypoint for real GLUE adapter experiments.")
    parser.add_argument("--project-dir", type=Path, default=None, help="Directory containing the experiment folder.")
    parser.add_argument("--task", default="sst2", help="GLUE task, e.g. sst2, mrpc, cola, qnli, rte, stsb.")
    parser.add_argument("--base-model", default="roberta-base", help="Hugging Face base model name.")
    parser.add_argument("--model-setting", default="lora_qv_r8", help="Name from model_settings registry.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to experiment output/colab_glue.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int, default=2000, help="Use -1 for the full train split.")
    parser.add_argument("--max-eval-samples", type=int, default=1000, help="Use -1 for the full validation split.")
    parser.add_argument("--no-train-classifier-head", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def none_if_negative(value: int) -> int | None:
    return None if value < 0 else value


def main() -> None:
    args = parse_args()
    experiment_dir = find_experiment_dir(args.project_dir)
    sys.path.insert(0, str(experiment_dir))

    from experiments.colab_glue_runner import GlueRunConfig, run_glue_experiment

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(experiment_dir / "output" / "colab_glue")

    config = GlueRunConfig(
        task=args.task,
        model_setting_name=args.model_setting,
        base_model_name=args.base_model,
        output_dir=output_dir,
        seed=args.seed,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        max_train_samples=none_if_negative(args.max_train_samples),
        max_eval_samples=none_if_negative(args.max_eval_samples),
        train_classifier_head=not args.no_train_classifier_head,
        fp16=not args.no_fp16,
        bf16=args.bf16,
    )
    summary = run_glue_experiment(config)

    compact = {
        "run_id": summary["run_id"],
        "task": summary["task"],
        "method": summary["model_setting"]["method"],
        "setting": summary["model_setting"]["name"],
        "trainable_params": summary["trainable_params"],
        "adapter_trainable_params": summary["adapter_trainable_params"],
        "trainable_ratio": summary["trainable_ratio"],
        "classifier_trainable_params": summary["classifier_trainable_params"],
        "svd_time_seconds": summary["svd_time_seconds"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_memory_gb": summary["peak_memory_bytes"] / 1024**3,
        "eval_metrics": summary["eval_metrics"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
