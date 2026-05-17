from __future__ import annotations

import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from model_settings import get_model_setting
from models.inject import apply_adapter, count_parameters, count_trainable_parameters


GLUE_TEXT_FIELDS: dict[str, tuple[str, str | None]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass(frozen=True)
class GlueRunConfig:
    task: str = "sst2"
    model_setting_name: str = "lora_qv_r8"
    base_model_name: str = "roberta-base"
    output_dir: str = "./colab_outputs"
    seed: int = 1
    max_length: int = 128
    learning_rate: float = 2e-4
    classifier_learning_rate: float | None = None
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    weight_decay: float = 0.0
    max_train_samples: int | None = 2000
    max_eval_samples: int | None = 1000
    train_classifier_head: bool = True
    fp16: bool = True
    bf16: bool = False
    report_to: str = "none"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_count_for_task(task: str) -> int:
    return 1 if task == "stsb" else 3 if task == "mnli" else 2


def validation_split_for_task(task: str) -> str:
    return "validation_matched" if task == "mnli" else "validation"


def tokenize_glue_dataset(dataset: Any, tokenizer: Any, task: str, max_length: int) -> Any:
    field_a, field_b = GLUE_TEXT_FIELDS[task]

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        if field_b is None:
            return tokenizer(batch[field_a], truncation=True, max_length=max_length)
        return tokenizer(batch[field_a], batch[field_b], truncation=True, max_length=max_length)

    tokenized = dataset.map(preprocess, batched=True)
    keep_columns = {"input_ids", "attention_mask", "label"}
    if "token_type_ids" in tokenized["train"].column_names:
        keep_columns.add("token_type_ids")
    remove_columns = [column for column in tokenized["train"].column_names if column not in keep_columns]
    return tokenized.remove_columns(remove_columns)


def unfreeze_classifier_head(model: torch.nn.Module) -> int:
    unfrozen = 0
    for name, parameter in model.named_parameters():
        if name.startswith(("classifier.", "score.", "regressor.")) or ".classifier." in name:
            parameter.requires_grad = True
            unfrozen += parameter.numel()
    return unfrozen


def build_compute_metrics(task: str):
    import evaluate

    metric = evaluate.load("glue", task)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        predictions, labels = eval_pred
        if task == "stsb":
            predictions = np.squeeze(predictions)
        else:
            predictions = np.argmax(predictions, axis=-1)
        metrics = metric.compute(predictions=predictions, references=labels)
        return {key: float(value) for key, value in metrics.items()}

    return compute_metrics


def make_training_arguments(config: GlueRunConfig, run_dir: Path) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir),
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "weight_decay": config.weight_decay,
        "logging_steps": 25,
        "save_strategy": "no",
        "report_to": config.report_to,
        "fp16": bool(config.fp16 and torch.cuda.is_available()),
        "bf16": bool(config.bf16 and torch.cuda.is_available()),
        "seed": config.seed,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def run_glue_experiment(config: GlueRunConfig) -> dict[str, Any]:
    """Run one real GLUE fine-tuning job from a public Hugging Face dataset."""

    if config.task not in GLUE_TEXT_FIELDS:
        raise ValueError(f"unsupported GLUE task {config.task!r}; choose from {sorted(GLUE_TEXT_FIELDS)}")

    set_seed(config.seed)
    run_id = f"{config.task}_{config.model_setting_name}_seed{config.seed}"
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    raw_dataset = load_dataset("glue", config.task)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, use_fast=True)
    tokenized = tokenize_glue_dataset(raw_dataset, tokenizer, config.task, config.max_length)

    train_dataset = tokenized["train"]
    eval_dataset = tokenized[validation_split_for_task(config.task)]
    if config.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(config.max_train_samples, len(train_dataset))))
    if config.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(config.max_eval_samples, len(eval_dataset))))

    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name,
        num_labels=label_count_for_task(config.task),
    )
    if config.task == "stsb":
        model.config.problem_type = "regression"

    setting = get_model_setting(config.model_setting_name)
    adapter_records = apply_adapter(model, **setting.adapter_kwargs())
    classifier_params = unfreeze_classifier_head(model) if config.train_classifier_head else 0

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = make_training_arguments(config, run_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(config.task),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    eval_metrics = trainer.evaluate()
    peak_memory_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "run_id": run_id,
        "task": config.task,
        "base_model_name": config.base_model_name,
        "model_setting": asdict(setting),
        "config": asdict(config),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / max(total_params, 1),
        "classifier_trainable_params": classifier_params,
        "adapter_records": [asdict(record) for record in adapter_records],
        "svd_time_seconds": sum(record.svd_time_seconds for record in adapter_records),
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_memory_bytes,
        "eval_metrics": eval_metrics,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary
