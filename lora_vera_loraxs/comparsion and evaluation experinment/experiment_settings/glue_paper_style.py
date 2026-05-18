from __future__ import annotations

from .base import ExperimentSetting


GLUE_PAPER_STYLE = ExperimentSetting(
    name="glue_paper_style",
    task_family="glue",
    base_model="roberta-large",
    tasks=("sst2", "mrpc", "cola", "qnli", "rte", "stsb"),
    model_settings=(
        "lora_qv_r8",
        "vera_qv_r256",
        "loraxs_qvo_fc1_r16",
    ),
    seeds=(1, 2, 3),
    notes="Paper-style comparison where LoRA-XS uses q/v/o/fc1 coverage.",
)
