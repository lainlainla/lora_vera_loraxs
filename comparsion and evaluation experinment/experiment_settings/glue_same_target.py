from __future__ import annotations

from .base import ExperimentSetting


GLUE_SAME_TARGET = ExperimentSetting(
    name="glue_same_target",
    task_family="glue",
    base_model="roberta-base",
    tasks=("sst2", "mrpc", "cola", "qnli"),
    model_settings=(
        "lora_qv_r8",
        "vera_qv_r256",
        "loraxs_qv_r8",
        "loraxs_qv_r16",
    ),
    seeds=(1,),
    notes="Fair comparison where all methods adapt query/value only.",
)
