from __future__ import annotations

from .base import ExperimentSetting


GLUE_COVERAGE_MATCHED = ExperimentSetting(
    name="glue_coverage_matched",
    task_family="glue",
    base_model="roberta-base",
    tasks=("sst2", "mrpc", "cola", "qnli"),
    model_settings=(
        "lora_qvo_fc1_r8",
        "vera_qvo_fc1_r256",
        "loraxs_qvo_fc1_r16",
    ),
    seeds=(1,),
    notes="Control experiment for target-module coverage confounding.",
)
