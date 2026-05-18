from __future__ import annotations

from .base import ExperimentSetting


GLUE_TINYLORA_TYING = ExperimentSetting(
    name="glue_tinylora_tying",
    task_family="glue",
    base_model="roberta-base",
    tasks=("sst2",),
    model_settings=(
        "tinylora_qv_full_tie_u1_r2",
        "tinylora_qv_per_layer_tie_u1_r2",
        "tinylora_qv_no_tie_u1_r2",
        "tinylora_qvo_fc1_full_tie_u1_r2",
        "tinylora_qvo_fc1_per_layer_tie_u1_r2",
        "tinylora_qvo_fc1_no_tie_u1_r2",
    ),
    seeds=(1,),
    notes="Real GLUE/SST-2 TinyLoRA tying sweep for q/v and q/v/o/fc1 coverage.",
)
