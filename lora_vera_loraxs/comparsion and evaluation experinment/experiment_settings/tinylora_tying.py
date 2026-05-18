from __future__ import annotations

from .base import ExperimentSetting


TINYLORA_TYING = ExperimentSetting(
    name="tinylora_tying",
    task_family="toy_or_reasoning",
    base_model="synthetic / qwen2.5-instruct",
    tasks=("toy_matrix_regression", "gsm8k_optional"),
    model_settings=(
        "tinylora_qvo_fc1_full_tie_u1_r2",
        "tinylora_qvo_fc1_per_layer_tie_u1_r2",
        "tinylora_qvo_fc1_no_tie_u1_r2",
        "tinylora_qvo_fc1_trainable_p_u1_r2",
        "loraxs_qvo_fc1_r16",
    ),
    seeds=(1, 2, 3),
    notes="Tracks whether stronger TinyLoRA tying causes sharp performance degradation.",
)
