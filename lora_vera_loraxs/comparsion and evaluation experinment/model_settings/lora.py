from __future__ import annotations

from .base import AdapterModelSetting


LORA_QV_R8 = AdapterModelSetting(
    name="lora_qv_r8",
    method="lora",
    target_modules=("query", "value"),
    rank=8,
    tags=("glue", "same-target", "paper-baseline"),
    notes="Classic LoRA baseline used on query and value projections.",
)

LORA_QV_R4 = AdapterModelSetting(
    name="lora_qv_r4",
    method="lora",
    target_modules=("query", "value"),
    rank=4,
    tags=("glue", "same-target", "rank-sweep"),
)

LORA_QV_R16 = AdapterModelSetting(
    name="lora_qv_r16",
    method="lora",
    target_modules=("query", "value"),
    rank=16,
    tags=("glue", "same-target", "rank-sweep"),
)

LORA_QVO_FC1_R8 = AdapterModelSetting(
    name="lora_qvo_fc1_r8",
    method="lora",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=8,
    tags=("glue", "coverage-matched"),
    notes="Coverage-matched LoRA control for q/v/o/fc1.",
)

ALL_LORA_SETTINGS = (
    LORA_QV_R4,
    LORA_QV_R8,
    LORA_QV_R16,
    LORA_QVO_FC1_R8,
)
