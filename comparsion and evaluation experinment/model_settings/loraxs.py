from __future__ import annotations

from .base import AdapterModelSetting


LORAXS_QV_R8 = AdapterModelSetting(
    name="loraxs_qv_r8",
    method="loraxs",
    target_modules=("query", "value"),
    rank=8,
    basis="svd_top",
    tags=("glue", "same-target", "rank-sweep"),
)

LORAXS_QV_R16 = AdapterModelSetting(
    name="loraxs_qv_r16",
    method="loraxs",
    target_modules=("query", "value"),
    rank=16,
    basis="svd_top",
    tags=("glue", "same-target", "rank-sweep"),
)

LORAXS_QVO_FC1_R16 = AdapterModelSetting(
    name="loraxs_qvo_fc1_r16",
    method="loraxs",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=16,
    basis="svd_top",
    tags=("glue", "paper-style", "coverage-matched"),
    notes="Paper-style LoRA-XS target coverage.",
)

LORAXS_QVO_FC1_RANDOM_R16 = AdapterModelSetting(
    name="loraxs_qvo_fc1_random_r16",
    method="loraxs",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=16,
    basis="random_orthogonal",
    tags=("basis-ablation", "random"),
)

LORAXS_QVO_FC1_BOTTOM_R16 = AdapterModelSetting(
    name="loraxs_qvo_fc1_bottom_r16",
    method="loraxs",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=16,
    basis="svd_bottom",
    tags=("basis-ablation", "bottom"),
)

ALL_LORAXS_SETTINGS = (
    LORAXS_QV_R8,
    LORAXS_QV_R16,
    LORAXS_QVO_FC1_R16,
    LORAXS_QVO_FC1_RANDOM_R16,
    LORAXS_QVO_FC1_BOTTOM_R16,
)
