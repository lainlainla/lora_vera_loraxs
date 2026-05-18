from __future__ import annotations

from .base import AdapterModelSetting


TINY_QV_FULL_TIE_U1_R2 = AdapterModelSetting(
    name="tinylora_qv_full_tie_u1_r2",
    method="tinylora",
    target_modules=("query", "value"),
    rank=2,
    basis="svd_top",
    tiny_projection_dim=1,
    tiny_tie_every=24,
    tags=("tinylora", "same-target", "full-model-tie", "ultra-low-param"),
    notes="For RoBERTa-base q/v with 12 layers and 2 modules per layer.",
)

TINY_QV_PER_LAYER_TIE_U1_R2 = AdapterModelSetting(
    name="tinylora_qv_per_layer_tie_u1_r2",
    method="tinylora",
    target_modules=("query", "value"),
    rank=2,
    basis="svd_top",
    tiny_projection_dim=1,
    tiny_tie_every=2,
    tags=("tinylora", "same-target", "per-layer-tie"),
)

TINY_QV_NO_TIE_U1_R2 = AdapterModelSetting(
    name="tinylora_qv_no_tie_u1_r2",
    method="tinylora",
    target_modules=("query", "value"),
    rank=2,
    basis="svd_top",
    tiny_projection_dim=1,
    tiny_tie_every=1,
    tags=("tinylora", "same-target", "no-tie"),
)

TINY_QVO_FC1_FULL_TIE_U1_R2 = AdapterModelSetting(
    name="tinylora_qvo_fc1_full_tie_u1_r2",
    method="tinylora",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=2,
    basis="svd_top",
    tiny_projection_dim=1,
    tiny_tie_every=48,
    tags=("tinylora", "full-model-tie", "ultra-low-param"),
    notes="For RoBERTa-base paper-style q/v/o/fc1 with 12 layers and 4 modules per layer.",
)

TINY_QVO_FC1_PER_LAYER_TIE_U1_R2 = AdapterModelSetting(
    name="tinylora_qvo_fc1_per_layer_tie_u1_r2",
    method="tinylora",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=2,
    basis="svd_top",
    tiny_projection_dim=1,
    tiny_tie_every=4,
    tags=("tinylora", "per-layer-tie"),
)

TINY_QVO_FC1_NO_TIE_U1_R2 = AdapterModelSetting(
    name="tinylora_qvo_fc1_no_tie_u1_r2",
    method="tinylora",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=2,
    basis="svd_top",
    tiny_projection_dim=1,
    tiny_tie_every=1,
    tags=("tinylora", "no-tie"),
)

TINY_QVO_FC1_TRAINABLE_P_U1_R2 = AdapterModelSetting(
    name="tinylora_qvo_fc1_trainable_p_u1_r2",
    method="tinylora",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=2,
    basis="svd_top",
    tiny_projection_dim=1,
    tiny_tie_every=1,
    tiny_trainable_projection=True,
    tags=("tinylora", "trainable-projection-ablation"),
)

ALL_TINYLORA_SETTINGS = (
    TINY_QV_FULL_TIE_U1_R2,
    TINY_QV_PER_LAYER_TIE_U1_R2,
    TINY_QV_NO_TIE_U1_R2,
    TINY_QVO_FC1_FULL_TIE_U1_R2,
    TINY_QVO_FC1_PER_LAYER_TIE_U1_R2,
    TINY_QVO_FC1_NO_TIE_U1_R2,
    TINY_QVO_FC1_TRAINABLE_P_U1_R2,
)
