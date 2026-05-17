from __future__ import annotations

from .base import AdapterModelSetting


VERA_QV_R256 = AdapterModelSetting(
    name="vera_qv_r256",
    method="vera",
    target_modules=("query", "value"),
    rank=256,
    tags=("glue", "same-target", "paper-baseline"),
    notes="VeRA parameter-sharing baseline on query/value.",
)

VERA_QV_R64 = AdapterModelSetting(
    name="vera_qv_r64",
    method="vera",
    target_modules=("query", "value"),
    rank=64,
    tags=("glue", "same-target", "rank-sweep"),
)

VERA_QV_R128 = AdapterModelSetting(
    name="vera_qv_r128",
    method="vera",
    target_modules=("query", "value"),
    rank=128,
    tags=("glue", "same-target", "rank-sweep"),
)

VERA_QVO_FC1_R256 = AdapterModelSetting(
    name="vera_qvo_fc1_r256",
    method="vera",
    target_modules=("query", "value", "output.dense", "intermediate.dense", "attention_output", "fc1"),
    rank=256,
    tags=("glue", "coverage-matched"),
    notes="Coverage-matched VeRA control for q/v/o/fc1.",
)

ALL_VERA_SETTINGS = (
    VERA_QV_R64,
    VERA_QV_R128,
    VERA_QV_R256,
    VERA_QVO_FC1_R256,
)
