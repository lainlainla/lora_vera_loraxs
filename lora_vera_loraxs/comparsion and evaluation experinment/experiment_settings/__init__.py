from __future__ import annotations

from .base import ExperimentSetting
from .glue_coverage_matched import GLUE_COVERAGE_MATCHED
from .glue_paper_style import GLUE_PAPER_STYLE
from .glue_same_target import GLUE_SAME_TARGET
from .tinylora_tying import TINYLORA_TYING


ALL_EXPERIMENT_SETTINGS: tuple[ExperimentSetting, ...] = (
    GLUE_SAME_TARGET,
    GLUE_PAPER_STYLE,
    GLUE_COVERAGE_MATCHED,
    TINYLORA_TYING,
)

EXPERIMENT_SETTING_REGISTRY = {setting.name: setting for setting in ALL_EXPERIMENT_SETTINGS}


def get_experiment_setting(name: str) -> ExperimentSetting:
    try:
        return EXPERIMENT_SETTING_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENT_SETTING_REGISTRY))
        raise KeyError(f"unknown experiment setting {name!r}; available: {available}") from exc
