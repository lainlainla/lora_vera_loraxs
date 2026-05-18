from __future__ import annotations

from .base import AdapterModelSetting
from .lora import ALL_LORA_SETTINGS
from .loraxs import ALL_LORAXS_SETTINGS
from .tinylora import ALL_TINYLORA_SETTINGS
from .vera import ALL_VERA_SETTINGS


ALL_MODEL_SETTINGS: tuple[AdapterModelSetting, ...] = (
    *ALL_LORA_SETTINGS,
    *ALL_VERA_SETTINGS,
    *ALL_LORAXS_SETTINGS,
    *ALL_TINYLORA_SETTINGS,
)

MODEL_SETTING_REGISTRY = {setting.name: setting for setting in ALL_MODEL_SETTINGS}


def get_model_setting(name: str) -> AdapterModelSetting:
    try:
        return MODEL_SETTING_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_SETTING_REGISTRY))
        raise KeyError(f"unknown model setting {name!r}; available: {available}") from exc
