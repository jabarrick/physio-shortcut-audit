"""Model factory.  `fallback_standin=True` substitutes a small patch transformer for
foundation models whose official code/weights are not configured (dry runs only;
results produced this way must never be reported)."""
from __future__ import annotations

from .base import EEGModel
from .convnets import CSOANetPlaceholder, EEGNet, RawFeatureLinear, ShallowConvNet
from .foundation import build_cbramod, build_labram, build_standin

CONV = {"eegnet": EEGNet, "shallow": ShallowConvNet, "csoanet": CSOANetPlaceholder, "linear_raw": RawFeatureLinear}
FOUNDATION = {"cbramod": build_cbramod, "labram": build_labram}


def build_model(name: str, cfg, n_ch: int, n_times: int, ch_names, fallback_standin: bool = False) -> EEGModel:
    sf = cfg["streams"]["model"]["sfreq"]
    if name in CONV:
        return CONV[name](n_ch, n_times, sfreq=sf)
    if name in FOUNDATION:
        try:
            return FOUNDATION[name](cfg, n_ch, n_times, ch_names)
        except RuntimeError:
            if not fallback_standin:
                raise
            m = build_standin(cfg, n_ch, n_times, ch_names, name=f"{name}-STANDIN")
            m.is_standin = True
            return m
    raise KeyError(name)


def is_foundation(model) -> bool:
    return bool(getattr(model, "uses_raw_scale", False))
