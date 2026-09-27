"""3.6.2 input interventions (model-stream, µV, applied to *padded* windows
then cropped, so the FIR band-stop has context and no 4-s edge artefacts).

SRI (2 × 2: region {posterior, central} × band {alpha, equal-width control}):
    SRI_post = [BA(x) − BA(post-α stop)] − [BA(x) − BA(post-control stop)]
             = BA(post-control stop) − BA(post-α stop)
Central-alpha band-stop is reported alongside (collateral removal check).

HEOG regression: regressor r = x[AF7] − x[AF8] (+ contamination), channel-wise
coefficients by least squares on *training* windows, cleaned x − b·r.
    ΔBA_reg = BA(x) − BA(x_clean)       (matched test set)
Contamination scan (0/low/mid): r += c·(sd_r/sd_m)·(x[C3] − x[C4]) — a regressor
that picks up lateralised sensorimotor activity, i.e. the task signal.
"""
from __future__ import annotations

import numpy as np

from ..data.streams import bandstop
from ..utils.common import balanced_accuracy, ch_index


def crop(Xp: np.ndarray, pad: int, core_len: int) -> np.ndarray:
    return Xp[..., pad:pad + core_len]


def bandstop_channels(Xp: np.ndarray, sfreq: float, ch_idx: np.ndarray, band, trans: float) -> np.ndarray:
    out = np.array(Xp, dtype=np.float64, copy=True)
    out[:, ch_idx, :] = bandstop(out[:, ch_idx, :], sfreq, band[0], band[1], trans)
    return out.astype(np.float32)


def sri(predict, Xp, y, pad, core_len, sfreq, ch_names, cfg) -> dict:
    """predict: callable(X_cropped) -> labels."""
    ic = cfg["intervention"]
    post = ch_index(ch_names, cfg["alpha_component"]["posterior"])
    cent = ch_index(ch_names, cfg["alpha_component"]["central"])
    base = balanced_accuracy(y, predict(crop(Xp, pad, core_len)))
    res = {"ba": base}
    for rname, idx in (("post", post), ("central", cent)):
        for bname, band in (("alpha", ic["alpha_band"]), ("ctrl", ic["control_band"])):
            Xi = bandstop_channels(Xp, sfreq, idx, band, ic["transition_hz"])
            res[f"ba_{rname}_{bname}"] = balanced_accuracy(y, predict(crop(Xi, pad, core_len)))
    res["sri_post"] = res["ba_post_ctrl"] - res["ba_post_alpha"]
    res["sri_central"] = res["ba_central_ctrl"] - res["ba_central_alpha"]
    res["drop_post_alpha"] = base - res["ba_post_alpha"]
    res["drop_central_alpha"] = base - res["ba_central_alpha"]
    return res


def heog_regressor(Xp: np.ndarray, ch_names, contamination: float = 0.0, scale_ref: tuple | None = None):
    i7, i8 = ch_index(ch_names, ["AF7", "AF8"])
    r = Xp[:, i7, :] - Xp[:, i8, :]
    if contamination:
        c3, c4 = ch_index(ch_names, ["C3", "C4"])
        m = Xp[:, c3, :] - Xp[:, c4, :]
        sd_r, sd_m = scale_ref if scale_ref is not None else (r.std(), m.std() + 1e-12)
        r = r + contamination * (sd_r / sd_m) * m
    return r


def fit_heog_regression(Xp_train: np.ndarray, ch_names, contamination: float = 0.0, method: str = "ols",
                        sfreq: float = 200.0, cfg=None, min_events: int = 20):
    """Channel-wise propagation coefficients b of the HEOG regressor.

    method = 'ols'             : least squares over all training samples.  With AF7−AF8
                                 (not a true EOG channel) b is dominated by the EEG background
                                 covariance and misses most of the ocular field.
    method = 'saccade_locked'  : event-related estimate (Croft & Barry-style): detect steps in the
                                 regressor, take post−pre step vectors Δx (all channels) and Δr,
                                 b = ΣΔx·Δr / ΣΔr².  Falls back to OLS if < min_events steps.
    """
    r = heog_regressor(Xp_train, ch_names, contamination)
    i7, i8 = ch_index(ch_names, ["AF7", "AF8"])
    c3, c4 = ch_index(ch_names, ["C3", "C4"])
    scale_ref = (float((Xp_train[:, i7] - Xp_train[:, i8]).std()),
                 float((Xp_train[:, c3] - Xp_train[:, c4]).std()) + 1e-12)
    used = "ols"
    b = None
    if method == "saccade_locked":
        from ..saccades.detect import detect_all
        w = int(round(0.1 * sfreq))
        dX, dR = [], []
        for i in range(len(r)):
            for sc in detect_all(r[i], sfreq, cfg):
                if sc.onset - w < 0 or sc.offset + w > r.shape[1]:
                    continue
                pre, post = slice(sc.onset - w, sc.onset), slice(sc.offset, sc.offset + w)
                dX.append(Xp_train[i][:, post].mean(-1) - Xp_train[i][:, pre].mean(-1))
                dR.append(r[i, post].mean() - r[i, pre].mean())
        if len(dR) >= min_events:
            dX, dR = np.asarray(dX), np.asarray(dR)
            b = dX.T @ dR / (dR @ dR)
            used = "saccade_locked"
    if b is None:
        rc = r - r.mean(-1, keepdims=True)
        Xc = Xp_train - Xp_train.mean(-1, keepdims=True)
        b = np.einsum("nct,nt->c", Xc, rc) / (rc ** 2).sum()
    return {"b": b, "contamination": contamination, "scale_ref": scale_ref, "method": used}


def apply_heog_regression(Xp: np.ndarray, ch_names, reg: dict) -> np.ndarray:
    r = heog_regressor(Xp, ch_names, reg["contamination"], reg["scale_ref"])
    r = r - r.mean(-1, keepdims=True)
    return (Xp - reg["b"][None, :, None] * r[:, None, :]).astype(np.float32)


def delta_ba_reg(predict, Xp, y, pad, core_len, ch_names, reg: dict) -> dict:
    base = balanced_accuracy(y, predict(crop(Xp, pad, core_len)))
    clean = balanced_accuracy(y, predict(crop(apply_heog_regression(Xp, ch_names, reg), pad, core_len)))
    return {"ba": base, "ba_regressed": clean, "delta_ba_reg": base - clean}
