"""3.4.4 step-component template by EOG regression on pilot subjects (P6, 2026-09-19 revision).

Replaces the split-half ICA construct (saccade_ica.py) after it proved unmeasurable on this
dataset (two defensible selection rules agreed on 3/15 pilot subjects).  Decided by Yu Gao,
supervisor not yet reviewed; see PILOT_LOG 10.5 and configs/default.yaml saccade.template_method.

Per pilot subject (same six lateral runs, same 0.1 Hz high-pass as saccade_ica):
  1. detect horizontal saccades on HEOG = AF7 − AF8 with the project detector (detect_all);
     no side labels and no cue times are used;
  2. per event, step on every channel: mean(x[off+g : off+g+w]) − mean(x[on−g−w : on−g]),
     w = saccade_detection.step_window_ms, g = regression.guard_ms;
  3. regress each channel's step on [ΔHEOG, ΔVEOG] over events (no intercept); the HEOG
     coefficient vector is the propagation template (Croft & Barry 2000).  VEOG = mean(Fp1, Fp2)
     enters only to keep blink/vertical leakage out of the HEOG coefficients;
  4. reliability: events alternately split into two halves (time order); template per half;
     split-half r = corr(b0, b1).  Pass iff each half has >= min_events_per_half events and
     r >= min_split_half.  Template = mean of the two unit-norm halves.
  5. feasibility: unchanged P6 rule (> max_failed_subjects failures -> infeasible).
"""
from __future__ import annotations

import numpy as np

from ..data.streams import fir
from ..saccades.detect import detect_all, heog
from ..utils.common import ch_index
from .saccade_ica import HalfResult, SaccadeTemplate


def event_steps(run, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns per-event channel steps (n_ev, n_ch), HEOG steps, VEOG steps."""
    sc, rg = cfg["saccade"], cfg["saccade"]["regression"]
    sf = run.sfreq
    x = fir(run.data, sf, sc["highpass"], None)
    h = heog(run, cfg)
    iv = ch_index(run.ch_names, cfg["streams"]["eye"]["veog"])
    v = x[iv].mean(0)
    w = max(2, int(round(cfg["saccade_detection"]["step_window_ms"] / 1000 * sf)))
    g = int(round(rg["guard_ms"] / 1000 * sf))
    D, dh, dv = [], [], []
    for s in detect_all(h, sf, cfg):
        a0, a1 = s.onset - g - w, s.onset - g
        b0, b1 = s.offset + g, s.offset + g + w
        if a0 < 0 or b1 > x.shape[1]:
            continue
        D.append(x[:, b0:b1].mean(1) - x[:, a0:a1].mean(1))
        dh.append(h[b0:b1].mean() - h[a0:a1].mean())
        dv.append(v[b0:b1].mean() - v[a0:a1].mean())
    n = len(D)
    return (np.asarray(D).reshape(n, x.shape[0]), np.asarray(dh), np.asarray(dv))


def propagation(D: np.ndarray, dh: np.ndarray, dv: np.ndarray) -> np.ndarray:
    A = np.column_stack([dh, dv])
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)      # (2, n_ch)
    return coef[0]


def template_for_subject(subject: int, runs, cfg, seed: int = 0) -> SaccadeTemplate:
    rg = cfg["saccade"]["regression"]
    parts = [event_steps(r, cfg) for r in runs]
    D = np.concatenate([p[0] for p in parts]); dh = np.concatenate([p[1] for p in parts])
    dv = np.concatenate([p[2] for p in parts])
    half = np.arange(dh.size) % 2
    res, pats = [], []
    for k in (0, 1):
        m = half == k
        n = int(m.sum())
        if n < rg["min_events_per_half"]:
            res.append(HalfResult(False, None, float("nan"), float(rg["min_events_per_half"]), n,
                                  f"only {n} events (< {rg['min_events_per_half']})"))
            continue
        b = propagation(D[m], dh[m], dv[m])
        pats.append(b)
        res.append(HalfResult(True, b, float(n), float(rg["min_events_per_half"]), n))
    if len(pats) < 2:
        return SaccadeTemplate(subject, False, None, float("nan"), res)
    p0, p1 = (p / np.linalg.norm(p) for p in pats)
    r_sh = float(np.corrcoef(p0, p1)[0, 1])
    ok = r_sh >= rg["min_split_half"]
    if not ok:
        for h in res:
            h.ok = False; h.reason = f"split-half r {r_sh:.2f} < {rg['min_split_half']}"
    return SaccadeTemplate(subject, bool(ok), (p0 + p1) / 2 if ok else None, r_sh, res)
