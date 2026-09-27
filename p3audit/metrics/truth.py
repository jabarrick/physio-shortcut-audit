"""3.4.6 ground truth and 3.8 reliability.

ΔBA_neu = BA(matched) − BA(neutral),  ΔBA_rev = BA(matched) − BA(reversed).
The three test sets share background windows and labels; only z differs.
Upper bounds for a fully confound-reliant model: p − 0.5 and 2p − 1.

Reliability is computed for the unit ("model × cell × seed") by splitting
background windows within each test subject into two halves (window-ID
parity), computing the quantity per subject in each half, correlating the
halves across subjects and applying Spearman–Brown.  Truth and metric are
cross-fitted on opposite halves and the two directions averaged.
"""
from __future__ import annotations

import numpy as np

from ..utils.common import balanced_accuracy


def ba_by(y, pred, mask=None) -> float:
    if mask is not None:
        y, pred = y[mask], pred[mask]
    return balanced_accuracy(y, pred) if y.size else float("nan")


def delta_ba(y, pred_matched, pred_other, mask=None) -> float:
    return ba_by(y, pred_matched, mask) - ba_by(y, pred_other, mask)


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")


def per_subject_halves(values_fn, subj: np.ndarray, half: np.ndarray) -> np.ndarray:
    """values_fn(mask) -> scalar.  Returns array (n_subjects, 2)."""
    out = []
    for s in np.unique(subj):
        out.append([values_fn((subj == s) & (half == h)) for h in (0, 1)])
    return np.asarray(out, dtype=float)


def split_half_reliability(halves: np.ndarray) -> dict:
    ok = np.all(np.isfinite(halves), axis=1)
    if ok.sum() < 3:
        return {"r_half": float("nan"), "reliability": float("nan"), "n": int(ok.sum())}
    r = float(np.corrcoef(halves[ok, 0], halves[ok, 1])[0, 1])
    return {"r_half": r, "reliability": spearman_brown(r), "n": int(ok.sum())}


def unit_truth(y, preds: dict[str, np.ndarray], subj, half) -> dict:
    """preds: {'matched','neutral','reversed'} -> predicted labels (window-aligned)."""
    out = {"ba_matched": ba_by(y, preds["matched"])}
    for other, tag in (("neutral", "neu"), ("reversed", "rev")):
        if other not in preds:
            continue
        out[f"ba_{other}"] = ba_by(y, preds[other])
        out[f"delta_ba_{tag}"] = delta_ba(y, preds["matched"], preds[other])
        for h in (0, 1):
            out[f"delta_ba_{tag}_half{h}"] = delta_ba(y, preds["matched"], preds[other], half == h)
        halves = per_subject_halves(lambda m: delta_ba(y, preds["matched"], preds[other], m), subj, half)
        rel = split_half_reliability(halves)
        out[f"delta_ba_{tag}_reliability"] = rel["reliability"]
    return out


def disattenuated_corr(r_xy: float, rel_x: float, rel_y: float) -> float:
    """Correction for attenuation, capped at 1 (3.8)."""
    if not (np.isfinite(rel_x) and np.isfinite(rel_y)) or rel_x <= 0 or rel_y <= 0:
        return float("nan")
    return float(np.clip(r_xy / np.sqrt(rel_x * rel_y), -1.0, 1.0))
