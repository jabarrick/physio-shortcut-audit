"""Per-subject sufficient statistics so every unit-level quantity can be
recomputed under the hierarchical bootstrap (seeds, then subjects; 3.8).

BA-type quantities are stored as per-subject confusion counts
[n(y=0,ŷ=0), n(y=0,ŷ=1), n(y=1,ŷ=0), n(y=1,ŷ=1)].
"""
from __future__ import annotations

import numpy as np


def confusion_by_subject(y, pred, subj) -> dict[int, list[int]]:
    out = {}
    for s in np.unique(subj):
        m = subj == s
        yy, pp = y[m], pred[m]
        out[int(s)] = [int(((yy == 0) & (pp == 0)).sum()), int(((yy == 0) & (pp == 1)).sum()),
                       int(((yy == 1) & (pp == 0)).sum()), int(((yy == 1) & (pp == 1)).sum())]
    return out


def ba_from_counts(counts: dict, subjects=None) -> float:
    subjects = counts.keys() if subjects is None else subjects
    c = np.zeros(4)
    for s in subjects:
        v = counts.get(s, counts.get(str(s)))
        if v is not None:
            c += np.asarray(v, dtype=float)
    r0 = c[0] / (c[0] + c[1]) if (c[0] + c[1]) else np.nan
    r1 = c[3] / (c[2] + c[3]) if (c[2] + c[3]) else np.nan
    return float(np.nanmean([r0, r1]))


def sums_by_subject(values: np.ndarray, subj: np.ndarray) -> dict[int, list[float]]:
    """values (n, k) -> per-subject column sums (+ count)."""
    values = np.asarray(values, dtype=float).reshape(len(subj), -1)
    return {int(s): values[subj == s].sum(0).tolist() + [int((subj == s).sum())] for s in np.unique(subj)}


def ratio_from_sums(sums: dict, num: int, den: int, subjects=None) -> float:
    subjects = sums.keys() if subjects is None else subjects
    n = d = 0.0
    for s in subjects:
        v = sums.get(s, sums.get(str(s)))
        if v is not None:
            n += v[num]; d += v[den]
    return float(n / d) if d else float("nan")
