"""3.6.1 encoding: linear probe on penultimate embeddings.

Main protocol: probe decodes the injection indicator z (z ⟂ y in the probe's
data), is fitted on *training-subject* embeddings with the L2 strength chosen
by subject-grouped nested CV on those same subjects, frozen, and evaluated on
test-subject embeddings.  D = (BA − 0.5)/0.5.

Sensitivity protocol (十·1): refit per test subject on that subject's own
windows (window-level folds), using only z (known by construction), never y.
PCA to 16/64/256 dims is a sensitivity analysis.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..utils.common import balanced_accuracy


def D_from_ba(ba: float) -> float:
    return (ba - 0.5) / 0.5


def _pipe(C: float, pca_dim: int | None, n_feat: int):
    steps = [StandardScaler()]
    if pca_dim is not None and pca_dim < n_feat:
        steps.append(PCA(n_components=pca_dim, random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=2000))
    return make_pipeline(*steps)


def fit_probe(E: np.ndarray, z: np.ndarray, groups: np.ndarray, Cs, inner_folds: int = 5,
              pca_dim: int | None = None):
    """Nested (subject-grouped) choice of C, then refit on all training subjects."""
    n_groups = np.unique(groups).size
    k = min(inner_folds, n_groups)
    best_C, best = Cs[0], -np.inf
    if k >= 2:
        for C in Cs:
            scores = []
            for tr, va in GroupKFold(n_splits=k).split(E, z, groups):
                if np.unique(z[tr]).size < 2:
                    continue
                m = _pipe(C, pca_dim, E.shape[1]).fit(E[tr], z[tr])
                scores.append(balanced_accuracy(z[va], m.predict(E[va])))
            if scores and np.mean(scores) > best:
                best, best_C = float(np.mean(scores)), C
    model = _pipe(best_C, pca_dim, E.shape[1]).fit(E, z)
    model.chosen_C_ = best_C
    return model


def probe_D(model, E: np.ndarray, z: np.ndarray) -> float:
    return D_from_ba(balanced_accuracy(z, model.predict(E)))


def within_subject_D(E: np.ndarray, z: np.ndarray, subj: np.ndarray, C: float, folds: int = 5,
                     pca_dim: int | None = None, seed: int = 0) -> dict:
    """Refit per test subject; pooled out-of-fold predictions -> D (plus per-subject D)."""
    pred = np.full(z.shape, -1)
    per = {}
    for s in np.unique(subj):
        m = subj == s
        zs = z[m]
        if np.bincount(zs, minlength=2).min() < 2:
            continue
        k = min(folds, np.bincount(zs).min())
        idx = np.where(m)[0]
        ps = np.empty(idx.size, dtype=int)
        for tr, va in StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, zs):
            mdl = _pipe(C, pca_dim, E.shape[1]).fit(E[idx[tr]], zs[tr])
            ps[va] = mdl.predict(E[idx[va]])
        pred[idx] = ps
        per[int(s)] = D_from_ba(balanced_accuracy(zs, ps))
    ok = pred >= 0
    return {"D": D_from_ba(balanced_accuracy(z[ok], pred[ok])) if ok.any() else float("nan"), "per_subject": per}
