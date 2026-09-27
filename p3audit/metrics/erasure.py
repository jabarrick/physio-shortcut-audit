"""3.6.2 in-distribution residual erasure (LEACE, Belrose et al. 2023).

Concept = residual r = z − E[z|y].  Main protocol: eraser fitted on training
embeddings (training coupling p), frozen, applied to test embeddings without
labels; the model's own linear head is kept fixed.
    ΔBA_keep = BA(head(h)) − BA(head(erase(h)))        on the matched test set
Sensitivity protocol (十一·2): refit per test subject on its own windows, still
erasing the residual z − E[z|y] (pseudo-label y is known by construction and
never seen by the model with z; its use is limited to residualisation).
ΔBA_retrain (amnesic probing) is a diagnostic only.
Principal angle between the erased direction and the task direction
(Haufe pattern Cov(h)·w_task) supports the angle–erasure-bias relation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import subspace_angles
from sklearn.linear_model import LogisticRegression

from ..utils.common import balanced_accuracy


def _sym_power(S: np.ndarray, power: float, eps: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(S)
    keep = vals > eps * vals.max()
    v = np.zeros_like(vals)
    v[keep] = vals[keep] ** power
    return (vecs * v) @ vecs.T


@dataclass
class LeaceEraser:
    mean: np.ndarray
    proj: np.ndarray            # (d, d): x ↦ x − proj @ (x − mean)
    direction: np.ndarray       # erased direction(s) in embedding space (d, k)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return X - (X - self.mean) @ self.proj.T


def fit_leace(X: np.ndarray, Z: np.ndarray, reg: float = 1e-6) -> LeaceEraser:
    """Closed-form LEACE: r(x) = x − W⁺ P W (x − μ),  W = Σ_XX^{-1/2},
    P = orthogonal projector onto colspace(W Σ_XZ)."""
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64).reshape(len(X), -1)
    mu = X.mean(0)
    Xc, Zc = X - mu, Z - Z.mean(0)
    n = len(X)
    S_xx = Xc.T @ Xc / n
    S_xz = Xc.T @ Zc / n
    W = _sym_power(S_xx, -0.5, reg)
    W_pinv = _sym_power(S_xx, 0.5, reg)
    U, s, _ = np.linalg.svd(W @ S_xz, full_matrices=False)
    U = U[:, s > 1e-12 * max(s.max(), 1e-300)]
    P = U @ U.T
    proj = W_pinv @ P @ W
    return LeaceEraser(mu, proj, W_pinv @ U)


def residual_concept(z: np.ndarray, y: np.ndarray, cond_means: dict | None = None):
    """r = z − E[z|y]; E[z|y] from `cond_means` (training estimates) or from (z, y)."""
    if cond_means is None:
        cond_means = {int(c): float(z[y == c].mean()) for c in np.unique(y)}
    return z - np.array([cond_means[int(c)] for c in y]), cond_means


def head_predict(head_W: np.ndarray, head_b: np.ndarray, H: np.ndarray) -> np.ndarray:
    return (H @ head_W.T + head_b).argmax(1)


def delta_ba_keep(eraser: LeaceEraser, head_W, head_b, H_test, y_test) -> dict:
    ba = balanced_accuracy(y_test, head_predict(head_W, head_b, H_test))
    ba_e = balanced_accuracy(y_test, head_predict(head_W, head_b, eraser(H_test)))
    return {"ba": ba, "ba_erased": ba_e, "delta_ba_keep": ba - ba_e}


def within_subject_keep(H, z, y, subj, head_W, head_b, reg=1e-6) -> dict:
    """Sensitivity protocol: per-subject eraser on residual z − E[z|y] (same estimator)."""
    y_hat = np.empty_like(y)
    for s in np.unique(subj):
        m = subj == s
        r, _ = residual_concept(z[m], y[m])
        er = fit_leace(H[m], r, reg)
        y_hat[m] = head_predict(head_W, head_b, er(H[m]))
    ba = balanced_accuracy(y, head_predict(head_W, head_b, H))
    return {"ba": ba, "ba_erased": balanced_accuracy(y, y_hat), "delta_ba_keep": ba - balanced_accuracy(y, y_hat)}


def delta_ba_retrain(eraser, H_train, y_train, H_test, y_test, C: float = 1.0) -> float:
    """Diagnostic: BA of a head retrained on erased embeddings (residual task information)."""
    clf = LogisticRegression(C=C, max_iter=2000).fit(eraser(H_train), y_train)
    return float(balanced_accuracy(y_test, clf.predict(eraser(H_test))))


def task_pattern(H: np.ndarray, head_W: np.ndarray) -> np.ndarray:
    """Haufe activation pattern of the head's decision direction: Cov(h)·w."""
    w = head_W[1] - head_W[0] if head_W.shape[0] == 2 else head_W[0]
    Hc = H - H.mean(0)
    return (Hc.T @ Hc / len(H)) @ w


def erasure_task_angle(eraser: LeaceEraser, H: np.ndarray, head_W: np.ndarray) -> float:
    """Smallest principal angle (degrees) between erased subspace and task pattern."""
    if eraser.direction.shape[1] == 0:          # residual concept has no variance (e.g. p = 1)
        return float("nan")
    a = task_pattern(H, head_W).reshape(-1, 1)
    return float(np.rad2deg(subspace_angles(eraser.direction, a).min()))
