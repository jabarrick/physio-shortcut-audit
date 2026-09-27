"""Compose semi-synthetic datasets from cached bases (3.4.5–3.4.6)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cache import SubjectCache
from .components import task_bases
from .design import Cell, assign_z


@dataclass
class ComposedSet:
    """Window-level description of one dataset; inputs are composed lazily.

    X_i = X0_i + Σ_b coef[b]_i · B_b,i     (model stream, padded)
    """
    caches: dict[int, SubjectCache]
    subj: np.ndarray
    idx: np.ndarray
    y: np.ndarray
    z: np.ndarray
    coef: dict[str, np.ndarray]
    wids: list[str]
    pad: int
    core_len: int
    info: dict = field(default_factory=dict)
    _mm: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return self.y.size

    def _arr(self, s: int, name: str):
        key = (s, name)
        if key not in self._mm:
            self._mm[key] = self.caches[s].load(name)
        return self._mm[key]

    def get(self, i: int, padded: bool = False) -> np.ndarray:
        s, j = int(self.subj[i]), int(self.idx[i])
        x = np.array(self._arr(s, "X0")[j], dtype=np.float32)
        for b, c in self.coef.items():
            if c[i] != 0.0:
                x += np.float32(c[i]) * self._arr(s, b)[j]
        return x if padded else x[:, self.pad:self.pad + self.core_len]

    def materialize(self, padded: bool = False, indices=None) -> np.ndarray:
        indices = np.arange(len(self)) if indices is None else np.asarray(indices)
        first = self.get(int(indices[0]), padded)
        out = np.empty((indices.size, *first.shape), dtype=np.float32)
        out[0] = first
        for k, i in enumerate(indices[1:], start=1):
            out[k] = self.get(int(i), padded)
        return out

    def subset(self, mask) -> "ComposedSet":
        m = np.asarray(mask)
        return ComposedSet(self.caches, self.subj[m], self.idx[m], self.y[m], self.z[m],
                           {k: v[m] for k, v in self.coef.items()},
                           [w for w, keep in zip(self.wids, m) if keep], self.pad, self.core_len,
                           dict(self.info), self._mm)


def task_coefficients(y: np.ndarray, s_val: float, cfg, suffix: str = "") -> dict[str, np.ndarray]:
    """Basis coefficients carrying the task signal (3.4.2).

    hemisphere: single — SYMMETRIC bidirectional modulation of the one selected component:
        y=1 -> g = 1 − s/2 (ERD present),  y=0 -> g = 1 + s/2.
    Symmetric rather than present/absent so that both classes carry the same processing
    footprint (cosine taper, filter ripple).  Under a present/absent rule the y=0 windows are
    untouched background, so "has been modulated" is itself class-discriminative and sits next
    to the ground truth that H1–H3 are meant to recover — a second, unintended signal.  The
    price, recorded in 5.2, is that the task signal is no longer a pure suppression relative to
    rest.  Between-class contrast in component log power is 2[ln(1+s/2) − ln(1−s/2)], not the
    2·ln(1−s) of the bilateral design, so P4 must re-bracket s_star / s_low / s_high.

    hemisphere: both — the v8.1 contralateral design: g = 1 − s on the hemisphere given by y.
    """
    tb = task_bases(cfg)
    if len(tb) == 1:
        return {f"{tb[0]}{suffix}": np.where(y == 1, -s_val / 2.0, s_val / 2.0)}
    return {f"mu_L{suffix}": np.where(y == 0, -s_val, 0.0),
            f"mu_R{suffix}": np.where(y == 1, -s_val, 0.0)}


def alpha_log_change(cfg, amp: float, sigma) -> np.ndarray:
    """Target change of alpha-component log power for a z = 1 window (3.4.3).

    reference: sigma  -> amp · a* · k_ref · σ_i            (v8.1; per-subject within-run SD)
    reference: ec_eo  -> amp · a* · ec_eo_log_ratio          (PILOT_LOG 11.13; one value for all)
    """
    ac = cfg["alpha_component"]
    sigma = np.asarray(sigma, dtype=float)
    if ac.get("reference", "sigma") == "ec_eo":
        return np.full_like(sigma, amp * ac["a_star"] * ac["ec_eo_log_ratio"])
    return amp * ac["a_star"] * ac["k_ref"] * sigma


def compose(caches: dict[int, SubjectCache], subjects: list[int], cell: Cell, cfg, q: float | None,
            key: tuple, amp: float | None = None, y_override: dict[int, np.ndarray] | None = None,
            role: str = "train") -> ComposedSet:
    """Build a dataset for `subjects` under `cell`, with coupling q
    (P(z=1|y=1)=q; None = no confound injection) and optional amplitude override
    (units of a*; used for the H1 test-amplitude series)."""
    subj, idx, y, wids = [], [], [], []
    for s in subjects:
        c = caches[s]
        ys = np.asarray(c.meta["y"] if y_override is None else y_override[s])
        subj.append(np.full(c.n, s)); idx.append(np.arange(c.n)); y.append(ys); wids += c.meta["wids"]
    subj = np.concatenate(subj); idx = np.concatenate(idx); y = np.concatenate(y).astype(np.int64)
    n = y.size

    s_val = cfg.task_strength(cell.s, cell.confound)
    coef: dict[str, np.ndarray] = dict(task_coefficients(y, s_val, cfg, cell.task_suffix))

    if q is None:
        z = np.zeros(n, dtype=np.int64)
    else:
        z = assign_z(y, subj, q, key)
    a = cell.amp if amp is None else amp
    if q is not None:        # test-time probing sets inject even for no-injection cells
        if cell.confound == "alpha":
            sig = np.array([caches[int(s)].meta["sigma_alpha"] for s in subj])
            coef[cell.alpha_basis] = z * (np.exp(alpha_log_change(cfg, a, sig) / 2.0) - 1.0)
        elif cell.confound == "saccade":
            amp_uv = a * cfg["saccade"]["amplitude_uv"]
            basis = cell.sacc_basis_test if (role == "test" and cell.sacc_basis_test) else cell.sacc_basis
            coef[basis] = z * amp_uv
            if cfg["saccade"]["mode"] == "direction":   # z=1 rightward, z=0 leftward
                coef["sacc_L"] = (1 - z) * amp_uv
        else:
            raise ValueError(cell.confound)
    any_c = next(iter(caches.values()))
    return ComposedSet(caches, subj, idx, y, z, coef, wids, any_c.meta["pad"], any_c.meta["core_len"],
                       {"cell": cell.cid, "q": q, "amp": a, "s": s_val})
