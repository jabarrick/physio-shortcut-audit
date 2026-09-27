"""3.4.4 step-component template by split-half ICA on pilot subjects (P6).

Procedure per pilot subject:
  1. six lateral runs (exec 3/7/11, imag 4/8/12), 0.1 Hz high-pass, epochs around cue;
  2. trials alternately split into two halves *within* execution and imagery so the
     exec/imag proportion is equal (七·4);
  3. per half: PCA to 30 → FastICA;
  4. selection: component whose 0–1 s low-frequency mean activation correlates most with
     target side (left −1 / right +1), with opposite-sign F7/AF7 vs F8/AF8 weights, and
     |r| above the 95th percentile of the permutation null of max|r| over the 30 components
     (1000 side permutations) (七·5);
  5. failure: any half without a qualifying component -> subject leaves the pool; > 5 of
     15 failures -> saccade injection infeasible (P6 decision rule).

The pool is restricted to pilot subjects to avoid label leakage: target side *is* the
true label in the L/R task (十·4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA, FastICA

from ..data.streams import fir
from ..utils.common import ch_index


@dataclass
class HalfResult:
    ok: bool
    pattern: np.ndarray | None
    r: float
    threshold: float
    component: int
    reason: str = ""
    n_gated: int = -1              # components surviving the ocular topography gate
    enrichment: float = float("nan")   # ocular energy share / chance share
    bipolarity: float = float("nan")   # |p[AF7] - p[AF8]| / max|p|


@dataclass
class SaccadeTemplate:
    subject: int
    ok: bool
    template: np.ndarray | None
    split_half_r: float
    halves: list[HalfResult] = field(default_factory=list)


def lateral_epochs(runs, cfg):
    """Returns epochs (n_trials, n_ch, n_t), side (−1/+1), is_exec (bool), sfreq."""
    sc = cfg["saccade"]
    X, side, is_exec = [], [], []
    sf = runs[0].sfreq
    t0, t1 = sc["epoch"]
    for r in runs:
        xh = fir(r.data, sf, sc["highpass"], None)
        for o, _, s in r.event_samples(("T1", "T2")):
            a, b = o + int(round(t0 * sf)), o + int(round(t1 * sf))
            if a < 0 or b > xh.shape[1]:
                continue
            X.append(xh[:, a:b]); side.append(-1 if s == "T1" else 1); is_exec.append(r.kind.startswith("exec"))
    return np.stack(X), np.array(side), np.array(is_exec), sf


def alternate_halves(is_exec: np.ndarray) -> np.ndarray:
    half = np.zeros(is_exec.size, dtype=int)
    for flag in (True, False):
        idx = np.where(is_exec == flag)[0]
        half[idx[1::2]] = 1
    return half


def _activation_feature(S_epochs, sf, cfg):
    sc = cfg["saccade"]
    t0 = sc["epoch"][0]
    low = fir(S_epochs, sf, None, sc["activation_lowpass"])
    i_base = slice(0, int(round(-t0 * sf)))
    a, b = sc["activation_window"]
    i_win = slice(int(round((a - t0) * sf)), int(round((b - t0) * sf)))
    return low[..., i_win].mean(-1) - low[..., i_base].mean(-1)       # (n_trials, n_comp)


def _corr_cols(F, s):
    Fc = F - F.mean(0); sc = s - s.mean()
    return (Fc * sc[:, None]).sum(0) / (np.sqrt((Fc ** 2).sum(0) * (sc ** 2).sum()) + 1e-12)


def ocular_features(p: np.ndarray, oc_idx, i7: int, i8: int) -> tuple[float, float]:
    """(enrichment, bipolarity) of a pattern.

    enrichment = ocular-site energy share / the share an isotropic pattern would give
                 (len(oc_idx) / n_ch), so 1.0 is chance.
    bipolarity = |p[AF7] - p[AF8]| / max|p|; a pure horizontal dipole whose extrema are
                 AF7/AF8 gives 2.0, a blink (vertical) gives ~0.

    Both are functions of the pattern ALONE and so are independent of the side labels — which
    is what allows the permutation family to be restricted to the gated components without
    biasing the null (3.4.4 / P6 redesign 2026-09-19)."""
    p = np.asarray(p, dtype=float)
    chance = len(oc_idx) / p.size
    enr = float((p[oc_idx] ** 2).sum() / (p**2).sum() / chance)
    bip = float(abs(p[i7] - p[i8]) / np.abs(p).max())
    return enr, bip


def ocular_gate(patterns: np.ndarray, ch_names, cfg) -> np.ndarray:
    """Indices of components whose topography is that of a horizontal eye movement."""
    sc = cfg["saccade"]
    oc = ch_index(ch_names, sc["ocular_sites"])
    i7, i8 = ch_index(ch_names, ["AF7", "AF8"])
    keep = []
    for k in range(patterns.shape[1]):
        enr, bip = ocular_features(patterns[:, k], oc, i7, i8)
        if enr >= sc["ocular_enrichment_min"] and bip >= sc["bipolarity_min"]:
            keep.append(k)
    return np.array(keep, dtype=int)


def _reason(*parts) -> str:
    return "; ".join(x for x in parts if x)


def fit_half(X, side, sf, ch_names, cfg, seed) -> HalfResult:
    sc = cfg["saccade"]
    n_tr, n_ch, n_t = X.shape
    cat = X.transpose(1, 0, 2).reshape(n_ch, -1)
    cat = cat - cat.mean(1, keepdims=True)
    n_comp = min(sc["n_pca"], n_ch - 1)
    pca = PCA(n_components=n_comp, random_state=seed).fit(cat.T)
    Z = pca.transform(cat.T)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    converged = True
    for attempt, (max_iter, tol) in enumerate(((3000, 1e-3), (10000, 5e-3))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            ica = FastICA(n_components=n_comp, random_state=seed + 101 * attempt, max_iter=max_iter, tol=tol,
                          whiten="unit-variance").fit(Z)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if converged:
            break
    S = ica.transform(Z).T                                          # (n_comp, total)
    patterns = pca.components_.T @ ica.mixing_                      # (n_ch, n_comp)
    S_ep = S.reshape(n_comp, n_tr, n_t).transpose(1, 0, 2)
    F = _activation_feature(S_ep, sf, cfg)
    rng = np.random.default_rng(seed)
    li = ch_index(ch_names, sc["left_sites"]); ri = ch_index(ch_names, sc["right_sites"])
    oc_idx = ch_index(ch_names, sc["ocular_sites"])
    i7, i8 = ch_index(ch_names, ["AF7", "AF8"])
    note = "" if converged else "ICA not converged"

    # P6 redesign 2026-09-19: gate on topography first, then rank by side-correlation.  The
    # old order ranked all n_pca components by |r| and paid an n_pca-way multiple-comparison
    # penalty in the permutation null, while the generator needs one stable OCULAR template.
    # The gate is label-independent, so restricting the null to it is legitimate.
    if sc.get("selection_order", "correlation_first") == "topography_first":
        gated = ocular_gate(patterns, ch_names, cfg)
        if gated.size == 0:
            return HalfResult(False, None, float("nan"), float("nan"), -1,
                              _reason("no component passes the ocular topography gate", note), 0)
        Fg = F[:, gated]
    else:
        gated = np.arange(F.shape[1])
        Fg = F

    r = _corr_cols(Fg, side.astype(float))
    null = np.array([np.abs(_corr_cols(Fg, rng.permutation(side).astype(float))).max()
                     for _ in range(sc["n_permutations"])])
    thr = float(np.quantile(null, sc["null_quantile"]))
    order = np.argsort(-np.abs(r))
    for j in order:
        k = int(gated[j])
        p = patterns[:, k] * np.sign(r[j])                          # activation ↑ for rightward target
        # 2026-09-19: the opposite-sign test is applied to the SPATIALLY MEAN-CENTRED pattern.
        # EEGMMIDB's recording reference is undocumented (PhysioNet states the 10-10 montage but
        # no reference; the BCI2000 paper's only reference sentence describes the SCP
        # implementation, not these Wadsworth recordings).  A single common reference adds the
        # same offset to every channel, i.e. a constant to the pattern, and if that reference
        # picks up horizontal EOG the constant is itself saccade-driven — which can flip the
        # left/right sign test without any change in the underlying dipole.  Removing the
        # spatial mean removes exactly that constant, so the test asks what it is meant to ask
        # whatever the reference was.  It is also the domain the model actually sees: both the
        # model and spectral streams average-reference (streams.py).
        p_c = p - p.mean()
        opposite = np.sign(p_c[li].mean()) != np.sign(p_c[ri].mean())
        if abs(r[j]) > thr and opposite:
            enr, bip = ocular_features(p, oc_idx, i7, i8)
            return HalfResult(True, p, float(abs(r[j])), thr, k, note, int(gated.size), enr, bip)
        if abs(r[j]) <= thr:
            break
    j0 = int(order[0]); k0 = int(gated[j0])
    enr, bip = ocular_features(patterns[:, k0], oc_idx, i7, i8)
    return HalfResult(False, None, float(abs(r[j0])), thr, k0,
                      _reason("no gated component passes |r|>null95 with opposite F7/AF7-F8/AF8 sign", note),
                      int(gated.size), enr, bip)


def template_for_subject(subject: int, runs, cfg, seed: int = 0) -> SaccadeTemplate:
    X, side, is_exec, sf = lateral_epochs(runs, cfg)
    halves = alternate_halves(is_exec)
    ch = runs[0].ch_names
    res = [fit_half(X[halves == h], side[halves == h], sf, ch, cfg, seed + h) for h in (0, 1)]
    if not all(h.ok for h in res):
        return SaccadeTemplate(subject, False, None, float("nan"), res)
    p0, p1 = (h.pattern / np.linalg.norm(h.pattern) for h in res)
    if p0 @ p1 < 0:
        p1 = -p1
    r_sh = float(np.corrcoef(p0, p1)[0, 1])
    return SaccadeTemplate(subject, True, (p0 + p1) / 2, r_sh, res)


def pool_feasible(templates: list[SaccadeTemplate], cfg) -> tuple[bool, int]:
    n_fail = sum(not t.ok for t in templates)
    return n_fail <= cfg["saccade"]["max_failed_subjects"], n_fail
