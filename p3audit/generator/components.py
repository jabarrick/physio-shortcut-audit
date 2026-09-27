"""Per-subject GED/SSD spatial components (3.4.2 mu, 3.4.3 posterior alpha).

GED: maximise  wᵀ C_S w / wᵀ C_R w  with shrinkage on C_R.
Pattern (Haufe 2014): a = C w / (wᵀ C w), C = covariance of the band-limited
data, so that wᵀa = 1 and  a·(wᵀx)  is the component's back-projection onto
all channels ("按完整拓扑反投影").

mu    : S = T0 rest, R = task (T1+T2 pooled, labels unused) in execution runs 3/7/11,
        8–13 Hz.  One left- and one right-hemisphere component.
alpha : S = eyes closed (run 2), R = eyes open (run 1), 8–13 Hz.

Only baseline and execution runs are read (SignalAccessGuard purposes
gen_alpha_component / gen_mu_component); labels and pseudo-label experiments
are never used, so estimation is allowed for test subjects (3.1, 十·3).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.signal import welch

from ..data.streams import fir
from ..utils.common import ch_index


@dataclass
class Component:
    kind: str                      # 'alpha' | 'mu_L' | 'mu_R' | 'mu' (selected hemisphere)
    w: np.ndarray                  # spatial filter (n_ch,)
    a: np.ndarray                  # pattern (n_ch,), wᵀa = 1
    eigval: float
    band: tuple[float, float]
    passed: bool
    checks: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["w"] = self.w.tolist(); d["a"] = self.a.tolist()
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["w"] = np.asarray(d["w"]); d["a"] = np.asarray(d["a"]); d["band"] = tuple(d["band"])
        return Component(**d)


# ------------------------------------------------------------------ GED core
def shrink(C: np.ndarray, gamma: float) -> np.ndarray:
    n = C.shape[0]
    return (1 - gamma) * C + gamma * np.trace(C) / n * np.eye(n)


def ged(C_S: np.ndarray, C_R: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues (desc) and filters W (n_ch, n_comp)."""
    evals, W = eigh(C_S, shrink(C_R, reg))
    order = np.argsort(evals)[::-1]
    return evals[order], W[:, order]


def pattern(C: np.ndarray, w: np.ndarray) -> np.ndarray:
    return C @ w / float(w @ C @ w)


def seg_cov(xb: np.ndarray, segments: list[tuple[int, int]]) -> np.ndarray:
    parts = [xb[:, a:b] for a, b in segments if b > a]
    X = np.concatenate(parts, axis=1)
    X = X - X.mean(axis=1, keepdims=True)
    return X @ X.T / X.shape[1]


# ------------------------------------------------------------------ spectral checks
def component_psd(x_broad: np.ndarray, w: np.ndarray, sfreq: float, nperseg_s: float = 2.0):
    s = w @ x_broad
    n = int(nperseg_s * sfreq)
    f, p = welch(s, fs=sfreq, nperseg=n, noverlap=n // 2)
    return f, p


def band_peak(f, p, band, fit_range=(2.0, 30.0), exclude=(7.0, 14.0), min_db=1.0) -> tuple[bool, float]:
    """Peak inside `band` rising ≥ min_db above a log-log power-law fit (fit excludes `exclude`)."""
    m_fit = (f >= fit_range[0]) & (f <= fit_range[1]) & ~((f >= exclude[0]) & (f <= exclude[1]))
    coef = np.polyfit(np.log10(f[m_fit]), np.log10(p[m_fit]), 1)
    m_b = (f >= band[0]) & (f <= band[1])
    resid_db = 10 * (np.log10(p[m_b]) - np.polyval(coef, np.log10(f[m_b])))
    return bool(resid_db.max() >= min_db), float(resid_db.max())


def posterior_ratio(a: np.ndarray, post_idx) -> float:
    """Share of the pattern's energy carried by posterior sites (3.4.2 anti-contamination).

    Reference value: with 18 of 64 channels posterior, an isotropic pattern gives 0.281."""
    a = np.asarray(a, dtype=float)
    return float((a[post_idx] ** 2).sum() / (a**2).sum())


def rising_high_band(f, p, band=(20.0, 40.0)) -> tuple[bool, float]:
    """True if the log spectrum rises with frequency over `band` (EMG-like)."""
    m = (f >= band[0]) & (f <= band[1])
    slope = np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0]
    return bool(slope > 0), float(slope)


# ------------------------------------------------------------------ alpha
def estimate_alpha(run_open, run_closed, cfg) -> Component:
    ac = cfg["alpha_component"]
    sf = run_open.sfreq
    band = tuple(ac["band"])
    xo = fir(run_open.data, sf, *band, trans=ac["fir_transition_hz"])
    xc = fir(run_closed.data, sf, *band, trans=ac["fir_transition_hz"])
    Co, Cc = seg_cov(xo, [(0, xo.shape[1])]), seg_cov(xc, [(0, xc.shape[1])])
    evals, W = ged(Cc, Co, ac["reg"])
    C_all = (Co + Cc) / 2
    ch = run_open.ch_names
    post = ch_index(ch, ac["posterior"])
    w = W[:, 0]
    a = pattern(C_all, w)
    if a[post].sum() < 0:
        w, a = -w, -a
    bp = ac["band_peak"]
    broad = fir(run_closed.data, sf, 1.0, 45.0)
    f, p = component_psd(broad, w, sf, bp["psd_nperseg_s"])
    peak_ok, peak_db = band_peak(f, p, band, tuple(bp["fit_range"]), tuple(bp["exclude"]), bp["min_db"])
    checks = {
        "posterior_peak": bool(np.argmax(np.abs(a)) in set(post.tolist())),
        "band_peak": peak_ok, "band_peak_db": peak_db,
        "closed_gt_open": bool(evals[0] > 1.0), "eigval": float(evals[0]),
    }
    passed = checks["posterior_peak"] and checks["band_peak"] and checks["closed_gt_open"]
    return Component("alpha", w, a, float(evals[0]), band, passed, checks)


# ------------------------------------------------------------------ mu
def _task_rest_segments(run):
    rest = [(o, o + d) for o, d, _ in run.event_samples("T0")]
    task = [(o, o + d) for o, d, _ in run.event_samples(("T1", "T2"))]
    return rest, task


def estimate_mu(exec_runs, cfg) -> tuple[Component, Component]:
    tc = cfg["task_component"]
    bp = tc["band_peak"]
    band = tuple(tc["band"])
    sf = exec_runs[0].sfreq
    ch = exec_runs[0].ch_names
    Cr, Ct, broads, broads_rest, broads_task = [], [], [], [], []
    for r in exec_runs:
        xb = fir(r.data, sf, *band, trans=tc["fir_transition_hz"])
        rest, task = _task_rest_segments(r)
        Cr.append(seg_cov(xb, rest)); Ct.append(seg_cov(xb, task))
        b_all = fir(r.data, sf, 1.0, 45.0)
        broads.append(b_all)
        broads_rest.append(np.concatenate([b_all[:, i:j] for i, j in rest], axis=1))
        broads_task.append(np.concatenate([b_all[:, i:j] for i, j in task], axis=1))
    C_rest, C_task = np.mean(Cr, axis=0), np.mean(Ct, axis=0)
    evals, W = ged(C_rest, C_task, tc["reg"])          # rest > task  <=> ERD
    C_all = (C_rest + C_task) / 2
    broad = np.concatenate(broads, axis=1)
    broad_rest = np.concatenate(broads_rest, axis=1)
    broad_task = np.concatenate(broads_task, axis=1)
    post_idx = ch_index(ch, cfg["alpha_component"]["posterior"])
    out = {}
    for side, neigh, anchor in (("L", tc["left_neighborhood"], "C3"), ("R", tc["right_neighborhood"], "C4")):
        nidx = set(ch_index(ch, neigh).tolist())
        anchor_i = ch_index(ch, [anchor])[0]
        best = None
        for k in range(min(tc["n_candidates"], W.shape[1])):
            if evals[k] <= 1.0:
                break          # task_lt_rest can never hold below here; screening deeper is pointless
            w = W[:, k]
            a = pattern(C_all, w)
            if a[anchor_i] < 0:
                w, a = -w, -a
            # P1a 2026-09-19: the 8-13 Hz peak is tested on T0 REST, not on rest+task concatenated.
            # Mu is suppressed during movement by construction (that suppression is the ERD the GED
            # was built to find), so the concatenation dilutes the very peak being tested.  Measured
            # on 105 subjects: median band-peak residual 3.15 dB on rest vs 2.15 dB concatenated,
            # 125/184 vs 110/184 sides clearing the 1 dB bar.  The bar itself is unchanged.
            f, p = component_psd(broad_rest, w, sf, bp["psd_nperseg_s"])
            peak_ok, peak_db = band_peak(f, p, band, tuple(bp["fit_range"]), tuple(bp["exclude"]),
                                         bp["min_db"])
            f_all, p_all = component_psd(broad, w, sf, bp["psd_nperseg_s"])
            rising, slope = rising_high_band(f_all, p_all, tuple(tc["emg_band"]))
            f_t, p_t = component_psd(broad_task, w, sf, bp["psd_nperseg_s"])
            checks = {"candidate": k, "topo_peak_in_neighborhood": bool(np.argmax(np.abs(a)) in nidx),
                      "band_peak": peak_ok, "band_peak_db": peak_db,
                      "band_peak_db_all": float(band_peak(f_all, p_all, band, tuple(bp["fit_range"]),
                                                          tuple(bp["exclude"]), bp["min_db"])[1]),
                      "task_lt_rest": bool(evals[k] > 1.0), "eigval": float(evals[k]),
                      "emg_not_rising": not rising, "emg_slope": slope,
                      "emg_slope_task": float(rising_high_band(f_t, p_t, tuple(tc["emg_band"]))[1]),
                      # recorded for every candidate; applied as a criterion only by select_mu,
                      # so mu_L/mu_R pass/fail stay comparable with the 2026-09-19 re-run
                      "posterior_ratio": posterior_ratio(a, post_idx)}
            passed = all(checks[c] for c in ("topo_peak_in_neighborhood", "band_peak", "task_lt_rest",
                                            "emg_not_rising"))
            comp = Component(f"mu_{side}", w, a, float(evals[k]), band, passed, checks)
            if passed:
                best = comp
                break
            if best is None and checks["topo_peak_in_neighborhood"]:
                best = comp            # keep the first topographically plausible failure for reporting
        if best is None:
            w = W[:, 0]; a = pattern(C_all, w)
            best = Component(f"mu_{side}", w, a, float(evals[0]), band, False,
                             {"no_candidate": True, "posterior_ratio": posterior_ratio(a, post_idx)})
        out[side] = best
    # the two hemispheres must be distinct components
    if out["L"].passed and out["R"].passed and out["L"].checks.get("candidate") == out["R"].checks.get("candidate"):
        out["R"].passed = False
        out["R"].checks["same_as_left"] = True
    return out["L"], out["R"]


def task_bases(cfg) -> tuple[str, ...]:
    """Component/basis names carrying the task signal under the current design.

    hemisphere: single -> ('mu',)        one selected hemisphere, symmetric modulation
    hemisphere: both   -> ('mu_L','mu_R')  the v8.1 contralateral design
    """
    return ("mu",) if cfg["task_component"].get("hemisphere", "both") == "single" else ("mu_L", "mu_R")


def required_components(cfg) -> tuple[str, ...]:
    """Components a subject must have for the generator to use them (3.4.2 + 3.4.3)."""
    return ("alpha", *task_bases(cfg))


def select_mu(mu_L: Component, mu_R: Component, cfg) -> Component:
    """3.4.2 (2026-09-19 revision): one hemisphere per subject.

    A side is ELIGIBLE when it passes every per-side criterion AND carries no more posterior
    energy than `task_component.posterior_ratio_max`.  That cap exists because with only one
    hemisphere retained the bilateral requirement no longer guards against a GED component that
    is really posterior alpha, which would make the task signal and the confound degenerate by
    construction.  Among eligible sides the rule takes the larger REST-segment band-peak
    residual; ties go to the left.

    Rule, cap and the cap's fitting principle were all declared in configs/default.yaml before
    this function existed; the expected L 40 / R 30 split is recorded there and in PILOT_LOG 5.
    `mu_L.passed` / `mu_R.passed` keep their original meaning and are unaffected by the cap.
    """
    tc = cfg["task_component"]
    cap = tc.get("posterior_ratio_max")
    sides = (mu_L, mu_R)
    eligible = []
    for c in sides:
        if not c.passed:
            continue
        pr = c.checks.get("posterior_ratio")
        if cap is not None and pr is not None and pr > cap:
            continue
        eligible.append(c)
    checks = {
        "side_rule": tc.get("side_rule"),
        "posterior_ratio_max": cap,
        "eligible": [c.kind[-1] for c in eligible],
        "passed_side": {c.kind[-1]: bool(c.passed) for c in sides},
        "band_peak_db": {c.kind[-1]: c.checks.get("band_peak_db") for c in sides},
        "posterior_ratio": {c.kind[-1]: c.checks.get("posterior_ratio") for c in sides},
    }
    if not eligible:
        return Component("mu", mu_L.w, mu_L.a, mu_L.eigval, mu_L.band, False,
                         {**checks, "side": None, "no_eligible_side": True})
    best = max(eligible, key=lambda c: (c.checks.get("band_peak_db", float("-inf")),
                                        c.kind.endswith("L")))
    return Component("mu", best.w, best.a, best.eigval, best.band, True,
                     {**checks, "side": best.kind[-1], "candidate": best.checks.get("candidate"),
                      "band_peak_db_selected": best.checks.get("band_peak_db"),
                      "posterior_ratio_selected": best.checks.get("posterior_ratio")})


# ------------------------------------------------------------------ group template (3.4.5 control)
def group_component(comps: list[Component], kind: str) -> Component:
    """Average of sign-aligned pilot patterns/filters, rescaled so wᵀa = 1."""
    A = np.stack([c.a / np.linalg.norm(c.a) for c in comps])
    Wf = np.stack([c.w / np.linalg.norm(c.w) for c in comps])
    a = A.mean(0); w = Wf.mean(0)
    w = w / float(w @ a)
    return Component(kind, w, a, float("nan"), comps[0].band, True, {"group_n": len(comps)})
