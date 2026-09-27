"""P1a design parameters: confound–task overlap, topographic dispersion and
realised injection effects (3.4.3, reported in 4.1)."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.signal import welch

from ..data.streams import fir
from ..utils.common import ch_index
from .inject import component_delta, cosine_envelope


def projection_ratio(a_alpha: np.ndarray, ch_names, central, posterior) -> float:
    """mean |a_α| over C3/Cz/C4  ÷  peak |a_α| over posterior sites."""
    c = np.abs(a_alpha[ch_index(ch_names, central)]).mean()
    p = np.abs(a_alpha[ch_index(ch_names, posterior)]).max()
    return float(c / p)


def abs_spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.corrcoef(a, b)[0, 1]))


def overlap_summary(comps: dict, ch_names, cfg) -> dict:
    ac = cfg["alpha_component"]
    rL = abs_spatial_corr(comps["mu_L"].a, comps["alpha"].a)
    rR = abs_spatial_corr(comps["mu_R"].a, comps["alpha"].a)
    out = {"projection_ratio": projection_ratio(comps["alpha"].a, ch_names, ac["central"], ac["posterior"]),
           "corr_muL_alpha": rL, "corr_muR_alpha": rR,
           "overlap": max(rL, rR)}           # H3 moderator (max over hemispheres; conservative)
    # Under the single-hemisphere task signal the moderator that matches what the generator
    # actually injects is the SELECTED side's correlation, not the max over hemispheres.  Both
    # are stored; which one 十·2 uses for H3 is a separate, explicit decision.
    mu = comps.get("mu")
    if mu is not None:
        side = mu.checks.get("side")
        out["overlap_selected"] = {"L": rL, "R": rR}.get(side)
        out["selected_side"] = side
    return out


def topo_dispersion(patterns: dict[int, np.ndarray]) -> dict:
    """Pairwise |spatial correlation| between subjects' patterns (十·1 item 1)."""
    ids = sorted(patterns)
    r = [abs_spatial_corr(patterns[i], patterns[j]) for i, j in itertools.combinations(ids, 2)]
    r = np.asarray(r)
    return {"n_subjects": len(ids), "n_pairs": int(r.size), "median": float(np.median(r)) if r.size else np.nan,
            "q25": float(np.quantile(r, 0.25)) if r.size else np.nan,
            "q75": float(np.quantile(r, 0.75)) if r.size else np.nan, "values": r.tolist()}


def _logpow(x, sf, band):
    f, p = welch(x, fs=sf, nperseg=min(x.shape[-1], int(sf)), axis=-1)
    m = (f >= band[0]) & (f <= band[1])
    return np.log(p[..., m].mean(-1))


def measure_injection(run, windows, comp, g: float, cfg, central) -> dict:
    """Realised effect of multiplicative component modulation with steady gain g.

    Returns mean change in component log power (target: 2·ln g = kσ; 七·3 asks to report
    the shortfall caused by filter ripple / transition band) and mean change in central
    8–13 Hz log power (overlap evidence).  Computed on window cores in the raw domain."""
    sf = run.sfreq
    band = comp.band
    n_taper = int(round(cfg["alpha_component"]["taper_s"] * sf))
    s = comp.w @ fir(run.data, sf, *band, trans=cfg["alpha_component"]["fir_transition_hz"])
    ci = ch_index(run.ch_names, central)
    d_comp, d_central = [], []
    for w in windows:
        core = slice(w.start, w.start + w.length)
        x = run.data[:, core]
        env = cosine_envelope(w.length, n_taper)
        delta = (g - 1.0) * component_delta(comp.a, s[core], slice(None), w.length, 0, env)
        xp = x + delta
        # evaluate the steady-state middle (exclude tapers)
        mid = slice(n_taper, w.length - n_taper)
        d_comp.append(_logpow(comp.w @ xp[:, mid], sf, band) - _logpow(comp.w @ x[:, mid], sf, band))
        d_central.append(_logpow(xp[ci][:, mid], sf, band).mean() - _logpow(x[ci][:, mid], sf, band).mean())
    return {"target_log_change": float(2 * np.log(g)),
            "component_log_change": float(np.mean(d_comp)),
            "central_log_change": float(np.mean(d_central)),
            "n_windows": len(windows)}
