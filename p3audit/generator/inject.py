"""Injection primitives (3.4.2–3.4.4), all in the raw 160-Hz domain.

Every function returns a *unit* Δ (n_ch, n_region) over a window's padded
region, zero outside the core; experimental conditions multiply it by a
scalar coefficient:

  task   : Δ = (g_task − 1)·e(t)·a_mu·(w_muᵀ x_mu(t))     coefficient  −s   (g_task = 1 − s)
  alpha  : Δ = (g − 1)·e(t)·a_α·(w_αᵀ x_α(t))            coefficient  exp(kσ_α/2) − 1
  saccade: Δ = amp·[step(t)·ã + SP(t)·ã_SP]·e(t)         coefficient  amp (µV on AF7−AF8)

x_mu, x_α are zero-phase FIR band-passed on the *continuous run* before
cutting (七·1): the caller passes the continuous component series.
"""
from __future__ import annotations

import re

import numpy as np

from ..constants import PHYSIONET_CHANNELS
from ..utils.common import ch_index


def cosine_envelope(n_core: int, n_taper: int) -> np.ndarray:
    """e(t): 0 at the core edges, raised-cosine ramp of n_taper samples, 1 in between."""
    e = np.ones(n_core)
    if n_taper > 0:
        r = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        e[:n_taper] = r
        e[-n_taper:] = r[::-1]
    return e


def component_delta(a: np.ndarray, comp_series: np.ndarray, core: slice, n_region: int, pad: int,
                    envelope: np.ndarray) -> np.ndarray:
    """Unit Δ = e(t)·a·s(t) placed in a padded region (zero outside the core).
    comp_series: the continuous band-limited component series sliced to the core."""
    d = np.zeros((a.size, n_region))
    d[:, pad:pad + envelope.size] = np.outer(a, envelope * comp_series)
    return d


# ------------------------------------------------------------------ saccade templates
_MIRROR_RE = re.compile(r"^([A-Za-z]+?)(\d+)$")


def mirror_index(ch_names=PHYSIONET_CHANNELS) -> np.ndarray:
    """Left-right mirror permutation for 10-10 names (odd <-> even+1; z unchanged)."""
    lookup = {c.lower(): i for i, c in enumerate(ch_names)}
    idx = np.arange(len(ch_names))
    for i, c in enumerate(ch_names):
        m = _MIRROR_RE.match(c)
        if m:
            base, num = m.group(1), int(m.group(2))
            partner = f"{base}{num + 1 if num % 2 else num - 1}"
            if partner.lower() in lookup:
                idx[i] = lookup[partner.lower()]
    return idx


def normalise_step_template(a: np.ndarray, ch_names, rightward_sign: int = -1) -> np.ndarray:
    """Scale so that AF7−AF8 = rightward_sign µV for a unit rightward saccade."""
    i7, i8 = ch_index(ch_names, ["AF7", "AF8"])
    diff = a[i7] - a[i8]
    if abs(diff) < 1e-12:
        raise ValueError("template has no AF7−AF8 difference")
    return a * (rightward_sign / diff)


def spike_template(ch_names, reference: list[str] | None, rightward: bool = True) -> np.ndarray:
    """Saccadic spike-potential topography (hypothesis after Keren et al. 2010 [待核实]):
    nose-referenced SP is maximal at the lateral peri-orbital site on the side the eyes move
    toward, smaller contralaterally, with a broad weak negativity over posterior sites.
    Converted to the recording's original reference by subtracting the mean over
    `reference` channels (3.4.4 / 七·9).  Returned unscaled (peak |value| = 1)."""
    w = np.zeros(len(ch_names))
    near = {"F8": 0.8, "AF8": 0.7, "FT8": 1.0, "T8": 0.6, "F10": 0.0}
    far = {"F7": 0.35, "AF7": 0.3, "FT7": 0.4, "T7": 0.25}
    post = ["P3", "Pz", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2"]
    lookup = {c.lower(): i for i, c in enumerate(ch_names)}
    for d_, sgn in ((near, 1.0), (far, 1.0)):
        for c, v in d_.items():
            if c.lower() in lookup:
                w[lookup[c.lower()]] = sgn * v
    for c in post:
        if c.lower() in lookup:
            w[lookup[c.lower()]] = -0.15
    if not rightward:
        w = w[mirror_index(ch_names)]
    if reference:
        w = w - w[ch_index(ch_names, reference)].mean()
    return w / np.abs(w).max()


def step_waveform(n_core: int, onset: int, ramp: int) -> np.ndarray:
    s = np.zeros(n_core)
    if onset >= n_core:
        return s
    r = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp)) if ramp > 0 else np.array([])
    end = min(n_core, onset + ramp)
    s[onset:end] = r[: end - onset]
    s[end:] = 1.0
    return s


def spike_waveform(n_core: int, onset: int, sfreq: float, duration_ms: float) -> np.ndarray:
    """Biphasic transient (first derivative of a Gaussian), band-limited below native Nyquist
    by construction (σ ≥ 2 samples), peak |value| = 1."""
    sig = max(duration_ms / 1000 * sfreq / 4, 2.0)
    t = np.arange(n_core) - onset
    g = -t / sig * np.exp(-0.5 * (t / sig) ** 2)
    return g / (np.abs(g).max() + 1e-12)


def jitter_topography(a: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    return a + rel * np.linalg.norm(a) / np.sqrt(a.size) * rng.standard_normal(a.size)


def perturb_template(a: np.ndarray, ch_names, rotation_deg: float, ft_ratio: float) -> np.ndarray:
    """Mismatch (3.4.5): rotate the dipole toward a vertical-EOG-like field and change the
    frontal/temporal weight ratio."""
    v = np.zeros_like(a)
    lookup = {c.lower(): i for i, c in enumerate(ch_names)}
    for c, val in (("Fp1", 1.0), ("Fpz", 1.0), ("Fp2", 1.0), ("AF7", 0.6), ("AF8", 0.6), ("AF3", 0.6), ("AF4", 0.6)):
        if c.lower() in lookup:
            v[lookup[c.lower()]] = val
    v = v / np.linalg.norm(v) * np.linalg.norm(a)
    th = np.deg2rad(rotation_deg)
    out = np.cos(th) * a + np.sin(th) * v
    frontal = [c for c in ("Fp1", "Fp2", "AF7", "AF8", "F7", "F8") if c.lower() in lookup]
    temporal = [c for c in ("FT7", "FT8", "T7", "T8", "TP7", "TP8") if c.lower() in lookup]
    out[ch_index(ch_names, frontal)] *= ft_ratio
    out[ch_index(ch_names, temporal)] /= ft_ratio
    return out


def saccade_delta(template: np.ndarray, ch_names, sfreq: float, n_region: int, pad: int,
                  envelope: np.ndarray, onset: int, cfg, rng: np.random.Generator,
                  rightward: bool = True, reference=None) -> np.ndarray:
    """Unit (1 µV on AF7−AF8) rightward/leftward saccade: step + spike potential, jittered."""
    sc = cfg["saccade"]
    n_core = envelope.size
    a = template if rightward else template[mirror_index(ch_names)]
    a = jitter_topography(a, sc["topo_jitter"], rng)
    a = normalise_step_template(a, ch_names, sc["rightward_sign"] if rightward else -sc["rightward_sign"])
    wave = np.outer(a, step_waveform(n_core, onset, int(round(sc["ramp_ms"] / 1000 * sfreq))))
    if sc["spike"]["enabled"]:
        sp = spike_template(ch_names, reference, rightward)
        lat = onset + int(round(rng.normal(0, sc["spike"]["latency_jitter_ms"]) / 1000 * sfreq))
        amp = sc["spike"]["amplitude_ratio"] * (1 + sc["spike"]["amp_jitter"] * rng.standard_normal())
        wave += amp * np.outer(sp, spike_waveform(n_core, max(lat, 0), sfreq, sc["spike"]["duration_ms"]))
    d = np.zeros((a.size, n_region))
    d[:, pad:pad + n_core] = wave * envelope
    return d
