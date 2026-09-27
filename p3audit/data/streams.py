"""3.3 signal streams, implemented as *linear* operators on raw (160 Hz, µV) data.

model     : resample 200 Hz -> FIR 0.5–75 Hz -> 60 Hz notch -> average reference
spectral  : resample 200 Hz -> FIR 1–40 Hz -> average reference
eye       : FIR 0.1 Hz high-pass at native rate, original reference
emg       : FIR band (40–55 Hz for PhysioNet) at native rate

Because every stream is linear and (on a 4-sample grid) shift-equivariant,
stream(x + Δ) = stream(x) + stream(Δ).  The generator therefore streams the
continuous background once and streams each injection Δ separately (see
generator/cache.py).  This is mathematically identical to injecting in the raw
domain before any filtering/re-referencing (3.4.2), avoids 4-s-window filter
edge artefacts, and lets every experimental condition be composed from cached
bases by scalar coefficients.  `tests/test_streams.py` checks the identity.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import mne
import numpy as np
from scipy.signal import resample_poly


@dataclass(frozen=True)
class StreamSpec:
    name: str
    sfreq: float | None          # None -> keep native
    l_freq: float | None
    h_freq: float | None
    notch: float | None
    avg_ref: bool


def stream_specs(cfg) -> dict[str, StreamSpec]:
    s = cfg["streams"]
    return {
        "model": StreamSpec("model", s["model"]["sfreq"], s["model"]["l_freq"], s["model"]["h_freq"],
                            s["model"]["notch"], s["model"]["avg_ref"]),
        "spectral": StreamSpec("spectral", s["spectral"]["sfreq"], s["spectral"]["l_freq"],
                               s["spectral"]["h_freq"], None, s["spectral"]["avg_ref"]),
        "eye": StreamSpec("eye", None, s["eye"]["l_freq"], None, None, s["eye"]["avg_ref"]),
        "emg": StreamSpec("emg", None, s["emg"]["band"][0], s["emg"]["band"][1], None, False),
    }


def _ratio(sf_in, sf_out):
    fr = Fraction(sf_out / sf_in).limit_denominator(1000)
    return fr.numerator, fr.denominator


def fir(x: np.ndarray, sfreq: float, l_freq, h_freq, trans: float | None = None) -> np.ndarray:
    """Zero-phase FIR (MNE firwin design).  `trans` overrides both transition widths."""
    kw = {}
    if trans is not None:
        if l_freq is not None:
            kw["l_trans_bandwidth"] = trans
        if h_freq is not None:
            kw["h_trans_bandwidth"] = trans
    return mne.filter.filter_data(np.asarray(x, dtype=np.float64), sfreq, l_freq, h_freq, method="fir",
                                  phase="zero", fir_design="firwin", pad="reflect_limited",
                                  verbose="ERROR", **kw)


def bandstop(x: np.ndarray, sfreq: float, lo: float, hi: float, trans: float = 1.0) -> np.ndarray:
    return mne.filter.filter_data(np.asarray(x, dtype=np.float64), sfreq, hi, lo, method="fir", phase="zero",
                                  fir_design="firwin", l_trans_bandwidth=trans, h_trans_bandwidth=trans,
                                  verbose="ERROR")


def apply_stream(x: np.ndarray, sfreq: float, spec: StreamSpec) -> tuple[np.ndarray, float]:
    """x: (..., n_ch, n_times) raw µV.  Returns (y, sfreq_out)."""
    y = np.asarray(x, dtype=np.float64)
    sf = sfreq
    if spec.sfreq is not None and spec.sfreq != sfreq:
        up, down = _ratio(sfreq, spec.sfreq)
        y = resample_poly(y, up, down, axis=-1)
        sf = spec.sfreq
    h = spec.h_freq
    if h is not None and h >= sf / 2:
        h = None
    if spec.l_freq is not None or h is not None:
        y = fir(y, sf, spec.l_freq, h)
    if spec.notch is not None and spec.notch < sf / 2:
        y = mne.filter.notch_filter(y, sf, [spec.notch], method="fir", phase="zero", verbose="ERROR")
    if spec.avg_ref:
        y = y - y.mean(axis=-2, keepdims=True)
    return y, sf


def stream_support_s(spec: StreamSpec, sfreq_in: float) -> float:
    """Half-length (s) of the stream's impulse response: a Δ embedded with this
    much zero padding on each side is streamed exactly as inside the run."""
    sf = spec.sfreq or sfreq_in
    total = 0.0
    h = spec.h_freq if (spec.h_freq is None or spec.h_freq < sf / 2) else None
    if spec.l_freq is not None or h is not None:
        k = mne.filter.create_filter(None, sf, spec.l_freq, h, method="fir", phase="zero",
                                     fir_design="firwin", verbose="ERROR")
        total += len(k) / sf / 2
    if spec.notch is not None and spec.notch < sf / 2:
        total += 3.3 / 1.0 / 2 + 0.5     # MNE default notch transition ~1 Hz
    if spec.sfreq is not None and spec.sfreq != sfreq_in:
        total += 0.25                    # polyphase kernel
    return total + 0.25


def out_index(i_in: int, sfreq_in: float, spec: StreamSpec) -> int:
    """Map a sample index at the native rate to the stream's rate (exact on the grid)."""
    if spec.sfreq is None or spec.sfreq == sfreq_in:
        return i_in
    up, down = _ratio(sfreq_in, spec.sfreq)
    if i_in % down:
        raise ValueError(f"index {i_in} not on the {down}-sample grid required for exact resampling")
    return i_in * up // down
