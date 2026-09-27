"""Spectral-probe-stream quantities (3.3): posterior alpha power, aperiodic
exponent (2–30 Hz, windows with fit R² below threshold rejected), and the
exploratory LZC / HFD complexity measures."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from ..utils.common import ch_index


def log_band_power(X: np.ndarray, sfreq: float, band, channels_idx=None) -> np.ndarray:
    """X (n, C, T) -> mean log power in band over selected channels, (n,)."""
    f, p = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(2 * sfreq)), axis=-1)
    m = (f >= band[0]) & (f <= band[1])
    bp = p[..., m].mean(-1)
    if channels_idx is not None:
        bp = bp[:, channels_idx]
    return np.log(bp).mean(-1)


def posterior_alpha(X, sfreq, ch_names, cfg) -> np.ndarray:
    return log_band_power(X, sfreq, cfg["alpha_component"]["band"], ch_index(ch_names, cfg["alpha_component"]["posterior"]))


def aperiodic_exponent(X: np.ndarray, sfreq: float, frange=(2.0, 30.0), exclude=(7.0, 14.0)) -> tuple[np.ndarray, np.ndarray]:
    """Channel-averaged spectrum; robust log-log line fit excluding the alpha range.
    Returns (exponent, R²) per window.  Use specparam for the preregistered version if installed."""
    f, p = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(2 * sfreq)), axis=-1)
    p = p.mean(1)
    m = (f >= frange[0]) & (f <= frange[1]) & ~((f >= exclude[0]) & (f <= exclude[1]))
    lf = np.log10(f[m])
    exps, r2s = [], []
    for row in np.log10(p[:, m]):
        coef = np.polyfit(lf, row, 1)
        pred = np.polyval(coef, lf)
        ss = ((row - row.mean()) ** 2).sum()
        exps.append(-coef[0]); r2s.append(1 - ((row - pred) ** 2).sum() / ss if ss > 0 else np.nan)
    return np.array(exps), np.array(r2s)


def lzc(x: np.ndarray) -> float:
    """Lempel–Ziv complexity (LZ76) of the median-binarised signal, normalised by n/log2 n."""
    s = (x > np.median(x)).astype(np.uint8)
    n = s.size
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c * np.log2(n) / n


def hfd(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi fractal dimension."""
    n = x.size
    lk = []
    ks = np.arange(1, kmax + 1)
    for k in ks:
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            length = np.abs(np.diff(x[idx])).sum() * (n - 1) / ((idx.size - 1) * k) / k
            lm.append(length)
        lk.append(np.mean(lm))
    return float(np.polyfit(np.log(1.0 / ks), np.log(lk), 1)[0])
