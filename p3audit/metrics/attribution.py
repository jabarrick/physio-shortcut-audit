"""3.6.2 attribution: Integrated Gradients over a filter-bank decomposition.

The window is split exactly into bands with FFT masks (Σ_b x_b = x).
Main baseline ("去除目标频段的同窗口"): for band b the path runs from
x − x_b to x, so channel relevance R[c, b] = Σ_t x_b ⊙ ∫∇f, and
Σ_c R[c, b] = f(x) − f(x − x_b)  (completeness).
Zero-baseline sensitivity: one path 0 → x; R[c, b] = Σ_t x_b ⊙ ∫∇f.
f = logit(1) − logit(0).

Summaries per unit:
  share   = Σ|R[target]| / Σ|R|                         (relevance share)
  norm_abs= mean_w Σ|R_w[target]| / mean_w |f(x_w)|      (normalised absolute relevance)
target = posterior × alpha (alpha confound) or frontal-lateral × delta (saccade).
"""
from __future__ import annotations

import numpy as np
import torch

from ..utils.common import ch_index


def band_masks(n_times: int, sfreq: float, bands: dict) -> tuple[list[str], torch.Tensor]:
    f = np.fft.rfftfreq(n_times, 1 / sfreq)
    names, masks = [], []
    covered = np.zeros_like(f, dtype=bool)
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        names.append(name); masks.append(m); covered |= m
    if (~covered).any():
        names.append("rest"); masks.append(~covered)
    return names, torch.as_tensor(np.stack(masks), dtype=torch.float32)


def decompose(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """x (B, C, T) -> (B, K, C, T) with Σ_K = x."""
    X = torch.fft.rfft(x, dim=-1)
    return torch.fft.irfft(X.unsqueeze(1) * masks[None, :, None, :], n=x.shape[-1], dim=-1)


def _f(model, x):
    out = model(x)
    return out[:, 1] - out[:, 0]


def ig_filterbank(model, X: np.ndarray, sfreq: float, bands: dict, steps: int = 32,
                  baseline: str = "band_removed", device=None, batch: int = 16) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (band_names, R (n, C, K), f(x) (n,))."""
    device = device or next(model.parameters()).device
    model.eval()
    names, masks = band_masks(X.shape[-1], sfreq, bands)
    masks = masks.to(device)
    alphas = (torch.arange(steps, device=device, dtype=torch.float32) + 0.5) / steps   # midpoint rule
    R_all, f_all = [], []
    for i in range(0, len(X), batch):
        x = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        xb = decompose(x, masks)                                  # (B, K, C, T)
        B, K, C, T = xb.shape
        R = torch.zeros(B, C, K, device=device)
        with torch.no_grad():
            f_all.append(_f(model, x).cpu().numpy())
        if baseline == "band_removed":
            for k in range(K):
                base = x - xb[:, k]
                inp = (base[None] + alphas[:, None, None, None] * xb[:, k][None]).reshape(-1, C, T)
                inp.requires_grad_(True)
                g, = torch.autograd.grad(_f(model, inp).sum(), inp)
                g = g.reshape(steps, B, C, T).mean(0)
                R[:, :, k] = (xb[:, k] * g).sum(-1)
        elif baseline == "zero":
            inp = (alphas[:, None, None, None] * x[None]).reshape(-1, C, T)
            inp.requires_grad_(True)
            g, = torch.autograd.grad(_f(model, inp).sum(), inp)
            g = g.reshape(steps, B, C, T).mean(0)
            R = (xb * g[:, None]).sum(-1).permute(0, 2, 1)
        else:
            raise ValueError(baseline)
        R_all.append(R.detach().cpu().numpy())
    return names, np.concatenate(R_all), np.concatenate(f_all)


def ig_batch_for(cfg, model_name: str) -> int:
    """Windows per ig_filterbank batch for `model_name` (PILOT_LOG 13.8).  Each window becomes
    `steps` model inputs, so the effective batch is batch x steps.  A pure performance setting:
    with the model in eval mode every window's attribution is computed independently, so the
    result does not depend on it (test_ig_batch_invariance).  ig.batch_by_model overrides
    ig.batch; a stand-in ('cbramod-STANDIN') follows its base name."""
    igc = cfg["ig"]
    per = igc.get("batch_by_model") or {}
    base = model_name.split("-")[0]
    return int(per.get(model_name, per.get(base, igc.get("batch", 16))))


def relevance_summary(R: np.ndarray, fx: np.ndarray, names: list[str], ch_names, target_channels,
                      target_band: str) -> dict:
    ci = ch_index(ch_names, target_channels)
    k = names.index(target_band)
    absR = np.abs(R)
    tgt = absR[:, ci, k].sum(1)
    return {"share": float(tgt.sum() / absR.sum()),
            "norm_abs": float(tgt.mean() / (np.abs(fx).mean() + 1e-12)),
            "signed_target": float(R[:, ci, k].sum(1).mean())}


def ig_targets(cfg, confound: str) -> tuple[list[str], str]:
    if confound == "alpha":
        return cfg["alpha_component"]["posterior"], "alpha"
    return cfg["ig"]["frontal_lateral"], "delta"
