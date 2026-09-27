"""P12 power simulation for H3 (SESOI and variance components are prereg
prerequisites).  Variance components come from P8 (units: model × cell × seed).

Generative model per unit u in model m:
    truth_u  = μ_cell + b_seed + e_t,             e_t ~ N(0, σ_t²)
    metric_u = β·truth_u + e_m,                   e_m ~ N(0, σ_m²)
The pooled within-model correlation is estimated and its bootstrap lower CI
compared with the preregistered threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inference import pooled_within_model_corr


def simulate_h3(n_models: int, cell_means: np.ndarray, n_seeds: int, sd_seed: float, sd_truth: float,
                beta: float, sd_metric: float, threshold: float, n_sim: int = 500, n_boot: int = 300,
                seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    hits = 0
    rs = []
    for _ in range(n_sim):
        rows = []
        for m in range(n_models):
            for c, mu in enumerate(cell_means):
                bs = rng.normal(0, sd_seed, n_seeds)
                for s in range(n_seeds):
                    t = mu + bs[s] + rng.normal(0, sd_truth)
                    rows.append({"model": m, "cell": c, "seed": s, "truth": t,
                                 "metric": beta * t + rng.normal(0, sd_metric)})
        df = pd.DataFrame(rows)
        r = pooled_within_model_corr(df)
        boots = []
        for _ in range(n_boot):
            ss = rng.choice(n_seeds, n_seeds)
            d = pd.concat([df[df.seed == s] for s in ss])
            boots.append(pooled_within_model_corr(d))
        lo = np.nanquantile(boots, 0.025)
        rs.append(r)
        hits += lo > threshold
    return {"power": hits / n_sim, "mean_r": float(np.mean(rs))}
