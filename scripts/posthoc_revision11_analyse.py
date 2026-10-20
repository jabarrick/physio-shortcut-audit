"""Post hoc (exploratory; fourth review): analysis of results/units_csp (CSP + LDA, point 18) and
results/units_erasure05 (erasure fitted at p = 0.5, point 12).  Writes results/analysis/posthoc_revision11.json.
CSP + LDA, per confound: mean ground truth per amplitude; standard deviation of the cell residuals of ground truth
(seeds 50-55 differ only in data composition) and within-cell split-half r; correlation with ground truth across units
(cross-fitted over window halves for HEOG regression, erasure, SRI and BA(matched); not for the amplitude and D).
Erasure variant, per confound: registered residual erasure (Delta BA_keep) and the variant
Delta BA_erase05 = BA(matched) - BA(matched, z erased by LEACE fitted at p = 0.5) against Delta BA_neu: mean bias,
median recovery where Delta BA_neu > 0.05, and the cross-fitted correlation with ground truth across units
(standardised within model).  Bootstrap over test subjects with seeds fixed (2000 resamples, seed 20261021).
Usage (repo root): python scripts/posthoc_revision11_analyse.py [--csp DIR] [--erasure DIR]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from p3audit.experiments import analysis as A
N_BOOT, SEED = 2000, 20261021


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 3 and a[k].std() > 0 and b[k].std() > 0 else np.nan


def zw(v, mod):
    out = np.full(len(v), np.nan)
    for m in np.unique(mod):
        k = (mod == m) & np.isfinite(v)
        if k.sum() > 1 and v[k].std() > 0: out[k] = (v[k] - v[k].mean()) / v[k].std()
    return out


def ba(r, path, subs, h):
    node = r
    for p in path: node = node[p]
    return A._ba(node[h], subs)


def metrics_csp(conf):
    m = {"BA_matched": lambda r, s, h: ba(r, ("truth_counts", "matched"), s, h),
         "erasure_keep": lambda r, s, h: ba(r, ("erasure", "counts_head"), s, h) - ba(r, ("erasure", "counts_erased"), s, h),
         "heog_reg": lambda r, s, h: ba(r, ("truth_counts", "matched"), s, h) - ba(r, ("heog", "0", "counts"), s, h)}
    if conf == "alpha":
        m["sri_post"] = lambda r, s, h: ba(r, ("sri", "counts", "post_ctrl"), s, h) - ba(r, ("sri", "counts", "post_alpha"), s, h)
    return m


def metrics_er05():
    return {"erasure_keep": lambda r, s, h: ba(r, ("erasure", "counts_head"), s, h) - ba(r, ("erasure", "counts_erased"), s, h),
            "erasure05": lambda r, s, h: ba(r, ("truth_counts", "matched"), s, h) - ba(r, ("erasure05", "counts_erased05_matched"), s, h)}


def xfit(us, subs, fns, mod):
    t = {h: np.array([A.truth(r, subs, h) for r in us]) for h in ("half0", "half1")}; o = {}
    for n, f in fns.items():
        m = {h: np.array([f(r, subs, h) for r in us]) for h in ("half0", "half1")}
        o[n] = float(np.nanmean([corr(zw(t["half0"], mod), zw(m["half1"], mod)), corr(zw(t["half1"], mod), zw(m["half0"], mod))]))
    return o


def boot(us, fns, mod, extra=None):
    subs = A._test_subjects(us); est = xfit(us, subs, fns, mod)
    if extra: est.update(extra(us, subs))
    rng = np.random.default_rng(SEED); B = []
    for _ in range(N_BOOT):
        sb = list(rng.choice(subs, len(subs))); b = xfit(us, sb, fns, mod)
        if extra: b.update(extra(us, sb))
        B.append(b)
    out = {}
    for k, v in est.items():
        bb = np.array([x[k] for x in B], float); bb = bb[np.isfinite(bb)]
        out[k] = {"est": v, "ci95": [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))] if len(bb) > 10 else None}
    return out


def load(d):
    p = ROOT / d
    return [json.load(open(f, encoding="utf-8")) for f in sorted(p.glob("*.json"))] if p.exists() else []


def main():
    global N_BOOT
    ap = argparse.ArgumentParser(); ap.add_argument("--csp", default="results/units_csp"); ap.add_argument("--erasure", default="results/units_erasure05")
    ap.add_argument("--n-boot", type=int, default=2000); a = ap.parse_args()
    N_BOOT = a.n_boot; out = {"n_boot": N_BOOT, "seed": SEED}
    csp = load(a.csp)
    for conf in ("alpha", "saccade"):
        us = [r for r in csp if r["cell"]["confound"] == conf]
        if len(us) < 4: continue
        subs = A._test_subjects(us); mod = np.array(["csp"] * len(us))
        t_all = np.array([A.truth(r, subs, "all") for r in us]); cid = np.array([r["cid"] for r in us])
        res = np.full(len(us), np.nan); t0 = np.array([A.truth(r, subs, "half0") for r in us]); t1 = np.array([A.truth(r, subs, "half1") for r in us])
        r0, r1 = t0.copy(), t1.copy()
        for c in np.unique(cid):
            k = cid == c; res[k] = t_all[k] - t_all[k].mean(); r0[k] -= t0[k].mean(); r1[k] -= t1[k].mean()
        amp = np.array([float(r["cell"]["amp"]) for r in us]); D = np.array([r["probe"][f"{float(r['cell']['amp']):g}"]["D"] for r in us])
        out["csp_" + conf] = {"n_units": len(us),
            "mean_truth_by_amp": {f"{x:g}": float(t_all[amp == x].mean()) for x in np.unique(amp)},
            "mean_BA_matched": float(np.mean([A._ba(r["truth_counts"]["matched"]["all"], subs) for r in us])),
            "within_cell_sd_truth_pp": float(np.nanstd(res) * 100), "within_cell_splithalf_r": corr(r0, r1),
            "r_amplitude": corr(zw(t_all, mod), zw(amp, mod)), "r_D": corr(zw(t_all, mod), zw(D, mod)),
            "xfit": boot(us, metrics_csp(conf), mod)}
        print("csp", conf, json.dumps(out["csp_" + conf])[:600])
    er = load(a.erasure)
    for conf in ("alpha", "saccade"):
        us = [r for r in er if r["cell"]["confound"] == conf]
        if len(us) < 4: continue
        subs = A._test_subjects(us); mod = np.array([r["model"] for r in us])
        t = np.array([A.truth(r, subs, "all") for r in us]); f = metrics_er05()
        v = {n: np.array([fn(r, subs, "all") for r in us]) for n, fn in f.items()}
        neu = np.array([ba(r, ("erasure05", "counts_head_neutral"), subs, "all") - ba(r, ("erasure05", "counts_erased05_neutral"), subs, "all") for r in us])
        k = t > 0.05
        out["erasure_" + conf] = {"n_units": len(us), "mean_truth": float(t.mean()),
            **{n: {"mean_bias": float(np.mean(x - t)), "median_recovery_truth_gt_0.05": float(np.median(x[k] / t[k])) if k.any() else None} for n, x in v.items()},
            "erasure05_change_on_neutral_mean": float(np.mean(neu)), "xfit": boot(us, f, mod)}
        print("erasure", conf, json.dumps(out["erasure_" + conf])[:600])
    p = ROOT / "results/analysis/posthoc_revision11.json"; json.dump(out, open(p, "w"), indent=1); print("written", p)


if __name__ == "__main__":
    main()
