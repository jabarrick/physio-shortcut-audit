"""Post hoc (exploratory; fifth review, point 7): specificity and detection.  No new runs.
Units: preregistered runs (results/units), weak-confound runs (results/units_weak) and Amendment 1 runs
(results/units_a1); frozen metric definitions (p3audit.experiments.analysis.DEPENDENCE_METRICS).
(a) Null behaviour: at p = 0.5 (label coupling 0.5, so the confound carries no label information and true reliance is
    about zero) and in the group-template control, the distribution of each metric on the accuracy scale (erasure,
    HEOG regression, SRI) and of the probe score D, and the share of units in which a metric exceeds the smallest effect
    size of interest (0.05).  In units without any injection only D is defined.
(b) Detection: area under the ROC curve for classifying units with Delta BA_neu > 0.05, per metric and confound, over all
    units in which the metric is defined; and the same within each confound amplitude (average of the per-amplitude
    AUCs, weighted by the number of units), which removes most of the design.  95% intervals from 2000 bootstrap
    resamples of units (seed 20261021).
Writes results/analysis/posthoc_revision12_specificity.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from p3audit.experiments import analysis as A
N_BOOT, SEED, SESOI = 2000, 20261021, 0.05


def load(d):
    return [json.load(open(p, encoding="utf-8")) for p in sorted((ROOT / d).glob("*.json"))]


def auc(score, lab):
    k = np.isfinite(score); s, l = score[k], lab[k]
    pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1.0
    # ties: average ranks
    allv = np.r_[pos, neg]; order = np.argsort(allv); ranks = np.empty(len(allv)); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strat_auc(score, lab, strata):
    vals, ws = [], []
    for s in np.unique(strata):
        k = strata == s; a = auc(score[k], lab[k])
        if np.isfinite(a): vals.append(a); ws.append(k.sum())
    return float(np.average(vals, weights=ws)) if vals else np.nan


def boot(fn, n, rng):
    b = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n); v = fn(i)
        if np.isfinite(v): b.append(v)
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))] if len(b) > 50 else None


pre = load("results/units"); weak = load("results/units_weak"); a1 = load("results/units_a1")
out = {"sesoi": SESOI, "n_boot": N_BOOT, "seed": SEED}
for conf in ("alpha", "saccade"):
    fns = {k: v for k, v in A.DEPENDENCE_METRICS[conf].items() if k in A.CONFIRMATORY[conf] or k == "heog_reg"}
    res = {}
    # (a) null behaviour
    for label, us in (("p0.5", [r for r in pre if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.5]),
                      ("group_template", [r for r in pre if r["cell"]["confound"] == conf and r["cell"]["module"] == "group_template"]),
                      ("no_injection", [r for r in pre if r["cell"]["confound"] == conf and r["cell"]["p"] is None])):
        if not us: continue
        subs = A._test_subjects(us); d = {"n_units": len(us)}
        t = np.array([A.truth(r, subs, "all") for r in us]) if label != "no_injection" else None
        if t is not None: d["truth_mean"] = float(np.nanmean(t)); d["truth_max"] = float(np.nanmax(t))
        Dv = np.array([r["probe"][f"{float(r['cell']['amp']):g}"]["D"] if f"{float(r['cell']['amp']):g}" in r["probe"] else r["probe"]["1"]["D"] for r in us])
        d["D"] = {"mean": float(Dv.mean()), "range": [float(Dv.min()), float(Dv.max())]}
        if label != "no_injection":
            for n, f in fns.items():
                v = np.array([f(r, subs, "all") for r in us]); v = v[np.isfinite(v)]
                if len(v) == 0: continue
                d[n] = {"mean": float(v.mean()), "q95": float(np.quantile(v, .95)), "share_above_sesoi": float(np.mean(v > SESOI))} if n != "ig_share" else {"mean": float(v.mean()), "q95": float(np.quantile(v, .95))}
        res["null_" + label] = d
    # (b) detection
    us = [r for r in pre + weak + a1 if r["cell"]["confound"] == conf and r["cell"]["p"] is not None and r["cell"]["module"] in ("factorial", "p_series", "group_template", "mismatch")]
    subs = A._test_subjects(us)
    t = np.array([A.truth(r, subs, "all") for r in us]); lab = t > SESOI
    amp = np.array([float(r["cell"]["amp"]) for r in us])
    sc = {n: np.array([f(r, subs, "all") for r in us]) for n, f in fns.items()}
    sc["D"] = np.array([r["probe"].get(f"{float(r['cell']['amp']):g}", {}).get("D", np.nan) for r in us])
    sc["amplitude"] = amp.copy()
    rng = np.random.default_rng(SEED); det = {"n_units": len(us), "n_positive": int(lab.sum())}
    for n, s in sc.items():
        a_all = auc(s, lab); a_str = strat_auc(s, lab, amp) if n != "amplitude" else np.nan
        det[n] = {"n": int(np.isfinite(s).sum()), "auc": a_all, "ci95": boot(lambda i: auc(s[i], lab[i]), len(s), rng),
                  "auc_within_amplitude": a_str,
                  "ci95_within": boot(lambda i: strat_auc(s[i], lab[i], amp[i]), len(s), rng) if n != "amplitude" else None}
    res["detection"] = det
    out[conf] = res
    print("==", conf); print(json.dumps(res, indent=0)[:2500])
json.dump(out, open(ROOT / "results/analysis/posthoc_revision12_specificity.json", "w"), indent=1)
