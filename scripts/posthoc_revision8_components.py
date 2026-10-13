"""Post hoc (exploratory, not registered; 3 Oct 2026): what varies between seeds within a design cell.

Units and test-window labels as in posthoc_revision8_truth.py.  For every unit the recall of each (subject, label)
stratum in the matched, neutral and reversed test sets is regressed on the realised congruent fraction f = mean(z == y)
(strata as fixed effects, per label, then averaged).  This gives
  slope  = A+ - A-     accuracy on congruent minus incongruent windows (reliance; ground truth = (p - 0.5) x slope)
  skill  = recall at f = 0.5 = (A+ + A-) / 2     accuracy that does not depend on the coupling
  A+ = skill + slope / 2,  A- = skill - slope / 2
so that BA(matched) = skill + (p - 0.5) x slope up to the rounding of the stratum counts.

Reported per unit set, on the residuals from the cell x model means (seeds within a cell), with a cluster bootstrap over
cell x model groups (4000 resamples):
  * the coupling imbalance of the window-parity halves (deviation of f in each half from the full test set);
  * within-cell SD (pooled, df = units - groups) of BA(matched), BA(neutral), BA(reversed), skill, slope, A+ and A-;
  * correlations between them on all test windows, and the split of var BA(matched) into skill, reliance and covariance;
  * split-half r of the raw accuracies and of the coupling-corrected quantities.
Writes results/analysis/posthoc_revision8_components.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_revision5_bamatched import N_BOOT, SEED
from posthoc_revision8_truth import corr, build, unit_sets, test_labels, SETS

QS = ("M", "N", "R", "skill", "slope", "Aplus", "Aminus", "MR2")
PAIRS = (("N", "MR2"), ("skill", "slope"), ("Aplus", "Aminus"), ("M", "slope"), ("N", "slope"), ("R", "slope"), ("M", "N"), ("M", "skill"))


def arrays(U):
    X = {}
    for part in ("all", "half0", "half1"):
        g = lambda k: np.array([u[part][k] for u in U], float)
        d = {"M": g("M"), "N": g("N"), "R": g("R"), "skill": g("skill"), "slope": g("slope")}
        d["Aplus"] = d["skill"] + d["slope"] / 2; d["Aminus"] = d["skill"] - d["slope"] / 2; d["MR2"] = (d["M"] + d["R"]) / 2
        X.update({f"{k}|{part}": v for k, v in d.items()})
    return X


def stat(X, groups, p=0.9):
    idx = np.concatenate(groups); lab = np.concatenate([np.full(len(m), j) for j, m in enumerate(groups)])
    mu = lambda v: (np.bincount(lab, v[idx], minlength=len(groups)) / np.bincount(lab, minlength=len(groups)))[lab]
    R = {k: v[idx] - mu(v) for k, v in X.items()}; df = len(idx) - len(groups); o = {}
    var = lambda k: float((R[k + "|all"] ** 2).sum() / df); cov = lambda a, b: float((R[a + "|all"] * R[b + "|all"]).sum() / df)
    for k in QS:
        o["sd_points", k] = 100 * np.sqrt(var(k)); o["split_half_r", k] = corr(R[k + "|half0"], R[k + "|half1"])
    for a, b in PAIRS:
        o["r_all", f"{a}~{b}"] = corr(R[a + "|all"], R[b + "|all"])
    o["r_cross_half", "N~MR2"] = np.mean([corr(R["N|half0"], R["MR2|half1"]), corr(R["N|half1"], R["MR2|half0"])])
    o["r_cross_half", "skill~slope"] = np.mean([corr(R["skill|half0"], R["slope|half1"]), corr(R["skill|half1"], R["slope|half0"])])
    vM = var("M"); k = p - 0.5
    o["var_BA_matched_share", "skill"] = var("skill") / vM; o["var_BA_matched_share", "reliance"] = k * k * var("slope") / vM
    o["var_BA_matched_share", "2cov"] = 2 * k * cov("skill", "slope") / vM
    o["mean", "slope"] = float(np.mean(X["slope|all"][idx])); o["mean", "N-(M+R)/2"] = float(np.mean((X["N|all"] - X["MR2|all"])[idx]))
    return o


def run(units, conf, metrics, labels):
    _, groups, U = build(units, conf, metrics, labels); X = arrays(U); est = stat(X, groups)
    rng = np.random.default_rng(SEED); B = [stat(X, [groups[k] for k in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
    out = {"n_units": len(units), "n_groups": len(groups), "half_coupling_deviation": {}}
    for k in SETS:
        d0 = np.array([u["half0"]["f_dev"][k] for u in U]); d1 = np.array([u["half1"]["f_dev"][k] for u in U])
        out["half_coupling_deviation"][k] = {"sd_half0": float(d0.std()), "mean_abs": float(np.abs(np.r_[d0, d1]).mean()), "r_half0_half1": corr(d0, d1)}
    for key, v in est.items():
        b = np.array([x[key] for x in B], float); b = b[np.isfinite(b)]
        out.setdefault(key[0], {})[key[1]] = {"est": float(v), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
    return out


if __name__ == "__main__":
    S = unit_sets(); labels = test_labels(sorted({int(k) for us, _, _ in S.values() for r in us for k in r["truth_counts"]["matched"]["all"]}))
    res = {name: run(us, conf, mets, labels) for name, (us, conf, mets) in S.items()}
    json.dump(res, open("results/analysis/posthoc_revision8_components.json", "w"), indent=1)
    f = lambda d: f"{d['est']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"
    for name, v in res.items():
        print(f"\n{name}: {v['n_units']} units, {v['n_groups']} groups")
        print("  half coupling deviation:", {k: (round(d["sd_half0"], 4), round(d["r_half0_half1"], 3)) for k, d in v["half_coupling_deviation"].items()})
        for sec in ("sd_points", "split_half_r", "r_all", "r_cross_half", "var_BA_matched_share", "mean"):
            print(f"  {sec}: " + "; ".join(f"{k} {f(d)}" for k, d in v[sec].items()))
