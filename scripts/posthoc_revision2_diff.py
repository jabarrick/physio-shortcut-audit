"""Paired cluster bootstrap: cross-fitted partial r of each metric minus that of the probe score D."""
import sys, json, warnings, numpy as np
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from p3audit.experiments import analysis as A
units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]
def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b); return np.corrcoef(a[k], b[k])[0, 1]
out = {}
for conf, metrics in A.DEPENDENCE_METRICS.items():
    us = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9 and r["cell"]["module"] in ("factorial", "p_series")]
    subs = A._test_subjects(us)
    keys = sorted({(r["model"], r["cid"]) for r in us}); gi = {k: i for i, k in enumerate(keys)}
    g = np.array([gi[(r["model"], r["cid"])] for r in us])
    X = {"t0": [A.truth(r, subs, "half0") for r in us], "t1": [A.truth(r, subs, "half1") for r in us],
         "D": [r["probe"][f"{float(r['cell']['amp']):g}"]["D"] for r in us]}
    for n, fn in metrics.items():
        X[n + "0"] = [fn(r, subs, "half0") for r in us]; X[n + "1"] = [fn(r, subs, "half1") for r in us]
    X = {k: np.array(v, float) for k, v in X.items()}
    members = [np.where(g == i)[0] for i in range(len(keys))]
    def stat(idx_groups):
        idx = np.concatenate(idx_groups); lab = np.concatenate([np.full(len(m), j) for j, m in enumerate(idx_groups)])
        def res(v):
            v = v[idx]; mu = np.bincount(lab, v) / np.bincount(lab); return v - mu[lab]
        R = {k: res(v) for k, v in X.items()}
        pD = np.mean([corr(R["t0"], R["D"]), corr(R["t1"], R["D"])])
        o = {"D": pD}
        for n in metrics:
            o[n] = np.mean([corr(R["t0"], R[n + "1"]), corr(R["t1"], R[n + "0"])])
        return o
    est = stat(members); rng = np.random.default_rng(11); B = []
    for _ in range(4000):
        B.append(stat([members[k] for k in rng.integers(0, len(keys), len(keys))]))
    res = {}
    for n in metrics:
        dif = np.array([b[n] - b["D"] for b in B]); dif = dif[np.isfinite(dif)]
        res[n] = {"xfit_partial": float(est[n]), "minus_D": float(est[n] - est["D"]),
                  "ci": [float(np.quantile(dif, .025)), float(np.quantile(dif, .975))],
                  "p_le0": float((dif <= 0).mean())}
    res["D"] = float(est["D"]); out[conf] = res
json.dump(out, open("results/analysis/posthoc_revision2_diff.json", "w"), indent=1); print(json.dumps(out, indent=1))
