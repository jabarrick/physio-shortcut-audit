"""Descriptive SHU-MI summary (supplementary H5; PILOT_LOG 17.22).  analyse_h5 reads only
results/real, so SHU records are summarised here.  Outside p3audit/ (frozen hash unaffected).
Collapsed run = one predicted class covers > 95% of test trials (reported, never dropped silently)."""
import glob, json, os, statistics as st
from collections import defaultdict

rows = []
for f in sorted(glob.glob("results/shu/*.json")):
    d = json.load(open(f, encoding="utf-8"))
    tp, fn, fp, tn = (sum(v[i] for v in d["counts"].values()) for i in range(4))   # counts order as stored
    n = tp + fn + fp + tn
    pred1 = (fn + tn) / n    # share of trials predicted as the second class (column order [.,1,.,1])
    rows.append({"model": d["model"], "fold": d["fold"], "seed": d["seed"], "ba": d["ba"],
                 "collapsed": max(pred1, 1 - pred1) > 0.95, "pred_share_max": max(pred1, 1 - pred1),
                 "sri_post": d["sri_post"], "erasure_keep": d["alpha_erasure"]["delta_ba_keep"],
                 "ig_alpha_share": d["ig_alpha"]["share"], "ig_frontal_delta_share": d["ig_frontal_delta"]["share"],
                 "qualitative_only": d["qualitative_only"]})
out = {"n_records": len(rows), "by_model": {}, "rows": rows,
       "note": "SHU-MI supplementary H5: descriptive only; LaBraM qualitative only (5 folds x 1 seed). "
               "Collapsed runs are listed; summaries are given with and without them."}
by = defaultdict(list)
for r in rows: by[r["model"]].append(r)
def ms(v): return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else None, "n": len(v)} if v else None
for m, rs in by.items():
    keep = [r for r in rs if not r["collapsed"]]
    out["by_model"][m] = {"n_runs": len(rs), "n_collapsed": sum(r["collapsed"] for r in rs),
                          "all": {k: ms([r[k] for r in rs]) for k in ("ba", "sri_post", "erasure_keep", "ig_alpha_share", "ig_frontal_delta_share")},
                          "non_collapsed": {k: ms([r[k] for r in keep]) for k in ("ba", "sri_post", "erasure_keep", "ig_alpha_share", "ig_frontal_delta_share")}}
os.makedirs("results/analysis", exist_ok=True)
json.dump(out, open("results/analysis/H5_shu.json", "w", encoding="utf-8"), indent=1)
for m, v in out["by_model"].items():
    a, k = v["all"]["ba"], v["non_collapsed"]["ba"]
    print(f"{m:8s} runs {v['n_runs']:2d} collapsed {v['n_collapsed']:2d}  BA all {a['mean']:.3f}±{(a['sd'] or 0):.3f}"
          + (f"  non-collapsed {k['mean']:.3f} (n={k['n']})" if k else "  non-collapsed: none"))
