"""Round-2 post hoc real-data analyses (exploratory; PILOT_LOG 17.29). Run from repo root.
(a) main - control accuracy gap by trial type, incl. incongruent trials (negative control for
    cue-directed saccade use); (b) in trials without a detected saccade, whether the gap grows with a
    sub-threshold HEOG deflection towards the cued side. Subject cluster bootstrap (2000)."""
import glob, json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from p3audit.constants import PHYSIONET_CHANNELS as CH
i7, i8 = CH.index("AF7"), CH.index("AF8")
d = pd.concat([pd.read_csv(f) for f in glob.glob("results/real/*_trials.csv")])
t = d.groupby(["model", "arm", "subject", "trial", "type", "label"], as_index=False).correct.mean()
w = t.pivot_table(index=["model", "subject", "trial", "type", "label"], columns="arm", values="correct").reset_index()
# sub-threshold HEOG deflection 0.3-1.2 s after cue minus -0.2-0 s, signed towards the cued side
defl = {}
for s in sorted(w.subject.unique()):
    z = np.load(f"results/real_cache/sub-{s:03d}.npz", allow_pickle=True)
    h = z["X"][:, i7, :] - z["X"][:, i8, :]; p = int(z["pad"])
    v = h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)
    sg = np.where(z["y"] == 1, 1.0, -1.0)
    for k in range(len(v)):
        defl[(s, k)] = v[k] * sg[k]
w["d"] = [defl[(s, k)] for s, k in zip(w.subject, w.trial)]
if w.loc[w.type == "congruent", "d"].mean() < 0:
    w["d"] = -w["d"]
# standardise d within subject (amplitude scales differ between subjects)
w["dz"] = w.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
w["gap"] = w["main"] - w["control"]
subs = np.array(sorted(w.subject.unique())); rng = np.random.default_rng(3)
none = w[w.type == "none"].copy()
q = none.groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))
none["tert"] = q
def stats(ww, nn):
    o = {}
    for ty in ("congruent", "incongruent", "none"):
        x = ww[ww.type == ty]; o[f"main_{ty}"] = x["main"].mean(); o[f"ctrl_{ty}"] = x["control"].mean(); o[f"gap_{ty}"] = x.gap.mean()
    o["inc_minus_none"] = o["gap_incongruent"] - o["gap_none"]; o["con_minus_none"] = o["gap_congruent"] - o["gap_none"]
    for k in (0, 1, 2):
        x = nn[nn.tert == k]; o[f"gap_none_t{k}"] = x.gap.mean(); o[f"main_none_t{k}"] = x["main"].mean(); o[f"ctrl_none_t{k}"] = x["control"].mean()
    o["gap_t2_minus_t0"] = o["gap_none_t2"] - o["gap_none_t0"]
    b = np.polyfit(nn.dz, nn.gap, 1)[0] if len(nn) > 10 else np.nan; o["gap_slope_per_sd"] = b
    o["main_slope_per_sd"] = np.polyfit(nn.dz, nn["main"], 1)[0]; o["ctrl_slope_per_sd"] = np.polyfit(nn.dz, nn["control"], 1)[0]
    return o
out = {}
for m in sorted(w.model.unique()):
    wm, nm = w[w.model == m], none[none.model == m]
    est = stats(wm, nm)
    gs, gn = {s: g for s, g in wm.groupby("subject")}, {s: g for s, g in nm.groupby("subject")}
    B = []
    for _ in range(2000):
        pick = rng.choice(subs, subs.size)
        B.append(stats(pd.concat([gs[s] for s in pick if s in gs]), pd.concat([gn[s] for s in pick if s in gn])))
    B = pd.DataFrame(B)
    out[m] = {k: {"est": float(v), "ci": [float(B[k].quantile(.025)), float(B[k].quantile(.975))]} for k, v in est.items()}
    out[m]["n_trials"] = {ty: int((wm.type == ty).sum()) for ty in ("congruent", "incongruent", "none")}
out["d_by_type_mean"] = w[w.model == w.model.iloc[0]].groupby("type").d.mean().to_dict()
json.dump(out, open("results/analysis/posthoc_revision2_real.json", "w"), indent=1)
for m in [k for k in out if k != "d_by_type_mean"]:
    print(m, {k: (round(v["est"], 3), [round(x, 3) for x in v["ci"]]) for k, v in out[m].items() if k != "n_trials"}, out[m]["n_trials"])
print(out["d_by_type_mean"])
