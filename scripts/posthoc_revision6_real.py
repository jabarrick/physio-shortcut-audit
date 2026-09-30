"""Post hoc (exploratory): summaries for scripts/posthoc_revision6_controls.py and the A2-R3 accuracy table.
Deflection, terciles, seed averaging, model pooling and subject bootstrap as in scripts/amendment2_real.py
(2000 resamples, seed 20261008; 95% intervals, no decisions).
(1) A2-R3 table (needs only existing records): accuracy per model and tercile of trials without a detected saccade,
    and all no-saccade trials, for main, heog_ols, main_late and heog_late.
(2) Decoders restricted to channel groups (frontal_loc from real_a2; post_loc, post_loc_late, cent_loc,
    frontal_loc_late, frontal_loc_heog from real_r6): accuracy (all trials), and highest-minus-lowest tercile
    accuracy in trials without a detected saccade.
(3) Controls: G(main, arm) for heog_ols, plc_shuf, plc_orth, plc_rev and the differences from G(main, heog_ols).
Arms that are not yet available are skipped.  --models restricts every arm (and main) to a subset of models;
the output name then carries the models, e.g. posthoc_revision6_real__csoanet_eegnet.json.
Writes results/analysis/posthoc_revision6_real.json.
"""
import argparse, glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from p3audit.constants import PHYSIONET_CHANNELS as CH

N_BOOT, SEED = 2000, 20261008
ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=None); ARGS = ap.parse_args()
i7, i8 = CH.index("AF7"), CH.index("AF8")
key = ["model", "subject", "trial", "type", "label"]


def load(pattern):
    fs = glob.glob(pattern)
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame(columns=key + ["arm", "correct"])


main = load("results/real/*_trials.csv"); main = main[main.arm == "main"]
if ARGS.models: main = main[main.model.isin(ARGS.models)]
src = pd.concat([load("results/real_r3/*_trials.csv"), load("results/real_a2/*_trials.csv"), load("results/real_r6/*_trials.csv")])
defl = {}
for s in sorted(main.subject.unique()):
    z = np.load(f"results/real_cache/sub-{s:03d}.npz", allow_pickle=True)
    h = z["X"][:, i7, :] - z["X"][:, i8, :]; p = int(z["pad"])
    v = (h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)) * np.where(z["y"] == 1, 1.0, -1.0)
    for k in range(len(v)):
        defl[(s, k)] = v[k]
prep = lambda df, name: df.assign(correct=df.correct.astype(float)).groupby(key, as_index=False).correct.mean().rename(columns={"correct": name})
wide = prep(main, "main")
ARMS = ["heog_ols", "plc_shuf", "main_late", "heog_late", "frontal_loc", "post_loc", "post_loc_late", "cent_loc",
        "frontal_loc_late", "frontal_loc_heog", "plc_orth", "plc_rev"]
present = []
for a in ARMS:
    df = src[(src.arm == a) & src.model.isin(main.model.unique())]
    if len(df) == 0 or set(df.model.unique()) != set(main.model.unique()) or df.groupby("model").seed.nunique().min() < 3:
        continue   # use an arm only when every model has all three seeds (all folds are checked by the merge below)
    t = prep(df, a)[key + [a]]
    if len(t) != len(wide):
        continue
    wide = wide.merge(t, on=key); present.append(a)
wide["d"] = [defl[(s, k)] for s, k in zip(wide.subject, wide.trial)]
if wide.loc[wide.type == "congruent", "d"].mean() < 0:
    wide["d"] *= -1
wide["dz"] = wide.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
n = wide.type == "none"
wide.loc[n, "tert"] = wide[n].groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))
out = {"arms_present": present, "n_trials": int(len(wide)), "n_subjects": int(wide.subject.nunique())}

# (1) A2-R3 table
tab = {}
for a in [x for x in ("main", "heog_ols", "main_late", "heog_late") if x == "main" or x in present]:
    nn = wide[n]
    g = nn.groupby(["model", "tert"])[a].mean().unstack()
    g["all_no_saccade"] = nn.groupby("model")[a].mean()
    tab[a] = {m: {("low", "middle", "high")[int(k)] if k in (0, 1, 2) else k: float(v) for k, v in row.items()} for m, row in g.iterrows()}
out["a2r3_table"] = tab

DEC = [a for a in ("frontal_loc", "post_loc", "post_loc_late", "cent_loc", "frontal_loc_late", "frontal_loc_heog") if a in present]
CTRL = [a for a in ("heog_ols", "plc_shuf", "plc_orth", "plc_rev") if a in present]


def stats(w):
    nn = w[w.type == "none"]; t0, t2 = nn[nn.tert == 0], nn[nn.tert == 2]
    o = {}
    for a in DEC:
        o[f"{a}_acc"] = w[a].mean(); o[f"{a}_acc_none"] = nn[a].mean(); o[f"{a}_gradient"] = t2[a].mean() - t0[a].mean()
    for a in CTRL:
        o[f"G_{a}"] = (t2.main - t2[a]).mean() - (t0.main - t0[a]).mean()
    for a in CTRL:
        if a != "heog_ols" and "heog_ols" in CTRL:
            o[f"G_heog_ols-G_{a}"] = o["G_heog_ols"] - o[f"G_{a}"]
    return o


rng = np.random.default_rng(SEED); subs = np.array(sorted(wide.subject.unique())); gs = {s: x for s, x in wide.groupby("subject")}
est = stats(wide); B = pd.DataFrame([stats(pd.concat([gs[s] for s in rng.choice(subs, subs.size)])) for _ in range(N_BOOT)])
out["estimates"] = {k: {"est": float(v), "ci_95": [float(B[k].quantile(.025)), float(B[k].quantile(.975))]} for k, v in est.items()}
out["per_model_acc"] = {a: wide.groupby("model")[a].mean().round(4).to_dict() for a in DEC}
out["models"] = sorted(main.model.unique())
fn = "results/analysis/posthoc_revision6_real" + ("__" + "_".join(sorted(ARGS.models)) if ARGS.models else "") + ".json"
json.dump(out, open(fn, "w"), indent=1)
print("arms:", present)
for k, v in out["estimates"].items():
    print(f"{k:32s} {v['est']:.3f} [{v['ci_95'][0]:.3f}, {v['ci_95'][1]:.3f}]")
