"""Tercile analysis for the round-3 control arms (exploratory).  Run from the repository root after
scripts/posthoc_revision3_controls.py.  Same deflection and tercile definition as
posthoc_revision2_real.py; subject cluster bootstrap (2000).

For each control arm (heog_ols, plc_P7P8, plc_O1O2) and model: accuracy of main and arm, and the
main - arm gap, by trial type and by tercile of the cue-signed AF7 - AF8 deflection in trials without a
detected saccade.  A placebo arm that shows the same tercile gradient as heog_ols would point to
over-subtraction of lateralised neural signal rather than to eye movements.
For the frontal arm: its own accuracy overall, by trial type and by tercile.
Writes results/analysis/posthoc_revision3_terciles.json.
"""
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from p3audit.constants import PHYSIONET_CHANNELS as CH

i7, i8 = CH.index("AF7"), CH.index("AF8")
main = pd.concat([pd.read_csv(f) for f in glob.glob("results/real/*_trials.csv")])
main = main[main.arm == "main"]
arms = pd.concat([pd.read_csv(f) for f in glob.glob("results/real_r3/*_trials.csv")])
key = ["model", "subject", "trial", "type", "label"]

defl = {}
for s in sorted(main.subject.unique()):
    z = np.load(f"results/real_cache/sub-{s:03d}.npz", allow_pickle=True)
    h = z["X"][:, i7, :] - z["X"][:, i8, :]; p = int(z["pad"])
    v = (h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)) * np.where(z["y"] == 1, 1.0, -1.0)
    for k in range(len(v)):
        defl[(s, k)] = v[k]


def prep(df):
    t = df.groupby(key, as_index=False).correct.mean()
    t["d"] = [defl[(s, k)] for s, k in zip(t.subject, t.trial)]
    return t


M = prep(main)
flip = 1.0 if M.loc[M.type == "congruent", "d"].mean() > 0 else -1.0
rng = np.random.default_rng(5)
out = {}


def tert(df):
    df = df.copy(); df["d"] *= flip
    df["dz"] = df.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    n = df.type == "none"
    df.loc[n, "tert"] = df[n].groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))
    return df


def boot(df, fn):
    subs = np.array(sorted(df.subject.unique())); g = {s: x for s, x in df.groupby("subject")}
    est = fn(df); B = []
    for _ in range(2000):
        B.append(fn(pd.concat([g[s] for s in rng.choice(subs, subs.size)])))
    B = pd.DataFrame(B)
    return {k: {"est": float(v), "ci": [float(B[k].quantile(.025)), float(B[k].quantile(.975))]} for k, v in est.items()}


def gap_stats(w):
    o = {}
    for ty in ("congruent", "incongruent", "none"):
        x = w[w.type == ty]; o[f"gap_{ty}"] = (x.main - x.arm).mean()
    for k in (0, 1, 2):
        x = w[(w.type == "none") & (w.tert == k)]
        o[f"main_t{k}"] = x.main.mean(); o[f"arm_t{k}"] = x.arm.mean(); o[f"gap_t{k}"] = (x.main - x.arm).mean()
    o["gap_t2_minus_t0"] = o["gap_t2"] - o["gap_t0"]
    return o


def own_stats(w):
    o = {"acc_all": w.arm.mean()}
    for ty in ("congruent", "incongruent", "none"):
        o[f"acc_{ty}"] = w[w.type == ty].arm.mean()
    for k in (0, 1, 2):
        o[f"acc_t{k}"] = w[(w.type == "none") & (w.tert == k)].arm.mean()
    return o


for arm in sorted(arms.arm.unique()):
    A = prep(arms[arms.arm == arm]).rename(columns={"correct": "arm"})
    for mdl in sorted(A.model.unique()):
        w = M[M.model == mdl].rename(columns={"correct": "main"}).merge(A[A.model == mdl][key + ["arm"]], on=key)
        w = tert(w)
        out.setdefault(arm, {})[mdl] = boot(w, own_stats if arm == "frontal" else gap_stats)
        out[arm][mdl]["n_trials"] = int(len(w))
json.dump(out, open("results/analysis/posthoc_revision3_terciles.json", "w"), indent=1)
for arm, mm in out.items():
    for mdl, v in mm.items():
        keys = ("acc_all", "acc_congruent", "acc_none", "acc_t0", "acc_t2") if arm == "frontal" else ("gap_t0", "gap_t1", "gap_t2", "gap_t2_minus_t0", "gap_congruent")
        print(arm, mdl, {k: (round(v[k]["est"], 3), [round(x, 3) for x in v[k]["ci"]]) for k in keys})


# ---------------------------------------------------------------- registered tests (Amendment 1)
# A1-R1 (placebo): pooled over models, the tercile gradient of the main - control gap
#   G(arm) = gap_t2 - gap_t0 is larger for heog_ols than for plc_P7P8:  G(heog_ols) - G(plc_P7P8) > 0.
# A1-R2 (frontal-only): pooled over models, frontal-only accuracy is above chance (acc_all > 0.5) and
#   rises across terciles of the deflection in trials without a detected saccade (acc_t2 - acc_t0 > 0).
# Decision: lower bound of the subject-bootstrap 95% interval > 0 (for acc_all: > 0.5).
wide = None
for arm in ("heog_ols", "plc_P7P8", "frontal"):
    if arm not in set(arms.arm):
        continue
    A = prep(arms[arms.arm == arm]).rename(columns={"correct": arm})[key + [arm]]
    wide = (M.rename(columns={"correct": "main"}) if wide is None else wide).merge(A, on=key)
if wide is not None and {"heog_ols", "plc_P7P8", "frontal"} <= set(wide.columns):
    wide = tert(wide)

    def reg_stats(w):
        o = {}
        nn = w[w.type == "none"]
        G = lambda arm: ((nn[nn.tert == 2].main - nn[nn.tert == 2][arm]).mean() - (nn[nn.tert == 0].main - nn[nn.tert == 0][arm]).mean())
        o["G_heog_ols"], o["G_plc_P7P8"] = G("heog_ols"), G("plc_P7P8")
        o["R1_diff"] = o["G_heog_ols"] - o["G_plc_P7P8"]
        o["R2_acc_all"] = w.frontal.mean()
        o["R2_gradient"] = nn[nn.tert == 2].frontal.mean() - nn[nn.tert == 0].frontal.mean()
        return o
    res = boot(wide, reg_stats)   # pooled over models: each model's trials enter once
    res["decisions"] = {"A1-R1": res["R1_diff"]["ci"][0] > 0,
                        "A1-R2": res["R2_acc_all"]["ci"][0] > 0.5 and res["R2_gradient"]["ci"][0] > 0}
    json.dump(res, open("results/analysis/amendment1_real.json", "w"), indent=1, default=bool)
    print("registered:", {k: v for k, v in res.items()})
