"""Eye-tracking dataset (Amendment 4), post hoc, no training.

(1) Positive control for the F8 - F7 OLS control: how much eye-movement-related signal is left in the input after the
    registered OLS regression (coefficients fitted on each fold's training subjects, as for the control models)?
    a. per-channel slope of the 0.3-1.2 s deflection on the measured horizontal gaze shift, within subject and cue side,
       before and after regression; residual fraction = norm of the slopes after / before;
    b. within-subject cross-validated ridge readout of the gaze shift (within subject and cue side) from the 60
       channel deflections, before and after regression (R^2);
    c. in gaze-congruent trials, per-subject AUC for the cue side of Fp1 - Fp2, AF3 - AF4, FT7 - FT8 and of a
       within-subject cross-validated shrinkage-LDA readout of all channels, before and after regression.
(2) Reliability in the 382 gaze-valid trials without a measured saccade: split-half reliability of the gaze shift and
    of the F8 - F7 deflection within this subset, their correlation, a joint regression of the trial gap on both, and
    the tercile gradient of the F8 - F7 deflection after residualising it on the gaze shift.
(3) Intervals: all-trial main-minus-control accuracy (98.75 %), tercile accuracies (95 %).
(4) Microsaccade detection rate by trial type (sensitivity reference for the detector of the round-6 analysis).
Output: results/analysis/posthoc_a4_control_leakage_reliability.json
"""
import sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import amendment4_gf2025 as a
rng = np.random.default_rng(20261102); NB = 2000
EEG = a.EEG; ci = lambda v, al=0.05: [float(np.nanquantile(v, al / 2)), float(np.nanquantile(v, 1 - al / 2))]

T = a.trial_table()                                   # registered trial classification and deflections
F = a.folds(); fold_of = {s: f["fold"] for f in F for s in f["test"]}
# ---------- per-subject sufficient statistics
D, S = {}, {}
for s in a.subjects_in_cache():
    z = np.load(a.CACHE / f"sub-{s:02d}.npz"); X = z["X"].astype(float); H = z["HOR"].astype(float)
    Xc = X - X.mean(-1, keepdims=True); Hc = H - H.mean(-1, keepdims=True)
    S[s] = (np.einsum("nct,nt->c", Xc, Hc), float((Hc ** 2).sum()))
    dX = X[:, :, a.idx(*a.W_DEFL)].mean(-1) - X[:, :, a.idx(*a.W_BASE)].mean(-1)
    dH = H[:, a.idx(*a.W_DEFL)].mean(-1) - H[:, a.idx(*a.W_BASE)].mean(-1)
    D[s] = (dX, dH)
check = []
rows = []
for f in F:
    C = sum(S[s][0] for s in f["train"]); G = sum(S[s][1] for s in f["train"]); b = C / G
    check.append(float(b[EEG.index("F8")] - b[EEG.index("F7")]))
    for s in f["test"]:
        dX, dH = D[s]; R = dX - np.outer(dH, b)
        for k in range(len(dH)):
            rows.append({"subject": s, "trial": k, "before": dX[k], "after": R[k]})
B = pd.DataFrame(rows).merge(T[["subject", "trial", "label", "gaze_valid", "type_gaze", "u_gaze"]], on=["subject", "trial"])
out = {"check_b_F8_minus_b_F7_per_fold": check}
V = B[B.gaze_valid & B.u_gaze.notna()].copy()
V["cue"] = np.where(V.label == 1, 1, -1)

def slopes(df, col):
    """Per-channel slope of the deflection on the gaze shift, within subject x cue side."""
    Y = np.stack(df[col].to_numpy()); u = df.u_gaze.to_numpy(float); grp = (df.subject.astype(str) + "_" + df.cue.astype(str)).to_numpy()
    Yd = Y - pd.DataFrame(Y).groupby(grp).transform("mean").to_numpy(); ud = u - pd.Series(u).groupby(grp).transform("mean").to_numpy()
    return (Yd * ud[:, None]).sum(0) / (ud ** 2).sum()
def resid_frac(df):
    b0, b1 = slopes(df, "before"), slopes(df, "after"); return float(np.linalg.norm(b1) / np.linalg.norm(b0)), b0, b1
est, b0, b1 = resid_frac(V)
subs = np.array(sorted(V.subject.unique())); g = {s: x for s, x in V.groupby("subject")}
boot = [resid_frac(pd.concat([g[s].assign(subject=f"{s}_{i}") for i, s in enumerate(rng.choice(subs, subs.size))]))[0] for _ in range(500)]
order = np.argsort(-np.abs(b1))[:6]
out["leak_slope"] = {"residual_fraction": est, "ci95": ci(boot), "n_trials": int(len(V)), "n_subjects": int(subs.size), "n_boot": 500,
                     "slope_uV_per_deg_before": {c: round(float(v), 3) for c, v in zip(EEG, b0)},
                     "slope_uV_per_deg_after": {c: round(float(v), 3) for c, v in zip(EEG, b1)},
                     "largest_after": {EEG[i]: [round(float(b1[i]), 3), round(float(b0[i]), 3)] for i in order}}
# b. ridge readout of the gaze shift within subject x cue side
def ridge_r2(df, col, lam=1.0):
    num = den = 0.0
    for s, x in df.groupby("subject"):
        if len(x) < 20: continue
        Y = np.stack(x[col].to_numpy()); u = x.u_gaze.to_numpy(float); cs = x.cue.to_numpy()
        for c in (-1, 1):
            m = cs == c; Y[m] -= Y[m].mean(0); u = u.copy(); u[m] -= u[m].mean()
        Y = Y / (Y.std(0) + 1e-9); k = np.arange(len(u)) % 5; np.random.default_rng(s).shuffle(k); pred = np.zeros_like(u)
        for j in range(5):
            tr, te = k != j, k == j; A = Y[tr]; w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ u[tr]); pred[te] = Y[te] @ w
        num += ((u - pred) ** 2).sum(); den += (u ** 2).sum()
    return float(1 - num / den)
out["leak_ridge_r2_gaze_shift"] = {"before": ridge_r2(V, "before"), "after": ridge_r2(V, "after"),
                                   "note": "within subject and cue side; 5-fold CV; ridge lambda = 1 x n on standardised channels"}
# c. AUC for the cue side in gaze-congruent and no-saccade trials
def auc(x, y):
    p, n = x[y == 1], x[y == 0]; return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
pairs = {"Fp1-Fp2": ("Fp1", "Fp2"), "AF3-AF4": ("AF3", "AF4"), "FT7-FT8": ("FT7", "FT8"), "F7-F8": ("F7", "F8")}
def lda_auc(x, col):
    Y = np.stack(x[col].to_numpy()); y = x.label.to_numpy(); k = np.arange(len(y)) % 5; np.random.default_rng(int(x.subject.iloc[0])).shuffle(k); sc = np.zeros(len(y))
    for j in range(5):
        tr, te = k != j, k == j
        if len(np.unique(y[tr])) < 2: return np.nan
        m1, m0 = Y[tr][y[tr] == 1].mean(0), Y[tr][y[tr] == 0].mean(0); Sw = np.cov((Y[tr] - np.where(y[tr][:, None] == 1, m1, m0)).T)
        Sw = 0.5 * Sw + 0.5 * np.trace(Sw) / len(Sw) * np.eye(len(Sw)); sc[te] = Y[te] @ np.linalg.solve(Sw, m1 - m0)
    return auc(sc, y)
res = {}
for ty in ("congruent", "none"):
    x0 = V[V.type_gaze == ty]; r = {}
    for nm, (l, rr) in pairs.items():
        il, ir = EEG.index(l), EEG.index(rr)
        for col in ("before", "after"):
            A = [auc(np.stack(x[col].to_numpy())[:, ir] - np.stack(x[col].to_numpy())[:, il], x.label.to_numpy()) for _, x in x0.groupby("subject") if x.label.nunique() == 2 and len(x) >= 10]
            r[f"{nm}_{col}"] = round(float(np.mean(A)), 3)
    for col in ("before", "after"):
        A = [lda_auc(x, col) for _, x in x0.groupby("subject") if x.label.nunique() == 2 and len(x) >= 15 and x.label.value_counts().min() >= 5]
        r[f"LDA_all_channels_{col}"] = round(float(np.nanmean(A)), 3); r["n_subjects_LDA"] = int(np.isfinite(A).sum())
    r["n_trials"] = int(len(x0)); res[ty] = r
out["auc_cue_side_right_minus_left"] = res

# ---------- (2) reliability etc. in gaze-valid trials without a measured saccade
recs = pd.concat([pd.read_csv(f) for f in glob.glob("results/gf2025/*_trials.csv")])
W = recs[recs.arm.isin(["main", "hor_ols"])].groupby(["arm", "model", "subject", "trial"]).correct.mean().unstack("arm").reset_index()
W["gap"] = W["main"] - W["hor_ols"]
Wt = W.groupby(["subject", "trial"])[["main", "hor_ols", "gap"]].mean().reset_index().merge(T, on=["subject", "trial"])
N = Wt[Wt.gaze_valid & (Wt.type_gaze == "none")].copy()
z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
rel = {}
for c in ("hor", "gaze"):
    x = N[[f"h1_{c}", f"h2_{c}", "subject"]].dropna(); zz = x.groupby("subject").transform(z)
    r = float(np.corrcoef(zz[f"h1_{c}"], zz[f"h2_{c}"])[0, 1]); rel[c] = {"r_halves": r, "spearman_brown": 2 * r / (1 + r), "n": int(len(x))}
N["zh"] = N.groupby("subject").d_hor.transform(z); N["zg"] = N.groupby("subject").d_gaze.transform(z)
out["no_saccade_subset"] = {"n_trials": int(len(N)), "split_half": rel,
    "corr_zgaze_zhor": float(N[["zh", "zg"]].corr().iloc[0, 1]),
    "sd_gaze_shift_deg": float(N.d_gaze.std()), "mean_gaze_shift_deg": float(N.d_gaze.mean()),
    "sd_hor_uV": float(N.d_hor.std())}
def joint(x):
    M = x[["zh", "zg", "gap"]].dropna(); A = np.c_[np.ones(len(M)), M.zh, M.zg]; return np.linalg.lstsq(A, M.gap.to_numpy(), rcond=None)[0][1:]
def Gres(x):
    M = x[["zh", "zg", "gap"]].dropna().copy(); A = np.c_[np.ones(len(M)), M.zg]; M["res"] = M.zh - A @ np.linalg.lstsq(A, M.zh.to_numpy(), rcond=None)[0]
    t = pd.qcut(M.res, 3, labels=False); return M.gap[t == 2].mean() - M.gap[t == 0].mean()
gN = {s: x for s, x in N.groupby("subject")}; sN = np.array(sorted(gN))
bj, bg = [], []
for _ in range(NB):
    xx = pd.concat([gN[s] for s in rng.choice(sN, sN.size)]); bj.append(joint(xx)); bg.append(Gres(xx))
bj = np.array(bj); ej = joint(N)
out["no_saccade_subset"]["joint_regression_gap_per_sd"] = {"hor": [float(ej[0])] + ci(bj[:, 0]), "gaze": [float(ej[1])] + ci(bj[:, 1])}
out["no_saccade_subset"]["G_hor_residualised_on_gaze"] = [float(Gres(N))] + ci(np.array(bg))
# ---------- (3) intervals
gA = {s: x for s, x in Wt.groupby("subject")}; sA = np.array(sorted(gA))
f_all = lambda x: (x["main"] - x["hor_ols"]).mean()
bA = [f_all(pd.concat([gA[s] for s in rng.choice(sA, sA.size)])) for _ in range(NB)]
out["all_trial_gap_mean_correctness"] = [float(f_all(Wt))] + ci(np.array(bA), 0.0125)
N["tert"] = N.tert_d_hor
def terc(x): return [x[x.tert == k][c].mean() for k in (0, 2) for c in ("main", "hor_ols")]
bt = np.array([terc(pd.concat([gN[s].assign(tert=gN[s].tert_d_hor) for s in rng.choice(sN, sN.size)])) for _ in range(NB)])
et = terc(N)
out["tercile_accuracy_hor"] = {"lowest_main": [et[0]] + ci(bt[:, 0]), "lowest_control": [et[1]] + ci(bt[:, 1]),
                               "highest_main": [et[2]] + ci(bt[:, 2]), "highest_control": [et[3]] + ci(bt[:, 3])}
json.dump(out, open("results/analysis/posthoc_a4_control_leakage_reliability.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("leak_slope",)}, indent=1)[:4000]); print(out["leak_slope"]["residual_fraction"], out["leak_slope"]["ci95"], out["leak_slope"]["largest_after"])
