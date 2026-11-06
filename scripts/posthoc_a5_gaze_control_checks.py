"""Checks without training of the Amendment 5 eye-tracking arms (post hoc).
(1) Composition of the exclusion training pool: trials kept without valid gaze, and the share of gaze-classified
    saccades among gaze-valid trials that the F8 - F7 detector called saccade-free (an estimate of the saccades left
    among the kept trials without valid gaze).
(2) Positive control for gaze_ols: per fold, the registered gaze regression (coefficients fitted on the fold's training
    trials with valid gaze) is applied to the 0.3-1.2 s deflections of all channels; within-subject cross-validated
    ridge readout of the gaze shift (within subject and cue side) and within-subject LDA readout of the cue side in
    gaze-congruent trials, before and after; trials without valid gaze are not corrected by the arm and are not used here.
Output: results/analysis/posthoc_a5_gaze_control_checks.json"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import amendment4_gf2025 as a4, amendment5_exclusion_gaze as m
km, T = m.gf_classes(); T["keep"] = [km[(int(s), int(k))] for s, k in zip(T.subject, T.trial)]
out = {"pool": {"kept": int(T.keep.sum()), "kept_gaze_valid": int((T.keep & T.gaze_valid).sum()), "kept_without_valid_gaze": int((T.keep & ~T.gaze_valid).sum()),
                "gaze_valid_trials_hor_none": int((T.gaze_valid & (T.type_hor == "none")).sum()),
                "of_which_gaze_saccade": int((T.gaze_valid & (T.type_hor == "none") & T.type_gaze.isin(["congruent", "incongruent"])).sum())}}
p = out["pool"]; p["share_gaze_saccade_among_hor_none"] = p["of_which_gaze_saccade"] / p["gaze_valid_trials_hor_none"]
p["estimated_saccade_trials_among_kept_without_gaze"] = p["kept_without_valid_gaze"] * p["share_gaze_saccade_among_hor_none"]
rows = []
for f in a4.folds():
    TR = m.gf_load(a4, f["train"]); R, ok = m.gaze_regressors(TR, a4); b = a4.fit_ols(TR["X"][ok], R[ok])
    TE = m.gf_load(a4, f["test"]); Rt, okt = m.gaze_regressors(TE, a4)
    dX = TE["X"][:, :, a4.idx(0.3, 1.2)].mean(-1) - TE["X"][:, :, a4.idx(-0.2, 0)].mean(-1)
    dR = Rt[:, :, a4.idx(0.3, 1.2)].mean(-1) - Rt[:, :, a4.idx(-0.2, 0)].mean(-1)
    after = dX - dR @ b.T
    for k in range(len(dX)):
        rows.append({"subject": int(TE["subj"][k]), "trial": int(TE["trial"][k]), "before": dX[k], "after": after[k], "corrected": bool(okt[k])})
D = pd.DataFrame(rows).merge(T[["subject", "trial", "label", "gaze_valid", "type_gaze", "u_gaze"]], on=["subject", "trial"])
V = D[D.gaze_valid & D.u_gaze.notna() & D.corrected].copy(); V["cue"] = np.where(V.label == 1, 1, -1); V["y"] = V.label
def auc(x, y):
    q, n = x[y == 1], x[y == 0]; return float((q[:, None] > n[None, :]).mean() + 0.5 * (q[:, None] == n[None, :]).mean())
def lda_auc(Y, y, seed):
    k = np.arange(len(y)) % 5; np.random.default_rng(seed).shuffle(k); sc = np.zeros(len(y))
    for j in range(5):
        tr, te = k != j, k == j
        if len(np.unique(y[tr])) < 2: return np.nan
        m1, m0 = Y[tr][y[tr] == 1].mean(0), Y[tr][y[tr] == 0].mean(0); Sw = np.cov((Y[tr] - np.where(y[tr][:, None] == 1, m1, m0)).T)
        Sw = 0.5 * Sw + 0.5 * np.trace(Sw) / len(Sw) * np.eye(len(Sw)); sc[te] = Y[te] @ np.linalg.solve(Sw, m1 - m0)
    return auc(sc, y)
def ridge_r2(df, col, lam=1.0):
    num = den = 0.0
    for s, x in df.groupby("subject"):
        if len(x) < 20: continue
        Y = np.stack(x[col].to_numpy()).copy(); u = x.u_gaze.to_numpy(float).copy(); cs = x.cue.to_numpy()
        for c in (-1, 1):
            mm = cs == c; Y[mm] -= Y[mm].mean(0); u[mm] -= u[mm].mean()
        Y = Y / (Y.std(0) + 1e-9); k = np.arange(len(u)) % 5; np.random.default_rng(int(s)).shuffle(k); pred = np.zeros_like(u)
        for j in range(5):
            a, b_ = k != j, k == j; A = Y[a]; w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ u[a]); pred[b_] = Y[b_] @ w
        num += ((u - pred) ** 2).sum(); den += (u ** 2).sum()
    return float(1 - num / den)
cg = V[V.type_gaze == "congruent"]
L = {c: float(np.nanmean([lda_auc(np.stack(x[c].to_numpy()), x.y.to_numpy(), int(s)) for s, x in cg.groupby("subject") if len(x) >= 15 and x.y.value_counts().min() >= 5])) for c in ("before", "after")}
out["gaze_ols_positive_control"] = {"n_trials": int(len(V)), "ridge_r2_before": ridge_r2(V, "before"), "ridge_r2_after": ridge_r2(V, "after"),
                                    "lda_auc_congruent_before": L["before"], "lda_auc_congruent_after": L["after"]}
json.dump(out, open("results/analysis/posthoc_a5_gaze_control_checks.json", "w"), indent=1); print(json.dumps(out, indent=1))
