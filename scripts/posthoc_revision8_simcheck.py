"""Post hoc (exploratory, not registered; 3 Oct 2026): simulation check of the coupling correction in
posthoc_revision8_truth.py.

Per-window predictions were not stored, so the registered window-parity halves cannot be re-drawn with an exact coupling
in each half.  This script asks whether the fixed-effects slope on the realised congruent fraction (neu_fe, all_fe)
recovers what such halves would give.  It keeps the real test windows of the Amendment 1 saccade units (labels,
subjects, halves and the stored z of the three test sets) and simulates only the predictions: each unit has a skill and
a reliance (slope = A+ - A-) drawn around its cell x model mean (within-cell SD and correlation near the observed
values), each subject an offset to both, and a window is classified correctly when a uniform draw, shared by the three
test sets, falls below A+ (z congruent with y) or A- (incongruent).  For a share SHARED of the windows the draw is common
to all units (window difficulty on which runs agree), which lowers the noise between seeds towards the observed level.

Compared, per half and per replicate, on the residuals from the cell x model means:
  parity halves, stored z :  neu (registered), rev, nr, neu_fe, all_fe
  stratified halves       :  neu with z re-assigned exactly within (subject, label, half)   (the reference)
Reported: split-half r, the correlation of the mean of the two halves with the true reliance, and the cross-fitted r of
BA(matched) of one half with the ground truth of the other.  Mean and 2.5-97.5 percentiles over replicates.
Writes results/analysis/posthoc_revision8_simcheck.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_revision8_truth import corr, build, unit_sets, test_labels

N_REP, SEED = 200, 20261009
SD_SKILL, SD_SLOPE, RHO = 0.013, 0.049, -0.8        # between seeds within a cell, as observed (posthoc_revision8_components.json)
SD_SUBJ_SKILL, SD_SUBJ_SLOPE = 0.03, 0.05           # between test subjects (assumed)
SHARED = 0.6                                        # share of windows whose difficulty draw is common to all units (assumed)
KEYS = ("neu", "rev", "nr", "neu_fe", "all_fe", "neu_stratified")


def ba(correct, y, m):
    return 0.5 * (correct[m & (y == 0)].mean() + correct[m & (y == 1)].mean())


def fe(correct, cong, st, y_of_st, m, ks):
    """Fixed-effects slope of stratum recall on the congruent fraction, per label, then averaged (as fe_slope)."""
    n = np.bincount(st[m], minlength=y_of_st.size).astype(float); keep = n > 0
    r = np.stack([np.bincount(st[m], correct[k][m], minlength=n.size)[keep] / n[keep] for k in ks])
    f = np.stack([np.bincount(st[m], cong[k][m], minlength=n.size)[keep] / n[keep] for k in ks])
    rc, fc = r - r.mean(0), f - f.mean(0); w = n[keep]; out = []
    for yy in (0, 1):
        s = y_of_st[keep] == yy; out.append((w[s] * rc[:, s] * fc[:, s]).sum() / (w[s] * fc[:, s] ** 2).sum())
    return float(np.mean(out))


def exact_z(rng, y, cell, q):
    """z with P(z = 1 | y = 1) = q, P(z = 1 | y = 0) = 1 - q, exact (rounded) within every cell, as assign_z."""
    order = np.lexsort((rng.random(y.size), cell)); c = cell[order]
    start = np.r_[0, np.flatnonzero(np.diff(c)) + 1]; size = np.diff(np.r_[start, c.size])
    rank = np.arange(c.size) - np.repeat(start, size); frac = np.where(y[order] == 1, q, 1 - q)
    z = np.zeros(y.size, int); z[order] = rank < np.round(frac * np.repeat(size, size)); return z


def main():
    units, conf, mets = unit_sets()["amendment1_saccade"]
    labels = test_labels(sorted({int(k) for r in units for k in r["truth_counts"]["matched"]["all"]}))
    _, groups, U = build(units, conf, mets, labels)
    gid = np.empty(len(units), int)
    for j, m in enumerate(groups): gid[m] = j
    skill0 = np.array([u["all"]["skill"] for u in U]); slope0 = np.array([u["all"]["slope"] for u in U])
    mu_skill = np.array([skill0[m].mean() for m in groups])[gid]; mu_slope = np.array([slope0[m].mean() for m in groups])[gid]
    W = []
    for r in units:
        subs = [int(k) for k in r["truth_counts"]["matched"]["all"]]
        y = np.concatenate([labels[s] for s in subs]); si = np.concatenate([np.full(labels[s].size, i) for i, s in enumerate(subs)])
        W.append({"y": y, "si": si, "st": si * 2 + y, "half": np.asarray(r["test_half"]), "p": float(r["cell"]["p"]),
                  "z": {k: np.asarray(r["z_test"][k]) for k in ("matched", "neutral", "reversed")}, "n_sub": len(subs)})
    res = lambda v: v - np.array([v[m].mean() for m in groups])[gid]
    rng = np.random.default_rng(SEED); cov = [[SD_SKILL ** 2, RHO * SD_SKILL * SD_SLOPE], [RHO * SD_SKILL * SD_SLOPE, SD_SLOPE ** 2]]
    out = {k: [] for k in ("split_half_r", "r_with_true", "xfit_BA_matched", "sd_true_points")}
    for _ in range(N_REP):
        dev = rng.multivariate_normal([0, 0], cov, len(units)); n_sub = max(w["n_sub"] for w in W)
        sub_skill = rng.normal(0, SD_SUBJ_SKILL, n_sub); sub_slope = rng.normal(0, SD_SUBJ_SLOPE, n_sub)
        u_common = rng.random(max(w["y"].size for w in W))
        E = {k: np.empty((len(units), 2)) for k in KEYS}; BAM = np.empty((len(units), 2)); BAMs = np.empty((len(units), 2)); true = np.empty(len(units))
        for i, w in enumerate(W):
            y, si, st, half, p = w["y"], w["si"], w["st"], w["half"], w["p"]; y_of_st = np.arange(w["n_sub"] * 2) % 2
            skill = np.clip(mu_skill[i] + dev[i, 0] + sub_skill[si], 0.03, 0.97); slope = mu_slope[i] + dev[i, 1] + sub_slope[si]
            a_plus = np.clip(skill + slope / 2, 0.01, 0.99); a_minus = np.clip(skill - slope / 2, 0.01, 0.99)
            true[i] = (p - 0.5) * 0.5 * sum((a_plus - a_minus)[y == yy].mean() for yy in (0, 1)); u = np.where(rng.random(y.size) < SHARED, u_common[:y.size], rng.random(y.size))
            def sets(zs):
                cong = {k: (z == y).astype(float) for k, z in zs.items()}
                return {k: (u < np.where(cong[k] > 0, a_plus, a_minus)).astype(float) for k in zs}, cong
            cp, gp = sets(w["z"])                                                           # parity halves, stored z
            cs, gs = sets({k: exact_z(rng, y, st * 2 + half, q) for k, q in (("matched", p), ("neutral", 0.5))})  # stratified
            for h in (0, 1):
                m = half == h; M, N, R = (ba(cp[k], y, m) for k in ("matched", "neutral", "reversed"))
                E["neu"][i, h] = M - N; E["rev"][i, h] = M - R; E["nr"][i, h] = N - R; BAM[i, h] = M
                E["neu_fe"][i, h] = (p - 0.5) * fe(cp, gp, st, y_of_st, m, ("matched", "neutral"))
                E["all_fe"][i, h] = (p - 0.5) * fe(cp, gp, st, y_of_st, m, ("matched", "neutral", "reversed"))
                BAMs[i, h] = ba(cs["matched"], y, m); E["neu_stratified"][i, h] = BAMs[i, h] - ba(cs["neutral"], y, m)
        t = res(true); out["sd_true_points"].append(100 * np.sqrt((t ** 2).sum() / (len(units) - len(groups))))
        row = {"split_half_r": {}, "r_with_true": {}, "xfit_BA_matched": {}}
        for k in KEYS:
            e0, e1 = res(E[k][:, 0]), res(E[k][:, 1]); B = BAMs if k == "neu_stratified" else BAM
            row["split_half_r"][k] = corr(e0, e1); row["r_with_true"][k] = corr((e0 + e1) / 2, t)
            row["xfit_BA_matched"][k] = np.mean([corr(e0, res(B[:, 1])), corr(e1, res(B[:, 0]))])
        for s in row: out[s].append(row[s])
    summ = lambda v: {"mean": float(np.mean(v)), "p2.5": float(np.quantile(v, .025)), "p97.5": float(np.quantile(v, .975))}
    final = {"n_rep": N_REP, "seed": SEED, "n_units": len(units), "n_groups": len(groups),
             "parameters": {"sd_skill": SD_SKILL, "sd_slope": SD_SLOPE, "rho": RHO, "sd_subject_skill": SD_SUBJ_SKILL, "sd_subject_slope": SD_SUBJ_SLOPE, "shared": SHARED},
             "sd_true_truth_points": summ(out["sd_true_points"])}
    for s in ("split_half_r", "r_with_true", "xfit_BA_matched"):
        final[s] = {k: summ([r[k] for r in out[s]]) for k in KEYS}
    json.dump(final, open("results/analysis/posthoc_revision8_simcheck.json", "w"), indent=1)
    print(f"{N_REP} replicates; within-cell SD of the true ground truth {final['sd_true_truth_points']['mean']:.2f} points")
    for s in ("split_half_r", "r_with_true", "xfit_BA_matched"):
        print(s + ": " + "; ".join(f"{k} {v['mean']:.2f} [{v['p2.5']:.2f}, {v['p97.5']:.2f}]" for k, v in final[s].items()))


if __name__ == "__main__":
    if len(sys.argv) > 1: N_REP = int(sys.argv[1])
    if len(sys.argv) > 2: SHARED = float(sys.argv[2])
    main()
