"""Post hoc (exploratory; fourth review, point 18): a linear, deterministically trained decoder (CSP + LDA) on the
semi-synthetic data, as a check of the generator and of the ground-truth definition.  Its training has no random
initialisation or stochastic optimisation, so differences between seeds of one cell come only from the composition
of the data (which windows receive the confound, injection sampling).  Comparing its within-cell variation of ground
truth with that of the deep networks shows how much of the latter is training randomness.
Design: p = 0.9 factorial cells (3 task strengths x 3 confound amplitudes), both confounds, seeds 50-55: 108 units.
Frozen generator, configuration and metric code (p3audit) are used unchanged; no torch needed.
Model: mne.decoding.CSP (6 components, Ledoit-Wolf, log-variance) on the broadband model input (0.5-75 Hz), then LDA;
fitted on at most 4000 training windows (stratified by subject).  Recorded as in a unit record: truth counts for the
matched/neutral/reversed test sets with the window-parity halves; probe D at the cell amplitude on the CSP features
(frozen probe code); residual erasure on the CSP features with the LDA head (frozen LEACE code); HEOG regression
(saccade_locked, contamination 0) and the SRI band-stop interventions.  No IG (not defined for CSP features).
Output: results/units_csp/<uid>.json.  Resumable.  CPU only.
Usage (repo root): python scripts/posthoc_revision11_csp_units.py [--confound alpha|saccade|both] [--limit N] [--out units_csp]
"""
import argparse, json, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SEEDS = [50, 51, 52, 53, 54, 55]
N_COMP, MAX_TRAIN = 6, 4000


def counts3(y, pred, subj, half):
    from p3audit.stats.suffstats import confusion_by_subject
    return {"all": confusion_by_subject(y, pred, subj),
            "half0": confusion_by_subject(y[half == 0], pred[half == 0], subj[half == 0]),
            "half1": confusion_by_subject(y[half == 1], pred[half == 1], subj[half == 1])}


def strat_idx(subj, n_max, seed):
    if len(subj) <= n_max:
        return np.arange(len(subj))
    rng = np.random.default_rng(seed); out = []
    for s in np.unique(subj):
        ii = np.where(subj == s)[0]; k = max(1, int(round(n_max * len(ii) / len(subj))))
        out.append(rng.choice(ii, min(k, len(ii)), replace=False))
    return np.sort(np.concatenate(out))


def run(ctx, cfg, unit, out_dir):
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from p3audit.data.windows import half_assignment
    from p3audit.generator.design import test_couplings
    from p3audit.generator.semisynth import compose
    from p3audit.metrics.truth import unit_truth
    from p3audit.metrics.erasure import fit_leace, residual_concept
    from p3audit.metrics.intervention import apply_heog_regression, bandstop_channels, crop, fit_heog_regression
    from p3audit.metrics.probe import D_from_ba, fit_probe
    from p3audit.stats.suffstats import ba_from_counts, confusion_by_subject
    from p3audit.utils.common import ch_index
    out = ctx.results / out_dir / f"{unit.uid}.json"
    if out.exists():
        return
    t0 = time.time(); cell, seed = unit.cell, unit.seed; sp = ctx.require_split()
    tr_s, te_s = ctx.usable(sp.train), ctx.usable(sp.test); caches = ctx.caches(tr_s + te_s); ch = ctx.ch_names
    tr = compose(caches, tr_s, cell, cfg, cell.p, key=(cell.cid, "train", seed))
    ii = strat_idx(tr.subj, MAX_TRAIN, seed)
    Xtr = tr.materialize(indices=ii).astype(np.float64); ytr = tr.y[ii]
    csp = CSP(n_components=N_COMP, reg="ledoit_wolf", log=True); Ftr = csp.fit_transform(Xtr, ytr)
    lda = LDA().fit(Ftr, ytr); del Xtr
    rec = {"uid": unit.uid, "model": "csp_lda", "cell": cell.as_dict(), "cid": cell.cid, "seed": seed, "config_hash": cfg.hash(),
           "n_train_windows": int(len(ii))}
    tests = {k: compose(caches, te_s, cell, cfg, q, key=(cell.cid, "test", k, seed), role="test") for k, q in test_couplings(cell.p).items()}
    base = tests["matched"]; half = half_assignment(base.wids, cfg["seeds"]["half_split"])
    preds, counts = {}, {}
    for k, cs in tests.items():
        preds[k] = lda.predict(csp.transform(cs.materialize().astype(np.float64))); counts[k] = counts3(cs.y, preds[k], cs.subj, half)
    rec["truth"] = unit_truth(base.y, preds, base.subj, half); rec["truth_counts"] = counts
    # probe on CSP features at the cell amplitude (frozen probe code)
    pc = cfg["probe"]; amp = float(cell.amp)
    ptr = compose(caches, tr_s, cell, cfg, 0.5, key=(cell.cid, "probe_train", amp, seed), amp=amp)
    pte = compose(caches, te_s, cell, cfg, 0.5, key=(cell.cid, "probe_test", amp, seed), amp=amp, role="test")
    pi = strat_idx(ptr.subj, pc["max_train_windows"], seed + 1)
    Ep = csp.transform(ptr.materialize(indices=pi).astype(np.float64)); Et = csp.transform(pte.materialize().astype(np.float64))
    pr = fit_probe(Ep, ptr.z[pi], ptr.subj[pi], pc["Cs"], pc["inner_folds"])
    rec["probe"] = {f"{amp:g}": {"D": D_from_ba(ba_from_counts(confusion_by_subject(pte.z, pr.predict(Et), pte.subj)))}}
    # residual erasure on CSP features with the LDA head
    Fte = csp.transform(base.materialize().astype(np.float64))
    r_tr, _ = residual_concept(tr.z[ii], ytr); er = fit_leace(Ftr, r_tr, cfg["erasure"]["reg"])
    rec["erasure"] = {"counts_head": counts3(base.y, lda.predict(Fte), base.subj, half),
                      "counts_erased": counts3(base.y, lda.predict(er(Fte)), base.subj, half)}
    # interventions
    pad, L, sf = base.pad, base.core_len, cfg["streams"]["model"]["sfreq"]; ic = cfg["intervention"]
    post = ch_index(ch, cfg["alpha_component"]["posterior"])
    hi = strat_idx(tr.subj, 600, seed + 2)
    reg = fit_heog_regression(tr.materialize(padded=True, indices=hi), ch, 0.0, ic.get("heog_method", "saccade_locked"), sf, cfg)
    pa = np.empty(len(base), int); pcn = np.empty(len(base), int); ph = np.empty(len(base), int)
    for s in np.unique(base.subj):
        jj = np.where(base.subj == s)[0]; Xp = base.materialize(padded=True, indices=jj)
        f = lambda X: lda.predict(csp.transform(crop(X, pad, L).astype(np.float64)))
        pa[jj] = f(bandstop_channels(Xp, sf, post, ic["alpha_band"], ic["transition_hz"]))
        pcn[jj] = f(bandstop_channels(Xp, sf, post, ic["control_band"], ic["transition_hz"]))
        ph[jj] = f(apply_heog_regression(Xp, ch, reg))
    rec["sri"] = {"counts": {"post_alpha": counts3(base.y, pa, base.subj, half), "post_ctrl": counts3(base.y, pcn, base.subj, half)}}
    rec["heog"] = {"0": {"counts": counts3(base.y, ph, base.subj, half), "method": reg["method"]}}
    rec["seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(rec, open(out, "w", encoding="utf-8"))
    print("done", unit.uid, "dBA_neu %.3f" % rec["truth"].get("delta_ba_neu", float("nan")), "%.0fs" % rec["seconds"], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--confound", default="both", choices=["alpha", "saccade", "both"])
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--out", default="units_csp"); ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    from p3audit.config import load_config
    from p3audit.experiments.context import Context
    from p3audit.generator.design import Cell, Unit
    cfg = load_config(a.config); print("config hash", cfg.hash(), flush=True); ctx = Context.create(cfg); d = cfg["design"]
    confs = ["saccade", "alpha"] if a.confound == "both" else [a.confound]
    units = [Unit("csp_lda", Cell(c, d["factorial_p"], s, float(am), "factorial"), sd)
             for c in confs for s in d["factorial_s"] for am in d["factorial_amp"] for sd in SEEDS]
    if a.limit:
        units = units[:a.limit]
    print(len(units), "units", flush=True)
    for u in units:
        try:
            run(ctx, cfg, u, a.out)
        except Exception as e:
            (ctx.results / (a.out + "_failed")).mkdir(parents=True, exist_ok=True)
            (ctx.results / (a.out + "_failed") / f"{u.uid}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print("FAILED", u.uid, e, flush=True)


if __name__ == "__main__":
    main()
