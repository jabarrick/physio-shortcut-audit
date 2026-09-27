"""Diagnostic: what ARE the components that pass |r| but fail the P6 sign rule?

Context.  Removing the spatial mean from the pattern before the opposite-sign test
(2026-09-19, to make the test invariant to the unknown recording reference) changed nothing:
8/15 still fail, bit-identically.  So the reference offset is not what breaks the sign rule.

The failure pattern points elsewhere.  Subjects 9 and 61 each fail a half in which |r| CLEARS
the permutation threshold (0.562 vs 0.450; 0.485 vs 0.452) — a component that tracks target
side strongly, but without opposite F7/AF7 vs F8/AF8 polarity.  In this task target side IS
the motor label, so every lateralised sensorimotor component correlates with side by
construction.  The sign rule is the only thing separating an ocular component from a motor one.

That matters for how P6 gets fixed.  PILOT_LOG 2.3 lists three ways to loosen the criterion
(n_pca, whole-trial selection, the sign rule as a diagnostic switch).  If the components
crowding the top of the |r| ranking are MOTOR, loosening any of them lets a motor component
into the saccade template pool — the injected confound would then be a copy of the task
signal, making confound and task degenerate by construction.  That is the exact failure mode
this paper exists to study; it must not appear inside its own generator.

WHAT THIS DECIDES, STATED BEFORE IT IS RUN.  Per half, components are ranked by |r| and the
top `--top` reported by where the pattern lives:

    ocular_share = energy over AF7, AF8, F7, F8, Fp1, Fp2, FT7, FT8
    motor_share  = energy over C5..C6, FC5..FC6, CP5..CP6 (the mu neighbourhoods)
    bipolarity   = |p[AF7] - p[AF8]| / max|p|   (large for a horizontal eye dipole)

The reading is fixed in advance:

  - sign-rule failures dominated by motor_share > ocular_share -> the criterion is working,
    P6 must NOT be loosened; the fix is an explicit ocular-topography requirement and the
    pool shrinks honestly.
  - failures ocular by topography but lacking clean opposite polarity -> the sign rule is too
    brittle for genuine eye components and can be replaced by a bipolarity threshold.
  - nothing near the top ocular at all -> these subjects have no usable saccade template and
    6.4 (2) applies on its merits.

Writes only results/pilots/P6_components_diag.json.  Touches no pool, template or config.

Usage:  python diag_p6_components.py [--top 5] [--subjects 9,61,39,53]
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.exceptions import ConvergenceWarning

from p3audit.config import load_config
from p3audit.experiments.context import Context
from p3audit.generator.saccade_ica import _activation_feature, _corr_cols, alternate_halves, lateral_epochs
from p3audit.utils.common import ch_index, write_json

OCULAR = ["AF7", "AF8", "F7", "F8", "Fp1", "Fp2", "FT7", "FT8"]
MOTOR = ["C5", "C3", "C1", "C2", "C4", "C6", "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
         "CP5", "CP3", "CP1", "CP2", "CP4", "CP6"]


def share(p, idx):
    p = np.asarray(p, dtype=float)
    return float((p[idx] ** 2).sum() / (p**2).sum())


def decompose(Xi, si, sf, cfg, seed):
    """Mirrors fit_half's decomposition, including its convergence retry."""
    sc = cfg["saccade"]
    n_tr, n_ch, n_t = Xi.shape
    cat = Xi.transpose(1, 0, 2).reshape(n_ch, -1)
    cat = cat - cat.mean(1, keepdims=True)
    n_comp = min(sc["n_pca"], n_ch - 1)
    pca = PCA(n_components=n_comp, random_state=seed).fit(cat.T)
    Z = pca.transform(cat.T)
    for attempt, (max_iter, tol) in enumerate(((3000, 1e-3), (10000, 5e-3))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            ica = FastICA(n_components=n_comp, random_state=seed + 101 * attempt,
                          max_iter=max_iter, tol=tol, whiten="unit-variance").fit(Z)
        if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
            break
    S = ica.transform(Z).T
    patterns = pca.components_.T @ ica.mixing_
    S_ep = S.reshape(n_comp, n_tr, n_t).transpose(1, 0, 2)
    F = _activation_feature(S_ep, sf, cfg)
    r = _corr_cols(F, si.astype(float))
    rng = np.random.default_rng(seed)
    null = np.array([np.abs(_corr_cols(F, rng.permutation(si).astype(float))).max()
                     for _ in range(sc["n_permutations"])])
    return patterns, r, float(np.quantile(null, sc["null_quantile"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()

    cfg = load_config(None, {})
    ctx = Context.create(cfg)
    sc = cfg["saccade"]
    pilots = list(ctx.require_split().pilot)
    if args.subjects:
        want = {int(x) for x in args.subjects.split(",")}
        pilots = [s for s in pilots if s in want]

    rows = []
    for s in pilots:
        ctx.guard.record(s, sc["lateral_runs"], "pilot_saccade_ica")
        runs = [ctx.source.load(s, r) for r in sc["lateral_runs"]]
        X, side, is_exec, sf = lateral_epochs(runs, cfg)
        ch = runs[0].ch_names
        oc, mo = ch_index(ch, OCULAR), ch_index(ch, MOTOR)
        li, ri = ch_index(ch, sc["left_sites"]), ch_index(ch, sc["right_sites"])
        i7, i8 = ch_index(ch, ["AF7", "AF8"])
        half = alternate_halves(is_exec)

        for h in (0, 1):
            m = half == h
            patterns, r, thr = decompose(X[m], side[m], sf, cfg, seed=h)
            cur = []
            for rank, k in enumerate(np.argsort(-np.abs(r))[: args.top]):
                p = patterns[:, k] * np.sign(r[k])
                pc = p - p.mean()
                cur.append({
                    "subject": s, "half": h, "rank": rank, "k": int(k),
                    "r": float(abs(r[k])), "threshold": thr, "r_clears": bool(abs(r[k]) > thr),
                    "peak_channel": ch[int(np.argmax(np.abs(p)))],
                    "ocular_share": share(p, oc), "motor_share": share(p, mo),
                    "bipolarity": float(abs(p[i7] - p[i8]) / np.abs(p).max()),
                    "opposite_raw": bool(np.sign(p[li].mean()) != np.sign(p[ri].mean())),
                    "opposite_centred": bool(np.sign(pc[li].mean()) != np.sign(pc[ri].mean())),
                    "spatial_mean_rel": float(abs(p.mean()) / np.abs(p).max()),
                })
            rows += cur
            print(f"sub {s:3d} half {h}: thr {thr:.3f} | " + "  ".join(
                f"#{d['rank']} r={d['r']:.2f}{'*' if d['r_clears'] else ' '} {d['peak_channel']:>4s} "
                f"oc={d['ocular_share']:.2f} mo={d['motor_share']:.2f} bip={d['bipolarity']:.2f} "
                f"sgn={'Y' if d['opposite_centred'] else 'n'}" for d in cur))

    write_json(ctx.results / "pilots" / "P6_components_diag.json",
               {"top": args.top, "ocular_sites": OCULAR, "motor_sites": MOTOR, "rows": rows})
    print("\nwrote results/pilots/P6_components_diag.json")

    fails = [d for d in rows if d["rank"] == 0 and d["r_clears"] and not d["opposite_centred"]]
    print(f"\ntop-ranked components clearing |r| but failing the sign rule: {len(fails)}")
    if fails:
        mo_dom = sum(1 for d in fails if d["motor_share"] > d["ocular_share"])
        print(f"  motor-dominant (motor_share > ocular_share): {mo_dom}/{len(fails)}")
        for d in fails:
            print(f"    sub {d['subject']:3d} half {d['half']}  peak {d['peak_channel']:>4s}  "
                  f"ocular {d['ocular_share']:.2f}  motor {d['motor_share']:.2f}  "
                  f"bipolarity {d['bipolarity']:.2f}  spatial_mean {d['spatial_mean_rel']:.3f}")
    ocu = [d for d in rows if d["rank"] == 0]
    print(f"\nreference-offset check: max |spatial mean| / max|p| over top components = "
          f"{max(d['spatial_mean_rel'] for d in ocu):.3f} "
          f"(median {np.median([d['spatial_mean_rel'] for d in ocu]):.3f}) — "
          f"small values confirm the reference contributes little to the topographies")


if __name__ == "__main__":
    main()
