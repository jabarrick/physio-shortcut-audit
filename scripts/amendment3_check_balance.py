"""Amendment 3 (in preparation): checks of the composition, no training.  Cell: saccade, p = 0.9, s_star, 1 a*;
seed 30; matched coupling.
  --calibrate  on the TRAINING subjects: injected power over [-0.35, -0.05] s before the event onset (background only)
               in z = 1 windows (event epoch) and z = 0 windows (sham epoch, gain 1); gain = sqrt(ratio), written to
               results/a3/sham_gain_<ARM>.json.  Uses composition only; no model, label or outcome.
  default      on the TEST subjects with the stored gain: (1) z is exact within subject x label x half in the three
               ground-truth sets; (2) the same pre-onset power ratio (should be near 1); (3) injected power by channel
               group and z.  Writes results/a3/check_balance_<ARM>.json.
Usage: python scripts/amendment3_check_balance.py --arm RH [--calibrate]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))


def pre_onset_power(cs, caches, arm):
    onset = {r["wid"]: r["sacc_onset"] for s in np.unique(cs.subj) for r in caches[int(s)].meta["rows"]}
    P = []
    for i in range(len(cs)):
        o = onset[cs.wids[i]]
        if o < int(0.5 * 160):
            continue
        b = f"sacc_{arm}" if cs.z[i] == 1 else f"sham_{arm}"
        x = cs.coef[b][i] * np.asarray(cs._arr(int(cs.subj[i]), b)[int(cs.idx[i])], dtype=np.float64)
        a0 = cs.pad + int(round((o - 0.35 * 160) * 1.25)); a1 = cs.pad + int(round((o - 0.05 * 160) * 1.25))
        P.append((cs.z[i], (x[:, a0:a1] ** 2).mean()))
    P = np.array(P); return float(P[P[:, 0] == 1, 1].mean()), float(P[P[:, 0] == 0, 1].mean()), len(P)


def main():
    from p3audit.config import load_config
    from p3audit.experiments.context import Context
    from p3audit.data.windows import half_assignment
    from p3audit.generator.design import Cell, test_couplings
    from p3audit.utils.common import ch_index
    import amendment3_common as A
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="RH"); ap.add_argument("--calibrate", action="store_true"); a = ap.parse_args()
    cfg = load_config("configs/default.yaml"); ctx = Context.create(cfg); sp = ctx.require_split(); seed = 30
    cell = Cell("saccade", 0.9, "s_star", 1.0, "factorial", sacc_basis=f"sacc_{a.arm}")
    if a.calibrate:
        compose = A.install(ctx, a.arm, patch_unit=False, gain=1.0); tr = ctx.usable(sp.train); caches = ctx.caches(tr)
        cs = compose(caches, tr, cell, cfg, 0.9, key=(cell.cid, "a3_calibration", seed))
        p1, p0, n = pre_onset_power(cs, caches, a.arm); g = float(np.sqrt(p1 / p0))
        out = {"arm": a.arm, "subjects": "training", "n_windows": n, "pre_onset_power_event": p1, "pre_onset_power_sham_gain1": p0, "gain": g}
        json.dump(out, open(f"results/a3/sham_gain_{a.arm}.json", "w"), indent=1); print(json.dumps(out, indent=1)); return
    compose = A.install(ctx, a.arm, patch_unit=False); te = ctx.usable(sp.test); caches = ctx.caches(te)
    out = {"arm": a.arm, "n_test_subjects": len(te), "sham_gain": A.sham_gain(a.arm)}
    for k, q in test_couplings(0.9).items():                      # couplings exactly as in the frozen code (reversed = 1 - p)
        cs = compose(caches, te, cell, cfg, q, key=(cell.cid, "test", k, seed), role="test"); half = half_assignment(cs.wids, cfg["seeds"]["half_split"])
        dev = [abs(cs.z[m].mean() - round((q if y == 1 else 1 - q) * m.sum()) / m.sum())
               for s in np.unique(cs.subj) for y in (0, 1) for h in (0, 1) for m in [(cs.subj == s) & (cs.y == y) & (half == h)] if m.sum()]
        out[f"z_exactness_{k}_max_dev"] = float(max(dev))
        if k == "matched":
            p1, p0, n = pre_onset_power(cs, caches, a.arm); out["pre_onset_power"] = {"z1_event": p1, "z0_sham": p0, "ratio": p1 / p0, "n_windows": n}
            ch = ctx.ch_names; front = ch_index(ch, ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8", "Fpz", "AF3", "AF4", "AFz", "F5", "F6"]); rest = np.setdiff1d(np.arange(len(ch)), front)
            P = []
            for i in range(len(cs)):
                sj, j = int(cs.subj[i]), int(cs.idx[i])
                x = sum(cs.coef[b][i] * np.asarray(cs._arr(sj, b)[j], dtype=np.float64) for b in (f"sacc_{a.arm}", f"sham_{a.arm}"))[:, cs.pad:cs.pad + cs.core_len]
                P.append([(x[front] ** 2).mean(), (x[rest] ** 2).mean()])
            P = np.array(P); z = cs.z
            out["injected_power_whole_window"] = {"frontal_z1": float(P[z == 1, 0].mean()), "frontal_z0": float(P[z == 0, 0].mean()),
                                                  "other_z1": float(P[z == 1, 1].mean()), "other_z0": float(P[z == 0, 1].mean())}
            Xb = np.stack([np.asarray(cs._arr(int(cs.subj[i]), "X0")[int(cs.idx[i])])[:, cs.pad:cs.pad + cs.core_len] for i in range(len(cs))])
            out["background_power_X0"] = {"frontal": float((Xb[:, front] ** 2).mean()), "other": float((Xb[:, rest] ** 2).mean())}
    json.dump(out, open(f"results/a3/check_balance_{a.arm}.json", "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
