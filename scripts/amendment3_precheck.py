"""Amendment 3 (in preparation; not yet registered): pre-check without training.  How much of an injected saccade does
the HEOG-regression metric remove, for the template generator of the registration and for the real-event libraries
R-H (horizontal) and R-V (with vertical and blink components) of scripts/amendment3_event_library.py?

Events are unit saccades (AF7 - AF8 step of -1 uV) on all 64 channels at 160 Hz, epoch [-0.5, 1.5] s:
  template  frozen p3audit.generator.inject.saccade_delta (step + spike potential, topography jitter) with the
            main-pool templates (results/saccade_pool.npz), 500 events;
  R-H, R-V  cache/a3_event_library_{RH,RV}.npz.
Each epoch is average-referenced (the model stream is) and mean-centred per channel.  The HEOG-regression metric
(p3audit/metrics/intervention.py, method saccade_locked) is mimicked: b = sum(dX dR) / sum(dR^2) from the steps of
the events of one half (dX, dR: mean over 100 ms after offset minus 100 ms before onset), then X - b (r - mean r) with
r = AF7 - AF8 on the other half.  Reported over 200 random event splits (mean, 2.5 and 97.5 percentiles):
  removed_energy      1 - sum(residual^2) / sum(X^2) over [-0.1, 1.0] s and all channels, pooled over events;
  removed_mean_event  the same for the average event (the part a decoder could learn as a fixed pattern);
  background_share    for real events: mean energy over [-0.5, -0.05] s (background only) relative to the mean energy
                      over [0, 1.0] s (saccade + background), average-referenced, before centring; median over events.
  sham_vs_event_power per real library: mean power of the sham epochs scaled like a random event of the library,
                      relative to the mean power of the events over [-0.5, 1.5] s.
This is descriptive: no unit is trained and no ground truth is involved.  Writes results/a3/precheck.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
N_SPLIT, N_TEMPLATE, SEED = 200, 500, 20261013


def template_events(cfg, ch, sf, n, onset, offset_rel):
    from p3audit.generator.inject import saccade_delta
    pool = np.load("results/saccade_pool.npz"); keys = sorted(pool.files); rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n):
        t = pool[keys[rng.integers(len(keys))]]
        out.append(saccade_delta(t, ch, sf, n_region=int(2.0 * sf), pad=0, envelope=np.ones(int(2.0 * sf)), onset=onset, cfg=cfg,
                                 rng=rng, rightward=True, reference=cfg["physionet"]["original_reference"]))
    return np.stack(out).astype(np.float32), np.full(n, offset_rel)


def stats(E, off_rel, onset, sf, i7, i8):
    E = E - E.mean(1, keepdims=True)                     # average reference
    pre_e = (E[:, :, onset - int(0.5 * sf):onset - int(0.05 * sf)] ** 2).mean((1, 2)); post_e = (E[:, :, onset:onset + int(1.0 * sf)] ** 2).mean((1, 2))
    E = E - E.mean(2, keepdims=True)
    w = int(round(0.1 * sf)); r = E[:, i7] - E[:, i8]
    pre = slice(onset - w, onset)
    dX = np.stack([E[k, :, onset + o:onset + o + w].mean(-1) - E[k, :, pre].mean(-1) for k, o in enumerate(off_rel)])
    dR = np.array([r[k, onset + o:onset + o + w].mean() - r[k, pre].mean() for k, o in enumerate(off_rel)])
    win = slice(onset - int(0.1 * sf), onset + int(1.0 * sf)); rng = np.random.default_rng(SEED); res = {"removed_energy": [], "removed_mean_event": []}
    for _ in range(N_SPLIT):
        m = rng.random(len(E)) < 0.5
        b = dX[m].T @ dR[m] / (dR[m] @ dR[m])
        X = E[~m][:, :, win]; rr = r[~m][:, win]; rr = rr - rr.mean(-1, keepdims=True)
        Rz = X - b[None, :, None] * rr[:, None, :]
        res["removed_energy"].append(1 - (Rz ** 2).sum() / (X ** 2).sum())
        mX, mR = X.mean(0), Rz.mean(0); res["removed_mean_event"].append(1 - (mR ** 2).sum() / (mX ** 2).sum())
    q = lambda v: {"mean": float(np.mean(v)), "p2.5": float(np.quantile(v, .025)), "p97.5": float(np.quantile(v, .975))}
    return {"n_events": len(E), **{k: q(v) for k, v in res.items()}, "background_share_median": float(np.median(pre_e / post_e))}


def main():
    from p3audit.config import load_config
    from p3audit.utils.common import ch_index
    cfg = load_config("configs/default.yaml"); out = {}
    L = {k: np.load(f"cache/a3_event_library_{k}.npz") for k in ("RH", "RV")}
    ch = [str(c) for c in L["RH"]["ch_names"]]; sf = float(L["RH"]["sfreq"]); onset = int(L["RH"]["onset_index"]); i7, i8 = ch_index(ch, ["AF7", "AF8"])
    ramp = int(round(cfg["saccade"]["ramp_ms"] / 1000 * sf))
    E, off = template_events(cfg, ch, sf, N_TEMPLATE, onset, ramp)
    out["template"] = stats(E, off, onset, sf, i7, i8)
    sh = np.load("cache/a3_event_library_sham.npz")["epochs"]; sh = sh - sh.mean(1, keepdims=True); rng = np.random.default_rng(SEED)
    for k in ("RH", "RV"):
        out[k] = stats(L[k]["epochs"], L[k]["offset_rel"], onset, sf, i7, i8)
        ev = L[k]["epochs"] - L[k]["epochs"].mean(1, keepdims=True); scale = 1 / np.abs(L[k]["dheog"])
        out[k]["sham_vs_event_power"] = float((sh ** 2).mean((1, 2)).mean() * (scale[rng.integers(len(scale), size=len(sh))] ** 2).mean() / (ev ** 2).mean((1, 2)).mean())
    Path("results/a3").mkdir(parents=True, exist_ok=True); json.dump(out, open("results/a3/precheck.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:9s} n={v['n_events']:4d}  removed energy {v['removed_energy']['mean']:.2f} [{v['removed_energy']['p2.5']:.2f}, {v['removed_energy']['p97.5']:.2f}]"
              f"  removed mean-event energy {v['removed_mean_event']['mean']:.2f} [{v['removed_mean_event']['p2.5']:.2f}, {v['removed_mean_event']['p97.5']:.2f}]"
              f"  background share {v['background_share_median']:.2f}" + (f"  sham/event power {v['sham_vs_event_power']:.2f}" if 'sham_vs_event_power' in v else ""))


if __name__ == "__main__":
    main()
