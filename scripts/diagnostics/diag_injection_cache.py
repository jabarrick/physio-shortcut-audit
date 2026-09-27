"""Injection check on the CACHE actually consumed by P4 (numpy only; runs without mne/torch).

Written 2026-09-20, BEFORE running.  Pilot subjects only (split.json 'pilot').
Question: is ΔBA_neu ≈ 0 in P4 because the alpha confound is (a) not injected / mis-scaled,
or (b) injected as specified but too weak to be learnt at the reference amplitude k = 1σ?

Per subject, on the core segment (pad:pad+core_len), 8–13 Hz via FFT mask:
  comp  : log band power of the component series wᵀX, X0 vs X0 + c·B
          alpha: c = exp(σ/2) − 1, target change = σ            (2·ln g)
          mu   : c = ∓s/2 at s = 0.5, target contrast = 2[ln(1+s/2) − ln(1−s/2)] = 1.022
  sensor: the same change at the channel with the largest |a| (peak channel), and the
          within-subject between-window SD of that channel's log band power (effect size d).

Pre-declared readings:
  R1  comp change within ±15 % of target for ≥ 8/10     -> injection is as specified; ΔBA_neu≈0
      is 'confound too weak for the models' -> P5 is the legitimate next step.
  R2  comp change ≈ 0 or off by a large factor           -> generator/cache bug; P4 results void.
  Sensor d is descriptive only (not used to choose any value).
"""
import json
import numpy as np
from pathlib import Path

CH = None
root = Path("cache")
split = json.load(open("results/split.json"))
pilot = split["pilot"]
S = 0.5
BAND = (8.0, 13.0)


def bandpow(x, sf):  # x (..., T)
    f = np.fft.rfftfreq(x.shape[-1], 1 / sf)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return (np.abs(np.fft.rfft(x, axis=-1))[..., m] ** 2).mean(-1)


out = []
for s in pilot:
    d = root / f"sub-{s:03d}"
    if not (d / "meta.json").exists():
        out.append({"subject": s, "cache": False}); continue
    meta = json.load(open(d / "meta.json"))
    pad, L, sf = meta["pad"], meta["core_len"], meta["sfreq"]
    core = slice(pad, pad + L)
    X0 = np.load(d / "X0.npy", mmap_mode="r")[:, :, core].astype(np.float64)
    row = {"subject": s, "n_windows": int(X0.shape[0]), "sigma_alpha": meta["sigma_alpha"]}
    for kind, name in (("alpha", "alpha"), ("mu", "mu")):
        c = meta["components"].get(kind)
        if c is None or not c["passed"] or not (d / f"{name}.npy").exists():
            row[kind] = None; continue
        w = np.array(json.loads(c["w"]) if isinstance(c["w"], str) else c["w"])
        a = np.array(json.loads(c["a"]) if isinstance(c["a"], str) else c["a"])
        B = np.load(d / f"{name}.npy", mmap_mode="r")[:, :, core].astype(np.float64)
        pk = int(np.argmax(np.abs(a)))
        if kind == "alpha":
            cc = np.exp(meta["sigma_alpha"] / 2) - 1
            hi, lo = X0 + cc * B, X0
            target = meta["sigma_alpha"]
        else:
            hi, lo = X0 + (S / 2) * B, X0 - (S / 2) * B      # y=0 arm, y=1 arm
            target = 2 * (np.log(1 + S / 2) - np.log(1 - S / 2))
        comp = np.log(bandpow(np.einsum("c,nct->nt", w, hi), sf)) - np.log(bandpow(np.einsum("c,nct->nt", w, lo), sf))
        sens = np.log(bandpow(hi[:, pk], sf)) - np.log(bandpow(lo[:, pk], sf))
        sd_pk = np.log(bandpow(X0[:, pk], sf)).std(ddof=1)
        # whole-scalp mean band power (what a model without spatial filtering sees)
        scalp = np.log(bandpow(hi, sf).mean(1)) - np.log(bandpow(lo, sf).mean(1))
        row[kind] = {"target": float(target), "comp_change_median": float(np.median(comp)),
                     "comp_ratio": float(np.median(comp) / target),
                     "peak_channel_idx": pk, "sensor_change_median": float(np.median(sens)),
                     "sensor_sd_between_windows": float(sd_pk),
                     "sensor_d": float(np.median(sens) / sd_pk),
                     "scalp_change_median": float(np.median(scalp)),
                     "B_to_X0_rms": float(np.sqrt((B ** 2).mean() / (X0 ** 2).mean())),
                     "wTa": float(w @ a)}
    out.append(row)

Path("results/pilots/injection_cache_diag.json").write_text(json.dumps(out, indent=1))
for r in out:
    print(r["subject"], r.get("n_windows"), end="  ")
    for k in ("alpha", "mu"):
        v = r.get(k)
        if v:
            print(f"{k}: tgt {v['target']:.3f} comp {v['comp_change_median']:.3f} (x{v['comp_ratio']:.2f}) "
                  f"sens {v['sensor_change_median']:.3f} d {v['sensor_d']:.2f} scalp {v['scalp_change_median']:.3f} "
                  f"B/X0 {v['B_to_X0_rms']:.3f} wTa {v['wTa']:.2f}", end=" | ")
        else:
            print(f"{k}: -", end=" | ")
    print()
