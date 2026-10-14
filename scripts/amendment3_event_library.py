"""Amendment 3 (in preparation; not yet registered): libraries of real recorded rightward saccades for the
independent saccade generators.  Outside the frozen package; uses the frozen loader, detector and filters.

Source: the 15 pilot subjects (results/split.json), lateral runs (configs: saccade.lateral_runs).  Pilot subjects
appear thereafter only in training, never in validation or test.  No labels or cue times are used.
Per run: HEOG = AF7 - AF8 and VEOG = mean(Fp1, Fp2) after the 0.1 Hz high-pass of the template code
(p3audit/generator/saccade_regression.py); saccades from the project detector (p3audit/saccades/detect.py).
Steps (dHEOG, dVEOG): mean over 100 ms after offset + 50 ms minus mean over 100 ms before onset - 50 ms, as in
event_steps.  Rules, applied to every rightward saccade:
  common  1. epoch [-0.5, +1.5] s around onset inside the run;
          2. isolated: no other detected saccade with onset in that epoch;
          3. |dHEOG| >= 75 uV;
          4. no artefact: max |x - baseline| over all channels and the epoch <= 1000 uV;
  R-H (primary, horizontal)   additionally  |dVEOG| <= 0.5 |dHEOG|  and  max |VEOG - baseline| over [-0.1, 0.6] s <= 150 uV;
  R-V (secondary, with vertical and blink components)  no further rule.
The rules were fixed after counting candidate events only (no model trained, no outcome seen); see
results/a3/event_library_summary.json.
Kept epoch = all 64 channels at 160 Hz minus their mean over [-0.3, -0.05] s, divided by |dHEOG|, so that the
AF7 - AF8 step is -1 uV (rightward sign of the project), the unit of the existing saccade bases.
Sham library: in the same runs, epochs starting every 1 s with no detected saccade onset within [-0.5, +1.5] s,
max |VEOG - baseline| <= 150 uV over the epoch and the artefact rule; baseline-subtracted, kept in uV (not scaled).
They are injected into z = 0 windows so that both classes of z receive a real background segment from another
subject and only the saccade differs (otherwise the added background itself would be a cue for z).
Writes results/a3/event_library.csv (every rightward saccade with its rule outcomes; tracked),
results/a3/event_library_summary.json (counts; tracked) and cache/a3_event_library_{RH,RV,sham}.npz (epochs; not tracked).
Access is logged to the signal-access log under the purpose 'amendment3_event_library'.
Usage (repo root): python scripts/amendment3_event_library.py [--data-root ~/mne_data]
"""
import argparse, csv, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EPOCH, BASE, STEP_W, GUARD, TRANS = (-0.5, 1.5), (-0.3, -0.05), 0.100, 0.050, (-0.1, 0.6)
MIN_STEP, MAX_ABS, MAX_VRATIO, MAX_VTRANS = 75.0, 1000.0, 0.5, 150.0
COMMON = ("r1_inside", "r2_isolated", "r3_step", "r4_no_artefact")


def main():
    from p3audit.config import load_config
    from p3audit.data.sources import PhysioNetSource
    from p3audit.data.streams import fir
    from p3audit.saccades.detect import detect_all, heog
    from p3audit.utils.access import SignalAccessGuard
    from p3audit.utils.common import ch_index
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--data-root", default=None)
    a = ap.parse_args(); cfg = load_config(a.config)
    pilots = json.load(open("results/split.json", encoding="utf-8"))["pilot"]; runs = cfg["saccade"]["lateral_runs"]
    src = PhysioNetSource(Path(a.data_root).expanduser() if a.data_root else cfg.resolve_path("data_root"), cfg["physionet"]["n_subjects"], download=False)
    guard = SignalAccessGuard(cfg.resolve_path("access_log"), set(pilots), enforce=False)
    rows, lib, sham = [], {"RH": [], "RV": []}, []
    for s in pilots:
        guard.record(s, runs, "amendment3_event_library")
        for rn in runs:
            run = src.load(s, rn); sf = run.sfreq; ch = run.ch_names
            x = fir(run.data, sf, cfg["saccade"]["highpass"], None); h = heog(run, cfg)
            v = x[ch_index(ch, cfg["streams"]["eye"]["veog"])].mean(0)
            sacs = detect_all(h, sf, cfg); on_all = np.array([q.onset for q in sacs])
            i = lambda t: int(round(t * sf)); e0, e1, b0, b1, t0, t1 = map(i, EPOCH + BASE + TRANS)
            w, g = max(2, i(STEP_W)), i(GUARD)
            for c in range(-e0, x.shape[1] - e1, int(sf)):          # sham epochs, onset grid of 1 s
                if ((on_all >= c + e0) & (on_all < c + e1)).any():
                    continue
                seg = x[:, c + e0:c + e1] - x[:, c + b0:c + b1].mean(1, keepdims=True)
                vt = float(np.abs(v[c + e0:c + e1] - v[c + b0:c + b1].mean()).max())
                if vt <= MAX_VTRANS and float(np.abs(seg).max()) <= MAX_ABS:
                    sham.append((s, rn, c, seg.astype(np.float32)))
            for q in sacs:
                if q.direction != 1:
                    continue
                r = {"subject": s, "run": rn, "onset": q.onset, "offset": q.offset, "detector_amp_uv": round(q.amplitude, 2)}
                r["r1_inside"] = q.onset + e0 >= 0 and q.onset + e1 <= x.shape[1] and q.onset - g - w >= 0 and q.offset + g + w <= x.shape[1]
                if r["r1_inside"]:
                    step = lambda y: float(y[q.offset + g:q.offset + g + w].mean() - y[q.onset - g - w:q.onset - g].mean())
                    r["r2_isolated"] = not bool(((on_all >= q.onset + e0) & (on_all < q.onset + e1) & (on_all != q.onset)).any())
                    dh, dv = step(h), step(v); r["dheog_uv"], r["dveog_uv"] = round(dh, 2), round(dv, 2); r["r3_step"] = abs(dh) >= MIN_STEP
                    seg = x[:, q.onset + e0:q.onset + e1] - x[:, q.onset + b0:q.onset + b1].mean(1, keepdims=True)
                    r["abs_max_uv"] = round(float(np.abs(seg).max()), 2); r["r4_no_artefact"] = r["abs_max_uv"] <= MAX_ABS
                    vb = v[q.onset + b0:q.onset + b1].mean(); r["veog_transient_uv"] = round(float(np.abs(v[q.onset + t0:q.onset + t1] - vb).max()), 2)
                    r["rh_vertical_ok"] = abs(dv) <= MAX_VRATIO * abs(dh) and r["veog_transient_uv"] <= MAX_VTRANS
                    r["kept_RV"] = all(r[k] for k in COMMON); r["kept_RH"] = r["kept_RV"] and r["rh_vertical_ok"]
                    for k in ("RH", "RV"):
                        if r["kept_" + k]:
                            lib[k].append((r, (seg / abs(dh)).astype(np.float32)))
                else:
                    r["kept_RV"] = r["kept_RH"] = False
                rows.append(r)
    keys = ["subject", "run", "onset", "offset", "detector_amp_uv", "dheog_uv", "dveog_uv", "abs_max_uv", "veog_transient_uv",
            "r1_inside", "r2_isolated", "r3_step", "r4_no_artefact", "rh_vertical_ok", "kept_RH", "kept_RV"]
    with open("results/a3/event_library.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, keys); wr.writeheader(); [wr.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    cum, left = {}, rows
    for k in COMMON:
        left = [r for r in left if r.get(k)]; cum[k] = len(left)
    summ = {"rules": {"min_step_uv": MIN_STEP, "max_abs_uv": MAX_ABS, "RH_max_dveog_over_dheog": MAX_VRATIO, "RH_max_veog_transient_uv": MAX_VTRANS,
                      "epoch_s": EPOCH, "baseline_s": BASE, "transient_window_s": TRANS},
            "rightward_saccades": len(rows), "remaining_after_common_rule": cum, "libraries": {}}
    Path("cache").mkdir(exist_ok=True); i7, i8 = ch_index(ch, ["AF7", "AF8"])
    for k, items in lib.items():
        R = [it[0] for it in items]; E = np.stack([it[1] for it in items])
        dh = np.abs([r["dheog_uv"] for r in R]); dv = np.array([r["dveog_uv"] for r in R]) / np.array([r["dheog_uv"] for r in R])
        summ["libraries"][k] = {"n": len(R), "per_subject": {str(s): sum(r["subject"] == s for r in R) for s in pilots},
                                "n_subjects": len({r["subject"] for r in R}), "largest_subject_share": max(sum(r["subject"] == s for r in R) for s in pilots) / len(R),
                                "dheog_uv_quantiles": {q: round(float(np.quantile(dh, q)), 1) for q in (0.1, 0.5, 0.9)},
                                "dveog_over_dheog_quantiles": {q: round(float(np.quantile(dv, q)), 2) for q in (0.1, 0.5, 0.9)},
                                "veog_transient_uv_quantiles": {q: round(float(np.quantile([r["veog_transient_uv"] for r in R], q)), 1) for q in (0.1, 0.5, 0.9)}}
        np.savez_compressed(f"cache/a3_event_library_{k}.npz", epochs=E, subject=np.array([r["subject"] for r in R]), run=np.array([r["run"] for r in R]),
                            onset=np.array([r["onset"] for r in R]), offset_rel=np.array([r["offset"] - r["onset"] for r in R]),
                            dheog=np.array([r["dheog_uv"] for r in R]), sfreq=sf, onset_index=-e0, ch_names=np.array(ch))
    summ["sham"] = {"n": len(sham), "per_subject": {str(s): sum(t[0] == s for t in sham) for s in pilots}}
    np.savez_compressed("cache/a3_event_library_sham.npz", epochs=np.stack([t[3] for t in sham]), subject=np.array([t[0] for t in sham]),
                        run=np.array([t[1] for t in sham]), onset=np.array([t[2] for t in sham]), sfreq=sf, onset_index=-e0, ch_names=np.array(ch))
    json.dump(summ, open("results/a3/event_library_summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
