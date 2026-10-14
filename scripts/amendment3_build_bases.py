"""Amendment 3 (in preparation; not yet registered): cached bases of the real-saccade generators.

For every subject with a cache (cache/sub-XXX), and for every cached window, a new basis in the model stream with the
same shape and unit as the existing saccade basis sacc_R (a rightward saccade whose AF7 - AF8 step is -1 uV):
  sacc_<ARM>  a real saccade epoch from cache/a3_event_library_<ARM>.npz (scripts/amendment3_event_library.py);
  sham_<ARM>  a real epoch without saccade or blink (cache/a3_event_library_sham.npz), scaled by 1/|dHEOG| of a
              randomly drawn event of the same library, so that z = 0 windows receive a background segment of the
              same size as the one that comes with an event in z = 1 windows.
ARM is RH (horizontal, primary) or RV (with vertical and blink components, secondary).
Placement follows the frozen saccade basis (p3audit/generator/cache.py): the event onset is the window's stored
'sacc_onset'; after the epoch ends, each channel holds the mean of its last 100 ms (a step persists, as in the
template); the first 100 ms of the epoch are faded in; the window's cosine envelope (taper_s) is applied; the Delta is
streamed with zero padding at least the stream's support on each side, which equals streaming inside the run
(the frozen cache builder relies on the same property).  Draws use keyed RNGs (pseudo-label seed, tag, window id);
a recipient that is itself a pilot subject never receives its own epochs (leave-one-out, as for the templates).
Output: cache_a3/sub-XXX/{sacc,sham}_<ARM>.npy and cache_a3/sub-XXX/draws_<ARM>.json (not tracked by git).
--validate rebuilds the frozen sacc_R basis of the given subjects with the same placement and streaming code and
reports the largest difference to the cached arrays.
Usage (repo root): python scripts/amendment3_build_bases.py --arm RH [--subjects 2 3 ...] [--validate]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUT = Path("cache_a3")


def stream_many(deltas, sf, spec):
    """Stream each region-Delta as if alone in a run: all in one zero buffer, separated by >= the support."""
    from p3audit.data.streams import apply_stream, out_index, stream_support_s
    grid = 4; sup = int(np.ceil(stream_support_s(spec, sf) * sf / grid) * grid); n = deltas[0].shape[1]
    step = n + sup; step += (-step) % grid
    buf = np.zeros((deltas[0].shape[0], sup + len(deltas) * step + sup))
    for i, d in enumerate(deltas):
        buf[:, sup + i * step: sup + i * step + n] = d
    y, _ = apply_stream(buf, sf, spec)
    return [y[:, out_index(sup + i * step, sf, spec): out_index(sup + i * step + n, sf, spec)] for i in range(len(deltas))]


def place(epoch, onset_index, onset, n_core, sf):
    """Epoch (C, T) with its saccade onset at column onset_index -> (C, n_core) with the onset at `onset`."""
    C, T = epoch.shape; e = epoch.astype(np.float64).copy(); f = int(round(0.1 * sf))
    e[:, :f] *= 0.5 * (1 - np.cos(np.pi * np.arange(f) / f))
    w = np.zeros((C, n_core)); a = onset - onset_index; b = a + T
    lo, hi = max(a, 0), min(b, n_core); w[:, lo:hi] = e[:, lo - a:hi - a]
    if b < n_core:
        w[:, b:] = e[:, -f:].mean(1, keepdims=True)
    return w


def main():
    from p3audit.config import load_config
    from p3audit.data.streams import stream_specs
    from p3audit.generator.cache import open_cache
    from p3audit.generator.inject import cosine_envelope, saccade_delta
    from p3audit.utils.common import rng as keyed_rng
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["RH", "RV"], default="RH"); ap.add_argument("--subjects", type=int, nargs="*")
    ap.add_argument("--validate", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(); cfg = load_config("configs/default.yaml"); spec = stream_specs(cfg)["model"]
    sf = 160.0; pad = int(round(cfg["streams"]["pad_s"] * sf)); n_core = int(round(cfg["background"]["window_s"] * sf))
    env = cosine_envelope(n_core, int(round(cfg["alpha_component"]["taper_s"] * sf))); seed = cfg["seeds"]["pseudo_label"]
    subs = a.subjects or sorted(int(p.name[4:]) for p in Path("cache").glob("sub-*") if (p / "meta.json").exists())
    if a.validate:
        pool = np.load("results/saccade_pool.npz"); ref = cfg["physionet"]["original_reference"]
        for s in subs:
            c = open_cache("cache", s); ch = list(__import__("p3audit.constants", fromlist=["x"]).PHYSIONET_CHANNELS)
            D = [saccade_delta(pool[f"s_{r['sacc_template']}"], ch, sf, n_core + 2 * pad, pad, env, r["sacc_onset"], cfg,
                               keyed_rng(seed, "sacc_R", r["wid"]), rightward=True, reference=ref) for r in c.meta["rows"]]
            B = np.stack(stream_many(D, sf, spec)); old = c.load("sacc_R")
            print(f"validate sub-{s:03d}: {len(D)} windows, max |new - cached| = {np.abs(B - old).max():.2e} (max |cached| {np.abs(old).max():.2e})")
        return
    L = np.load(f"cache/a3_event_library_{a.arm}.npz"); S = np.load("cache/a3_event_library_sham.npz")
    E, esub, scale, oi = L["epochs"], L["subject"], 1.0 / np.abs(L["dheog"]), int(L["onset_index"])
    SE, ssub, soi = S["epochs"], S["subject"], int(S["onset_index"])
    for s in subs:
        out = OUT / f"sub-{s:03d}"
        if (out / f"draws_{a.arm}.json").exists() and not a.overwrite:          # written last: marks a complete subject
            continue
        c = open_cache("cache", s); ev_ok = np.where(esub != s)[0]; sh_ok = np.where(ssub != s)[0]
        Ds, Dh, draws = [], [], []
        for r in c.meta["rows"]:
            k = int(keyed_rng(seed, f"a3_sacc_{a.arm}", r["wid"]).choice(ev_ok))
            g = keyed_rng(seed, f"a3_sham_{a.arm}", r["wid"]); m = int(g.choice(sh_ok)); j = int(g.choice(ev_ok))
            for D, ep, o, sc in ((Ds, E[k], oi, 1.0), (Dh, SE[m], soi, scale[j])):
                d = np.zeros((E.shape[1], n_core + 2 * pad)); d[:, pad:pad + n_core] = place(ep, o, r["sacc_onset"], n_core, sf) * sc * env; D.append(d)
            draws.append({"wid": r["wid"], "event": k, "sham": m, "sham_scale_event": j})
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"sacc_{a.arm}.npy", np.stack(stream_many(Ds, sf, spec)).astype(np.float32))
        np.save(out / f"sham_{a.arm}.npy", np.stack(stream_many(Dh, sf, spec)).astype(np.float32))
        json.dump(draws, open(out / f"draws_{a.arm}.json", "w"))
        print(f"sub-{s:03d}: {len(draws)} windows", flush=True)


if __name__ == "__main__":
    main()
