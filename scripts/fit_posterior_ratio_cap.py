"""Fit the anti-contamination cap `task_component.posterior_ratio_max` (PILOT_LOG 7.3).

Principle, declared in configs/default.yaml BEFORE this value was computed and independent of
any mu pass rate:

    posterior_ratio(a) = sum(a[alpha_component.posterior]**2) / sum(a**2)

    cap = 10th percentile of posterior_ratio over the alpha components ACCEPTED FOR THE
          PILOT SUBJECTS.

Reading: a mu candidate carrying as large a posterior share as the least-posterior verified
alpha component cannot be told apart from posterior alpha on topography, so it is rejected.
With only one hemisphere retained the bilateral requirement no longer provides that guard.

Estimated on pilot subjects only.  The block headed "not used to set the cap" is printed after
the value is fixed, for the record; it cannot change it.

Usage:  python scripts/fit_posterior_ratio_cap.py [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from p3audit.constants import PHYSIONET_CHANNELS
from p3audit.utils.common import ch_index, write_json

ROOT = Path(__file__).resolve().parents[1]


def posterior_ratio(a: np.ndarray, post_idx: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    return float((a[post_idx] ** 2).sum() / (a**2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the fitted cap into configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/default.yaml", encoding="utf-8"))
    split = json.loads((ROOT / "results/split.json").read_text(encoding="utf-8"))
    pilot = set(split["pilot"])
    post_idx = ch_index(list(PHYSIONET_CHANNELS), cfg["alpha_component"]["posterior"])

    comps = {}
    for f in sorted((ROOT / "results/components").glob("sub-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        comps[d["subject"]] = d

    # ---- the cap: pilot subjects' ACCEPTED alpha components only
    ref = []
    for s in sorted(pilot):
        c = comps.get(s, {}).get("components", {}).get("alpha")
        if c and c.get("passed"):
            ref.append((s, posterior_ratio(c["a"], post_idx)))
    if len(ref) < 5:
        raise SystemExit(f"only {len(ref)} accepted pilot alpha components; refusing to fit a cap")

    vals = np.array([v for _, v in ref])
    cap = float(np.percentile(vals, 10))

    print(f"reference set: {len(ref)} accepted alpha components, pilot subjects "
          f"{[s for s, _ in ref]}")
    print("posterior_ratio of those alpha components:")
    for s, v in ref:
        print(f"   sub {s:3d}  {v:.3f}")
    print(f"\n  min {vals.min():.3f}   p10 {cap:.3f}   median {np.median(vals):.3f}   "
          f"max {vals.max():.3f}")
    print(f"\n==> posterior_ratio_max = {cap:.3f}")

    # ---- record only; declared after the value is fixed and cannot change it
    print("\n--- not used to set the cap: what it would exclude ---")
    for scope, subs in (("pilot", sorted(pilot)), ("all", sorted(comps))):
        n_side = n_cut = 0
        cut = []
        for s in subs:
            for k in ("mu_L", "mu_R"):
                c = comps.get(s, {}).get("components", {}).get(k)
                if not (c and c.get("passed")):
                    continue
                n_side += 1
                if posterior_ratio(c["a"], post_idx) > cap:
                    n_cut += 1
                    cut.append((s, k))
        print(f"  {scope:5s}: {n_cut}/{n_side} currently-passing mu sides exceed the cap"
              + (f" -> {cut}" if cut and scope == "pilot" else ""))

    out = {
        "principle": "p10 of posterior_ratio over pilot subjects' accepted alpha components",
        "reference_subjects": [s for s, _ in ref],
        "reference_values": {str(s): v for s, v in ref},
        "posterior_ratio_max": cap,
        "min": float(vals.min()), "median": float(np.median(vals)), "max": float(vals.max()),
    }
    write_json(ROOT / "results/pilots/P1a_posterior_ratio_cap.json", out)
    print("\nwrote results/pilots/P1a_posterior_ratio_cap.json")

    if args.write:
        p = ROOT / "configs/default.yaml"
        s = p.read_text(encoding="utf-8")
        old = "  posterior_ratio_max: null   # filled by scripts/fit_posterior_ratio_cap.py"
        new = (f"  posterior_ratio_max: {cap:.3f}   # p10 of pilot accepted-alpha posterior share "
               f"(scripts/fit_posterior_ratio_cap.py)")
        if old not in s:
            raise SystemExit("config line not found (already written?); not touching the file")
        p.write_text(s.replace(old, new), encoding="utf-8")
        print(f"wrote posterior_ratio_max = {cap:.3f} into configs/default.yaml")


if __name__ == "__main__":
    main()
