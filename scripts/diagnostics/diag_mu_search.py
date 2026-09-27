"""Diagnostic: is task_component.n_candidates a binding constraint on mu inclusion?

3.4.2 fixes four inclusion criteria for the mu component (topographic peak in the
hemisphere's sensorimotor neighbourhood; 8-13 Hz spectral peak; task power below rest;
no monotonic rise over 20-40 Hz).  `n_candidates` is NOT one of them: it is the number of
GED components screened before giving up.  The first `prepare --stage components` run
capped it at 6 and 60 of 106 mu failures came back as `no_candidate`, i.e. no screened
component even had its topographic peak in the right neighbourhood.

This script re-runs estimate_mu at a larger search budget WITHOUT touching any inclusion
criterion and WITHOUT writing results/components/, and reports:
  - how many subjects recover, and at which rank their component was found
  - for the subjects that still fail, which criterion actually rejects them
  - the rank distribution, so the budget can be set from the data rather than by fiat

Principle stated before the numbers are read: the search budget will be set to cover every
component with eigenvalue > 1 (every component with rest > task, i.e. every ERD candidate),
because a component rejected for not being searched is not evidence about the subject.
The four inclusion criteria stay exactly as preregistered.

Usage:
    python diag_mu_search.py                 # all subjects in the split, budget 40
    python diag_mu_search.py 40              # explicit budget
    python diag_mu_search.py 40 pilot        # pilot subjects only (fast check first)

Writes results/pilots/mu_search_diag.json .  Run time is roughly (budget / 6) times the
components stage, so budget 40 over 105 subjects takes on the order of ten minutes.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from p3audit.config import load_config
from p3audit.experiments.context import Context
from p3audit.generator.components import estimate_mu

CRITERIA = ("topo_peak_in_neighborhood", "band_peak", "task_lt_rest", "emg_not_rising")

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SCOPE = sys.argv[2] if len(sys.argv) > 2 else "all"


def main() -> int:
    cfg = load_config(None, {})
    ctx = Context.create(cfg)
    split = ctx.require_split()
    subjects = split.pilot if SCOPE == "pilot" else ctx.all_subjects

    big = cfg.copy()
    big["task_component"]["n_candidates"] = BUDGET

    old_pass, new_pass, rows = {}, {}, []
    t0 = time.time()
    for i, s in enumerate(subjects, 1):
        prev = ctx.load_components(s)
        runs = [ctx.source.load(s, r) for r in (3, 7, 11)]
        mu_l, mu_r = estimate_mu(runs, big)
        row = {"subject": s, "pilot": s in split.pilot, "role": split.role(s)}
        for side, comp in (("mu_L", mu_l), ("mu_R", mu_r)):
            was = bool(prev[side].passed) if prev else None
            ch = comp.checks
            row[side] = {
                "was_passed": was,
                "now_passed": bool(comp.passed),
                "rank": ch.get("candidate"),
                "recovered": bool(comp.passed and was is False),
                "fails": [c for c in CRITERIA if ch.get(c) is False],
                "no_candidate": bool(ch.get("no_candidate")),
                "eigval": comp.eigval,
                "band_peak_db": ch.get("band_peak_db"),
                "emg_slope": ch.get("emg_slope"),
            }
            old_pass[side] = old_pass.get(side, 0) + (1 if was else 0)
            new_pass[side] = new_pass.get(side, 0) + (1 if comp.passed else 0)
        rows.append(row)
        if i % 10 == 0 or i == len(subjects):
            el = time.time() - t0
            print(f"[{i:3d}/{len(subjects)}] {el/60:.1f} min elapsed, "
                  f"ETA {(len(subjects) - i) * el / i / 60:.1f} min", flush=True)

    def both(key):
        return sum(1 for r in rows if r["mu_L"][key] and r["mu_R"][key])

    ranks = [r[s]["rank"] for r in rows for s in ("mu_L", "mu_R")
             if r[s]["now_passed"] and r[s]["rank"] is not None]
    over_6 = [x for x in ranks if x >= 6]
    still_fail = {}
    for r in rows:
        for s in ("mu_L", "mu_R"):
            if r[s]["now_passed"]:
                continue
            key = "+".join(r[s]["fails"]) or ("no_candidate" if r[s]["no_candidate"] else "UNKNOWN")
            still_fail[f"{s}:{key}"] = still_fail.get(f"{s}:{key}", 0) + 1

    out = {
        "budget": BUDGET, "scope": SCOPE, "n_subjects": len(subjects),
        "pass_counts": {"before": old_pass, "after": new_pass},
        "both_mu_before": both("was_passed"), "both_mu_after": both("now_passed"),
        "recovered_sides": sum(1 for r in rows for s in ("mu_L", "mu_R") if r[s]["recovered"]),
        "accepted_rank": {
            "median": float(np.median(ranks)) if ranks else None,
            "p90": float(np.percentile(ranks, 90)) if ranks else None,
            "max": int(max(ranks)) if ranks else None,
            "n_beyond_old_cap_6": len(over_6),
            "histogram": {str(k): int(sum(1 for x in ranks if x == k)) for k in sorted(set(ranks))},
        },
        "still_failing_signatures": dict(sorted(still_fail.items(), key=lambda kv: -kv[1])),
        "per_subject": rows,
        "note": "diagnostic only; results/components/ untouched. Inclusion criteria unchanged — "
                "only the number of GED components screened.",
    }
    p = ctx.results / "pilots" / "mu_search_diag.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    print(f"\nbudget {BUDGET}, {len(subjects)} subjects")
    print(f"  both mu ok : {out['both_mu_before']} -> {out['both_mu_after']}")
    print(f"  sides recovered: {out['recovered_sides']}")
    print(f"  accepted rank: median {out['accepted_rank']['median']}, "
          f"p90 {out['accepted_rank']['p90']}, max {out['accepted_rank']['max']}, "
          f"{out['accepted_rank']['n_beyond_old_cap_6']} beyond the old cap of 6")
    print("  still failing:")
    for k, v in out["still_failing_signatures"].items():
        print(f"    {v:4d}  {k}")
    print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
