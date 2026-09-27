"""Signal-access guard (3.1).

The outline restricts *signal* statistics during exclusion checks and pilot
statistics to the 15 pilot subjects, while the generator (3.4.2/3.4.3) may
read every subject's baseline and execution runs (never labels).  Every signal
read goes through `SignalAccessGuard.record`, which

* appends a JSONL line (subject, runs, purpose) that becomes part of the
  preregistration record ("预实验看过的数据与结果清单", 6.3 item 1);
* raises if a purpose restricted to pilot subjects touches a non-pilot subject;
* raises if a generator purpose touches runs other than those allowed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PILOT_ONLY_PURPOSES = {
    "pilot_t0_stats", "pilot_return_saccades", "pilot_saccade_ica", "pilot_saccade_amplitude",
    "pilot_sensitivity_curve", "pilot_trial_counts", "pilot_injection_check", "pilot_saccade_regression",
}
# generator component estimation: allowed on every subject, restricted runs, no labels
GENERATOR_PURPOSES = {
    "gen_alpha_component": {1, 2},
    "gen_mu_component": {3, 7, 11},
}
# purposes allowed for any subject once the design is frozen
OPEN_PURPOSES = {"background_windows", "real_audit", "natural_counterfactual", "training", "test"}


class SignalAccessGuard:
    def __init__(self, log_path: str | Path | None, pilot_subjects: set[int] | None = None,
                 enforce: bool = True):
        self.log_path = Path(log_path) if log_path else None
        self.pilot = set(pilot_subjects or [])
        self.enforce = enforce
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, subject: int, runs, purpose: str) -> None:
        runs = sorted({int(r) for r in runs})
        if self.enforce:
            if purpose in PILOT_ONLY_PURPOSES and subject not in self.pilot:
                raise PermissionError(f"purpose '{purpose}' is pilot-only; subject {subject} is not a pilot subject")
            if purpose in GENERATOR_PURPOSES and not set(runs) <= GENERATOR_PURPOSES[purpose]:
                raise PermissionError(f"purpose '{purpose}' may only read runs {sorted(GENERATOR_PURPOSES[purpose])}")
            if (purpose not in PILOT_ONLY_PURPOSES and purpose not in GENERATOR_PURPOSES
                    and purpose not in OPEN_PURPOSES):
                raise ValueError(f"unknown access purpose '{purpose}'")
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "subject": subject, "runs": runs, "purpose": purpose}) + "\n")


NULL_GUARD = SignalAccessGuard(None, enforce=False)
