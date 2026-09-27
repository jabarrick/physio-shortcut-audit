"""Parallel pre-fetch of the EEGMMIDB runs that `p3audit prepare --stage split` needs.

Downloads subjects in DESCENDING order so that it does not collide with the main
`p3audit` process, which walks subjects in ascending order. The two meet in the
middle; whichever gets there first wins and the other simply finds the file
already cached.

Usage (from a second terminal tab, while `prepare --stage split` keeps running):

    python prefetch.py                 # subjects 109 down to 40, 4 threads
    python prefetch.py 109 40          # explicit range (high low)
    python prefetch.py 109 40 6        # ... with 6 threads

Safe to interrupt (Ctrl-C) and safe to re-run: pooch downloads each file to a
temporary name and renames it on completion, so an interrupted file is re-fetched
rather than left half-written. Files already present are skipped.
"""
from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from mne.datasets import eegbci

# Runs required by 3.1: baselines 1-2, execution 3/7/11, imagery 4/8/12 and 6/10/14.
RUNS = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14]

HIGH = int(sys.argv[1]) if len(sys.argv) > 1 else 109
LOW = int(sys.argv[2]) if len(sys.argv) > 2 else 40
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

_lock = threading.Lock()
_done = 0
_t0 = time.time()


def fetch(subject: int) -> tuple[int, str]:
    global _done
    err = ""
    for attempt in (1, 2, 3):
        try:
            eegbci.load_data(subject, RUNS, update_path=True, verbose="ERROR")
            err = ""
            break
        except Exception as e:  # noqa: BLE001 - keep going, report at the end
            err = f"{type(e).__name__}: {e}"
            if attempt < 3:
                time.sleep(3 * attempt)
    with _lock:
        _done += 1
        n = _done
    total = abs(HIGH - LOW) + 1
    rate = n / max(time.time() - _t0, 1e-9) * 60
    eta = (total - n) / rate if rate > 0 else float("inf")
    tag = "ok  " if not err else "FAIL"
    print(f"[{n:3d}/{total}] {tag} S{subject:03d}   {rate:.1f} subj/min   ETA {eta:.0f} min",
          flush=True)
    return subject, err


def main() -> int:
    step = -1 if HIGH >= LOW else 1
    subjects = list(range(HIGH, LOW + step, step))
    print(f"pre-fetching {len(subjects)} subjects ({HIGH} -> {LOW}), "
          f"{len(RUNS)} runs each, {WORKERS} threads", flush=True)
    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(fetch, s) for s in subjects]
        try:
            for f in as_completed(futures):
                s, err = f.result()
                if err:
                    failures.append((s, err))
        except KeyboardInterrupt:
            print("\ninterrupted; cancelling remaining subjects", flush=True)
            for f in futures:
                f.cancel()
            return 130
    mins = (time.time() - _t0) / 60
    print(f"\ndone in {mins:.1f} min; {len(failures)} subject(s) failed", flush=True)
    for s, err in sorted(failures):
        print(f"  S{s:03d}  {err}", flush=True)
    if failures:
        print("\nre-run this script to retry the failures (cached files are skipped).", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
