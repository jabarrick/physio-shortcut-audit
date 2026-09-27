"""Watchdog for hanging p3audit runs (PILOT_LOG 10.8 / 11.10 / 13.8).

P4/P5 have hung twice with the process alive, one core spinning and nothing
written (10.8, 11.10); `train_eval` there does training and BA only, so a long
silence in those stages really is a hang.

*** READ THIS BEFORE SETTING --stall-min ***
A unit run (`pilot P8`, `run-units`) writes its json only when the WHOLE unit is
finished, and a unit ending in integrated gradients is legitimately silent for a
long time: IG does K+1 = 7 forward+backward passes at an effective batch of
ig.batch x ig.steps for each of ig.windows_per_unit windows, which on CBraMod is
tens of minutes per unit.  A --stall-min below that kills real work.  Default is
therefore 45 minutes; lower it only for stages without IG (P4/P5 calibration).

This wrapper launches a p3audit command, watches the directory it writes results
into, and when nothing has been written for --stall-min minutes it

  1. dumps the Python stack of every thread with py-spy into
     results/hang_dumps/<timestamp>.txt   (so the NEXT hang is diagnosed, not lost),
  2. kills the process tree,
  3. waits --cooldown seconds for the CUDA context to go away, and
  4. relaunches the same command.

Relaunching is safe because every p3audit stage is resumable from files already on
disk: run_unit returns the existing json (unit.py:72), and the calibration stages
skip finished rows in their *_partial.jsonl.  Nothing is retrained.

This wrapper changes NO project code and takes no decisions; it only restarts a
command that was already safe to restart by hand.

KEEP-AWAKE (PILOT_LOG 13.11).  On Windows the wrapper also asks the OS, for as long
as it runs, not to enter Modern Standby on idle and not to turn the display off
(SetThreadExecutionState).  The Windows event log ties both P4 hangs and the
2026-09-23 GPU loss to Modern Standby entered while a CUDA process was alive;
"sleep: never" does not prevent it, because on these laptops the idle screen-off
IS the entry into standby.  Closing the lid or pressing the power button still
forces standby - keep the lid open and the charger in.  Check it is active with
`powercfg /requests` in an ADMINISTRATOR window (python.exe under SYSTEM and DISPLAY).

GPU-LOSS CHECK (PILOT_LOG 13.12).  Every --gpu-check seconds the wrapper runs
`nvidia-smi`.  If the GPU is gone ("GPU is lost", or nvidia-smi fails) it dumps the
stack, kills the run and EXITS with code 3 instead of waiting --stall-min and
restarting: a lost GPU only comes back after a reboot, and with training.device =
cuda a restart would fail at once anyway.  After the reboot, run the same command
again; finished units are skipped.

GPU TELEMETRY.  While the run is alive, `nvidia-smi` samples temperature, power,
clocks, utilisation, memory and throttle reasons every --telemetry seconds into
results/gpu_telemetry/<timestamp>.csv, so the last rows before a loss are kept.

Examples
--------
    python watchdog.py --watch results/pilots/p8_units -- pilot P8
    python watchdog.py --watch results/units -- run-units --model cbramod
    python watchdog.py --watch results/pilots --stall-min 15 -- pilot P4 --confound saccade
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def newest_mtime(watch: Path) -> float:
    """Most recent mtime under `watch` (recursive), or 0.0 if it holds no files."""
    best = 0.0
    if not watch.exists():
        return best
    for p in watch.rglob("*"):
        try:
            if p.is_file():
                best = max(best, p.stat().st_mtime)
        except OSError:          # file vanished between listing and stat
            continue
    return best


def dump_stacks(pid: int, out_dir: Path) -> Path | None:
    """py-spy dump of `pid` -> a text file.  Returns None if py-spy is unavailable."""
    exe = shutil.which("py-spy")
    if exe is None:
        print("[watchdog] py-spy not installed - no stack dump.  pip install py-spy", flush=True)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"hang_{datetime.now():%Y%m%d_%H%M%S}_pid{pid}.txt"
    try:
        r = subprocess.run([exe, "dump", "--pid", str(pid), "--locals"],
                           capture_output=True, text=True, timeout=120)
        out.write_text((r.stdout or "") + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""),
                       encoding="utf-8")
        print(f"[watchdog] stack dump written to {out}", flush=True)
        if r.returncode != 0:
            print("[watchdog] py-spy returned nonzero; try an elevated shell for the next one",
                  flush=True)
        return out
    except Exception as e:                                        # noqa: BLE001
        print(f"[watchdog] py-spy failed: {e}", flush=True)
        return None


def keep_awake(on: bool) -> bool:
    """Windows only: hold (on=True) or release a system+display 'required' request for this
    thread.  Returns True if the request was accepted.  No-op elsewhere."""
    if os.name != "nt":
        return False
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
    fn = ctypes.windll.kernel32.SetThreadExecutionState
    fn.argtypes, fn.restype = [ctypes.c_uint], ctypes.c_uint
    flags = ES_CONTINUOUS | ((ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if on else 0)
    return fn(flags) != 0


def gpu_state() -> tuple[bool | None, str]:
    """(True, name) if nvidia-smi sees a GPU; (False, message) if it reports none or a lost one;
    (None, reason) if nvidia-smi is not installed or did not answer - then nothing is concluded."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, "nvidia-smi not found"
    try:
        r = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except Exception as e:                                        # noqa: BLE001
        return None, f"nvidia-smi did not answer: {e}"
    msg = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0 or "lost" in msg.lower() or "no devices" in msg.lower() or not msg:
        return False, msg or f"nvidia-smi exit code {r.returncode}"
    return True, msg.splitlines()[0]


TELEMETRY_FIELDS = ("timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,"
                    "memory.used,memory.total,pstate,clocks_throttle_reasons.active")


def start_telemetry(out_dir: Path, every_s: int):
    """Sample nvidia-smi every `every_s` s from a thread; append and fsync EVERY row.
    (v1 used `nvidia-smi -l -f`, which buffers on Windows: the 2026-09-23 19:40 run lost the GPU
    and left an empty file - PILOT_LOG 13.13.)  A failed sample is written too, with the error,
    so the file shows the moment the GPU went.  Returns a stop() callable or None."""
    import threading
    exe = shutil.which("nvidia-smi")
    if exe is None or every_s <= 0:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gpu_{datetime.now():%Y%m%d_%H%M%S}.csv"
    stop = threading.Event()

    def write(line: str) -> None:
        with open(out, "a", encoding="utf-8") as f:
            f.write(line + "\n"); f.flush(); os.fsync(f.fileno())

    def loop():
        write("local_time," + TELEMETRY_FIELDS)
        while not stop.is_set():
            now = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            try:
                r = subprocess.run([exe, f"--query-gpu={TELEMETRY_FIELDS}", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=20)
                txt = " | ".join(((r.stdout or "") + (r.stderr or "")).split("\n")).strip(" |")
                write(f"{now},{txt}" if r.returncode == 0 else f"{now},ERROR rc={r.returncode}: {txt}")
            except Exception as e:                                # noqa: BLE001
                write(f"{now},ERROR {e}")
            stop.wait(every_s)

    threading.Thread(target=loop, daemon=True).start()
    print(f"[watchdog] GPU telemetry every {every_s}s -> {out}", flush=True)
    return stop.set


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, text=True)
    else:
        proc.kill()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("[watchdog] process did not die; kill it from Task Manager", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Restart a hanging p3audit command, dumping its stack first.")
    ap.add_argument("--watch", required=True,
                    help="directory whose file writes count as progress, e.g. results/pilots/p8_units")
    ap.add_argument("--stall-min", type=float, default=45.0,
                    help="minutes without a write before the run counts as hung.  Default 45: a "
                         "unit writes only when finished, and a CBraMod unit ending in integrated "
                         "gradients is silent for tens of minutes.  Use ~15 for P4/P5, which have "
                         "no IG and write after every training")
    ap.add_argument("--max-restarts", type=int, default=4)
    ap.add_argument("--cooldown", type=int, default=45,
                    help="seconds to wait after killing, so the CUDA context is released")
    ap.add_argument("--poll", type=int, default=20, help="seconds between checks")
    ap.add_argument("--gpu-check", type=int, default=60,
                    help="seconds between nvidia-smi checks for a lost GPU (0 = off)")
    ap.add_argument("--telemetry", type=int, default=10,
                    help="seconds between GPU telemetry samples (0 = off)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="p3audit arguments, after a literal --  (e.g. -- pilot P8)")
    a = ap.parse_args()

    cmd = [x for x in a.cmd if x != "--"]
    if not cmd:
        ap.error("no p3audit command given; put it after a literal --,  e.g.  -- pilot P8")

    watch = (ROOT / a.watch) if not Path(a.watch).is_absolute() else Path(a.watch)
    dumps = ROOT / "results" / "hang_dumps"
    stall = a.stall_min * 60.0
    # -u: unbuffered, so the child's progress lines appear here immediately
    full = [sys.executable, "-u", "-m", "p3audit.cli", *cmd]

    print(f"[watchdog] command : {' '.join(full)}")
    print(f"[watchdog] watching: {watch}")
    print(f"[watchdog] stall   : {a.stall_min:g} min without a write -> dump, kill, restart "
          f"(up to {a.max_restarts} times)")
    if a.stall_min < 30:
        print("[watchdog] NOTE: a unit writes only when it finishes, and integrated gradients on "
              "CBraMod is silent for tens of minutes - below 30 min you may kill real work",
              flush=True)

    ok, msg = gpu_state()
    if ok is False:
        print(f"[watchdog] GPU not available before start: {msg}\n"
              f"[watchdog] reboot, check nvidia-smi, then run this command again", flush=True)
        return 3
    if ok:
        print(f"[watchdog] GPU     : {msg}", flush=True)
    tele = start_telemetry(ROOT / "results" / "gpu_telemetry", a.telemetry)
    if os.name == "nt":
        ok = keep_awake(True)
        print(f"[watchdog] keep-awake: {'ON' if ok else 'FAILED - set screen-off to Never by hand'} "
              f"(idle standby and screen-off blocked; lid and power button still sleep)", flush=True)
    try:
        return _run(a, full, watch, dumps, stall)
    finally:
        keep_awake(False)
        if tele is not None:
            tele()


def _run(a, full, watch, dumps, stall) -> int:
    for attempt in range(a.max_restarts + 1):
        if attempt:
            print(f"\n[watchdog] === restart {attempt}/{a.max_restarts} ===", flush=True)
        proc = subprocess.Popen(full, cwd=str(ROOT))
        last_seen = max(newest_mtime(watch), time.time())   # grace period for startup
        last_gpu = time.time()
        while True:
            time.sleep(a.poll)
            rc = proc.poll()
            if rc is not None:
                print(f"\n[watchdog] command exited with code {rc}", flush=True)
                return rc
            if a.gpu_check > 0 and time.time() - last_gpu >= a.gpu_check:
                last_gpu = time.time()
                ok, msg = gpu_state()
                if ok is False:
                    print(f"\n[watchdog] GPU LOST at {datetime.now():%Y-%m-%d %H:%M:%S}: {msg}",
                          flush=True)
                    dump_stacks(proc.pid, dumps)
                    kill_tree(proc)
                    print("[watchdog] not restarting - a lost GPU needs a reboot.  Reboot, check "
                          "nvidia-smi, then run the same command again (finished units are skipped).",
                          flush=True)
                    return 3
            m = newest_mtime(watch)
            if m > last_seen:
                last_seen = m
                continue
            idle = time.time() - last_seen
            if idle > stall:
                print(f"\n[watchdog] no write to {watch} for {idle / 60:.1f} min - treating as hung",
                      flush=True)
                dump_stacks(proc.pid, dumps)
                kill_tree(proc)
                print(f"[watchdog] cooling down {a.cooldown}s", flush=True)
                time.sleep(a.cooldown)
                break

    print(f"[watchdog] gave up after {a.max_restarts} restarts; "
          f"see {dumps} for the stack dumps", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
