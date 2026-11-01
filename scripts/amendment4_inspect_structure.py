"""Amendment 4, structure check BEFORE registration (prints no signal values).

Lists, for the GuttmannFlury2025 motor-imagery archive: every file with its size; for each BDF the header only
(channel names, channel types as read by MNE, sampling rate, duration); for each *_tobii.csv and *_sync.csv the
column names and the number of rows; for each *_tobii.csv the median interval of the first time-stamp-like column
(to give the sampling rate).  No EEG, EOG or gaze value is read or printed.

Usage (repo root):
  pip install moabb==1.5.0
  python scripts/amendment4_inspect_structure.py --download          # MOABB fetches the 31 subject ZIPs (~26 GB)
  python scripts/amendment4_inspect_structure.py --root <folder with S01..S31>  > amendment4_structure.txt
  python scripts/amendment4_inspect_structure.py --zips <folder with S01.zip..S31.zip> > amendment4_structure.txt
Without --root or --zips, the MOABB cache folder of the dataset is searched.  With --zips the archives are read in
place: BDF headers are parsed from the first bytes of each member (no samples are decoded).
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
from pathlib import Path


def find_root():
    from mne import get_config
    base = Path(get_config("MNE_DATASETS_GUTTMANNFLURY2025_PATH") or get_config("MNE_DATA") or Path.home() / "mne_data")
    hits = sorted({Path(p).parents[1] for p in glob.glob(str(base / "**" / "MI*.bdf"), recursive=True)})
    return hits


def csv_header(p: Path, ts_names=("time", "timestamp", "Time", "TimeStamp", "system_time_stamp", "device_time_stamp")):
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        r = csv.reader(f, dialect); cols = next(r)
        ts_idx = next((i for i, c in enumerate(cols) if any(c.strip().lower() == n.lower() for n in ts_names)), None)
        if ts_idx is None:
            ts_idx = next((i for i, c in enumerate(cols) if "time" in c.lower()), None)
        n, prev, d = 0, None, []
        for row in r:
            n += 1
            if ts_idx is not None and len(d) < 5000 and len(row) > ts_idx:
                try:
                    t = float(row[ts_idx])
                    if prev is not None:
                        d.append(t - prev)
                    prev = t
                except ValueError:
                    pass
    med = sorted(d)[len(d) // 2] if d else None
    return cols, n, (cols[ts_idx] if ts_idx is not None else None), med


def bdf_header(f):
    """Parse an EDF/BDF header from a binary stream; reads only the header bytes."""
    h = f.read(256)
    ns = int(h[252:256].decode("ascii", "replace").strip()); nrec = int(h[236:244].decode("ascii", "replace").strip())
    dur = float(h[244:252].decode("ascii", "replace").strip())
    e = f.read(256 * ns)
    fld = lambda off, w, i: e[off * ns + w * i: off * ns + w * (i + 1)].decode("ascii", "replace").strip()
    labels = [fld(0, 16, i) for i in range(ns)]
    # offsets in units of ns: label 16, transducer 80, phys dim 8, pmin 8, pmax 8, dmin 8, dmax 8, prefilt 80, nsamp 8
    base_ns = 16 + 80 + 8 + 8 + 8 + 8 + 8 + 80
    nsamp = [int(e[base_ns * ns + 8 * i: base_ns * ns + 8 * (i + 1)].decode("ascii", "replace").strip()) for i in range(ns)]
    dims = [e[(16 + 80) * ns + 8 * i:(16 + 80) * ns + 8 * (i + 1)].decode("ascii", "replace").strip() for i in range(ns)]
    return ns, nrec, dur, labels, [n / dur if dur else None for n in nsamp], dims


def inspect_zips(folder: Path):
    import io, zipfile
    for z in sorted(folder.glob("S*.zip")):
        print("# archive", z.name, z.stat().st_size)
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                if m.is_dir():
                    continue
                print(f"{m.filename}\t{m.file_size}")
                low = m.filename.lower()
                if low.endswith((".bdf", ".edf")):
                    with zf.open(m) as f:
                        ns, nrec, dur, labels, sf, dims = bdf_header(f)
                    print(f"  BDF n_ch={ns} n_records={nrec} record_s={dur} duration_s={nrec * dur:.1f}")
                    print("  channels:", ", ".join(f"{l}:{r:g}Hz:{d}" for l, r, d in zip(labels, sf, dims)))
                elif low.endswith(("_tobii.csv", "_sync.csv")):
                    with zf.open(m) as fb:
                        txt = io.TextIOWrapper(fb, encoding="utf-8-sig", errors="replace", newline="")
                        import tempfile; tmpp = Path(tempfile.gettempdir()) / "amendment4_tmp_header.csv"
                        import itertools; head = list(itertools.islice(txt, 5001))
                        rest = sum(1 for _ in txt)
                    tmpp.write_text("".join(head), encoding="utf-8")
                    cols, n, ts, med = csv_header(tmpp); tmpp.unlink()
                    print(f"  CSV rows={n + rest} timestamp_col={ts} median_dt_first5000={med}")
                    print("  columns:", " | ".join(cols))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--zips"); ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.zips:
        inspect_zips(Path(a.zips)); return
    if a.download:
        from moabb.datasets import GuttmannFlury2025_MI
        ds = GuttmannFlury2025_MI()
        for s in ds.subject_list:
            print("subject", s, ds.data_path(s), flush=True)
        return
    import mne
    roots = [Path(a.root)] if a.root else find_root()
    if not roots:
        sys.exit("archive not found; pass --root")
    for root in roots:
        print("# root", root)
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            print(f"{p.relative_to(root)}\t{p.stat().st_size}")
            if p.suffix.lower() in (".bdf", ".edf"):
                raw = mne.io.read_raw(p, preload=False, verbose="ERROR")
                print(f"  BDF sfreq={raw.info['sfreq']} n_ch={len(raw.ch_names)} duration_s={raw.n_times / raw.info['sfreq']:.1f}")
                print("  channels:", ", ".join(f"{c}:{t}" for c, t in zip(raw.ch_names, raw.get_channel_types())))
            elif p.name.endswith(("_tobii.csv", "_sync.csv")):
                cols, n, ts, med = csv_header(p)
                print(f"  CSV rows={n} timestamp_col={ts} median_dt={med}")
                print("  columns:", " | ".join(cols))


if __name__ == "__main__":
    main()
