# D-2026-11-01: Amendment 4 training started before the OSF timestamp

Amendment 4 (OSF Registries, osf.io/vmeas, doi:10.17605/OSF.IO/VMEAS) was registered on 1 November 2026 at
10:32:48 (UTC; Lisbon time). Section 9 of the amendment states that no listed run starts before the amendment has an
OSF timestamp.

What happened. The training step (`python watchdog.py --watch results/gf2025 --script scripts/amendment4_gf2025.py
-- --models csoanet eegnet shallow cbramod`) was started at 10:08:08, about 25 minutes before the timestamp. By
10:32:48, 22 records of fold 0 had been written: arm `main` for all four models and seeds 0-2 (12 records), and arm
`hor_ols` for MS-CNN, EEGNet and ShallowConvNet (seeds 0-2) and CBraMod (seed 0) (10 records). Their balanced
accuracies were printed on the author's screen.

What did not change. The amendment text, code and data fingerprints had been fixed before the run started (commit
442370ed690913a2012922722c2fb3a56c8b8719, pushed and checked on 1 November 2026 at about 09:50; final text written
before 10:00) and were registered unchanged. No statistic of section 6 of the amendment was computed before
registration.

Handling (decided by the author on 1 November 2026, before any registered statistic was computed). The 22 records
(JSON and per-trial CSV) were moved unchanged to `results/gf2025_before_registration/`. After the remaining records
are complete, the same command is run again; because the script skips existing records, it re-trains exactly these
22 records with the same folds, seeds and settings. The registered analysis uses only records written after the
timestamp. The moved records are kept and their agreement with the re-trained ones (balanced accuracy per record)
is reported as a check.

Addendum (1 November 2026, 21:00). The re-training step was skipped by mistake: `--analyse` was run at 20:50 on 338
of the 360 records (fold 0 lacked the 22 moved records of arms `main` and `hor_ols`), and its printed output was seen
by the author. That output is invalid as the registered analysis and is kept, unchanged, as
`results/analysis/amendment4_INCOMPLETE_338records_20261101T2050.json`. The 22 records are then re-trained and
`--analyse` is run again on all 360 records; only that run is reported as the registered result.
