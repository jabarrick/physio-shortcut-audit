# Do physiological shortcut audits work for EEG decoders?

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22994247.svg)](https://doi.org/10.5281/zenodo.22994247)

Code and unit-level results for

> Yu Gao. *Do Physiological Shortcut Audits Work for EEG Decoders? A Semi-Synthetic Ground-Truth Test with a Natural Counterfactual Anchor.* Manuscript, 2026.

The study was **preregistered on OSF** after the pilot phase and before any confirmatory unit was trained:
[doi:10.17605/OSF.IO/8Q7N6](https://doi.org/10.17605/OSF.IO/8Q7N6) (project [osf.io/37s6b](https://osf.io/37s6b)).
The registration fixed the configuration and the analysis code by hash; this repository lets you check both (see [Verifying the registered code](#verifying-the-registered-code)).

Release v1.0.0 of this repository is archived on Zenodo: [doi:10.5281/zenodo.22994247](https://doi.org/10.5281/zenodo.22994247).

The package is called `p3audit` (the working name of the project); the name is kept because the registered code hash covers the package as it was frozen.

---

## What is in the repository

```
p3audit/            analysis package (frozen by hash - do not edit, see below)
  data/             EEGMMIDB metadata, exclusion rules, subject split, signal streams, windowing, SHU-MI loader
  generator/        semi-synthetic generator: mu task component, posterior alpha, saccade step + spike potential, basis cache
  models/           EEGNet-8,2, ShallowConvNet, the "csoanet" multi-scale CNN, CBraMod / LaBraM adapters
  metrics/          ground truth, probe (encoding), erasure, HEOG regression / SRI, integrated gradients, spectral
  saccades/         saccade detector and detection-sensitivity curve (natural counterfactual)
  experiments/      pilots, calibration, semi-synthetic units, real audit, SHU-MI audit, H1-H5 analysis
  stats/            sufficient statistics, hierarchical bootstrap, power
  utils/access.py   signal-access guard: logs every signal read with its purpose and restricts which subjects each purpose may read
configs/default.yaml   every numerical setting (config_hash d3d4d36229fe8662)
configs/dryrun.yaml    synthetic dry-run settings
tests/                 unit tests (pytest)
watchdog.py            runner used for all long jobs: restarts hung runs, records GPU telemetry
deviations/            the single post-registration code deviation D-2026-09-26 (diff + frozen originals)
scripts/               post-registration summaries, figures and post hoc analyses (outside the hashed package)
  diagnostics/         one-off pilot-phase diagnostics
  data/                data download / extraction helpers
  archive/             the overnight pilot script as it was run (from the repo root, before scripts/ was reorganised)
results/               unit-level results (see below)
figures/               manuscript figures (scripts/make_figures.py)
docs/                  original implementation notes (Chinese, 2026-09-18, pre-pilot; partly superseded)
```

Comments in the code refer to sections of the execution log (`PILOT_LOG x.y`) and of the study outline (`3.4.2`, `6.3`, ...). The execution log is deposited on OSF with the preregistration.

## Models

| name in code | model | notes |
|---|---|---|
| `eegnet` | EEGNet-8,2 (Lawhern et al., 2018) | |
| `shallow` | ShallowConvNet (Schirrmeister et al., 2017) | |
| `csoanet` | compact multi-scale CNN, class `CSOANetPlaceholder` | three temporal branches with ~72 / 200 / 776 ms receptive fields feeding a depthwise-spatial / separable stage (3,510 parameters). It reproduces only this multi-scale property of CSOANet; it is **not** the published CSOANet implementation. It was used unchanged in every experiment, because the code was frozen by hash. |
| `cbramod` | CBraMod (official repository + weights, official fine-tuning recipe) | |
| `labram` | LaBraM-base (official repository, commit `c431221e`, bundled checkpoint) | supplementary; excluded from the real PhysioNet audit because EEGMMIDB is in its pretraining corpus |

## Installation

Tested with Python 3.12.7, PyTorch 2.5.1 + CUDA 12.1, numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, pandas 2.2.3, mne 1.11.0, statsmodels 0.14.6, pytest 9.0.3, timm 1.0.30, on Windows with one NVIDIA RTX 4060 Laptop GPU (8 GB).

```bash
pip install -e .[dev]            # core + pytest
pip install -e .[foundation]     # timm, einops (LaBraM)
pip install -e .[figures]        # matplotlib (scripts/make_figures.py)
pytest -q
```

### External resources

| resource | where | configured by |
|---|---|---|
| PhysioNet EEG Motor Movement/Imagery (EEGMMIDB) | downloaded automatically by `mne.datasets.eegbci`; `scripts/data/prefetch.py` downloads the needed runs in parallel | `paths.data_root` (`~/mne_data`) |
| SHU-MI | figshare 19228725 (the archive is encrypted; access is granted by the dataset authors) | `paths.shu_root` |
| EEGEyeNet (pilot P10a only) | `Direction_task_with_dots_synchronised_min.npz`, linked from OSF ktv7m | `pilot P10a --npz <file>` |
| CBraMod | <https://github.com/wjq-learning/CBraMod>; weights `pretrained_weights.pth` from Hugging Face `weighting666/CBraMod` | `foundation.cbramod.repo_path`, `.checkpoint` |
| LaBraM | <https://github.com/935963004/LaBraM> at commit `c431221e` (includes `checkpoints/labram-base.pth`) | `foundation.labram.repo_path`, `.checkpoint` |

`configs/default.yaml` points the foundation models to `D:/project/CBraMod` and `D:/project/LaBraM`, as on the machine used for the study. To use other locations, put the overrides in a small YAML file and pass it with `--config` (it is merged over `default.yaml`), or use `--set key.path=value`. **The configuration hash includes these paths**, so a run with other paths reports a config hash different from the registered `d3d4d36229fe8662` even when every scientific setting is identical.

## Reproducing the study

All commands run from the repository root. Every stage writes one file per unit and resumes from completed units. Long jobs were run through the watchdog, e.g. `python watchdog.py --watch results/units -- --confirmatory run-units`.

```bash
# dry run on synthetic subjects (no data download; numbers are meaningless, only code paths are exercised)
p3audit --config configs/dryrun.yaml --synthetic 14 prepare --stage all
p3audit --config configs/dryrun.yaml --synthetic 14 run-units --max-epochs 3

# 1. data preparation: exclusion, split, per-subject components, saccade template pool, basis cache (~10 GB)
p3audit prepare --stage split
p3audit prepare --stage components
p3audit prepare --stage pool
p3audit prepare --stage cache

# 2. pilots (already run; results in results/pilots/), e.g.
p3audit pilot P4 --confound alpha

# 3. confirmatory semi-synthetic units (358 units; ~32 GPU hours on the hardware above)
p3audit --confirmatory run-units          # filters: --module --model --confound --shard i/n

# 4. real-data audits
p3audit --confirmatory real-audit         # PhysioNet-MI, 5 folds x 3 seeds x 4 core models
p3audit --confirmatory shu-audit          # SHU-MI (supplementary H5)

# 5. registered analysis (H1-H5) -> results/analysis/
p3audit analyse

# 6. summaries, sensitivity and post hoc analyses, figures
python scripts/units_summary.py
python scripts/real_summary.py
python scripts/summarize_shu.py
python scripts/h3_without_stress.py       # registered sensitivity analysis
python scripts/posthoc_revision.py        # exploratory, not preregistered
python scripts/posthoc_revision2.py       # exploratory, not preregistered
python scripts/posthoc_revision2_diff.py  # exploratory, not preregistered
python scripts/posthoc_revision2_real.py  # exploratory, not preregistered
python scripts/make_figures.py
```

`--confirmatory` refuses to run while any setting in the configuration's TBD registry is still unconfirmed. `p3audit tbd` lists them; `p3audit freeze` writes the registration snapshot (`results/prereg_snapshot.json`).

## Results

| path | content |
|---|---|
| `results/units/` | the 358 confirmatory semi-synthetic units (per-subject sufficient statistics for every metric and the ground truth) |
| `results/units_VOID_stale_cache_20260924/` | 12 units voided before any test statistic was examined (two stale background files); re-run in `units/` |
| `results/real/` | real PhysioNet-MI audit, 60 records + per-trial tables |
| `results/shu/` | SHU-MI audit, 65 records |
| `results/analysis/` | H1-H5 outputs, descriptive tables, sensitivity and post hoc results |
| `results/pilots/` | every pilot result the design was based on |
| `results/prereg_snapshot.json` | registration snapshot: config hash, code hash, split, signal-access record |
| `results/signal_access_log.jsonl` | every signal read by the pipeline, with purpose (test-subject blinding) |
| `results/split.json`, `exclusion.csv`, `unit_list.txt` | subject exclusion and split, registered unit list |
| `results/components/`, `saccade_pool*`, `group_components.json` | generator components and saccade templates |
| `results/gpu_telemetry/`, `results/hang_dumps/` | run provenance |

The basis cache (`cache/`, ~10 GB) and the real-audit input cache (`results/real_cache/`, ~1.3 GB) are not included; they are regenerated by `prepare --stage cache` and `real-audit`.

## Verifying the registered code

```bash
python scripts/verify_code_hash.py
```

The registration recorded `code_sha256 = 4e1ffab9...6e21`, a SHA-256 over the bytes of every `p3audit/**/*.py` file in sorted path order (as computed by `p3audit freeze`). After registration the code changed once:

* **D-2026-09-26** - integrated gradients ran out of GPU memory in the real audit, so `real_audit.py` and `shu_audit.py` now call IG with the registered per-model batch size. The diff and the two original files are in `deviations/`. The resulting code hash is `0d594c15...785a`. All 358 semi-synthetic units ran under the frozen code.

The script recomputes both hashes (the frozen one by substituting the two original files) and the configuration hash. The git history mirrors this: tag `prereg-frozen` is the registered package, tag `deviation-D-2026-09-26` adds the deviation. Note that git commit dates are not evidence of timing; the OSF timestamp and the hashes are.

`.gitattributes` disables line-ending conversion so the hashes also hold on Windows checkouts.

## Licence

Code: MIT (see `LICENSE`). The datasets are subject to their own licences (EEGMMIDB: ODC-By 1.0 on PhysioNet; SHU-MI and EEGEyeNet: see their repositories). CBraMod and LaBraM are not redistributed here.

## Citation

See `CITATION.cff`. Please cite the article, this software (doi:10.5281/zenodo.22994247) and the preregistration (doi:10.17605/OSF.IO/8Q7N6).
