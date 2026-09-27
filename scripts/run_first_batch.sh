#!/usr/bin/env bash
# First pilot batch (outline 6.1), in the prescribed order, on PhysioNet-MI.
# Run locally (physionet.org may be blocked from cloud machines).
set -euo pipefail
CFG=${CFG:-configs/default.yaml}

p3audit --config "$CFG" prepare --stage split        # 3.1 metadata-only exclusion + 3.2 split
p3audit --config "$CFG" pilot P2                     # T0 stats, return saccades -> window_s / trim (edit config!)
p3audit --config "$CFG" pilot P6                     # split-half ICA saccade templates (pilot only)
p3audit --config "$CFG" prepare --stage components   # per-subject GED alpha / mu (no labels)
p3audit --config "$CFG" pilot P1a                    # injection check, overlap, dispersion, mu failures
p3audit --config "$CFG" pilot sensitivity            # detection sensitivity curve (miss-rate lower bound)
p3audit --config "$CFG" pilot P7                     # natural-counterfactual trial counts (provisional)
p3audit --config "$CFG" pilot P3                     # timing / memory (run in parallel on the GPU box)

echo "Now update configs (window_s, trim_ms, trim_mode, amplitude_uv, ...) and run:"
echo "  p3audit --config \$CFG prepare --stage cache"
