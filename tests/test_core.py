"""Fast unit tests of the mathematical core (run: pytest -q)."""
import json
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")

from p3audit.config import assert_confirmatory_ready, load_config, tbd_status
from p3audit.constants import PHYSIONET_CHANNELS
from p3audit.data.exclusion import exclusion_table, included_subjects
from p3audit.data.sources import SyntheticSource
from p3audit.data.splits import make_split, real_audit_folds
from p3audit.data.streams import apply_stream, out_index, stream_specs
from p3audit.data.windows import half_assignment, pseudo_labels, t0_windows
from p3audit.generator.cache import stream_window_deltas
from p3audit.generator.components import (Component, estimate_alpha, ged, pattern,
                                          required_components, select_mu, task_bases)
from p3audit.generator.saccade_ica import ocular_features, ocular_gate
from p3audit.generator.semisynth import task_coefficients
from p3audit.generator.design import assign_z, enumerate_units
from p3audit.generator.inject import cosine_envelope, mirror_index, saccade_delta
from p3audit.metrics.attribution import ig_filterbank
from p3audit.metrics.erasure import fit_leace, residual_concept
from p3audit.models.convnets import CSOANetPlaceholder, EEGNet, RawFeatureLinear, ShallowConvNet
from p3audit.stats.suffstats import ba_from_counts, confusion_by_subject
from p3audit.utils.access import SignalAccessGuard
from p3audit.utils.common import balanced_accuracy, ch_index


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"background.window_s": 3.0})


@pytest.fixture(scope="module")
def src():
    return SyntheticSource(n_subjects=6)


def test_stream_linearity_exact(cfg, src):
    """stream(x + Δ) == stream(x) + stream(Δ) per window (injection before filtering, 3.4.2)."""
    run = src.load(1, 4)
    spec = stream_specs(cfg)["model"]
    wins, _ = t0_windows(run, cfg)
    rng = np.random.default_rng(0)
    deltas, emb = [], np.zeros_like(run.data)
    for w in wins:
        a, b = w.region
        d = np.zeros((64, b - a))
        d[:, w.pad:w.pad + w.length] = rng.standard_normal((64, w.length)) * cosine_envelope(w.length, 40)
        deltas.append(d); emb[:, a:b] += d
    full, _ = apply_stream(run.data + emb, run.sfreq, spec)
    base, _ = apply_stream(run.data, run.sfreq, spec)
    sd = stream_window_deltas(deltas, wins, run.data.shape[1], run.sfreq, spec)
    for w, d in zip(wins, sd):
        oa, ob = out_index(w.region[0], run.sfreq, spec), out_index(w.region[1], run.sfreq, spec)
        assert np.abs(full[:, oa:ob] - (base[:, oa:ob] + d)).max() < 1e-5 * np.abs(d).max()


def test_ged_pattern_normalised():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((8, 8)); S = A @ A.T + np.eye(8); B = rng.standard_normal((8, 8)); R = B @ B.T + np.eye(8)
    ev, W = ged(S, R, 0.01)
    assert np.all(np.diff(ev) <= 1e-9)
    a = pattern(S, W[:, 0])
    assert abs(W[:, 0] @ a - 1) < 1e-9


def test_alpha_component_on_synthetic(cfg, src):
    c = estimate_alpha(src.load(2, 1), src.load(2, 2), cfg)
    assert c.passed and c.checks["posterior_peak"]


def test_saccade_unit_amplitude(cfg):
    ch = PHYSIONET_CHANNELS
    tmpl = np.zeros(64); tmpl[ch.index("AF7")] = 1.0; tmpl[ch.index("AF8")] = -1.0; tmpl[ch.index("F7")] = 0.5
    c2 = cfg.copy(); c2["saccade"]["spike"]["enabled"] = False; c2["saccade"]["topo_jitter"] = 0.0
    env = cosine_envelope(480, 40)
    d = saccade_delta(tmpl, ch, 160.0, 800, 160, env, 100, c2, np.random.default_rng(0), rightward=True)
    i7, i8 = ch_index(ch, ["AF7", "AF8"])
    plateau = d[i7, 160 + 200:160 + 400] - d[i8, 160 + 200:160 + 400]
    assert np.allclose(plateau, c2["saccade"]["rightward_sign"], atol=1e-9)
    m = mirror_index(ch)
    assert np.array_equal(m[m], np.arange(64)) and ch[m[ch.index("C3")]] == "C4"


def test_coupling_exact():
    y = np.tile([0, 1], 50); subj = np.repeat([1, 2], 50)
    z = assign_z(y, subj, 0.9, ("k",))
    assert abs(z[y == 1].mean() - 0.9) < 0.05 and abs(z[y == 0].mean() - 0.1) < 0.05
    z5 = assign_z(y, subj, 0.5, ("k",))
    assert abs(np.corrcoef(z5, y)[0, 1]) < 0.1


def test_units_unique():
    u = enumerate_units(load_config())
    assert len({x.uid for x in u}) == len(u)


def test_leace_removes_linear_information():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n); z = (rng.random(n) < np.where(y == 1, 0.9, 0.1)).astype(int)
    X = rng.standard_normal((n, 10)); X[:, 0] += 2 * z; X[:, 1] += 1.5 * y
    r, _ = residual_concept(z, y)
    er = fit_leace(X, r)
    Xe = er(X)
    cov = ((Xe - Xe.mean(0)) * (r - r.mean())[:, None]).mean(0)
    assert np.abs(cov).max() < 1e-8
    # task information (y) mostly survives because only the residual is erased
    from sklearn.linear_model import LogisticRegression
    assert LogisticRegression().fit(Xe, y).score(Xe, y) > 0.7


def test_ba_from_counts_matches():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300); p = rng.integers(0, 2, 300); s = rng.integers(0, 5, 300)
    assert abs(ba_from_counts(confusion_by_subject(y, p, s)) - balanced_accuracy(y, p)) < 1e-12


def test_ig_completeness():
    torch.manual_seed(0)
    m = EEGNet(8, 400).eval()
    X = np.random.default_rng(0).standard_normal((3, 8, 400)).astype(np.float32) * 10
    bands = {"alpha": [8.0, 13.0], "beta": [13.0, 30.0]}
    names, R, fx = ig_filterbank(m, X, 200.0, bands, steps=64, baseline="band_removed")
    from p3audit.metrics.attribution import band_masks, decompose
    _, masks = band_masks(400, 200.0, bands)
    xt = torch.as_tensor(X); xb = decompose(xt, masks)
    with torch.no_grad():
        out = m(xt); f = out[:, 1] - out[:, 0]
        ob = m(xt - xb[:, 0]); fb = ob[:, 1] - ob[:, 0]
    assert np.allclose(R[:, :, 0].sum(1), (f - fb).numpy(), atol=5e-2 * max(1e-3, float((f - fb).abs().max())))


@pytest.mark.parametrize("cls", [EEGNet, ShallowConvNet, CSOANetPlaceholder, RawFeatureLinear])
def test_models_shapes(cls):
    m = cls(64, 800)
    x = torch.randn(2, 64, 800)
    assert m(x).shape == (2, 2) and m.embed(x).shape[1] == m.embed_dim


@pytest.mark.parametrize("head", ["all_patch_reps", "avgpooling_patch_reps"])
def test_foundation_heads_and_recipe(cfg, head):
    """PILOT_LOG 11.7: official CBraMod head and fine-tuning recipe (stand-in backbone)."""
    from p3audit.models.foundation import build_standin
    from p3audit.training.trainer import make_optimizer
    import copy
    c = copy.deepcopy(cfg)
    c["foundation"]["cbramod_head"] = {"type": head, "dropout": 0.1}
    m = build_standin(c, 64, 600, PHYSIONET_CHANNELS[:64], name="cbramod-STANDIN")
    x = torch.randn(2, 64, 600)
    assert m(x).shape == (2, 2) and m.embed(x).shape[1] == m.embed_dim
    assert m.embed_dim == (200 if head == "all_patch_reps" else 64)
    opt, sched, clip, loss_fn = make_optimizer(m, c, 100)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert len(lrs) == 2 and lrs[0] == pytest.approx(1e-4)
    assert lrs[1] == pytest.approx(1e-3 * (c["training"]["batch_size"] / 256) ** 0.5)
    assert sched is not None and clip == 1.0 and loss_fn.label_smoothing == pytest.approx(0.1)
    # conv nets keep the single-group `training` optimiser
    opt2, sched2, clip2, _ = make_optimizer(EEGNet(64, 600), c, 100)
    assert len(opt2.param_groups) == 1 and sched2 is None and clip2 == 0.0


def test_alpha_reference_ec_eo(cfg):
    """PILOT_LOG 11.13: EC-EO reference gives one log-power change for all subjects."""
    import copy
    from p3audit.generator.semisynth import alpha_log_change
    c = copy.deepcopy(cfg)
    sig = np.array([0.2, 0.6, 1.0])
    c["alpha_component"]["reference"] = "sigma"
    assert np.allclose(alpha_log_change(c, 2.0, sig), 2.0 * c["alpha_component"]["a_star"] * sig)
    c["alpha_component"]["reference"] = "ec_eo"
    out = alpha_log_change(c, 1.5, sig)
    assert np.allclose(out, 1.5 * c["alpha_component"]["a_star"] * c["alpha_component"]["ec_eo_log_ratio"])
    assert np.ptp(out) == 0


def test_per_confound_s_and_crossing(cfg):
    """PILOT_LOG 11.16: per-confound s* and the non-monotone ±10 pp crossing."""
    import copy
    import pandas as pd
    from p3audit.experiments.calibration import _crossing
    c = copy.deepcopy(cfg)
    c["task_component"]["by_confound"] = {"alpha": {"s_star": 1.077, "s_low": None}, "saccade": {"s_star": 1.333}}
    assert c.task_strength("s_star", "alpha") == pytest.approx(1.077)
    assert c.task_strength("s_star", "saccade") == pytest.approx(1.333)
    assert c.task_strength("s_star", None) == pytest.approx(float(c["task_component"]["s_star"]))
    with pytest.raises(ValueError):
        c.task_strength("s_low", "alpha")
    curve = pd.Series([0.50, 0.58, 0.55, 0.65, 0.70, 0.80], index=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    s, m = _crossing(curve, 1.0, 0.60, -1)
    assert m["reached"] and 0.6 < s < 0.8
    s, m = _crossing(curve, 1.0, 0.95, +1)
    assert not m["reached"] and s == 1.2 and m["achieved_diff_pp"] == pytest.approx(10.0)


def test_exclusion_and_split(cfg):
    src = SyntheticSource(n_subjects=90)
    tab = exclusion_table(src, cfg)
    assert tab.set_index("subject").loc[88, "excluded"]
    inc = included_subjects(tab)
    sp = make_split(inc, cfg)
    assert len(sp.pilot) == 15 and len(sp.test) == 34 and len(sp.val) == 10
    assert set(sp.pilot) <= set(sp.train) and not set(sp.test) & set(sp.train)
    folds = real_audit_folds(inc, sp.pilot, 5, 0)
    assert all(set(sp.pilot) <= set(f["train"]) for f in folds)


def test_half_split_within_subject():
    w = [f"{s:03d}-04-{k:02d}" for s in (1, 2) for k in range(10)]
    h = half_assignment(w, 7)
    assert h[:10].sum() == 5 and h[10:].sum() == 5
    assert np.array_equal(h, half_assignment(w, 7))


def test_pseudo_labels_balanced():
    y = pseudo_labels(3, 41, 1)
    assert abs(y.sum() - 20) <= 1


def test_access_guard(tmp_path):
    g = SignalAccessGuard(tmp_path / "log.jsonl", pilot_subjects={1, 2})
    g.record(1, [4], "pilot_t0_stats")
    with pytest.raises(PermissionError):
        g.record(5, [4], "pilot_t0_stats")
    with pytest.raises(PermissionError):
        g.record(5, [4], "gen_alpha_component")     # generator may only read runs 1, 2
    g.record(5, [1, 2], "gen_alpha_component")


def test_tbd_blocks_confirmatory():
    """Mechanism: an unconfirmed registry key blocks a confirmatory run.  Since PILOT_LOG 15.14 the
    real config has every TBD confirmed, so the mechanism is tested on a copy with one key removed."""
    import copy
    cfg = load_config()
    assert all(r["confirmed"] for r in tbd_status(cfg))        # 15.14: registry fully confirmed
    assert_confirmatory_ready(cfg)
    c2 = copy.deepcopy(cfg)
    c2["confirmed"] = [k for k in c2["confirmed"] if k != "stats.h2_delta"]
    assert any(not r["confirmed"] for r in tbd_status(c2))
    with pytest.raises(RuntimeError):
        assert_confirmatory_ready(c2)


# ---------------------------------------------------------------- 3.4.2 single-hemisphere rule
def _mu_side(side, passed, band_peak_db, post_ratio, candidate=0):
    w = np.zeros(64); w[0] = 1.0
    a = np.zeros(64); a[0] = 1.0
    return Component(f"mu_{side}", w, a, 2.0, (8.0, 13.0), passed,
                     {"candidate": candidate, "band_peak_db": band_peak_db,
                      "posterior_ratio": post_ratio})


def test_select_mu_side_rule():
    """Larger rest-segment band-peak residual wins, ties go left, the cap removes a side."""
    c = {"task_component": {"side_rule": "higher_rest_band_peak_db", "posterior_ratio_max": 0.425}}
    m = select_mu(_mu_side("L", True, 2.0, 0.2), _mu_side("R", True, 5.0, 0.2), c)
    assert m.kind == "mu" and m.passed and m.checks["side"] == "R"
    m = select_mu(_mu_side("L", True, 5.0, 0.2), _mu_side("R", True, 2.0, 0.2), c)
    assert m.checks["side"] == "L"
    m = select_mu(_mu_side("L", True, 3.0, 0.2), _mu_side("R", True, 3.0, 0.2), c)
    assert m.checks["side"] == "L"                      # tie -> left
    # the anti-contamination cap drops an otherwise-passing (and stronger) side
    m = select_mu(_mu_side("L", True, 9.0, 0.9), _mu_side("R", True, 2.0, 0.2), c)
    assert m.checks["side"] == "R" and m.checks["eligible"] == ["R"]
    # a side that failed its own criteria is never eligible, however strong its peak
    m = select_mu(_mu_side("L", False, 9.0, 0.2), _mu_side("R", True, 2.0, 0.2), c)
    assert m.checks["side"] == "R"
    # no eligible side -> no task component for this subject
    m = select_mu(_mu_side("L", True, 9.0, 0.9), _mu_side("R", False, 2.0, 0.2), c)
    assert not m.passed and m.checks["no_eligible_side"] and m.checks["side"] is None
    # mu_L / mu_R pass flags are untouched by the cap
    left = _mu_side("L", True, 9.0, 0.9)
    select_mu(left, _mu_side("R", True, 2.0, 0.2), c)
    assert left.passed


def test_select_mu_roundtrips_through_json():
    c = {"task_component": {"side_rule": "higher_rest_band_peak_db", "posterior_ratio_max": 0.425}}
    m = select_mu(_mu_side("L", True, 5.0, 0.2), _mu_side("R", True, 2.0, 0.2), c)
    back = Component.from_dict(json.loads(json.dumps(m.to_dict())))
    assert back.kind == "mu" and back.checks["side"] == "L" and back.passed


def _hemi_cfg(mode):
    return {"task_component": {"hemisphere": mode, "side_rule": "higher_rest_band_peak_db",
                               "posterior_ratio_max": 0.425}}


def test_task_bases_follow_hemisphere():
    assert task_bases(_hemi_cfg("single")) == ("mu",)
    assert task_bases(_hemi_cfg("both")) == ("mu_L", "mu_R")
    assert required_components(_hemi_cfg("single")) == ("alpha", "mu")
    assert required_components(_hemi_cfg("both")) == ("alpha", "mu_L", "mu_R")
    assert task_bases({"task_component": {}}) == ("mu_L", "mu_R")      # default = v8.1 design


def test_task_coefficients_single_is_symmetric():
    """Both classes are modulated by the same magnitude, so 'has been modulated' cannot
    itself be class-discriminative (3.4.2 revision 2026-09-19)."""
    y = np.array([0, 1, 0, 1])
    s_val = 0.5
    c = task_coefficients(y, s_val, _hemi_cfg("single"))
    assert set(c) == {"mu"}
    v = c["mu"]
    assert np.allclose(v[y == 1], -s_val / 2) and np.allclose(v[y == 0], +s_val / 2)
    assert np.allclose(np.abs(v), s_val / 2)          # no untouched class
    assert abs(v.sum()) < 1e-12                       # equal and opposite
    g = 1.0 + v                                       # basis is a unit delta: gain = 1 + coef
    assert np.all(g > 0)
    contrast = 2 * (np.log(1 + s_val / 2) - np.log(1 - s_val / 2))
    assert abs((np.log(g[y == 0] ** 2) - np.log(g[y == 1] ** 2)).mean() - contrast) < 1e-12


def test_task_coefficients_both_keeps_v81():
    y = np.array([0, 1, 0, 1])
    c = task_coefficients(y, 0.5, _hemi_cfg("both"))
    assert set(c) == {"mu_L", "mu_R"}
    assert np.allclose(c["mu_L"][y == 0], -0.5) and np.allclose(c["mu_L"][y == 1], 0.0)
    assert np.allclose(c["mu_R"][y == 1], -0.5) and np.allclose(c["mu_R"][y == 0], 0.0)


def test_task_coefficients_group_suffix():
    y = np.array([0, 1])
    assert set(task_coefficients(y, 0.4, _hemi_cfg("single"), "_group")) == {"mu_group"}
    assert set(task_coefficients(y, 0.4, _hemi_cfg("both"), "_group")) == {"mu_L_group", "mu_R_group"}


# ---------------------------------------------------------------- 3.4.4 P6 ocular gate
def _pat(**vals):
    ch = PHYSIONET_CHANNELS
    p = np.zeros(len(ch))
    for c, v in vals.items():
        p[ch.index(c)] = v
    return p


def test_ocular_features_scale():
    ch = PHYSIONET_CHANNELS
    oc = ch_index(ch, ["AF7", "AF8", "F7", "F8", "Fp1", "Fp2", "FT7", "FT8"])
    i7, i8 = ch_index(ch, ["AF7", "AF8"])
    # a pure horizontal dipole with its extrema at AF7/AF8 is the reference case: bipolarity 2
    enr, bip = ocular_features(_pat(AF7=1.0, AF8=-1.0), oc, i7, i8)
    assert abs(bip - 2.0) < 1e-9 and enr > 4.0
    # a blink is vertical: large at Fp1/Fp2 with the SAME sign -> no horizontal contrast
    enr, bip = ocular_features(_pat(Fp1=1.0, Fp2=1.0, AF7=0.2, AF8=0.2), oc, i7, i8)
    assert bip < 0.1 and enr > 4.0        # ocular by energy, not by geometry
    # an isotropic pattern sits at chance enrichment
    rng = np.random.default_rng(0)
    e = np.mean([ocular_features(rng.standard_normal(64), oc, i7, i8)[0] for _ in range(400)])
    assert abs(e - 1.0) < 0.15


def test_ocular_gate_keeps_dipole_rejects_blink_and_motor(cfg):
    ch = PHYSIONET_CHANNELS
    dipole = _pat(AF7=1.0, AF8=-1.0, F7=0.5, F8=-0.5)
    blink = _pat(Fp1=1.0, Fp2=1.0, AF7=0.2, AF8=0.2)
    motor = _pat(C3=1.0, C1=0.6, CP3=0.5)
    keep = ocular_gate(np.stack([dipole, blink, motor], axis=1), ch, cfg)
    assert keep.tolist() == [0]


def test_ocular_gate_is_label_independent(cfg):
    """The gate must depend on the pattern only — that is what lets the permutation null be
    taken over the gated family instead of all n_pca components."""
    import inspect
    src = inspect.getsource(ocular_gate)
    assert "side" not in src and "label" not in src


def test_bad_window_ocular_exclusion():
    """2026-09-19: max_abs skips saccade.ocular_sites; flat check still covers every channel."""
    import numpy as np
    from p3audit.config import load_config
    from p3audit.constants import PHYSIONET_CHANNELS
    from p3audit.data.windows import amp_channels, bad_window_reason
    cfg = load_config(None, {})
    cfg["background"]["bad_window"]["max_abs_exclude"] = "ocular"   # revision-1 path
    ch = list(PHYSIONET_CHANNELS)
    rng = np.random.default_rng(0)
    core = rng.normal(0, 10, (len(ch), 600))
    mask = amp_channels(ch, cfg)
    assert mask.sum() == len(ch) - len(cfg["saccade"]["ocular_sites"])
    assert bad_window_reason(core, cfg, mask) == ""
    blink = core.copy(); blink[ch.index("Fp1"), 100:140] += 400.0
    assert bad_window_reason(blink, cfg, mask) == ""            # ocular: ignored
    assert bad_window_reason(blink, cfg, None) == "max_abs"     # v8.1 behaviour
    pop = core.copy(); pop[ch.index("C3"), 300] += 400.0
    assert bad_window_reason(pop, cfg, mask) == "max_abs"       # non-ocular pop still caught
    flat = core.copy(); flat[ch.index("Fp2")] = 0.0
    assert bad_window_reason(flat, cfg, mask) == "flat"         # flat check covers ocular sites


def test_robust_z_rule():
    """2026-09-19 rev 2: robust z flags only outliers; amp=False skips the absolute check."""
    import numpy as np
    from p3audit.config import load_config
    from p3audit.data.windows import bad_window_reason, robust_z
    x = np.log10(np.r_[np.full(50, 300.0) * np.linspace(0.9, 1.1, 50), 5000.0])
    z = robust_z(x)
    assert z[-1] > 3 and (np.abs(z[:-1]) < 3).all()
    assert (robust_z(np.ones(5)) == 0).all()
    cfg = load_config(None, {})
    big = np.random.default_rng(1).normal(0, 150, (64, 600))   # > 200 uV peaks, not flat
    assert bad_window_reason(big, cfg, None, amp=False) == ""
    assert bad_window_reason(big, cfg, None, amp=True) == "max_abs"


def test_eog_regression_propagation():
    """2026-09-19: HEOG coefficient recovers the propagation vector despite vertical leakage."""
    import numpy as np
    from p3audit.generator.saccade_regression import propagation
    rng = np.random.default_rng(0)
    b, c = rng.normal(size=64), rng.normal(size=64)
    dh, dv = rng.normal(0, 50, 200), rng.normal(0, 80, 200)
    D = np.outer(dh, b) + np.outer(dv, c) + rng.normal(0, 1, (200, 64))
    assert np.corrcoef(propagation(D, dh, dv), b)[0, 1] > 0.999


def test_p1a_reads_reference_and_per_confound_s(cfg):
    """PILOT_LOG 12.4 (3): P1a must follow alpha_component.reference and the per-confound s*,
    not the sigma formula and not the dead global task_component.s_star placeholder."""
    import copy
    import inspect
    import numpy as np
    from p3audit.experiments import pilots
    from p3audit.generator.semisynth import alpha_log_change
    c = copy.deepcopy(cfg)
    src = inspect.getsource(pilots.p1a)
    # the sigma-only amplitude formula and the global placeholder are gone
    assert 'k_ref' not in src and 'cfg["task_component"]["s_star"]' not in src
    assert 'alpha_log_change(cfg, 1.0, sigma)' in src
    assert 'cfg.task_strength("s_star", conf)' in src
    # and the amplitude it would report equals the generator's, under either reference
    sig = np.array([0.4, 0.9])
    c["alpha_component"]["reference"] = "ec_eo"
    assert np.allclose(alpha_log_change(c, 1.0, sig), c["alpha_component"]["ec_eo_log_ratio"])


def test_ig_batch_invariance(cfg):
    """PILOT_LOG 13.8: ig.batch is a pure performance setting - attributions do not depend on it
    (model in eval mode, every window independent), including a ragged last batch."""
    import copy
    from p3audit.metrics.attribution import ig_batch_for
    torch.manual_seed(0)
    m = EEGNet(8, 400).eval()
    X = np.random.default_rng(1).standard_normal((5, 8, 400)).astype(np.float32) * 10
    bands = {"alpha": [8.0, 13.0], "beta": [13.0, 30.0]}
    for bl in ("band_removed", "zero"):
        _, R1, f1 = ig_filterbank(m, X, 200.0, bands, steps=8, baseline=bl, batch=1)
        for b in (2, 5):                     # 2 leaves a ragged last batch of 1
            _, Rb, fb = ig_filterbank(m, X, 200.0, bands, steps=8, baseline=bl, batch=b)
            tol = 1e-4 * float(np.abs(R1).max())
            assert np.allclose(R1, Rb, rtol=1e-3, atol=tol), (bl, b)
            assert np.allclose(f1, fb, rtol=1e-4, atol=1e-5)
    c = copy.deepcopy(cfg)
    c["ig"]["batch"] = 16
    c["ig"]["batch_by_model"] = {"cbramod": 4}
    assert ig_batch_for(c, "cbramod") == 4 and ig_batch_for(c, "cbramod-STANDIN") == 4
    assert ig_batch_for(c, "eegnet") == 16
    c["ig"].pop("batch"); c["ig"].pop("batch_by_model")
    assert ig_batch_for(c, "eegnet") == 16   # the ig_filterbank default run_unit used before 13.8


def test_resolve_device_cuda_is_strict(monkeypatch):
    """PILOT_LOG 13.9: with training.device = cuda, a missing GPU is an error, never a CPU run."""
    from p3audit.training.trainer import resolve_device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")
    assert resolve_device("auto").type == "cpu" and resolve_device("cpu").type == "cpu"


def test_p8b_verdict_and_p12_inputs(cfg):
    """PILOT_LOG 15.4 D1/D2: the P8b rule (mean of 8 runs vs window) and the P12 input filter
    (extra seeds appended, excluded cell dropped, duplicates refused)."""
    from p3audit.experiments.pilots import p12_inputs, p8b_verdict
    w = cfg["calibration"]["target_delta_ba_neu"]
    assert p8b_verdict([0.25] * 8, w, 1.333)["verdict"] == "keep_s_star"
    assert p8b_verdict([0.28] * 8, w, 1.333)["verdict"] == "keep_s_star"          # bounds inclusive
    assert p8b_verdict([0.30] * 8, w, 1.333)["verdict"] == "try_s_1.6"
    assert p8b_verdict([0.10] * 8, w, 1.333)["verdict"] == "try_s_1.077"
    assert p8b_verdict([0.25] * 6, w, 1.333)["verdict"] == "incomplete"
    u = lambda m, c, s, d: {"model": m, "cell": c, "seed": s, "delta_ba_neu": d}
    base = [u("a", "alpha_p0.9_s_star_a1", 0, .1), u("a", "alpha_p0.9_s_low_a1.5", 0, .4),
            u("a", "saccade_p0.9_s_star_a1", 0, .3)]
    # filter logic with an explicit list (independent of the config's current contents)
    df = p12_inputs(base, [[u("a", "saccade_p0.9_s_star_a1", 2, .2)]], ["alpha_p0.9_s_low_a1.5"])
    assert "alpha_p0.9_s_low_a1.5" not in set(df["cell"]) and len(df) == 3
    # the configured list (PILOT_LOG 15.11): alpha 1.5 a* and the old-s* saccade cells are both dropped
    ex = cfg["stats"]["p12_exclude_cells"]
    assert {"alpha_p0.9_s_low_a1.5", "saccade_p0.9_s_star_a1"} <= set(ex)
    df = p12_inputs(base, [[u("a", "saccade_p0.9_1.6_a1", 0, .2)]], ex)
    assert set(df["cell"]) == {"alpha_p0.9_s_star_a1", "saccade_p0.9_1.6_a1"}
    with pytest.raises(ValueError):
        p12_inputs(base, [[base[0]]], [])


def test_labram_recipe_helpers(cfg):
    """PILOT_LOG 15.26: official LaBraM checkpoint filtering, channel indexing, layer ids and
    layer-wise-decay parameter groups (no repo or checkpoint needed)."""
    from torch import nn
    from p3audit.models.foundation import labram_filter_checkpoint, labram_input_chans
    from p3audit.training.trainer import labram_layer_id, labram_param_groups
    st = {"model": {"logit_scale": 1, "student.cls_token": 2, "student.blocks.0.attn.relative_position_index": 3,
                    "student.blocks.0.norm1.weight": 4}}
    assert labram_filter_checkpoint(st) == {"cls_token": 2, "blocks.0.norm1.weight": 4}
    std = ["FP1", "FPZ", "C3", "CZ", "IZ"]
    assert labram_input_chans(["Fp1", "Cz", "Iz."], std) == [0, 1, 4, 5]
    with pytest.raises(RuntimeError):
        labram_input_chans(["XX1"], std)
    assert labram_layer_id("backbone.m.cls_token", 12) == 0
    assert labram_layer_id("backbone.m.patch_embed.conv1.weight", 12) == 0
    assert labram_layer_id("backbone.m.blocks.3.attn.qkv.weight", 12) == 4
    assert labram_layer_id("backbone.m.fc_norm.weight", 12) == 13
    assert labram_layer_id("head.weight", 12) == 13

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            inner = nn.Module()
            inner.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
            inner.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
            self.backbone = nn.Module(); self.backbone.m = inner
            self.head = nn.Linear(4, 2)
    ft = cfg["foundation"]["labram_finetune"]
    g = {tuple(x["names"]): x for x in labram_param_groups(M(), ft, 2)}
    assert g[("backbone.m.cls_token",)]["weight_decay"] == 0.0            # skip list
    assert g[("backbone.m.blocks.0.weight",)]["weight_decay"] == ft["weight_decay"]
    assert g[("backbone.m.blocks.0.bias",)]["weight_decay"] == 0.0        # bias
    assert abs(g[("backbone.m.blocks.0.weight",)]["lr_scale"] - ft["layer_decay"] ** 2) < 1e-12
    assert g[("head.weight",)]["lr_scale"] == 1.0
