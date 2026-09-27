"""Structural check of confirmatory unit files (PILOT_LOG 17.10).  Read-only; lives outside
p3audit/ so the frozen code hash is unaffected.  Checks fields, subject coverage, counts and
non-finite values -- never interprets results."""
import glob, json, math, os, sys
sys.path.insert(0, os.getcwd())
from collections import Counter
from p3audit.config import load_config

cfg = load_config()
HASH = cfg.hash()
split = json.load(open("results/split.json"))
cache = {int(d[4:]) for d in os.listdir("cache") if os.path.exists(f"cache/{d}/meta.json")}
TEST = sorted(str(s) for s in split["test"] if s in cache)
UNITS = set(open("results/unit_list.txt", encoding="utf-8").read().split())
ER_P, IG_P = cfg["erasure"]["p_levels"], cfg["ig"]["p_levels"]
AMPS = cfg["design"]["h1_test_amps"]; BANDS = list(cfg["ig"]["bands"])
# band_masks() appends "rest" for bins no band covers (75 Hz .. Nyquist 100 Hz at 200 Hz) -- by design
if max(hi for lo, hi in cfg["ig"]["bands"].values()) < cfg["streams"]["model"]["sfreq"] / 2: BANDS.append("rest")
HEOG = {f"{c:g}" for c in cfg["design"]["regressor_contamination"]} | {"ols"}
NW = cfg["ig"]["windows_per_unit"]; MAXEP = cfg["training"]["max_epochs"]

def nonfinite(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from nonfinite(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from nonfinite(v, f"{path}[{i}]")
    elif isinstance(x, float) and not math.isfinite(x):
        yield path

def cnt_ok(c, n=4):
    return set(c) == set(TEST) and all(len(v) == n and all(isinstance(t, int) and t >= 0 for t in v) for v in c.values())

def c3_ok(c):
    return set(c) == {"all", "half0", "half1"} and cnt_ok(c["all"])

problems, nan_paths, n = [], Counter(), 0
for f in sorted(glob.glob("results/units/*.json")):
    n += 1; name = os.path.basename(f)[:-5]; E = lambda m: problems.append(f"{name}: {m}")
    try: d = json.load(open(f, encoding="utf-8"))
    except Exception as e: E(f"unreadable {e}"); continue
    p = d["cell"]["p"]
    if d.get("uid") != name: E("uid != filename")
    if name not in UNITS: E("uid not in unit_list")
    if d.get("model") != name.split("__")[0] or f"seed{d.get('seed')}" != name.split("__")[-1]: E("model/seed mismatch")
    if d.get("cid") != name.split("__")[1]: E(f"cid {d.get('cid')} mismatch")
    if d.get("config_hash") != HASH: E(f"config_hash {d.get('config_hash')}")
    if d.get("device") != "cuda" or d.get("cuda_available") is not True: E(f"device {d.get('device')}")
    if d.get("standin") is not False: E("standin")
    if d.get("n_subjects") != {"train": 37, "val": 7, "test": 20}: E(f"n_subjects {d.get('n_subjects')}")
    if sorted(d.get("overlap", {})) != TEST: E("overlap subjects")
    tr = d["training"]
    if not (1 <= tr["epochs"] <= MAXEP and len(tr["history"]) == tr["epochs"]): E(f"epochs {tr['epochs']}/{len(tr['history'])}")
    L = len(d["test_half"])
    if set(d["z_test"]) != set(d["truth_counts"]) or any(len(v) != L for v in d["z_test"].values()): E("z_test/test_half length")
    if not all(c3_ok(v) for v in d["truth_counts"].values()): E("truth_counts structure")
    want_amps = {f"{a:g}" for a in AMPS} | {f"{d['cell']['amp']:g}"}
    if set(d["probe"]) != want_amps: E(f"probe amps {sorted(d['probe'])}")
    for a, r in d["probe"].items():
        if not cnt_ok(r["counts"]): E(f"probe {a} counts")
        if a in ("1", f"{d['cell']['amp']:g}"):
            if "within_subject" not in r or set(r["within_subject"]["per_subject"]) != set(TEST): E(f"probe {a} within_subject")
            if "pca" not in r: E(f"probe {a} pca missing")
    has = {k for k in ("erasure", "sri", "heog", "ig") if k in d}
    want = set() if p is None else {"sri", "heog"} | ({"erasure"} if p in ER_P else set()) | ({"ig"} if p in IG_P else set())
    if has != want: E(f"blocks {sorted(has)} expected {sorted(want)}")
    if "erasure" in d:
        e = d["erasure"]
        if not (c3_ok(e["counts_erased"]) and c3_ok(e["counts_head"])): E("erasure counts")
        if e["unidentifiable"] != (p >= 1.0): E("erasure unidentifiable flag")
    if "sri" in d and not all(c3_ok(v) for v in d["sri"]["counts"].values()): E("sri counts")
    if "heog" in d:
        if set(d["heog"]) != HEOG: E(f"heog keys {sorted(d['heog'])}")
        if not all(c3_ok(v["counts"]) for v in d["heog"].values()): E("heog counts")
    if "ig" in d:
        if set(d["ig"]) != {"band_removed", "zero"}: E(f"ig baselines {sorted(d['ig'])}")
        for bl, s in d["ig"].items():
            if s["bands"] != BANDS: E(f"ig {bl} bands {s['bands']}")
            mr = s["mean_relevance"]
            if len(mr) != 64 or any(len(r) != len(BANDS) for r in mr): E(f"ig {bl} mean_relevance shape")
            a = s["sums"]["all"]
            if set(a) != set(TEST): E(f"ig {bl} subjects {len(a)}")
            if sum(v[3] for v in a.values()) != NW: E(f"ig {bl} windows {sum(v[3] for v in a.values())}")
            if sum(v[3] for h in ("half0", "half1") for v in s["sums"][h].values()) != NW: E(f"ig {bl} half windows")
    for path in nonfinite(d):
        if path == ".erasure.angle_deg" and p >= 1.0:   # by design: no residual-concept variance at p = 1
            nan_paths["(expected) .erasure.angle_deg at p = 1"] += 1; continue
        key = path.split("[")[0]
        for s in TEST: key = key.replace(f".{s}.", ".<subj>.").replace(f".{s}", ".<subj>")
        nan_paths[key] += 1

print(f"units checked: {n} / {len(UNITS)}; config_hash {HASH}; test subjects {len(TEST)}")
print("problems:", len(problems))
for m in problems: print("  ", m)
print("non-finite values (path pattern: occurrences):", "none" if not nan_paths else "")
for k, v in sorted(nan_paths.items()): print(f"   {k}: {v}")
