#!/usr/bin/env python3
"""V182, candidate side of questions 1 and 2 (compatibility and leakage) and the practical-behaviour timings.
Candidate, not QEIv18. The certified engine is not imported here; its side of the comparison runs in the
certified repository on the same inputs and reads this script's stored numbers.

    bash run.sh tests/compat_leak.py

Bases: the six base signals of the certified leakage test (qeiv18-esa-adb records/v182/leak_v1813_2026-11-14.json):
four CHB-MIT tiles of 2,400 rows and two constructed signals.
  A. legacy_run on each base: one sha256 per output (to be compared with the certified run on the same input).
  B. measure([x], end=e) at rows 300, 800, 1400, 2000 and 2399 with entropy_window 256: S, dS, kappa, Phi and
     their status, stored for comparison with the certified series at the same rows.
  C. Leakage: two-channel measure on each base (EEG: the tile's first two channels; constructed: a and b) at
     ends cut-1, cut-128 and cut-255 for cuts 800, 1400 and 2000, with the samples from the cut on replaced by
     their reverse x 1.7 plus 0.3 standard deviations; every value and every coherence array must be
     bit-identical.
  D. Timings and peak memory: measure on six channels at 200 ends; legacy_run per call.
Writes readings/compat/points.npz (not committed) and records/compat_leak_<date>.json (hashes, pass/fail,
differences, timings; no EEG value).
"""
import hashlib
import json
import os
import resource
import sys
import time

sys.dont_write_bytecode = True
CANDIDATE = os.environ.get("QEI_CANDIDATE_ROOT", "/root/neoamorfic/QEIv18.2-candidate")
CERT = os.environ.get("QEI_CERTIFIED_REPO", "/root/neoamorfic/satellite_physics")
sys.path.insert(0, CANDIDATE)
import numpy as np                    # noqa: E402
from qeiv182 import Gateway, Config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEGMENT = 2400
CUTS = (800, 1400, 2000)
POINTS = (300, 800, 1400, 2000, 2399)
ROLES = ("left_front_temporal", "left_back_temporal", "right_front_temporal", "right_back_temporal",
         "left_central", "right_central")
CONFIG = dict(entropy_window=256, coherence_window=256, segment_length=32)
ENTROPY = ("S", "dS", "kappa", "Phi")
PAIR_ARRAYS = ("raw", "adjusted", "phase", "jackknife_spread", "valid_bins")
# The outputs the certified project stores from run (seven series, four tile values, MSE); the same list on both sides.
LEGACY_KEYS = ("S", "dS", "kappa", "coherence", "resonance", "Phi", "decision",
               "lyapunov", "lyapunov_S", "attractor_radius", "attractor_curvature", "MSE")


def array_sha256(arr):
    """The certified repository's fingerprint of an array (s1_measure.array_sha256), restated."""
    h = hashlib.sha256()
    for i in range(0, arr.shape[0], 1 << 22):
        h.update(np.ascontiguousarray(arr[i:i + (1 << 22)]).tobytes())
    return h.hexdigest()


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


def constructed():
    t = np.arange(SEGMENT, dtype=float)
    v, lg = 0.123, np.empty(SEGMENT)
    for i in range(SEGMENT):
        v = 3.99 * v * (1.0 - v)
        lg[i] = v
    a = np.sin(2 * np.pi * t / 64.0) + 0.5 * (lg - 0.5)
    step = np.where(t < 1200, 0.0, 1.0) + 0.2 * np.sin(2 * np.pi * t / 23.0)
    b = np.roll(a, 7) + 0.3 * np.cos(2 * np.pi * t / 23.0)
    return {"constructed/sine64_logistic": [a, b], "constructed/step_sine23": [step, a]}


def eeg_base(name, plan):
    """chbmit/<case>/<stem>/<role>@<start> -> all six channels of that tile, each checked against the plan record."""
    body, start = name.split("@")
    _, case, stem, _ = body.split("/")
    ds = "%s/%s" % (case, stem)
    rec = plan["datasets"][ds]
    out = []
    for role in ROLES:
        x = np.load(os.path.join(CERT, "data", "eeg", "arrays", ds, role + ".npy"), mmap_mode="r")
        if array_sha256(np.asarray(x)) != rec["arrays_sha256"][role]:
            raise ValueError("%s/%s: the raw array is not the one on record" % (ds, role))
        s = int(start)
        out.append(np.array(x[s:s + SEGMENT], dtype=np.float64))
    return out


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        a, b = a.astype(np.float64), b.astype(np.float64)
        return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
    return bool(np.array_equal(a, b))


def compare_measures(m1, m2):
    """Every channel value and status, every pair array and status: identical?"""
    diffs = []
    for i, (c1, c2) in enumerate(zip(m1["channels"], m2["channels"])):
        for k in c1["values"]:
            if not same(c1["values"][k], c2["values"][k]) or c1["status"].get(k) != c2["status"].get(k):
                diffs.append("channel %d %s" % (i, k))
    for p in m1["pairs"]:
        a, b = m1["pairs"][p], m2["pairs"][p]
        if a["status"] != b["status"]:
            diffs.append("pair %s status" % p)
        for k in PAIR_ARRAYS:
            if not same(a[k], b[k]):
                diffs.append("pair %s %s" % (p, k))
    return diffs


def main():
    import scipy
    t0 = time.time()
    with open(os.path.join(CERT, "records", "v182", "leak_v1813_2026-11-14.json")) as f:
        leak = json.load(f)
    with open(os.path.join(CERT, leak["eeg_plan_record"])) as f:
        plan = json.load(f)
    bases = {}
    for name in leak["bases"]:
        bases[name] = eeg_base(name, plan) if name.startswith("chbmit/") else constructed()[name]
    g = Gateway(Config(**CONFIG))
    print("V182 candidate compatibility and leakage: %d bases; config %s; numpy %s, scipy %s" % (len(bases), CONFIG, np.__version__, scipy.__version__), flush=True)

    # A. legacy_run hashes
    legacy, legacy_sec = {}, []
    for name, chans in bases.items():
        t = time.perf_counter()
        out = g.legacy_run([chans[0]])
        legacy_sec.append(time.perf_counter() - t)
        legacy[name] = dict((k, sha(np.asarray(out[k])[0] if np.asarray(out[k]).ndim >= 1 and np.asarray(out[k]).shape[0] == 1 else out[k]))
                            for k in LEGACY_KEYS if k in out)
    print("  A. legacy_run on %d bases (%.1f s per call)" % (len(bases), float(np.median(legacy_sec))), flush=True)

    # B. entropy values at fixed rows
    pts = {}
    for name, chans in bases.items():
        for e in POINTS:
            m = g.measure([chans[0]], end=e)
            ch = m["channels"][0]
            pts["%s|%d" % (name, e)] = ([float(ch["values"][k]) for k in ENTROPY], [str(ch["status"][k]) for k in ENTROPY])
    os.makedirs(os.path.join(ROOT, "readings", "compat"), exist_ok=True)
    keys = sorted(pts)
    vals = np.array([pts[k][0] for k in keys], dtype=np.float64)
    status = np.array([pts[k][1] for k in keys])
    pts_path = os.path.join(ROOT, "readings", "compat", "points.npz")
    np.savez(pts_path, keys=np.array(keys), values=vals, status=status, names=np.array(ENTROPY))
    with open(pts_path, "rb") as f:
        pts_sha = hashlib.sha256(f.read()).hexdigest()
    print("  B. entropy values at %d points stored" % len(keys), flush=True)

    # C. leakage
    leak_rows, n_cmp = [], 0
    for name, chans in bases.items():
        x, y = chans[0], chans[1]
        for cut in CUTS:
            xa, ya = x.copy(), y.copy()
            xa[cut:] = x[cut:][::-1] * 1.7 + 0.3 * (float(np.std(x)) or 1.0)
            ya[cut:] = y[cut:][::-1] * 1.7 + 0.3 * (float(np.std(y)) or 1.0)
            for e in (cut - 1, cut - 128, cut - 255):
                d = compare_measures(g.measure([x, y], end=e), g.measure([xa, ya], end=e))
                n_cmp += 1
                if d:
                    leak_rows.append({"base": name, "cut": cut, "end": e, "differs": d[:20]})
    print("  C. leakage: %d comparisons, %d with any difference" % (n_cmp, len(leak_rows)), flush=True)

    # D. timings on six channels
    chans6 = bases[[n for n in bases if n.startswith("chbmit/")][0]]
    ends = np.linspace(300, SEGMENT - 1, 200).astype(int)
    t = time.perf_counter()
    for e in ends:
        g.measure(chans6, end=int(e))
    per6 = (time.perf_counter() - t) / len(ends)
    print("  D. measure on six channels: %.2f ms per call" % (1000 * per6), flush=True)

    rec = {"record": "V182 candidate side: compatibility, leakage, timings", "label": "candidate, not QEIv18",
           "plan": "qeiv18-esa-adb: records/dev_V182_candidate_plan_2026-11-14.md", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "code_commit": os.popen("git -C '%s' rev-parse HEAD" % ROOT).read().strip(), "candidate_root": CANDIDATE,
           "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "config": CONFIG,
           "bases": list(bases), "legacy_run_sha256": legacy, "legacy_run_seconds_median": round(float(np.median(legacy_sec)), 3),
           "points": {"file": "readings/compat/points.npz", "sha256": pts_sha, "rows": list(POINTS), "outputs": list(ENTROPY)},
           "leakage": {"comparisons": n_cmp, "with_difference": len(leak_rows), "details": leak_rows, "cuts": list(CUTS),
                       "ends": "cut-1, cut-128, cut-255", "alteration": "samples from the cut on replaced by their reverse x 1.7 plus 0.3 standard deviations"},
           "measure_six_channels_ms": round(1000 * per6, 3), "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
           "seconds": round(time.time() - t0, 1),
           "data_in_this_file": "hashes, pass/fail, timings; no EEG value; the entropy values are in readings/ on the droplet, not committed"}
    os.makedirs(os.path.join(ROOT, "records"), exist_ok=True)
    out = os.path.join(ROOT, "records", "compat_leak_%s.json" % time.strftime("%Y-%m-%d", time.gmtime()))
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("record: %s" % os.path.relpath(out, ROOT))
    print("COMPAT LEAK DONE")


if __name__ == "__main__":
    main()
