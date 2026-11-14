#!/usr/bin/env python3
"""Read-only probe of the candidate's API (QEIv18.2.0-candidate.1): candidate, not QEIv18; part of
development test V182 in qeiv18-esa-adb. Runs in its own process with only the candidate folder on the path (its README: "Use an
isolated Python process"); the certified engine is not imported here.

    bash run.sh tests/probe.py

Constructed signals only (no EEG). Records the structure of every answer (names, types, shapes, the first
values of constructed-signal answers), the public names of the API, the configuration fields, the library
versions and the time per call. The candidate's source is not read. Writes records/probe_<date>.json.
"""
import dataclasses
import json
import os
import resource
import sys
import time

sys.dont_write_bytecode = True
CANDIDATE = os.environ.get("QEI_CANDIDATE_ROOT", "/root/neoamorfic/QEIv18.2-candidate")
sys.path.insert(0, CANDIDATE)
import numpy as np                    # noqa: E402
from qeiv182 import Gateway, Config  # noqa: E402  (the candidate's own gateway, as its README directs)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N = 2400


def signals():
    t = np.arange(N, dtype=float)
    v, lg = 0.123, np.empty(N)
    for i in range(N):
        v = 3.99 * v * (1.0 - v)
        lg[i] = v
    a = np.sin(2 * np.pi * t / 64.0) + 0.5 * (lg - 0.5)
    b = np.roll(a, 7) + 0.3 * np.cos(2 * np.pi * t / 23.0)
    w, lg2 = 0.377, np.empty(N)
    for i in range(N):
        w = 3.99 * w * (1.0 - w)
        lg2[i] = w
    return {"a_sine64_logistic": a, "b_a_delayed_plus_cos23": b, "c_logistic_other_start": lg2 - 0.5,
            "constant": np.ones(N), "step": np.where(t < 1200, 0.0, 1.0)}


def describe(o, depth=0):
    if depth > 6:
        return "..."
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {"<dataclass %s>" % type(o).__name__: describe(dataclasses.asdict(o), depth + 1)}
    if isinstance(o, dict):
        return dict((str(k), describe(v, depth + 1)) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        if len(o) and all(not isinstance(x, (dict, list, tuple, np.ndarray)) for x in o):
            return "%s len=%d first=%s" % (type(o).__name__, len(o), repr(list(o[:4]))[:160])
        return [describe(x, depth + 1) for x in o[:3]] + (["... %d items" % len(o)] if len(o) > 3 else [])
    if isinstance(o, np.ndarray):
        flat = o.ravel()
        return "ndarray shape=%s dtype=%s first=%s nan=%d" % (o.shape, o.dtype, np.array2string(flat[:4], precision=5),
                                                               int(np.isnan(flat).sum()) if o.dtype.kind == "f" else 0)
    if hasattr(o, "__dict__") and not isinstance(o, type):
        return {"<object %s>" % type(o).__name__: describe(dict((k, v) for k, v in vars(o).items() if not k.startswith("_")), depth + 1)}
    return "%s %s" % (type(o).__name__, repr(o)[:160])


def attempt(label, fn, out, timings, reps=1):
    try:
        t0 = time.perf_counter()
        for _ in range(reps):
            r = fn()
        timings[label] = round((time.perf_counter() - t0) / reps, 5)
        out[label] = describe(r)
        return r
    except Exception as e:
        out[label] = "RAISED %s: %s" % (type(e).__name__, str(e)[:300])
        return None


def main():
    import scipy
    s = signals()
    rec = {"record": "Read-only probe of the candidate's API", "plan": "qeiv18-esa-adb: records/dev_V182_candidate_plan_2026-11-14.md", "label": "candidate, not QEIv18",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "candidate_root": CANDIDATE,
           "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
           "api_gateway_public": sorted(n for n in dir(Gateway) if not n.startswith("_")),
           "api_config_public": sorted(n for n in dir(Config) if not n.startswith("_"))}
    try:
        rec["config_default"] = describe(Config())
    except Exception as e:
        rec["config_default"] = "RAISED %s: %s" % (type(e).__name__, str(e)[:300])
    answers, timings = {}, {}
    ours = dict(entropy_window=256, coherence_window=256, segment_length=32)
    g_default = attempt("Gateway(Config())", lambda: Gateway(Config()), answers, timings)
    g = attempt("Gateway(Config(entropy_window=256, coherence_window=256, segment_length=32))", lambda: Gateway(Config(**ours)), answers, timings)
    if g is not None:
        a, b, c = s["a_sine64_logistic"], s["b_a_delayed_plus_cos23"], s["c_logistic_other_start"]
        attempt("measure([a, b], end=1999)", lambda: g.measure([a, b], end=1999), answers, timings, reps=20)
        attempt("measure([a, c], end=1999)", lambda: g.measure([a, c], end=1999), answers, timings)
        attempt("measure([a, a], end=1999)", lambda: g.measure([a, a], end=1999), answers, timings)
        attempt("measure([a, b], end=100)", lambda: g.measure([a, b], end=100), answers, timings)
        attempt("measure([constant, b], end=1999)", lambda: g.measure([s["constant"], b], end=1999), answers, timings)
        attempt("measure([a, b, c], end=1999)", lambda: g.measure([a, b, c], end=1999), answers, timings)
        attempt("measure([a], end=1999)", lambda: g.measure([a], end=1999), answers, timings)
        attempt("entropy_segment(a[1744:2000])", lambda: g.entropy_segment(a[1744:2000]), answers, timings, reps=20)
        attempt("entropy_segment(constant[1744:2000])", lambda: g.entropy_segment(s["constant"][1744:2000]), answers, timings)
        attempt("legacy_run([a])", lambda: g.legacy_run([a]), answers, timings)
        for name in rec["api_gateway_public"]:
            if name not in ("measure", "entropy_segment", "legacy_run", "diagnose") and callable(getattr(g, name, None)):
                answers["other public callable: %s" % name] = "present (not called)"
        attempt("diagnose([a, b])", lambda: g.diagnose([a, b]), answers, timings)
    if g_default is not None:
        attempt("default config: measure([a, b], end=1999)", lambda: g_default.measure([s["a_sine64_logistic"], s["b_a_delayed_plus_cos23"]], end=1999), answers, timings)
    rec.update({"answers": answers, "seconds_per_call": timings,
                "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
                "data_in_this_file": "structure of answers on constructed signals; no EEG value, no engine source"})
    os.makedirs(os.path.join(ROOT, "records"), exist_ok=True)
    out = os.path.join(ROOT, "records", "probe_%s.json" % time.strftime("%Y-%m-%d", time.gmtime()))
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("python", "numpy", "scipy", "api_gateway_public", "seconds_per_call", "peak_memory_mb")}, indent=1))
    print("record: %s" % os.path.relpath(out, ROOT))
    print("CANDIDATE PROBE DONE")


if __name__ == "__main__":
    main()
