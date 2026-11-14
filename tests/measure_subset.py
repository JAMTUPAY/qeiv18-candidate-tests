#!/usr/bin/env python3
"""V182, candidate side: the candidate's readings on the eight-unit CHB-MIT subset. Candidate, not QEIv18.

    bash run.sh tests/measure_subset.py

For every recording of the subset (qeiv18-esa-adb records/v182/subset_2026-11-14.json), every step of the
certified project's step grid (256 rows from row 255 on, plus a final step ending at the last row) gets one
call  measure([six channels], end=last row of the step)  with entropy_window 256, coherence_window 256 and
segment_length 32: trailing windows only, the step's own 256 rows. Stored per recording, in readings/subset/
(not committed):
  ent        (steps, 6, 4)  S, dS, kappa, Phi as the candidate reports them;  ent_ok  their status == "ok"
  coh_band   (steps, 15)    per channel pair, the mean of coherence_white_null_adjusted_v2 over the valid bins
                            from 2 to 26 Hz (bins 1 to 13 at 64 Hz; the signals are low-passed at 28 Hz);
                            NaN when the pair's status is not "ok" or no bin in that band is valid
  pair_ok    (steps, 15)    the pair's status == "ok"
Pairs in the order the candidate returns them ("i:j", i < j, channels in the certified project's order).
Fixed before any network sees a candidate reading. No seizure mask is read. Writes
records/measure_subset_<date>.json (counts, shares of valid values, timings, the sha256 of every stored file).
"""
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
CANDIDATE = os.environ.get("QEI_CANDIDATE_ROOT", "/root/neoamorfic/QEIv18.2-candidate")
CERT = os.environ.get("QEI_CERTIFIED_REPO", "/root/neoamorfic/satellite_physics")
sys.path.insert(0, CANDIDATE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                    # noqa: E402
from qeiv182 import Gateway, Config  # noqa: E402
from compat_leak import ROLES, CONFIG, array_sha256   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STEP, FIRST = 256, 255
HZ = 64.0
BAND_HZ = (2.0, 26.0)
ENTROPY = ("S", "dS", "kappa", "Phi")
OUT_DIR = os.path.join(ROOT, "readings", "subset")


def step_starts(n):
    """The certified project's step grid (eeg_features.step_starts), restated."""
    starts = np.arange(FIRST, n - STEP + 1, STEP, dtype=np.int64)
    if starts.size and starts[-1] + STEP < n:
        starts = np.append(starts, np.int64(n - STEP))
    return starts


def band_mean(pair):
    if pair["status"] != "ok":
        return np.nan
    f = np.asarray(pair["frequencies"], dtype=np.float64) * HZ
    adj = np.asarray(pair["adjusted"], dtype=np.float64)
    sel = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1]) & np.asarray(pair["valid_bins"], dtype=bool) & np.isfinite(adj)
    return float(adj[sel].mean()) if sel.any() else np.nan


def main():
    t0 = time.time()
    with open(os.path.join(CERT, "records", "v182", "subset_2026-11-14.json")) as f:
        subset = json.load(f)
    with open(os.path.join(CERT, subset["eeg_plan_record"])) as f:
        plan = json.load(f)
    g = Gateway(Config(**CONFIG))
    os.makedirs(OUT_DIR, exist_ok=True)
    files, pair_names, totals, call_sec = {}, None, {"steps": 0, "pair_ok": 0, "pairs": 0, "ent_ok": 0, "ents": 0}, []
    print("V182 candidate readings on the subset: %d recordings; config %s; band %s Hz" % (len(subset["recordings"]), CONFIG, BAND_HZ), flush=True)
    for i, ds in enumerate(subset["recordings"]):
        rec = plan["datasets"][ds]
        n = int(rec["rows"])
        xs = []
        for role in ROLES:
            x = np.load(os.path.join(CERT, "data", "eeg", "arrays", ds, role + ".npy"))
            if array_sha256(x) != rec["arrays_sha256"][role]:
                raise ValueError("%s/%s: the raw array is not the one on record" % (ds, role))
            xs.append(np.asarray(x, dtype=np.float64))
        starts = step_starts(n)
        ent = np.full((starts.size, len(ROLES), len(ENTROPY)), np.nan)
        ent_ok = np.zeros((starts.size, len(ROLES), len(ENTROPY)), dtype=bool)
        coh, pair_ok = None, None
        t = time.perf_counter()
        for j, s in enumerate(starts):
            m = g.measure(xs, end=int(s + STEP - 1))
            for c, ch in enumerate(m["channels"]):
                for k, name in enumerate(ENTROPY):
                    ent[j, c, k] = float(ch["values"][name])
                    ent_ok[j, c, k] = ch["status"][name] == "ok"
            if coh is None:
                pair_names = sorted(m["pairs"], key=lambda p: tuple(int(v) for v in p.split(":")))
                coh = np.full((starts.size, len(pair_names)), np.nan)
                pair_ok = np.zeros((starts.size, len(pair_names)), dtype=bool)
            for q, p in enumerate(pair_names):
                coh[j, q] = band_mean(m["pairs"][p])
                pair_ok[j, q] = m["pairs"][p]["status"] == "ok"
        call_sec.append((time.perf_counter() - t) / max(1, starts.size))
        path = os.path.join(OUT_DIR, ds.replace("/", "__") + ".npz")
        np.savez(path, starts=starts, ent=ent, ent_ok=ent_ok, coh_band=coh, pair_ok=pair_ok, pairs=np.array(pair_names))
        with open(path, "rb") as f:
            files[ds] = hashlib.sha256(f.read()).hexdigest()
        totals["steps"] += int(starts.size)
        totals["pair_ok"] += int(pair_ok.sum())
        totals["pairs"] += int(pair_ok.size)
        totals["ent_ok"] += int(ent_ok.sum())
        totals["ents"] += int(ent_ok.size)
        print("  %d/%d %s: %d steps, valid pairs %.1f%%, %.1f ms per step (%.0f s)" % (
            i + 1, len(subset["recordings"]), ds, starts.size, 100.0 * pair_ok.mean(), 1000 * call_sec[-1], time.time() - t0), flush=True)
    rec = {"record": "V182 candidate readings on the subset", "label": "candidate, not QEIv18",
           "plan": "qeiv18-esa-adb: records/dev_V182_candidate_plan_2026-11-14.md", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "code_commit": os.popen("git -C '%s' rev-parse HEAD" % ROOT).read().strip(), "candidate_root": CANDIDATE, "numpy": np.__version__,
           "config": CONFIG, "band_hz": BAND_HZ, "step": STEP, "first_row": FIRST, "roles": list(ROLES), "pairs": pair_names,
           "files": dict((ds, {"path": "readings/subset/%s.npz" % ds.replace("/", "__"), "sha256": h}) for ds, h in files.items()),
           "totals": totals, "valid_pair_share": round(totals["pair_ok"] / float(max(1, totals["pairs"])), 5),
           "valid_entropy_share": round(totals["ent_ok"] / float(max(1, totals["ents"])), 5),
           "ms_per_step_median": round(1000 * float(np.median(call_sec)), 3), "seconds": round(time.time() - t0, 1),
           "data_in_this_file": "counts, shares, timings, file hashes; no EEG value, no seizure position"}
    out = os.path.join(ROOT, "records", "measure_subset_%s.json" % time.strftime("%Y-%m-%d", time.gmtime()))
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print("record: %s" % os.path.relpath(out, ROOT))
    print("MEASURE SUBSET DONE")


if __name__ == "__main__":
    main()
