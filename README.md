# qeiv18-candidate-tests

Tests of **uncertified** engine candidates. Everything here is labelled "candidate, not QEIv18": nothing produced here is a QEIv18 reading, and nothing ships under the QEIv18 name.

Set up on 14 November 2026 by the owner's decision, after the QEIv18 compliance check correctly refused candidate-calling code in the certified repository (`JAMTUPAY/qeiv18-esa-adb`, `records/v182/compliance_refusal_2026-11-14.md`).

## Rules

- The candidate under test is `QEIv18.2.0-candidate.1`, on the droplet at `/root/neoamorfic/QEIv18.2-candidate`. It is used only through its own API (`from qeiv182 import Gateway, Config`), in its own process, as its README directs. Its source is not read, copied or summarised here.
- Every run goes through `run.sh`. Nothing runs unless the candidate's identity check (`verify_candidate.py`) prints `"state": "PASS"`. Console lines that would quote candidate source in an error trace are withheld before anything is saved.
- Deterministic runs only: `PYTHONHASHSEED=0` and single-threaded numerical libraries.
- No candidate file, no engine file, no password, key or token, and no EEG values are committed. Records hold structure, counts, hashes, differences, timings and scores. Measured arrays stay on the droplet under `readings/`, which is not committed.
- The certified engine, its folder, the finance products and the certified repository are never written to from here.

## Running

On the droplet:

    cd /root/neoamorfic/qeiv18-candidate-tests && git pull && bash run.sh tests/<name>.py
