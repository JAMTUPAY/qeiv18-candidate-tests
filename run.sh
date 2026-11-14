#!/usr/bin/env bash
# Usage on the droplet:  bash run.sh tests/<name>.py [arguments]
# Candidate identity check first; then the script; then the console and records are committed and pushed.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
CANDIDATE="${QEI_CANDIDATE_ROOT:-/root/neoamorfic/QEIv18.2-candidate}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export QEI_CANDIDATE_ROOT="$CANDIDATE"
git pull -q --rebase
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p records readings
LOG="records/console_$(basename "${1:-none}" .py)_${STAMP}.txt"
run() {
  echo "host $(hostname)   commit $(git rev-parse --short HEAD)   utc ${STAMP}   script $*"
  echo "--- the candidate's identity check"
  local id
  id="$(cd "$CANDIDATE" && python3 -B verify_candidate.py 2>&1)"
  echo "$id"
  if ! echo "$id" | grep -q '"state": "PASS"'; then
    echo "NOT RUN: the candidate's identity check did not pass"
    return 3
  fi
  nice -n 10 python3 -B "$@"
}
run "$@" 2>&1 | sed -u "\\|File \"${CANDIDATE}|{n;s|.*|    [a line of candidate source is withheld]|;}" > "$LOG"
CODE="${PIPESTATUS[0]}"
echo "exit code: ${CODE}" >> "$LOG"
git add records > /dev/null 2>&1
git commit -q -m "Run of ${1:-none}, ${STAMP} (exit ${CODE})" > /dev/null 2>&1
if git push -q > /dev/null 2>&1; then PUSHED="RECORD PUSHED"; else PUSHED="RECORD NOT PUSHED (it is saved on the droplet in records/)"; fi
tail -30 "$LOG"
echo "${PUSHED}"
