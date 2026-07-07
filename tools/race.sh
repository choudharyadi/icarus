#!/usr/bin/env bash
# Headless qualifier run: launches Webots in batch mode, the bridge
# auto-spawns the pilot, and everything quits when the race finishes.
#
# Usage:
#   tools/race.sh                 # Qualifier 1
#   tools/race.sh 2               # Qualifier 2
#   tools/race.sh 1 mylog.log     # custom log file

set -euo pipefail
cd "$(dirname "$0")/.."

COURSE="${1:-1}"
LOG="${2:-/tmp/icarus_qualifier${COURSE}.log}"
WEBOTS="${WEBOTS_BIN:-/Applications/Webots.app/Contents/MacOS/webots}"

echo "Racing Qualifier ${COURSE} (log: ${LOG}) ..."
ICARUS_AUTOQUIT=1 ICARUS_MAX_SIM_T="${ICARUS_MAX_SIM_T:-300}" \
  "${WEBOTS}" --batch --stdout --stderr --mode=realtime \
  "worlds/Qualifier${COURSE}.wbt" > "${LOG}" 2>&1 || true

echo "--- result ---------------------------------------"
grep -E "split|Gates passed|FINAL TIME|COLLISION|Did not finish" "${LOG}" \
  | sed 's/^\[RACE\] //'
