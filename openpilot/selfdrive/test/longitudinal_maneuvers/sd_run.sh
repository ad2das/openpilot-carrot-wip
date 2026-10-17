#!/bin/bash
# CP-SD 240-set sweep launcher (Linux pod / fresh checkout).
# Runs the stop-phase-delay sweep standalone; H0 (identity over every
# scenario+variant, stop delay disabled vs landed) runs first and aborts the
# job on any mismatch.
#
# Usage:
#   openpilot/selfdrive/test/longitudinal_maneuvers/sd_run.sh [nproc]
#
# Env:
#   SD_OUT    output JSON path (default ./stop_delay.json)
#   SD_PLANT  plant mode for OpenpilotPrefix rows (default real_sg)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
export SD_OUT="${SD_OUT:-$REPO_ROOT/stop_delay.json}"
export SD_PLANT="${SD_PLANT:-${BUTTER_SD_PLANT:-real_sg}}"
export BUTTER_SD_PLANT="$SD_PLANT"
NPROC="${1:-8}"
echo "SD_OUT=$SD_OUT SD_PLANT=$SD_PLANT nproc=$NPROC"
exec python openpilot/selfdrive/test/longitudinal_maneuvers/sd_sweep_driver.py "$NPROC"
