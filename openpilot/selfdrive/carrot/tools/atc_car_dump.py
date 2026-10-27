#!/usr/bin/env python3
"""Parked lever-capture markers for the Palisade ATC turn-signal investigation.

openpilot keeps running, so its rlog records every CAN frame on every bus with
logMonoTime. This script only prints timed prompts and writes a cloudlog event
(`atc_dump_prompt`, recorded as logMessage in the same rlog) at each step, so
the PC can slice the rlog by step. Nothing is sent on the CAN bus.

Run parked, ignition on, openpilot running:
    cd /data/openpilot && ./openpilot/selfdrive/carrot/tools/atc_car_dump.py

Keep the ignition on for about two minutes afterwards so the segment closes and
log_sync uploads it.
"""
import os
import sys
import time

BASEDIR = os.environ.get("BASEDIR", "/data/openpilot")
# Same order as launch_chffrplus.sh: pydeps first (pyserial etc.), then the repo.
for path in (BASEDIR, os.path.join(BASEDIR, "pydeps")):
  if path not in sys.path:
    sys.path.insert(0, path)

from openpilot.common.swaglog import cloudlog  # noqa: E402

# (seconds from start, step id, prompt shown to the driver)
SCHEDULE = [
  (0.0, "idle_start", "idle - do nothing"),
  (5.0, "left_on", "LEFT lever ON (hold it)"),
  (12.0, "left_off", "lever OFF"),
  (17.0, "right_on", "RIGHT lever ON (hold it)"),
  (24.0, "right_off", "lever OFF"),
  (29.0, "hazard_on", "HAZARD ON"),
  (36.0, "hazard_off", "HAZARD OFF"),
  (41.0, "done", "done"),
]


def main():
  print("ATC lever capture. Car parked, ignition on, openpilot running.")
  start = time.monotonic()
  for at, step, prompt in SCHEDULE:
    delay = start + at - time.monotonic()
    if delay > 0:
      time.sleep(delay)
    cloudlog.event("atc_dump_prompt", step=step, mono_ns=time.monotonic_ns())
    print(f"  [{at:4.1f}s] >>> {prompt}")
  print("\nDone. Keep the ignition on ~2 min so the rlog segment closes and uploads.")


if __name__ == "__main__":
  main()
