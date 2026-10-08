#!/usr/bin/env python3
"""Guard that a device model tree is backed by QCOM-compiled tinygrad models.

CI can only compile tinygrad programs for the CPU backend, so a published
prebuilt tree must never let a comma 3/3X run those artifacts: on the device
every entry the driving and driver-monitoring models load must be QCOM.
Off-device (no /TICI marker) the guard always passes so source checkouts and
the publish pipeline itself stay usable.

Exit status 0 means the tree passes the guard; any parse error, missing entry
or non-QCOM value exits 1.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

MODELD_KEY = "openpilot.selfdrive.modeld.modeld"
DMONITORING_KEY = "openpilot.selfdrive.modeld.dmonitoringmodeld"


def check(path: str, tici_marker: str) -> int:
  if not os.path.exists(tici_marker):
    print(f"backend guard: no {tici_marker} marker; passing off-device")
    return 0

  try:
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    if not isinstance(data, dict):
      raise ValueError("top level is not an object")
    modeld = data.get(MODELD_KEY)
    dmonitoring = data.get(DMONITORING_KEY)
    if not isinstance(modeld, dict) or not isinstance(dmonitoring, dict):
      raise ValueError("missing modeld or dmonitoringmodeld entry")
    modeld_default = modeld.get("default")
    dmonitoring_default = dmonitoring.get("default")
    if not isinstance(modeld_default, dict) or not isinstance(dmonitoring_default, dict):
      raise ValueError("missing default backend map")
    entries = {
      "modeld WARP_DEV": modeld_default.get("WARP_DEV"),
      "modeld QUEUE_DEV": modeld_default.get("QUEUE_DEV"),
      "dmonitoringmodeld DEV": dmonitoring_default.get("DEV"),
    }
  except Exception as exc:  # noqa: BLE001 - any error is a guard failure
    print(f"backend guard failed: {path}: {exc}", file=sys.stderr)
    return 1

  failures = [f"{name}={value!r}" for name, value in entries.items() if value != "QCOM"]
  if failures:
    print("backend guard failed: " + ", ".join(failures), file=sys.stderr)
    return 1

  print("backend guard OK: modeld WARP_DEV/QUEUE_DEV and dmonitoringmodeld DEV are QCOM")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("path", help="path to models/tg_input_devices.json")
  parser.add_argument("--tici-marker", default="/TICI", help="device marker file (default: /TICI)")
  args = parser.parse_args()
  return check(args.path, args.tici_marker)


if __name__ == "__main__":
  sys.exit(main())
