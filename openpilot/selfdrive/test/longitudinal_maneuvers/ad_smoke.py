#!/usr/bin/env python3
"""AD smoke: 2 delays (0.25, 0.8) on 1 replica + 1 following maneuver.

Prints every metric; asserts finite and that action_t differs between delays.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
  try:
    import w84_win_harness  # noqa: F401
  except Exception as exc:
    print(f"ad_smoke: harness preload failed: {exc!r}", file=sys.stderr)

from openpilot.selfdrive.test.longitudinal_maneuvers import test_butter_nod_plant as H


def main():
  _d, stops = H.load_stops()
  key = "445@7.27"
  profile = H.RealStopReplicaProfile(H._cs_find_stop(stops, key))

  follow = next(m for m in H.create_maneuvers({"e2e": False, "force_decel": False})
                if m.title.startswith("steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2"))

  rows = []
  for delay in (0.25, 0.8):
    rep = H._ad_fixture_row(profile, "replica", "stop", delay, f"replica {key}",
                            fixed_grade=profile.grade)
    fol = H._ad_scenario_row(follow, "std", "follow", delay)
    rows.append((delay, rep, fol))

  print("=== AD SMOKE ===")
  keys = ("action_t", "a_stop", "nod_swing", "peak_decel", "peak_jerk_plant",
          "p95_jerk_plant", "final_gap_m", "min_gap_m", "min_ttc",
          "post_stop_travel", "start_delay_s", "follow_std", "cruise_over_kph")
  for delay, rep, fol in rows:
    for label, r in (("replica", rep), ("follow", fol)):
      print(f"delay={delay} {label}: " + " ".join(f"{k}={r.get(k)}" for k in keys))

  # action_t must differ between delays.
  at_025 = rows[0][1]["action_t"]
  at_08 = rows[1][1]["action_t"]
  print(f"action_t@0.25={at_025} action_t@0.8={at_08} differs={at_025 != at_08}")
  assert at_025 is not None and at_08 is not None, "action_t None"
  assert abs(at_08 - at_025) > 0.4, f"action_t did not change enough: {at_025} {at_08}"

  # Every metric must be finite (except min_ttc may be None when no approach).
  for _delay, rep, fol in rows:
    for label, r in (("replica", rep), ("follow", fol)):
      for k in keys:
        v = r.get(k)
        if v is None:
          continue
        assert math.isfinite(v), f"non-finite {label} {k}={v}"
  print("AD-SMOKE-OK")


if __name__ == "__main__":
  main()
