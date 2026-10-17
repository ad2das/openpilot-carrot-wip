#!/usr/bin/env python3
"""CP-SD smoke: 2 candidate sets x 6 scenarios, printing every output column.

Prints nod/gap/w-onset for each set and the H0 identity check; asserts finite
metrics and that the stop-phase weight actually activates somewhere.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
  try:
    import w84_win_harness  # noqa: F401
  except Exception as exc:
    print(f"sd_smoke: harness preload failed: {exc!r}", file=sys.stderr)

from openpilot.selfdrive.test.longitudinal_maneuvers import test_butter_nod_plant as H


def main():
  follow = next(m for m in H.create_maneuvers({"e2e": False, "force_decel": False})
                if m.title.startswith("steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2"))

  # Default: one briefed set (V_LO 8.0 / DELAY_LO 0.65 / A_REQ_MAX 1.0 / M 0.3).
  # Override any axis with SD_SMOKE_* env vars.
  p = {
    "DELAY_LO": float(os.environ.get("SD_SMOKE_DELAY_LO", 0.65)),
    "V_LO": float(os.environ.get("SD_SMOKE_V_LO", 8.0)),
    "A_REQ_MAX": float(os.environ.get("SD_SMOKE_A_REQ_MAX", 1.0)),
    "MARGIN_SD": float(os.environ.get("SD_SMOKE_MARGIN_SD", 0.3)),
    "APPLY_TO_LATCH": os.environ.get("SD_SMOKE_APPLY_TO_LATCH", "0") == "1",
  }
  sets = [p]

  print("=== SD H0 (disabled == landed) ===")
  for row in H.sd_h0_check():
    print(f"H0 {row['name']}: identical={row['identical']}")
    assert row["identical"], f"H0 identity failed: {row['name']}"

  keys = ("delay_eff_max", "w_max", "w_onset_t", "w_onset_v", "a_stop", "nod_swing",
          "peak_decel", "peak_jerk_plant", "final_gap_m", "min_gap_m", "min_ttc",
          "post_stop_travel", "start_delay_s", "fcw_steps", "collision")
  print("=== SD SMOKE ===")
  any_active = False
  for p in sets:
    specs = H._sd_build_specs("real_sg")
    b1_cache = H._sd_b1_cache(specs)
    analysis, hard, comfort, passes = H._sd_eval_params(
      p, plant_mode="real_sg", b1_cache=b1_cache, specs=specs)
    by_name = {n: (b1, sd, cls) for n, b1, sd, cls in analysis["pairs"]}
    print(f"--- set {p} passes={passes} s1a={analysis['s1a_ok']} "
          f"s1a_mean={analysis['s1a_mean']:.4f} s1b={analysis['s1b_ok']} "
          f"stop_worse={analysis['stop_worse']:.4f} hard={len(hard)} comfort={len(comfort)}")
    sample = ["stopped lead 40 km/h gap 60 m", "b1", "replica 445@7.27",
              "replica 445@8.52", "replica 44c@6.21", follow.title]
    for nm in sample:
      if nm not in by_name:
        continue
      b1, sdrow, cls = by_name[nm]
      if sdrow.get("w_max", 0.0) > 0.0:
        any_active = True
      print(f"SD {nm[:46]:48} cls={cls} " + " ".join(f"{k}={sdrow.get(k)}" for k in keys))
      for k in keys:
        v = sdrow.get(k)
        if v is None:
          continue
        assert math.isfinite(v), f"{nm}:{k}={v} not finite"
      # Follow scenarios must keep the stop-phase weight at 0.
      if cls == "follow":
        assert sdrow.get("w_max", 0.0) == 0.0, f"follow {nm} weight activated: {sdrow.get('w_max')}"
    # Per-set nod table (B1 vs set).
    nr = analysis["nod_report"]
    print("SD NOD " + " ".join(
      f"{k}:[{nr[k]['b1']}->{nr[k]['sd']}]" for k in nr if k.startswith("replica") or k == "b1"))
    print(f"SD NOD a_group:[{nr['a_group']['b1']}->{nr['a_group']['sd']}]")
    print(f"SD {len(hard)} hard / {len(comfort)} comfort: "
          f"{(hard + comfort)[:3]}")

  assert any_active, "stop-phase weight never activated in the smoke scenarios"
  print("SD-SMOKE-OK")


if __name__ == "__main__":
  main()
