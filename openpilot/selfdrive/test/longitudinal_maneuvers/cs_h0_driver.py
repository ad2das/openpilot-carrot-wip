#!/usr/bin/env python3
"""CP-H0 diagnosis driver for the chauffeur-stop (CS) task.

Runs landed ButterStop (B1) on the 3 lead replicas and the 3 a-group stopped-lead
scenarios on PLANT_REAL, and prints the last 4 s before the stop at 50 ms
resolution. Pure diagnosis: no runtime code, no parameter changes.

Columns: t, v, a_plant, a_ego, stock_out, floor_raw, floor_applied, out, bound,
a_end, plan_stop_d, lead_avail, lc_state.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Preload the Windows desktop-test shims (params_pyx, pyray, fcntl, msgq, swaglog)
# when running on Windows; harmless no-op elsewhere.
if sys.platform == "win32":
  try:
    import w84_win_harness  # noqa: F401  (registers the substitutes in sys.modules)
  except Exception as exc:  # pragma: no cover
    print(f"cs_h0: harness preload failed: {exc!r}", file=sys.stderr)

from openpilot.selfdrive.test.longitudinal_maneuvers import test_butter_nod_plant as H


def _fmt(x, nd=4):
  if x is None:
    return "None"
  try:
    f = float(x)
  except (TypeError, ValueError):
    return str(x)
  if not np.isfinite(f):
    return "nan"
  return f"{f:.{nd}f}"


def _print_trace(name, trace):
  t = np.array([r["t"] for r in trace], dtype=float)
  v = np.array([r["v"] for r in trace], dtype=float)
  stand = np.flatnonzero(v < 0.05)
  t_stop = float(t[stand[0]]) if stand.size else float(t[-1])
  lo = t_stop - 4.0
  print(f"\n===== {name} =====  t_stop={t_stop:.3f}s  frames={len(trace)}")
  header = ("t       v      a_plant a_ego  stock   f_raw   f_app   out     "
            "bound  a_end  plan_d  lead_a  lc")
  print(header)
  last = None
  for r in trace:
    if r["t"] < lo or r["t"] > t_stop + 0.05:
      continue
    # 50 ms decimation: emit a row every 0.05 s (plus the first in window).
    emit = (last is None) or (r["t"] - last >= 0.05 - 1e-9) or (abs(r["t"] - t_stop) < 1e-9)
    if not emit:
      continue
    last = r["t"]
    print(f"{_fmt(r['t'],3):>7} {_fmt(r['v'],3):>6} {_fmt(r['plant_a'],4):>7} "
          f"{_fmt(r['a_ego'],3):>6} {_fmt(r['stock_out'],3):>7} {_fmt(r['floor_raw'],3):>7} "
          f"{_fmt(r['floor_applied'],3):>7} {_fmt(r['out'],3):>7} "
          f"{str(r['bound']):>6} {_fmt(r['a_end'],3):>6} {_fmt(r['plan_stop_d'],2):>7} "
          f"{_fmt(r['lead_avail'],2):>6} {int(r['long_state'])}")


def main():
  scenarios = [
    ("replica 445@7.27", "replica", "445@7.27"),
    ("replica 445@8.52", "replica", "445@8.52"),
    ("replica 44c@6.21", "replica", "44c@6.21"),
    ("stopped lead 20 km/h gap 30 m", "a", None),
    ("stopped lead 40 km/h gap 60 m", "a", None),
    ("stopped lead 60 km/h gap 90 m", "a", None),
  ]
  for name, group, key in scenarios:
    run = H._cs_scenario_runner(name, group, key)
    trace = run()
    _print_trace(name, trace)
    # one-line summary
    m = H.cs_trace_metrics(trace)
    print(f"SUMMARY {name}: stopped={m['stopped']} a_stop={_fmt(m['a_stop'],3)} "
          f"nod={_fmt(m['nod_swing'],3)} peak_decel={_fmt(m['peak_decel'],3)} "
          f"final_gap={_fmt(m['final_gap_m'],2)} min_gap={_fmt(m['min_gap_m'],2)} "
          f"floor_none={m['floor_none_steps']} floor_active={m['floor_active_steps']}")


if __name__ == "__main__":
  main()
