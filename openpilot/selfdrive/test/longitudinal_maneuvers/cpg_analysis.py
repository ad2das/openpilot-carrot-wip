"""STOP-NOD CP-G analysis driver (run inside the WSL openpilot venv).

Task 1: trade-off table for the top-5 sets (from butter_bse.json) on 7 scenarios.
Task 2: jerk-spike forensics for the best set on 3 runs.
Read-only analysis; no parameter changes, no new sweep.
"""
import json
import os
import numpy as np

from openpilot.common.prefix import OpenpilotPrefix
import openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_nod_plant as H

TOP5_JSON = "/mnt/c/Users/Administrator/AppData/Local/Temp/opencode/butter_bse.json"
OUT = "/mnt/c/Users/Administrator/AppData/Local/Temp/opencode/cpg.json"
BEST = {"K_E": 0.5, "L_E": 0.0, "MARGIN_E": 0.0, "EXTRA_MAX": 0.2, "V_E": 5.0}


def _fmt(x, w=8, p=3):
  return "n/a".rjust(w) if x is None else f"{x:{w}.{p}f}"


def _load_top5():
  with open(TOP5_JSON, encoding="utf-8") as fh:
    d = json.load(fh)
  return [t["params"] for t in d["top5"]]


def task1(sets):
  rows = []
  for si, p in enumerate(sets):
    for name, group, key in H._CPG_SCENARIOS:
      b1, bse = H.cpg_run_scenario(name, group, key, p, p)
      m1 = H._trace_metrics(b1)
      m2 = H._trace_metrics(bse)
      rows.append({"set": si, "params": p, "scenario": name,
                   "b1": m1, "bse": m2})
  return rows


def _print_table(rows):
  print("NOD-CPG-TABLE")
  print(f"{'set':>3} {'scenario':<48} {'who':<4} {'a_stop':>8} {'nod':>8} "
        f"{'decel':>8} {'jerk':>8} {'gap':>8}")
  for r in rows:
    for who, m in (("B1", r["b1"]), ("BSE", r["bse"])):
      print(f"{r['set']:>3} {r['scenario']:<48} {who:<4} {_fmt(m['a_stop'])} "
            f"{_fmt(m['nod_swing'])} {_fmt(m['peak_decel'])} {_fmt(m['peak_jerk'])} "
            f"{_fmt(m['final_gap_m'])}")


def task2(best):
  """Jerk-spike forensics on 3 runs for the best set."""
  runs = [
    ("replica 44c@6.21", "replica", "44c@6.21"),
    ("b1 [dn3]", "fixture_b1_dn3", None),
    ("b1", "fixture_b1", None),
  ]
  reports = []
  for name, kind, key in runs:
    H.bse_apply_params(None)
    try:
      if kind == "replica":
        tr_b1 = H._cpg_scenario_runner(name, "replica", key)(best)
      else:
        tr_b1 = _fixture_trace("real_sg", 0.0 if kind == "fixture_b1" else -0.03)
    finally:
      H.bse_restore_defaults()
    H.bse_apply_params(best)
    try:
      if kind == "replica":
        tr = H._cpg_scenario_runner(name, "replica", key)(best)
      else:
        tr = _fixture_trace("real_sg", 0.0 if kind == "fixture_b1" else -0.03)
    finally:
      H.bse_restore_defaults()
    reports.append(_spike_report(name, tr, tr_b1))
  return reports


def _fixture_trace(plant_mode, grade):
  from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import FixtureProfile
  fixture = H.load_fixture("b1")
  loop = H._RealFixtureLoop(dict(fixture.get("params", {})), FixtureProfile(fixture),
                            H.PLANT_REAL, plant_mode=plant_mode,
                            grade_rad=float(np.arcsin(grade)))
  H._apply_baseline_params(H.Params(), loop)
  loop._trace = []
  loop.run()
  return loop._trace


def _spike_report(name, trace, b1_trace):
  t = np.array([r["t"] for r in trace], dtype=float)
  out = np.array([r["out"] if r["out"] is not None else np.nan for r in trace], dtype=float)
  jerk = np.gradient(out, t)
  k = int(np.argmax(np.abs(jerk)))
  tk = t[k]
  b1_out = np.array([r["out"] if r["out"] is not None else np.nan for r in b1_trace], dtype=float)
  b1_jerk = np.gradient(b1_out, b1_trace_t(b1_trace)) if b1_trace else np.zeros_like(jerk)
  # 1 s window around the spike at ~10 ms resolution (trace step is DT_CTRL=0.01).
  lo, hi = tk - 0.5, tk + 0.5
  idx = np.flatnonzero((t >= lo) & (t <= hi))
  print(f"NOD-CPG-SPIKE {name} peak_jerk={jerk[k]:+.3f} at t={tk:.3f} "
        f"b1_peak={np.max(np.abs(b1_jerk)) if b1_jerk.size else float('nan'):+.3f}")
  # B1-off comparison at the same instant: is the step BSE-caused or the
  # controller's own pid->stopping handoff?
  tb = b1_trace_t(b1_trace)
  ob = np.array([r["out"] if r["out"] is not None else np.nan for r in b1_trace], dtype=float)
  if tb.size:
    kb = int(np.argmin(np.abs(tb - tk)))
    lo_b = max(0, kb - 3)
    print(f"NOD-CPG-SPIKE-B1 {name} b1_off_out[{tb[max(0,kb-1)]:.3f}..{tb[min(tb.size-1,kb+1)]:.3f}]="
          f"{np.array2string(ob[lo_b:kb+2], precision=4)} "
          f"b1_off_lc=" + ",".join(str(r["long_state"]) for r in b1_trace[lo_b:kb+2]))
  print(f"NOD-CPG-SPIKE-HDR {name}: t v stock ceiling extra out lc_state target plant_a bse_active")
  step = max(1, int(round(0.01 / (t[1] - t[0])))) if t.size > 1 else 1
  for i in idx[::step]:
    r = trace[i]
    print(f"NOD-CPG-SPIKE-ROW {name} {r['t']:+.3f} {r['v']:.4f} "
          f"{_fmt(r['stock_out'])} {_fmt(r['ceiling'])} {_fmt(r['extra'])} "
          f"{_fmt(r['out'])} {r['long_state']} {_fmt(r['plant_target'])} "
          f"{_fmt(r['plant_a'])} {r['bse_active']}")
  return {"name": name, "peak_jerk": float(jerk[k]), "t": float(tk),
          "b1_peak_jerk": float(np.max(np.abs(b1_jerk))) if b1_jerk.size else None}


def b1_trace_t(trace):
  return np.array([r["t"] for r in trace], dtype=float)


def main():
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    top5 = _load_top5()
    # ---- smoke: 1 set x 1 scenario, every column finite and nonzero.
    print(f"NOD-CPG-SMOKE set0 scenario=replica 445@7.27")
    b1, bse = H.cpg_run_scenario("replica 445@7.27", "replica", "445@7.27", top5[0], top5[0])
    m1, m2 = H._trace_metrics(b1), H._trace_metrics(bse)
    print(f"NOD-CPG-SMOKE b1={json.dumps(m1)}")
    print(f"NOD-CPG-SMOKE bse={json.dumps(m2)}")
    for who, m in (("b1", m1), ("bse", m2)):
      for col in ("a_stop", "nod_swing", "peak_decel", "peak_jerk", "final_gap_m"):
        assert m[col] is not None and np.isfinite(m[col]), f"smoke {who}.{col} not finite"
      assert abs(m["peak_decel"]) > 1e-9 and m["peak_jerk"] > 1e-9, f"smoke {who} zeros"

    rows = task1(top5)
    _print_table(rows)
    reports = task2(BEST)
    with open(OUT, "w", encoding="utf-8") as fh:
      json.dump({"table": rows, "spikes": reports}, fh, indent=2)


if __name__ == "__main__":
  main()
