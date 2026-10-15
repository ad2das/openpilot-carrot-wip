"""ButterStop real-stop replay (W112 task 4, diagnostic only).

Measurement replay of route 00000441--93d6b5c585 (build 59f28f940, ButterStop
3.2): the car stopped 7.2 m behind a stopped lead after a long slow crawl.
Fixture b1 is the [T-10, T+25] window around that stop. Two plant models are
compared: OFF (ideal, as in the other maneuver tests) and ON (automatic-
transmission creep torque plus 0.3 s actuator lag, fit from this log).

Variants: B0 = ButterStop 0; B1 = ButterStop 1 (as logged); B1d = ButterStop 1
with butter_stop.D_EXTRA = 0; B1f = ButterStop 1 with the taper floor disabled
but the MPC stop shift kept. The 9 normal a/b/c scenarios of
test_butter_stop_maneuvers also run with the plant ON for B0/B1.

Assertions: only no-collision and fixture presence; everything else is
reported to BUTTER_REAL_OUT. Simulation only; no vehicle validation.
"""
import json
import os
from contextlib import contextmanager

import numpy as np

from openpilot.selfdrive.controls.lib import butter_stop, longcontrol
from openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_stop_maneuvers import (
  ButterStopLoop, _CASES,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  SENTINEL, FarLeadLoop, FixtureProfile, load_fixture,
)

VARIANTS = ("B0", "B1", "B1d", "B1f")
LC_NAMES = {0: "off", 1: "pid", 2: "stopping", 3: "starting"}
COLLISION_GAP_M = 0.4  # same crash criterion as maneuver.py
FIDELITY_FINAL_GAP_M = 7.2
FIDELITY_FINAL_GAP_TOL_M = 1.0
FIDELITY_BELOW7_S = 13.6
FIDELITY_BELOW7_TOL_S = 4.0

RESULTS: dict = {"b1": {}, "scenarios_plant_on": {}, "log_reference": {}, "fidelity": {}, "report": {}}


def _no_floor(*_args, **_kwargs):
  return None


@contextmanager
def _butter_patch(*, d_extra=None, no_floor=False):
  saved = (butter_stop.D_EXTRA, longcontrol.butter_floor)
  try:
    if d_extra is not None:
      butter_stop.D_EXTRA = d_extra
    if no_floor:
      longcontrol.butter_floor = _no_floor
    yield
  finally:
    butter_stop.D_EXTRA, longcontrol.butter_floor = saved


def _unscale(values, scale):
  arr = np.array(values, dtype=float)
  arr[arr <= SENTINEL] = np.nan
  return arr / scale


def _log_reference(fixture):
  data, scale = fixture["data"], fixture["scale"]
  t = np.array(data["t_cs"], dtype=float) / scale["t"]
  v = _unscale(data["vEgo"], scale["speed"])
  x = _unscale(data["sEgo"], scale["dist"])
  gap = _unscale(data["leadOne_dRel"], scale["dist"])
  v_lead = _unscale(data["leadOne_vLead"], scale["speed"])
  status = _unscale(data["leadOne_status"], 1)
  stand = np.flatnonzero(v < 0.05)
  below7 = np.flatnonzero(v < 7.0 / 3.6)
  stopped = bool(stand.size)
  ref = {
    "final_gap_m": None if not stopped else round(float(gap[stand[0]]), 3),
    "below7_time_s": None if not (stopped and below7.size) else round(float(t[stand[0]] - t[below7[0]]), 2),
    "below7_dist_m": None if not (stopped and below7.size) else round(float(x[stand[0]] - x[below7[0]]), 2),
    "vEgo_0_kph": round(float(v[0]) * 3.6, 2),
    "vEgo_end_mps": round(float(v[-1]), 3),
  }
  lead_stop = np.flatnonzero((status > 0.5) & (v_lead < 0.3))
  ref["lead_stop_t_s"] = None if not lead_stop.size else round(float(t[lead_stop[0]]), 2)
  ref["lead_stop_gap_m"] = None if not lead_stop.size else round(float(gap[lead_stop[0]]), 2)
  if lead_stop.size and stopped:
    ref["lead_stop_to_standstill_s"] = round(float(t[stand[0]] - t[lead_stop[0]]), 2)
  else:
    ref["lead_stop_to_standstill_s"] = None
  return ref


def _state_timeline(t, lc_state):
  out = []
  prev = None
  for ti, state in zip(t, lc_state, strict=True):
    value = int(state)
    if value != prev:
      out.append({"t_s": round(float(ti), 2), "state": LC_NAMES.get(value, str(value))})
      prev = value
  return out


def _b1_metrics(run):
  t = np.array(run["t"], dtype=float)
  v = np.array(run["v"], dtype=float)
  a = np.array(run["a_actual"], dtype=float)
  x = np.array(run["x"], dtype=float)
  gap = np.array(run["gap"], dtype=float)
  active = np.array(run["active"], dtype=bool)
  v_lead = np.array(run["v_lead"], dtype=float)
  a_target = np.array(run["a_target"], dtype=float)
  should_stop = np.array(run["should_stop"], dtype=bool)

  stand = np.flatnonzero(v < 0.05)
  below7 = np.flatnonzero(v < 7.0 / 3.6)
  stopped = bool(stand.size)
  metrics: dict = {
    "stopped": stopped,
    "min_gap_m": None if not active.any() else round(float(np.min(gap[active])), 3),
    "collision": bool(np.any(gap[active] <= 0.0)),
  }
  if stopped and below7.size and below7[0] < stand[0]:
    i_stop, i_b7 = int(stand[0]), int(below7[0])
    metrics["final_gap_m"] = round(float(gap[i_stop]), 3)
    metrics["crawl_time_s"] = round(float(t[i_stop] - t[i_b7]), 2)
    metrics["crawl_dist_m"] = round(float(x[i_stop] - x[i_b7]), 2)
    window = (t >= t[i_stop] - 0.5) & (t <= t[i_stop])
    metrics["mean_accel_last_0.5s"] = round(float(np.mean(a[window])), 4) if window.any() else None
    jerk = np.gradient(a, t)
    jwindow = (t >= max(0.0, t[i_stop] - 1.5)) & (t <= t[i_stop])
    metrics["peak_jerk_last_1.5s"] = round(float(np.max(np.abs(jerk[jwindow]))), 4) if jwindow.any() else None
    metrics["max_abs_a_target_crawl"] = round(float(np.max(np.abs(a_target[i_b7:i_stop + 1]))), 4)
  else:
    metrics.update({"final_gap_m": None, "crawl_time_s": None, "crawl_dist_m": None,
                    "mean_accel_last_0.5s": None, "peak_jerk_last_1.5s": None,
                    "max_abs_a_target_crawl": None})

  lead_stop = np.flatnonzero(active & (v_lead < 0.3))
  if lead_stop.size and stopped:
    metrics["lead_stop_to_standstill_s"] = round(float(t[stand[0]] - t[lead_stop[0]]), 2)
  else:
    metrics["lead_stop_to_standstill_s"] = None
  fire = np.flatnonzero(should_stop)
  if fire.size:
    metrics["should_stop_gap_m"] = round(float(gap[fire[0]]), 3)
    metrics["should_stop_v_mps"] = round(float(v[fire[0]]), 3)
  else:
    metrics["should_stop_gap_m"] = None
    metrics["should_stop_v_mps"] = None
  metrics["lc_timeline"] = _state_timeline(t, run["lc_state"])
  metrics["fcw_steps"] = int(run["fcw_steps"])
  return metrics


def _run_b1(variant, plant):
  fixture = load_fixture("b1")
  header = dict(fixture["params"])
  patch_kwargs: dict = {}
  if variant == "B0":
    header["ButterStop"] = 0
  elif variant == "B1d":
    patch_kwargs["d_extra"] = 0.0
  elif variant == "B1f":
    patch_kwargs["no_floor"] = True
  with _butter_patch(**patch_kwargs):
    run = FarLeadLoop(header, FixtureProfile(fixture), plant=plant).run()
  return _b1_metrics(run)


def test_butter_real_b1():
  fixture = load_fixture("b1")
  assert fixture["grid"]["n"] > 100
  RESULTS["log_reference"] = _log_reference(fixture)
  for variant in VARIANTS:
    for plant in (False, True):
      key = f"{variant}_plant_{'on' if plant else 'off'}"
      metrics = _run_b1(variant, plant)
      RESULTS["b1"][key] = metrics
      assert not metrics["collision"], f"{key} collides (min gap {metrics['min_gap_m']})"

  sim = RESULTS["b1"]["B1_plant_on"]
  fidelity = {
    "final_gap_sim_m": sim["final_gap_m"],
    "final_gap_target_m": FIDELITY_FINAL_GAP_M,
    "final_gap_pass": sim["final_gap_m"] is not None
                      and abs(sim["final_gap_m"] - FIDELITY_FINAL_GAP_M) <= FIDELITY_FINAL_GAP_TOL_M,
    "below7_time_sim_s": sim["crawl_time_s"],
    "below7_time_target_s": FIDELITY_BELOW7_S,
    "below7_time_pass": sim["crawl_time_s"] is not None
                        and abs(sim["crawl_time_s"] - FIDELITY_BELOW7_S) <= FIDELITY_BELOW7_TOL_S,
  }
  RESULTS["fidelity"] = fidelity
  gap_ok = "pass" if fidelity["final_gap_pass"] else "FAIL"
  time_ok = "pass" if fidelity["below7_time_pass"] else "FAIL"
  print(f"BUTTER-REAL fidelity: final gap {fidelity['final_gap_sim_m']} ({gap_ok}), below-7 time {fidelity['below7_time_sim_s']} ({time_ok})")


def _scenario_metrics(logs, name):
  t, v = logs[:, 0], logs[:, 1]
  a = logs[:, 10]
  gaps = logs[:, 4] - logs[:, 3]
  stand = np.flatnonzero(v < 0.05)
  stopped = bool(stand.size)
  entry = {
    "scenario": name,
    "stopped": stopped,
    "final_gap_m": round(float(gaps[stand[0]]), 3) if stopped else None,
    "min_gap_m": round(float(np.min(gaps)), 3),
    "collision": bool(np.min(gaps) < COLLISION_GAP_M),
  }
  if stopped:
    i = int(stand[0])
    window = (t >= t[i] - 0.5) & (t <= t[i])
    entry["end_decel_mean_abs"] = round(float(np.mean(np.abs(a[window]))), 4) if window.any() else None
  else:
    entry["end_decel_mean_abs"] = None
  return entry


def test_butter_real_scenarios_plant_on():
  rows = []
  for group, maneuver in _CASES:
    if group not in ("a", "b", "c"):
      continue
    for butter in (0, 1):
      logs = ButterStopLoop(maneuver, butter=butter, plant=True).run()
      entry = _scenario_metrics(logs, maneuver.title)
      entry["group"] = group
      entry["butter"] = butter
      rows.append(entry)
      assert not entry["collision"], f"{maneuver.title} butter={butter} collides (min gap {entry['min_gap_m']})"
  RESULTS["scenarios_plant_on"] = rows


def test_write_butter_real_table():
  out_path = os.environ.get("BUTTER_REAL_OUT", "/tmp/butter_real.json")
  table = {
    "log_reference": RESULTS["log_reference"],
    "b1": RESULTS["b1"],
    "fidelity": RESULTS["fidelity"],
    "scenarios_plant_on": RESULTS["scenarios_plant_on"],
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  text = json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False)
  print(text)
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(text + "\n")
