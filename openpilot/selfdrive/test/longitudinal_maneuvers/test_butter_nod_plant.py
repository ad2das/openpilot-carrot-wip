"""STOP-NOD step 1: real-car low-speed plant calibration + baseline (test only).

Diagnostic / calibration harness for the "car nods at the stop instant" complaint.
It fits a candidate low-speed plant (`plant_real_step`: command dead time L, a
first-order lag tau, and a speed-dependent creep term) to six real auto-stops
(a_imu_fwd truth, see data/butter_nod/real_stops_nod_50hz.json) with ONE
parameter set, reports per-stop fidelity against fixed PASS targets, and runs a
closed-loop baseline with the best plant ("real") over the b1 fixture and every
maneuver scenario of test_butter_stop_maneuvers (plus the start scenarios).

This file changes no production code. Existing plants (ideal / creep / scc) are
untouched: the "real" plant lives entirely here. Simulation only; no vehicle
validation. Assertions are limited to the fixture sha256, non-collision and that
the fit ran; fidelity pass/fail is REPORTED, not asserted, in this step.
"""
import hashlib
import json
import math
import os
import types

import numpy as np

from opendbc.car.hyundai.interface import CarInterface
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_stop_maneuvers import (
  ButterStopLoop, _CASES, _START_SCENARIOS, _FakeSM, PLANNER_PERIOD,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  FarLeadLoop, FixtureProfile, load_fixture,
)

NOD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "butter_nod",
                        "real_stops_nod_50hz.json")
NOD_SHA256 = "07f34cdd72e1eab0638c4fb75df557ef3cd3cfc3e17787842d24fab86a8d1933"
FIT_DT = 0.02
FIT_START_S = -4.0
COLLISION_GAP_M = 0.4
GAS_COL = 9

# Grid axes (brief step 1.2).
P_L = [round(0.1 * i, 1) for i in range(7)]            # 0.0 .. 0.6
P_TAU = [round(0.1 * (i + 1), 1) for i in range(10)]   # 0.1 .. 1.0
P_CMAX = [round(0.05 * i, 2) for i in range(11)]       # 0.0 .. 0.5
P_VHI = [round(0.2 * (i + 1), 1) for i in range(8)]    # 0.2 .. 1.6
P_CLOW = [round(-0.3 + 0.1 * i, 1) for i in range(7)]  # -0.3 .. 0.3

# PASS targets (brief step 1.3).
T_RMS = 0.12
T_STOP_DECEL = 0.15
T_STOP_TIME = 0.4
T_DIST = 0.25

BASELINE_PARAMS = {"ButterStop": 1, "VEgoStopping": 10, "StoppingAccel": -50}


def _trapz(y, x):
  return float(np.trapezoid(y, x))


# --- candidate plant -------------------------------------------------------

def plant_real_step(state, cmd, v, dt, p):
  """One step of the candidate real-car plant.

  state: mutable dict (holds the command history and the lag state).
  cmd:   commanded acceleration (m/s^2) at this step.
  v:     current speed (m/s). p: {"L","tau","C_MAX","V_HI","C_LOW"}.

  The command is delayed by L, then first-order lagged with tau; a creep term
  c(v) = C_MAX for v >= V_HI else linear C_LOW..C_MAX is added only while the
  lagged command is < 0.3 and v > 0.01; at v == 0 the car holds (a_act = 0).
  Returns (a_act, v_next).
  """
  L = float(p["L"])
  tau = float(p["tau"])
  C_MAX = float(p["C_MAX"])
  V_HI = float(p["V_HI"])
  C_LOW = float(p["C_LOW"])

  hist = state.setdefault("hist", [])
  hist.append(float(cmd))
  n = int(round(L / dt)) if dt > 0.0 else 0
  if n <= 0:
    cmd_d = hist[-1]
  else:
    idx = len(hist) - 1 - n
    cmd_d = hist[idx] if idx >= 0 else hist[0]

  a_lag = float(state.get("a_lag", cmd_d))
  a_lag += (dt / tau) * (cmd_d - a_lag) if tau > 0.0 else (cmd_d - a_lag)
  state["a_lag"] = a_lag

  if v >= V_HI:
    c = C_MAX
  else:
    c = C_LOW + (C_MAX - C_LOW) * (v / V_HI if V_HI > 0.0 else 0.0)
  a_act = a_lag
  if a_lag < 0.3 and v > 0.01:
    a_act += c
  if v <= 0.0:
    a_act = 0.0
  v_next = max(0.0, v + a_act * dt)
  return a_act, v_next


# --- fixture load ----------------------------------------------------------

def _read_nod_raw():
  with open(NOD_PATH, "rb") as handle:
    return handle.read()


def load_stops():
  d = json.loads(_read_nod_raw().decode("utf-8"))
  stops = []
  for entry in d["stops"]:
    rows = np.array(entry["rows"], dtype=float)
    if rows[:, GAS_COL].max() > 0.5:
      # Stop 445@1.43 (and any gas-tainted stop): use only rows after the last gas.
      gas = np.flatnonzero(rows[:, GAS_COL] > 0.5)
      rows = rows[gas[-1] + 1:]
    stops.append({"route": entry["route"], "minute": entry["minute"], "rows": rows})
  return d, stops


def _prepare_stop(stop):
  """Native fixture samples from t_rel -4.0 on (irregular ~50 Hz timestamps)."""
  rows = stop["rows"]
  t = rows[:, 0]
  i0 = int(np.flatnonzero(t >= FIT_START_S)[0])
  tt = t[i0:]
  dts = np.diff(tt, prepend=tt[0] - (tt[1] - tt[0]) if tt.size > 1 else FIT_DT)
  dts = np.maximum(dts, 1e-6)
  return {
    "t_full": t, "v_full": rows[:, 1], "imu_full": rows[:, 3],
    "t": tt, "dts": dts,
    "v0": float(rows[i0, 1]),
    "cmd": rows[i0:, 4].copy(),
    "imu": rows[i0:, 3].copy(),
  }


# --- open-loop fit (vectorized over the creep axes) -----------------------

def _vec_rms(prep, L, tau, CM, VH, CL):
  cmd = prep["cmd"]
  imu = prep["imu"]
  dts = prep["dts"]
  N = cmd.size
  a_lag = np.empty(N)
  a_lag[0] = cmd[0]
  for i in range(1, N):
    n = int(round(L / dts[i]))
    cmd_d = cmd[max(0, i - n)]
    a_lag[i] = a_lag[i - 1] + (dts[i] / tau) * (cmd_d - a_lag[i - 1])

  K = CM.size
  v = np.full(K, prep["v0"])
  done = np.zeros(K, dtype=bool)
  err2 = np.zeros(K)
  cnt = np.zeros(K)
  for i in range(N):
    c = np.where(v >= VH, CM, CL + (CM - CL) * (v / VH))
    gate = (a_lag[i] < 0.3) & (v > 0.01)
    a_act = a_lag[i] + np.where(gate, c, 0.0)
    a_act = np.where(v <= 0.0, 0.0, a_act)
    m = ~done
    err2 += np.where(m, (a_act - imu[i]) ** 2, 0.0)
    cnt += m
    v_next = np.maximum(0.0, v + a_act * dts[i])
    done = done | (m & (v_next < 0.01))
    v = v_next
  rms = np.sqrt(err2 / np.maximum(cnt, 1.0))
  rms[cnt == 0.0] = np.nan
  return rms


def _mesh(cmax_list, vhi_list, clow_list):
  cm, vh, cl = np.meshgrid(np.array(cmax_list), np.array(vhi_list), np.array(clow_list),
                           indexing="ij")
  return cm.ravel(), vh.ravel(), cl.ravel()


def fit_real_plant(stops, L_list, tau_list, cmax_list, vhi_list, clow_list):
  """Grid-search ONE parameter set for all stops; returns (loss, params) sorted."""
  preps = [_prepare_stop(s) for s in stops]
  CM, VH, CL = _mesh(cmax_list, vhi_list, clow_list)
  K = CM.size
  results = []
  for L in L_list:
    for tau in tau_list:
      per_stop = np.empty((len(preps), K))
      for si, prep in enumerate(preps):
        per_stop[si] = _vec_rms(prep, L, tau, CM, VH, CL)
      avg = np.nanmean(per_stop, axis=0)
      for k in range(K):
        if np.isfinite(avg[k]):
          results.append((float(avg[k]), {"L": L, "tau": tau, "C_MAX": float(CM[k]),
                                          "V_HI": float(VH[k]), "C_LOW": float(CL[k])}))
  results.sort(key=lambda item: item[0])
  return results


def _simulate_scalar(prep, p):
  cmd = prep["cmd"]
  dts = prep["dts"]
  N = cmd.size
  v = prep["v0"]
  a_lag = cmd[0]
  vs = np.empty(N)
  aas = np.empty(N)
  stop_i = N - 1
  for i in range(N):
    if i > 0:
      n = int(round(p["L"] / dts[i]))
      cmd_d = cmd[max(0, i - n)]
      a_lag += (dts[i] / p["tau"]) * (cmd_d - a_lag)
    c = p["C_MAX"] if v >= p["V_HI"] else p["C_LOW"] + (p["C_MAX"] - p["C_LOW"]) * (v / p["V_HI"])
    a_act = a_lag + (c if (a_lag < 0.3 and v > 0.01) else 0.0)
    if v <= 0.0:
      a_act = 0.0
    vs[i] = v
    aas[i] = a_act
    v = max(0.0, v + a_act * dts[i])
    if v < 0.01:
      stop_i = i
      break
  return vs[:stop_i + 1], aas[:stop_i + 1], stop_i


def fidelity_stop(stop, prep, p):
  vs, aas, stop_i = _simulate_scalar(prep, p)
  t = prep["t"]
  imu = prep["imu"]
  k = stop_i
  out = {"route": stop["route"], "minute": stop["minute"]}
  rms = float(np.sqrt(np.mean((aas - imu[:k + 1]) ** 2)))
  w = max(0, np.searchsorted(t[:k + 1], t[k] - 0.6))
  sim_decel = float(np.min(aas[w:k + 1]))
  tf = prep["t_full"]
  imuf = prep["imu_full"]
  mask = (tf >= -0.6) & (tf <= 0.0)
  imu_decel = float(np.min(imuf[mask])) if mask.any() else float(np.min(imuf))
  stop_time = float(t[k])
  sim_dist = _trapz(vs, t[:k + 1])
  log_dist = _trapz(np.interp(t[:k + 1], tf, prep["v_full"]), t[:k + 1])
  out.update({
    "accel_rms": round(rms, 4),
    "stop_decel_sim": round(sim_decel, 4),
    "stop_decel_imu": round(imu_decel, 4),
    "stop_decel_err": round(abs(sim_decel - imu_decel), 4),
    "stop_time_sim_s": round(stop_time, 3),
    "stop_time_err_s": round(abs(stop_time), 3),
    "dist_sim_m": round(sim_dist, 3),
    "dist_log_m": round(log_dist, 3),
    "dist_err_m": round(abs(sim_dist - log_dist), 3),
  })
  out["pass"] = bool(rms <= T_RMS and out["stop_decel_err"] <= T_STOP_DECEL
                     and out["stop_time_err_s"] <= T_STOP_TIME and out["dist_err_m"] <= T_DIST)
  return out


def fidelity_all(stops, p):
  preps = [_prepare_stop(s) for s in stops]
  rows = [fidelity_stop(stop, prep, p) for stop, prep in zip(stops, preps, strict=True)]
  return {"stops": rows,
          "targets": {"accel_rms": T_RMS, "stop_decel_err": T_STOP_DECEL,
                      "stop_time_err_s": T_STOP_TIME, "dist_err_m": T_DIST},
          "all_pass": all(r["pass"] for r in rows)}


# --- closed-loop baseline with the real plant ------------------------------

def _run_loop_real(loop):
  """Shared body of ButterStopLoop._run_loop / FarLeadLoop.run with the real plant.

  `loop` is a ButterStopLoop- or FarLeadLoop-shaped instance whose plant / scc
  flags are off and which carries `real_p` and `_real_state`. Returns the same
  array / dict shapes those loops return.
  """
  maneuver = getattr(loop, "maneuver", None)
  fixture_mode = maneuver is None
  dt = DT_CTRL
  if fixture_mode:
    profile = loop.profile
    max_frames = profile.n_steps * PLANNER_PERIOD
    maneuver_enabled = True
  else:
    frames = int(round(maneuver.duration / dt))
    max_frames = frames + 1
    maneuver_enabled = bool(maneuver.enabled)

  fw_rows = []
  log_t, log_v, log_accel, log_x = [], [], [], []
  log_gap, log_active, log_vlead, log_atarget = [], [], [], []
  log_a_actual, log_should_stop, log_lc_state = [], [], []
  log_desired_distance = []
  fcw_steps = 0
  fcw = 0.0
  msgs = None
  plan_view = None
  if fixture_mode:
    v_cruise_kph = profile.v_cruise_kph(0.0)
  standstill_frames = 0
  stopped = False
  for frame in range(max_frames):
    t = frame * dt
    if fixture_mode:
      v_lead = None
      standstill = loop.ego_v < 0.01
      v_cruise_kph = profile.v_cruise_kph(t)
      v_cruise_ms = v_cruise_kph / 3.6
      cs_cruise = v_cruise_kph  # FarLeadLoop._cs_view takes km/h
    else:
      v_lead = float(np.interp(t, maneuver.breakpoints, maneuver.speed_lead_values))
      v_cruise_ms = float(np.interp(t, maneuver.breakpoints, maneuver.cruise_values))
      cs_cruise = v_cruise_ms  # ButterStopLoop._cs_view takes m/s
      pitch = float(np.interp(t, maneuver.breakpoints, maneuver.pitch_values))
      prob_lead = float(np.interp(t, maneuver.breakpoints, maneuver.prob_lead_values))
      prob_throttle = float(np.interp(t, maneuver.breakpoints, maneuver.prob_throttle_values))
      standstill = loop.ego_v < 0.01

    if frame % PLANNER_PERIOD == 0:
      if fixture_mode:
        specs = profile.sample(t, loop.ego_x)
        radar_state = loop._radar_message(specs)
        msgs = {"radarState": radar_state}
        msgs.update(loop._sm_messages(v_cruise_kph, standstill))
        loop.planner.update(_FakeSM(msgs, frame), loop.carrot)
        plan_view = loop._plan_view()
        if loop.planner.mpc.crash_cnt > 0:
          fcw_steps += 1
      else:
        a_lead = (v_lead - loop.v_lead_prev) / (PLANNER_PERIOD * dt)
        loop.v_lead_prev = v_lead
        if maneuver.lead_relevancy:
          status = True if maneuver.only_radar else prob_lead > 0.5
          d_rel = max(0.0, loop.lead_x - loop.ego_x)
          if loop.radar_noise_std > 0.0:
            d_rel += float(loop._radar_rng.normal(0.0, loop.radar_noise_std))
          lead = {"d_rel": d_rel, "v_rel": v_lead - loop.ego_v, "v_lead": v_lead,
                  "a_lead": a_lead if not np.isnan(a_lead) else 0.0,
                  "prob": prob_lead if status else 0.0}
        else:
          lead = None
        radar_state = loop._radar_message(lead)
        msgs = {"radarState": radar_state}
        msgs.update(loop._sm_messages(v_cruise_ms, pitch, prob_throttle, standstill))
        loop.planner.update(_FakeSM(msgs, frame), loop.carrot)
        plan_view = loop._plan_view()
        fcw = 1.0 if loop.planner.mpc.crash_cnt > 0 else 0.0

    t_since_plan = (frame % PLANNER_PERIOD) * dt
    limits = CarInterface.get_pid_accel_limits(loop.cp, loop.ego_v, v_cruise_ms)
    radar_view = types.SimpleNamespace(leadOne=msgs["radarState"].leadOne,
                                       leadTwo=msgs["radarState"].leadTwo)
    accel, _, _ = loop.long_control.update(
      maneuver_enabled, loop._cs_view(cs_cruise, standstill), plan_view,
      limits, t_since_plan, radar_view)
    accel = float(accel)
    applied, _ = plant_real_step(loop._real_state, accel, loop.ego_v, dt, loop.real_p)

    if fixture_mode:
      gap = np.nan
      v_lead_now = np.nan
      active = False
      for spec in profile.sample(t, loop.ego_x):
        if spec is not None and spec[3]:
          g = spec[0] - loop.ego_x
          if not active or g < gap:
            gap, v_lead_now, active = g, spec[1], True
      log_t.append(t)
      log_v.append(loop.ego_v)
      log_accel.append(accel)
      log_x.append(loop.ego_x)
      log_gap.append(gap)
      log_active.append(active)
      log_vlead.append(v_lead_now)
      log_atarget.append(float(plan_view.aTarget))
      log_a_actual.append(applied)
      log_should_stop.append(bool(plan_view.shouldStop))
      log_lc_state.append(int(loop.long_control.long_control_state))
      log_desired_distance.append(float(plan_view.desiredDistance))
    else:
      floor_active = 1.0 if loop.long_control.butter_floor_last is not None else 0.0
      floor_val = loop.long_control.butter_floor_last
      floor_raw = loop.long_control.butter_floor_raw_last
      bound = loop.long_control.butter_bound_last
      bound_code = -1.0 if floor_active == 0.0 else (1.0 if bound == "lead" else 0.0)
      a_end = loop.long_control.butter_a_end_last
      fw_rows.append((t, loop.ego_v, accel, loop.ego_x, loop.lead_x, floor_active,
                      floor_val if floor_val is not None else 0.0, bound_code,
                      floor_raw if floor_raw is not None else 0.0,
                      a_end if a_end is not None else 0.0,
                      applied, float(loop.long_control.long_control_state), fcw))

    loop.ego_v = max(0.0, loop.ego_v + applied * dt)
    loop.ego_x = loop.ego_x + loop.ego_v * dt
    if fixture_mode:
      if loop.ego_v < 0.05:
        standstill_frames += 1
        if standstill_frames >= 300:
          stopped = True
          break
      else:
        standstill_frames = 0
    else:
      loop.lead_x = loop.lead_x + v_lead * dt

  if fixture_mode:
    return {
      "t": log_t, "v": log_v, "accel": log_accel, "x": log_x,
      "gap": log_gap, "active": log_active, "v_lead": log_vlead, "a_target": log_atarget,
      "a_actual": log_a_actual, "should_stop": log_should_stop, "lc_state": log_lc_state,
      "desired_distance": log_desired_distance,
      "fcw_steps": fcw_steps, "stopped": bool(stopped or loop.ego_v < 0.05),
    }
  return np.array(fw_rows)


class _RealScenarioLoop(ButterStopLoop):
  def __init__(self, maneuver, butter, p):
    super().__init__(maneuver, butter=butter, plant=False)
    self.real_p = dict(p)
    self._real_state = None

  def run(self):
    self._real_state = {"hist": []}
    return _run_loop_real(self)


class _RealFixtureLoop(FarLeadLoop):
  def __init__(self, header, profile, p):
    super().__init__(header, profile, plant=False)
    self.real_p = dict(p)
    self._real_state = None

  def run(self):
    self._real_state = {"hist": []}
    return _run_loop_real(self)


def _stop_row(name, group, butter, t, v, applied, gap, fcw_steps, start_delay):
  t = np.asarray(t, dtype=float)
  v = np.asarray(v, dtype=float)
  applied = np.asarray(applied, dtype=float)
  gap = np.asarray(gap, dtype=float)
  finite = np.isfinite(gap)
  stand = np.flatnonzero(v < 0.01)
  stopped = bool(stand.size)
  row = {"name": name, "group": group, "butter": int(butter), "stopped": stopped,
         "stop_decel_at_stop": None, "nod_swing": None, "tail_s": None,
         "final_gap_m": None,
         "min_gap_m": None if not finite.any() else round(float(np.nanmin(gap)), 3),
         "collision": bool(finite.any() and np.nanmin(gap) < COLLISION_GAP_M),
         "fcw_steps": int(fcw_steps), "start_delay_s": start_delay}
  if stopped:
    i = int(stand[0])
    step = float(t[1] - t[0]) if t.size > 1 else DT_CTRL
    w = (t >= t[i] - 0.6) & (t <= t[i])
    decel = float(np.min(applied[w])) if w.any() else None
    w2 = (t > t[i]) & (t <= t[i] + 0.3)
    swing = float(np.max(applied[w2]) - decel) if (w2.any() and decel is not None) else None
    tail_w = (t >= max(0.0, t[i] - 6.0)) & (t <= t[i])
    row["stop_decel_at_stop"] = None if decel is None else round(decel, 4)
    row["nod_swing"] = None if swing is None else round(swing, 4)
    row["tail_s"] = round(float(np.sum((v > 0.05) & (v < 0.5) & tail_w)) * step, 3)
    row["final_gap_m"] = None if not finite.any() else round(float(gap[i]), 3)
  return row


def _start_delay(t, v):
  v = np.asarray(v, dtype=float)
  moving = np.flatnonzero(v > 0.3)
  return None if not moving.size else round(float(np.asarray(t, dtype=float)[moving[0]] - 5.0), 3)


def _apply_baseline_params(params, loop):
  """Seed the brief's baseline params (VEgoStopping 0.10, StoppingAccel -0.5) and
  force LongControl to re-read them (it caches stopping_accel at construction)."""
  params.put_int("VEgoStopping", BASELINE_PARAMS["VEgoStopping"])
  params.put_int("StoppingAccel", BASELINE_PARAMS["StoppingAccel"])
  loop.long_control._refresh_stopping_accel()
  loop.long_control._refresh_butter_stop()


def run_baseline(p):
  rows = []
  params = Params()
  fixture = load_fixture("b1")
  for butter in (1, 0):
    header = dict(fixture["params"])
    if butter == 0:
      header["ButterStop"] = 0
    loop = _RealFixtureLoop(header, FixtureProfile(fixture), p)
    _apply_baseline_params(params, loop)
    run = loop.run()
    rows.append(_stop_row("b1", "b1", butter, run["t"], run["v"], run["a_actual"],
                          run["gap"], run["fcw_steps"], None))
  for group, maneuver in _CASES:
    for butter in (1, 0):
      loop = _RealScenarioLoop(maneuver, butter, p)
      _apply_baseline_params(params, loop)
      logs = loop.run()
      gap = logs[:, 4] - logs[:, 3]
      rows.append(_stop_row(maneuver.title, group, butter, logs[:, 0], logs[:, 1],
                            logs[:, 10], gap, int(np.sum(logs[:, 12] > 0.5)), None))
  for maneuver in _START_SCENARIOS:
    for butter in (1, 0):
      loop = _RealScenarioLoop(maneuver, butter, p)
      _apply_baseline_params(params, loop)
      logs = loop.run()
      gap = logs[:, 4] - logs[:, 3]
      rows.append(_stop_row(maneuver.title, "start", butter, logs[:, 0], logs[:, 1],
                            logs[:, 10], gap, int(np.sum(logs[:, 12] > 0.5)),
                            _start_delay(logs[:, 0], logs[:, 1])))
  return rows


# --- tests -----------------------------------------------------------------

def test_butter_nod_fixture_sha256():
  assert hashlib.sha256(_read_nod_raw()).hexdigest() == NOD_SHA256


def test_butter_nod_smoke():
  """2x2x2x2x2 mini-grid on 2 stops: every output finite, loss non-constant."""
  _d, stops = load_stops()
  subset = [stops[0], stops[2]]
  cmax = [0.0, 0.25]
  vhi = [0.4, 1.0]
  clow = [-0.1, 0.2]
  fits = fit_real_plant(subset, [0.0, 0.3], [0.1, 0.6], cmax, vhi, clow)
  assert fits, "mini-grid produced no finite fits"
  losses = np.array([f[0] for f in fits], dtype=float)
  assert np.all(np.isfinite(losses)), f"non-finite loss: {losses}"
  assert losses.max() - losses.min() > 1e-6, f"loss is constant: {losses}"

  best = fits[0][1]
  prep = _prepare_stop(subset[0])
  CM, VH, CL = _mesh(cmax, vhi, clow)
  vec = _vec_rms(prep, best["L"], best["tau"], CM, VH, CL)
  _vs, _aas, _k = _simulate_scalar(prep, best)
  scalar_rms = float(np.sqrt(np.mean((_aas - prep["imu"][:_aas.size]) ** 2)))
  match = np.flatnonzero((np.abs(CM - best["C_MAX"]) < 1e-9)
                         & (np.abs(VH - best["V_HI"]) < 1e-9)
                         & (np.abs(CL - best["C_LOW"]) < 1e-9))
  assert match.size, "best creep params not found in mesh"
  assert abs(vec[match[0]] - scalar_rms) < 1e-6, f"vectorized {vec[match[0]]} != scalar {scalar_rms}"

  fid = fidelity_all(subset, best)
  print(f"NOD-SMOKE loss min={losses.min():.5f} max={losses.max():.5f} n={losses.size}")
  print(f"NOD-SMOKE best={best}")
  print(f"NOD-SMOKE fidelity={json.dumps(fid['stops'], sort_keys=True)}")
  for row in fid["stops"]:
    for key in ("accel_rms", "stop_decel_sim", "stop_decel_imu", "stop_decel_err",
                "stop_time_sim_s", "stop_time_err_s", "dist_sim_m", "dist_log_m", "dist_err_m"):
      assert math.isfinite(row[key]), f"{key} not finite: {row[key]}"


def test_butter_nod_fit_baseline():
  out_path = os.environ.get("BUTTER_NOD_OUT", "/tmp/butter_nod.json")
  raw = _read_nod_raw()
  assert hashlib.sha256(raw).hexdigest() == NOD_SHA256

  _d, stops = load_stops()
  fits = fit_real_plant(stops, P_L, P_TAU, P_CMAX, P_VHI, P_CLOW)
  assert fits, "fit produced no finite parameter sets"
  top5 = [{"loss": round(loss, 6), "params": params} for loss, params in fits[:5]]
  best_loss, best_p = fits[0]

  fid = fidelity_all(stops, best_p)
  base = run_baseline(best_p)
  assert not any(r["collision"] for r in base), "baseline collision"

  table = {
    "fixture_sha256": NOD_SHA256,
    "fit": {
      "grid": {"L": P_L, "tau": P_TAU, "C_MAX": P_CMAX, "V_HI": P_VHI, "C_LOW": P_CLOW,
               "dt": FIT_DT, "start_s": FIT_START_S, "n_combo": len(fits)},
      "loss": round(best_loss, 6),
      "best": {"loss": round(best_loss, 6), "params": best_p},
      "top5": top5,
    },
    "fidelity": fid,
    "baseline": {"params": {"plant": "real", **BASELINE_PARAMS}, "rows": base},
    "note": "desktop open-loop fit / closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  print(f"NOD-FIT best loss={best_loss:.6f} params={best_p} n_combo={len(fits)}")
  for i, row in enumerate(fid["stops"]):
    print(f"NOD-FIDELITY stop{i} {row['route'][-8:]}@{row['minute']} pass={row['pass']} {row}")
  print(f"NOD-BASELINE rows={len(base)} collisions={sum(r['collision'] for r in base)}")
