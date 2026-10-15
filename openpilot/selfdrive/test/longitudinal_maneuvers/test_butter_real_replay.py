"""ButterStop real-stop fidelity (W112 task 5, diagnostic only).

Part 1 — open-loop planner replay: feed the LOGGED ego state (vEgo, aEgo,
carState flags, controlsState/selfdriveState fields) and LOGGED radar leads to
the real LongitudinalPlanner + CarrotPlanner over the b1 fixture (warm-up
window [T-20, T+25], 10 s before the reported stop) and compare per step with
the logged longitudinalPlan (aTarget, shouldStop, vTargetNow, tFollow,
desiredDistance, source). The ego is NOT simulated here.

Part 2 — closed-loop with three plants (ideal / creep / creep+SCC stop):
B0/B1/B1d/B1f as in task 4, plus a Hyundai SCC stop model — while LongControl
is in the stopping state the car brakes to standstill by itself
(actual accel = min(a_cmd_lagged + creep(v), -0.6), hold 0 at v <= 0.01).
Fidelity target (creep+SCC, B1): final gap 7.2 +- 1.0 m, below-7 km/h time
13.6 +- 4 s.

Assertions: only no-collision and fixture presence; everything else is
reported to BUTTER_REAL_OUT. Simulation only; no vehicle validation.

Part 3 (task 9) — offset-cap sweep + stock reference, gated by
BUTTER_REAL_SWEEP_OUT: rows OFF (memory disabled entirely: BrakingGapState
update forced enabled=False), NOCAP (cap inf), candidates K2.0/V3, K2.0/V5,
K2.0/V8, K1.5/V5 (FLOOR_SPEED_CAP_S / FLOOR_CAP_OFFSET_V) over b1 (ideal and
creep+SCC plants) and the 9 ButterStop scenarios (both plants), each for
ButterStop 0 and 1. Criteria a-f are judged against the OFF row (see
_OFFSET_SWEEP_CRITERIA); references (OFF/NOCAP) are judged but not gated.
Autopick: smallest b1 ideal B1 crawl, then smallest FLOOR_CAP_OFFSET_V. The
sweep asserts that the current default pair passes a-f, so CI guards the
shipped defaults; collision anywhere also fails the test. The b1 B0/B1
ideal-plant telemetry (floor/margin/stopped_pair/fast_release/cap per 0.05 s)
goes to BUTTER_REAL_OUT.
"""
import json
import math
import os
from contextlib import contextmanager

import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.radar_constants import LEAD_ACCEL_TAU
from openpilot.selfdrive.controls.lib import butter_stop, longcontrol, longitudinal_braking_gap
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner
from openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_stop_maneuvers import (
  ButterStopLoop, _CASES, _case_failures as _butter_case_failures,
  compute_metrics as _butter_metrics,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  CAR_MODEL, SENTINEL, FarLeadLoop, FixtureProfile, _FakeSM, _apply_header_params,
  _seed_param_defaults, load_fixture,
)
from opendbc.car.hyundai.interface import CarInterface

VARIANTS = ("B0", "B1", "B1d", "B1f")
PLANT_MODES = ("ideal", "creep", "scc")
LC_NAMES = {0: "off", 1: "pid", 2: "stopping", 3: "starting"}
LC_ENUM = (LongCtrlState.off, LongCtrlState.pid, LongCtrlState.stopping, LongCtrlState.starting)
SOURCE_IDS = {"cruise": 0, "lead0": 1, "lead1": 2, "lead2": 3, "e2e": 4}
SOURCE_NAMES = {v: k for k, v in SOURCE_IDS.items()}
COLLISION_GAP_M = 0.4  # same crash criterion as maneuver.py
FIDELITY_FINAL_GAP_M = 7.2
FIDELITY_FINAL_GAP_TOL_M = 1.0
FIDELITY_BELOW7_S = 13.6
FIDELITY_BELOW7_TOL_S = 4.0
OPEN_LOOP_WINDOW_S = -10.0  # metric window; replay starts at fixture start (-20 s)

RESULTS: dict = {"b1": {}, "b1_telemetry": {}, "open_loop": {}, "stop_release": {},
                 "scenarios_plant_on": {}, "offset_sweep": {},
                 "log_reference": {}, "fidelity": {}, "report": {}}


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
  ss_log = np.flatnonzero(np.array(data["shouldStopLog"], dtype=float) > 0.5)
  ref["should_stop_t_s"] = None if not ss_log.size else round(float(t[ss_log[0]]), 2)
  ref["should_stop_gap_m"] = None if not ss_log.size else round(float(gap[ss_log[0]]), 3)
  ref["should_stop_v_mps"] = None if not ss_log.size else round(float(v[ss_log[0]]), 3)
  dd = _unscale(data["desiredDistanceLog"], scale["dist"])
  ref["desired_distance_standstill_m"] = None if not stopped else round(float(dd[stand[0]]), 3)
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


class OpenLoopPlannerReplay:
  """Logged inputs -> real planner; no ego simulation, no LongControl."""

  def __init__(self, fixture):
    _seed_param_defaults()
    params = Params()
    _apply_header_params(params, fixture["params"])
    self.cp = CarInterface.get_non_essential_params(CAR_MODEL)
    self.cp.openpilotLongitudinalControl = True

    data, scale = fixture["data"], fixture["scale"]
    self.t = np.array(data["t_cs"], dtype=float) / scale["t"]
    self.v_ego = _unscale(data["vEgo"], scale["speed"])
    self.a_ego = _unscale(data["aEgo"], scale["accel"])
    self.v_ego_cluster = _unscale(data["vEgoCluster"], scale["speed"])
    self.v_cruise_kph = _unscale(data["vCruiseKph"], scale["cruise"])
    self.v_clu_ratio = _unscale(data["vCluRatio"], scale["ratio"])
    self.gas = np.array(data["gasLog"], dtype=float)
    self.brake = np.array(data["brakeLog"], dtype=float)
    self.standstill = np.array(data["standstillLog"], dtype=float)
    self.soft_hold = np.array(data["softHoldLog"], dtype=float)
    self.carrot_cruise = np.array(data["carrotCruiseLog"], dtype=float)
    self.force_decel = np.array(data["forceDecelLog"], dtype=float)
    self.desired_curv = _unscale(data["desiredCurvatureLog"], scale["curv"])
    self.curv = _unscale(data["curvatureLog"], scale["curv"])
    self.exp_mode = np.array(data["experimentalModeLog"], dtype=float)
    self.personality = np.array(data["personalityLog"], dtype=float)
    self.lc_state = np.array(data["lcStateLog"], dtype=float)
    self.leads = []
    for name in ("leadOne", "leadTwo"):
      self.leads.append({
        "dRel": _unscale(data[f"{name}_dRel"], scale["dist"]),
        "vRel": _unscale(data[f"{name}_vRel"], scale["speed"]),
        "vLead": _unscale(data[f"{name}_vLead"], scale["speed"]),
        "aLeadK": _unscale(data[f"{name}_aLeadK"], scale["accel"]),
        "status": _unscale(data[f"{name}_status"], 1),
        "radar": _unscale(data[f"{name}_radar"], 1),
        "track": _unscale(data[f"{name}_radarTrackId"], 1),
        "prob": _unscale(data[f"{name}_modelProb"], scale["prob"]),
      })
    self.log = {
      "aTarget": _unscale(data["aTargetLog"], scale["a_target"]),
      "shouldStop": np.array(data["shouldStopLog"], dtype=float),
      "vTargetNow": _unscale(data["vTargetNowLog"], scale["speed"]),
      "tFollow": _unscale(data["tFollow"], scale["tf"]),
      "desiredDistance": _unscale(data["desiredDistanceLog"], scale["dist"]),
      "source": np.array(data["sourceLog"], dtype=float),
    }
    self.planner = LongitudinalPlanner(self.cp, init_v=float(np.nan_to_num(self.v_ego[0])))
    self.carrot = CarrotPlanner()

  @staticmethod
  def _num(value, default=0.0):
    value = float(value)
    return default if not np.isfinite(value) else value

  def _messages(self, i):
    ego_v = self._num(self.v_ego[i])
    ego_a = self._num(self.a_ego[i])
    model = messaging.new_message('modelV2')
    pos = log.XYZTData.new_message()
    pos.x = [float(x) for x in (ego_v + 0.5) * np.array(ModelConstants.T_IDXS)]
    pos.y = [0.0] * len(ModelConstants.T_IDXS)
    pos.z = [0.0] * len(ModelConstants.T_IDXS)
    model.modelV2.position = pos
    vel = log.XYZTData.new_message()
    vel.x = [float(x) for x in (ego_v + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
    vel.x[0] = float(ego_v)
    model.modelV2.velocity = vel
    acc = log.XYZTData.new_message()
    acc.x = [0.0] * len(ModelConstants.T_IDXS)
    model.modelV2.acceleration = acc
    model.modelV2.action.desiredAcceleration = float(ego_a)
    model.modelV2.meta.disengagePredictions.gasPressProbs = [1.0] * 6

    car_state = messaging.new_message('carState')
    car_state.carState.vEgo = ego_v
    car_state.carState.aEgo = ego_a
    car_state.carState.vCluRatio = self._num(self.v_clu_ratio[i], 1.0)
    car_state.carState.standstill = bool(self.standstill[i] > 0.5)
    car_state.carState.vCruise = self._num(self.v_cruise_kph[i])
    car_state.carState.vEgoCluster = self._num(self.v_ego_cluster[i])
    car_state.carState.gasPressed = bool(self.gas[i] > 0.5)
    car_state.carState.brakePressed = bool(self.brake[i] > 0.5)
    car_state.carState.softHoldActive = int(round(self.soft_hold[i]))
    car_state.carState.carrotCruise = int(round(self.carrot_cruise[i]))

    ss = messaging.new_message('selfdriveState')
    ss.selfdriveState.experimentalMode = bool(self.exp_mode[i] > 0.5)
    ss.selfdriveState.personality = int(round(self.personality[i]))

    control = messaging.new_message('controlsState')
    raw_lc = int(round(self.lc_state[i]))
    control.controlsState.longControlState = LC_ENUM[raw_lc] if 0 <= raw_lc < len(LC_ENUM) else LC_ENUM[0]
    control.controlsState.forceDecel = bool(self.force_decel[i] > 0.5)
    control.controlsState.desiredCurvature = self._num(self.desired_curv[i])
    control.controlsState.curvature = self._num(self.curv[i])

    car_control = messaging.new_message('carControl')
    car_control.carControl.orientationNED = [0.0, 0.0, 0.0]

    radar = messaging.new_message('radarState')
    for name, lead in zip(("leadOne", "leadTwo"), self.leads, strict=True):
      msg = log.RadarState.LeadData.new_message()
      status = bool(lead["status"][i] > 0.5)
      msg.status = status
      msg.dRel = float(max(0.0, self._num(lead["dRel"][i], 200.0)))
      msg.vRel = self._num(lead["vRel"][i])
      msg.vLead = self._num(lead["vLead"][i])
      msg.vLeadK = msg.vLead
      msg.aLeadK = self._num(lead["aLeadK"][i])
      msg.aRel = msg.aLeadK - ego_a
      msg.modelProb = self._num(lead["prob"][i])
      msg.radar = bool(lead["radar"][i] > 0.5)
      track = lead["track"][i]
      msg.radarTrackId = -1 if not np.isfinite(track) else int(round(track))
      msg.aLeadTau = float(LEAD_ACCEL_TAU)
      setattr(radar.radarState, name, msg)

    msgs = {
      "carState": car_state.carState,
      "modelV2": model.modelV2,
      "selfdriveState": ss.selfdriveState,
      "controlsState": control.controlsState,
      "carControl": car_control.carControl,
      "radarState": radar.radarState,
    }
    return _FakeSM(msgs, int(i * DT_MDL * 1e9))

  def run(self):
    planner = self.planner
    rows = []
    n = len(self.t)
    for i in range(n):
      planner.update(self._messages(i), self.carrot)
      margin_row = planner.mpc.lead_gap_margins[0]
      rows.append({
        "aTarget": float(planner.output_a_target),
        "shouldStop": bool(planner.output_should_stop),
        "vTargetNow": float(planner.output_v_target_now),
        "tFollow": float(planner.mpc.t_follow),
        "desiredDistance": float(planner.mpc.desired_distance),
        "source": SOURCE_IDS.get(str(planner.mpc.source), -1),
        "marginLead0": float(margin_row[0]),
        "marginLead1": float(margin_row[1]),
        "xObstacle0": float(planner.mpc.params[0, 2]),
        "pTFollow0": float(planner.mpc.params[0, 4]),
        "pStopDistance0": float(planner.mpc.params[0, 7]),
        "pDangerTf0": float(planner.mpc.params[0, 8]),
      })
    return rows


def _open_loop_metrics(fixture, replay, rows):
  t = replay.t
  log_at = replay.log["aTarget"]
  log_ss = replay.log["shouldStop"]
  sim_at = np.array([r["aTarget"] for r in rows], dtype=float)
  sim_ss = np.array([r["shouldStop"] for r in rows], dtype=bool)
  sim_src = np.array([r["source"] for r in rows], dtype=float)
  sim_tf = np.array([r["tFollow"] for r in rows], dtype=float)
  sim_dd = np.array([r["desiredDistance"] for r in rows], dtype=float)
  log_src = replay.log["source"]
  log_tf = replay.log["tFollow"]
  log_dd = replay.log["desiredDistance"]

  valid = np.isfinite(log_at) & np.isfinite(sim_at)
  mask = valid & (t >= OPEN_LOOP_WINDOW_S)
  delta = np.abs(sim_at - log_at)
  rms = float(np.sqrt(np.mean(delta[mask] ** 2)))
  dmax = float(np.max(delta[mask]))
  imax = int(np.flatnonzero(mask)[int(np.argmax(delta[mask]))])
  order = np.argsort(np.where(mask, delta, -1.0))[::-1][:3]

  def divergence(i):
    i = int(i)
    return {
      "t_s": round(float(t[i]), 2),
      "aTarget_sim": round(float(sim_at[i]), 4),
      "aTarget_log": round(float(log_at[i]), 4),
      "tFollow_sim": round(float(sim_tf[i]), 4),
      "tFollow_log": round(float(log_tf[i]), 4),
      "desiredDistance_sim": round(float(sim_dd[i]), 3),
      "desiredDistance_log": round(float(log_dd[i]), 3),
      "source_sim": SOURCE_NAMES.get(int(sim_src[i]), str(sim_src[i])),
      "source_log": SOURCE_NAMES.get(int(log_src[i]), str(log_src[i])),
      "marginLead0": round(rows[i]["marginLead0"], 3),
      "marginLead1": round(rows[i]["marginLead1"], 3),
      "xObstacle0": round(rows[i]["xObstacle0"], 3),
      "pTFollow0": round(rows[i]["pTFollow0"], 4),
      "pStopDistance0": round(rows[i]["pStopDistance0"], 3),
      "pDangerTf0": round(rows[i]["pDangerTf0"], 4),
    }

  sim_fire = np.flatnonzero(sim_ss)
  log_fire = np.flatnonzero(log_ss > 0.5)
  gap_log = _unscale(fixture["data"]["leadOne_dRel"], fixture["scale"]["dist"])
  v_log = _unscale(fixture["data"]["vEgo"], fixture["scale"]["speed"])
  metrics: dict = {
    "n_steps": int(len(t)),
    "window_s": [float(OPEN_LOOP_WINDOW_S), round(float(t[-1]), 2)],
    "rms_delta_aTarget": round(rms, 4),
    "max_delta_aTarget": round(dmax, 4),
    "max_delta_t_s": round(float(t[imax]), 2),
    "shouldStop_first_sim_s": None if not sim_fire.size else round(float(t[sim_fire[0]]), 2),
    "shouldStop_first_log_s": None if not log_fire.size else round(float(t[log_fire[0]]), 2),
    "shouldStop_sim_gap_m": None if not sim_fire.size else round(float(gap_log[sim_fire[0]]), 3),
    "shouldStop_sim_v_mps": None if not sim_fire.size else round(float(v_log[sim_fire[0]]), 3),
    "shouldStop_mismatch_steps": int(np.sum(sim_ss != (log_ss > 0.5))),
    "source_match_frac": round(float(np.mean(sim_src[mask] == log_src[mask])), 4),
    "tFollow_match_frac_0_05": round(float(np.mean(np.abs(sim_tf[mask] - log_tf[mask]) <= 0.05)), 4),
    "desiredDistance_rms": round(float(np.sqrt(np.mean((sim_dd[mask] - log_dd[mask]) ** 2))), 3),
    "top_divergences": [divergence(i) for i in order],
    "final_sim": {
      "tFollow": round(float(sim_tf[-1]), 4),
      "desiredDistance": round(float(sim_dd[-1]), 3),
      "source": SOURCE_NAMES.get(int(sim_src[-1]), str(sim_src[-1])),
      "pStopDistance0": round(rows[-1]["pStopDistance0"], 3),
    },
    "carrot_end": {
      "butter_stop": bool(getattr(replay.carrot, "butter_stop", False)),
      "stop_distance": round(float(getattr(replay.carrot, "stop_distance", -1.0)), 3),
      "myDrivingMode": int(getattr(replay.carrot, "myDrivingMode", -1).value
                          if hasattr(getattr(replay.carrot, "myDrivingMode", None), "value")
                          else getattr(replay.carrot, "myDrivingMode", -1)),
      "t_follow_last": round(float(getattr(replay.carrot, "t_follow_last", -1.0)), 4),
      "t_follow_danger": round(float(getattr(replay.carrot, "t_follow_danger", -1.0)), 4),
    },
  }
  return metrics


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
  desired_distance = np.array(run["desired_distance"], dtype=float)

  stand = np.flatnonzero(v < 0.05)
  below7 = np.flatnonzero(v < 7.0 / 3.6)
  stopped = bool(stand.size)
  metrics: dict = {
    "stopped": stopped,
    "min_gap_m": None if not active.any() else round(float(np.min(gap[active])), 3),
    "collision": bool(np.any(gap[active] <= 0.0)),
    "desired_distance_standstill_m": None if not stopped else round(float(desired_distance[stand[0]]), 3),
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


def _run_b1(variant, mode):
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
    run = FarLeadLoop(header, FixtureProfile(fixture), plant=(mode != "ideal"),
                      scc_stop=(mode == "scc")).run()
  return _b1_metrics(run)


def _run_stop_release(butter, release_tau, mode):
  fixture = load_fixture("b1")
  header = dict(fixture["params"])
  if butter == 0:
    header["ButterStop"] = 0
  saved = longitudinal_braking_gap.STOPPED_LEAD_RELEASE_S
  longitudinal_braking_gap.STOPPED_LEAD_RELEASE_S = release_tau
  try:
    run = FarLeadLoop(header, FixtureProfile(fixture), plant=(mode != "ideal"),
                      scc_stop=(mode == "scc")).run()
  finally:
    longitudinal_braking_gap.STOPPED_LEAD_RELEASE_S = saved
  return _b1_metrics(run)


def test_butter_real_stop_release():
  for butter in (0, 1):
    for label, tau in (("M0", 8.0), ("M1", 1.0)):
      for mode in ("ideal", "scc"):
        key = f"{'B1' if butter else 'B0'}_{label}_plant_{mode}"
        metrics = _run_stop_release(butter, tau, mode)
        RESULTS["stop_release"][key] = metrics
        assert not metrics["collision"], f"{key} collides (min gap {metrics['min_gap_m']})"


def test_butter_real_b1():
  fixture = load_fixture("b1")
  assert fixture["grid"]["n"] > 100
  RESULTS["log_reference"] = _log_reference(fixture)
  for variant in VARIANTS:
    for mode in PLANT_MODES:
      key = f"{variant}_plant_{mode}"
      metrics = _run_b1(variant, mode)
      RESULTS["b1"][key] = metrics
      assert not metrics["collision"], f"{key} collides (min gap {metrics['min_gap_m']})"

  sim = RESULTS["b1"]["B1_plant_scc"]
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
  print(f"BUTTER-REAL fidelity (B1 scc): final gap {fidelity['final_gap_sim_m']} ({gap_ok}), below-7 time {fidelity['below7_time_sim_s']} ({time_ok})")


def test_butter_real_open_loop_planner():
  fixture = load_fixture("b1")
  replay = OpenLoopPlannerReplay(fixture)
  rows = replay.run()
  metrics = _open_loop_metrics(fixture, replay, rows)
  metrics["fixture_params"] = {k: fixture["params"][k] for k in (
    "LongActuatorDelay", "VEgoStopping", "StopDistanceCarrot", "ButterStop", "SpeedTFFactor",
    "TFollowGap3", "TFollowDecelBoost", "MyDrivingMode", "LongitudinalPersonality",
    "TrafficStopDistanceAdjust", "DynamicTFollowCL", "DynamicTFollowLC", "CruiseCoastingPercent",
  ) if k in fixture["params"]}
  RESULTS["open_loop"] = metrics
  ol1 = f"BUTTER-REAL open-loop: RMS |d aTarget| {metrics['rms_delta_aTarget']} m/s^2, max {metrics['max_delta_aTarget']} at t={metrics['max_delta_t_s']} s"
  ol2 = f"BUTTER-REAL open-loop shouldStop: sim {metrics['shouldStop_first_sim_s']} s vs log {metrics['shouldStop_first_log_s']} s"
  print(ol1)
  print(ol2)


def _scenario_metrics(logs, name):
  t, v = logs[:, 0], logs[:, 1]
  a = logs[:, 10]
  gaps = logs[:, 4] - logs[:, 3]
  stand = np.flatnonzero(v < 0.05)
  below7 = np.flatnonzero(v < 7.0 / 3.6)
  stopped = bool(stand.size)
  entry = {
    "scenario": name,
    "stopped": stopped,
    "final_gap_m": round(float(gaps[stand[0]]), 3) if stopped else None,
    "min_gap_m": round(float(np.min(gaps)), 3),
    "collision": bool(np.min(gaps) < COLLISION_GAP_M),
    "fcw_steps": int(np.sum(logs[:, 12] > 0.5)),
  }
  if stopped:
    i = int(stand[0])
    window = (t >= t[i] - 0.5) & (t <= t[i])
    entry["end_decel_mean_abs"] = round(float(np.mean(np.abs(a[window]))), 4) if window.any() else None
    entry["crawl_time_s"] = round(float(t[i] - t[below7[0]]), 2) if below7.size and below7[0] < i else None
  else:
    entry["end_decel_mean_abs"] = None
    entry["crawl_time_s"] = None
  return entry


def test_butter_real_scenarios_plant_on():
  rows = []
  saved = longitudinal_braking_gap.STOPPED_LEAD_RELEASE_S
  try:
    for group, maneuver in _CASES:
      if group not in ("a", "b", "c"):
        continue
      for butter in (0, 1):
        for label, tau in (("M0", 8.0), ("M1", 1.0)):
          longitudinal_braking_gap.STOPPED_LEAD_RELEASE_S = tau
          logs = ButterStopLoop(maneuver, butter=butter, plant=True).run()
          entry = _scenario_metrics(logs, maneuver.title)
          entry["group"] = group
          entry["butter"] = butter
          entry["release"] = label
          rows.append(entry)
          assert not entry["collision"], f"{maneuver.title} butter={butter} {label} collides (min gap {entry['min_gap_m']})"
  finally:
    longitudinal_braking_gap.STOPPED_LEAD_RELEASE_S = saved
  RESULTS["scenarios_plant_on"] = rows


# --- Task 9: offset-cap sweep + stock OFF reference ------------------------

# (label, FLOOR_SPEED_CAP_S, FLOOR_CAP_OFFSET_V, memory_off)
_OFFSET_SWEEP = (
  ("OFF", None, None, True),
  ("NOCAP", float('inf'), 0.0, False),
  ("K2.0/V3", 2.0, 3.0, False),
  ("K2.0/V5", 2.0, 5.0, False),
  ("K2.0/V8", 2.0, 8.0, False),
  ("K1.5/V5", 1.5, 5.0, False),
)
_OFFSET_REFERENCES = ("OFF", "NOCAP")
_CAP_PLANTS = ("ideal", "scc")
_CAP_SWEEP_SCENARIOS = [(g, m) for g, m in _CASES if g in ("a", "b", "c")]
_CAP_LEAD_DECEL = (
  "steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2",
  "steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2",
  "steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2",
)
_OFFSET_SWEEP_CRITERIA = (
  "a) b1 ideal: B0 final gap <= 6.5 m and desiredDistance at standstill <= 6.5 m",
  "b) |B1 - B0| final gap <= 0.5 m in b1 and all 9 scenarios, both plants",
  "c) end decel (ideal): B1 <= max(0.20, B0 - 0.25) for a-group (except stopped-60), "
  "approach and b1; for braking-lead and stopped-60: B1 <= OFF-row B1 + 0.05",
  "d) crawl (below-7 km/h time before standstill) in b1 and approach-25/20, both plants, "
  "B0 and B1: <= OFF-row of the same ButterStop setting + 1.0 s",
  "e) safety: 0 collisions, 0 FCW anywhere; lead-decel-1/2/3 min gap >= OFF-row value - 0.3 m, "
  "both plants, B0 and B1",
  "f) existing ButterStop-suite criteria re-evaluated in-sweep for the 9 overlapping scenarios "
  "(ideal plant, B0 vs B1, same checks as _case_failures); the s-scenarios and the remaining "
  "suites are verified in the final CI round at the chosen defaults",
)


def _run_b1_offset(row, butter, plant, telemetry=None):
  _label, cap_s, offset_v, memory_off = row
  fixture = load_fixture("b1")
  header = dict(fixture["params"])
  if butter == 0:
    header["ButterStop"] = 0
  saved_cap = longitudinal_braking_gap.FLOOR_SPEED_CAP_S
  saved_offset = longitudinal_braking_gap.FLOOR_CAP_OFFSET_V
  saved_update = longitudinal_braking_gap.BrakingGapState.update
  if memory_off:
    # OFF = the braking-gap memory fully disabled: the planner exposes the
    # BrakingGapState.update(enabled=...) kwarg, so force it False. Age resets
    # every tick, the floor never captures and comfort_relief never accrues.
    def disabled_update(self, **kwargs):
      kwargs["enabled"] = False
      return saved_update(self, **kwargs)
    longitudinal_braking_gap.BrakingGapState.update = disabled_update
  else:
    longitudinal_braking_gap.FLOOR_SPEED_CAP_S = cap_s
    longitudinal_braking_gap.FLOOR_CAP_OFFSET_V = offset_v
  try:
    run = FarLeadLoop(header, FixtureProfile(fixture), plant=(plant != "ideal"),
                      scc_stop=(plant == "scc"), telemetry=telemetry).run()
  finally:
    longitudinal_braking_gap.FLOOR_SPEED_CAP_S = saved_cap
    longitudinal_braking_gap.FLOOR_CAP_OFFSET_V = saved_offset
    longitudinal_braking_gap.BrakingGapState.update = saved_update
  return _b1_metrics(run)


def _run_scenario_offset(row, maneuver, butter, plant):
  _label, cap_s, offset_v, memory_off = row
  saved_cap = longitudinal_braking_gap.FLOOR_SPEED_CAP_S
  saved_offset = longitudinal_braking_gap.FLOOR_CAP_OFFSET_V
  saved_update = longitudinal_braking_gap.BrakingGapState.update
  if memory_off:
    def disabled_update(self, **kwargs):
      kwargs["enabled"] = False
      return saved_update(self, **kwargs)
    longitudinal_braking_gap.BrakingGapState.update = disabled_update
  else:
    longitudinal_braking_gap.FLOOR_SPEED_CAP_S = cap_s
    longitudinal_braking_gap.FLOOR_CAP_OFFSET_V = offset_v
  try:
    logs = ButterStopLoop(maneuver, butter=butter, plant=(plant != "ideal"),
                          scc_stop=(plant == "scc")).run()
  finally:
    longitudinal_braking_gap.FLOOR_SPEED_CAP_S = saved_cap
    longitudinal_braking_gap.FLOOR_CAP_OFFSET_V = saved_offset
    longitudinal_braking_gap.BrakingGapState.update = saved_update
  return logs


def _cap_gap_pair(label, b0, b1m):
  g0, g1 = b0.get("final_gap_m"), b1m.get("final_gap_m")
  if g0 is None or g1 is None:
    return [f"b) {label}: final gap missing (B0 {g0}, B1 {g1})"]
  if abs(g1 - g0) > 0.5 + 1e-9:
    return [f"b) {label}: |B1 - B0| {abs(g1 - g0):.3f} > 0.5 m (B0 {g0}, B1 {g1})"]
  return []


def _collect_offset_row(row):
  b1: dict = {}
  scen: dict = {}
  scen_logs: dict = {}
  for plant in _CAP_PLANTS:
    for butter in (0, 1):
      b1[(plant, butter)] = _run_b1_offset(row, butter, plant)
  for _group, maneuver in _CAP_SWEEP_SCENARIOS:
    for plant in _CAP_PLANTS:
      for butter in (0, 1):
        logs = _run_scenario_offset(row, maneuver, butter, plant)
        if plant == "ideal":
          scen_logs[(maneuver.title, plant, butter)] = logs
        scen[(maneuver.title, plant, butter)] = _scenario_metrics(logs, maneuver.title)
  suite_fails = []
  for group, maneuver in _CAP_SWEEP_SCENARIOS:
    off_metrics = _butter_metrics(scen_logs[(maneuver.title, "ideal", 0)], maneuver.title, 0)
    on_metrics = _butter_metrics(scen_logs[(maneuver.title, "ideal", 1)], maneuver.title, 1)
    for msg in _butter_case_failures(group, off_metrics, on_metrics):
      suite_fails.append(f"f) ideal plant: {msg}")
  return {"b1": b1, "scen": scen, "suite_fails": suite_fails}


def _offset_combo_failures(row, data, off):
  """Criteria a-e for one sweep row; references (OFF/NOCAP) carry suite fails only."""
  label = row[0]
  fails = [f"{label}: {msg}" for msg in data["suite_fails"]]
  b1, scen = data["b1"], data["scen"]
  off_b1, off_scen = off["b1"], off["scen"]

  def fail(msg):
    fails.append(f"{label}: {msg}")

  # a) b1 ideal B0 stops at the normal distance.
  m = b1[("ideal", 0)]
  if m["final_gap_m"] is None or m["final_gap_m"] > 6.5 + 1e-9:
    fail(f"a) b1 ideal B0 final gap {m['final_gap_m']} > 6.5 m")
  dd = m["desired_distance_standstill_m"]
  if dd is None or dd > 6.5 + 1e-9:
    fail(f"a) b1 ideal B0 desiredDistance at standstill {dd} > 6.5 m")

  # b) ButterStop must not change the stop point: b1 + all 9 scenarios.
  for plant in _CAP_PLANTS:
    for msg in _cap_gap_pair(f"b1 {plant}", b1[(plant, 0)], b1[(plant, 1)]):
      fail(msg)
    for _group, maneuver in _CAP_SWEEP_SCENARIOS:
      for msg in _cap_gap_pair(f"{maneuver.title} {plant}",
                               scen[(maneuver.title, plant, 0)], scen[(maneuver.title, plant, 1)]):
        fail(msg)

  # c) End decel, ideal plant: normal rule vs this row's B0 for b1, the
  #    a-group (except stopped-60) and the approach scenarios; braking-lead
  #    and stopped-60 only must stay within OFF-row B1 + 0.05.
  a0 = b1[("ideal", 0)].get("mean_accel_last_0.5s")
  a1 = b1[("ideal", 1)].get("mean_accel_last_0.5s")
  if a0 is None or a1 is None:
    fail(f"c) b1 ideal: end decel missing (B0 {a0}, B1 {a1})")
  elif abs(a1) > max(0.20, abs(a0) - 0.25) + 1e-9:
    fail(f"c) b1 ideal: B1 end decel {abs(a1)} > max(0.20, B0 {abs(a0)} - 0.25)")
  for group, maneuver in _CAP_SWEEP_SCENARIOS:
    if group == "c":
      continue
    d0 = scen[(maneuver.title, "ideal", 0)].get("end_decel_mean_abs")
    d1 = scen[(maneuver.title, "ideal", 1)].get("end_decel_mean_abs")
    if d1 is None:
      fail(f"c) {maneuver.title} ideal: end decel missing")
    elif group == "b" or "60 km/h" in maneuver.title:
      ref = off_scen[(maneuver.title, "ideal", 1)].get("end_decel_mean_abs")
      if ref is None or d1 > ref + 0.05 + 1e-9:
        fail(f"c) {maneuver.title} ideal: B1 end decel {d1} > OFF-row B1 {ref} + 0.05")
    else:
      if d0 is None:
        fail(f"c) {maneuver.title} ideal: B0 end decel missing")
      elif d1 > max(0.20, d0 - 0.25) + 1e-9:
        fail(f"c) {maneuver.title} ideal: B1 end decel {d1} > max(0.20, B0 {d0} - 0.25)")

  # d) No crawl increase vs the OFF row OF THE SAME ButterStop setting: the
  #    taper itself adds B1 crawl vs B0 even memory-off, so B0 is judged
  #    against OFF B0 and B1 against OFF B1 (task-9 review ruling).
  for plant in _CAP_PLANTS:
    for butter in (0, 1):
      ref = off_b1[(plant, butter)].get("crawl_time_s")
      c = b1[(plant, butter)].get("crawl_time_s")
      if c is None or ref is None:
        fail(f"d) b1 {plant} B{butter}: crawl time missing ({c} vs OFF {ref})")
      elif c > ref + 1.0 + 1e-9:
        fail(f"d) b1 {plant} B{butter}: crawl {c} > OFF B{butter} {ref} + 1.0 s")
    for _group, maneuver in _CAP_SWEEP_SCENARIOS:
      if "approach stopped car" not in maneuver.title:
        continue
      for butter in (0, 1):
        ref = off_scen[(maneuver.title, plant, butter)].get("crawl_time_s")
        c = scen[(maneuver.title, plant, butter)].get("crawl_time_s")
        if c is None or ref is None:
          fail(f"d) {maneuver.title} {plant} B{butter}: crawl time missing ({c} vs OFF {ref})")
        elif c > ref + 1.0 + 1e-9:
          fail(f"d) {maneuver.title} {plant} B{butter}: crawl {c} > OFF B{butter} {ref} + 1.0 s")

  # e) Safety: zero collisions and zero FCW anywhere; lead-decel min gaps
  #    within OFF - 0.3 m, both plants, B0 and B1.
  for plant in _CAP_PLANTS:
    for butter in (0, 1):
      metrics = b1[(plant, butter)]
      if metrics.get("collision"):
        fail(f"e) b1 {plant} B{butter}: collision (min gap {metrics['min_gap_m']})")
      if metrics.get("fcw_steps", 0):
        fail(f"e) b1 {plant} B{butter}: FCW {metrics['fcw_steps']} planner ticks")
  for _group, maneuver in _CAP_SWEEP_SCENARIOS:
    for plant in _CAP_PLANTS:
      for butter in (0, 1):
        metrics = scen[(maneuver.title, plant, butter)]
        if metrics.get("collision"):
          fail(f"e) {maneuver.title} {plant} B{butter}: collision (min gap {metrics['min_gap_m']})")
        if metrics.get("fcw_steps", 0):
          fail(f"e) {maneuver.title} {plant} B{butter}: FCW {metrics['fcw_steps']} planner ticks")
    if maneuver.title not in _CAP_LEAD_DECEL:
      continue
    for plant in _CAP_PLANTS:
      for butter in (0, 1):
        ref = off_scen[(maneuver.title, plant, butter)]["min_gap_m"]
        mg = scen[(maneuver.title, plant, butter)]["min_gap_m"]
        if mg < ref - 0.3 - 1e-9:
          fail(f"e) {maneuver.title} {plant} B{butter}: min gap {mg} < OFF {ref} - 0.3")
  return fails


def _row_payload(row, data, fails):
  label, cap_s, offset_v, memory_off = row
  return {
    "label": label,
    "cap_s": None if cap_s is None or not math.isfinite(cap_s) else cap_s,
    "no_cap": cap_s == float('inf'),
    "offset_v": offset_v,
    "memory_off": memory_off,
    "reference": label in _OFFSET_REFERENCES,
    "all_pass": not fails,
    "failures": fails,
    "b1": {f"{plant}_{butter}": data["b1"][(plant, butter)]
           for plant in _CAP_PLANTS for butter in (0, 1)},
    "scenarios": {maneuver.title: {f"{plant}_{butter}": data["scen"][(maneuver.title, plant, butter)]
                                   for plant in _CAP_PLANTS for butter in (0, 1)}
                  for _group, maneuver in _CAP_SWEEP_SCENARIOS},
  }


def test_butter_real_offset_sweep():
  out_path = os.environ.get("BUTTER_REAL_SWEEP_OUT")
  if not out_path:
    pytest.skip("offset sweep runs only when BUTTER_REAL_SWEEP_OUT is set")
  titles = {m.title for _, m in _CASES}
  missing = [t for t in _CAP_LEAD_DECEL if t not in titles]
  assert not missing, f"lead-decel titles missing from _CASES: {missing}"

  row_data: dict = {}
  for row in _OFFSET_SWEEP:
    row_data[row[0]] = _collect_offset_row(row)
    print(f"OFFSET-SWEEP {row[0]}: collected")
  off = row_data["OFF"]

  rows = []
  for row in _OFFSET_SWEEP:
    fails = _offset_combo_failures(row, row_data[row[0]], off)
    rows.append(_row_payload(row, row_data[row[0]], fails))
    print(f"OFFSET-SWEEP {row[0]}: pass={not fails} fails={len(fails)}")

  default_row = next((r for r in rows if not r["reference"]
                      and r["cap_s"] == longitudinal_braking_gap.FLOOR_SPEED_CAP_S
                      and r["offset_v"] == longitudinal_braking_gap.FLOOR_CAP_OFFSET_V), None)
  assert default_row is not None, "default pair row missing from the sweep"
  assert default_row["all_pass"], (
    f"default pair (FLOOR_SPEED_CAP_S {longitudinal_braking_gap.FLOOR_SPEED_CAP_S}, "
    f"FLOOR_CAP_OFFSET_V {longitudinal_braking_gap.FLOOR_CAP_OFFSET_V}) must pass a-f: "
    + "; ".join(default_row["failures"][:5]))

  candidates = [r for r in rows if not r["reference"]]

  def pick_key(r):
    crawl = r["b1"]["ideal_1"]["crawl_time_s"]
    return (float('inf') if crawl is None else crawl, r["offset_v"])

  passing = [r for r in candidates if r["all_pass"]]
  chosen = min(passing, key=pick_key) if passing else None
  closest = sorted(candidates, key=lambda r: (len(r["failures"]), r["offset_v"]))

  best = chosen if chosen is not None else (closest[0] if closest else None)
  telemetry: dict = {}
  if best is not None:
    row = next(r for r in _OFFSET_SWEEP if r[0] == best["label"])
    for butter in (0, 1):
      trace: dict = {}
      metrics = _run_b1_offset(row, butter, "ideal", telemetry=trace)
      telemetry[f"B{butter}"] = {"label": best["label"], "metrics": metrics, "trace": trace}
  RESULTS["b1_telemetry"] = telemetry

  if chosen is not None:
    print(f"OFFSET-PICK: {chosen['label']}")
  else:
    print("OFFSET-PICK: none passed; per-candidate failures:")
    for r in closest:
      print(f"  {r['label']}: {len(r['failures'])} failures")

  table = {
    "criteria": list(_OFFSET_SWEEP_CRITERIA),
    "rows": rows,
    "chosen": chosen,
    "closest": closest,
    "off_reference": off is not None,
    "b1_telemetry": telemetry,
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  collisions = []
  for row in rows:
    for label, metrics in list(row["b1"].items()) + [(f"{title} {label}", m)
                                                     for title, runs in row["scenarios"].items()
                                                     for label, m in runs.items()]:
      if metrics.get("collision"):
        collisions.append(f"{row['label']} {label}")
  assert not collisions, "offset sweep collision: " + "; ".join(collisions[:5])


def test_write_butter_real_table():
  out_path = os.environ.get("BUTTER_REAL_OUT", "/tmp/butter_real.json")
  table = {
    "log_reference": RESULTS["log_reference"],
    "b1": RESULTS["b1"],
    "b1_telemetry": RESULTS["b1_telemetry"],
    "stop_release": RESULTS["stop_release"],
    "open_loop": RESULTS["open_loop"],
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
