"""Far-lead closed-loop replay: real-drive fixtures + synthetic safety maneuvers.

Runs where the native longitudinal MPC solver exists (CI build_release), the
same environment as the longitudinal maneuver tests. The loop is the real
LongitudinalPlanner + real acados MPC + real CarrotPlanner + real LongControl
at 100 Hz with the planner stepping at 20 Hz (plant.py is not used: same
reasons as test_butter_stop_maneuvers.py).

Replay episodes (data/far_lead/e1-e6,s1.json) were extracted from the user's
2019 Palisade rlogs (route 0000043f/0000042e/0000042b/00000438) where the car
braked hard on a slower lead that was still far away. Lead world position is
x_L(t) = s_ego_log(t) + dRel_log(t); in sim dRel = x_L(t) - s_sim(t),
vRel = vLead_log - v_sim, and status/radar/trackId/modelProb come from the log
per step, so the logged radar flicker is reproduced exactly.

Variants (module switches read at call time; restored after each run):
  V0 = DANGER_STATIC_T_FOLLOW=False, FAR_LEAD_APPROACH=False (pre-W112)
  V1 = FAR_LEAD_APPROACH=False (static danger TF only)
  V3 = both on (static danger TF + smooth-approach relief)
Maneuvers run V0@SpeedTFFactor10 (reference), V0@15 and V3@15.

PASS criteria (asserted):
  Fidelity: |V0 peak - log peak| <= 0.4 in >= 4/6 AND V0 never gentler than
  log - 0.4 in any comfort episode.
  Comfort e1-e6 (V3 vs V0): every episode V3 peak >= V0 peak - 0.05; mean
  peak-decel reduction >= 35%; every episode V3 min time-gap >= min(V0
  min time-gap, 1.0 s) - 0.05; no collision; FCW V3 <= V0; every episode
  V3 onset jerk >= V0 onset jerk - 0.1.
  Safety s1: V3 min gap >= V0 - 0.5 m; V3 min TTC >= min(V0, 3.0 s) - 0.1;
  no collision; FCW V3 <= V0.
  Maneuvers m1-m6: no collision for ANY variant; final gap >= 2.0 m when the
  lead stops; V3@15 min gap >= V0@10 min gap - 0.5 m; FCW V3@15 <= V0@10;
  m6: V3@15 peak decel >= V0@10 peak decel - 0.5.
Output JSON to FAR_LEAD_SIM_OUT (default /tmp/far_lead_sim.json).
Simulation only; no vehicle validation.
"""
import json
import os
import types

import numpy as np
import pytest

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.radar_constants import LEAD_ACCEL_TAU
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib import far_lead_approach
from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR

CAR_MODEL = CAR.HYUNDAI_PALISADE
PLANNER_PERIOD = 5  # 100 Hz loop, 20 Hz planner/model, as in the car
SENTINEL = -32767
EPISODES = ("e1", "e2", "e3", "e4", "e5", "e6")
ALL_EPISODES = EPISODES + ("s1",)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "far_lead")

RESULTS: dict = {"episodes": {}, "maneuvers": {}, "fidelity": {}, "report": {}}


def _seed_param_defaults():
  """Each test runs in a fresh OpenpilotPrefix; seed catalog defaults like manager.set_default_params."""
  params = Params()
  for key in params.all_keys():
    default = params.get_default_value(key)
    if default is None:
      continue
    try:
      params.put(key, default)
    except (UnknownKeyName, TypeError, ValueError):
      pass


def _apply_header_params(params, header):
  for name, value in header.items():
    try:
      if isinstance(value, bool):
        params.put_bool(name, value)
      elif isinstance(value, int):
        params.put_int(name, value)
      elif isinstance(value, float):
        params.put_float(name, value)
    except (UnknownKeyName, TypeError, ValueError):
      pass


class _Flags(dict):
  def __missing__(self, key):
    return False


class _FakeSM:
  def __init__(self, msgs, mono_ns):
    self._msgs = dict(msgs)
    self.logMonoTime = _Flags(dict.fromkeys(self._msgs, mono_ns))
    self.valid = _Flags(dict.fromkeys(self._msgs, True))
    self.alive = _Flags(dict.fromkeys(self._msgs, True))
    self.seen = _Flags(dict.fromkeys(self._msgs, True))
    self.recv_time = _Flags(dict.fromkeys(self._msgs, 0.0))

  def __getitem__(self, key):
    return self._msgs[key]

  def all_checks(self, *args, **kwargs):
    return True


def _unscale(values, scale):
  arr = np.array(values, dtype=float)
  arr[arr <= SENTINEL] = np.nan
  return arr / scale


def load_fixture(eid):
  with open(os.path.join(DATA_DIR, eid + ".json"), encoding="utf-8") as handle:
    return json.load(handle)


class FixtureProfile:
  """Episode trajectory from the rlog fixture; discrete fields step-held per grid point."""

  def __init__(self, fixture):
    self.dt = float(fixture["grid"]["dt"])
    self.n_steps = int(fixture["grid"]["n"])
    self.duration = self.n_steps * self.dt
    data, scale = fixture["data"], fixture["scale"]
    self.t = np.arange(self.n_steps) * self.dt
    v_ego = _unscale(data["vEgo"], scale["speed"])
    a_ego = _unscale(data["aEgo"], scale["accel"])
    self.initial_speed = float(np.nan_to_num(v_ego[0]))
    self.initial_accel = float(np.nan_to_num(a_ego[0]))
    self.vc_kph = _unscale(data["vCruiseKph"], scale["cruise"])
    s_ego = _unscale(data["sEgo"], scale["dist"])
    self.leads = []
    for name in ("leadOne", "leadTwo"):
      self.leads.append({
        "x": s_ego + np.nan_to_num(_unscale(data[f"{name}_dRel"], scale["dist"]), nan=0.0),
        "v": _unscale(data[f"{name}_vLead"], scale["speed"]),
        "a": _unscale(data[f"{name}_aLeadK"], scale["accel"]),
        "status": _unscale(data[f"{name}_status"], 1),
        "radar": _unscale(data[f"{name}_radar"], 1),
        "track": _unscale(data[f"{name}_radarTrackId"], 1),
        "prob": _unscale(data[f"{name}_modelProb"], scale["prob"]),
      })

  def _index(self, t):
    return min(max(int(round(t / self.dt)), 0), self.n_steps - 1)

  def sample(self, t, _s_ego):
    i = self._index(t)
    specs = []
    for lead in self.leads:
      track = -1 if not np.isfinite(lead["track"][i]) else int(round(lead["track"][i]))
      specs.append((
        float(np.interp(t, self.t, lead["x"])),
        float(np.nan_to_num(lead["v"][i])),
        float(np.nan_to_num(lead["a"][i])),
        bool(lead["status"][i] > 0.5),
        bool(lead["radar"][i] > 0.5),
        track,
        float(np.nan_to_num(lead["prob"][i])),
      ))
    return specs

  def reset(self):
    pass

  def v_cruise_kph(self, t):
    return float(np.nan_to_num(self.vc_kph[self._index(t)]))


class _LeadSpec:
  """Analytic lead for the synthetic maneuvers (brake-to-stop or cut-in)."""

  def __init__(self, x0=0.0, v0=0.0, track_id=7, brake_t=None, brake_a=0.0,
               cut_in_t=None, cut_in_gap=0.0):
    self.x0 = float(x0)
    self.v0 = float(v0)
    self.track_id = int(track_id)
    self.brake_t = brake_t
    self.brake_a = float(brake_a)
    self.cut_in_t = cut_in_t
    self.cut_in_gap = float(cut_in_gap)
    self._cut_base = None

  def state(self, t, s_ego):
    if self.cut_in_t is not None:
      if t < self.cut_in_t:
        return None
      if self._cut_base is None:
        self._cut_base = s_ego + self.cut_in_gap
      return (self._cut_base + self.v0 * (t - self.cut_in_t), self.v0, 0.0)
    if self.brake_t is not None and t > self.brake_t:
      dt = t - self.brake_t
      xb = self.x0 + self.v0 * self.brake_t
      stop_t = self.v0 / abs(self.brake_a) if self.brake_a < 0.0 else 0.0
      if dt <= stop_t:
        x = xb + self.v0 * dt + 0.5 * self.brake_a * dt * dt
        return (x, max(0.0, self.v0 + self.brake_a * dt), self.brake_a)
      x = xb + 0.5 * self.v0 * stop_t
      return (x, 0.0, 0.0)
    return (self.x0 + self.v0 * t, self.v0, 0.0)


class _BrakeThenHoldSpec:
  """m6: constant speed, then brake at brake_t for brake_dur, then constant."""

  def __init__(self, x0, v0, brake_t, brake_a, brake_dur, track_id=7):
    self.x0 = float(x0)
    self.v0 = float(v0)
    self.brake_t = float(brake_t)
    self.brake_a = float(brake_a)
    self.brake_dur = float(brake_dur)
    self.track_id = int(track_id)

  def state(self, t, _s_ego):
    if t <= self.brake_t:
      return (self.x0 + self.v0 * t, self.v0, 0.0)
    dt = t - self.brake_t
    xb = self.x0 + self.v0 * self.brake_t
    if dt <= self.brake_dur:
      x = xb + self.v0 * dt + 0.5 * self.brake_a * dt * dt
      return (x, self.v0 + self.brake_a * dt, self.brake_a)
    v_end = self.v0 + self.brake_a * self.brake_dur
    x_end = xb + self.v0 * self.brake_dur + 0.5 * self.brake_a * self.brake_dur ** 2
    return (x_end + v_end * (dt - self.brake_dur), v_end, 0.0)


class SyntheticProfile:
  def __init__(self, ego_kph, cruise_kph, lead1, lead2=None, duration=40.0):
    self.initial_speed = ego_kph / 3.6
    self.initial_accel = 0.0
    self.cruise_kph = float(cruise_kph)
    self.lead_specs = (lead1, lead2)
    self.dt = 0.05
    self.n_steps = int(round(duration / self.dt))
    self.duration = self.n_steps * self.dt

  def sample(self, t, s_ego):
    specs = []
    for spec in self.lead_specs:
      if spec is None:
        specs.append(None)
        continue
      state = spec.state(t, s_ego)
      if state is None:
        specs.append(None)
      else:
        x, v, a = state
        specs.append((x, v, a, True, True, spec.track_id, 1.0))
    return specs

  def v_cruise_kph(self, _t):
    return self.cruise_kph


class FarLeadLoop:
  def __init__(self, header, profile, speed_tf=None):
    _seed_param_defaults()
    params = Params()
    _apply_header_params(params, header)
    if speed_tf is not None:
      params.put_int("SpeedTFFactor", int(speed_tf))
    self.personality = int(header.get("LongitudinalPersonality", 1))

    self.profile = profile
    cp = CarInterface.get_non_essential_params(CAR_MODEL)
    cp.openpilotLongitudinalControl = True
    self.cp = cp

    self.ego_v = float(profile.initial_speed)
    self.ego_x = 0.0

    self.planner = LongitudinalPlanner(cp, init_v=self.ego_v)
    self.long_control = LongControl(cp)
    self.long_control.last_output_accel = float(profile.initial_accel)
    self.carrot = CarrotPlanner()

  def _radar_message(self, specs):
    radar = messaging.new_message('radarState')
    for name, spec in zip(("leadOne", "leadTwo"), specs, strict=True):
      lead = log.RadarState.LeadData.new_message()
      if spec is None:
        lead.status = False
        lead.dRel = 200.0
        lead.vRel = 0.0
        lead.vLead = float(self.ego_v)
        lead.vLeadK = float(self.ego_v)
        lead.aLeadK = 0.0
        lead.aRel = 0.0
        lead.modelProb = 0.0
        lead.radar = True
        lead.radarTrackId = -1
      else:
        x, v, a, status, radar_flag, track_id, prob = spec
        lead.status = bool(status)
        lead.dRel = float(max(0.0, x - self.ego_x))
        lead.vRel = float(v - self.ego_v)
        lead.vLead = float(v)
        lead.vLeadK = float(v)
        lead.aLeadK = float(a)
        lead.aRel = float(a - self.long_control.last_output_accel)
        lead.modelProb = float(prob)
        lead.radar = bool(radar_flag)
        lead.radarTrackId = int(track_id)
      lead.aLeadTau = float(LEAD_ACCEL_TAU)
      setattr(radar.radarState, name, lead)
    return radar.radarState

  def _sm_messages(self, v_cruise_kph, standstill):
    model = messaging.new_message('modelV2')
    pos = log.XYZTData.new_message()
    pos.x = [float(x) for x in (self.ego_v + 0.5) * np.array(ModelConstants.T_IDXS)]
    pos.y = [0.0] * len(ModelConstants.T_IDXS)
    pos.z = [0.0] * len(ModelConstants.T_IDXS)
    model.modelV2.position = pos
    vel = log.XYZTData.new_message()
    vel.x = [float(x) for x in (self.ego_v + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
    vel.x[0] = float(self.ego_v)
    model.modelV2.velocity = vel
    acc = log.XYZTData.new_message()
    acc.x = [0.0] * len(ModelConstants.T_IDXS)
    model.modelV2.acceleration = acc
    model.modelV2.action.desiredAcceleration = float(self.long_control.last_output_accel + 0.1)
    model.modelV2.meta.disengagePredictions.gasPressProbs = [1.0] * 6

    car_state = messaging.new_message('carState')
    car_state.carState.vEgo = float(self.ego_v)
    car_state.carState.aEgo = float(self.long_control.last_output_accel)
    car_state.carState.vCluRatio = 1.0
    car_state.carState.standstill = bool(standstill)
    car_state.carState.vCruise = float(v_cruise_kph)
    car_state.carState.gasPressed = False
    car_state.carState.brakePressed = False

    ss = messaging.new_message('selfdriveState')
    ss.selfdriveState.experimentalMode = False
    ss.selfdriveState.personality = int(self.personality)

    control = messaging.new_message('controlsState')
    control.controlsState.longControlState = self.long_control.long_control_state
    control.controlsState.forceDecel = False
    control.controlsState.desiredCurvature = 0.0
    control.controlsState.curvature = 0.0

    car_control = messaging.new_message('carControl')
    car_control.carControl.orientationNED = [0.0, 0.0, 0.0]

    return {
      "carState": car_state.carState,
      "modelV2": model.modelV2,
      "selfdriveState": ss.selfdriveState,
      "controlsState": control.controlsState,
      "carControl": car_control.carControl,
    }

  def _plan_view(self):
    planner = self.planner
    return types.SimpleNamespace(
      aTarget=float(planner.output_a_target), vTargetNow=float(planner.output_v_target_now),
      jTargetNow=float(planner.output_j_target_now), shouldStop=bool(planner.output_should_stop),
      fcw=bool(planner.fcw), speeds=planner.v_desired_trajectory,
      longitudinalPlanSource=planner.mpc.source,
      cruiseCoastingTarget=float(planner.coasting_target),
      cruiseCoastingPercent=int(planner.coasting_percent),
      cruiseTarget=float(planner.v_cruise_kph),
    )

  def _cs_view(self, v_cruise_kph, standstill):
    return types.SimpleNamespace(
      vEgo=float(self.ego_v), aEgo=float(self.long_control.last_output_accel),
      brakePressed=False, gasPressed=False, softHoldActive=0,
      standstill=bool(standstill),
      cruiseState=types.SimpleNamespace(standstill=False),
      vCruise=float(v_cruise_kph), carrotCruise=0,
    )

  def run(self):
    dt = DT_CTRL
    profile = self.profile
    max_frames = profile.n_steps * PLANNER_PERIOD
    log_t, log_v, log_accel, log_x = [], [], [], []
    log_gap, log_active, log_vlead, log_atarget = [], [], [], []
    fcw_steps = 0
    approach_steps = 0
    approach_scale_max = 0.0
    msgs = None
    plan_view = None
    v_cruise_kph = profile.v_cruise_kph(0.0)
    standstill_frames = 0
    stopped = False
    for frame in range(max_frames):
      t = frame * dt
      standstill = self.ego_v < 0.01
      v_cruise_kph = profile.v_cruise_kph(t)
      v_cruise_ms = v_cruise_kph / 3.6

      if frame % PLANNER_PERIOD == 0:
        specs = profile.sample(t, self.ego_x)
        radar_state = self._radar_message(specs)
        msgs = {"radarState": radar_state}
        msgs.update(self._sm_messages(v_cruise_kph, standstill))
        self.planner.update(_FakeSM(msgs, frame), self.carrot)
        plan_view = self._plan_view()
        if self.planner.mpc.crash_cnt > 0:
          fcw_steps += 1
        scale = float(getattr(self.planner.mpc, 'approach_scale', 0.0))
        approach_scale_max = max(approach_scale_max, scale)
        if scale > 0.0:
          approach_steps += 1

      t_since_plan = (frame % PLANNER_PERIOD) * dt
      limits = CarInterface.get_pid_accel_limits(self.cp, self.ego_v, v_cruise_ms)
      radar_view = types.SimpleNamespace(leadOne=msgs["radarState"].leadOne,
                                         leadTwo=msgs["radarState"].leadTwo)
      accel, _, _ = self.long_control.update(
        True, self._cs_view(v_cruise_kph, standstill), plan_view,
        limits, t_since_plan, radar_view)
      accel = float(accel)

      gap = np.nan
      v_lead = np.nan
      active = False
      for spec in profile.sample(t, self.ego_x):
        if spec is not None and spec[3]:
          g = spec[0] - self.ego_x
          if not active or g < gap:
            gap, v_lead, active = g, spec[1], True
      log_t.append(t)
      log_v.append(self.ego_v)
      log_accel.append(accel)
      log_x.append(self.ego_x)
      log_gap.append(gap)
      log_active.append(active)
      log_vlead.append(v_lead)
      log_atarget.append(float(plan_view.aTarget))

      self.ego_v = max(0.0, self.ego_v + accel * dt)
      self.ego_x = self.ego_x + self.ego_v * dt
      if self.ego_v < 0.05:
        standstill_frames += 1
        if standstill_frames >= 300:
          stopped = True
          break
      else:
        standstill_frames = 0
    return {
      "t": log_t, "v": log_v, "accel": log_accel, "x": log_x,
      "gap": log_gap, "active": log_active, "v_lead": log_vlead, "a_target": log_atarget,
      "fcw_steps": fcw_steps, "stopped": bool(stopped or self.ego_v < 0.05),
      "approach_steps": approach_steps, "approach_scale_max": round(approach_scale_max, 4),
    }


def compute_metrics(run):
  v = np.array(run["v"])
  a = np.array(run["accel"])
  gap = np.array(run["gap"])
  active = np.array(run["active"])
  v_lead = np.array(run["v_lead"])
  a_target = np.array(run["a_target"])

  min_a_target = float(np.min(a_target))
  min_accel = float(np.min(a))
  if a.size > 51:
    jerk = (a[50:] - a[:-50]) / 0.5
    onset_jerk = float(np.min(jerk))
    peak_jerk_abs = float(np.max(np.abs(jerk)))
  else:
    onset_jerk = 0.0
    peak_jerk_abs = 0.0

  collision = bool(np.any(gap[active] <= 0.0))
  min_gap = float(np.min(gap[active])) if bool(np.any(active)) else None

  tg = gap[active] / np.maximum(v[active], 1e-6)
  tg = tg[v[active] > 10.0]
  min_time_gap = float(np.min(tg)) if tg.size else None

  closing = v[active] - v_lead[active]
  ttc = gap[active] / np.maximum(closing, 1e-6)
  ttc = ttc[closing > 0.5]
  min_ttc = float(np.min(ttc)) if ttc.size else None

  stand = np.flatnonzero(v < 0.05)
  final_gap = float(gap[stand[0]]) if stand.size else None

  metrics = {
    "min_a_target": round(min_a_target, 4),
    "min_accel": round(min_accel, 4),
    "onset_jerk": round(onset_jerk, 4),
    "peak_jerk_abs": round(peak_jerk_abs, 4),
    "min_gap_m": None if min_gap is None else round(min_gap, 4),
    "min_time_gap_s": None if min_time_gap is None else round(min_time_gap, 4),
    "min_ttc_s": None if min_ttc is None else round(min_ttc, 4),
    "fcw_steps": int(run["fcw_steps"]),
    "collision": collision,
    "stopped": bool(run["stopped"]),
    "final_gap_m": None if final_gap is None else round(final_gap, 4),
    "approach_steps": int(run.get("approach_steps", 0)),
    "approach_scale_max": float(run.get("approach_scale_max", 0.0)),
  }
  return metrics


def _run(profile, header, *, static, approach=None, speed_tf=None):
  saved_static = long_mpc.DANGER_STATIC_T_FOLLOW
  long_mpc.DANGER_STATIC_T_FOLLOW = static
  saved_approach = far_lead_approach.FAR_LEAD_APPROACH
  if approach is not None:
    far_lead_approach.FAR_LEAD_APPROACH = approach
  try:
    return compute_metrics(FarLeadLoop(header, profile, speed_tf=speed_tf).run())
  finally:
    long_mpc.DANGER_STATIC_T_FOLLOW = saved_static
    far_lead_approach.FAR_LEAD_APPROACH = saved_approach


def _run_episode(eid):
  fixture = load_fixture(eid)
  header = fixture["params"]
  profile = FixtureProfile(fixture)
  v0 = _run(profile, header, static=False, approach=False)
  v1 = _run(profile, header, static=True, approach=False)
  v3 = _run(profile, header, static=True, approach=True)
  return fixture, {"V0": v0, "V1": v1, "V3": v3}


def _comfort_failures(eid, res):
  fails = []
  v0, v3 = res["V0"], res["V3"]
  if v3["min_a_target"] < v0["min_a_target"] - 0.05:
    fails.append(f"{eid}: V3 peak decel {v3['min_a_target']} < V0 {v0['min_a_target']} - 0.05")
  bound = min(v0["min_time_gap_s"] if v0["min_time_gap_s"] is not None else 1e9, 1.0) - 0.05
  if v3["min_time_gap_s"] is None or v3["min_time_gap_s"] < bound - 1e-9:
    fails.append(f"{eid}: V3 min time-gap {v3['min_time_gap_s']} < {bound}")
  if v3["collision"]:
    fails.append(f"{eid}: V3 collides")
  if v3["fcw_steps"] > v0["fcw_steps"]:
    fails.append(f"{eid}: FCW V3 {v3['fcw_steps']} > V0 {v0['fcw_steps']}")
  if v3["onset_jerk"] < v0["onset_jerk"] - 0.1:
    fails.append(f"{eid}: V3 onset jerk {v3['onset_jerk']} < V0 {v0['onset_jerk']} - 0.1")
  return fails


def _safety_failures(eid, res):
  fails = []
  v0, v3 = res["V0"], res["V3"]
  if v3["min_gap_m"] is None or v0["min_gap_m"] is None:
    fails.append(f"{eid}: missing gap metrics")
    return fails
  if v3["min_gap_m"] < v0["min_gap_m"] - 0.5:
    fails.append(f"{eid}: V3 min gap {v3['min_gap_m']} < V0 {v0['min_gap_m']} - 0.5")
  v0_ttc = v0["min_ttc_s"] if v0["min_ttc_s"] is not None else 1e9
  bound = min(v0_ttc, 3.0) - 0.1
  if v3["min_ttc_s"] is None or v3["min_ttc_s"] < bound - 1e-9:
    fails.append(f"{eid}: V3 min TTC {v3['min_ttc_s']} < {bound}")
  if v3["collision"]:
    fails.append(f"{eid}: V3 collides")
  if v3["fcw_steps"] > v0["fcw_steps"]:
    fails.append(f"{eid}: FCW V3 {v3['fcw_steps']} > V0 {v0['fcw_steps']}")
  return fails


@pytest.mark.parametrize("eid", EPISODES)
def test_far_lead_replay_comfort(eid):
  fixture, res = _run_episode(eid)
  log_min_at = min(fixture["data"]["aTargetLog"]) / fixture["scale"]["a_target"]
  fidelity = abs(res["V0"]["min_a_target"] - log_min_at)
  RESULTS["episodes"][eid] = {**res, "log_min_aTarget": round(log_min_at, 4),
                              "fidelity_delta": round(fidelity, 4)}
  RESULTS["fidelity"][eid] = fidelity <= 0.4 and res["V0"]["min_a_target"] <= log_min_at + 0.4
  fails = _comfort_failures(eid, res)
  assert not fails, "; ".join(fails)


def test_far_lead_replay_safety():
  eid = "s1"
  fixture, res = _run_episode(eid)
  log_min_at = min(fixture["data"]["aTargetLog"]) / fixture["scale"]["a_target"]
  RESULTS["episodes"][eid] = {**res, "log_min_aTarget": round(log_min_at, 4),
                              "fidelity_delta": round(abs(res["V0"]["min_a_target"] - log_min_at), 4)}
  fails = _safety_failures(eid, res)
  assert not fails, "; ".join(fails)


def test_far_lead_aggregate():
  eps = RESULTS["episodes"]
  missing = [e for e in EPISODES if e not in eps]
  assert not missing, f"missing episode results: {missing}"
  reductions = []
  for e in EPISODES:
    m0 = eps[e]["V0"]["min_a_target"]
    m3 = eps[e]["V3"]["min_a_target"]
    if m0 < 0.0:
      reductions.append((abs(m0) - abs(m3)) / abs(m0))
  mean_reduction = float(np.mean(reductions)) if reductions else 0.0
  fidelity_count = int(sum(RESULTS["fidelity"].values()))
  gentle_v3 = sum(1 for e in EPISODES
                  if eps[e]["V3"]["min_a_target"] >= -1.5 and eps[e]["V3"]["onset_jerk"] >= -2.5)
  RESULTS["report"] = {
    "mean_peak_reduction": round(mean_reduction, 4),
    "fidelity_within_0.4_count": fidelity_count,
    "comfort_episodes": len(EPISODES),
    "v3_peak_ge_-1.5_and_onset_jerk_ge_-2.5_count": gentle_v3,
  }
  print(f"FAR-LEAD: fidelity {fidelity_count}/{len(EPISODES)}; mean peak reduction {mean_reduction * 100:.1f}%; gentle V3 {gentle_v3}")
  assert mean_reduction >= 0.35, f"mean peak-decel reduction {mean_reduction * 100:.1f}% < 35%"


def _maneuvers():
  lead1 = _LeadSpec
  return {
    "m1": SyntheticProfile(110, 110, lead1(x0=100.0, v0=70 / 3.6, brake_t=8.0, brake_a=-4.0)),
    "m2": SyntheticProfile(110, 110, lead1(x0=100.0, v0=70 / 3.6, brake_t=8.0, brake_a=-6.0)),
    "m3": SyntheticProfile(110, 110, lead1(x0=100.0, v0=70 / 3.6, brake_t=2.0, brake_a=-6.0)),
    "m4": SyntheticProfile(100, 100, lead1(x0=64.0, v0=100 / 3.6),
                           lead2=lead1(v0=80 / 3.6, track_id=8, cut_in_t=4.0, cut_in_gap=30.0)),
    "m5": SyntheticProfile(90, 90, lead1(x0=120.0, v0=0.0)),
    "m6": SyntheticProfile(110, 110, _BrakeThenHoldSpec(x0=100.0, v0=70 / 3.6, brake_t=4.0, brake_a=-3.0, brake_dur=3.0)),
  }


def _maneuver_failures(mid, profile, res):
  fails = []
  v0_10, v3_15 = res["V0_10"], res["V3_15"]
  for name in ("V0_10", "V0_15", "V3_15"):
    if res[name]["collision"]:
      fails.append(f"{mid}: {name} collides (min gap {res[name]['min_gap_m']})")
  end_speed = profile.lead_specs[0].state(profile.duration, 0.0)
  lead_stops = end_speed[1] <= 0.1
  if lead_stops and v3_15["final_gap_m"] is not None and v3_15["final_gap_m"] < 2.0:
    fails.append(f"{mid}: V3@15 final gap {v3_15['final_gap_m']} < 2.0 m")
  if v0_10["min_gap_m"] is not None and v3_15["min_gap_m"] is not None:
    if v3_15["min_gap_m"] < v0_10["min_gap_m"] - 0.5:
      fails.append(f"{mid}: V3@15 min gap {v3_15['min_gap_m']} < V0@10 {v0_10['min_gap_m']} - 0.5")
  if v3_15["fcw_steps"] > v0_10["fcw_steps"]:
    fails.append(f"{mid}: FCW V3@15 {v3_15['fcw_steps']} > V0@10 {v0_10['fcw_steps']}")
  if mid == "m6" and v3_15["min_a_target"] < v0_10["min_a_target"] - 0.5:
    fails.append(f"{mid}: V3@15 peak decel {v3_15['min_a_target']} < V0@10 {v0_10['min_a_target']} - 0.5")
  return fails


@pytest.mark.parametrize("mid", ["m1", "m2", "m3", "m4", "m5", "m6"])
def test_far_lead_synthetic_maneuver(mid):
  header = load_fixture("e1")["params"]
  res = {
    "V0_10": _run(_maneuvers()[mid], header, static=False, approach=False, speed_tf=10),
    "V0_15": _run(_maneuvers()[mid], header, static=False, approach=False, speed_tf=15),
    "V3_15": _run(_maneuvers()[mid], header, static=True, approach=True, speed_tf=15),
  }
  RESULTS["maneuvers"][mid] = res
  fails = _maneuver_failures(mid, _maneuvers()[mid], res)
  assert not fails, "; ".join(fails)


def test_write_far_lead_sim_table():
  out_path = os.environ.get("FAR_LEAD_SIM_OUT", "/tmp/far_lead_sim.json")
  table = {
    "episodes": RESULTS["episodes"],
    "maneuvers": RESULTS["maneuvers"],
    "report": RESULTS["report"],
    "fidelity": RESULTS["fidelity"],
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  text = json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False)
  print(text)
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(text + "\n")


_SWEEP_MARGINS = (1.1, 1.25, 1.4)
_SWEEP_A_MAX = (1.6, 2.0)


def _sweep_output_path():
  out = os.environ.get("FAR_LEAD_SWEEP_OUT")
  if out:
    return out
  sim_out = os.environ.get("FAR_LEAD_SIM_OUT")
  if sim_out:
    return os.path.join(os.path.dirname(sim_out), "far_lead_sweep.json")
  return None


@pytest.mark.skipif(os.environ.get("FAR_LEAD_SWEEP") != "1",
                    reason="approach constant sweep runs only when FAR_LEAD_SWEEP=1")
def test_far_lead_approach_sweep(monkeypatch):
  """APPROACH_MARGIN {1.1, 1.25, 1.4} x APPROACH_A_MAX {1.6, 2.0} over everything.

  Auto-pick: among combos passing the BRIEF_v2 criteria, the largest
  APPROACH_MARGIN whose mean V3 peak decel is within 0.10 m/s^2 of the best
  (gentlest) passing combo; tie -> smaller APPROACH_A_MAX. No assertions here:
  when nothing passes, the closest two are reported with their failures.
  """
  episodes = {}
  for eid in EPISODES:
    fixture = load_fixture(eid)
    profile = FixtureProfile(fixture)
    episodes[eid] = {
      "fixture": fixture, "profile": profile,
      "log_min": min(fixture["data"]["aTargetLog"]) / fixture["scale"]["a_target"],
      "v0": _run(profile, fixture["params"], static=False, approach=False),
    }
  s1f = load_fixture("s1")
  s1 = {"profile": FixtureProfile(s1f), "fixture": s1f,
        "log_min": min(s1f["data"]["aTargetLog"]) / s1f["scale"]["a_target"]}
  s1["v0"] = _run(s1["profile"], s1f["params"], static=False, approach=False)
  header = episodes["e1"]["fixture"]["params"]
  man_v0 = {mid: {"V0_10": _run(p, header, static=False, approach=False, speed_tf=10),
                  "V0_15": _run(p, header, static=False, approach=False, speed_tf=15)}
            for mid, p in _maneuvers().items()}

  fidelity_count = int(sum(1 for e in episodes.values()
                           if abs(e["v0"]["min_a_target"] - e["log_min"]) <= 0.4))
  fidelity_gentle = all(e["v0"]["min_a_target"] <= e["log_min"] + 0.4 for e in episodes.values())
  fidelity_fails = [] if (fidelity_count >= 4 and fidelity_gentle) else [f"fidelity {fidelity_count}/6"]

  rows = []
  for margin in _SWEEP_MARGINS:
    for a_max in _SWEEP_A_MAX:
      monkeypatch.setattr(far_lead_approach, "APPROACH_MARGIN", margin)
      monkeypatch.setattr(far_lead_approach, "APPROACH_A_MAX", a_max)
      fails = list(fidelity_fails)
      peaks = []
      worst = (None, 0.0)
      approach_steps = {}
      reductions = []
      for eid, e in episodes.items():
        v3 = _run(e["profile"], e["fixture"]["params"], static=True, approach=True)
        fails += _comfort_failures(eid, {"V0": e["v0"], "V3": v3})
        peaks.append(v3["min_a_target"])
        approach_steps[eid] = v3["approach_steps"]
        if e["v0"]["min_a_target"] < 0.0:
          reductions.append((abs(e["v0"]["min_a_target"]) - abs(v3["min_a_target"])) / abs(e["v0"]["min_a_target"]))
        if worst[0] is None or v3["min_a_target"] < worst[1]:
          worst = (eid, v3["min_a_target"])
      mean_reduction = float(np.mean(reductions)) if reductions else 0.0
      if mean_reduction < 0.35:
        fails.append(f"mean peak-decel reduction {mean_reduction * 100:.1f}% < 35%")
      v3_s1 = _run(s1["profile"], s1f["params"], static=True, approach=True)
      fails += _safety_failures("s1", {"V0": s1["v0"], "V3": v3_s1})
      approach_steps["s1"] = v3_s1["approach_steps"]
      for mid, profile in _maneuvers().items():
        res = dict(man_v0[mid])
        res["V3_15"] = _run(profile, header, static=True, approach=True, speed_tf=15)
        fails += _maneuver_failures(mid, profile, res)
      rows.append({
        "margin": margin, "a_max": a_max, "pass": not fails, "failures": fails,
        "mean_v3_peak": round(float(np.mean(peaks)), 4),
        "mean_reduction": round(mean_reduction, 4),
        "worst_episode": worst[0], "worst_v3_peak": round(worst[1], 4),
        "approach_steps": approach_steps,
      })

  passing = [r for r in rows if r["pass"]]
  chosen = None
  if passing:
    best_mean = max(r["mean_v3_peak"] for r in passing)
    eligible = [r for r in passing if r["mean_v3_peak"] >= best_mean - 0.10]
    pick = sorted(eligible, key=lambda r: (-r["margin"], r["a_max"]))[0]
    chosen = {"margin": pick["margin"], "a_max": pick["a_max"],
              "mean_v3_peak": pick["mean_v3_peak"]}
  closest = sorted((r for r in rows if not r["pass"]),
                   key=lambda r: (len(r["failures"]), -r["mean_v3_peak"]))[:2]

  for r in rows:
    msg = 'sweep M {} A {}: pass={} fails={} mean_v3_peak={} worst={} {}'.format(
      r["margin"], r["a_max"], r["pass"], len(r["failures"]), r["mean_v3_peak"],
      r["worst_episode"], r["worst_v3_peak"])
    print(msg)
  if chosen is not None:
    print(f'sweep chosen: {chosen}')
  else:
    print("sweep: no combo passed; closest two:")
    for r in closest:
      print(f'  M {r["margin"]} A {r["a_max"]}: {r["failures"]}')

  path = _sweep_output_path()
  if path:
    criteria = "BRIEF_v2 PASS criteria; auto-pick = largest margin whose mean V3 peak is within 0.10 of the best passing combo, tie -> smaller a_max"
    text = json.dumps({
      "combos": rows, "chosen": chosen, "closest": closest,
      "fidelity_count": fidelity_count, "fidelity_gentle": fidelity_gentle,
      "criteria": criteria,
    }, ensure_ascii=False, indent=2, sort_keys=False)
    out_dir = os.path.dirname(path)
    if out_dir:
      os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
      handle.write(text + "\n")
