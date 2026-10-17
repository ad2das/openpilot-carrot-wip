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
  V0 = DANGER_STATIC_T_FOLLOW=False (pre-W112 danger boundary)
  V1 = default (static danger TF)
Maneuvers run V0@SpeedTFFactor10 (reference), V0@15 and V1@15.

PASS criteria (asserted):
  Fidelity: |V0 peak - log peak| <= 0.4 in >= 4/6 AND V0 never gentler than
  log - 0.4 in any comfort episode.
  Comfort e1-e6 (V1 vs V0), every episode: V1 peak decel >= V0 peak - 0.05;
  V1 onset jerk >= V0 onset jerk - 0.1; V1 min time-gap >= min(V0 min
  time-gap, 1.0 s) - 0.05; no collision; FCW V1 <= V0. Aggregate: mean
  peak-decel magnitude reduction > 0.0 (regression guard; the value is
  reported).
  Safety s1: V1 min gap >= V0 min gap - 0.5 m; V1 min TTC >= min(V0 min TTC, 3.0 s) - 0.1;
  no collision; FCW V1 <= V0.
  Maneuvers m1-m6: no collision for any variant; final gap >= 2.0 m when the
  lead stops; V1@15 min gap >= V0@10 min gap - 0.5 m; FCW V1@15 <= V0@10;
  m6: V1@15 peak decel >= V0@10 peak decel - 0.5.
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
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR

CAR_MODEL = CAR.HYUNDAI_PALISADE
PLANNER_PERIOD = 5  # 100 Hz loop, 20 Hz planner/model, as in the car
SENTINEL = -32767
PLANT_TAU = 0.3     # actuator first-order lag (s); automatic-transmission creep plant
PLANT_A_GATE = 0.3  # creep torque adds only while the lagged command is below this
PLANT_SCC_BRAKE = -0.6  # SCC StopReq braking ceiling in the stopping state (m/s^2)


def plant_creep(v):
  """Automatic-transmission creep torque vs speed (fit from the b1 log, W112 task 4)."""
  if v < 1.5:
    return 0.25
  if v < 3.0:
    return 0.25 * (3.0 - v) / 1.5
  return 0.0


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
  def __init__(self, header, profile, speed_tf=None, plant=False, scc_stop=False, telemetry=None):
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
    self.plant = bool(plant)
    self.scc_stop = bool(scc_stop)
    self.telemetry = telemetry
    if telemetry is not None:
      for key in ("t", "v", "v_lead", "gap", "floor", "margin", "stopped_pair",
                  "fast_release", "cap", "desired_distance", "a_target", "butter_floor", "lc_state"):
        telemetry.setdefault(key, [])
    self._plant_lag = float(profile.initial_accel)
    self.a_actual = float(profile.initial_accel)

    self.planner = LongitudinalPlanner(cp, init_v=self.ego_v)
    self.long_control = LongControl(cp)
    self.long_control.last_output_accel = float(profile.initial_accel)
    self.carrot = CarrotPlanner()

  def _record_telemetry(self, t, specs, plan_view):
    """Per-planner-tick trace: ego/lead v, gap, BrakingGapState (lead 0), plan, floor."""
    state = self.planner.mpc.braking_gap_states[0]
    spec0 = specs[0] if specs else None
    floor = state.floor
    bf = self.long_control.butter_floor_last
    tel = self.telemetry
    tel["t"].append(round(float(t), 2))
    tel["v"].append(round(float(self.ego_v), 3))
    tel["v_lead"].append(None if spec0 is None or not np.isfinite(spec0[1]) else round(float(spec0[1]), 3))
    tel["gap"].append(None if spec0 is None or not np.isfinite(spec0[0]) else round(float(spec0[0]) - self.ego_x, 3))
    tel["floor"].append(None if floor is None else round(float(floor), 3))
    tel["margin"].append(round(float(state.margin), 3))
    tel["stopped_pair"].append(bool(state.stopped_pair))
    tel["fast_release"].append(bool(state.fast_release))
    tel["cap"].append(None if state.speed_cap is None else round(float(state.speed_cap), 3))
    tel["desired_distance"].append(round(float(plan_view.desiredDistance), 3))
    tel["a_target"].append(round(float(plan_view.aTarget), 3))
    tel["butter_floor"].append(None if bf is None else round(float(bf), 3))
    tel["lc_state"].append(int(self.long_control.long_control_state))

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
      desiredDistance=float(planner.mpc.desired_distance),
      cruiseCoastingTarget=float(planner.coasting_target),
      cruiseCoastingPercent=int(planner.coasting_percent),
      cruiseTarget=float(planner.v_cruise_kph),
      actionTime=float(getattr(planner, "action_t_last", float("nan"))),
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
    log_a_actual, log_should_stop, log_lc_state = [], [], []
    log_desired_distance = []
    fcw_steps = 0
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
        if self.telemetry is not None:
          self._record_telemetry(t, specs, plan_view)

      t_since_plan = (frame % PLANNER_PERIOD) * dt
      limits = CarInterface.get_pid_accel_limits(self.cp, self.ego_v, v_cruise_ms)
      radar_view = types.SimpleNamespace(leadOne=msgs["radarState"].leadOne,
                                         leadTwo=msgs["radarState"].leadTwo)
      accel, _, _ = self.long_control.update(
        True, self._cs_view(v_cruise_kph, standstill), plan_view,
        limits, t_since_plan, radar_view)
      accel = float(accel)

      if self.plant:
        self._plant_lag += (dt / PLANT_TAU) * (accel - self._plant_lag)
        applied = self._plant_lag
        if accel < PLANT_A_GATE and self.ego_v > 0.01:
          applied += plant_creep(self.ego_v)
      else:
        applied = accel
      if self.scc_stop and self.long_control.long_control_state == LongCtrlState.stopping:
        # SCC owns the stop in the stopping state: at least -0.6 m/s^2 until
        # standstill, then hold.
        applied = 0.0 if self.ego_v <= 0.01 else min(applied, PLANT_SCC_BRAKE)
      self.a_actual = applied

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
      log_a_actual.append(applied)
      log_should_stop.append(bool(plan_view.shouldStop))
      log_lc_state.append(int(self.long_control.long_control_state))
      log_desired_distance.append(float(plan_view.desiredDistance))

      self.ego_v = max(0.0, self.ego_v + applied * dt)
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
      "a_actual": log_a_actual, "should_stop": log_should_stop, "lc_state": log_lc_state,
      "desired_distance": log_desired_distance,
      "fcw_steps": fcw_steps, "stopped": bool(stopped or self.ego_v < 0.05),
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
  }
  return metrics


def _run(profile, header, *, static, speed_tf=None):
  saved_static = long_mpc.DANGER_STATIC_T_FOLLOW
  long_mpc.DANGER_STATIC_T_FOLLOW = static
  try:
    return compute_metrics(FarLeadLoop(header, profile, speed_tf=speed_tf).run())
  finally:
    long_mpc.DANGER_STATIC_T_FOLLOW = saved_static


def _run_episode(eid):
  fixture = load_fixture(eid)
  header = fixture["params"]
  profile = FixtureProfile(fixture)
  v0 = _run(profile, header, static=False)
  v1 = _run(profile, header, static=True)
  return fixture, {"V0": v0, "V1": v1}


def _comfort_failures(eid, res):
  fails = []
  v0, v1 = res["V0"], res["V1"]
  if v1["min_a_target"] < v0["min_a_target"] - 0.05:
    fails.append(f"{eid}: V1 peak decel {v1['min_a_target']} < V0 {v0['min_a_target']} - 0.05")
  if v1["onset_jerk"] < v0["onset_jerk"] - 0.1:
    fails.append(f"{eid}: V1 onset jerk {v1['onset_jerk']} < V0 {v0['onset_jerk']} - 0.1")
  bound = min(v0["min_time_gap_s"] if v0["min_time_gap_s"] is not None else 1e9, 1.0) - 0.05
  if v1["min_time_gap_s"] is None or v1["min_time_gap_s"] < bound - 1e-9:
    fails.append(f"{eid}: V1 min time-gap {v1['min_time_gap_s']} < {bound}")
  if v1["collision"]:
    fails.append(f"{eid}: V1 collides")
  if v1["fcw_steps"] > v0["fcw_steps"]:
    fails.append(f"{eid}: FCW V1 {v1['fcw_steps']} > V0 {v0['fcw_steps']}")
  return fails


def _safety_failures(eid, res):
  fails = []
  v0, v1 = res["V0"], res["V1"]
  if v1["min_gap_m"] is None or v0["min_gap_m"] is None:
    fails.append(f"{eid}: missing gap metrics")
    return fails
  if v1["min_gap_m"] < v0["min_gap_m"] - 0.5:
    fails.append(f"{eid}: V1 min gap {v1['min_gap_m']} < V0 {v0['min_gap_m']} - 0.5")
  v0_ttc = v0["min_ttc_s"] if v0["min_ttc_s"] is not None else 1e9
  bound = min(v0_ttc, 3.0) - 0.1
  if v1["min_ttc_s"] is None or v1["min_ttc_s"] < bound - 1e-9:
    fails.append(f"{eid}: V1 min TTC {v1['min_ttc_s']} < {bound}")
  if v1["collision"]:
    fails.append(f"{eid}: V1 collides")
  if v1["fcw_steps"] > v0["fcw_steps"]:
    fails.append(f"{eid}: FCW V1 {v1['fcw_steps']} > V0 {v0['fcw_steps']}")
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
    m1 = eps[e]["V1"]["min_a_target"]
    if m0 < 0.0:
      reductions.append((abs(m0) - abs(m1)) / abs(m0))
  mean_reduction = float(np.mean(reductions)) if reductions else 0.0
  fidelity_count = int(sum(RESULTS["fidelity"].values()))
  gentle_v1 = sum(1 for e in EPISODES
                  if eps[e]["V1"]["min_a_target"] >= -1.5 and eps[e]["V1"]["onset_jerk"] >= -2.5)
  RESULTS["report"] = {
    "mean_peak_reduction": round(mean_reduction, 4),
    "fidelity_within_0.4_count": fidelity_count,
    "comfort_episodes": len(EPISODES),
    "v1_peak_ge_-1.5_and_onset_jerk_ge_-2.5_count": gentle_v1,
  }
  print(f"FAR-LEAD: fidelity {fidelity_count}/{len(EPISODES)}; mean peak reduction {mean_reduction * 100:.1f}%; gentle V1 {gentle_v1}")
  assert fidelity_count >= 4, f"fidelity within 0.4 in only {fidelity_count}/6"
  assert mean_reduction > 0.0, f"mean peak-decel reduction {mean_reduction * 100:.1f}% <= 0"


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
  v0_10, v1_15 = res["V0_10"], res["V1_15"]
  for name in ("V0_10", "V0_15", "V1_15"):
    if res[name]["collision"]:
      fails.append(f"{mid}: {name} collides (min gap {res[name]['min_gap_m']})")
  end_speed = profile.lead_specs[0].state(profile.duration, 0.0)
  lead_stops = end_speed[1] <= 0.1
  if lead_stops and v1_15["final_gap_m"] is not None and v1_15["final_gap_m"] < 2.0:
    fails.append(f"{mid}: V1@15 final gap {v1_15['final_gap_m']} < 2.0 m")
  if v0_10["min_gap_m"] is not None and v1_15["min_gap_m"] is not None:
    if v1_15["min_gap_m"] < v0_10["min_gap_m"] - 0.5:
      fails.append(f"{mid}: V1@15 min gap {v1_15['min_gap_m']} < V0@10 {v0_10['min_gap_m']} - 0.5")
  if v1_15["fcw_steps"] > v0_10["fcw_steps"]:
    fails.append(f"{mid}: FCW V1@15 {v1_15['fcw_steps']} > V0@10 {v0_10['fcw_steps']}")
  if mid == "m6" and v1_15["min_a_target"] < v0_10["min_a_target"] - 0.5:
    fails.append(f"{mid}: V1@15 peak decel {v1_15['min_a_target']} < V0@10 {v0_10['min_a_target']} - 0.5")
  return fails


@pytest.mark.parametrize("mid", ["m1", "m2", "m3", "m4", "m5", "m6"])
def test_far_lead_synthetic_maneuver(mid):
  header = load_fixture("e1")["params"]
  res = {
    "V0_10": _run(_maneuvers()[mid], header, static=False, speed_tf=10),
    "V0_15": _run(_maneuvers()[mid], header, static=False, speed_tf=15),
    "V1_15": _run(_maneuvers()[mid], header, static=True, speed_tf=15),
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
