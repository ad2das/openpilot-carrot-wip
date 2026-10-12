"""ButterStop closed-loop maneuver validation on the real acados solver.

Runs where the native longitudinal MPC solver exists (CI build_release), the
same environment as the longitudinal maneuver tests. The loop is the real
LongitudinalPlanner + real acados MPC + real LongControl at 100 Hz with the
planner stepping at 20 Hz, fed the same straight-line model/radar messages
plant.py builds.

plant.Plant is not used directly: its update(sm) call predates this fork's
update(sm, carrot) signature, it never runs LongControl (where the ButterStop
floor and creep guard live) and it has no ButterStop hook. Maneuver provides
the scenario definitions. Simulation only; no vehicle validation.
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
from openpilot.selfdrive.controls.lib import longcontrol
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR

CAR_MODEL = CAR.HYUNDAI_PALISADE
PLANNER_PERIOD = 5  # 100 Hz loop, 20 Hz planner/model, as in the car
COLLISION_GAP_M = 0.4  # same crash criterion as maneuver.py

RESULTS: list[dict] = []


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
  params.put_int("MyDrivingMode", 3)
  assert params.get_int("MyDrivingMode") == 3


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


class _NeutralCarrot:
  """ButterStop code path absent for the planner: pins butter_stop=False."""

  def __init__(self, inner):
    self._inner = inner
    self.butter_stop = False

  def update(self, sm, v_cruise_kph, mode):
    return self._inner.update(sm, v_cruise_kph, mode)

  def __getattr__(self, name):
    return getattr(self._inner, name)


def _no_floor(*_args, **_kwargs):
  return None


class ButterStopLoop:
  def __init__(self, maneuver, butter, neutralize=False):
    _seed_param_defaults()
    params = Params()
    params.put_int("ButterStop", int(butter))

    self.maneuver = maneuver
    self.neutralize = neutralize
    cp = CarInterface.get_non_essential_params(CAR_MODEL)
    cp.openpilotLongitudinalControl = True
    self.cp = cp

    self.ego_v = float(maneuver.speed)
    self.ego_x = 0.0
    self.lead_x = float(maneuver.distance_lead)
    self.v_lead_prev = 0.0

    self.planner = LongitudinalPlanner(cp, init_v=self.ego_v)
    self.long_control = LongControl(cp)
    carrot = CarrotPlanner()
    self.carrot = _NeutralCarrot(carrot) if neutralize else carrot

  def _radar_message(self, lead):
    radar = messaging.new_message('radarState')
    l1 = log.RadarState.LeadData.new_message()
    if lead is None:
      l1.status = False
      l1.dRel = 200.0
      l1.vRel = 0.0
      l1.vLead = float(self.ego_v)
      l1.vLeadK = float(self.ego_v)
      l1.aLeadK = 0.0
      l1.aRel = 0.0
      l1.modelProb = 0.0
      l1.radar = True
      l1.radarTrackId = -1
    else:
      l1.status = True
      l1.dRel = float(max(0.0, lead["d_rel"]))
      l1.vRel = float(lead["v_rel"])
      l1.vLead = float(lead["v_lead"])
      l1.vLeadK = float(lead["v_lead"])
      l1.aLeadK = float(lead["a_lead"])
      l1.aRel = float(lead["a_lead"] - self.long_control.last_output_accel)
      l1.modelProb = float(lead["prob"])
      l1.radar = True
      l1.radarTrackId = 7
    l1.aLeadTau = float(LEAD_ACCEL_TAU)
    radar.radarState.leadOne = l1
    l2 = log.RadarState.LeadData.new_message()
    l2.status = False
    l2.dRel = 200.0
    l2.vLead = float(self.ego_v)
    l2.vLeadK = float(self.ego_v)
    radar.radarState.leadTwo = l2
    return radar.radarState

  def _sm_messages(self, v_cruise_ms, pitch, prob_throttle, standstill):
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
    model.modelV2.meta.disengagePredictions.gasPressProbs = [float(prob_throttle)] * 6

    car_state = messaging.new_message('carState')
    car_state.carState.vEgo = float(self.ego_v)
    car_state.carState.aEgo = float(self.long_control.last_output_accel)
    car_state.carState.vCluRatio = 1.0
    car_state.carState.standstill = bool(standstill)
    car_state.carState.vCruise = float(v_cruise_ms * 3.6)
    car_state.carState.gasPressed = False
    car_state.carState.brakePressed = False

    ss = messaging.new_message('selfdriveState')
    ss.selfdriveState.experimentalMode = False
    ss.selfdriveState.personality = 1

    control = messaging.new_message('controlsState')
    control.controlsState.longControlState = self.long_control.long_control_state
    control.controlsState.forceDecel = False
    control.controlsState.desiredCurvature = 0.0
    control.controlsState.curvature = 0.0

    car_control = messaging.new_message('carControl')
    car_control.carControl.orientationNED = [0.0, float(pitch), 0.0]

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

  def _cs_view(self, v_cruise_ms, standstill):
    return types.SimpleNamespace(
      vEgo=float(self.ego_v), aEgo=float(self.long_control.last_output_accel),
      brakePressed=False, gasPressed=False, softHoldActive=0,
      standstill=bool(standstill),
      cruiseState=types.SimpleNamespace(standstill=False),
      vCruise=float(v_cruise_ms), carrotCruise=0,
    )

  def run(self):
    if self.neutralize:
      # "ButterStop code path absent": neutralize every ON-only primitive the
      # real loop reaches so the controller behaves exactly like the OFF one.
      saved = (longcontrol.butter_floor, longcontrol.CREEP_V, longcontrol.HOLD_RAMP)
      longcontrol.butter_floor = _no_floor
      longcontrol.CREEP_V = float("inf")
      longcontrol.HOLD_RAMP = self.cp.stoppingDecelRate
    try:
      return self._run_loop()
    finally:
      if self.neutralize:
        longcontrol.butter_floor, longcontrol.CREEP_V, longcontrol.HOLD_RAMP = saved

  def _run_loop(self):
    maneuver = self.maneuver
    dt = DT_CTRL
    frames = int(round(maneuver.duration / dt))
    logs = np.zeros((frames + 1, 5))
    msgs = None
    plan_view = None
    for frame in range(frames + 1):
      t = frame * dt
      v_lead = float(np.interp(t, maneuver.breakpoints, maneuver.speed_lead_values))
      v_cruise_ms = float(np.interp(t, maneuver.breakpoints, maneuver.cruise_values))
      pitch = float(np.interp(t, maneuver.breakpoints, maneuver.pitch_values))
      prob_lead = float(np.interp(t, maneuver.breakpoints, maneuver.prob_lead_values))
      prob_throttle = float(np.interp(t, maneuver.breakpoints, maneuver.prob_throttle_values))
      standstill = self.ego_v < 0.01

      if frame % PLANNER_PERIOD == 0:
        a_lead = (v_lead - self.v_lead_prev) / (PLANNER_PERIOD * dt)
        self.v_lead_prev = v_lead
        if maneuver.lead_relevancy:
          status = True if maneuver.only_radar else prob_lead > 0.5
          lead = {
            "d_rel": max(0.0, self.lead_x - self.ego_x),
            "v_rel": v_lead - self.ego_v,
            "v_lead": v_lead,
            "a_lead": a_lead if not np.isnan(a_lead) else 0.0,
            "prob": prob_lead if status else 0.0,
          }
        else:
          lead = None
        radar_state = self._radar_message(lead)
        msgs = {"radarState": radar_state}
        msgs.update(self._sm_messages(v_cruise_ms, pitch, prob_throttle, standstill))
        self.planner.update(_FakeSM(msgs, frame), self.carrot)
        plan_view = self._plan_view()

      t_since_plan = (frame % PLANNER_PERIOD) * dt
      limits = CarInterface.get_pid_accel_limits(self.cp, self.ego_v, v_cruise_ms)
      radar_view = types.SimpleNamespace(leadOne=msgs["radarState"].leadOne,
                                         leadTwo=msgs["radarState"].leadTwo)
      accel, _, _ = self.long_control.update(
        bool(maneuver.enabled), self._cs_view(v_cruise_ms, standstill), plan_view,
        limits, t_since_plan, radar_view)
      accel = float(accel)

      logs[frame] = (t, self.ego_v, accel, self.ego_x, self.lead_x)
      self.ego_v = max(0.0, self.ego_v + accel * dt)
      self.ego_x = self.ego_x + self.ego_v * dt
      self.lead_x = self.lead_x + v_lead * dt
    return logs


def compute_metrics(logs, name, butter):
  t, v, a = logs[:, 0], logs[:, 1], logs[:, 2]
  gaps = logs[:, 4] - logs[:, 3]
  stand_idx = np.flatnonzero(v < 0.05)
  stopped = bool(stand_idx.size)
  stop_i = int(stand_idx[0]) if stopped else len(t) - 1
  final_gap = float(gaps[stop_i])
  min_gap = float(np.min(gaps))
  if stopped:
    ts = t[stop_i]
    window = (t >= ts - 0.5) & (t <= ts)
    accel_last = float(np.mean(np.abs(a[window]))) if window.any() else None
    jerk_window = (t >= max(0.0, ts - 1.5)) & (t <= ts)
    jerk = np.gradient(a, t)
    peak_jerk = float(np.max(np.abs(jerk[jerk_window]))) if jerk_window.any() else None
    below3 = np.flatnonzero(v < 3.0)
    time_3 = float(ts - t[below3[0]]) if below3.size else None
  else:
    accel_last = peak_jerk = time_3 = None
  return {
    "name": name, "butter": int(butter),
    "final_gap_m": round(final_gap, 3), "min_gap_m": round(min_gap, 3),
    "collision": bool(min_gap < COLLISION_GAP_M), "stopped": stopped,
    "accel_last_0.5s_mean_abs": None if accel_last is None else round(accel_last, 4),
    "peak_jerk_last_1.5s_abs": None if peak_jerk is None else round(peak_jerk, 4),
    "time_3ms_to_standstill_s": None if time_3 is None else round(time_3, 3),
  }


def _stopped_lead(speed_kph, gap_m, duration):
  return Maneuver(
    f"stopped lead {speed_kph} km/h gap {gap_m} m",
    duration=duration,
    initial_speed=speed_kph / 3.6,
    lead_relevancy=1,
    initial_distance_lead=float(gap_m),
    speed_lead_values=[0.0, 0.0],
    breakpoints=[0.0, float(duration)],
  )


def _lead_braking():
  lead_v = 40 / 3.6
  brake_t = 2.0
  stop_t = brake_t + lead_v / 2.0
  return Maneuver(
    "lead 40 km/h braking to stop at 2 m/s^2 gap 25 m",
    duration=35.0,
    initial_speed=lead_v,
    lead_relevancy=1,
    initial_distance_lead=25.0,
    speed_lead_values=[lead_v, lead_v, 0.0, 0.0],
    breakpoints=[0.0, brake_t, stop_t, 35.0],
  )


_A_SCENARIOS = [
  _stopped_lead(20, 30, 20.0),
  _stopped_lead(40, 60, 25.0),
  _stopped_lead(60, 90, 30.0),
]
_B_SCENARIO = _lead_braking()
# The first five maneuvers of the upstream list: the two stopped-car approaches
# and the lead-decel 1/2/3 m/s^2 following cases.
_C_SCENARIOS = create_maneuvers({"e2e": False, "force_decel": False})[:5]

_CASES = ([("a", m) for m in _A_SCENARIOS] + [("b", _B_SCENARIO)]
          + [("c", m) for m in _C_SCENARIOS])
_CASE_IDS = [f"{group}-{maneuver.title}" for group, maneuver in _CASES]


@pytest.mark.parametrize("group,maneuver", _CASES, ids=_CASE_IDS)
def test_butter_stop_maneuver(group, maneuver):
  off = ButterStopLoop(maneuver, butter=0).run()
  on = ButterStopLoop(maneuver, butter=1).run()
  off_metrics = compute_metrics(off, maneuver.title, 0)
  on_metrics = compute_metrics(on, maneuver.title, 1)
  entry = {"scenario": maneuver.title, "group": group,
           "off": off_metrics, "on": on_metrics}
  RESULTS.append(entry)

  if off_metrics["collision"]:
    entry["mark"] = "xfail: baseline OFF collides"
    pytest.xfail("baseline OFF collides")

  assert not on_metrics["collision"], (
    f"{maneuver.title}: ON collides with min gap {on_metrics['min_gap_m']} m")
  off_gap = off_metrics["final_gap_m"]
  on_gap = on_metrics["final_gap_m"]
  assert on_gap >= off_gap - 0.1, f"{maneuver.title}: ON final gap {on_gap} < OFF {off_gap} - 0.1"
  assert on_gap >= 3.0, (
    f"{maneuver.title}: ON final gap {on_gap} < 3.0 m")

  if group == "a":
    off_soft = off_metrics["accel_last_0.5s_mean_abs"]
    on_soft = on_metrics["accel_last_0.5s_mean_abs"]
    assert on_soft is not None
    assert off_soft is not None
    assert on_soft <= 0.45, f"{maneuver.title}: ON last-0.5s mean |accel| {on_soft} > 0.45"
    assert on_soft < off_soft, f"{maneuver.title}: ON last-0.5s mean |accel| {on_soft} not below OFF {off_soft}"

  if group == "a" and "stopped lead 40" in maneuver.title:
    # ButterStop=0 must be identical to the same run with the whole ButterStop
    # code path absent: floor neutralized, creep guard unreachable, hold ramp
    # at the stock stopping rate, and no planner D_EXTRA.
    neutral = ButterStopLoop(maneuver, butter=1, neutralize=True).run()
    equal = bool(np.array_equal(off[:, 2], neutral[:, 2]))
    entry["off_equals_neutralized"] = equal
    assert equal, "ButterStop=0 accel trace differs from the neutralized ON run"


def test_write_butter_sim_table():
  out_path = os.environ.get("BUTTER_SIM_OUT", "/tmp/butter_sim.json")
  table = {"scenarios": RESULTS, "note": "desktop closed-loop simulation only; no vehicle validation"}
  text = json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False)
  print(text)
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(text + "\n")
