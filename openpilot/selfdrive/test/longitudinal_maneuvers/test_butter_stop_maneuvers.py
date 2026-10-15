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

PASS criteria (enforced by _case_failures):
- ALL scenarios: no collision; hold_creep <= 0.05 m (when stopped); final_gap_on >= min(3.0, final_gap_off) − 1e-9.
- Normal (groups a/b/c, not gated): stopped; |gap_on − gap_off| <= 0.30; soft_on <= 0.30 OR
  soft_on <= 0.70 × soft_off; peak_jerk_on <= max(peak_jerk_off + 0.1, 0.81);
  whole-run peak |jerk| ON <= max(OFF + 0.1, 0.81).
- Gated (groups a/b/c, floor never active): stopped; gap_off − 0.10 <= gap_on <= gap_off + D_EXTRA + 0.20
  (stopping further back is the safe direction when the floor is gated off); soft_on <= soft_off + 0.05;
  peak_jerk_on <= max(peak_jerk_off + 0.1, 0.81);
  whole-run peak |jerk| ON <= max(OFF + 0.1, 0.81).
- Safety s1/s3/s3b/s3c: gap_on >= gap_off − 0.30 and min_gap_on >= min(3.0, min_gap_off) − 0.10;
  s3/s3b/s3c additionally: peak_jerk_last_1.5s ON <= max(OFF + 0.2, 0.81) (smoothness under noise);
  whole-run peak |jerk| ON <= max(OFF + 0.1, 0.81).
  s2: min_gap_on >= min(3.0, min_gap_off) − 0.10; whole-run peak |jerk| ON <= OFF + 0.2 (metric
  `peak_jerk_run_abs`); `stopped` not required.

A scenario is gated when the ON run has floor_active_steps == 0. A scenario
whose OFF baseline collides or never stops has no comparable baseline and is
skipped. The tuning sweep fixes LEAD_SLACK 1.2, MARGIN 0.0, V_TAPER 3.0,
A_MIN 0.10, A_MAX 1.0 and covers D_EXTRA {0.4, 0.6, 0.8} x K_TAPER
{0.4, 0.5, 0.6} x TAU_LEAD {0.3, 0.6} x FLOOR_RELAX_JERK {0.8, 1.5}
(36 combos, all 14 scenarios) and auto-picks the passing combo with the
smallest D_EXTRA, then the lowest mean soft_on over non-gated normal
scenarios, then the larger TAU_LEAD.
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
from openpilot.selfdrive.controls.lib import butter_stop
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  PLANT_A_GATE, PLANT_TAU, plant_creep,
)
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
  """ButterStop code path absent for the planner: pins butter_stop=False.

  This also suppresses the v3 long_mpc stop-point shift, so the neutralized
  ON run matches the OFF baseline exactly.
  """

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
  def __init__(self, maneuver, butter, neutralize=False, plant=False):
    _seed_param_defaults()
    params = Params()
    params.put_int("ButterStop", int(butter))

    self.maneuver = maneuver
    self.neutralize = neutralize
    self.plant = bool(plant)
    self._plant_lag = 0.0
    self.radar_noise_std = float(getattr(maneuver, "radar_noise_std", 0.0))
    self._radar_rng = np.random.default_rng(int(getattr(maneuver, "radar_noise_seed", 0)))
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
      # The _NeutralCarrot wrapper also suppresses the v3 long_mpc stop shift.
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
    logs = np.zeros((frames + 1, 12))
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
          d_rel = max(0.0, self.lead_x - self.ego_x)
          if self.radar_noise_std > 0.0:
            # Same per-scenario seed for OFF and ON: one draw per radar message.
            d_rel += float(self._radar_rng.normal(0.0, self.radar_noise_std))
          lead = {
            "d_rel": d_rel,
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
      if self.plant:
        self._plant_lag += (dt / PLANT_TAU) * (accel - self._plant_lag)
        applied = self._plant_lag
        if accel < PLANT_A_GATE and self.ego_v > 0.01:
          applied += plant_creep(self.ego_v)
      else:
        applied = accel
      floor_active = 1.0 if self.long_control.butter_floor_last is not None else 0.0
      floor_val = self.long_control.butter_floor_last
      floor_raw = self.long_control.butter_floor_raw_last
      bound = self.long_control.butter_bound_last
      bound_code = -1.0 if floor_active == 0.0 else (1.0 if bound == "lead" else 0.0)
      a_end = self.long_control.butter_a_end_last

      logs[frame] = (t, self.ego_v, accel, self.ego_x, self.lead_x, floor_active,
                     floor_val if floor_val is not None else 0.0, bound_code,
                     floor_raw if floor_raw is not None else 0.0,
                     a_end if a_end is not None else 0.0,
                     applied, float(self.long_control.long_control_state))
      self.ego_v = max(0.0, self.ego_v + applied * dt)
      self.ego_x = self.ego_x + self.ego_v * dt
      self.lead_x = self.lead_x + v_lead * dt
    return logs


def compute_metrics(logs, name, butter):
  t, v, a = logs[:, 0], logs[:, 1], logs[:, 2]
  gaps = logs[:, 4] - logs[:, 3]
  floor_active_steps = int(np.sum(logs[:, 5] > 0.5))
  stand_idx = np.flatnonzero(v < 0.05)
  stopped = bool(stand_idx.size)
  stop_i = int(stand_idx[0]) if stopped else len(t) - 1
  final_gap = float(gaps[stop_i])
  hold_creep = float(final_gap - np.min(gaps[stop_i:])) if stopped else None
  min_gap = float(np.min(gaps))
  jerk = np.gradient(a, t) if t.size > 1 else np.zeros_like(a)
  peak_jerk_run = float(np.max(np.abs(jerk))) if t.size > 1 else 0.0
  floor_seq = logs[:, 6][logs[:, 5] > 0.5]
  floor_switches = int(np.sum(np.abs(np.diff(floor_seq)) > 0.01)) if floor_seq.size > 1 else 0
  floor_raw_seq = logs[:, 8][logs[:, 5] > 0.5]
  floor_raw_switches = int(np.sum(np.abs(np.diff(floor_raw_seq)) > 0.01)) if floor_raw_seq.size > 1 else 0
  bound_seq = logs[:, 7][logs[:, 5] > 0.5]
  bound_switches = int(np.sum(np.diff(bound_seq.astype(int)) != 0)) if bound_seq.size > 1 else 0
  if stopped:
    ts = t[stop_i]
    window = (t >= ts - 0.5) & (t <= ts)
    accel_last = float(np.mean(np.abs(a[window]))) if window.any() else None
    jerk_window = (t >= max(0.0, ts - 1.5)) & (t <= ts)
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
    "peak_jerk_run_abs": round(peak_jerk_run, 4),
    "time_3ms_to_standstill_s": None if time_3 is None else round(time_3, 3),
    "hold_creep_m": None if hold_creep is None else round(hold_creep, 3),
    "floor_active_steps": floor_active_steps,
    "floor_relax_switches": floor_switches,
    "floor_raw_switches": floor_raw_switches,
    "bound_switches": bound_switches,
  }


def _stopped_lead(speed_kph, gap_m, duration, title=None):
  return Maneuver(
    title or f"stopped lead {speed_kph} km/h gap {gap_m} m",
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

# Safety scenarios. s3/s3b/s3c add Gaussian noise to dRel on every radar
# message at different seeds/levels; both runs of a scenario use the scenario's
# seed so ON and OFF see the identical noise sequence. The noise hook lives in
# ButterStopLoop (maneuver.radar_noise_std / radar_noise_seed); Maneuver itself
# is untouched.
_S1 = _stopped_lead(15, 9, 20.0, title="s1 short-gap stopped lead 15 km/h gap 9 m")
_S2 = Maneuver(
  "s2 lead pulls away: stopped at 30 m, ego 20 km/h, lead 0->5 m/s at 6 s",
  duration=20.0,
  initial_speed=20 / 3.6,
  lead_relevancy=1,
  initial_distance_lead=30.0,
  speed_lead_values=[0.0, 0.0, 5.0, 5.0],
  breakpoints=[0.0, 6.0, 9.0, 20.0],
)
_S3 = _stopped_lead(40, 60, 25.0, title="s3 noisy radar stopped lead 40 km/h gap 60 m")
_S3.radar_noise_std = 0.3
_S3.radar_noise_seed = 0
_S3B = _stopped_lead(40, 60, 25.0, title="s3b noisy radar seed 1 stopped lead 40 km/h gap 60 m")
_S3B.radar_noise_std = 0.3
_S3B.radar_noise_seed = 1
_S3C = _stopped_lead(40, 60, 25.0, title="s3c noisy radar std 0.15 seed 2 stopped lead 40 km/h gap 60 m")
_S3C.radar_noise_std = 0.15
_S3C.radar_noise_seed = 2

_CASES = ([("a", m) for m in _A_SCENARIOS] + [("b", _B_SCENARIO)]
          + [("c", m) for m in _C_SCENARIOS]
          + [("s1", _S1), ("s2", _S2), ("s3", _S3), ("s3b", _S3B), ("s3c", _S3C)])
_CASE_IDS = [f"{group}-{maneuver.title}" for group, maneuver in _CASES]


def _case_failures(group, off_metrics, on_metrics):
  """ButterStop v3.1 PASS criteria for one scenario; an empty list means it passes.

  The criteria are listed verbatim in the module docstring; they implement:
  ALL: no collision; hold_creep <= 0.05 m; final_gap_on >= min(3.0, final_gap_off) − 1e-9.
  Normal a/b/c (not gated): stopped; |gap_on − gap_off| <= 0.30; soft_on <= 0.30 OR <= 0.70 × soft_off;
  peak_jerk_on <= max(peak_jerk_off + 0.1, 0.81).
  Gated a/b/c: stopped; gap_off − 0.10 <= gap_on <= gap_off + D_EXTRA + 0.20; soft_on <= soft_off + 0.05;
  peak_jerk_on <= max(peak_jerk_off + 0.1, 0.81).
  s1/s3/s3b/s3c: gap_on >= gap_off − 0.30; min_gap_on >= min(3.0, min_gap_off) − 0.10;
  s3/s3b/s3c additionally peak_jerk_last_1.5s ON <= max(OFF + 0.2, 0.81).
  s2: min_gap_on >= min(3.0, min_gap_off) − 0.10; peak_jerk_run_abs ON <= OFF + 0.2; stopped not required.
  An OFF baseline that collides or never stops is skipped (no comparable baseline).
  """
  title = on_metrics["name"]
  fails = []
  if off_metrics["collision"] or not off_metrics["stopped"]:
    return fails
  if on_metrics["collision"]:
    fails.append(f"{title}: ON collides with min gap {on_metrics['min_gap_m']} m")
  if on_metrics["stopped"]:
    creep = on_metrics["hold_creep_m"]
    if creep is None or creep > 0.05 + 1e-9:
      fails.append(f"{title}: ON hold creep {creep} > 0.05 m")
  off_gap = off_metrics["final_gap_m"]
  on_gap = on_metrics["final_gap_m"]
  if on_gap < min(3.0, off_gap) - 1e-9:
    fails.append(f"{title}: ON final gap {on_gap} < min(3.0, OFF {off_gap})")

  if group == "s1" or group.startswith("s3"):
    if on_gap < off_gap - 0.30 - 1e-9:
      fails.append(f"{title}: ON final gap {on_gap} < OFF {off_gap} - 0.30")
    min_bound = min(3.0, off_metrics["min_gap_m"]) - 0.10
    if on_metrics["min_gap_m"] < min_bound - 1e-9:
      fails.append(f"{title}: ON min gap {on_metrics['min_gap_m']} < min(3.0, OFF {off_metrics['min_gap_m']}) - 0.10")
    if group.startswith("s3"):
      off_jerk = off_metrics["peak_jerk_last_1.5s_abs"]
      on_jerk = on_metrics["peak_jerk_last_1.5s_abs"]
      limit = off_jerk + 0.2 if off_jerk is not None else None
      if limit is not None:
        limit = max(limit, 0.81)
      if on_jerk is None or limit is None:
        fails.append(f"{title}: missing peak jerk")
      elif on_jerk > limit + 1e-9:
        fails.append(f"{title}: ON peak jerk last 1.5 s {on_jerk} > max(OFF {off_jerk} + 0.2, 0.81)")
    off_run_jerk = off_metrics["peak_jerk_run_abs"]
    on_run_jerk = on_metrics["peak_jerk_run_abs"]
    run_limit = max(off_run_jerk + 0.1, 0.81)
    if on_run_jerk > run_limit + 1e-9:
      fails.append(f"{title}: ON peak |jerk| run {on_run_jerk} > max(OFF {off_run_jerk} + 0.1, 0.81)")
    return fails

  if group == "s2":
    min_bound = min(3.0, off_metrics["min_gap_m"]) - 0.10
    if on_metrics["min_gap_m"] < min_bound - 1e-9:
      fails.append(f"{title}: ON min gap {on_metrics['min_gap_m']} < min(3.0, OFF {off_metrics['min_gap_m']}) - 0.10")
    if on_metrics["peak_jerk_run_abs"] > off_metrics["peak_jerk_run_abs"] + 0.2 + 1e-9:
      fails.append(f"{title}: ON peak |jerk| run {on_metrics['peak_jerk_run_abs']} > OFF {off_metrics['peak_jerk_run_abs']} + 0.2")
    return fails

  # groups a/b/c
  if not on_metrics["stopped"]:
    fails.append(f"{title}: ON never stops")
    return fails
  off_soft = off_metrics["accel_last_0.5s_mean_abs"]
  on_soft = on_metrics["accel_last_0.5s_mean_abs"]
  gated = on_metrics["floor_active_steps"] == 0
  if gated:
    if on_gap < off_gap - 0.10 - 1e-9:
      fails.append(f"{title}: gated ON final gap {on_gap} < OFF {off_gap} - 0.10")
    if on_gap > off_gap + butter_stop.D_EXTRA + 0.20 + 1e-9:
      fails.append(f"{title}: gated ON final gap {on_gap} > OFF {off_gap} + D_EXTRA + 0.20")
    if off_soft is None or on_soft is None:
      fails.append(f"{title}: missing last-0.5s accel")
    elif on_soft > off_soft + 0.05 + 1e-9:
      fails.append(f"{title}: gated ON last-0.5s mean |accel| {on_soft} > OFF {off_soft} + 0.05")
  else:
    if abs(on_gap - off_gap) > 0.30 + 1e-9:
      fails.append(f"{title}: |final gap ON {on_gap} - OFF {off_gap}| > 0.30 m")
    if off_soft is None or on_soft is None:
      fails.append(f"{title}: missing last-0.5s accel")
    elif not (on_soft <= 0.30 + 1e-9 or on_soft <= 0.70 * off_soft + 1e-9):
      fails.append(f"{title}: ON last-0.5s mean |accel| {on_soft} > 0.30 and > 0.70 x OFF {off_soft}")
  off_jerk = off_metrics["peak_jerk_last_1.5s_abs"]
  on_jerk = on_metrics["peak_jerk_last_1.5s_abs"]
  if off_jerk is None or on_jerk is None:
    fails.append(f"{title}: missing peak jerk")
  elif on_jerk > max(off_jerk + 0.1, 0.81) + 1e-9:
    fails.append(f"{title}: ON peak jerk {on_jerk} > max(OFF {off_jerk} + 0.1, 0.81)")
  off_run_jerk = off_metrics["peak_jerk_run_abs"]
  on_run_jerk = on_metrics["peak_jerk_run_abs"]
  run_limit = max(off_run_jerk + 0.1, 0.81)
  if on_run_jerk > run_limit + 1e-9:
    fails.append(f"{title}: ON peak |jerk| run {on_run_jerk} > max(OFF {off_run_jerk} + 0.1, 0.81)")
  return fails


@pytest.mark.parametrize("group,maneuver", _CASES, ids=_CASE_IDS)
def test_butter_stop_maneuver(group, maneuver):
  off = ButterStopLoop(maneuver, butter=0).run()
  on = ButterStopLoop(maneuver, butter=1).run()
  off_metrics = compute_metrics(off, maneuver.title, 0)
  on_metrics = compute_metrics(on, maneuver.title, 1)
  entry = {"scenario": maneuver.title, "group": group,
           "off": off_metrics, "on": on_metrics,
           "gated": group in ("a", "b", "c") and on_metrics["floor_active_steps"] == 0}
  RESULTS.append(entry)

  if off_metrics["collision"]:
    entry["mark"] = "xfail: baseline OFF collides"
    pytest.xfail("baseline OFF collides")

  if off_metrics["final_gap_m"] < 3.0:
    entry["mark"] = "baseline OFF final gap below 3.0 m"
  fails = _case_failures(group, off_metrics, on_metrics)
  assert not fails, "; ".join(fails)

  if group == "a" and "stopped lead 40" in maneuver.title:
    # ButterStop=0 must be identical to the same run with the whole ButterStop
    # code path absent: floor neutralized, creep guard unreachable, hold ramp
    # at the stock stopping rate and the v3 planner shift suppressed by the
    # neutralized carrot's butter_stop=False.
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


_SWEEP_D_EXTRA = (0.4, 0.6, 0.8)
_SWEEP_K_TAPER = (0.4, 0.5, 0.6)
_SWEEP_TAU_LEAD = (0.3, 0.6)
_SWEEP_FLOOR_RELAX_JERK = (0.8, 1.5)
_SWEEP_FIXED = {"LEAD_SLACK": 1.2, "MARGIN": 0.0, "V_TAPER": 3.0, "A_MIN": 0.10, "A_MAX": 1.0}

_SWEEP_CRITERIA = [
  "ALL: no collision",
  "ALL: hold_creep <= 0.05 m (when stopped)",
  "ALL: final_gap_on >= min(3.0, final_gap_off) - 1e-9",
  "a/b/c normal: stopped",
  "a/b/c normal: |gap_on - gap_off| <= 0.30 m",
  "a/b/c normal: soft_on <= 0.30 or soft_on <= 0.70 x soft_off",
  "a/b/c normal: peak_jerk_on <= max(peak_jerk_off + 0.1, 0.81)",
  "a/b/c normal: peak_jerk_run ON <= max(OFF + 0.1, 0.81)",
  "a/b/c gated: stopped",
  "a/b/c gated: gap_off - 0.10 <= gap_on <= gap_off + D_EXTRA + 0.20",
  "a/b/c gated: soft_on <= soft_off + 0.05",
  "a/b/c gated: peak_jerk_on <= max(peak_jerk_off + 0.1, 0.81)",
  "a/b/c gated: peak_jerk_run ON <= max(OFF + 0.1, 0.81)",
  "s1/s3/s3b/s3c: gap_on >= gap_off - 0.30 and min_gap_on >= min(3.0, min_gap_off) - 0.10",
  "s3/s3b/s3c: peak_jerk_last_1.5s ON <= max(OFF + 0.2, 0.81)",
  "s1/s3/s3b/s3c: peak_jerk_run ON <= max(OFF + 0.1, 0.81)",
  "s2: min_gap_on >= min(3.0, min_gap_off) - 0.10",
  "s2: peak_jerk_run_abs ON <= OFF + 0.2",
]


def _sweep_output_path():
  """Where the sweep writes: BUTTER_SWEEP_OUT, else next to BUTTER_SIM_OUT.

  CI sets BUTTER_SIM_OUT for the butter-sim artifact, so mapping the sweep
  file next to it lets the same artifact carry butter_sweep.json.
  """
  out = os.environ.get("BUTTER_SWEEP_OUT")
  if out:
    return out
  sim_out = os.environ.get("BUTTER_SIM_OUT")
  if sim_out:
    return os.path.join(os.path.dirname(sim_out), "butter_sweep.json")
  return None


@pytest.mark.skipif(_sweep_output_path() is None,
                    reason="tuning sweep runs only when a sweep or sim output path is set")
def test_butter_stop_tuning_sweep(monkeypatch):
  """Sweep D_EXTRA x K_TAPER x TAU_LEAD x FLOOR_RELAX_JERK over every scenario.

  PASS criteria per combo and scenario: _case_failures (same criteria as the
  normal maneuver table; the butter_stop module is monkeypatched, and long_mpc,
  LeadPosFilter and slew_floor read their constants through the module at call
  time). Auto-pick: among the passing combos the smallest D_EXTRA, then the
  lowest mean soft_on over non-gated normal scenarios, then the larger
  TAU_LEAD. When no combo passes, the closest three are reported by (failure
  count, mean soft_on) with the failing criteria and nothing is shipped.
  """
  off_cache = {m.title: compute_metrics(ButterStopLoop(m, butter=0).run(), m.title, 0) for _, m in _CASES}
  rows = []
  for d_extra in _SWEEP_D_EXTRA:
    for k_taper in _SWEEP_K_TAPER:
      for tau_lead in _SWEEP_TAU_LEAD:
        for floor_relax_jerk in _SWEEP_FLOOR_RELAX_JERK:
          monkeypatch.setattr(butter_stop, "D_EXTRA", d_extra)
          monkeypatch.setattr(butter_stop, "K_TAPER", k_taper)
          monkeypatch.setattr(butter_stop, "TAU_LEAD", tau_lead)
          monkeypatch.setattr(butter_stop, "FLOOR_RELAX_JERK", floor_relax_jerk)
          for name, value in _SWEEP_FIXED.items():
            monkeypatch.setattr(butter_stop, name, value)
          cases = []
          fails = []
          gated = []
          normal_softs = []
          normal_dgaps = []
          for group, m in _CASES:
            off_metrics = off_cache[m.title]
            on_metrics = compute_metrics(ButterStopLoop(m, butter=1).run(), m.title, 1)
            case_fails = _case_failures(group, off_metrics, on_metrics)
            fails += case_fails
            is_normal = group in ("a", "b", "c")
            is_gated = is_normal and on_metrics["floor_active_steps"] == 0
            if is_gated:
              gated.append(m.title)
            if is_normal and not is_gated:
              if on_metrics["accel_last_0.5s_mean_abs"] is not None:
                normal_softs.append(on_metrics["accel_last_0.5s_mean_abs"])
              if off_metrics["stopped"] and not off_metrics["collision"]:
                normal_dgaps.append(abs(on_metrics["final_gap_m"] - off_metrics["final_gap_m"]))
            cases.append({"scenario": m.title, "group": group, "fails": case_fails,
                          "off": off_metrics, "on": on_metrics})
          rows.append({
            "d_extra": d_extra, "k_taper": k_taper, "tau_lead": tau_lead,
            "floor_relax_jerk": floor_relax_jerk,
            "all_pass": not fails, "failures": fails, "gated": gated,
            "mean_soft_on": round(float(np.mean(normal_softs)), 5) if normal_softs else None,
            "max_abs_dgap": round(float(np.max(normal_dgaps)), 4) if normal_dgaps else None,
            "cases": cases,
          })

  passing = [r for r in rows if r["all_pass"]]
  chosen = None
  if passing:
    chosen = min(passing, key=lambda r: (r["d_extra"],
                                         r["mean_soft_on"] if r["mean_soft_on"] is not None else 1e9,
                                         -r["tau_lead"]))
  closest = sorted((r for r in rows if not r["all_pass"]),
                   key=lambda r: (len(r["failures"]),
                                  r["mean_soft_on"] if r["mean_soft_on"] is not None else 1e9))[:3]

  for r in rows:
    label = (f'D {r["d_extra"]} K {r["k_taper"]} TAU {r["tau_lead"]} RJ {r["floor_relax_jerk"]}')
    stats = f'pass={r["all_pass"]} fails={len(r["failures"])} gated={len(r["gated"])}'
    print(f'sweep {label}: {stats} soft_mean={r["mean_soft_on"]} max_dgap={r["max_abs_dgap"]}')
  if chosen is not None:
    label = f'D {chosen["d_extra"]} K {chosen["k_taper"]} TAU {chosen["tau_lead"]} RJ {chosen["floor_relax_jerk"]}'
    print(f'chosen: {label} soft_mean {chosen["mean_soft_on"]} max_dgap {chosen["max_abs_dgap"]}')
  else:
    print("no combo passed all criteria; closest:")
    for r in closest:
      print(f'  D_EXTRA {r["d_extra"]} K_TAPER {r["k_taper"]} TAU_LEAD {r["tau_lead"]} FLOOR_RELAX_JERK {r["floor_relax_jerk"]}:')
      for failure in r["failures"]:
        print(f'    {failure}')

  text = json.dumps({"criteria": _SWEEP_CRITERIA, "sweep": rows,
                     "chosen": chosen, "closest": closest}, ensure_ascii=False, indent=2)
  with open(_sweep_output_path(), "w", encoding="utf-8") as handle:
    handle.write(text + "\n")
