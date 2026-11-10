import math
import numpy as np
from openpilot.cereal import car
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.cruise_coasting import (
  CruiseCoastingControl, MAX_PLAN_AGE, coasting_relief, no_coasting_lead,
)
from openpilot.selfdrive.controls.lib import butter_stop as butter_stop_module
from openpilot.selfdrive.controls.lib.butter_stop import (
  CREEP_ACCEL, CREEP_V, EARLY_REQ_ENABLED, EARLY_V, FINAL_A0_DIST, FINAL_A0_MAX, FINAL_A0_MIN,
  FINAL_ROLL_RESET_V, FINAL_ROLL_V, FINAL_ROLL_VLEAD, FINAL_U_MAX,
  FINAL_U_MIN, FINAL_WINDOW_ENABLED, HOLD_RAMP, LeadPosFilter, MIN_FINAL_GAP_ABS,
  PROBE_GUARD_T, PROBE_GUARD_V, PROBE_MAX_T, PROBE_V_MAX, STANDSTILL_T, STANDSTILL_V,
  STOP_DISTANCE_DEFAULT, butter_eligible, butter_floor, early_req_gap_ok,
  final_profile_travel, final_roll_limit, plan_stop_distance, probe_accel, probe_weak,
  rest_ramp_floor,
)

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

HYUNDAI_LONGITUDINAL_KP = 1.0
HYUNDAI_LONGITUDINAL_KI = 0.0
HYUNDAI_LONGITUDINAL_KF = 1.0

# Departure guard: right after leaving a stop the car surges ahead of vTargetNow (the planner's
# v_desired_filter lags by ~2 s) and the speed-error term cut the command to ~0 while the plan
# still accelerated, so the K5 hesitated for ~1 s (route 6db). While the lead pulls away, the
# command stays >= aTarget + DEPART_P_FLOOR for DEPART_WINDOW_T after leaving rest.
DEPART_START_V = 0.3       # m/s, the stop -> pid transition must start from rest
DEPART_WINDOW_T = 4.0      # s
DEPART_P_FLOOR = -0.2      # m/s^2 below aTarget
DEPART_MIN_DREL = 4.0      # m
DEPART_RELEASE_JERK = 1.0  # m/s^3, floor relax when the window ends or the plan stops accelerating

# ButterStopFinal firm finish (classic-CAN Hyundai/Kia): hold at least FIRM_A of
# braking through the last metres of a planned stop so the car is nearly stopped
# when the Palisade ESC standstill timer (0.51 s after 0.794 m/s) releases the
# brakes. Off unless ButterStop and ButterStopFinal are both on.
FIRM_V_HI = 2.0            # m/s, band top (car responds ~0.6 s late)
FIRM_V_LO = 0.3            # m/s, band bottom
FIRM_ATGT = -0.3           # m/s^2, plan must command real braking (P7)
FIRM_VPLAN_END = 0.3       # m/s, plan must end at a stop (P7)
FIRM_JERK = 2.0            # m/s^3, cap build rate
FIRM_RELEASE_JERK = 2.0    # m/s^3, cap relax rate on a soft exit
FIRM_A = 0.8               # m/s^2, ButterStopFinal firm-finish decel on classic-CAN Hyundai/Kia


def firm_gate(firm_a, v_ego, a_target, plan_speeds, t_since_plan, gas, brake, cruise_standstill):
  if firm_a <= 0.0 or gas or brake or cruise_standstill:
    return False
  if not (0.0 <= t_since_plan <= MAX_PLAN_AGE) or len(plan_speeds) == 0:
    return False
  v_end = float(plan_speeds[-1])
  if not (math.isfinite(v_ego) and math.isfinite(a_target) and math.isfinite(v_end)):
    return False
  return a_target < FIRM_ATGT and v_end < FIRM_VPLAN_END and FIRM_V_LO <= v_ego <= FIRM_V_HI


def firm_cap_step(cap, last_output, firm_a, gate, hard_exit):
  if gate:
    base = last_output if cap is None else cap
    return max(-firm_a, min(float(base), 0.0) - FIRM_JERK * DT_CTRL)
  if cap is None or hard_exit:
    return None
  cap += FIRM_RELEASE_JERK * DT_CTRL
  return None if cap >= 0.0 else cap


STOPPING_ACCEL_DEFAULT = -50  # Params use hundredths of m/s^2
STOPPING_ACCEL_MIN = -100
STOPPING_ACCEL_MAX = -50

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill, a_ego, stopping_accel, radarState,
                             final_hold=False, early_handoff=False):
  stopping_condition = should_stop
  stopping_accel = stopping_accel if stopping_accel < 0.0 else -0.5
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        leadOne = radarState.leadOne
        fcw_stop = leadOne.status and leadOne.dRel < 4.0
        if (fcw_stop and not final_hold) or (a_ego > stopping_accel and not final_hold) or early_handoff:
          long_control_state = LongCtrlState.stopping
        elif long_control_state == LongCtrlState.starting:
          long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state

class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             k_f=CP.longitudinalTuning.kf, rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.coasting = CruiseCoastingControl()


    self.params = Params()
    self.readParamCount = 0
    self.final_stop = False
    self.final_window = False
    # Which final-stop behaviour ButterStopFinal selects. The early hand-off is
    # for the classic-CAN Palisade ESC, which flags standstill while the wheels
    # still roll. A CAN-FD car (K5, ...) flags it only after the physical stop
    # (d_ass median +0.92 s), so an early hand-off there is a plain early StopReq
    # with no upside; those cars, and every other brand, get the final window.
    # Assigned before the first _refresh_butter_stop() call, which reads it.
    self._early_req_capable = CP.brand == "hyundai" and not (CP.flags & HyundaiFlags.CANFD)
    self._final_window_capable = CP.brand == "hyundai" and bool(CP.flags & HyundaiFlags.CANFD)
    self._probe_weak_t = 0.0
    self._probe_guard_logged = False
    self._refresh_stopping_accel()
    self._refresh_butter_stop()
    self._firm_cap = None
    self._firm_logged = False
    self.j_lead = 0.0

    # ButterStop stop-episode state. The standstill hold/release decision itself
    # stays in long_control_state_trans and is not touched here.
    self._butter_standstill_time = 0.0
    self._butter_standstill_confirmed = False
    self._butter_creep_latched = False
    self._butter_creep_logged = False
    self._butter_near_rest = False
    self._butter_rest_floor = None
    self._butter_v_min = float("inf")
    self._depart_t = None
    self._depart_floor = None
    self.depart_guard_last = False
    self.butter_floor_last = None
    self.butter_floor_raw_last = None
    self.butter_bound_last = None
    self.butter_a_end_last = None
    self._butter_floor_prev = None
    self._butter_s_ego = 0.0
    self._butter_lead_filter = LeadPosFilter()
    self._butter_roll_limit = None
    self._butter_diag = {}
    # ButterStopFinal window state. One window per stop: _final_t is the
    # elapsed profile time (None = no window), _final_done blocks a second
    # window in the same stop and clears once the car is moving again.
    self._final_t = None
    self._final_v0 = 0.0
    self._final_a0 = 0.0
    self._final_s_target = 0.0
    self._final_s_trav = 0.0
    self._final_done = False

    # ButterStopFinal early hand-off log latch: one event per stop, cleared
    # above 1 m/s.
    self._early_logged = False

    self.hyundai_fixed_longitudinal_tuning = CP.brand == "hyundai"
    if self.hyundai_fixed_longitudinal_tuning:
      self._apply_hyundai_longitudinal_tuning()
    elif CP.brand == "toyota":
      self._apply_toyota_longitudinal_tuning()

    self.use_accel_pid = False
    if CP.brand == "toyota":
      self.use_accel_pid = True

  def _refresh_stopping_accel(self):
    try:
      value = float(self.params.get_float("StoppingAccel"))
    except (TypeError, ValueError):
      value = STOPPING_ACCEL_DEFAULT
    if not math.isfinite(value):
      value = STOPPING_ACCEL_DEFAULT
    # Enforce the menu bounds even for stale Params or direct writes.
    self.stopping_accel = min(STOPPING_ACCEL_MAX, max(STOPPING_ACCEL_MIN, value)) * 0.01

  def _refresh_butter_stop(self):
    try:
      value = float(self.params.get_int("ButterStop"))
    except (TypeError, ValueError, AttributeError):
      value = 0.0
    self.butter_stop = math.isfinite(value) and value != 0.0
    self.final_stop = False
    self.final_window = False
    self.early_req = False
    self.firm_a = 0.0
    if not self.butter_stop:
      return
    try:
      final_value = float(self.params.get_int("ButterStopFinal"))
    except (TypeError, ValueError, AttributeError):
      final_value = 0.0
    self.final_stop = math.isfinite(final_value) and final_value > 0.0
    # One switch, two behaviours. Classic-CAN Hyundai/Kia get the firm finish:
    # at least FIRM_A of braking through the last 2.0 -> 0.3 m/s, so the car is
    # nearly stopped when the ESC standstill timer releases the brakes. The
    # early StopReq hand-off is held off by EARLY_REQ_ENABLED (W-SS1: a StopReq
    # sent before the flag did not stop the release). CAN-FD Hyundai/Kia drive
    # the K5 measurement probe (W-K5B1): the final window holds pid below
    # PROBE_V_MAX and commands the decel the car's own stopping delivers there,
    # with an under-response guard that hands back to stock. Every other brand
    # gets nothing from this switch.
    self.early_req = self.final_stop and self._early_req_capable and EARLY_REQ_ENABLED
    self.firm_a = FIRM_A if (self.final_stop and self._early_req_capable) else 0.0
    self.final_window = self.final_stop and self._final_window_capable and FINAL_WINDOW_ENABLED
    try:
      dist = float(self.params.get_float("StopDistanceCarrot")) * 0.01
    except (TypeError, ValueError, AttributeError):
      dist = float("nan")
    self.stop_distance_orig = dist if math.isfinite(dist) and dist >= 0.0 else STOP_DISTANCE_DEFAULT

  def _butter_floor(self, active, CS, long_plan, t_since_plan, radarState):
    if not self.butter_stop:
      return None
    self._butter_diag["bound"] = None
    self._butter_diag["a_end"] = None
    try:
      t_age = float(t_since_plan)
    except (TypeError, ValueError):
      return None
    try:
      v_ego = float(CS.vEgo)
    except (TypeError, ValueError, AttributeError):
      return None
    lead = getattr(radarState, "leadOne", None) if radarState is not None else None
    if not butter_eligible(
        active=bool(active), v_ego=v_ego,
        brake_pressed=bool(getattr(CS, "brakePressed", False)),
        gas_pressed=bool(getattr(CS, "gasPressed", False)),
        fcw=bool(getattr(long_plan, "fcw", False)),
        t_since_plan=t_age, lead=lead):
      return None
    plan_speeds = getattr(long_plan, "speeds", None)
    if plan_speeds is None:
      return None
    # The lead bound uses the smoothed lead position (d_eff); every gate above
    # still uses the raw radar values. s_ego is maintained by update().
    d_eff = None
    v_lead = float("nan")
    if lead is not None and bool(getattr(lead, "status", False)):
      try:
        d_rel_raw = float(lead.dRel)
      except (TypeError, ValueError, AttributeError):
        return None
      d_eff = self._butter_lead_filter.update(True, d_rel_raw, self._butter_s_ego)
      if d_eff is None:
        return None
      try:
        v_lead = float(lead.vLead)
      except (TypeError, ValueError, AttributeError):
        v_lead = float("nan")
    else:
      self._butter_lead_filter.reset()
      # A latch taken with a lead in view is invalid without one.
      self._butter_roll_limit = None
    # Final-roll latch: taken once per stop, and only behind a stopped lead.
    lead_stopped = math.isfinite(v_lead) and v_lead < FINAL_ROLL_VLEAD
    roll_avail = None
    if self._butter_roll_limit is not None and (v_ego > FINAL_ROLL_RESET_V or not lead_stopped):
      self._butter_roll_limit = None
    if self._butter_roll_limit is None and d_eff is not None and lead_stopped and v_ego <= FINAL_ROLL_V:
      self._butter_roll_limit = final_roll_limit(v_ego, d_eff, self._butter_s_ego)
    if self._butter_roll_limit is not None:
      roll_avail = min(self._butter_roll_limit - self._butter_s_ego, d_eff - MIN_FINAL_GAP_ABS)
    return butter_floor(v_ego, CONTROL_N_T_IDX, plan_speeds, lead, self.stop_distance_orig,
                        d_rel_eff=d_eff, diag=self._butter_diag, roll_avail=roll_avail)

  def _apply_hyundai_longitudinal_tuning(self):
    # Hyundai, Kia, and Genesis all use the opendbc "hyundai" brand. Keep the
    # complete acceleration/deceleration feedforward path intact instead of
    # allowing a stale or unsafe persistent tuning value to override it.
    self.pid._k_p = ([0.0], [HYUNDAI_LONGITUDINAL_KP])
    self.pid._k_i = ([0.0], [HYUNDAI_LONGITUDINAL_KI])
    self.pid.k_f = HYUNDAI_LONGITUDINAL_KF

  def _refresh_longitudinal_tuning(self):
    if self.hyundai_fixed_longitudinal_tuning:
      self._apply_hyundai_longitudinal_tuning()
    elif self.CP.brand == "toyota":
      self._apply_toyota_longitudinal_tuning()
    elif len(self.CP.longitudinalTuning.kpBP) == 1 and len(self.CP.longitudinalTuning.kiBP) == 1:
      longitudinalTuningKpV = self.params.get_float("LongTuningKpV") * 0.01
      longitudinalTuningKiV = self.params.get_float("LongTuningKiV") * 0.001
      self.pid._k_p = (self.CP.longitudinalTuning.kpBP, [longitudinalTuningKpV])
      self.pid._k_i = (self.CP.longitudinalTuning.kiBP, [longitudinalTuningKiV])
      self.pid.k_f = self.params.get_float("LongTuningKf") * 0.01

  def _apply_toyota_longitudinal_tuning(self):
    # Toyota's CarController closes the acceleration loop at the PCM send rate.
    # Pass the planner target through; a second acceleration PID here feeds
    # measured-acceleration noise and duplicate correction into that loop.
    self.pid._k_p = ([0.0], [0.0])
    self.pid._k_i = ([0.0], [0.0])
    self.pid.k_f = 1.0

  def reset(self):
    self.pid.reset()
    self.coasting.reset()

  def _depart_guard(self, output_accel, a_target_ff, should_stop, CS, radarState):
    self.depart_guard_last = False
    if not self.butter_stop:
      self._depart_t = None
      self._depart_floor = None
      return output_accel
    if self._depart_t is None and self._depart_floor is None:
      return output_accel
    try:
      v_ego = float(CS.vEgo)
      a_ff = float(a_target_ff)
      lead = radarState.leadOne
      lead_ok = (not lead.status) or (float(lead.vLead) >= v_ego and float(lead.dRel) >= DEPART_MIN_DREL)
    except (TypeError, ValueError, AttributeError):
      v_ego = a_ff = float("nan")
      lead_ok = False
    if not (lead_ok and not should_stop and math.isfinite(v_ego) and math.isfinite(a_ff)):
      # Safety exits drop the guard at once and end it for this departure.
      self._depart_t = None
      self._depart_floor = None
      return output_accel
    if self._depart_t is not None and self._depart_t <= DEPART_WINDOW_T and a_ff > 0.0:
      self._depart_floor = a_ff + DEPART_P_FLOOR
    else:
      # Window over or the plan stopped accelerating: end the guard and relax its floor.
      self._depart_t = None
      if self._depart_floor is not None:
        self._depart_floor -= DEPART_RELEASE_JERK * DT_CTRL
    if self._depart_floor is None or output_accel >= self._depart_floor:
      if self._depart_t is None:
        self._depart_floor = None
      return output_accel
    self.depart_guard_last = True
    return self._depart_floor

  def _probe_guard_tripped(self, CS):
    # Under-response guard: the car must brake at least as hard as the weakest
    # tenth of today's stock K5 stops; PROBE_GUARD_T weaker than that hands back.
    try:
      v = float(CS.vEgo)
      a = float(CS.aEgo)
    except (TypeError, ValueError, AttributeError):
      return True
    if not (math.isfinite(v) and math.isfinite(a)):
      return True
    if v >= PROBE_GUARD_V and a > probe_weak(v):
      self._probe_weak_t += DT_CTRL
    else:
      self._probe_weak_t = 0.0
    if self._probe_weak_t < PROBE_GUARD_T - 1e-9:
      return False
    if not self._probe_guard_logged:
      cloudlog.event("butterStopProbeGuard", vEgo=v, aEgo=a, t=float(self._final_t))
      self._probe_guard_logged = True
    return True

  def _final_window_accel(self, CS, long_plan, radarState):
    """One tick of the K5 measurement window, then advance its clock.

    Commands the decel the K5's own stopping state delivers at this speed
    (probe_accel over the 44-stop table), so the drive records how the car
    answers our pid below 0.3 m/s at that strength. The lead guard stays at
    full strength against the remaining gap: the gap is a hard safety budget
    and may brake harder than the probe. long_plan is no longer read. The PID
    integrator is reset because nothing in this window is a PID decision.
    """
    try:
      v_ego = float(CS.vEgo)
    except (TypeError, ValueError, AttributeError):
      v_ego = float("nan")
    u = probe_accel(v_ego) if math.isfinite(v_ego) else probe_accel(PROBE_V_MAX)
    # Lead guard: unchanged and at full strength. The gap is a hard safety
    # budget, so it is allowed to brake harder than the probe table.
    lead = getattr(radarState, "leadOne", None) if radarState is not None else None
    if lead is not None and bool(getattr(lead, "status", False)):
      try:
        gap = float(lead.dRel) - MIN_FINAL_GAP_ABS
      except (TypeError, ValueError, AttributeError):
        gap = None
      if gap is not None and math.isfinite(gap) and math.isfinite(v_ego):
        u = min(u, -(v_ego * v_ego) / (2.0 * max(gap, 0.05)))
    self.reset()
    if math.isfinite(v_ego):
      self._final_s_trav += max(v_ego, 0.0) * DT_CTRL
    self._final_t += DT_CTRL
    return float(np.clip(u, FINAL_U_MIN, FINAL_U_MAX))

  def update(self, active, CS, long_plan, accel_limits, t_since_plan, radarState):

    soft_hold_active = CS.softHoldActive > 0
    a_target_ff = long_plan.aTarget
    v_target_now = long_plan.vTargetNow
    j_target_now = long_plan.jTargetNow
    should_stop = long_plan.shouldStop

    self.readParamCount += 1
    if self.readParamCount >= 100:
      self.readParamCount = 0
      self._refresh_stopping_accel()
      self._refresh_butter_stop()
    elif self.readParamCount == 10:
      self._refresh_longitudinal_tuning()


    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    # ButterStopFinal window, resolved before the state machine so final_hold
    # can suppress the ordinary hand-off this tick. The window keeps its own
    # lead threshold: the 4 m fcw radius would end it while the distance guard
    # is still able to stop the car inside the lead gap, and the ordinary
    # stopping state that follows is exactly the hard end this removes.
    # Partial stand-ins (no leadOne, or a lead without dRel) are never close.
    _lead = getattr(radarState, "leadOne", None)
    _d_rel = float(getattr(_lead, "dRel", float("inf")))
    _lead_on = bool(getattr(_lead, "status", False))
    final_close = _lead_on and _d_rel < MIN_FINAL_GAP_ABS
    if self._final_t is not None:
      exit_window = (not (active and should_stop and self.final_window)
                     or self.long_control_state != LongCtrlState.pid
                     or final_close
                     or CS.vEgo < STANDSTILL_V
                     or self._final_t >= PROBE_MAX_T
                     or self._probe_guard_tripped(CS))
      if exit_window:
        self._final_t = None
        self._final_done = True
    elif (not self._final_done and self.final_window and active and should_stop
          and self.long_control_state == LongCtrlState.pid
          and CS.aEgo > self.stopping_accel and CS.vEgo <= PROBE_V_MAX
          and not final_close):
      self._final_t = 0.0
      self._final_v0 = max(float(CS.vEgo), 0.05)
      self._final_a0 = min(FINAL_A0_MAX, max(FINAL_A0_DIST, FINAL_A0_MIN, -float(self.last_output_accel)))
      self._probe_weak_t = 0.0
      self._probe_guard_logged = False
      # Freeze the plan target at entry. Once the plan's speeds reach zero the
      # live plan_stop_distance collapses to ~0; the frozen target stays
      # meaningful for the whole window, and the profile's own travel is the
      # floor, so a plan that has already stopped short cannot shorten it.
      s_plan0 = None
      plan_speeds = getattr(long_plan, "speeds", None)
      if plan_speeds is not None:
        s_plan0 = plan_stop_distance(CONTROL_N_T_IDX, plan_speeds)
      travel = final_profile_travel(self._final_v0, self._final_a0)
      self._final_s_target = max(
        s_plan0 if (s_plan0 is not None and math.isfinite(s_plan0)) else 0.0, travel)
      self._final_s_trav = 0.0
      cloudlog.event("butterStopFinal", vEgo=float(CS.vEgo), a0=float(self._final_a0),
                     sTarget=float(self._final_s_target))
    if CS.vEgo > 1.0:
      self._final_done = False
    final_hold = self._final_t is not None

    # ButterStopFinal on classic-CAN Hyundai/Kia: hand the stop to the car
    # before its ESC standstill timer runs out. Only with a lead whose gap
    # survives weak braking. final_hold wins, so a final-stop window is never
    # cut short; on those cars the window never opens at all.
    early_handoff = (self.early_req and self._early_req_capable and active and should_stop
                     and self.long_control_state == LongCtrlState.pid
                     and not final_hold and float(CS.vEgo) <= EARLY_V
                     and early_req_gap_ok(float(CS.vEgo), _lead_on, _d_rel))
    if early_handoff and not self._early_logged:
      cloudlog.event("butterStopEarlyReq", vEgo=float(CS.vEgo), dRel=float(_d_rel), aEgo=float(CS.aEgo))
      self._early_logged = True
    if CS.vEgo > 1.0:
      self._early_logged = False

    prev_long_control_state = self.long_control_state
    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill, CS.aEgo, self.stopping_accel, radarState,
                                                       final_hold=final_hold, early_handoff=early_handoff)
    if active and soft_hold_active:
      self.long_control_state = LongCtrlState.stopping

    if self.long_control_state == LongCtrlState.pid:
      if (prev_long_control_state in (LongCtrlState.stopping, LongCtrlState.starting)
          and float(CS.vEgo) < DEPART_START_V):
        self._depart_t = 0.0
        self._depart_floor = None
      elif self._depart_t is not None:
        self._depart_t += DT_CTRL
    else:
      self._depart_t = None
      self._depart_floor = None

    # ButterStop episode bookkeeping. This only decides how much braking the
    # taper may remove while still rolling and how fast the confirmed hold
    # ramps; it never changes the state machine, hold or release conditions.
    butter_floor_value = None
    butter_floor_raw_value = None
    if self.butter_stop:
      if (not active) or self.long_control_state != LongCtrlState.stopping:
        self._butter_standstill_time = 0.0
        self._butter_standstill_confirmed = False
        self._butter_creep_latched = False
        self._butter_creep_logged = False
        self._butter_near_rest = False
        self._butter_rest_floor = None
        self._butter_v_min = float("inf")
      else:
        try:
          v_ego_now = float(CS.vEgo)
        except (TypeError, ValueError):
          v_ego_now = float("nan")
        # The hold is confirmed only after STANDSTILL_T below STANDSTILL_V. CS.standstill
        # (Hyundai: wheel speed <= 0.104 m/s) comes ~0.4 s before the car is at rest, and
        # confirming on it ramped to the hold decel while still rolling (K5 6d5/6d6).
        if v_ego_now < STANDSTILL_V:
          self._butter_standstill_time = min(STANDSTILL_T, self._butter_standstill_time + DT_CTRL)
        else:
          self._butter_standstill_time = 0.0
        if v_ego_now < STANDSTILL_V or bool(getattr(CS, "standstill", False)):
          self._butter_near_rest = True
        if math.isfinite(v_ego_now):
          self._butter_v_min = min(self._butter_v_min, v_ego_now)
        if self._butter_standstill_time >= STANDSTILL_T:
          self._butter_standstill_confirmed = True
      floor_live = (self.long_control_state == LongCtrlState.pid
                    or (self.long_control_state == LongCtrlState.stopping
                        and not self._butter_standstill_confirmed))
      if floor_live:
        # Ego odometry frame for the lead-position filter; it spans the whole
        # floor-active episode (pid and the rolling part of stopping).
        try:
          v_ego_odom = float(CS.vEgo)
        except (TypeError, ValueError):
          v_ego_odom = float("nan")
        if math.isfinite(v_ego_odom):
          self._butter_s_ego += v_ego_odom * DT_CTRL
        butter_floor_raw_value = self._butter_floor(active, CS, long_plan, t_since_plan, radarState)
      else:
        self._butter_s_ego = 0.0
        self._butter_lead_filter.reset()
        self._butter_roll_limit = None
      # One-way floor slew: tightening (more negative) is instant, relaxing
      # (rising toward 0) is rate-limited at FLOOR_RELAX_JERK so radar noise
      # cannot flap the applied braking; that constant is read at call time so
      # the tuning sweep can patch it. On first activation (prev None) the slew
      # is seeded with the previously applied accel, so the floor starts at or
      # below what the car was already doing and relaxes at the same rate.
      floor_prev = self._butter_floor_prev
      if floor_prev is None and butter_floor_raw_value is not None:
        try:
          applied = float(self.last_output_accel)
        except (TypeError, ValueError):
          applied = float("nan")
        floor_prev = min(applied, butter_floor_raw_value) if math.isfinite(applied) else butter_floor_raw_value
      butter_floor_value = butter_stop_module.slew_floor(floor_prev, butter_floor_raw_value)
      self._butter_floor_prev = butter_floor_value
    else:
      self._butter_floor_prev = None
      self._butter_lead_filter.reset()
      self._butter_roll_limit = None
      self._butter_s_ego = 0.0
    self.butter_floor_raw_last = butter_floor_raw_value
    self.butter_floor_last = butter_floor_value
    if butter_floor_raw_value is None:
      self.butter_bound_last = None
      self.butter_a_end_last = None
    else:
      self.butter_bound_last = self._butter_diag.get("bound")
      self.butter_a_end_last = self._butter_diag.get("a_end")

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel

      if soft_hold_active:
        output_accel = self.CP.stopAccel
      # ButterStop ramps to the standstill hold faster once standstill is
      # confirmed; the soft-hold path and every other case keep today's rate.
      ramp_rate = self.CP.stoppingDecelRate
      if self.butter_stop and self._butter_standstill_confirmed and not soft_hold_active:
        ramp_rate = HOLD_RAMP
      # Restore the original one-way ramp. Do not unwind stronger braking.
      if output_accel > self.stopping_accel:
        output_accel = min(output_accel, 0.0)
        output_accel -= ramp_rate * DT_CTRL
      if self.butter_stop and not self._butter_standstill_confirmed:
        floor = None
        if butter_floor_value is not None:
          # The taper only limits braking while the car is still rolling.
          floor = butter_floor_value
          self._butter_rest_floor = butter_floor_value
        elif self._butter_rest_floor is not None and float(CS.vEgo) < STANDSTILL_V:
          # Below STANDSTILL_V the taper is ineligible; keep its last value until the hold is confirmed.
          floor = self._butter_rest_floor
        if floor is not None:
          # Rest ramp: near rest the floor eases down to the stop decel, so the ESC does not let go
          # on the taper's soft end; the one-way ramp above still limits how fast braking grows.
          if math.isfinite(self._butter_v_min):
            floor = min(floor, rest_ramp_floor(self._butter_v_min, self.stopping_accel))
          output_accel = max(output_accel, floor)
      if self.butter_stop and (self._butter_standstill_confirmed or self._butter_near_rest):
        # Creep guard: once the car was near rest, never let it creep toward the lead.
        if float(CS.vEgo) > CREEP_V and not bool(getattr(CS, "gasPressed", False)):
          self._butter_creep_latched = True
        if self._butter_creep_latched:
          output_accel = min(output_accel, CREEP_ACCEL)
          if not self._butter_creep_logged:
            cloudlog.event("butterStopCreepGuard", vEgo=float(CS.vEgo))
            self._butter_creep_logged = True
      self.reset()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.reset()

    else:  # LongCtrlState.pid
      if self._final_t is not None:
        # ButterStopFinal owns the command for the whole window: no coasting
        # relief, no butter floor and no depart guard inside it.
        self.coasting.reset()
        output_accel = self._final_window_accel(CS, long_plan, radarState)
        self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
        return self.last_output_accel, a_target_ff, j_target_now
      target = getattr(long_plan, 'cruiseCoastingTarget', 0.0)
      percent = getattr(long_plan, 'cruiseCoastingPercent', 0)
      relief = 0.0
      if (target > 0.0 and percent > 0 and self.CP.openpilotLongitudinalControl and
          0.0 <= t_since_plan <= MAX_PLAN_AGE and
          long_plan.longitudinalPlanSource == 'cruise' and not long_plan.fcw and not should_stop and
          not CS.brakePressed and not CS.gasPressed and not CS.carrotCruise and
          abs(long_plan.cruiseTarget - CS.vCruise) < 0.001 and
          not CS.cruiseState.standstill and no_coasting_lead(radarState) and
          math.isfinite(CS.aEgo) and math.isfinite(a_target_ff) and math.isfinite(v_target_now) and
          accel_limits[1] >= 0.0):
        relief = coasting_relief(CS.vEgo, target, percent)
      if relief == 0.0:
        self.coasting.reset()
      if self.use_accel_pid:
        error = a_target_ff - CS.aEgo
      else:
        error = v_target_now - CS.vEgo
      previous_integral = self.pid.i
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff)
      if relief > 0.0 and output_accel < 0.0:
        # Preserve the ordinary PID exactly for positive commands and at 0%.
        # Do not integrate a speed error whose braking output we are suppressing.
        self.pid.i = previous_integral
        output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff, freeze_integrator=True)
        output_accel = self.coasting.apply(output_accel, relief, DT_CTRL)
      else:
        self.coasting.reset()
      if butter_floor_value is not None and output_accel < butter_floor_value:
        # ButterStop taper: only remove braking above the floor, never add any.
        # Freeze the integrator while limited, like the coasting relief.
        self.pid.i = previous_integral
        output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff, freeze_integrator=True)
        output_accel = min(0.0, max(output_accel, butter_floor_value))
      # Off-identity: with the param off and no cap in flight the block is not
      # entered at all, so it cannot even read CS. The _firm_cap half keeps the
      # soft release alive when the param is switched off mid-stop.
      if self.firm_a > 0.0 or self._firm_cap is not None:
        hard_exit = bool(CS.gasPressed) or bool(CS.brakePressed) or bool(CS.cruiseState.standstill)
        # getattr, not attribute access: several callers hand update() a partial plan
        # (no .speeds) and a bare read raises on every tick before the gate is even
        # entered. An absent plan is "no plan", which the gate rejects.
        plan_speeds = getattr(long_plan, "speeds", ())
        gate = firm_gate(self.firm_a, float(CS.vEgo), float(a_target_ff), plan_speeds, t_since_plan,
                         bool(CS.gasPressed), bool(CS.brakePressed), bool(CS.cruiseState.standstill))
        was_capped = self._firm_cap is not None
        self._firm_cap = firm_cap_step(self._firm_cap, self.last_output_accel, self.firm_a, gate, hard_exit)
        if gate and not was_capped and not self._firm_logged:
          cloudlog.event("butterStopFirm", vEgo=float(CS.vEgo), aTarget=float(a_target_ff), firmA=self.firm_a)
          self._firm_logged = True
        if float(CS.vEgo) > FIRM_V_HI + 1.0:
          self._firm_logged = False
        if self._firm_cap is not None and output_accel > self._firm_cap:
          # Do not integrate the speed error the cap is overriding.
          self.pid.i = previous_integral
          output_accel = self._firm_cap
      guarded = self._depart_guard(output_accel, a_target_ff, should_stop, CS, radarState)
      if guarded > output_accel:
        # Do not integrate the speed error the guard is overriding.
        self.pid.i = previous_integral
        output_accel = guarded

    if self.long_control_state != LongCtrlState.pid:
      self._firm_cap = None
    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel, a_target_ff, j_target_now
