import math
import numpy as np
from openpilot.cereal import car
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
  CREEP_ACCEL, CREEP_V, FINAL_ROLL_RESET_V, FINAL_ROLL_V, FINAL_ROLL_VLEAD, HOLD_RAMP,
  LeadPosFilter, MIN_FINAL_GAP_ABS, STANDSTILL_T, STANDSTILL_V,
  STOP_DISTANCE_DEFAULT, butter_eligible, butter_floor, final_roll_limit,
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

STOPPING_ACCEL_DEFAULT = -50  # Params use hundredths of m/s^2
STOPPING_ACCEL_MIN = -100
STOPPING_ACCEL_MAX = -50

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill, a_ego, stopping_accel, radarState):
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
        if a_ego > stopping_accel or fcw_stop:
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
    self._refresh_stopping_accel()
    self._refresh_butter_stop()
    self.j_lead = 0.0

    # ButterStop stop-episode state. The standstill hold/release decision itself
    # stays in long_control_state_trans and is not touched here.
    self._butter_standstill_time = 0.0
    self._butter_standstill_confirmed = False
    self._butter_creep_latched = False
    self._butter_creep_logged = False
    self._butter_near_rest = False
    self._butter_rest_floor = None
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
    if not self.butter_stop:
      return
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

    prev_long_control_state = self.long_control_state
    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill, CS.aEgo, self.stopping_accel, radarState)
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
        if butter_floor_value is not None:
          # The taper only limits braking while the car is still rolling.
          output_accel = max(output_accel, butter_floor_value)
          self._butter_rest_floor = butter_floor_value
        elif self._butter_rest_floor is not None and float(CS.vEgo) < STANDSTILL_V:
          # Below STANDSTILL_V the taper is ineligible; keep its last value until the hold is confirmed.
          output_accel = max(output_accel, self._butter_rest_floor)
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
      guarded = self._depart_guard(output_accel, a_target_ff, should_stop, CS, radarState)
      if guarded > output_accel:
        # Do not integrate the speed error the guard is overriding.
        self.pid.i = previous_integral
        output_accel = guarded

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel, a_target_ff, j_target_now
