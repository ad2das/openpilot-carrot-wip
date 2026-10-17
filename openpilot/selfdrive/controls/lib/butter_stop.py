"""ButterStop: distance-exact taper floor and standstill creep guard.

When ButterStop is on the MPC plans its stop D_EXTRA earlier than the OFF
target; this module turns that slack into a soft end: below V_TAPER the floor
caps braking at -(a_end + K_TAPER*v), where a_end is solved so the taper stops
the car within the plan's remaining distance plus D_EXTRA. The car therefore
rolls through the whole slack and still ends at the OFF stop point, bounded by
the lead gap. The lead bound uses a smoothed lead position (LeadPosFilter) in
the ego odometry frame instead of raw dRel, so radar noise cannot flap the
bound, and LongControl may only relax the floor at FLOOR_RELAX_JERK (tightening
is instant). All standstill hold/release logic lives unchanged in the
LongControl stopping state machine; this module provides the floor and the
creep guard constants used there.
"""
import math

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.cruise_coasting import MAX_PLAN_AGE

V_TAPER = 3.0
K_TAPER = 0.5        # taper slope: a(v) = -(A + K_TAPER * v)
A_MIN = 0.10         # softest end decel the taper may use (m/s^2)
A_MAX = 1.0          # above this the taper is no softer than stock -> plan wins (None)
D_EXTRA = 0.4        # MPC stop-point shift (m) when ButterStop is on
MARGIN = 0.0         # distance kept unused at the end of the slack (m)
LEAD_SLACK = 1.2     # lead bound: final gap >= max(MIN_FINAL_GAP_ABS, stop_distance_orig - LEAD_SLACK)

# Chauffeur Stop (CS): lag-compensated taper. The taper is evaluated one plant
# lag ahead and the distance the commanded accel still spends is banked, so the
# ease reaches the wheels before the stop. The defaults are chosen to reproduce
# the landed ButterStop (B1) bit-identically: T_LAG=0 means no prediction, and
# the other slopes/slack match their B1 counterparts. The CP-H2 sweep patches
# these at run time.
V_TAPER_CS = 3.0     # eligibility/floor taper ceiling (replaces V_TAPER for CS)
T_LAG = 0.0          # plant lag predicted ahead (s); 0 == B1 (no prediction)
K_CS = 0.5           # CS taper slope (== K_TAPER at parity)
LEAD_SLACK_CS = 1.2  # CS lead-bound slack (replaces LEAD_SLACK in the CS floor)
MARGIN_CS = 0.0      # extra distance (m) held back from avail_cs (safety margin)
BUTTER_CS = False    # when True, LongControl uses butter_floor_cs; else butter_floor
SOLVE_ITERS = 30
LEAD_JUMP_M = 1.5      # raw-lead jump that forces a filter reset (m)
LEAD_RAW_ALLOW = 0.5   # d_eff may never exceed raw dRel by more than this (m)
TAU_LEAD = 0.3         # lead-position filter time constant (s)
FLOOR_RELAX_JERK = 0.8  # max floor relaxation rate (m/s^3); tightening is instant
MIN_FINAL_GAP_ABS = 3.0
STANDSTILL_V = 0.05
STANDSTILL_T = 0.2
HOLD_RAMP = 2.0
CREEP_V = 0.15
CREEP_ACCEL = -1.5
PLAN_STOP_V = 0.05
AVAIL_MIN = 0.05
LEAD_ACCEL_MIN = -1.5
LEAD_CLOSING_TIME_MIN = 2.5
STOP_DISTANCE_DEFAULT = 5.5


def plan_stop_distance(plan_t, plan_speeds):
  """Trapezoid distance along the plan until it first drops below standstill."""
  t = np.asarray(plan_t, dtype=float)
  v = np.asarray(plan_speeds, dtype=float)
  if t.ndim != 1 or v.ndim != 1 or v.shape != t.shape or t.size < 2:
    return None
  if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
    return None
  below = np.flatnonzero(v < PLAN_STOP_V)
  if below.size == 0:
    return None
  idx = int(below[0]) + 1  # include the first sub-standstill sample
  if idx < 2:
    return 0.0
  dt = np.diff(t[:idx])
  v_avg = 0.5 * (v[1:idx] + v[:idx - 1])
  return float(np.sum(v_avg * dt))


class LeadPosFilter:
  """Smoothed lead position in the ego-odometry frame.

  x_raw = dRel + s_ego is the lead position relative to where the ego was when
  the filter's episode began; for a stationary lead it is constant even while
  the ego rolls, so radar noise does not move it. A lead that jumps closer
  resets immediately (safe direction); a farther jump must persist for
  JUMP_VERIFY_S before it is adopted. d_eff = min(x_f - s_ego, dRel +
  LEAD_RAW_ALLOW) can therefore never exceed the raw gap by more than
  LEAD_RAW_ALLOW.
  """

  JUMP_VERIFY_S = 0.5

  def __init__(self):
    self._x_f = None
    self._far_jump_time = 0.0

  def reset(self):
    self._x_f = None
    self._far_jump_time = 0.0

  def update(self, status, d_rel, s_ego, dt=DT_CTRL, tau=None):
    if not status or d_rel is None:
      self.reset()
      return None
    tau = TAU_LEAD if tau is None else tau
    try:
      x_raw = float(d_rel) + float(s_ego)
      s = float(s_ego)
    except (TypeError, ValueError):
      self.reset()
      return None
    if not (math.isfinite(x_raw) and math.isfinite(s)):
      self.reset()
      return None
    if self._x_f is None:
      self._x_f = x_raw
      self._far_jump_time = 0.0
    elif x_raw < self._x_f - LEAD_JUMP_M:
      # Closer jump: adopt immediately, the safe direction.
      self._x_f = x_raw
      self._far_jump_time = 0.0
    elif x_raw > self._x_f + LEAD_JUMP_M:
      self._far_jump_time += dt
      if self._far_jump_time + 1e-9 >= self.JUMP_VERIFY_S:
        self._x_f = x_raw
        self._far_jump_time = 0.0
    else:
      self._far_jump_time = 0.0
      self._x_f += (dt / tau) * (x_raw - self._x_f)
    return min(self._x_f - s, float(d_rel) + LEAD_RAW_ALLOW)


def taper_stop_distance(v, a_end, k):
  """Distance to stop from v under a(v) = -(a_end + k*v); a_end > 0, k >= 0."""
  if k <= 0.0:
    return v * v / (2.0 * a_end)
  return v / k - (a_end / (k * k)) * math.log1p(k * v / a_end)


def solve_taper_a_end(v, avail, k):
  """Smallest a_end in [A_MIN, A_MAX] whose taper stops within avail, or None.

  taper_stop_distance decreases monotonically in a_end. Returns A_MIN when
  even the softest taper fits; None when even A_MAX overruns avail. Bisection
  returns the upper bracket so the result always satisfies S <= avail.
  """
  if taper_stop_distance(v, A_MIN, k) <= avail:
    return A_MIN
  if taper_stop_distance(v, A_MAX, k) > avail:
    return None
  lo, hi = A_MIN, A_MAX
  for _ in range(SOLVE_ITERS):
    mid = 0.5 * (lo + hi)
    if taper_stop_distance(v, mid, k) <= avail:
      hi = mid
    else:
      lo = mid
  return hi


def slew_floor(prev, floor_raw, dt=DT_CTRL):
  """One-way floor slew: tightening is instant, relaxing is rate-limited.

  Returns None (and lets the caller reset prev) when floor_raw is None; the
  first raw value is adopted unchanged. FLOOR_RELAX_JERK is read at call time
  so the tuning sweep can monkeypatch the module attribute.
  """
  if floor_raw is None:
    return None
  if prev is None:
    return floor_raw
  return min(floor_raw, prev + FLOOR_RELAX_JERK * dt)


def butter_floor(v_ego, plan_t, plan_speeds, lead, stop_distance_orig, d_rel_eff=None, diag=None):
  """Most-negative allowed acceleration at v_ego, or None to let the plan win.

  The taper distance budget is the plan's own remaining stop distance plus
  D_EXTRA (the slack the shifted MPC left in front of it), so rolling through
  the slack still ends at the OFF stop point; the lead bound keeps the whole
  slack outside the minimum final gap. d_rel_eff replaces the raw lead dRel
  when given (the smoothed lead position LongControl maintains); diag, when
  given, records the binding term and the solved a_end for the sim telemetry.
  """
  if diag is not None:
    diag["bound"] = None
    diag["a_end"] = None
  if not math.isfinite(v_ego) or v_ego <= 0.0:
    return None
  if stop_distance_orig is None or not math.isfinite(stop_distance_orig) or stop_distance_orig < 0.0:
    return None
  dist = plan_stop_distance(plan_t, plan_speeds)
  if dist is None:
    return None
  avail = dist + D_EXTRA - MARGIN
  bound = "plan"
  if lead is not None and getattr(lead, "status", False):
    if d_rel_eff is not None:
      try:
        d_rel = float(d_rel_eff)
      except (TypeError, ValueError):
        return None
    else:
      try:
        d_rel = float(lead.dRel)
      except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(d_rel):
      return None
    lead_avail = d_rel - max(MIN_FINAL_GAP_ABS, stop_distance_orig - LEAD_SLACK)
    if lead_avail < avail:
      bound = "lead"
    avail = min(avail, lead_avail)
  if avail <= AVAIL_MIN:
    return None
  a_end = solve_taper_a_end(v_ego, avail, K_TAPER)
  if a_end is None:
    return None
  if diag is not None:
    diag["bound"] = bound
    diag["a_end"] = a_end
  return -(a_end + K_TAPER * v_ego)


def _cta_avail(plan_t, plan_speeds, lead, stop_distance_orig, d_rel_eff, lead_slack):
  """Shared CS/B1 taper budget: min(plan + D_EXTRA - MARGIN, lead bound).

  Returns (avail, bound) or (None, None) when there is no planned stop / a
  non-finite lead gap. The lead bound is d_rel - max(MIN_FINAL_GAP_ABS,
  stop_distance_orig - lead_slack); lead_slack is LEAD_SLACK for B1 and
  LEAD_SLACK_CS for CS.
  """
  if stop_distance_orig is None or not math.isfinite(stop_distance_orig) or stop_distance_orig < 0.0:
    return None, None
  dist = plan_stop_distance(plan_t, plan_speeds)
  if dist is None:
    return None, None
  avail = dist + D_EXTRA - MARGIN
  bound = "plan"
  if lead is not None and getattr(lead, "status", False):
    if d_rel_eff is not None:
      try:
        d_rel = float(d_rel_eff)
      except (TypeError, ValueError):
        return None, None
    else:
      try:
        d_rel = float(lead.dRel)
      except (TypeError, ValueError, AttributeError):
        return None, None
    if not math.isfinite(d_rel):
      return None, None
    lead_avail = d_rel - max(MIN_FINAL_GAP_ABS, stop_distance_orig - lead_slack)
    if lead_avail < avail:
      bound = "lead"
    avail = min(avail, lead_avail)
  return avail, bound


def butter_floor_cs(v_ego, a_ego, plan_t, plan_speeds, lead, stop_distance_orig,
                    d_rel_eff=None, diag=None):
  """Lag-compensated taper floor (Chauffeur Stop), or None to let the plan win.

  The eater evaluates the taper at the PREDICTED speed one plant lag
  (T_LAG) ahead and subtracts the distance the current accel still spends in
  that lag, so the commanded ease reaches the wheels before the car stops. With
  T_LAG == 0 and V_TAPER_CS/K_CS/LEAD_SLACK_CS equal to their B1 counterparts
  this reproduces butter_floor exactly (bit-identical parity).

  a_ego is the currently applied accel; only the braking part (min(a_ego, 0)) is
  used for the prediction so the floor never assumes the car accelerates.
  """
  if diag is not None:
    diag["bound"] = None
    diag["a_end"] = None
    diag["v_p"] = None
    diag["d_lag"] = None
    diag["avail_cs"] = None
  if not math.isfinite(v_ego) or v_ego <= 0.0:
    return None
  avail, bound = _cta_avail(plan_t, plan_speeds, lead, stop_distance_orig, d_rel_eff,
                            LEAD_SLACK_CS)
  if avail is None:
    return None
  # Lag prediction: where will v be one lag ahead, and how far does the car
  # still travel while a = min(a_ego, 0) acts over T_LAG?
  try:
    a = min(float(a_ego), 0.0) if a_ego is not None and math.isfinite(float(a_ego)) else 0.0
  except (TypeError, ValueError):
    a = 0.0
  tau = T_LAG
  if v_ego + a * tau > 0.0:
    d_lag = v_ego * tau + 0.5 * a * tau * tau
    v_p = v_ego + a * tau
  else:
    # The car reaches standstill inside the lag; spend v^2/(2|a|) then hold.
    d_lag = v_ego * tau if a == 0.0 else v_ego * v_ego / (2.0 * abs(a))
    v_p = 0.0
  avail_cs = avail - d_lag - MARGIN_CS
  if diag is not None:
    diag["v_p"] = v_p
    diag["d_lag"] = d_lag
    diag["avail_cs"] = avail_cs
  if avail_cs <= AVAIL_MIN:
    return None
  if v_p <= STANDSTILL_V:
    floor = -A_MIN
  else:
    a_end = solve_taper_a_end(v_p, avail_cs, K_CS)
    if a_end is None:
      return None
    floor = -(a_end + K_CS * v_p)
  if diag is not None:
    diag["bound"] = bound
    diag["a_end"] = a_end if v_p > STANDSTILL_V else A_MIN
  return floor


def butter_eligible_cs(active, v_ego, brake_pressed, gas_pressed, fcw, t_since_plan, lead):
  """CS eligibility: identical to butter_eligible except the taper ceiling."""
  return butter_eligible_impl(active, v_ego, brake_pressed, gas_pressed, fcw,
                              t_since_plan, lead, V_TAPER_CS)


def butter_eligible(active, v_ego, brake_pressed, gas_pressed, fcw, t_since_plan, lead):
  """All eligibility gates except plan-array finiteness (checked by butter_floor)."""
  return butter_eligible_impl(active, v_ego, brake_pressed, gas_pressed, fcw,
                              t_since_plan, lead, V_TAPER)


def butter_eligible_impl(active, v_ego, brake_pressed, gas_pressed, fcw, t_since_plan,
                         lead, v_taper):
  if not active:
    return False
  if not (math.isfinite(v_ego) and math.isfinite(t_since_plan)):
    return False
  if not (STANDSTILL_V < v_ego < v_taper):
    return False
  if brake_pressed or gas_pressed or fcw:
    return False
  if not 0.0 <= t_since_plan <= MAX_PLAN_AGE:
    return False
  if lead is not None and getattr(lead, "status", False):
    try:
      d_rel = float(lead.dRel)
      a_lead = float(lead.aLeadK)
      v_lead = float(lead.vLead)
    except (TypeError, ValueError, AttributeError):
      return False
    if not all(map(math.isfinite, (d_rel, a_lead, v_lead))):
      return False
    if d_rel < MIN_FINAL_GAP_ABS:
      return False
    if a_lead < LEAD_ACCEL_MIN:
      return False
    if v_lead < v_ego and d_rel / (v_ego - v_lead) < LEAD_CLOSING_TIME_MIN:
      return False
  return True
