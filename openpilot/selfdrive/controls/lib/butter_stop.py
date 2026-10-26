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
# Rest ramp (K5 routes 6da..6e3, 10-26): ending the stop on the taper's last value (~-0.13) let the
# ESC let go at 0.04-0.05 m/s, and idle creep surged at +0.11..+0.25 m/s^2 in 5 of 10 stops; the
# -0.5 request at that speed before (routes 6d5..6d8) held in 11 of 16. Below REST_RAMP_V the floor
# eases down with the lowest speed of the stop and reaches the stop decel at STANDSTILL_V.
REST_RAMP_V = 0.3
PLAN_STOP_V = 0.05
AVAIL_MIN = 0.05
LEAD_ACCEL_MIN = -1.5
LEAD_CLOSING_TIME_MIN = 2.5
STOP_DISTANCE_DEFAULT = 5.5

# Latched final-roll limit. Once per stop, behind a lead that is no longer
# closing, the ego-odometry position where the car must be at rest is latched
# to at most stock rolling distance + FINAL_ROLL_M from where it is at that
# moment. The taper may then use that slack, so the last metres keep rolling
# instead of being given back to the plan at the stop point.
FINAL_ROLL_M = 0.15      # extra rolling distance granted at the latch (m)
FINAL_ROLL_V = 0.6       # latch when v_ego <= this (m/s)
FINAL_ROLL_RESET_V = 1.0 # a latch above this speed is stale, drop it (m/s)
FINAL_ROLL_VLEAD = 0.3   # lead must be slower than this to latch (m/s)
STOCK_ROLL_DECEL = 0.5   # stock stopping decel used for the stock rolling distance


def final_roll_limit(v_ego, d_eff, s_ego):
  """Latched ego-odometry position where the taper must have stopped the car.

  stock is the distance stock stopping would still cover at v_ego; the extra
  FINAL_ROLL_M on top of it is the only slack the latch may add, and the lead
  budget still caps the result MIN_FINAL_GAP_ABS short of the lead.
  """
  stock = v_ego * v_ego / (2.0 * STOCK_ROLL_DECEL)
  return s_ego + min(stock + FINAL_ROLL_M, d_eff - MIN_FINAL_GAP_ABS)


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


def rest_ramp_floor(v_min, stopping_accel):
  """Braking floor near rest: 0 at REST_RAMP_V, stopping_accel at STANDSTILL_V and below."""
  return float(np.interp(v_min, [STANDSTILL_V, REST_RAMP_V], [min(stopping_accel, 0.0), 0.0]))


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


def butter_floor(v_ego, plan_t, plan_speeds, lead, stop_distance_orig, d_rel_eff=None, diag=None,
                 roll_avail=None):
  """Most-negative allowed acceleration at v_ego, or None to let the plan win.

  The taper distance budget is the plan's own remaining stop distance plus
  D_EXTRA (the slack the shifted MPC left in front of it), so rolling through
  the slack still ends at the OFF stop point; the lead bound keeps the whole
  slack outside the minimum final gap. d_rel_eff replaces the raw lead dRel
  when given (the smoothed lead position LongControl maintains); diag, when
  given, records the binding term and the solved a_end for the sim telemetry.
  roll_avail is the remaining distance to the latched final-roll limit; it only
  ever raises the lead budget, and the caller has already clipped it to the
  lead, so it can never push the car past MIN_FINAL_GAP_ABS.
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
    if roll_avail is not None and math.isfinite(roll_avail):
      lead_avail = max(lead_avail, roll_avail)
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


def butter_eligible(active, v_ego, brake_pressed, gas_pressed, fcw, t_since_plan, lead):
  """All eligibility gates except plan-array finiteness (checked by butter_floor)."""
  if not active:
    return False
  if not (math.isfinite(v_ego) and math.isfinite(t_since_plan)):
    return False
  if not (STANDSTILL_V < v_ego < V_TAPER):
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
