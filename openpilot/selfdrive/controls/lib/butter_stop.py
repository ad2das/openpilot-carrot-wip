"""ButterStop: distance-exact taper floor and standstill creep guard.

When ButterStop is on the MPC plans its stop D_EXTRA earlier than the OFF
target; this module turns that slack into a soft end: below V_TAPER the floor
caps braking at -(a_end + K_TAPER*v), where a_end is solved so the taper stops
the car within the plan's remaining distance plus D_EXTRA. The car therefore
rolls through the whole slack and still ends at the OFF stop point, bounded by
the lead gap. All standstill hold/release logic lives unchanged in the
LongControl stopping state machine; this module provides the floor and the
creep guard constants used there.
"""
import math

import numpy as np

from openpilot.selfdrive.controls.lib.cruise_coasting import MAX_PLAN_AGE

V_TAPER = 3.0
K_TAPER = 0.4        # taper slope: a(v) = -(A + K_TAPER * v)
A_MIN = 0.10         # softest end decel the taper may use (m/s^2)
A_MAX = 1.0          # above this the taper is no softer than stock -> plan wins (None)
D_EXTRA = 0.9        # MPC stop-point shift (m) when ButterStop is on
MARGIN = 0.0         # distance kept unused at the end of the slack (m)
LEAD_SLACK = 1.2     # lead bound: final gap >= max(MIN_FINAL_GAP_ABS, stop_distance_orig - LEAD_SLACK)
SOLVE_ITERS = 30
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


def butter_floor(v_ego, plan_t, plan_speeds, lead, stop_distance_orig):
  """Most-negative allowed acceleration at v_ego, or None to let the plan win.

  The taper distance budget is the plan's own remaining stop distance plus
  D_EXTRA (the slack the shifted MPC left in front of it), so rolling through
  the slack still ends at the OFF stop point; the lead bound keeps the whole
  slack outside the minimum final gap.
  """
  if not math.isfinite(v_ego) or v_ego <= 0.0:
    return None
  if stop_distance_orig is None or not math.isfinite(stop_distance_orig) or stop_distance_orig < 0.0:
    return None
  dist = plan_stop_distance(plan_t, plan_speeds)
  if dist is None:
    return None
  avail = dist + D_EXTRA - MARGIN
  if lead is not None and getattr(lead, "status", False):
    try:
      d_rel = float(lead.dRel)
    except (TypeError, ValueError, AttributeError):
      return None
    if not math.isfinite(d_rel):
      return None
    avail = min(avail, d_rel - max(MIN_FINAL_GAP_ABS, stop_distance_orig - LEAD_SLACK))
  if avail <= AVAIL_MIN:
    return None
  a_end = solve_taper_a_end(v_ego, avail, K_TAPER)
  if a_end is None:
    return None
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
