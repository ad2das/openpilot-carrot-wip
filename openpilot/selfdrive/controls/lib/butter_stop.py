"""ButterStop: speed-tapered final stop floor and standstill creep guard.

The taper only ever limits how hard braking may build up below V_TAPER. All
standstill hold/release logic lives unchanged in the LongControl stopping
state machine; this module provides the bounded floor and the creep guard
constants used there.
"""
import math

import numpy as np

from openpilot.selfdrive.controls.lib.cruise_coasting import MAX_PLAN_AGE

V_TAPER = 3.0
A_END = 0.25
K_TAPER = 0.4
D_EXTRA = 1.0
MARGIN_BASE = 0.2
MARGIN_V_SLOPE = 0.15
MIN_FINAL_GAP_SLACK = 0.6
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


def butter_floor(v_ego, plan_t, plan_speeds, lead, stop_distance_orig):
  """Most-negative allowed acceleration at v_ego, or None to let the plan win."""
  if not math.isfinite(v_ego) or v_ego <= 0.0:
    return None
  if stop_distance_orig is None or not math.isfinite(stop_distance_orig) or stop_distance_orig < 0.0:
    return None
  dist = plan_stop_distance(plan_t, plan_speeds)
  if dist is None:
    return None
  avail = dist + D_EXTRA - (MARGIN_BASE + MARGIN_V_SLOPE * v_ego)
  if lead is not None and getattr(lead, "status", False):
    try:
      d_rel = float(lead.dRel)
    except (TypeError, ValueError, AttributeError):
      return None
    if not math.isfinite(d_rel):
      return None
    avail = min(avail, d_rel - max(MIN_FINAL_GAP_ABS, stop_distance_orig - MIN_FINAL_GAP_SLACK))
  if avail <= AVAIL_MIN:
    return None
  a_kin = -(v_ego * v_ego) / (2.0 * avail)
  return min(-(A_END + K_TAPER * v_ego), a_kin)


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
