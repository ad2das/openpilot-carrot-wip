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
PLAN_STOP_V = 0.05
AVAIL_MIN = 0.05
LEAD_ACCEL_MIN = -1.5
LEAD_CLOSING_TIME_MIN = 2.5
STOP_DISTANCE_DEFAULT = 5.5

# ButterStop early-ease (BSE): adds braking EARLIER (0.3..V_E m/s) so the car
# arrives slow with distance left and the lagging actuator decays before the
# stop, removing the end-of-stop nod. These are the sweepable defaults; the
# tuning sweep overrides the module attributes (read at call time).
K_E = 0.8            # BSE taper slope: ceiling(v) = -(a_end + K_E * v)
L_E = 0.3            # dead-time budget (s): v_ego * L_E metres reserved
MARGIN_E = 0.0       # extra distance kept unused before the ease (m)
EXTRA_MAX = 0.6      # most BSE may brake harder than stock (m/s^2)
V_E = 4.0            # BSE upper speed bound (m/s)
BSE_V_MIN = 0.3      # BSE lower speed bound (m/s)
BSE_LEAD_V = 0.3     # lead counts as stationary below this (m/s)
BSE_LEAD_A_MIN = -0.5  # lead accel floor for "stationary" (m/s^2)
BSE_INELIG_END_S = 0.5  # ineligible this long ends the episode


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


def solve_taper_a_end_range(v, avail, k, a_min=None, a_max=None):
  """Parameterised solve_taper_a_end; does not read/mutate the module constants.

  Same monotonic bisection as solve_taper_a_end but with caller-provided bounds
  so BSE can use its own K_E without changing the existing taper behaviour.
  """
  a_min = A_MIN if a_min is None else a_min
  a_max = A_MAX if a_max is None else a_max
  if taper_stop_distance(v, a_min, k) <= avail:
    return a_min
  if taper_stop_distance(v, a_max, k) > avail:
    return None
  lo, hi = a_min, a_max
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


def bse_eligible(active, v_ego, brake_pressed, gas_pressed, fcw, t_since_plan,
                 long_state_pid, lead):
  """BSE eligibility (brief 2.1).

  Same gates as butter_eligible except the speed window uses BSE's own
  0.3 < v_ego <= V_E bound (V_E may exceed V_TAPER), and the long state must be
  pid. The lead must be valid and stationary: vLead < BSE_LEAD_V and
  aLeadK > BSE_LEAD_A_MIN. K_E/L_E/... are read at call time so the sweep can
  patch them.
  """
  if not active or not long_state_pid:
    return False
  if not (math.isfinite(v_ego) and math.isfinite(t_since_plan)):
    return False
  if not (BSE_V_MIN < v_ego <= V_E):
    return False
  if brake_pressed or gas_pressed or fcw:
    return False
  if not 0.0 <= t_since_plan <= MAX_PLAN_AGE:
    return False
  if lead is None or not getattr(lead, "status", False):
    return False
  try:
    d_rel = float(lead.dRel)
    a_lead = float(lead.aLeadK)
    v_lead = float(lead.vLead)
  except (TypeError, ValueError, AttributeError):
    return False
  if not all(map(math.isfinite, (d_rel, a_lead, v_lead))):
    return False
  if v_lead >= BSE_LEAD_V or a_lead <= BSE_LEAD_A_MIN:
    return False
  return True


def butter_ceiling(v_ego, lead, stop_distance_orig, d_rel_eff=None, diag=None):
  """Most-negative allowed acceleration BSE may add, or None.

  Room: avail_e = d_rel_eff - max(MIN_FINAL_GAP_ABS, stop_distance_orig -
  LEAD_SLACK) - MARGIN_E - v_ego*L_E. The v_ego*L_E term budgets the actuator
  dead time so the ease completes before the stop. Taper:
  a_end = solve_taper_a_end_range(v_ego, avail_e, K_E); when avail_e <= AVAIL_MIN
  or a_end is None, returns None (stock wins). Otherwise
  ceiling = -(a_end + K_E*v_ego). diag records avail_e/a_end for telemetry.
  """
  if diag is not None:
    diag["ceiling"] = None
    diag["avail_e"] = None
    diag["a_end"] = None
  if not (math.isfinite(v_ego) and BSE_V_MIN < v_ego <= V_E):
    return None
  if lead is None or not getattr(lead, "status", False):
    return None
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
  if stop_distance_orig is None or not math.isfinite(stop_distance_orig) or stop_distance_orig < 0.0:
    return None
  room = d_rel - max(MIN_FINAL_GAP_ABS, stop_distance_orig - LEAD_SLACK) - MARGIN_E - v_ego * L_E
  if diag is not None:
    diag["avail_e"] = room
  if room <= AVAIL_MIN:
    return None
  a_end = solve_taper_a_end_range(v_ego, room, K_E)
  if a_end is None:
    return None
  if diag is not None:
    diag["a_end"] = a_end
  ceiling = -(a_end + K_E * v_ego)
  if diag is not None:
    diag["ceiling"] = ceiling
  return ceiling


def ceiling_extra(stock_out, ceiling):
  """Target extra braking (stock - out) the ceiling clamp asks for (brief 2.4).

  out_target = max(min(stock, ceiling), stock - EXTRA_MAX); returns
  stock - out_target >= 0. 0 when the ceiling does not bind. The caller slews
  this value with slew_extra so leaving BSE never steps the braking down.
  """
  if ceiling is None:
    return 0.0
  clamped = max(min(stock_out, ceiling), stock_out - EXTRA_MAX)
  return max(0.0, stock_out - clamped)


def slew_extra(prev, target, dt=DT_CTRL):
  """Symmetric jerk-limited slew for the BSE extra braking (invariant I4 + G6).

  The extra is rate-limited in BOTH directions at FLOOR_RELAX_JERK so that
  neither entering nor leaving BSE steps the braking (G6 caps cmd |jerk| at
  2.5 m/s^3). Growing toward more braking and decaying toward 0 both obey the
  same limit; I4 ("leaving never steps down") is the decaying half. Returns the
  applied extra. FLOOR_RELAX_JERK is read at call time so the sweep can patch it.
  """
  if prev is None:
    return target
  lo = prev - FLOOR_RELAX_JERK * dt
  hi = prev + FLOOR_RELAX_JERK * dt
  return min(max(target, lo), hi)


class BseState:
  """BSE episode latch (brief 2.5).

  Starts when the ceiling first binds (eligible and a ceiling exists). Ends on a
  hard reason (lead not stationary, gas/brake/FCW, v_ego > V_E + 0.5, or the
  stopping state is entered) immediately, or after BSE_INELIG_END_S of any other
  ineligibility. On end the extra decays to 0 at FLOOR_RELAX_JERK (never steps).
  Pure state + arithmetic; the caller supplies stock_out, status and timing.
  """

  def __init__(self):
    self.active = False
    self.inelig_time = 0.0
    self.extra_prev = None
    self.ceiling_last = None
    self.avail_e_last = None
    self.a_end_last = None

  def reset(self):
    self.active = False
    self.inelig_time = 0.0
    self.extra_prev = None
    self.ceiling_last = None
    self.avail_e_last = None
    self.a_end_last = None

  def update(self, stock_out, eligible, hard_end, ceiling, dt=DT_CTRL):
    """Advance one step; returns the BSE output (<= stock_out).

    `extra_prev` is a float for the whole BSE episode lifetime (seeded at 0 on
    first activation) so the jerk limit applies to entering, changing and leaving
    equally; it is only cleared by reset().
    """
    if not self.active:
      if eligible and ceiling is not None:
        self.active = True
        self.inelig_time = 0.0
        if self.extra_prev is None:
          self.extra_prev = 0.0
    else:
      if hard_end:
        self.active = False
      elif not eligible:
        self.inelig_time += dt
        if self.inelig_time >= BSE_INELIG_END_S:
          self.active = False
      else:
        self.inelig_time = 0.0

    target = ceiling_extra(stock_out, ceiling) if self.active else 0.0
    applied = slew_extra(self.extra_prev, target, dt)
    self.extra_prev = applied
    out = min(stock_out, stock_out - applied)
    self.ceiling_last = ceiling if self.active else None
    return out

