"""Smooth far-lead approach relief: the comfort cost plans at the decel a stop needs."""

import math

import numpy as np

FAR_LEAD_APPROACH = True     # False = V1 behaviour (tests only)
APPROACH_MARGIN = 1.25       # planned decel = APPROACH_MARGIN * a_req   (swept)
APPROACH_A_MAX = 1.6         # m/s^2, a_req above this -> no relief     (swept)
APPROACH_A_MIN = 0.5         # planned decel never gentler than this
APPROACH_V_MIN = 15.0        # m/s ego speed (54 km/h) - highway feature only
APPROACH_TTC_MIN = 4.0       # s
APPROACH_LEAD_A_MIN = -0.5   # m/s^2, lead decelerating harder -> no relief
APPROACH_RAMP_S = 1.0        # ramp-in time of the relief scale
APPROACH_HOLD_S = 1.0        # radar dropout hold
APPROACH_RECOVER_S = 2.0     # plan must clear the static danger boundary by this time


def get_safe_obstacle_distance(v_ego, t_follow, comfort_brake, stop_distance):
  """Same one-line formula as long_mpc.get_safe_obstacle_distance (kept local
  so this pure module has no import cycle with the acados solver library)."""
  return (v_ego ** 2) / (2 * comfort_brake) + t_follow * v_ego + stop_distance


def approach_path_safe(times, danger_margins, physical_gaps, ego_speeds, stop_distance):
  late = times >= APPROACH_RECOVER_S
  early = ~late
  return bool(np.all(np.isfinite(danger_margins)) and np.all(np.isfinite(physical_gaps))
              and np.all(np.isfinite(ego_speeds))
              and np.min(danger_margins[late]) >= 3.0
              and np.min(danger_margins[early]) >= min(float(danger_margins[0]), 3.0) - 0.5
              and np.min(physical_gaps - stop_distance - 0.7 * ego_speeds) >= 2.0)


class ApproachState:
  def __init__(self):
    self.scale = 0.0
    self.hold = 0.0

  def relief(self, *, dt, enabled, lead_present, v_ego, gap, v_lead, a_lead, t_follow,
             t_follow_danger, danger_factor, comfort_brake, stop_distance, times):
    if not FAR_LEAD_APPROACH or not enabled:
      self.scale = 0.0
      self.hold = 0.0
      return np.zeros_like(times)

    if not lead_present:
      if self.hold > 0.0 and self.scale > 0.0:
        self.hold -= dt
        self.scale = max(0.0, self.scale - dt / APPROACH_RAMP_S)
      else:
        self.scale = 0.0
      return np.zeros_like(times)

    values = (dt, v_ego, gap, v_lead, a_lead, t_follow, t_follow_danger, danger_factor,
              comfort_brake, stop_distance)
    if (not all(math.isfinite(v) for v in values) or dt <= 0.0
        or v_ego < APPROACH_V_MIN or v_lead < 5.0):
      self.scale = 0.0
      self.hold = 0.0
      return np.zeros_like(times)

    c = v_ego - v_lead
    if not (0.5 <= c <= 15.0) or a_lead < APPROACH_LEAD_A_MIN:
      self.scale = 0.0
      self.hold = 0.0
      return np.zeros_like(times)

    if gap / c < APPROACH_TTC_MIN:
      self.scale = 0.0
      self.hold = 0.0
      return np.zeros_like(times)

    d = gap - (t_follow * v_lead + stop_distance)
    if d < 5.0:
      self.scale = 0.0
      self.hold = 0.0
      return np.zeros_like(times)

    a_req = c * c / (2.0 * d)
    if a_req > APPROACH_A_MAX:
      self.scale = 0.0
      self.hold = 0.0
      return np.zeros_like(times)

    a_s = float(np.clip(APPROACH_MARGIN * a_req, APPROACH_A_MIN, APPROACH_A_MAX))
    v_p = np.maximum(v_lead, v_ego - a_s * times)
    x_p = v_ego * times - 0.5 * a_s * times ** 2
    stop_time = (v_ego - v_lead) / a_s if v_ego > v_lead else 0.0
    late = times > stop_time
    x_p[late] = v_ego * stop_time - 0.5 * a_s * stop_time ** 2 + v_lead * (times[late] - stop_time)
    lead_x = gap + v_lead * times
    lead_obstacle = lead_x + (v_lead ** 2) / (2.0 * comfort_brake)

    danger_margins = lead_obstacle - x_p - danger_factor * get_safe_obstacle_distance(
      v_p, t_follow_danger, comfort_brake, stop_distance)
    physical_gaps = lead_x - x_p
    if not approach_path_safe(times, danger_margins, physical_gaps, v_p, stop_distance):
      self.scale = 0.0
      self.hold = 0.0
      return np.zeros_like(times)

    shortfall = get_safe_obstacle_distance(v_p, t_follow, comfort_brake, stop_distance) - (lead_obstacle - x_p)
    relief = np.maximum(0.0, shortfall)

    self.scale = min(1.0, self.scale + dt / APPROACH_RAMP_S)
    self.hold = APPROACH_HOLD_S
    return self.scale * relief
