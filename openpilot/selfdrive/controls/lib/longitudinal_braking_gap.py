"""Distance memory and bounded comfort slack; physical MPC obstacles stay intact."""

from copy import copy
import math

import numpy as np


STOPPED_LEAD_RELEASE_S = 1.0  # release tau for a stopped lead while the ego still rolls
# Speed-proportional cap on the remembered floor (s). The captured bound is
# re-applied every step, so the memory shrinks with the speed instead of
# lagging behind it on the 8 s decay. FLOOR_CAP_OFFSET_V shifts the cap so it
# reaches the base distance early: with the chosen pair (2.0, 3.0) the
# remembered gap is fully released below 3 m/s (about 11 km/h), before the
# final metres. Chosen by the offset sweep (task 9 review; row K2.0/V3).
FLOOR_SPEED_CAP_S = 2.0
FLOOR_CAP_OFFSET_V = 3.0


def comfort_path_has_reserve(ego_speeds, physical_gaps, danger_margins, times, stop_distance):
  near = times <= 3.0
  return bool(np.all(np.isfinite(danger_margins)) and np.all(np.isfinite(physical_gaps))
              and np.all(np.isfinite(ego_speeds))
              and np.min(danger_margins[near]) > 3.0
              and np.min(physical_gaps[near] - stop_distance - 0.7 * ego_speeds[near]) > 2.0)


class BrakingGapState:
  def __init__(self):
    self.key = None
    self.age = 0.0
    self.distance = 0.0
    self.relative_speed = 0.0
    self.floor = None
    self.margin = 0.0
    self.relief = 0.0
    self.stopped_pair = False
    self.fast_release = False
    self.speed_cap = None

  def update(self, *, enabled, track_id, dt, distance, ego_speed, ego_accel, lead_speed,
             lead_accel, relative_speed, base_tf, stop_distance, desired_distance):
    values = (dt, distance, ego_speed, ego_accel, lead_speed, lead_accel, relative_speed,
              base_tf, stop_distance, desired_distance)
    if (not enabled or track_id < 0 or not all(map(math.isfinite, values))
        or not 0.0 < dt <= 0.2 or distance <= 0.0 or ego_speed < 0.0):
      self.__init__()
      return 0.0
    if (self.key != track_id
        or abs(distance - self.distance - self.relative_speed * dt) > 3.0):
      self.__init__()
      self.key = track_id
    self.age += dt
    base_distance = stop_distance + ego_speed * base_tf
    if (self.floor is None and self.age >= 0.3 and ego_speed > 1.0
        and ego_accel < -0.2 and lead_accel < -0.2 and relative_speed < -0.2):
      # Capture once per braking episode, not once per sample. Bound distant
      # acquisition and do not carry one vehicle's gap into a replacement lead.
      self.floor = min(distance, base_distance + 10.0, stop_distance + 2.5 * ego_speed)
    self.distance, self.relative_speed = distance, relative_speed
    self._advance(dt, base_distance, desired_distance, ego_speed, lead_speed, stop_distance,
                  releasing=(ego_accel > 0.1 or lead_accel > 0.3))
    return self.margin

  def _advance(self, dt, base_distance, desired_distance, ego_speed, lead_speed, stop_distance, releasing=False):
    # A stopped pair must not slowly creep forward just because its reference
    # expires. Departure evidence releases this episode's distance preference.
    stopped_pair = ego_speed < 0.3 and lead_speed < 0.3
    # A stopped lead ends the following episode. The release tau additionally
    # applies once the ego itself has stopped before the lead. This memory only
    # ever ADDS comfort distance in the cost reference; releasing it returns to
    # the base stop distance, and the physical constraints / danger margins are
    # untouched.
    fast_release = lead_speed < 0.3
    self.stopped_pair = stopped_pair
    self.fast_release = fast_release
    self.speed_cap = None if not math.isfinite(FLOOR_SPEED_CAP_S) else \
      max(base_distance, stop_distance + FLOOR_SPEED_CAP_S * max(0.0, ego_speed - FLOOR_CAP_OFFSET_V))
    if self.floor is None:
      return
    if not stopped_pair or releasing:
      target = min(self.floor, base_distance)
      tau = 2.0 if releasing else (STOPPED_LEAD_RELEASE_S if fast_release else 8.0)
      self.floor += (target - self.floor) * -math.expm1(-dt / tau)
      # The capture bound is re-applied every step and reaches the base
      # distance at FLOOR_CAP_OFFSET_V, so the remembered gap is fully returned
      # before walking speed instead of during the final metres. Never below
      # the base distance; the stopped pair still freezes.
      if self.speed_cap is not None:
        self.floor = min(self.floor, self.speed_cap)
    requested = max(0.0, self.floor - desired_distance)
    self.margin = min(requested, self.margin + dt) if requested >= self.margin else max(requested, self.margin - dt)
    if releasing and self.margin < 0.01 and self.floor <= base_distance + 0.05:
      self.floor = None

  def margins(self, times, ego_speeds, lead_speeds, desired_distances, base_tf, stop_distance):
    predicted = copy(self)
    result = np.zeros_like(times)
    previous = 0.0
    for i, time_s in enumerate(times):
      if time_s > previous:
        predicted._advance(float(time_s - previous), stop_distance + ego_speeds[i] * base_tf,
                           desired_distances[i], ego_speeds[i], lead_speeds[i], stop_distance)
      result[i] = predicted.margin
      previous = time_s
    return result

  def comfort_relief(self, *, dt, enabled, base_tf, ego_speeds, lead_speeds, lead_accel,
                     physical_gaps, danger_margins, times, stop_distance):
    """Spend at most 0.4 s of comfort TF, never modify the danger constraint.

    0.7 s is a lower comfort-reference bound, NOT a safety guarantee. Closing
    speed, observed clearance and the unchanged lead forecast must also leave
    reserve. New leads, stops and already constrained approaches get no relief.
    """
    v = float(ego_speeds[0])
    eligible = (enabled and self.age >= 1.0 and self.margin <= 0.01
                and base_tf > 0.7 and v > 5.0 and lead_speeds[0] > 2.0
                and lead_accel < -0.3 and -2.5 < self.relative_speed < 0.0
                and self.distance - stop_distance > v * (base_tf + 0.2)
                and comfort_path_has_reserve(ego_speeds, physical_gaps, danger_margins, times, stop_distance))
    # Hazard withdrawal is immediate. This is a preference, never a brake cap.
    self.relief = min(1.0, self.relief + dt / 0.5) if eligible else 0.0
    if not eligible:
      return np.zeros_like(ego_speeds)
    return self.relief * min(0.4, max(0.0, base_tf - 0.7)) * np.maximum(ego_speeds, 0.0)


def braking_gap_reference(obstacles, margins, ego_speeds):
  preferred = obstacles.copy()
  preferred[:, :2] -= margins
  # Signed shift allows comfort slack while retaining every other obstacle.
  return (np.min(obstacles, axis=1) - np.min(preferred, axis=1)) / (np.maximum(ego_speeds, 0.0) + 10.0)
