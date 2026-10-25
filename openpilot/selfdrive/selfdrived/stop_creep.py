"""Warn the driver when the car keeps rolling while openpilot asks it to stop.

Route 6db 18:02 (K5, 10-25): the ESC let go just short of rest and the car crawled 7.7 m at
~1 km/h for 30 s while openpilot kept requesting the stop. Nothing in the car warned the driver.
"""
NEAR_REST_V = 0.15   # m/s; the stop must first have come down to this
CREEP_V = 0.20       # m/s; rolling faster than this after that counts as a creep
CREEP_T = 1.0        # s of continuous creep before the warning


class StopCreepWarning:
  def __init__(self, dt: float):
    self.dt = dt
    self.near_rest = False
    self.creep_t = 0.0

  def update(self, enabled: bool, should_stop: bool, v_ego: float, gas: bool, brake: bool) -> bool:
    if not (enabled and should_stop) or gas or brake:
      self.near_rest = False
      self.creep_t = 0.0
      return False
    if v_ego < NEAR_REST_V:
      self.near_rest = True
    self.creep_t = self.creep_t + self.dt if (self.near_rest and v_ego > CREEP_V) else 0.0
    return self.creep_t >= CREEP_T
