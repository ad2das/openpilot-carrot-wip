"""Opt-in log-fitted longitudinal plant for the ButterStop sim loops.

Source
------
Fitted per car on real Palisade and K5 ButterStop stops in the W-PLANT2 study:
  .analysis/scratch/2026-11-08-perf/W-PLANT2/fitted_plant2.py  (model FittedPlant2)
  .analysis/scratch/2026-11-08-perf/W-PLANT2/kf.py             (car speed filter)
  .analysis/scratch/2026-11-08-perf/W-PLANT2/params2_palisade.json
  .analysis/scratch/2026-11-08-perf/W-PLANT2/params2_k5.json

Model kinds ported here (chosen on held-out events, W-PLANT2/raw/gate2.txt):
  palisade -> "M"   end-decel err 0.0473 (M2 0.1031, M3 0.1011)
  k5       -> "M3"  end-decel err 0.0336 (M2 0.0476, M  0.2252)

What it models
--------------
Command -> motion, with an actuator lag (two time constants split at a speed
threshold), AT creep torque, a stop blend near rest, and the car's own
2-state speed Kalman filter as the output stage -- so the loop is fed vEgo/aEgo
the way the car reports them, not a true speed. "M" has no stopping-state
branch; "M3" additionally replaces the command with the car's own
brake-pressure-vs-speed schedule while longControlState == stopping (2) and
applies a low-speed command gain ahead of the lag.

Scope
-----
Distribution-level model. Held-out scoring of the source class passes on
end-decel error but misses per-event stop time and stop distance, so this plant
reproduces the shape of this car's stops, not any individual stop. It was fit
with u = carOutput.actuatorsOutput.accel (after the CarController rate limits);
the sims feed LongControl's output directly, and on the W-PLANT1 event caches
the two differ by median 0.005/0.008 and p95 0.029/0.034 m/s^2 for
Palisade/K5 over the pid (longControlState == 1) samples.

The class is a straight port of FittedPlant2 plus the kf.py gain computation;
test_fitted_plant.py pins it to the source class through a golden trace.

Usage
-----
    plant = FittedPlant("palisade")
    plant.reset(v0, a0)
    a_true, v_true, v_kf, a_kf, standstill = plant.step(u, lcs, dt)
"""
import numpy as np

DT_CTRL = 0.01
KF_Q = np.array([[0.0, 0.0], [0.0, 100.0]])
KF_R = 0.3
KF_A = np.array([[1.0, DT_CTRL], [0.0, 1.0]])
KF_C = np.array([[1.0, 0.0]])
RESET_JUMP = 2.0           # |v_true - v_kf| above this re-seeds the filter
BLEND_RAMP_DEFAULT = 0.20  # m/s; fixed numerical width of the stop blend
LCS_STOPPING = 2


def get_kalman_gain(dt, a, c, q, r, iterations=100):
  """opendbc.car.common.simple_kalman.get_kalman_gain, verbatim."""
  p = np.zeros_like(q)
  for _ in range(iterations):
    p = a.dot(p).dot(a.T) + dt * q
    s = c.dot(p).dot(c.T) + r
    k = p.dot(c.T).dot(np.linalg.inv(s))
    p = (np.eye(len(p)) - k.dot(c)).dot(p)
  return k


_K = get_kalman_gain(DT_CTRL, KF_A, KF_C, KF_Q, KF_R)
K0 = float(_K[0, 0])
K1 = float(_K[1, 0])
_A_K = KF_A - _K.dot(KF_C)
A_K = (float(_A_K[0, 0]), float(_A_K[0, 1]), float(_A_K[1, 0]), float(_A_K[1, 1]))

# Fitted parameters, copied verbatim (repr of floats) from params2_*.json.
# Keys are the model dict's own keys after FittedPlant2.from_json flattens
# them; the standing-still edge comes from the file's top-level key.
CARS = {
  "palisade": {
    "kind": "M",
    "standstill_edge_speed": 0.096,
    "params": {
      "delay": 0.1,
      "tau_hi": 0.287945362707177,
      "tau_lo": 1.486805212862769,
      "v_sw": 2.0127599220487564,
      "c0": 0.10937088065844244,
      "v1": 1.870631297678612,
      "dv": 1.3972041333913905,
      "v_e": 0.8193796165044444,
      "beta": 0.6450982211979248,
      "a_e": -0.05214955353126603,
    },
  },
  "k5": {
    "kind": "M3",
    "standstill_edge_speed": 0.097,
    "params": {
      "delay": 0.680354603714616,
      "tau_hi": 1.4999999999582367,
      "tau_lo": 0.21210645071031278,
      "v_sw": 2.2790436863004646,
      "c0": 0.13611261231312716,
      "v1": 1.9999999999996565,
      "dv": 2.9999999999999996,
      "v_e": 1.4999999999964049,
      "beta": 0.5250236547727688,
      "a_e": -0.3818577452078355,
      "a_stop": -0.9999999999941585,
      "v0": 0.2952905076953211,
      "dv_stop": 0.10182940172513096,
      "k1": -0.7223612109386464,
      "v_g": 1.1900173534768912,
    },
  },
}


def creep_of(v, c0, v1, v2):
  """AT creep torque vs speed: flat c0 below v1, linear to 0 at v2."""
  if c0 <= 0.0:
    return 0.0
  if v < v1:
    return c0
  if v < v2:
    return c0 * (v2 - v) / (v2 - v1)
  return 0.0


class FittedPlant:
  """Command -> motion plant + the car's speed filter + stopping-state branch."""

  def __init__(self, car):
    if car not in CARS:
      raise ValueError(f"unknown fitted car {car!r}; choose from {sorted(CARS)}")
    spec = CARS[car]
    self.car = car
    self.kind = spec["kind"]
    p = spec["params"]
    self.delay = float(p.get("delay", 0.0))
    self.tau_hi = float(p.get("tau_hi", p.get("tau", 0.3)))
    self.tau_lo = float(p.get("tau_lo", self.tau_hi))
    self.v_sw = float(p.get("v_sw", 0.0))
    self.c0 = float(p.get("c0", 0.0))
    self.v1 = float(p.get("v1", 0.0))
    self.v2 = self.v1 + float(p.get("dv", 0.0))
    self.v_e = float(p.get("v_e", 0.0))
    self.beta = float(p.get("beta", 0.0))
    self.a_e = float(p.get("a_e", 0.0))
    self.has_stop = "a_stop" in p
    self.a_stop = float(p.get("a_stop", 0.0))
    self.s_v0 = float(p.get("v0", 0.0))
    self.s_dv = float(p.get("dv_stop", 1.0))
    self.has_gain = "k1" in p
    self.k1 = float(p.get("k1", 0.0))
    self.v_g = float(p.get("v_g", 1.0))
    self.blend_w = min(BLEND_RAMP_DEFAULT, self.v_e) if self.v_e > 0 else 0.0
    self.do_blend = self.v_e > 0.0 and self.beta > 0.0
    self.standstill_edge = float(spec["standstill_edge_speed"])
    self.reset(0.0)

  # -- state ---------------------------------------------------------------
  def reset(self, v0, a0=0.0):
    """Seed the plant at true speed v0.

    a0 only fills the reported a_kf before the first step; the filter itself is
    seeded on the first step from the command in force (the source class's
    convention, which is what its batch fit does).
    """
    self.v = float(v0)
    self.a_lag = None            # set from the first command
    self.kf_x0 = None            # set on the first step
    self.kf_x1 = None
    self.first = True
    self._v_kf = float(v0)
    self._a_kf = float(a0)
    self._standstill = bool(self._v_kf <= self.standstill_edge)

  @property
  def v_kf(self):
    """Last reported speed (the car's filter output); v0 before any step."""
    return self._v_kf

  @property
  def a_kf(self):
    """Last reported acceleration (the car's filter output); a0 before any step."""
    return self._a_kf

  @property
  def standstill(self):
    """True while the reported speed is at or below this car's standstill edge."""
    return self._standstill

  # -- step ----------------------------------------------------------------
  def step(self, u, lcs, dt):
    """One Euler step.

    u   = command (already delayed by the caller's own convention)
    lcs = longControlState (2 = stopping; "M3" then ignores u)
    dt  = step size

    Returns (a_true, v_true, v_kf, a_kf, standstill).
    """
    u = float(u)
    dt = float(dt)
    lcs = int(lcs)
    if self.first:
      self.a_lag = u
    # --- command path -----------------------------------------------------
    u_in = u
    if self.has_stop and lcs == LCS_STOPPING:
      arg = (self.v - self.s_v0) / self.s_dv
      arg = 0.0 if arg < 0.0 else (1.0 if arg > 1.0 else arg)
      u_in = self.a_stop * arg
    elif self.has_gain:
      arg = (self.v_g - self.v) / self.v_g
      arg = 0.0 if arg < 0.0 else (1.0 if arg > 1.0 else arg)
      u_in = u * (1.0 + self.k1 * arg)
    # --- actuator lag -----------------------------------------------------
    tau = self.tau_hi if self.v >= self.v_sw else self.tau_lo
    self.a_lag += (dt / tau) * (u_in - self.a_lag)
    a = self.a_lag
    a += creep_of(self.v, self.c0, self.v1, self.v2)
    # --- stop blend near rest ---------------------------------------------
    if self.do_blend and self.v < self.v_e:
      frac = 1.0 if (self.v_e - self.v) >= self.blend_w \
          else (self.v_e - self.v) / self.blend_w
      b = self.beta * frac
      a = (1.0 - b) * a + b * self.a_e
    # --- integrate --------------------------------------------------------
    if self.v <= 0.0 and a <= 0.0:
      self.v = 0.0
    else:
      self.v = max(0.0, self.v + a * dt)
    v_true = self.v
    # --- output stage: the car's own filter -------------------------------
    if self.kf_x0 is None:
      # The car's filter is never cold at the start of a log; it sits at the
      # command level in force then, not at 0.
      self.kf_x0 = v_true
      self.kf_x1 = u
    if abs(v_true - self.kf_x0) > RESET_JUMP:
      self.kf_x0 = v_true
      self.kf_x1 = 0.0
    x0 = A_K[0] * self.kf_x0 + A_K[1] * self.kf_x1 + K0 * v_true
    x1 = A_K[2] * self.kf_x0 + A_K[3] * self.kf_x1 + K1 * v_true
    self.kf_x0 = x0
    self.kf_x1 = x1
    self.first = False
    self._v_kf = float(x0)
    self._a_kf = float(x1)
    self._standstill = bool(x0 <= self.standstill_edge)
    return a, v_true, x0, x1, self._standstill
