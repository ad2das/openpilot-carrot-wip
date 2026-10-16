import json
import math

import numpy as np

from openpilot.common.realtime import DT_MDL

# Learned aim offset (m, model path-y units) per road curvature knot (1/m, path-y units).
KNOTS = np.array([-0.010, -0.005, -0.002, 0.0, 0.002, 0.005, 0.010])
OFFSET_MAX = 0.30        # per-knot aim offset cap, m
T_I = 5.0                # learning time constant, s (route replay: only 5 s reaches 70% of the
                         # left-curve target within one drive on both reviewer routes; 8/10 stall)
DEADBAND = 0.05          # no learning below this car-to-centre error, m
LOOKAHEAD_S = 2.0        # pure-pursuit lookahead time for offset -> curvature
LOOKAHEAD_MIN = 20.0
LOOKAHEAD_MAX = 60.0
LAT_ACCEL_MAX = 0.25     # |dk| * v^2 cap, m/s^2
V_OFF = 30.0 / 3.6
V_ON = 40.0 / 3.6
PROB_MIN = 0.7
WIDTH_MIN = 2.6
WIDTH_MAX = 4.2
KAPPA_MAX = 0.02         # |road curvature| above this (R < 50 m): fade out
LANE_CHANGE_HOLD_S = 3.0
E_MAX = 0.6              # no learning when the car is further than this from centre
JUMP_M = 0.3             # lane-centre jump per model frame
JUMP_HOLD_S = 2.0
# Avoid thresholds calibrated on 89,061 usable route frames (v>=40 km/h mask for
# learning; exclusions: lane changes, desire != none, |e| jumps, lane probs < 0.7).
# Pooled p99.5: |r| 0.519 m, |r_rate| 0.237 m/s, |r_dev| 0.503 m. The r_dev cap of
# 0.35 m is the binding calibration limit, so |r_dev| above 0.35 m alone trips.
AVOID_R_MAX = 0.52       # max |model path - lane centre| within 3 s, m
AVOID_DEV_MAX = 0.35     # |(path - centre at 3 s) - its learned baseline|, m
T_BASE = 10.0            # baseline tracking time constant while not avoiding, s
T_BASE_HOLD = 60.0       # baseline tracking time constant while avoiding, s
AVOID_RATE_MAX = 0.237   # |d/dt (path - centre) at 1.5 s, m/s
AVOID_RATE_TAU = 0.5
AVOID_CLEAR_S = 2.0
# A driver's short wheel blips (median 0.2-0.25 s in the route logs) must not
# interrupt learning; only a sustained hold does.
STEER_BLOCK_MIN_S = 0.5  # steering_pressed must last this long to block learning
STEER_BLOCK_HOLD_S = 1.0 # once blocked, keep blocking this long after release
FIT_HORIZON_S = 2.0
FIT_MIN_M = 20.0
R_HORIZON_S = 3.0
R_RATE_T = 1.5
W_UP = 1.0               # output weight ramp, 1/s
W_DOWN = 2.0

GATE_OFF = 1
GATE_LANE_MODE = 2
GATE_SPEED = 4
GATE_PROB = 8
GATE_WIDTH = 16
GATE_LANE_CHANGE = 32
GATE_CURVE = 64
GATE_AVOID = 128
GATE_STEER = 256
GATE_E_LARGE = 512
GATE_JUMP = 1024
GATE_INVALID = 2048

TRAJECTORY_SIZE = 33


def load_table(text) -> np.ndarray:
  try:
    if isinstance(text, bytes):
      text = text.decode("utf-8")
    values = json.loads(text)
    table = np.array([float(v) for v in values], dtype=float)
    if table.shape != KNOTS.shape or not np.all(np.isfinite(table)):
      return np.zeros_like(KNOTS)
    return np.clip(table, -OFFSET_MAX, OFFSET_MAX)
  except Exception:
    return np.zeros_like(KNOTS)


def dump_table(table: np.ndarray) -> str:
  return json.dumps([round(float(v), 4) for v in table])


BASE_MAX = AVOID_R_MAX + AVOID_DEV_MAX  # baseline persistence clip, m


def load_base(text) -> np.ndarray:
  try:
    if isinstance(text, bytes):
      text = text.decode("utf-8")
    values = json.loads(text)
    base = np.array([float(v) for v in values], dtype=float)
    if base.shape != KNOTS.shape or not np.all(np.isfinite(base)):
      return np.zeros_like(KNOTS)
    return np.clip(base, -BASE_MAX, BASE_MAX)
  except Exception:
    return np.zeros_like(KNOTS)


def dump_base(base: np.ndarray) -> str:
  return json.dumps([round(float(v), 4) for v in base])


def knot_weights(kappa: float) -> np.ndarray:
  k = float(np.clip(kappa, KNOTS[0], KNOTS[-1]))
  return np.array([np.interp(k, KNOTS, np.eye(len(KNOTS))[i]) for i in range(len(KNOTS))])


class LaneCenterBlend:
  def __init__(self, table=None, base=None):
    self.table = np.zeros_like(KNOTS) if table is None else np.clip(np.array(table, dtype=float), -OFFSET_MAX, OFFSET_MAX)
    self.w = 0.0
    self.offset = 0.0
    self.curvature = 0.0
    self.gates = GATE_OFF
    self.e = 0.0
    self.kappa_road = 0.0
    self.e_prev = None
    self.r_prev = None
    self.r_rate = 0.0
    self.r_base = np.zeros_like(KNOTS) if base is None else np.clip(np.array(base, dtype=float), -BASE_MAX, BASE_MAX)
    self.r_far = 0.0
    self.r_dev = 0.0
    self.lane_change_t = 0.0
    self.jump_t = 0.0
    self.avoid_clear_t = 0.0
    self.steer_held_s = 0.0
    self.steer_block = False
    self.steer_release_s = 0.0

  def reset(self):
    self.table = np.zeros_like(KNOTS)
    self.r_base = np.zeros_like(KNOTS)
    self.steer_held_s = 0.0
    self.steer_block = False
    self.steer_release_s = 0.0

  def update(self, enabled, lane_mode, v_ego, l_prob, r_prob, ll_x, lll_y, rll_y,
             path_t, path_x, path_y, lane_change_active, steering_pressed) -> float:
    if not enabled:
      self.w = 0.0
      self.offset = 0.0
      self.curvature = 0.0
      self.gates = GATE_OFF
      self.e_prev = None
      self.r_prev = None
      self.r_rate = 0.0
      return 0.0

    gates = 0
    self.lane_change_t = LANE_CHANGE_HOLD_S if lane_change_active else max(0.0, self.lane_change_t - DT_MDL)
    self.jump_t = max(0.0, self.jump_t - DT_MDL)

    ll_x = np.asarray(ll_x, dtype=float)
    lll_y = np.asarray(lll_y, dtype=float)
    rll_y = np.asarray(rll_y, dtype=float)
    path_t = np.asarray(path_t, dtype=float)
    path_x = np.asarray(path_x, dtype=float)
    path_y = np.asarray(path_y, dtype=float)
    valid = (all(len(a) == TRAJECTORY_SIZE for a in (ll_x, lll_y, rll_y, path_t, path_x, path_y)) and
             all(np.all(np.isfinite(a)) for a in (ll_x, lll_y, rll_y, path_t, path_x, path_y)) and
             math.isfinite(v_ego) and np.all(np.diff(ll_x) > 0))

    avoid_now = False
    if valid:
      centre = (lll_y + rll_y) / 2.0
      self.e = float(centre[0])
      width = float(rll_y[0] - lll_y[0])
      fit = ll_x <= max(FIT_HORIZON_S * v_ego, FIT_MIN_M)
      self.kappa_road = float(2.0 * np.polyfit(ll_x[fit], centre[fit], 2)[0]) if np.count_nonzero(fit) >= 3 else 0.0
      near = path_t <= R_HORIZON_S
      r = path_y[near] - np.interp(path_x[near], ll_x, centre)
      r_mid = float(np.interp(R_RATE_T, path_t, path_y) - np.interp(np.interp(R_RATE_T, path_t, path_x), ll_x, centre))
      if self.r_prev is not None:
        alpha = DT_MDL / (AVOID_RATE_TAU + DT_MDL)
        self.r_rate += alpha * ((r_mid - self.r_prev) / DT_MDL - self.r_rate)
      self.r_prev = r_mid
      self.r_far = float(np.interp(R_HORIZON_S, path_t, path_y) - np.interp(np.interp(R_HORIZON_S, path_t, path_x), ll_x, centre))
      self.r_dev = self.r_far - float(knot_weights(self.kappa_road) @ self.r_base)
      avoid_now = (float(np.max(np.abs(r))) > AVOID_R_MAX or abs(self.r_rate) > AVOID_RATE_MAX or
                   abs(self.r_dev) > AVOID_DEV_MAX)
      if self.e_prev is not None and abs(self.e - self.e_prev) > JUMP_M:
        self.jump_t = JUMP_HOLD_S
      self.e_prev = self.e

      if lane_mode:
        gates |= GATE_LANE_MODE
      if v_ego < V_OFF:
        gates |= GATE_SPEED
      if l_prob < PROB_MIN or r_prob < PROB_MIN:
        gates |= GATE_PROB
      if not (WIDTH_MIN <= width <= WIDTH_MAX):
        gates |= GATE_WIDTH
      if self.lane_change_t > 0.0:
        gates |= GATE_LANE_CHANGE
      if abs(self.kappa_road) > KAPPA_MAX:
        gates |= GATE_CURVE
      if avoid_now:
        gates |= GATE_AVOID
    else:
      gates |= GATE_INVALID
      self.e_prev = None
      self.r_prev = None
      self.r_rate = 0.0

    self.avoid_clear_t = 0.0 if (avoid_now or not valid) else self.avoid_clear_t + DT_MDL
    output_ok = gates == 0
    w_target = float(np.interp(v_ego, [V_OFF, V_ON], [0.0, 1.0])) if output_ok else 0.0
    if w_target > self.w:
      self.w = min(w_target, self.w + W_UP * DT_MDL)
    else:
      self.w = max(w_target, self.w - W_DOWN * DT_MDL)

    if steering_pressed:
      self.steer_held_s += DT_MDL
      if self.steer_held_s >= STEER_BLOCK_MIN_S:
        self.steer_block = True
      self.steer_release_s = 0.0
    else:
      self.steer_held_s = 0.0
      if self.steer_block:
        self.steer_release_s += DT_MDL
        if self.steer_release_s >= STEER_BLOCK_HOLD_S:
          self.steer_block = False

    learn_block = 0
    if self.steer_block:
      learn_block |= GATE_STEER
    if valid and abs(self.e) > E_MAX:
      learn_block |= GATE_E_LARGE
    if self.jump_t > 0.0:
      learn_block |= GATE_JUMP
    learn = output_ok and learn_block == 0 and v_ego >= V_ON and self.avoid_clear_t >= AVOID_CLEAR_S
    self.gates = gates | learn_block

    h = knot_weights(self.kappa_road)
    # The table is persistent knowledge: a blocked frame pauses learning but never
    # erodes what was learned, so gated stretches (avoids, lane changes, low
    # confidence) cannot shrink the map.
    if learn and abs(self.e) > DEADBAND:
      self.table = np.clip(self.table + h * self.e * DT_MDL / T_I, -OFFSET_MAX, OFFSET_MAX)

    if (gates & ~GATE_AVOID) == 0 and not steering_pressed:
      t_base = T_BASE_HOLD if avoid_now else T_BASE
      self.r_base = np.clip(self.r_base + h * (self.r_far - float(h @ self.r_base)) * DT_MDL / t_base,
                            -BASE_MAX, BASE_MAX)

    self.offset = float(h @ self.table)
    lookahead = float(np.clip(LOOKAHEAD_S * v_ego, LOOKAHEAD_MIN, LOOKAHEAD_MAX))
    k_max = LAT_ACCEL_MAX / max(v_ego, 1.0) ** 2
    self.curvature = float(np.clip(2.0 * self.w * self.offset / lookahead ** 2, -k_max, k_max))
    return self.curvature
