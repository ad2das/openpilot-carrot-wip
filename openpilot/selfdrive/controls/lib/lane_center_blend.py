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

# Lane recenter assist (LRA).  It only accelerates a return the model already
# plans: a monotone path-to-centre offset that shrinks over the 3 s horizon.
REC_HORIZON_S = 3.0
R_ON = 0.15              # activation threshold and hard output gate, m
R_OFF = 0.08             # release the latch below this offset, m
R_MAX = 0.8              # larger offsets are not a normal lane-centre return, m
R_NEAR = 0.05            # offsets below this may cross the sign without a flip, m
R_MONO_TOL = 0.03        # allowed |r| growth between samples while returning, m
RET_MIN = 0.05           # required |r0| - |r(3 s)| shrinkage, m
REC_T_L = 1.5            # lookahead used to read the model's planned offset, s
ENTER_HOLD_S = 0.2       # the full entry condition must hold this long first, s
R_REENTER_S = 0.5        # hold-off after any release before the latch may re-enter, s
EXIT_HOLD_S = 0.2        # soft conditions must fail this long before the latch drops, s
REC_FILT_TAU = 0.25      # EMA time constant for the soft-check offset profile, s
WIDTH_DELTA_MAX = 0.4    # lane-width change across the lookahead that blocks entry, m
BOOST_S = 0.5            # aim at this fraction of the planned offset
LRA_ACC_MAX = 0.3        # |k_lra| * v^2 cap, m/s^2
LRA_JERK = 0.6           # |dk_lra| * v^2 per second cap, m/s^3
LAT_ON = 1.0             # |kappa_road| * v^2 limit for assist, m/s^2

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


def knot_weights(kappa: float) -> np.ndarray:
  k = float(np.clip(kappa, KNOTS[0], KNOTS[-1]))
  return np.array([np.interp(k, KNOTS, np.eye(len(KNOTS))[i]) for i in range(len(KNOTS))])


def recenter_profile(ll_x, centre, path_t, path_x, path_y, horizon=REC_HORIZON_S):
  """Model path offset from lane centre, r(t) = path_y(t) - centre(path_x(t)).

  Samples at the model's t_idxs up to `horizon`.  Positive r means the model
  path (and so the car, at t=0) is right of the lane centre in path-y units.
  Returns (t, r) or (None, None) when the horizon holds fewer than two samples.
  """
  near = path_t <= horizon
  if np.count_nonzero(near) < 2:
    return None, None
  r = path_y[near] - np.interp(path_x[near], ll_x, centre)
  return path_t[near], r


def recenter_return_checks(t, r, r0):
  """Monotone-return checks shared by the runtime assist and its tests.

  Returns (sign_ok, mono_ok, retract_ok):
    sign_ok    every sample keeps r0's sign, or is within R_NEAR of centre
    mono_ok    |r| never grows by more than R_MONO_TOL between samples
    retract_ok the horizon end is at least RET_MIN closer than r0
  """
  sign_ok = bool(np.all((np.sign(r) == np.sign(r0)) | (np.abs(r) < R_NEAR)))
  mono_ok = bool(np.all(np.abs(r[1:]) <= np.abs(r[:-1]) + R_MONO_TOL))
  retract_ok = abs(r0) - abs(float(r[-1])) >= RET_MIN
  return sign_ok, mono_ok, retract_ok


def recenter_width_delta(ll_x, lll_y, rll_y, v_ego):
  """Lane-width change from the bumper to the lane-recenter lookahead, m.

  Tapers and gores widen (or narrow) the lane ahead, which moves the estimated
  lane centre and makes a model return look real.  The width is interpolated at
  x_L = clip(v*REC_T_L, 15, 45) m and compared with the width at x=0.
  """
  x_l = float(np.clip(v_ego * REC_T_L, 15.0, 45.0))
  width0 = float(rll_y[0] - lll_y[0])
  width_l = float(np.interp(x_l, ll_x, rll_y - lll_y))
  return width_l - width0


class LaneCenterBlend:
  def __init__(self, table=None):
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
    self.r_base = np.zeros_like(KNOTS)
    self.r_far = 0.0
    self.r_dev = 0.0
    self.lane_change_t = 0.0
    self.jump_t = 0.0
    self.avoid_clear_t = 0.0
    self.steer_held_s = 0.0
    self.steer_block = False
    self.steer_release_s = 0.0
    # Lane recenter assist (only used through update(..., recenter_enabled=True)).
    self.recenter_curvature = 0.0
    self.recenter_active = False
    self.recenter_k = 0.0
    self.recenter_latched = False
    self.recenter_r0 = 0.0
    self.recenter_hold_s = 0.0
    self.recenter_enter_s = 0.0
    self.recenter_soft_s = 0.0
    self.recenter_r_filt = None
    self.recenter_releases = {}
    self.recenter_width_delta = 0.0

  def reset(self):
    self.table = np.zeros_like(KNOTS)
    self.r_base = np.zeros_like(KNOTS)
    self.steer_held_s = 0.0
    self.steer_block = False
    self.steer_release_s = 0.0
    self.recenter_curvature = 0.0
    self.recenter_active = False
    self.recenter_k = 0.0
    self.recenter_latched = False
    self.recenter_r0 = 0.0
    self.recenter_hold_s = 0.0
    self.recenter_enter_s = 0.0
    self.recenter_soft_s = 0.0
    self.recenter_r_filt = None
    self.recenter_releases = {}
    self.recenter_width_delta = 0.0

  def _update_steer_block(self, steering_pressed):
    """Debounced driver-steering latch shared by both blend branches."""
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

  def update(self, enabled, lane_mode, v_ego, l_prob, r_prob, ll_x, lll_y, rll_y,
             path_t, path_x, path_y, lane_change_active, steering_pressed,
             recenter_enabled=False) -> float:
    if not enabled and not recenter_enabled:
      self.w = 0.0
      self.offset = 0.0
      self.curvature = 0.0
      self.gates = GATE_OFF
      self.e_prev = None
      self.r_prev = None
      self.r_rate = 0.0
      self._update_recenter(False, v_ego, 0, None, steering_pressed)
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
    if not enabled:
      # LaneCenterBlend=0. Every blend output keeps its OFF value, but the
      # optional lane recenter assist still needs the same model geometry.
      self.w = 0.0
      self.offset = 0.0
      self.curvature = 0.0
      self.gates = GATE_OFF
      geom = None
      self.recenter_width_delta = 0.0
      if valid:
        centre = (lll_y + rll_y) / 2.0
        self.e = float(centre[0])
        width = float(rll_y[0] - lll_y[0])
        fit = ll_x <= max(FIT_HORIZON_S * v_ego, FIT_MIN_M)
        self.kappa_road = float(2.0 * np.polyfit(ll_x[fit], centre[fit], 2)[0]) if np.count_nonzero(fit) >= 3 else 0.0
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
        r_mid = float(np.interp(R_RATE_T, path_t, path_y) - np.interp(np.interp(R_RATE_T, path_t, path_x), ll_x, centre))
        if self.r_prev is not None:
          alpha = DT_MDL / (AVOID_RATE_TAU + DT_MDL)
          self.r_rate += alpha * ((r_mid - self.r_prev) / DT_MDL - self.r_rate)
        self.r_prev = r_mid
        self.recenter_width_delta = recenter_width_delta(ll_x, lll_y, rll_y, v_ego)
        if abs(self.kappa_road) * v_ego ** 2 >= LAT_ON:
          geom = None
        else:
          geom = (centre, ll_x, path_t, path_x, path_y)
      else:
        gates |= GATE_INVALID
        self.e_prev = None
        self.r_prev = None
        self.r_rate = 0.0
      self._update_steer_block(steering_pressed)
      if self.steer_block or self.jump_t > 0.0:
        gates |= GATE_STEER
      self._update_recenter(recenter_enabled, v_ego, gates, geom, steering_pressed)
      return 0.0

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
      self.recenter_width_delta = recenter_width_delta(ll_x, lll_y, rll_y, v_ego)
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
      geom = None
    if valid:
      geom = (centre, ll_x, path_t, path_x, path_y)

    self.avoid_clear_t = 0.0 if (avoid_now or not valid) else self.avoid_clear_t + DT_MDL
    output_ok = gates == 0
    w_target = float(np.interp(v_ego, [V_OFF, V_ON], [0.0, 1.0])) if output_ok else 0.0
    if w_target > self.w:
      self.w = min(w_target, self.w + W_UP * DT_MDL)
    else:
      self.w = max(w_target, self.w - W_DOWN * DT_MDL)

    self._update_steer_block(steering_pressed)

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
      self.r_base = self.r_base + h * (self.r_far - float(h @ self.r_base)) * DT_MDL / t_base

    self.offset = float(h @ self.table)
    lookahead = float(np.clip(LOOKAHEAD_S * v_ego, LOOKAHEAD_MIN, LOOKAHEAD_MAX))
    k_max = LAT_ACCEL_MAX / max(v_ego, 1.0) ** 2
    self.curvature = float(np.clip(2.0 * self.w * self.offset / lookahead ** 2, -k_max, k_max))
    recenter_gates = gates & ~(GATE_AVOID | GATE_CURVE)
    if self.steer_block or self.jump_t > 0.0:
      recenter_gates |= GATE_STEER
    recenter_geom = geom if (valid and abs(self.kappa_road) * v_ego ** 2 < LAT_ON) else None
    self._update_recenter(recenter_enabled, v_ego, recenter_gates, recenter_geom, steering_pressed)
    return self.curvature

  def _recenter_target_k(self, t, r, v_ego):
    """Curvature that makes the model's planned offset shrink twice as fast.

    The planned offset at REC_T_L is `y_m`; aiming at `BOOST_S * y_m` over the
    same distance needs k = 2 * dy / L^2.  Returns the speed-weighted, capped k.
    """
    y_m = float(np.interp(REC_T_L, t, r))
    dy = (BOOST_S - 1.0) * y_m
    length = float(np.clip(v_ego * REC_T_L, 15.0, 45.0))
    k = 2.0 * dy / length ** 2
    k_cap = LRA_ACC_MAX / max(v_ego, 1.0) ** 2
    weight = float(np.interp(v_ego, [V_OFF, V_ON], [0.0, 1.0]))
    return float(np.clip(k * weight, -k_cap, k_cap))

  def _release_recenter(self, reason):
    """Drop the latch and start the re-entry hold-off for *every* release kind."""
    if self.recenter_latched:
      self.recenter_releases[reason] = self.recenter_releases.get(reason, 0) + 1
    self.recenter_latched = False
    self.recenter_hold_s = R_REENTER_S
    self.recenter_enter_s = 0.0
    self.recenter_soft_s = 0.0
    self.recenter_r_filt = None

  def _soft_return_checks(self, t, r, r0):
    """mono/retract on the EMA-smoothed offset; sign stays on the raw profile.

    Per-frame model noise (vision jitter, knot re-fitting) makes a genuinely
    monotone return fail `|r|` non-increase and the retraction margin on raw
    samples.  The soft checks only decide whether the assist is *allowed*, so
    they run on a 0.25 s EMA.  The output target and the sign guard stay raw.
    """
    if self.recenter_r_filt is None or len(self.recenter_r_filt) != len(r):
      self.recenter_r_filt = np.array(r, dtype=float)
    else:
      alpha = DT_MDL / (REC_FILT_TAU + DT_MDL)
      self.recenter_r_filt += alpha * (np.asarray(r, dtype=float) - self.recenter_r_filt)
    _, mono_ok, retract_ok = recenter_return_checks(t, self.recenter_r_filt, float(r0))
    return mono_ok, retract_ok

  def _update_recenter(self, enabled, v_ego, gates, geom, steering_pressed):
    """Lane recenter assist: extra curvature that doubles an already-planned return.

    Only ever runs when the caller asks for it, the laneless branch is used and
    the road is not turning.  OFF leaves `recenter_curvature` exactly 0.0; when a
    gate closes the output is slewed back to zero rather than stepped.
    """
    if not enabled:
      self.recenter_curvature = 0.0
      self.recenter_k = 0.0
      self.recenter_active = False
      self.recenter_latched = False
      self.recenter_hold_s = 0.0
      self.recenter_enter_s = 0.0
      self.recenter_soft_s = 0.0
      self.recenter_r_filt = None
      return

    prev_k = self.recenter_k
    target_k = 0.0
    active = False
    self.recenter_hold_s = max(0.0, self.recenter_hold_s - DT_MDL)

    r0 = 0.0
    if geom is not None:
      t, r = recenter_profile(geom[1], geom[0], geom[2], geom[3], geom[4])
      if t is not None:
        r0 = float(r[0])
        # Sign and the r0 band are RAW: the |r0| < R_ON = 0 guarantee is on the
        # real, instantaneous offset.  Only mono/retract use the smoothed plan.
        sign_ok = bool(np.all((np.sign(r) == np.sign(r0)) | (np.abs(r) < R_NEAR)))
        mono_ok, retract_ok = self._soft_return_checks(t, r, r0)
        rate_away = bool(np.sign(self.r_rate) == np.sign(r0)) and abs(self.r_rate) >= 0.05
        base_ok = (sign_ok and mono_ok and retract_ok and not rate_away and
                   gates == 0 and not steering_pressed and v_ego >= V_OFF)
        if self.recenter_latched:
          # Hard conditions drop the latch at once; soft ones only stop this
          # frame's output and must fail EXIT_HOLD_S of consecutive frames
          # before the latch goes, so one noisy sample cannot end a real return.
          reason = None
          if not sign_ok:
            reason = 'sign'
          elif abs(r0) < R_OFF:
            reason = 'r_off'
          elif abs(r0) > R_MAX:
            reason = 'r_max'
          elif v_ego < V_OFF:
            reason = 'speed'
          elif abs(self.recenter_width_delta) > WIDTH_DELTA_MAX:
            reason = 'width'
          elif steering_pressed:
            reason = 'steer'
          elif gates != 0:
            reason = 'gate_steer' if (gates & GATE_STEER) else 'gate'

          soft_reason = None
          if not mono_ok:
            soft_reason = 'mono'
          elif not retract_ok:
            soft_reason = 'retract'
          elif rate_away:
            soft_reason = 'rate'

          if reason is not None:
            self._release_recenter(reason)
          elif soft_reason is not None:
            # Output stops immediately (slew down); the latch survives.
            target_k = 0.0
            active = False
            self.recenter_soft_s += DT_MDL
            if self.recenter_soft_s >= EXIT_HOLD_S - 1e-12:
              self._release_recenter(soft_reason)
          else:
            self.recenter_soft_s = 0.0
            # Same consecutive-frame rule for the intent as for entry: dropping
            # below R_ON (still latched) stops the assist, and it only comes
            # back after the full hold again, so the band cannot chatter.
            if abs(r0) >= R_ON:
              self.recenter_enter_s += DT_MDL
            else:
              self.recenter_enter_s = 0.0
            if abs(r0) >= R_ON and self.recenter_enter_s >= ENTER_HOLD_S - 1e-12:
              target_k = self._recenter_target_k(t, r, v_ego)
              active = target_k != 0.0
        if not self.recenter_latched:
          enter_ok = (base_ok and R_ON <= abs(r0) <= R_MAX and
                      abs(self.recenter_width_delta) <= WIDTH_DELTA_MAX)
          if enter_ok and self.recenter_hold_s <= 0.0:
            # Consecutive-frame debounce: one failing frame restarts the count.
            self.recenter_enter_s += DT_MDL
            if self.recenter_enter_s >= ENTER_HOLD_S - 1e-12:
              self.recenter_latched = True
              self.recenter_r0 = r0
              # The intent is already through the debounce; keep the counter
              # saturated so the latched branch does not drop it next frame.
              self.recenter_enter_s = ENTER_HOLD_S
              target_k = self._recenter_target_k(t, r, v_ego)
              active = target_k != 0.0
          else:
            self.recenter_enter_s = 0.0
    else:
      self.recenter_enter_s = 0.0
      if self.recenter_latched:
        self._release_recenter('no_geometry')

    slew = LRA_JERK / max(v_ego, 1.0) ** 2 * DT_MDL
    self.recenter_k = float(np.clip(target_k, prev_k - slew, prev_k + slew))
    if abs(self.recenter_k) < 1e-9:
      self.recenter_k = 0.0
    self.recenter_curvature = self.recenter_k
    # `active` is the intent (assist requested this frame); the curvature itself
    # can still be slewing down after the intent drops.
    self.recenter_active = bool(active)
