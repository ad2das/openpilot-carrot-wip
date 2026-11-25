"""Lane-centre hybrid: eligibility masks, guards, the curve gate and the slew-limited added
lateral acceleration.

Per-frame port of the validated offline reference
(.analysis/scratch/2026-11-20-lane-hybrid/W-SHD6/report_hybrid5.py, md5 3676b6a3aafad1c48d818afbc3c247fc)
and of the eligibility masks in W-HYB1/report_hyb.py:masks().  Constants, comparison directions
and the frame order are copied unchanged; only the array form became a per-frame state machine.
simulate()'s alpha ramp is not ported (a_add never consumes it; SPEC_2B section 1).

Chain per frame, identical to report_hybrid5.metrics_route:
  cont_of -> masks base/strict (AND isfinite(k_model)) -> guards (re-armed) -> gate (ON/latch/drop)
  -> blend_parts (dk, dk_rest, w_c) -> slew_blend (a_add, tgt, zc)
"""
import math
from dataclasses import dataclass

# a frame continues the previous one in time when 0 < dt <= 0.2 s
CONT_MAX_DT = 0.2
# a step is integrated (hold timer, slew) only while 0 < dt <= 0.1 s
STEP_MAX_DT = 0.1

# --- eligibility (report_hyb.py:14-22) ------------------------------------------------------
STRICT_PROB = 0.7
STRICT_STD = 0.25
STRICT_STD_HOLD = 0.35
W_LO, W_HI = 2.45, 4.2
W_TAPER = 0.45
JUMP_C = 0.25
JUMP_W = 0.35
SPIKE_OK_RUN = 10
V_MIN = 40.0 / 3.6

# --- guards (report_hybrid5.py:41-49) -------------------------------------------------------
G_SHAPE_BAR = 0.45
G_W45_TAPER = 0.55
G_LW_WIN_S = 0.5
G_LW_BAR = 0.6
GUARD_REARM_S = 0.5

# --- gate (report_hybrid5.py:50-58) ---------------------------------------------------------
CAP_A = 0.8
GATE_ON_A = 0.65
GATE_REENTRY_A = 0.45
GATE_REENTRY_CS = 0.20
GATE_ROAD_A = 0.3
GATE_HOLD_A = 0.45
HOLD_LATCH_CS = 0.20
HOLD_RELEASE_CS = 0.15
HOLD_S = 0.0
W_C_LO = 0.10
W_C_SPAN = 0.15
ALPHA_MAX = 1.0
J_UP = 0.27
J_DOWN = 0.27

# --- module (SPEC_2B section 1) -------------------------------------------------------------
CS_MEDIAN_N = 3
PREWARM_ROAD_A = 0.20
PREWARM_CURVE_A = 0.40
RUN_V_MIN = 35.0 / 3.6

GUARD_SHAPE = 1
GUARD_W45 = 2
GUARD_LW = 4
GUARD_BAD = 8

NAN = float('nan')


@dataclass
class HybridFrame:
  """One modelV2 frame and everything the chain reads from it."""
  t: float = NAN
  v: float = NAN
  lat: bool = False
  sp: bool = False
  lane_change: bool = False
  lane_mode: bool = False
  c_s: float = NAN
  c0: float = NAN
  w0: float = NAN
  w15: float = NAN
  w30: float = NAN
  lprob: float = NAN
  rprob: float = NAN
  std1: float = NAN
  std2: float = NAN
  spike_judge: bool = False
  w0_g: float = NAN
  w45: float = NAN
  lwl: float = NAN
  lwr: float = NAN
  ymod0: float = NAN
  ymod10: float = NAN
  ylane0: float = NAN
  ylane10: float = NAN
  k_lane: float = NAN
  lane_status: int = -1
  lane_nan: bool = False
  dc: float = NAN
  kappa: float = NAN
  k_model: float = NAN


def _finite(x):
  return isinstance(x, (int, float)) and math.isfinite(x)


def _clip(x, lo, hi):
  """numpy.clip on scalars: a NaN bound propagates."""
  if not _finite(x) or not _finite(lo) or not _finite(hi):
    return NAN
  return lo if x < lo else (hi if x > hi else x)


def _sign(x):
  return 0.0 if x == 0.0 else (1.0 if x > 0.0 else -1.0)


class CausalMedian:
  """Trailing median of the last n stored c values.

  The buffer is cleared when reset is True or when the frame does not continue the previous one
  in time (0 < t - t_prev <= 0.2 s).  A non-finite c is not stored; an empty buffer yields NaN.
  """

  def __init__(self, n=CS_MEDIAN_N):
    self.n = int(n)
    self.reset()

  def reset(self):
    self.buf = []
    self.t_prev = None

  def update(self, t, c, reset=False):
    t = float(t)
    if reset or self.t_prev is None or not (0.0 < t - self.t_prev <= CONT_MAX_DT):
      self.buf = []
    self.t_prev = t
    if _finite(c):
      self.buf.append(float(c))
      if len(self.buf) > self.n:
        del self.buf[0]
    if not self.buf:
      return NAN
    s = sorted(self.buf)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


class CurveLaneHybrid:
  """Per-frame gate from the model lane measurements to a slew-limited a_add (m/s^2)."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.a_add = 0.0
    self.tgt = 0.0
    self.on = False
    self.eligible = False
    self.guard = False
    self.guard_bits = 0
    self.drop = False
    self.zc = False
    self.cont = False
    self.curvature = 0.0
    self._t_prev = None
    self._on = False
    self._latch = False
    self._ht = 0.0
    self._a = 0.0
    self._guard_last_t = None
    self._lw = []
    self._spike_run = SPIKE_OK_RUN
    self._prev_c0 = NAN
    self._prev_w0 = NAN

  # --- masks (report_hyb.py:57-89) ----------------------------------------------------------
  def _base(self, f):
    return bool(f.lat and not f.lane_mode and not f.lane_change and not f.sp
                and _finite(f.v) and f.v > V_MIN
                and _finite(f.c_s) and _finite(f.c0))

  def _strict_inputs(self, f):
    std_lim = STRICT_STD_HOLD if self._on else STRICT_STD
    return bool(f.lprob >= STRICT_PROB and f.rprob >= STRICT_PROB
                and f.std1 <= std_lim and f.std2 <= std_lim
                and W_LO <= f.w0 <= W_HI and W_LO <= f.w15 <= W_HI and W_LO <= f.w30 <= W_HI
                and abs(f.w30 - f.w0) <= W_TAPER)

  def _spike_ok(self, f):
    """A judged c0/w0 jump zeroes the run; SPIKE_OK_RUN clean frames restore it.

    A time gap never resets the counter: the caller decides whether a jump is judged at all
    (offline: both frames in one segment; on the car: a previous frame exists).
    """
    bad = bool(f.spike_judge) and (abs(f.c0 - self._prev_c0) > JUMP_C
                                   or abs(f.w0 - self._prev_w0) > JUMP_W)
    self._prev_c0 = f.c0
    self._prev_w0 = f.w0
    self._spike_run = 0 if bad else self._spike_run + 1
    return self._spike_run >= SPIKE_OK_RUN

  # --- guards (report_hybrid5.py:171-218) ---------------------------------------------------
  def _lane_width_guard(self, f, cont):
    """|max - min| of laneWidthLeft / Right over the last 0.5 s of the same segment."""
    if not cont:
      self._lw = []
    self._lw.append((f.t, f.lwl, f.lwr))
    while len(self._lw) > 1 and f.t - self._lw[0][0] > G_LW_WIN_S:
      del self._lw[0]
    if len(self._lw) < 2:
      return False
    wl = [x for _t, x, _y in self._lw if _finite(x)]
    wr = [y for _t, _x, y in self._lw if _finite(y)]
    return bool((len(wl) > 1 and (max(wl) - min(wl)) > G_LW_BAR)
                or (len(wr) > 1 and (max(wr) - min(wr)) > G_LW_BAR))

  def _update_guards(self, f, cont):
    """guard_bits is this frame's raw truth; guard stays true for GUARD_REARM_S after the last."""
    bits = 0
    if abs((f.ylane10 - f.ylane0) - (f.ymod10 - f.ymod0)) > G_SHAPE_BAR:
      bits |= GUARD_SHAPE
    if (not _finite(f.w45) or abs(f.w45 - f.w0_g) > G_W45_TAPER
            or f.w45 < W_LO or f.w45 > W_HI):
      bits |= GUARD_W45
    if self._lane_width_guard(f, cont):
      bits |= GUARD_LW
    if (not _finite(f.k_lane)) or f.lane_status != 0 or bool(f.lane_nan):
      bits |= GUARD_BAD
    if not cont:
      self._guard_last_t = None
    raw = bits != 0
    if raw:
      self._guard_last_t = f.t
    guarded = raw or (self._guard_last_t is not None
                      and (f.t - self._guard_last_t) <= GUARD_REARM_S + 1e-9)
    self.guard_bits = bits
    self.guard = bool(guarded)

  # --- gate (report_hybrid5.py:223-287 without the alpha ramp) ------------------------------
  def _gate(self, f, dt, eligible, a_lat, a_road):
    self.drop = False
    if f.sp or f.lane_change or not f.lat:
      self._on, self._latch, self._ht = False, False, 0.0
      self.drop = True
      self.on = False
      return
    if not self._on:
      enter = bool(eligible and (a_lat >= GATE_ON_A
                                 or (a_lat >= GATE_REENTRY_A and abs(f.c_s) >= GATE_REENTRY_CS)))
      if enter and a_road >= GATE_ROAD_A:
        self._on, self._latch, self._ht = True, False, 0.0
    elif not eligible:
      self._on, self._latch = False, False
    elif a_lat > GATE_HOLD_A:
      self._latch = False
    elif not self._latch:
      if abs(f.c_s) >= HOLD_LATCH_CS:
        self._latch, self._ht = True, 0.0
      else:
        self._on = False
    else:
      self._ht += dt
      if abs(f.c_s) < HOLD_RELEASE_CS or self._ht >= HOLD_S:
        self._on, self._latch = False, False
    self.on = self._on

  # --- slew (report_hybrid5.py:315-362) ----------------------------------------------------
  def _slew(self, f, dt):
    a = self._a
    if a * (-f.c_s) < 0:
      a = 0.0
      self.zc = True
    g = self.tgt
    if abs(g) > abs(a):
      step = J_UP * dt
      if step >= abs(g) - abs(a):
        a = g
      else:
        a += (1.0 if g > 0 else -1.0) * step
    else:
      step = J_DOWN * dt
      if step >= abs(g - a):
        a = g
      else:
        a += (1.0 if g - a > 0 else -1.0) * step
    self._a = a
    self.a_add = a

  def update(self, f: HybridFrame) -> float:
    """Advance one frame and return a_add (m/s^2)."""
    t = float(f.t)
    dt = NAN if self._t_prev is None else t - self._t_prev
    self.cont = _finite(dt) and 0.0 < dt <= CONT_MAX_DT
    broke = not self.cont
    if broke:
      self._on, self._latch, self._ht = False, False, 0.0
      self._a = 0.0
      dt = 0.0
    elif not (0.0 < dt <= STEP_MAX_DT):
      dt = 0.0
    self._t_prev = t

    self.zc = False
    self._update_guards(f, self.cont)

    base = self._base(f)
    strict_inputs = self._strict_inputs(f)
    spike_ok = self._spike_ok(f)
    self.eligible = bool(base and strict_inputs and spike_ok and _finite(f.k_model))

    v2 = f.v * f.v
    cap = CAP_A / v2 if (_finite(f.v) and f.v > 0.0) else NAN
    dk = _clip(f.k_lane - f.k_model, -cap, cap)
    dk_rest = dk if dk * (-_sign(f.c_s)) > 0 else 0.0
    w_c = _clip((abs(f.c_s) - W_C_LO) / W_C_SPAN, 0.0, 1.0)

    a_lat = abs(f.dc) * v2
    a_road = abs(f.kappa) * v2
    curve_a = max(abs(f.dc), abs(f.kappa)) * v2

    self._gate(f, dt, self.eligible and not self.guard, curve_a, a_road)

    tgt = w_c * dk_rest * v2 if self.on else 0.0
    self.tgt = tgt if _finite(tgt) else 0.0

    if self.drop:
      self._a = 0.0
      self.a_add = 0.0
    else:
      self._slew(f, dt)

    self.curvature = self.a_add / v2 if (_finite(f.v) and f.v > 0.0) else 0.0
    return self.a_add
