"""ButterStop stop law for classic-CAN Hyundai/Kia: one continuous decel to the stop point behind a stopped lead.

StopLaw is the decel law. From the speed, the measured accel and the metres left to the stop
point it steers the car to a steady A_H at the moment the ESC standstill timer fires: a soft
glide while the stop point is far (q < 1), then a latched hold that is never softer than the
kinematic need. The command is a model inverse of the fitted brake response (gain G, offset C,
lag TAU_B, delay ND) with a two-parameter RLS correction and a small feedback on (r - a).

StopCtl decides when StopLaw owns the command, from signals LongControl has: it tracks the lead
position (the same filter as butter_stop.LeadPosFilter), corrects the radar range lag, starts
the law behind a lead that is stopped or nearly stopped, holds the command once the standstill
timer has fired, and keeps a lead guard and a plan floor that only ever add braking.

Both classes are pure Python with no openpilot imports. They are a line-for-line copy of the
reference validated on the Palisade stop twin; test_stop_law.py replays its recorded fixtures.

Per 100 Hz tick, in this order:
  u = ctl.update(allow, v, a, lead_status, d_rel, v_lead, a_target, u_plan)   # None = the law is not in charge
      (or ctl.idle() on a tick where the caller is not in its pid state)
  ... the caller applies u (or its own command when u is None) and clips it ...
  ctl.commit(u_applied)                                                        # every tick, law or not
"""
import math

DT = 0.01             # s, one controls tick (test_stop_law.py checks it equals DT_CTRL)
A_H = -0.70          # hand-off decel the law steers to (m/s^2)
M_EARLY = 2.0        # glide exponent while q < 1
M_LATE = 1.5         # glide exponent while q >= 1
V_LOCK = 2.0         # m/s, the lock may latch at or below this speed
Q_LATCH = 0.85       # ... once q has reached this
V_FLOOR = 1.0        # m/s, the lock latches here whatever q is
A_MIN = 0.15         # softest glide decel (m/s^2)
J_EASE = 2.0         # m/s^3, reference easing rate
J_FIRM = 2.0         # m/s^3, reference firming rate
KP = 0.3             # feedback on (r - a)
LAM = 0.98           # RLS forgetting
G = 0.9335664129807782      # brake response gain (fit on 104 Palisade stops)
C = 0.0664617646816226      # brake response offset (m/s^2)
TAU_B = 0.5713120493245386  # brake response lag (s)
ND = 20              # brake response delay in steps (0.20 s)
BETA = DT / TAU_B
P0 = (0.04, 0.09)
TH_LO = (-0.5, -1.0)
TH_HI = (0.8, 1.0)
U_MIN = -2.0
U_MAX = 0.0
DU_LIM = 2.5         # m/s^3, command rate limit
DEN_FLOOR = 0.05     # m
X_CORR = 0.1429370435473798   # m, travel after the standstill timer fires that the law leaves room for
W_V_HI = 2.0
W_V_LO = 0.45


def dw(v):
  x = (W_V_HI - v) / (W_V_HI - W_V_LO)
  return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class StopLaw:
  def __init__(self, a0, u_hist):
    """a0: measured accel at entry. u_hist: the last applied commands, oldest first, len >= ND + 1."""
    uh = [float(x) for x in u_hist]
    if len(uh) < ND + 1:
      raise ValueError("u_hist needs at least ND + 1 samples")
    self.uh = uh
    self.u_prev = uh[-1]
    self.r_prev = float(a0)
    self.locked = False
    self.q = float("nan")
    self.r = float("nan")
    self.fb = float(a0)
    self.fu = 0.0
    self.fw = 0.0
    self.th0 = 0.0
    self.th1 = 0.0
    self.p00 = P0[0]
    self.p01 = 0.0
    self.p11 = P0[1]
    self.v_prev = None

  def _adapt(self, a_m):
    if self.v_prev is None:
      return
    u_d = G * self.uh[-1 - ND]
    wk = dw(self.v_prev)
    self.fb += (u_d + C - self.fb) * BETA
    self.fu += (u_d - self.fu) * BETA
    self.fw += (wk - self.fw) * BETA
    p0, p1 = self.fu, self.fw
    t0 = self.p00 * p0 + self.p01 * p1
    t1 = self.p01 * p0 + self.p11 * p1
    den = LAM + p0 * t0 + p1 * t1
    if den != 0.0:
      k0 = t0 / den
      k1 = t1 / den
      res = (a_m - self.fb) - (p0 * self.th0 + p1 * self.th1)
      th0 = self.th0 + k0 * res
      th1 = self.th1 + k1 * res
      self.p00 = (self.p00 - k0 * t0) / LAM
      self.p01 = (self.p01 - k0 * t1) / LAM
      self.p11 = (self.p11 - k1 * t1) / LAM
      self.th0 = min(max(th0, TH_LO[0]), TH_HI[0])
      self.th1 = min(max(th1, TH_LO[1]), TH_HI[1])

  def _ref(self, v, d_stop):
    den = d_stop - X_CORR
    if den < DEN_FLOOR:
      den = DEN_FLOOR
    a_kin = -v * v / (2.0 * den)
    q = a_kin / A_H
    self.q = q
    if not self.locked and (v > V_LOCK or (q < Q_LATCH and v > V_FLOOR)):
      r_t = A_H * (q ** (M_EARLY if q < 1.0 else M_LATE))
      if r_t < U_MIN:
        r_t = U_MIN
      elif r_t > -A_MIN:
        r_t = -A_MIN
    else:
      self.locked = True
      r_t = a_kin if a_kin < A_H else A_H      # never softer than a_kin once a_kin is firmer than A_H
    dr = r_t - self.r_prev
    if dr < 0.0:
      lim = -J_FIRM * DT
      r = self.r_prev + (dr if dr > lim else lim)
    else:
      lim = J_EASE * DT
      r = self.r_prev + (dr if dr < lim else lim)
    self.r_prev = r
    return r

  def step(self, v, a_m, d_stop):
    """One 100 Hz step. v: speed, a_m: measured accel (aEgo), d_stop: metres left to the stop point."""
    self._adapt(a_m)
    self.v_prev = v
    r = self._ref(v, d_stop)
    v_pred = v + a_m * ND * DT
    if v_pred < 0.0:
      v_pred = 0.0
    u_ff = (r - C - self.th1 * dw(v_pred)) / (G * (1.0 + self.th0))
    u_cmd = u_ff + KP * (r - a_m)
    if u_cmd < U_MIN:
      u_cmd = U_MIN
    elif u_cmd > U_MAX:
      u_cmd = U_MAX
    lim = DU_LIM * DT
    du = u_cmd - self.u_prev
    u = self.u_prev + (lim if du > lim else (-lim if du < -lim else du))
    self.u_prev = u
    self.r = r
    self.uh.append(u)
    if len(self.uh) > 256:
      del self.uh[:-64]
    return u

  def commit(self, u_applied):
    """Tell the law which command was actually applied this tick (after guards and clips)."""
    u = float(u_applied)
    self.uh[-1] = u
    self.u_prev = u


CFG = dict(
  GAP=4.8,             # m, final radar gap the law aims at
  T_RADAR=0.135,       # s, radar range lag against wheel odometry
  V_ENTER=3.5,         # m/s, the law may start at or below this speed
  V_ENTER_MIN=1.0,     # m/s, and only above this one
  V_EXIT=4.0,          # m/s, the law ends when the car is faster again
  Q_ENTER=0.4,         # needed decel / |A_H| at which the law may start
  VL_IN=1.0,           # m/s, the lead counts as stopping below this
  VL_OUT=1.5,          # m/s, the law ends when the lead is faster
  D_MIN=3.0,           # m, hard minimum gap: no start below it, and the guard stops the car short of it
  NEED_PLAN_BRAKE=True,  # start only while the plan itself brakes (a_target < 0)
  Q_PLAN=float("inf"),   # above this q the command is never softer than u_plan
  V_FLAG=0.794,        # m/s, the ESC standstill timer starts below this
  T_FLAG=0.509,        # s, ... and fires after this long; the command is held from then on
  V_FLAG_RESET=1.0,    # m/s, the timer clears above this
  TAU_LEAD=0.3,        # s, lead position filter (butter_stop.TAU_LEAD)
  LEAD_JUMP_M=1.5,     # m (butter_stop.LEAD_JUMP_M)
  LEAD_RAW_ALLOW=0.5,  # m (butter_stop.LEAD_RAW_ALLOW)
  JUMP_VERIFY_S=0.5,   # s (butter_stop.LeadPosFilter.JUMP_VERIFY_S)
)
UH_KEEP = 64           # applied-command history kept for seeding the law (>= ND + 1)


class LeadPos:
  """butter_stop.LeadPosFilter, verbatim logic: smoothed lead position in the ego-odometry frame."""

  def __init__(self, tau, jump_m, raw_allow, verify_s):
    self.tau, self.jump_m, self.raw_allow, self.verify_s = tau, jump_m, raw_allow, verify_s
    self.x_f = None
    self.far_t = 0.0

  def reset(self):
    self.x_f = None
    self.far_t = 0.0

  def update(self, d_rel, s_ego):
    x_raw = d_rel + s_ego
    if self.x_f is None:
      self.x_f = x_raw
      self.far_t = 0.0
    elif x_raw < self.x_f - self.jump_m:
      self.x_f = x_raw
      self.far_t = 0.0
    elif x_raw > self.x_f + self.jump_m:
      self.far_t += DT
      if self.far_t + 1e-9 >= self.verify_s:
        self.x_f = x_raw
        self.far_t = 0.0
    else:
      self.far_t = 0.0
      self.x_f += (DT / self.tau) * (x_raw - self.x_f)
    return min(self.x_f - s_ego, d_rel + self.raw_allow)


class StopCtl:
  def __init__(self, **over):
    for k in over:
      if k not in CFG:
        raise KeyError(k)
    self.c = dict(CFG)
    self.c.update(over)
    c = self.c
    self.filt = LeadPos(c["TAU_LEAD"], c["LEAD_JUMP_M"], c["LEAD_RAW_ALLOW"], c["JUMP_VERIFY_S"])
    self.uh = []            # applied commands, oldest first
    self.s_ego = 0.0
    self.law = None
    self.t_flag = 0.0
    self.frozen = False
    self.u_out = 0.0
    # telemetry
    self.d_eff = None
    self.d_stop = float("nan")
    self.q = float("nan")
    self.why = ""           # reason of the last drop
    self.n_enter = 0
    self.n_guard = 0        # ticks the lead guard bound
    self.n_plan = 0         # ticks the plan floor bound

  def _drop(self, why):
    if self.law is not None:
      self.why = why
    self.law = None
    self.t_flag = 0.0
    self.frozen = False

  def idle(self):
    """A tick on which the law cannot run (not in pid, pedal, fcw, stale plan): drop it and forget the lead."""
    self._drop("allow")
    self.filt.reset()
    self.s_ego = 0.0
    self.d_eff = None

  def update(self, allow, v, a, lead_status, d_rel, v_lead, a_target, u_plan=float("inf")):
    c = self.c
    fin = math.isfinite
    if not (allow and fin(v) and fin(a)):
      self.idle()
      return None
    self.s_ego += v * DT
    if not (lead_status and d_rel is not None and fin(d_rel) and fin(v_lead)):
      self._drop("lead")
      self.filt.reset()
      self.d_eff = None
      return None
    d_eff = self.filt.update(d_rel, self.s_ego)
    self.d_eff = d_eff
    d_stop = d_eff - v * c["T_RADAR"] - c["GAP"]
    den = d_stop - X_CORR
    if den < DEN_FLOOR:
      den = DEN_FLOOR
    q = (v * v / (2.0 * den)) / (-A_H)
    self.d_stop = d_stop
    self.q = q
    if self.law is None:
      if not (c["V_ENTER_MIN"] < v <= c["V_ENTER"] and v_lead < c["VL_IN"] and d_rel >= c["D_MIN"]
              and q >= c["Q_ENTER"] and (a_target < 0.0 or not c["NEED_PLAN_BRAKE"])
              and len(self.uh) >= ND + 1):
        return None
      self.law = StopLaw(a, self.uh[-UH_KEEP:])
      self.t_flag = 0.0
      self.frozen = False
      self.n_enter += 1
    else:
      if v_lead >= c["VL_OUT"]:
        self._drop("lead_go")
        return None
      if v > c["V_EXIT"]:
        self._drop("fast")
        return None
    if v < c["V_FLAG"]:
      self.t_flag += DT
    elif v > c["V_FLAG_RESET"]:
      self.t_flag = 0.0
      self.frozen = False
    if self.t_flag >= c["T_FLAG"] - 1e-9:
      self.frozen = True
    u = self.u_out if self.frozen else self.law.step(v, a, d_stop)
    # Lead guard: only when stopping short of D_MIN needs more than |A_H|. It never eases.
    room = (d_eff if d_eff < d_rel else d_rel) - c["D_MIN"]
    g = -v * v / (2.0 * (room if room > 0.05 else 0.05))
    if g < A_H and g < u:
      u = g
      self.n_guard += 1
    # Plan floor: in a late approach (q above Q_PLAN) the command is never softer than the ordinary path.
    if q > c["Q_PLAN"] and u_plan < u:
      u = u_plan
      self.n_plan += 1
    self.u_out = u
    return u

  def commit(self, u_applied):
    u = float(u_applied)
    self.uh.append(u)
    if len(self.uh) > 4 * UH_KEEP:
      del self.uh[:-UH_KEEP]
    if self.law is not None:
      self.law.commit(u)
      self.u_out = u
