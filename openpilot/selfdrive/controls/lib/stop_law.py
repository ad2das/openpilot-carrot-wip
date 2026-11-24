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
TH_LO = (-0.25, -0.40)
TH_HI = (0.40, 1.0)
U_MIN = -2.0
U_MAX = -0.15        # keep a minimum brake while the law owns the stop; a full release let idle creep push the car
DU_LIM = 2.5         # m/s^3, command rate limit
TAU_S = 0.12         # s, the measured-accel smoothing lag (fb, fu, fw and u_f match it)
P_MAX = 0.60         # covariance ceiling; conditional forgetting keeps trace(p00+p11) <= P_MAX
GUARD = True         # asymmetric RLS vetoes (deadband + fast glide release)
SLEW = True          # slew-limit the parameter step per tick
U_DB = 0.40          # m/s^2, |command| below which the deadband guard may veto
FU_REL = 0.12        # m/s^2, band above u_f the release-lag guard keys on
TH0_SLEW = 0.6       # 1/s, th0 slew limit (applied when SLEW)
TH1_SLEW = 1.2       # 1/s, th1 slew limit (applied when SLEW)
DEN_FLOOR = 0.05     # m
DEN_LOCK = 0.35      # m, den_l floor in the locked branch only
X_CORR = 0.1429370435473798   # m, travel after the standstill timer fires that the law leaves room for
W_V_HI = 2.0
W_V_LO = 0.45


def dw(v):
  x = (W_V_HI - v) / (W_V_HI - W_V_LO)
  return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class StopLaw:
  def __init__(self, a0, u_hist, tol_lock=0.0):
    """a0: measured accel at entry. u_hist: the last applied commands, oldest first, len >= ND + 1.
    tol_lock: metres the locked hold may run past the stop point before it brakes firmer than A_H."""
    uh = [float(x) for x in u_hist]
    if len(uh) < ND + 1:
      raise ValueError("u_hist needs at least ND + 1 samples")
    self.uh = uh
    self.tol_lock = float(tol_lock)
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
    self.fb_s = float(a0)
    self.fu_s = 0.0
    self.fw_s = 0.0
    self.u_f = G * uh[-1 - ND]
    self.v_prev = None

  def _adapt(self, a_m):
    if self.v_prev is None:
      return
    u_d = G * self.uh[-1 - ND]
    wk = dw(self.v_prev)
    self.fb += (u_d + C - self.fb) * BETA
    self.fu += (u_d - self.fu) * BETA
    self.fw += (wk - self.fw) * BETA
    self.u_f += (u_d - self.u_f) * BETA
    self.fb_s += (self.fb - self.fb_s) * (DT / TAU_S)
    self.fu_s += (self.fu - self.fu_s) * (DT / TAU_S)
    self.fw_s += (self.fw - self.fw_s) * (DT / TAU_S)
    p0, p1 = self.fu_s, self.fw_s
    t0 = self.p00 * p0 + self.p01 * p1
    t1 = self.p01 * p0 + self.p11 * p1
    den = LAM + p0 * t0 + p1 * t1
    if den != 0.0:
      k0 = t0 / den
      k1 = t1 / den
      res = (a_m - self.fb_s) - (p0 * self.th0 + p1 * self.th1)
      if GUARD:
        if abs(self.uh[-1 - ND]) < U_DB and (res < 0.0 or self.th1 >= 0.0):    # deadband: a weak brake fakes a -offset; a +res may only correct a negative th1
          return
        if u_d > self.u_f + FU_REL and (res > 0.0 or not self.locked):           # fast release: the measured decel leads the model, res is unreliable; locked only vetoes the false +res
          return
      th0 = self.th0 + k0 * res
      th1 = self.th1 + k1 * res
      q00 = self.p00 - k0 * t0
      q01 = self.p01 - k0 * t1
      q11 = self.p11 - k1 * t1
      lam = max(LAM, min(1.0, (q00 + q11) / P_MAX))                               # conditional forgetting: trace <= P_MAX
      self.p00 = q00 / lam
      self.p01 = q01 / lam
      self.p11 = q11 / lam
      if SLEW:
        s0 = TH0_SLEW * DT
        s1 = TH1_SLEW * DT
        e0 = th0 - self.th0
        e1 = th1 - self.th1
        th0 = self.th0 + (s0 if e0 > s0 else (-s0 if e0 < -s0 else e0))
        th1 = self.th1 + (s1 if e1 > s1 else (-s1 if e1 < -s1 else e1))
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
      if self.tol_lock > 0.0:
        den_l = d_stop + self.tol_lock - X_CORR
        if den_l < DEN_LOCK:
          den_l = DEN_LOCK
        a_kin = -v * v / (2.0 * den_l)
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
  VL_IN=0.5,           # m/s, the lead counts as stopping only once nearly at rest (< 1.8 km/h)
  VL_STILL=0.5,        # m/s, below this the lead counts as stopped regardless of closing speed
  DV_ENTER=0.6,        # m/s, when VL_IN > VL_STILL (caller override), require ego closing speed >= this
  VL_OUT=1.5,          # m/s, the law ends when the lead is faster
  VL_CREEP=0.75,       # m/s, creep exit: the lead is at least this fast ...
  DV_RISE=0.65,        # m/s, ... and this much faster than its slowest (>= 0) since the law started ...
  VL_CREEP_HYST=0.15,  # m/s, (the creep timer clears this far below that threshold and holds in between)
  T_CREEP=0.6,         # s, ... for this long ...
  DX_CREEP=0.8,        # m, ... and has moved this far past its nearest filtered position since that slowest point (lead_go)
  VL_REIN=0.5,         # m/s, after a lead_go the law starts again only once the lead is slower than this
  PG_V=0.25,           # m/s, once the command is held or the car is below this speed, ...
  PG_A=0.05,           # m/s^2, ... a plan asking for more than this ...
  PG_T=0.2,            # s, ... for this long ends the law (plan_go)
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
  TOL_LOCK=0.3,        # m, once locked the hold may run this far past the stop point before braking firmer than A_H
)
UH_KEEP = 64           # applied-command history kept for seeding the law (>= ND + 1)

GAP_OFFSET = 0.7      # m, law rest gap = StopDistanceCarrot - GAP_OFFSET; 550 cm keeps the field-tuned 4.8 m
SETTING_MIN_M = 4.0   # StopDistanceCarrot menu range 400-1000 cm (carrot_settings.json)
SETTING_MAX_M = 10.0


def stop_law_gap(stop_distance_m):
  """Law rest gap for StopDistanceCarrot in metres. Outside the menu range (unset Params read as 0, NaN, direct
  writes) -> CFG GAP, the old fixed value."""
  try:
    d = float(stop_distance_m)
  except (TypeError, ValueError):
    return CFG["GAP"]
  if not (math.isfinite(d) and SETTING_MIN_M <= d <= SETTING_MAX_M):
    return CFG["GAP"]
  return round(d - GAP_OFFSET, 3)


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
    self.vl_min = None       # slowest lead speed (>= 0) since the law started
    self.x_creep_ref = None  # nearest filtered lead position since that slowest point (odometry frame)
    self.t_creep = 0.0
    self.t_plan_go = 0.0
    self.creep_lock = False  # set by a lead_go drop: no new start until idle, lead loss, v > V_EXIT or v_lead < VL_REIN
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
    self.vl_min = None
    self.x_creep_ref = None
    self.t_creep = 0.0
    self.t_plan_go = 0.0
    self.creep_lock = why == "lead_go"

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
      if self.creep_lock and (v > c["V_EXIT"] or v_lead < c["VL_REIN"]):
        self.creep_lock = False
      lead_stopping = v_lead < c["VL_STILL"] or (v_lead < c["VL_IN"] and (v - v_lead) >= c["DV_ENTER"])
      if not (c["V_ENTER_MIN"] < v <= c["V_ENTER"] and lead_stopping and d_rel >= c["D_MIN"]
              and q >= c["Q_ENTER"] and (a_target < 0.0 or not c["NEED_PLAN_BRAKE"])
              and len(self.uh) >= ND + 1 and not self.creep_lock):
        return None
      self.law = StopLaw(a, self.uh[-UH_KEEP:], c["TOL_LOCK"])
      self.t_flag = 0.0
      self.frozen = False
      self.n_enter += 1
      self.vl_min = v_lead if v_lead > 0.0 else 0.0
      self.x_creep_ref = self.filt.x_f
      self.t_creep = 0.0
      self.t_plan_go = 0.0
    else:
      if v_lead >= c["VL_OUT"]:
        self._drop("lead_go")
        return None
      # A lead creeping away below VL_OUT must not hold the car at rest behind it. Only a rise above the lead's
      # slowest speed since the law started counts, so a lead that is still slowing to its own stop never does.
      vl_pos = v_lead if v_lead > 0.0 else 0.0
      if vl_pos < self.vl_min:
        self.vl_min = vl_pos
        self.x_creep_ref = self.filt.x_f
        self.t_creep = 0.0
      elif self.filt.x_f < self.x_creep_ref:
        self.x_creep_ref = self.filt.x_f
      vl_thr = self.vl_min + c["DV_RISE"]
      if vl_thr < c["VL_CREEP"]:
        vl_thr = c["VL_CREEP"]
      if v_lead >= vl_thr:
        self.t_creep += DT
      elif v_lead < vl_thr - c["VL_CREEP_HYST"]:
        self.t_creep = 0.0
      if (v_lead >= vl_thr and self.t_creep >= c["T_CREEP"] - 1e-9
          and self.filt.x_f - self.x_creep_ref >= c["DX_CREEP"]):
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
    # The plan wants to move off while the law holds the car (held command or nearly at rest): hand back.
    if (self.frozen or v < c["PG_V"]) and a_target > c["PG_A"]:
      self.t_plan_go += DT
    else:
      self.t_plan_go = 0.0
    if self.t_plan_go >= c["PG_T"] - 1e-9:
      self._drop("plan_go")
      return None
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
