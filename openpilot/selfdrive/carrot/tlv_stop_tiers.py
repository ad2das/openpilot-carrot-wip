"""W-TLT1: two tiers for the experimental CarrotTlvStopGo toggle.

Tier 1 (soft) applies a continuous physics-based deceleration cap (0 to -1.0
m/s^2) derived from the fused monocular range to the red light (width +
calibrated grade-compensated elevation) and current speed: far reds coast
without braking or burning the moving-timeout clock. Without the driving
model's own slowdown cue the cap is the range cap scaled into [-0.5, 0.0];
only the cue unlocks -1.0 m/s^2. A no-cue tier 1 that started at 80 km/h or
above releases after 4.5 s of braking. A new no-cue tier 1 then waits 10 s
(the cue bypasses that). When no range is available, tier 1
falls back to -0.5 m/s^2 until the model cue.
Tier 2 (stop) is the existing commit: it stops at the driving model's own stop
distance, allowed up to 100 km/h and, at 40 km/h and above, gated on the
model's own slowdown cue (rx = x_end / (10 * v0)).

Pure python by contract: no cereal, no params, no clock. Time is an input, so
the planner and an offline replay drive exactly the same state machine.
"""
from collections import deque
from dataclasses import dataclass
import math
import statistics

TIER_H_S = 2.5                 # red hold; 0000049c had a 2.249 s gap
SOFT_DECEL = 0.5               # m/s^2 fallback cap when range is unknown and no cue
SOFT_FIRM_DECEL = 1.0          # m/s^2 max tier-1 decel cap
SOFT_MIN_MPS = 30.0 / 3.6
SOFT_MAX_MPS = 100.0 / 3.6
SOFT_MAX_MOVING_S = 12.0       # = TLV_COAST_MAX_MOVING_S
SOFT_RANGE_WIN_S = 1.0         # rolling median window for fused range_m
SOFT_RANGE_HOLD_S = TIER_H_S   # odometry propagation horizon across brief dropouts
SOFT_REACT_S = 1.0             # reaction distance buffer in a_stop_req = v^2 / (2*(d - t*v))
SOFT_MIN_EFF_RANGE_M = 10.0
SOFT_A_DEAD = 0.85             # m/s^2: below this, red is far enough to coast without braking
SOFT_A_FULL = 1.35             # m/s^2: at or above this, full -1.0 m/s^2 cap
SOFT_TIMER_MIN_DECEL = 0.15    # m/s^2: moving timeout advances only while actually braking
STOP_MIN_MPS = 0.3
STOP_MAX_MPS = 100.0 / 3.6
STOP_MIN_DIST_M = 5.0
STOP_MAX_DIST_M = 220.0
STOP_DIST_PER_MPS = 10.5
STOP_MAX_Y_M = 5.0
STOP_AGE_MPS = 22.2            # = TLV_FAST_GATE_MPS
SOFT_FAST_MPS = STOP_AGE_MPS   # 80 km/h: a no-cue tier 1 started at or above this gets the short timeout
SOFT_FAST_NOCUE_S = 4.5        # s of braking a no-cue fast tier 1 may spend (real >= 80 km/h reds cued within 3.3 s)
SOFT_FAST_TIMEOUT_COOLDOWN_S = 10.0  # s after a fast no-cue timeout before a new no-cue tier 1 may start (~250 m at 90 km/h)
STOP_AGE_S = 1.0
VETO_MPS = 40.0 / 3.6
CUE_RX_MAX = 0.80
CUE_MIN_S = 0.5
CUE_BRAKE_HOLD_S = 2.0
LEAD_CLEAR_S = 0.3           # a stop commit needs the lead gone this long (no commit inside a lead dropout)
LEAD_REARM_S = 1.0           # after a lead release, a new stop commit waits this long
LEAD_SOLID_S = 0.5           # a lead releases either tier only once seen this long (gaps <= LEAD_CLEAR_S bridged)


@dataclass
class TierIn:
  """One planner tick, as the tiers need it."""
  t: float
  v_ego: float
  gas: bool
  brake: bool
  blinker: bool
  steer_ok: bool
  lead: bool
  raw_red: bool             # src-1 trafficState 1 or 4, fresh, NOT fast-gated
  conf: bool                # src-1 state 1, fresh
  go: bool                  # src-1 state 2 or 3, fresh
  gated_red: bool           # tlv_red_active after the fast gate
  model_stop: bool
  x_end: float              # position.x[-1]
  v0: float                 # velocity.x[0]
  y_end: float
  stop_x: float             # stop_model_x_rl
  comfort_brake: float
  cooldown: bool            # tlv_gas_cooldown_frames > 0
  starting: bool            # traffic_starting_count != 0
  enabled: bool             # toggle 1 and TrafficLightDetectMode >= 1 and not High
  range_m: float = 0.0      # fused monocular range to relevant red light (m); <=0 = unknown


class TlvStopTiers:
  """Tier 1 (range-aware soft decel) and tier 2 (model-agreed stop) as one state machine."""

  def __init__(self):
    self.soft_active = False
    self.committed = False
    self.events = []
    self.prev_t = None
    self.last_brake_t = None
    self.cue_t0 = None
    self.last_raw_t = None
    self.ep_t0 = None
    self.ep_v0 = None
    self.last_conf_t = None
    self.last_go_t = None
    self.last_gated_t = None
    self.last_lead_t = None
    self.last_lead_rel_t = None
    self.lead_run_t0 = None
    self.gas_ep = False
    self.soft_latched = False
    self.soft_stop_seen = False
    self.soft_moving_s = 0.0
    self.soft_firm = False
    self.soft_cap = None
    self.soft_range_m = None
    self._soft_cue_latched = False
    self.soft_v0 = None
    self.last_fast_timeout_t = None
    self._range_hist = deque(maxlen=64)
    self._last_range_t = None
    self._last_range_m = None

  def accel_cap(self):
    """Acceleration upper limit this tick (m/s^2 in [-1.0, 0.0]), or None when tier 1 is off."""
    if not self.soft_active:
      return None
    if self.soft_cap is not None:
      return self.soft_cap
    return -SOFT_FIRM_DECEL if self.soft_firm else -SOFT_DECEL

  def _update_range(self, t: float, range_m: float, v_ego: float):
    """1.0 s rolling median with odometry decay across brief detector dropouts."""
    if math.isfinite(range_m) and range_m > 0.0:
      self._range_hist.append((t, float(range_m)))
      cutoff = t - SOFT_RANGE_WIN_S
      while self._range_hist and self._range_hist[0][0] < cutoff:
        self._range_hist.popleft()
      self._last_range_m = float(statistics.median(v for _, v in self._range_hist))
      self._last_range_t = t
    if self._last_range_t is None:
      return None
    age = t - self._last_range_t
    if age < 0.0 or age > SOFT_RANGE_HOLD_S:
      return None
    return max(0.0, self._last_range_m - max(0.0, v_ego) * age)

  @staticmethod
  def _range_cap(v_ego: float, d_est: float) -> float:
    """Continuous deceleration cap in [-1.0, 0.0] from stopping physics."""
    d_eff = max(d_est - SOFT_REACT_S * max(0.0, v_ego), SOFT_MIN_EFF_RANGE_M)
    a_req = (max(0.0, v_ego) ** 2) / (2.0 * d_eff)
    if a_req <= SOFT_A_DEAD:
      return 0.0
    frac = min(1.0, (a_req - SOFT_A_DEAD) / (SOFT_A_FULL - SOFT_A_DEAD))
    return -SOFT_FIRM_DECEL * frac

  def update(self, i: TierIn) -> None:
    self.events = []
    dt = 0.0 if self.prev_t is None else min(max(i.t - self.prev_t, 0.0), 0.1)
    self.prev_t = i.t

    # model cue (every tick)
    if i.brake:
      self.last_brake_t = i.t
    valid = (not i.brake and not i.gas
             and (self.last_brake_t is None or i.t - self.last_brake_t > CUE_BRAKE_HOLD_S))
    rx = i.x_end / (10.0 * max(i.v0, 1.0))
    self.cue_t0 = (self.cue_t0 if self.cue_t0 is not None else i.t) if (valid and rx <= CUE_RX_MAX) else None
    cue = self.cue_t0 is not None and i.t - self.cue_t0 >= CUE_MIN_S

    # raw red episode (tier 2)
    if i.raw_red:
      if self.last_raw_t is None or i.t - self.last_raw_t > TIER_H_S:
        self.ep_t0, self.ep_v0 = i.t, i.v_ego
      self.last_raw_t = i.t
    if i.conf:
      self.last_conf_t = i.t
    if i.go:
      self.last_go_t = i.t

    # a green seen after the last red ends that red for any new commit
    # (no re-commit while the held red decays)
    red_newer = self.last_raw_t is not None and (self.last_go_t is None or self.last_raw_t > self.last_go_t)
    held_raw = self.last_raw_t is not None and i.t - self.last_raw_t <= TIER_H_S
    conf_recent = self.last_conf_t is not None and i.t - self.last_conf_t <= TIER_H_S

    # gated red episode (tier 1)
    if i.gated_red:
      self.last_gated_t = i.t
    gated_held = self.last_gated_t is not None and i.t - self.last_gated_t <= TIER_H_S
    if not gated_held:
      self.soft_latched = False
      self._range_hist.clear()
      self._last_range_t = None
      self._last_range_m = None
    d_est = self._update_range(i.t, i.range_m, i.v_ego)
    self.soft_range_m = d_est

    # driver gas while a confirmed red is recent overrides both tiers for the rest of that red episode
    if not (held_raw or gated_held):
      self.gas_ep = False
    elif i.gas and conf_recent:
      self.gas_ep = True
    if i.lead:
      if self.last_lead_t is None or i.t - self.last_lead_t > LEAD_CLEAR_S:
        self.lead_run_t0 = i.t
      self.last_lead_t = i.t
    # one- or two-tick radar blips (000004ad 20:14:56-15:01) must not drop a tier
    lead_solid = i.lead and self.lead_run_t0 is not None and i.t - self.lead_run_t0 >= LEAD_SOLID_S
    lead_clear = ((self.last_lead_t is None or i.t - self.last_lead_t > LEAD_CLEAR_S)
                  and (self.last_lead_rel_t is None or i.t - self.last_lead_rel_t > LEAD_REARM_S))

    # tier 1: without the model cue the range-scaled cap stays in [-0.5, 0.0]; only the cue unlocks -1.0
    if self.soft_active:
      if i.model_stop:
        self.soft_stop_seen = True
      cue_latched_now = not self._soft_cue_latched and cue
      if cue_latched_now:
        self._soft_cue_latched = True
        self.events.append(('soft', 'firm', 'cue'))
      if self._soft_cue_latched:
        self.soft_cap = -SOFT_FIRM_DECEL
      elif d_est is not None:
        self.soft_cap = (SOFT_DECEL / SOFT_FIRM_DECEL) * self._range_cap(i.v_ego, d_est)
      else:
        self.soft_cap = -SOFT_DECEL
      self.soft_firm = self._soft_cue_latched
      # Advance the moving timeout only while actually braking (never while waiting far out).
      if self.soft_stop_seen or cue_latched_now:
        self.soft_moving_s = 0.0
      elif self.soft_cap <= -SOFT_TIMER_MIN_DECEL:
        self.soft_moving_s += dt
      fast_timeout = (not self._soft_cue_latched and not self.soft_stop_seen and self.soft_v0 is not None
                      and self.soft_v0 >= SOFT_FAST_MPS and self.soft_moving_s >= SOFT_FAST_NOCUE_S - 1e-9)
      r = ('disabled' if not i.enabled else 'go' if i.go else 'red_end' if not gated_held else
           'gas' if (i.gas or i.cooldown) else 'lead' if lead_solid else
           'timeout' if self.soft_moving_s > SOFT_MAX_MOVING_S else
           'timeout' if fast_timeout else
           'slow' if i.v_ego < SOFT_MIN_MPS else None)
      if r:
        if r == 'timeout' and fast_timeout:
          self.last_fast_timeout_t = i.t
        self.soft_active = False
        self.soft_latched = True
        self.soft_firm = False
        self.soft_cap = None
        self._soft_cue_latched = False
        self.soft_v0 = None
        self.events.append(('soft', 'release', r))
    elif (i.enabled and not self.soft_latched and not self.gas_ep and i.conf and i.gated_red
          and SOFT_MIN_MPS <= i.v_ego < SOFT_MAX_MPS
          and not i.lead and not i.gas and not i.brake and not i.cooldown and not i.blinker
          and (cue or self.last_fast_timeout_t is None or i.t - self.last_fast_timeout_t >= SOFT_FAST_TIMEOUT_COOLDOWN_S)):
      self.soft_active = True
      self.soft_stop_seen = i.model_stop
      self.soft_moving_s = 0.0
      self._soft_cue_latched = cue
      self.soft_v0 = i.v_ego
      if cue:
        self.soft_cap = -SOFT_FIRM_DECEL
      elif d_est is not None:
        self.soft_cap = (SOFT_DECEL / SOFT_FIRM_DECEL) * self._range_cap(i.v_ego, d_est)
      else:
        self.soft_cap = -SOFT_DECEL
      self.soft_firm = self._soft_cue_latched
      self.events.append(('soft', 'start', 'confirmed_red'))

    # tier 2
    if self.committed:
      r = ('disabled' if not i.enabled else 'go' if i.go else 'red_end' if not held_raw else
           'gas' if (i.gas or i.cooldown) else 'lead' if lead_solid else None)
      if r:
        self.committed = False
        if r == 'lead':
          self.last_lead_rel_t = i.t
        self.events.append(('stop', 'release', r))
    elif i.enabled:
      lo = max(STOP_MIN_DIST_M, i.v_ego ** 2 / (2.0 * i.comfort_brake)) if i.comfort_brake > 0 else float('inf')
      hi = max(STOP_MAX_DIST_M, STOP_DIST_PER_MPS * i.v_ego)
      age_ok = i.v_ego < STOP_AGE_MPS or (self.ep_t0 is not None and i.t - self.ep_t0 >= STOP_AGE_S)
      veto_on = i.v_ego >= VETO_MPS or (self.ep_v0 is not None and self.ep_v0 >= VETO_MPS)
      if (held_raw and red_newer and not i.go and not self.gas_ep and lead_clear and conf_recent and age_ok
          and (cue or not veto_on)
          and STOP_MIN_MPS < i.v_ego < STOP_MAX_MPS and not i.gas and not i.cooldown and not i.lead
          and not i.blinker and i.steer_ok and not i.starting and abs(i.y_end) < STOP_MAX_Y_M
          and lo <= i.stop_x <= hi):
        self.committed = True
        self.events.append(('stop', 'commit', 'cue' if veto_on else 'slow_no_veto'))
