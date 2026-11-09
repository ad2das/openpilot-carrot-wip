"""W-TLT1: two tiers for the experimental CarrotTlvStopGo toggle.

Tier 1 (soft) caps the acceleration at SOFT_DECEL on a confirmed TLV red in the
30-100 km/h band, so a phantom detection costs a few km/h instead of a stop.
Tier 2 (stop) is the existing commit: it stops at the driving model's own stop
distance, now allowed up to 100 km/h and, at 40 km/h and above, gated on the
model's own slowdown cue (rx = x_end / (10 * v0)).

Pure python by contract: no cereal, no params, no clock. Time is an input, so
the planner and an offline replay drive exactly the same state machine.
"""
from dataclasses import dataclass

TIER_H_S = 2.5                 # red hold; 0000049c had a 2.249 s gap
SOFT_DECEL = 0.5               # m/s^2 accel upper limit while tier 1 is on
SOFT_MIN_MPS = 30.0 / 3.6
SOFT_MAX_MPS = 100.0 / 3.6
SOFT_MAX_MOVING_S = 12.0       # = TLV_COAST_MAX_MOVING_S
STOP_MIN_MPS = 0.3
STOP_MAX_MPS = 100.0 / 3.6
STOP_MIN_DIST_M = 5.0
STOP_MAX_DIST_M = 220.0
STOP_DIST_PER_MPS = 10.5
STOP_MAX_Y_M = 5.0
STOP_AGE_MPS = 22.2            # = TLV_FAST_GATE_MPS
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


class TlvStopTiers:
  """Tier 1 (soft decel) and tier 2 (model-agreed stop) as one state machine."""

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

  def accel_cap(self):
    """Acceleration upper limit this tick, or None when tier 1 is off."""
    return -SOFT_DECEL if self.soft_active else None

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

    # tier 1
    if self.soft_active:
      if i.model_stop:
        self.soft_stop_seen = True
      self.soft_moving_s = 0.0 if self.soft_stop_seen else self.soft_moving_s + dt
      r = ('disabled' if not i.enabled else 'go' if i.go else 'red_end' if not gated_held else
           'gas' if (i.gas or i.cooldown) else 'lead' if lead_solid else
           'timeout' if self.soft_moving_s > SOFT_MAX_MOVING_S else
           'slow' if i.v_ego < SOFT_MIN_MPS else None)
      if r:
        self.soft_active = False
        self.soft_latched = True
        self.events.append(('soft', 'release', r))
    elif (i.enabled and not self.soft_latched and not self.gas_ep and i.conf and i.gated_red
          and SOFT_MIN_MPS <= i.v_ego < SOFT_MAX_MPS
          and not i.lead and not i.gas and not i.cooldown and not i.blinker):
      self.soft_active = True
      self.soft_stop_seen = i.model_stop
      self.soft_moving_s = 0.0
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
