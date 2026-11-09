"""W-TSG1/W-TLT1: experimental CarrotTlvStopGo drives the two TLV tiers.

The toggle decides WHETHER a confirmed red may act; the state machine itself
lives in the pure `tlv_stop_tiers` module and is unit-tested there. What this
file pins is the wiring: the tier-1 soft cap reaches `get_carrot_accel`, the
tier-2 commit reaches the existing e2eStop path at the model's own distance,
and the release edges (gas, lead, red_end, standstill, green) behave.

At the default 0 the planner must behave exactly as before (the first case pins
that), so every case here is either a commit gate or a release edge of
`_update_tlv_stop_go`.

The state machine runs on time, so the replay namespace's `time` and `cloudlog`
are replaced with a controllable clock and an event recorder: every `step()`
advances the clock by DT_MDL. That keeps the 0.5 s model cue, the 2.5 s red hold
and the 12 s soft timeout exact instead of wall-clock dependent.

Full-update cases run the real `update()` through `planner_class`, which stubs
`_params_update`; the toggle, the detect mode and the carrotMan readers are
therefore set on the instance. `inputs()` has no carrotMan, so the TLV state is
injected through `_read_tlv_red_active` / `_read_tlv_confirmed_red` /
`_read_tlv_go`.

Ticks are counted, never compared for exact float equality at a boundary: a
"not yet" assertion stops a full tick short of the boundary and the "done"
assertion runs a couple of ticks past it.
"""
import ast
import time

from test_tlv_fusion import carrot_man_message, gate_sm, planner_for_gate

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.carrot.tlv_green_override import GreenHold, override_ok
from tools.signal_analysis.replay_signal_assist import REPO, planner_class
from tools.signal_analysis.test_signal_assist_planner import inputs

# The replay planner is exec'd in its own namespace, so its enums are distinct
# classes from the module-level ones and must come back out of that namespace.
# `planner_class` imports only t_follow helpers, so the two names CarrotPlanner
# resolves from module globals but lives in tlv_green_override must be added
# before the real __init__ runs.
_CACHE = {}
_T0 = 1000.0


class _Clock:
  """Deterministic stand-in for the `time` module inside the replay namespace."""

  def __init__(self):
    self.n = 0

  def reset(self):
    self.n = 0

  def advance(self, dt=DT_MDL):
    self.n += 1

  def monotonic(self):
    return _T0 + self.n * DT_MDL

  def time(self):
    return self.monotonic()


class _EventLog:
  """Captures cloudlog.event calls so the tier-1 events are observable."""

  def __init__(self):
    self.events = []

  def clear(self):
    self.events = []

  def event(self, name, **kwargs):
    self.events.append((name, kwargs))

  def named(self, name):
    return [kwargs for event_name, kwargs in self.events if event_name == name]


# `planner_class` re-execs only the TLV_* constants, so the plain module-level
# names CarrotPlanner reads survive as nothing at all. `get_carrot_accel` reads
# A_CRUISE_MAX_BP_CARROT, which the pre-W-TLT1 tests never called and these do.
_NEEDED_GLOBALS = ('A_CRUISE_MAX_BP_CARROT',)


def _inject_globals(ns, source):
  """Bind the named module-level assignments of the real file into the replay ns."""
  tree = ast.parse(source)
  for node in tree.body:
    if not isinstance(node, ast.Assign):
      continue
    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
    if any(name in _NEEDED_GLOBALS for name in names):
      exec(compile(ast.Module([node], type_ignores=[]), 'carrot_source_replay', 'exec'), ns)


def replay():
  if not _CACHE:
    clock = _Clock()
    log = _EventLog()
    source = (REPO / 'openpilot/selfdrive/carrot/carrot_functions.py').read_text(encoding='utf-8')
    cls, ns = planner_class(source)
    ns['GreenHold'] = GreenHold
    ns['override_ok'] = override_ok
    ns['time'] = clock
    ns['cloudlog'] = log
    _inject_globals(ns, source)
    _CACHE.update(cls=cls, clock=clock, log=log,
                  XState=ns['XState'], TrafficState=ns['TrafficState'],
                  DrivingMode=ns['DrivingMode'], cooldown=ns['TLV_GAS_COOLDOWN_FRAMES'])
  return _CACHE


def inputs_at(speed, stop_x, lead=False):
  """`inputs()` whose model x[31] is exactly `stop_x` (it ramps distance*31/32)."""
  return inputs(speed=speed, distance=stop_x * 32. / 31., lead=lead)


def set_rx(sm, rx):
  """Pin the model cue ratio x[-1] / (10 * v[0]) without moving x[31]."""
  v0 = float(sm['modelV2'].velocity.x[0])
  x = list(sm['modelV2'].position.x)
  x[-1] = rx * 10.0 * v0
  sm['modelV2'].position.x = x


def planner(mode=2, toggle=1):
  c = replay()
  c['clock'].reset()
  c['log'].clear()
  p = c['cls']()
  p.tlvStopGo = toggle
  p.trafficLightDetectMode = mode
  p.myDrivingMode = c['DrivingMode'].Normal
  calls = []
  p._log_tlv_stop_go = lambda action, reason, v_ego, stop_x: calls.append((action, reason))
  return p, calls


def step(p, sm, red_active=True, confirmed=True, go=False):
  p._read_tlv_red_active = lambda _sm, _v=red_active: _v
  p._read_tlv_confirmed_red = lambda _sm, _v=confirmed: _v
  p._read_tlv_go = lambda _sm, _v=go: _v
  replay()['clock'].advance()
  p.update(sm, 80., 'acc')


def steps(p, sm, count, **kwargs):
  for _ in range(count):
    step(p, sm, **kwargs)


# Tick budgets. CUE needs CUE_MIN_S = 0.5 s of rx <= 0.80, the red hold is
# TIER_H_S = 2.5 s, the fast-speed episode age is 1.0 s: 20 ticks/s.
CUE_SAFE = 13      # 0.65 s: past the 0.5 s cue
CUE_SHORT = 9      # 0.45 s: still inside the cue
AGE_SAFE = 24      # 1.20 s: past the 1.0 s episode age
AGE_SHORT = 18     # 0.90 s: still inside it
HOLD_SHORT = 49    # 2.45 s: still inside the 2.5 s red hold
HOLD_SAFE = 52     # 2.60 s: past it


# ---------------------------------------------------------------------------
# Default 0 must not touch anything
# ---------------------------------------------------------------------------

def test_toggle_off_never_commits_and_leaves_the_model_alone():
  XState, TrafficState = replay()['XState'], replay()['TrafficState']
  p, calls = planner(2, toggle=0)
  reference, _ = planner(2, toggle=0)
  assert p.tlvStopGo == 0
  steps(p, inputs_at(13.89, 100.), 3)  # 50 km/h, cue satisfied by geometry
  assert p.tlv_stop_committed is False
  assert p.tlv_soft_active is False
  assert p.xState == XState.e2eCruise
  assert p.trafficState != TrafficState.red  # the model's own cue stays in charge
  assert calls == []
  assert replay()['log'].named('tlvSoftDecel') == []
  # The accel limit is exactly the pre-W-TLT1 one: the existing coast cap (the
  # red is still read, the toggle only decides what may act on it) and nothing
  # else. With the cap cleared the untouched cruise formula is back.
  assert p.coast_cap_active is True
  assert p.get_carrot_accel(20.0) == 0.0
  p.coast_cap_active = False
  assert p.get_carrot_accel(20.0) == reference.get_carrot_accel(20.0)
  assert p.get_carrot_accel(20.0) > 0.0


def test_confirmed_red_commits_the_existing_stop_at_the_model_distance():
  XState, TrafficState = replay()['XState'], replay()['TrafficState']
  p, calls = planner(2, toggle=1)
  sm = inputs_at(13.89, 100.)  # min_dist = max(5, 13.89^2/4.8) = 40.2 m
  # 50 km/h is inside the model-veto band (>= 40 km/h), so the commit waits for
  # the model's own slowdown cue, here rx = 100 / (10 * 13.89) = 0.72.
  steps(p, sm, CUE_SHORT)
  assert p.tlv_stop_committed is False
  assert p.tlv_soft_active is True  # tier 1 started on the first tick
  steps(p, sm, 4)
  assert p.tlv_stop_committed is True
  assert p.trafficState == TrafficState.red
  assert p.xState == XState.e2eStop
  assert calls == [('commit', 'cue')]
  # Next frame: the stop branch is now active and applies the existing
  # adjustment (trafficStopAdjustRatio 0.85 at 50 km/h, so 100 m -> 85 m, minus
  # one frame of travel). The lamp contributed nothing to that number.
  step(p, sm)
  assert 83. < p.actual_stop_distance <= 85.1
  assert p.tlv_stop_committed is True
  assert len(calls) == 1


def test_graded_red_likely_alone_never_commits():
  XState = replay()['XState']
  p, calls = planner(2, toggle=1)
  step(p, inputs_at(13.89, 100.), red_active=True, confirmed=False)  # state 4 only
  assert p.tlv_stop_committed is False
  assert p.tlv_soft_active is False
  assert p.xState != XState.e2eStop
  assert calls == []


def test_mode_zero_and_high_mode_never_commit():
  p, calls = planner(0, toggle=1)
  steps(p, inputs_at(13.89, 100.), 3)
  assert p.tlv_stop_committed is False and calls == []
  assert p.tlv_soft_active is False

  p, calls = planner(2, toggle=1)
  p.myDrivingMode = replay()['DrivingMode'].High
  steps(p, inputs_at(13.89, 100.), 3)
  assert p.tlv_stop_committed is False and calls == []
  assert p.tlv_soft_active is False


# ---------------------------------------------------------------------------
# Commit gates
# ---------------------------------------------------------------------------

def test_commit_is_vetoed_by_lead_blinkers_steering_and_y():
  cases = {
    'lead': lambda sm: None,
    'left_blinker': lambda sm: setattr(sm['carState'], 'leftBlinker', True),
    'right_blinker': lambda sm: setattr(sm['carState'], 'rightBlinker', True),
    'steering': lambda sm: setattr(sm['carState'], 'steeringAngleDeg', 60.),
  }
  for name, mutate in cases.items():
    p, calls = planner(2, toggle=1)
    sm = inputs_at(10., 100., lead=name == 'lead')  # 36 km/h: below the veto band
    mutate(sm)
    steps(p, sm, 3)
    assert p.tlv_stop_committed is False, name
    assert calls == [], name

  p, calls = planner(2, toggle=1)
  sm = inputs_at(10., 100.)
  sm['modelV2'].position.y = [0.] * 32 + [6.]
  steps(p, sm, 3)
  assert p.tlv_stop_committed is False
  assert calls == []


def test_commit_is_vetoed_by_the_speed_and_distance_bounds():
  # 100.5 km/h = 27.92 m/s > STOP_MAX_MPS = 100/3.6 = 27.78 m/s. The cue holds
  # (rx 0.72) and the window is open at that speed (lo = 27.92^2/4.8 = 162.4,
  # hi = max(220, 293.2) = 293.2), so only the speed bound can veto; detector
  # support keeps the TLV red alive through the W139c fast gate.
  p, calls = planner(2, toggle=1)
  sm = inputs_at(100.5 / 3.6, 200.)
  sm['carrotMan'] = carrot_man_message(state=1, src=1, det_support=2)
  steps(p, sm, AGE_SAFE)
  assert p.tlv_stop_committed is False
  assert calls == []

  for stop_x in (30., 230.):  # below v^2/(2*2.4) at 50 km/h, and above 220 m
    p, calls = planner(2, toggle=1)
    sm = inputs_at(13.89, stop_x)
    steps(p, sm, CUE_SAFE)
    assert p.tlv_stop_committed is False, stop_x
    assert calls == [], stop_x


def test_above_eighty_two_kph_commits_with_the_model_cue():
  # 81.9 km/h used to sit on the 82 km/h bound. The tier-2 bound is now 100
  # km/h; at this speed the free-cruise plan is not slowing (rx ~0.97), so the
  # commit waits for the model's own cue instead of a speed bound.
  p, calls = planner(2, toggle=1)
  sm = inputs_at(22.75, 213.5)
  sm['carrotMan'] = carrot_man_message(state=1, src=1, det_support=2)
  steps(p, sm, AGE_SAFE)
  assert p.tlv_stop_committed is False
  assert calls == []

  # Same speed with the cue satisfied (rx 0.75) and stop_x 200 inside the
  # window (lo = 22.75^2/4.8 = 107.8, hi = max(220, 238.9)); the 1 s episode
  # age is what delays the commit.
  p, calls = planner(2, toggle=1)
  sm = inputs_at(22.75, 200.)
  set_rx(sm, 0.75)
  sm['carrotMan'] = carrot_man_message(state=1, src=1, det_support=2)
  steps(p, sm, AGE_SHORT)
  assert p.tlv_stop_committed is False
  steps(p, sm, AGE_SAFE - AGE_SHORT)
  assert p.tlv_stop_committed is True
  assert calls == [('commit', 'cue')]


def test_distance_window_tracks_the_comfort_brake():
  # 50 km/h, comfort brake 2.4: the floor is 40.19 m, so 45 m commits and 40 m
  # must not; the lamp never creates a distance, only this plausibility window.
  for stop_x, expected in ((100., True), (45., True), (40., False)):
    p, _ = planner(2, toggle=1)
    sm = inputs_at(13.89, stop_x)
    steps(p, sm, CUE_SAFE)
    assert p.tlv_stop_committed is expected, stop_x


# ---------------------------------------------------------------------------
# Release edges
# ---------------------------------------------------------------------------

def test_red_end_holds_for_two_and_a_half_seconds_then_releases():
  XState = replay()['XState']
  p, calls = planner(2, toggle=1)
  sm = inputs_at(10., 100.)  # 36 km/h: below the veto band, so no cue needed
  step(p, sm)
  assert p.tlv_stop_committed is True and p.xState == XState.e2eStop
  assert calls == [('commit', 'slow_no_veto')]
  # 0000049c dropped the TLV red for 2.249 s inside one real red, so a commit
  # survives any red gap up to TIER_H_S = 2.5 s.
  steps(p, sm, HOLD_SHORT, red_active=False, confirmed=False)
  assert p.tlv_stop_committed is True
  steps(p, sm, HOLD_SAFE - HOLD_SHORT, red_active=False, confirmed=False)
  assert p.tlv_stop_committed is False
  assert p.xState == XState.e2eCruise
  assert calls[-1] == ('release', 'red_end')


def test_gas_releases_and_holds_off_for_the_rest_of_that_red():
  p, calls = planner(2, toggle=1)
  cooldown = replay()['cooldown']
  sm = inputs_at(10., 100.)
  step(p, sm)
  assert p.tlv_stop_committed is True
  assert p.tlv_soft_active is True

  sm['carState'].gasPressed = True
  step(p, sm)
  assert p.tlv_stop_committed is False
  assert p.tlv_soft_active is False
  assert calls[-1] == ('release', 'gas')
  assert p.tlv_gas_cooldown_frames == cooldown

  sm['carState'].gasPressed = False
  # The driver overrode this red: no new commit for the rest of it, even
  # after the 10 s gas cooldown has run out (00000497 07:54:09.854).
  steps(p, sm, cooldown + 20)
  assert p.tlv_gas_cooldown_frames == 0
  assert p.tlv_stop_committed is False
  assert p.tlv_soft_active is False
  assert calls[-1] == ('release', 'gas')

  # Once the red has been gone longer than the 2.5 s hold, a new red commits again.
  steps(p, sm, HOLD_SAFE, red_active=False, confirmed=False)
  assert p.tlv_stop_committed is False
  step(p, sm)
  assert p.tlv_stop_committed is True
  assert p.tlv_soft_active is True
  assert calls[-1] == ('commit', 'slow_no_veto')


def test_lead_appearance_releases():
  # A lead releases the commit only once seen LEAD_SOLID_S = 0.5 s (tlv_stop_tiers v5):
  # one- or two-tick radar blips (000004ad) keep the stop.
  p, calls = planner(2, toggle=1)
  step(p, inputs_at(10., 100.))
  assert p.tlv_stop_committed is True
  lead = inputs_at(10., 100., lead=True)
  steps(p, lead, 9)
  assert p.tlv_stop_committed is True
  assert ('release', 'lead') not in calls
  steps(p, lead, 3)
  assert p.tlv_stop_committed is False
  assert calls[-1] == ('release', 'lead')


def test_standstill_hold_survives_the_model_and_green_follows_the_mode():
  XState, TrafficState = replay()['XState'], replay()['TrafficState']
  for mode, starts in ((2, True), (1, False)):
    p, calls = planner(mode, toggle=1)
    sm = inputs_at(10., 100.)
    step(p, sm)
    assert p.tlv_stop_committed is True

    # Standstill with the model's own stop cue off: nothing else holds the car,
    # so the committed red must keep trafficState red by itself.
    sm['carState'].vEgo = 0.2
    sm['carState'].vEgoCluster = 0.2
    sm['modelV2'].velocity.x = [0.2] + [15.] * 32
    for _ in range(15):
      step(p, sm)
    assert p.xState == XState.e2eStopped, mode
    assert p.trafficState == TrafficState.red, mode
    assert p.tlv_stop_committed is True, mode

    # TLV green ends the commit after the 2.5 s red hold, then the model's own
    # start cue needs a few frames to reach trafficState green; after that the
    # existing departure rule decides, per the detect mode.
    steps(p, sm, HOLD_SAFE, red_active=False, confirmed=False)
    assert p.tlv_stop_committed is False, mode
    assert calls[-1] == ('release', 'red_end'), mode
    for _ in range(10):
      step(p, sm, red_active=False, confirmed=False)
    assert (p.xState == XState.e2eCruise) is starts, mode


# ---------------------------------------------------------------------------
# Tier 1 (soft decel) reaches the accel limit and releases
# ---------------------------------------------------------------------------

def test_soft_starts_on_a_confirmed_red_and_caps_the_accel():
  p, calls = planner(2, toggle=1)
  sm = inputs_at(20., 220.)  # 72 km/h
  set_rx(sm, 1.0)            # the model is not slowing: no commit cue
  step(p, sm)
  assert p.tlv_soft_active is True
  assert p.get_carrot_accel(20.0) <= -0.5
  steps(p, sm, AGE_SAFE)
  assert p.tlv_stop_committed is False  # rx 1.0 never satisfies the cue
  assert calls == []
  soft = replay()['log'].named('tlvSoftDecel')
  assert soft and soft[0]['action'] == 'start' and soft[0]['reason'] == 'confirmed_red'
  assert all(entry['rx'] == 1.0 for entry in soft)


def test_soft_releases_on_gas_and_arms_the_cooldown():
  p, _ = planner(2, toggle=1)
  sm = inputs_at(20., 220.)
  set_rx(sm, 1.0)
  step(p, sm)
  assert p.tlv_soft_active is True
  sm['carState'].gasPressed = True
  step(p, sm)
  assert p.tlv_soft_active is False
  assert p.tlv_gas_cooldown_frames == replay()['cooldown']
  released = [e for e in replay()['log'].named('tlvSoftDecel') if e['action'] == 'release']
  assert released and released[-1]['reason'] == 'gas'


# ---------------------------------------------------------------------------
# The existing TLV green start is not touched (W143 path, real module)
# ---------------------------------------------------------------------------

def test_existing_tlv_green_start_still_leaves_e2e_stopped_in_mode_two():
  p = planner_for_gate(stop_sign=False)
  p.trafficLightDetectMode = 2
  sm = gate_sm(carrot_man_message(state=1, src=1))
  sm.recv_time['carrotMan'] = time.monotonic()
  p._update_carrot_man(sm, 0.0, 60.0)
  assert p.trafficState_carrot == 1
  sm = gate_sm(carrot_man_message(state=2, src=1))
  sm.recv_time['carrotMan'] = time.monotonic()
  p._update_carrot_man(sm, 0.0, 60.0)
  assert p.xState.name == 'e2eCruise'
