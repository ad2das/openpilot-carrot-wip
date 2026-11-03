"""W-TSG1: experimental CarrotTlvStopGo commits the existing e2eStop on a
confirmed TLV red.

The toggle only decides WHETHER a confirmed red may enter the stop the model's
own cue already uses; the stop distance is always stop_model_x_rl. At the
default 0 the planner must behave byte-identically to today (the first case
pins that), so every case here is either a commit gate or a release edge of
`_update_tlv_stop_go`.

Full-update cases run the real `update()` through `planner_class`, which stubs
`_params_update`; the toggle, the detect mode and the two carrotMan readers are
therefore set on the instance. `inputs()` has no carrotMan, so the TLV state is
injected through `_read_tlv_red_active` / `_read_tlv_confirmed_red`.
"""
import time

from test_tlv_fusion import carrot_man_message, gate_sm, planner_for_gate

from openpilot.selfdrive.carrot.tlv_green_override import GreenHold, override_ok
from tools.signal_analysis.replay_signal_assist import REPO, planner_class
from tools.signal_analysis.test_signal_assist_planner import inputs

# The replay planner is exec'd in its own namespace, so its enums are distinct
# classes from the module-level ones and must come back out of that namespace.
# `planner_class` imports only t_follow helpers, so the two names CarrotPlanner
# resolves from module globals but lives in tlv_green_override must be added
# before the real __init__ runs.
_CACHE = {}


def replay():
  if not _CACHE:
    cls, ns = planner_class((REPO / 'openpilot/selfdrive/carrot/carrot_functions.py').read_text(encoding='utf-8'))
    ns['GreenHold'] = GreenHold
    ns['override_ok'] = override_ok
    _CACHE.update(cls=cls, XState=ns['XState'], TrafficState=ns['TrafficState'],
                  DrivingMode=ns['DrivingMode'], cooldown=ns['TLV_GAS_COOLDOWN_FRAMES'])
  return _CACHE


def inputs_at(speed, stop_x, lead=False):
  """`inputs()` whose model x[31] is exactly `stop_x` (it ramps distance*31/32)."""
  return inputs(speed=speed, distance=stop_x * 32. / 31., lead=lead)


def planner(mode=2, toggle=1):
  c = replay()
  p = c['cls']()
  p.tlvStopGo = toggle
  p.trafficLightDetectMode = mode
  p.myDrivingMode = c['DrivingMode'].Normal
  calls = []
  p._log_tlv_stop_go = lambda action, reason, v_ego, stop_x: calls.append((action, reason))
  return p, calls


def step(p, sm, red_active=True, confirmed=True, v_cruise=80.):
  p._read_tlv_red_active = lambda _sm, _v=red_active: _v
  p._read_tlv_confirmed_red = lambda _sm, _v=confirmed: _v
  p.update(sm, v_cruise, 'acc')


# ---------------------------------------------------------------------------
# Default 0 must not touch anything
# ---------------------------------------------------------------------------

def test_toggle_off_never_commits_and_leaves_the_model_alone():
  XState, TrafficState = replay()['XState'], replay()['TrafficState']
  p, calls = planner(2, toggle=0)
  assert p.tlvStopGo == 0
  step(p, inputs_at(13.89, 100.))  # 50 km/h
  assert p.tlv_stop_committed is False
  assert p.xState == XState.e2eCruise
  assert p.trafficState != TrafficState.red  # the model's own cue stays in charge
  assert calls == []


def test_confirmed_red_commits_the_existing_stop_at_the_model_distance():
  XState, TrafficState = replay()['XState'], replay()['TrafficState']
  p, calls = planner(2, toggle=1)
  sm = inputs_at(13.89, 100.)  # min_dist = max(5, 13.89^2/4.8) = 40.2 m
  step(p, sm)
  assert p.tlv_stop_committed is True
  assert p.trafficState == TrafficState.red
  assert p.xState == XState.e2eStop
  assert calls == [('commit', 'confirmed_red')]
  # Second frame: the stop branch is now active and applies the existing
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
  assert p.xState != XState.e2eStop
  assert calls == []


def test_mode_zero_and_high_mode_never_commit():
  p, calls = planner(0, toggle=1)
  step(p, inputs_at(13.89, 100.))
  assert p.tlv_stop_committed is False and calls == []

  p, calls = planner(2, toggle=1)
  p.myDrivingMode = replay()['DrivingMode'].High
  step(p, inputs_at(13.89, 100.))
  assert p.tlv_stop_committed is False and calls == []


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
    sm = inputs_at(13.89, 100., lead=name == 'lead')
    mutate(sm)
    step(p, sm)
    assert p.tlv_stop_committed is False, name
    assert calls == [], name

  p, calls = planner(2, toggle=1)
  sm = inputs_at(13.89, 100.)
  sm['modelV2'].position.y = [0.] * 32 + [6.]
  step(p, sm)
  assert p.tlv_stop_committed is False
  assert calls == []


def test_commit_is_vetoed_by_speed_and_distance_bounds():
  # 83 km/h > 82/3.6 = 22.78 m/s. stop_x 150 m is inside the distance window at
  # that speed and detector support keeps the TLV red alive through the fast
  # gate, so the speed bound itself is what vetoes.
  p, calls = planner(2, toggle=1)
  sm = inputs_at(83. / 3.6, 150.)
  sm['carrotMan'] = carrot_man_message(state=1, src=1, det_support=2)
  step(p, sm)
  assert p.tlv_stop_committed is False
  assert calls == []

  for stop_x in (30., 230.):  # below v^2/(2*2.4) at 50 km/h, and above 220 m
    p, calls = planner(2, toggle=1)
    step(p, inputs_at(13.89, stop_x))
    assert p.tlv_stop_committed is False, stop_x
    assert calls == [], stop_x


def test_eighty_one_point_nine_kph_commits_under_the_220_m_cap():
  # Free-cruise x[31] = v*10*(31/32)^2 = 213.5 m at 81.9 km/h, which the old
  # 200 m cap blocked. Detector support keeps the TLV red alive through the
  # W139c fast gate; 22.75 m/s is still under 82/3.6 = 22.78 m/s and the floor
  # at that speed is 22.75^2/4.8 = 107.8 m.
  p, calls = planner(2, toggle=1)
  sm = inputs_at(22.75, 213.5)
  sm['carrotMan'] = carrot_man_message(state=1, src=1, det_support=2)
  step(p, sm)
  assert p.tlv_stop_committed is True
  assert calls == [('commit', 'confirmed_red')]


def test_distance_window_tracks_the_comfort_brake():
  # 50 km/h, comfort brake 2.4: the floor is 40.19 m, so 45 m commits and 40 m
  # must not; the lamp never creates a distance, only this plausibility window.
  for stop_x, expected in ((100., True), (45., True), (40., False)):
    p, _ = planner(2, toggle=1)
    step(p, inputs_at(13.89, stop_x))
    assert p.tlv_stop_committed is expected, stop_x


# ---------------------------------------------------------------------------
# Release edges
# ---------------------------------------------------------------------------

def test_red_end_releases_back_to_cruise():
  XState = replay()['XState']
  p, calls = planner(2, toggle=1)
  sm = inputs_at(10., 100.)  # 36 km/h
  step(p, sm)
  assert p.tlv_stop_committed is True and p.xState == XState.e2eStop
  step(p, sm, red_active=False, confirmed=False)
  assert p.tlv_stop_committed is False
  assert p.xState == XState.e2eCruise
  assert calls == [('commit', 'confirmed_red'), ('release', 'red_end')]


def test_gas_releases_and_holds_off_for_the_cooldown():
  p, calls = planner(2, toggle=1)
  cooldown = replay()['cooldown']
  sm = inputs_at(10., 100.)
  step(p, sm)
  assert p.tlv_stop_committed is True

  sm['carState'].gasPressed = True
  step(p, sm)
  assert p.tlv_stop_committed is False
  assert calls[-1] == ('release', 'gas')
  assert p.tlv_gas_cooldown_frames == cooldown

  sm['carState'].gasPressed = False
  # The cooldown decrements before the commit gate, so the frame that takes it
  # from 1 to 0 is the first frame that may commit again.
  for _ in range(cooldown):
    if p.tlv_gas_cooldown_frames <= 1:
      break
    step(p, sm)
    assert p.tlv_stop_committed is False  # still cooling down
  assert p.tlv_gas_cooldown_frames == 1
  step(p, sm)
  assert p.tlv_stop_committed is True
  assert calls[-1] == ('commit', 'confirmed_red')


def test_lead_appearance_releases():
  p, calls = planner(2, toggle=1)
  step(p, inputs_at(10., 100.))
  assert p.tlv_stop_committed is True
  step(p, inputs_at(10., 100., lead=True))
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

    step(p, sm, red_active=False, confirmed=False)  # TLV green ends the commit
    assert p.tlv_stop_committed is False, mode
    assert calls[-1] == ('release', 'red_end'), mode
    # The model's own start cue needs 5 frames to reach trafficState green; after
    # that the existing departure rule decides, per the detect mode.
    for _ in range(10):
      step(p, sm, red_active=False, confirmed=False)
    assert (p.xState == XState.e2eCruise) is starts, mode


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
