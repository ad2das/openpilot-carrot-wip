"""Execute the production sender/controller with deterministic ECU replies.

These tests exercise CAN bytes and cleanup, not physical lamps or ECU support.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.hyundai.atc_blinker import PalisadeAtcBlinker, classic_atc_enabled
from opendbc.car.hyundai.values import CAR, HyundaiFlags, HyundaiSafetyFlags


def ack(frame):
  address, data, bus = frame
  assert (address, bus) == (0x770, 0)
  payload = b'\x50\x03' if data[1] == 0x10 else b'\x6f' + data[2:5]
  return (0x778, (bytes([len(payload)]) + payload).ljust(8, b'\xaa'), 0)


def is_on(frame):
  return frame[1][1] == 0x2f and frame[1][4] == 3


class Rig:
  def __init__(self):
    self.sender = PalisadeAtcBlinker()
    self.now = 10_000_000_000
    self.cs = structs.CarState(canValid=True, vEgo=10.0, vEgoRaw=10.0, gearShifter='drive')
    self.cc = structs.CarControl(enabled=True, latActive=True)
    self.replies = []
    self.history = []
    self.direction = 0
    self.phase = 'none'
    self.episode = 0

  def step(self, *, dt=10_000_000, respond=True, cc_present=True, control_valid=True,
           frozen=False, replies=None, switches=(0, 0, 0)):
    self.now += dt
    self.cs.turnSignalInput = dict(valid=True, sampleMonoTime=self.now,
                                   leftSwitch=switches[0], rightSwitch=switches[1], hazardSwitch=switches[2])
    if not frozen:
      self.cc.atcSignal = dict(sourceId=100, episode=self.episode, sourceMonoTime=self.now,
                              direction=self.direction, phase=self.phase, valid=True,
                              requestMonoTime=self.now)
    incoming = self.replies if replies is None else replies
    self.replies = []
    out = self.sender.update(self.cc if cc_present else None, self.cs, incoming, self.now, control_valid=control_valid)
    self.history.extend((self.now, f) for f in out)
    if respond:
      self.replies = [(*ack(f), self.now + 1) for f in out]
    return out

  def start(self, direction=1):
    assert self.step() == []
    self.direction, self.phase, self.episode = direction, 'turn', 1
    assert self.step()[0][1] == bytes.fromhex('0210030000000000')
    assert self.step()[0][1] == bytes.fromhex('042fbc1500000000')
    assert self.step()[0][1] == bytes.fromhex('042fbc1600000000')
    out = self.step()
    assert len(out) == 1 and is_on(out[0])
    self.step()


@pytest.mark.parametrize('direction', [1, 2])
def test_real_sender_pulses_with_both_release_acks_between_on_commands(direction):
  rig = Rig()
  rig.start(direction)
  for _ in range(170):
    rig.step()
  on = [(t, f) for t, f in rig.history if is_on(f)]
  assert len(on) >= 3
  assert all(f[1][3] == 0x14 + direction for _, f in on)
  assert all(b[0] - a[0] >= 700_000_000 for a, b in zip(on, on[1:], strict=False))
  for (start, _), (end, _) in zip(on, on[1:], strict=False):
    release = [(t, f) for t, f in rig.history if start < t < end and f[1][4] == 0]
    assert {f[1][3] for _, f in release} == {0x15, 0x16}
    assert 350_000_000 <= release[0][0] - start <= 360_000_000
    assert end - release[-1][0] >= 350_000_000


@pytest.mark.parametrize('stop', ['driver_left', 'driver_right', 'hazard', 'controls_dead', 'event_invalid',
                                 'lat_off', 'can_invalid', 'trailer', 'opposite_torque', 'park', 'steer_fault', 'navigation_end'])
def test_live_stop_releases_both_and_same_episode_never_resumes(stop):
  rig = Rig()
  rig.start()
  kwargs = {}
  if stop.startswith('driver'):
    kwargs['switches'] = (int(stop == 'driver_left'), int(stop == 'driver_right'), 0)
  elif stop == 'hazard':
    kwargs['switches'] = (0, 0, 1)
  elif stop == 'controls_dead':
    kwargs['cc_present'] = False
  elif stop == 'event_invalid':
    kwargs['control_valid'] = False
  elif stop == 'lat_off':
    rig.cc.latActive = False
  elif stop == 'can_invalid':
    rig.cs.canValid = False
  elif stop == 'trailer':
    rig.cs.trailerConnected = True
  elif stop == 'opposite_torque':
    rig.cs.steeringPressed, rig.cs.steeringTorque = True, -1.0
  elif stop == 'park':
    rig.cs.gearShifter = 'park'
  elif stop == 'steer_fault':
    rig.cs.steerFaultTemporary = True
  else:
    rig.direction, rig.phase = 0, 'none'
  history_start = len(rig.history)
  rig.step(**kwargs)
  rig.cc.latActive = rig.cs.canValid = True
  rig.cs.trailerConnected = rig.cs.steeringPressed = rig.cs.steerFaultTemporary = False
  rig.cs.gearShifter = 'drive'
  rig.direction, rig.phase = 1, 'turn'
  for _ in range(120):
    rig.step()
  sent = [f for _, f in rig.history[history_start:]]
  assert {f[1][3] for f in sent} == {0x15, 0x16}
  assert all(not is_on(f) for f in sent)
  assert rig.sender.planner is None


def test_frozen_control_publication_stops_even_with_fresh_can():
  rig = Rig()
  rig.start()
  start = rig.now
  for _ in range(100):
    rig.step(frozen=True)
  releases = [(t, f) for t, f in rig.history if t > start]
  assert len(releases) == 2
  assert releases[0][0] - start <= 160_000_000
  assert not any(is_on(f) for _, f in releases)


@pytest.mark.parametrize('bad', ['missing', 'negative', 'echo', 'wrong_bus', 'wrong_did', 'future', 'short_dlc'])
def test_bad_replies_never_acquire_and_cleanup_is_bounded(bad):
  rig = Rig()
  rig.step()
  rig.direction, rig.phase, rig.episode = 1, 'turn', 1
  rig.step()
  reply = rig.replies[0]
  changes = {'missing': [], 'negative': [(0x778, bytes.fromhex('037f103100000000'), 0, rig.now + 1)],
             'echo': [(reply[0], reply[1], 128, reply[3])], 'wrong_bus': [(reply[0], reply[1], 1, reply[3])],
             'wrong_did': [(0x778, bytes.fromhex('046fbc1603000000'), 0, rig.now + 1)],
             'future': [(reply[0], reply[1], 0, rig.now + 1_000_000_000)],
             'short_dlc': [(reply[0], reply[1][:3], 0, reply[3])]}
  rig.step(replies=changes[bad], respond=False)
  for _ in range(300):
    rig.step(respond=False)
  assert not any(is_on(f) for _, f in rig.history)
  releases = [f for _, f in rig.history if f[1][1] == 0x2f]
  assert len(releases) == 6
  assert sum(f[1][3] == 0x15 for f in releases) == 3
  assert rig.sender.status['phase'] == 'fault'


def test_restart_active_and_old_schema_never_transmit():
  rig = Rig()
  rig.direction, rig.phase, rig.episode = 1, 'turn', 5
  for _ in range(30):
    assert rig.step() == []
  rig = Rig()
  rig.step()
  rig.direction, rig.phase, rig.episode = 1, 'turn', 1
  rig.cc.atcSignal = dict(sourceId=100, episode=1, sourceMonoTime=rig.now, direction=1, phase='turn', valid=True, requestMonoTime=0)
  assert rig.step(frozen=True) == []


def test_never_acquires_an_existing_oem_one_touch_signal():
  rig = Rig()
  rig.step()
  rig.cs.leftBlinker = True
  rig.direction, rig.phase, rig.episode = 1, 'turn', 1
  for _ in range(20):
    assert rig.step() == []
  rig.cs.leftBlinker = False
  for _ in range(20):
    assert rig.step() == []  # the blocked navigation episode cannot resume


def test_interface_retains_short_manual_press_even_after_reply_snapshot_is_full():
  from opendbc.car.hyundai.tests.test_classic_atc_interface import make_receive_interface
  ci, _, _ = make_receive_interface()
  ci.update([(100, [(0x778, bytes(8), 0)] * 33),
             (101, [(0x541, (1 << 42).to_bytes(8, 'little'), 0)]),
             (102, [(0x541, bytes(8), 0)])])
  assert ci.CS.atc_last_manual_press_ns == 101
  assert ci.CS.atc_previous_switches == 0
  ci.update([(103, [(0x541, (1 << 41).to_bytes(8, 'little'), 128)])])
  assert ci.CS.atc_last_manual_press_ns == 101  # returned echoes are never driver input
  ci.update([])
  assert ci.CS.atc_last_manual_press_ns == 101


def test_received_batch_cannot_ack_commands_generated_in_same_call():
  rig = Rig()
  rig.step()
  rig.direction, rig.phase, rig.episode = 1, 'turn', 1
  replies = [(0x778, bytes.fromhex(h), 0, rig.now + 10_000_000) for h in
             ('0250030000000000', '046fbc1500000000', '046fbc1600000000')]
  out = rig.step(replies=replies)
  assert [f[1] for f in out] == [bytes.fromhex('0210030000000000')]
  assert rig.sender.status['phase'] == 'session'
  # Even an apparently matching collection after the session may only release
  # the first side; the second ACK has not happened after its command yet.
  rig.step(replies=[(a, d, b, rig.now + 1) for a, d, b, _ in replies])
  assert rig.sender.status['phase'] == 'releasing'
  assert not any(is_on(f) for _, f in rig.history)


def test_episode_time_limit_latches_fault_and_releases():
  rig = Rig()
  rig.start()
  for _ in range(4800):
    rig.step()
  assert rig.sender.status['fault'] == 'maximum_episode_duration'
  assert rig.sender.status['phase'] == 'fault'
  assert rig.history[-1][1][1][4] == 0


def test_new_episode_requires_new_session_and_release_handshake():
  rig = Rig()
  rig.start()
  rig.direction, rig.phase = 0, 'none'
  for _ in range(120):
    rig.step()
  rig.direction, rig.phase, rig.episode = 2, 'turn', 2
  assert rig.step()[0][1] == bytes.fromhex('0210030000000000')
  assert rig.step()[0][1][3:5] == b'\x15\x00'
  assert rig.step()[0][1][3:5] == b'\x16\x00'
  assert rig.step()[0][1][3:5] == b'\x16\x03'


def make_controller(monkeypatch, enabled):
  from opendbc.car.hyundai import carcontroller
  params = SimpleNamespace(get_int=lambda k: int(k == 'HyundaiCameraSCC'), get_bool=lambda k: False, get_float=lambda k: 0.0)
  monkeypatch.setattr(carcontroller, 'Params', lambda: params)
  cp = structs.CarParams(carFingerprint=CAR.HYUNDAI_PALISADE, wheelbase=2.9, steerRatio=15.5,
                        flags=int(HyundaiFlags.CAMERA_SCC | HyundaiFlags.SEND_LFA | (HyundaiFlags.CLASSIC_ATC if enabled else 0)),
                        safetyConfigs=[dict(safetyModel='hyundai', safetyParam=int(HyundaiSafetyFlags.CLASSIC_ATC if enabled else 0))])
  controller = carcontroller.CarController({Bus.pt: 'hyundai_kia_generic'}, cp)
  controller.frame, controller.lkas11_active = 254, True
  parser = CANParser('hyundai_kia_generic', [('LKAS11', 100)], 0)
  parser.update([1_000_000_000, [(0x340, bytes.fromhex('0400002406014011'), 0)]])
  state = SimpleNamespace(out=None, modelV2=None, is_metric=True, lkas11=dict(parser.vl['LKAS11']), clu11=None,
                          mdps12=None, paddle_button_prev=0, softHoldActive=0, scc11=None, scc12=None, scc13=None,
                          scc14=None, fca11=None, atc_control_valid=True, atc_diagnostic_replies=[])
  return controller, state


def test_actual_controller_tx_and_existing_adas_are_identical(monkeypatch):
  enabled, state = make_controller(monkeypatch, True)
  disabled, baseline = make_controller(monkeypatch, False)
  assert classic_atc_enabled(enabled.CP) and not classic_atc_enabled(disabled.CP)
  rig = Rig()
  incoming, atc_sent = [], []
  for i in range(200):
    if i == 2:
      rig.direction, rig.phase, rig.episode = 1, 'turn', 1
    rig.step()
    state.out = baseline.out = rig.cs.as_reader()
    state.atc_diagnostic_replies = incoming
    cc = rig.cc.as_reader()
    actual_actuators, actual = enabled.update(cc, state, rig.now)
    expected_actuators, expected = disabled.update(cc, baseline, rig.now)
    assert actual_actuators.to_dict() == expected_actuators.to_dict()
    assert [f for f in actual if f[0] != 0x770] == expected
    frames = [f for f in actual if f[0] == 0x770]
    incoming = [(*ack(f), rig.now + 1) for f in frames]
    atc_sent.extend(frames)
    assert state.atc_diagnostic_replies == []
  assert sum(is_on(f) for f in atc_sent) >= 3
  cleanup = enabled.shutdown_atc(state, rig.now)
  assert {f[1][3] for f in cleanup} == {0x15, 0x16}
  assert all(not is_on(f) for f in cleanup)
  assert disabled.shutdown_atc(baseline, rig.now) == []


def test_card_cleanup_executes_only_sender_stop(monkeypatch):
  # Execute the actual production method without importing hardware-only msgq.
  root = Path(__file__).resolve().parents[5]
  tree = ast.parse((root / 'openpilot/selfdrive/car/card.py').read_text(encoding='utf-8'))
  method = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'atc_cleanup')
  namespace = dict(REPLAY=True, can_list_to_can_capnp=lambda frames, **kw: (frames, kw))
  exec(compile(ast.Module(body=[method], type_ignores=[]), 'card.py', 'exec'), namespace)
  controller, state = make_controller(monkeypatch, True)
  rig = Rig()
  rig.start()
  controller.atc_blinker = rig.sender
  state.out = rig.cs.as_reader()
  state.atc_diagnostic_replies = rig.replies
  sent = []
  obj = SimpleNamespace(CI=SimpleNamespace(CC=controller, CS=state), can_log_mono_time=rig.now + 10_000_000,
                        pm=SimpleNamespace(send=lambda service, msg: sent.append((service, msg))))
  namespace['atc_cleanup'](obj)
  assert len(sent) == 1 and sent[0][0] == 'sendcan'
  assert sent[0][1][0][0][1][3:5] == b'\x15\x00'
  assert all(f[0] == 0x770 and not is_on(f) for f in sent[0][1][0])


@pytest.mark.parametrize('failure', ['none', 'controls_dead', 'negative_reply'])
def test_runtime_trace_accepted_by_compiled_panda_hooks(failure):
  from opendbc.safety.tests.test_hyundai_atc_c import run_compiled_probe
  rig, trace = Rig(), []
  for i in range(250):
    if i == 2:
      rig.direction, rig.phase, rig.episode = 1, 'turn', 1
    alive = not (failure == 'controls_dead' and i >= 120)
    frames = rig.step(cc_present=alive, respond=False)
    stamp, controls = rig.now // 1000, int(alive)
    trace.append(f'{stamp} {controls} R 541 0 0 0000000000000000 1')
    for frame in frames:
      address, data, bus = frame
      trace.append(f'{stamp} {controls} T {address:x} {bus} 0 {data.hex()} 1')
      response = ack(frame)[1]
      if failure == 'negative_reply' and is_on(frame):
        response = bytes.fromhex('037f2f3100000000')
      rig.replies.append((0x778, response, 0, rig.now + 1_000_000))
      trace.append(f'{stamp + 1000} {controls} R 778 0 0 {response.hex()} 1')
  assert any(is_on(f) for _, f in rig.history)
  result = run_compiled_probe('\n'.join(trace))
  assert f'HYUNDAI_ATC_REPLAY_EVENTS={len(trace)}' in result


def test_pulses_alone_keep_session_alive_through_long_signal():
  # Simulated ECU drops the extended session after 5 s without a diagnostic
  # request. ON/release pulses every ~0.35 s must keep it open for 20 s.
  rig = Rig()
  rig.start()
  last_diag, session_open, gaps = rig.now, True, []
  for _ in range(2000):
    out = rig.step(respond=False)
    if session_open and rig.now - last_diag > 5_000_000_000:
      session_open = False
    if out:
      gaps.append(rig.now - last_diag)
      last_diag = rig.now
    rig.replies = [(*ack(f), rig.now + 1) if session_open else (0x778, bytes.fromhex('037f2f7faaaaaaaa'), 0, rig.now + 1) for f in out]
  assert session_open and max(gaps) < 1_000_000_000
  assert rig.sender.planner is not None and rig.sender.planner.status.fault is None
  assert any(is_on(f) for t, f in rig.history if t > rig.now - 2_000_000_000)


class CheckRig(Rig):
  """Engaged, stopped in D; lamp(rig) models the CGW1 left lamp."""
  def __init__(self, lamp=lambda rig: 0):
    super().__init__()
    self.cs.vEgo = self.cs.vEgoRaw = 0.0
    self.cs.standstill = True
    self.lamp = lamp
    self.request = 0

  def step(self, *, dt=10_000_000, respond=True, switches=(0, 0, 0), **_):
    self.now += dt
    lamp = self.lamp(self)
    self.cs.turnSignalInput = dict(valid=True, sampleMonoTime=self.now, leftSwitch=switches[0],
                                   rightSwitch=switches[1], hazardSwitch=switches[2], leftLamp=lamp)
    self.cs.leftBlinker = bool(lamp)
    self.cc.atcSignal = dict(sourceId=100, episode=self.episode, sourceMonoTime=self.now, direction=self.direction,
                             phase=self.phase, valid=True, requestMonoTime=self.now)
    incoming, self.replies = self.replies, []
    out = self.sender.update(self.cc, self.cs, incoming, self.now, control_valid=True, test_request=self.request)
    self.history.extend((self.now, f) for f in out)
    if respond:
      self.replies = [(*ack(f), self.now + 1) for f in out]
    return out

  def run(self, steps=700, **kwargs):
    self.step()
    self.request = self.now
    for _ in range(steps):
      self.step(**kwargs)
      if self.sender.test_result is not None and self.sender.test is None:
        break
    return self.sender.test_result


def steady_lamp(rig):
  return int(rig.sender._on_since is not None)


def flashing_lamp(rig):
  since = rig.sender._on_since
  return 0 if since is None else int((rig.now / 1e9 - since) // 0.33 % 2 == 0)


def test_stationary_check_steady_lamp_long_first_on_then_pulses_and_release():
  rig = CheckRig(steady_lamp)
  result = rig.run()
  assert result['verdict'] == 'steady', result
  assert result['icuPositive'] >= 3 and result['icuNegative'] == 0 and not result['fault']
  assert result['firstOnLampRatio'] >= 0.95 and result['firstOnLampRises'] == 0
  assert result['lampAfter'] == 0 and result['pulses'] >= 3
  on = [t for t, f in rig.history if is_on(f)]
  assert all(f[1][3] == 0x15 for t, f in rig.history if is_on(f))  # left only
  first_release = next(t for t, f in rig.history if t > on[0] and f[1][4] == 0)
  assert 0.9e9 <= first_release - on[0] <= 0.91e9 < 1.0e9  # below planner max_active_s
  assert on[-1] - rig.request < 3.0e9
  last = rig.history[-1][1][1]
  assert last[1] == 0x2f and last[4] == 0  # ends by returning control
  assert {f[1][3] for t, f in rig.history if t > on[-1] and f[1][4] == 0} == {0x15, 0x16}
  assert rig.sender.planner is None and rig.sender.test is None


def test_stationary_check_detects_icu_self_flash():
  result = CheckRig(flashing_lamp).run()
  assert result['verdict'] == 'icu_flashes', result
  assert result['firstOnLampRises'] >= 1 and result['firstOnLampRatio'] < 0.8


def test_stationary_check_reports_lamp_left_on_after_release():
  rig = CheckRig(lambda r: int(r.sender._on_since is not None or r.sender.test_result is None and r.request and r.now - r.request > 1e9))
  result = rig.run()
  assert result['lampAfter'] == 1


def test_stationary_check_without_icu_reply_faults_and_never_turns_on():
  rig = CheckRig()
  result = rig.run(respond=False)
  assert result['verdict'] == 'no_icu_response' and result['fault'] == 'ack_timeout', result
  assert not any(is_on(f) for _, f in rig.history)


@pytest.mark.parametrize('block', ['not_engaged', 'not_stopped', 'gear', 'driver_signal', 'steering', 'car_fault'])
def test_stationary_check_refuses_unsafe_start_without_sending(block):
  rig = CheckRig()
  kwargs = {}
  if block == 'not_engaged':
    rig.cc.enabled = False
  elif block == 'not_stopped':
    rig.cs.standstill, rig.cs.vEgo = False, 2.0
  elif block == 'gear':
    rig.cs.gearShifter = 'park'
  elif block == 'driver_signal':
    kwargs['switches'] = (0, 1, 0)
  elif block == 'steering':
    rig.cs.steeringPressed = True
  elif block == 'car_fault':
    rig.cs.steerFaultTemporary = True
  result = rig.run(steps=5, **kwargs)
  assert result['verdict'] == 'canceled' and result['cancel'] == block, result
  assert rig.history == []


def test_navigation_signal_preempts_check_and_check_never_starts_over_it():
  rig = CheckRig(steady_lamp)
  rig.step()
  rig.request = rig.now
  while not any(is_on(f) for _, f in rig.history):
    rig.step()
  rig.direction, rig.phase, rig.episode = 2, 'turn', 1
  cut = rig.now
  for _ in range(300):
    rig.step()
  assert rig.sender.test_result['cancel'] == 'navigation_request'
  after = [(t, f) for t, f in rig.history if t > cut]
  left_on_after = [t for t, f in after if is_on(f) and f[1][3] == 0x15]
  first_right = next(t for t, f in after if is_on(f) and f[1][3] == 0x16)
  assert not left_on_after and {f[1][3] for t, f in after if t < first_right and f[1][1] == 0x2f and f[1][4] == 0} == {0x15, 0x16}
  rig.request = rig.now  # navigation owns the ICU: a new check is refused unsent
  sent = len(rig.history)
  rig.step(respond=False)
  assert rig.sender.test_result['cancel'] == 'busy' and rig.sender.test is None
  assert all(f[1][3] == 0x16 or f[1][4] == 0 or f[1][1] == 0x10 for _, f in rig.history[sent:])


def test_stale_or_repeated_request_never_runs():
  rig = CheckRig()
  rig.step()
  rig.request = rig.now - 3_000_000_000
  rig.step()
  assert rig.sender.test_result['cancel'] == 'stale_request' and rig.history == []
  first = rig.sender.test_result
  for _ in range(50):
    rig.step()
  assert rig.sender.test_result is first and rig.history == []


@pytest.mark.parametrize('stop', ['driver_signal', 'not_stopped', 'not_engaged'])
def test_stationary_check_cancel_mid_test_releases_both_and_stops(stop):
  rig = CheckRig(steady_lamp)
  rig.step()
  rig.request = rig.now
  while not any(is_on(f) for _, f in rig.history):
    rig.step()
  kwargs = {}
  if stop == 'driver_signal':
    kwargs['switches'] = (1, 0, 0)
  elif stop == 'not_stopped':
    rig.cs.standstill, rig.cs.vEgo = False, 1.0
  else:
    rig.cc.enabled = False
  cut = rig.now
  for _ in range(300):
    rig.step(**kwargs)
  result = rig.sender.test_result
  assert result['verdict'] == 'canceled' and result['cancel'] == stop, result
  after = [f for t, f in rig.history if t > cut]
  assert not any(is_on(f) for f in after)
  assert {f[1][3] for f in after if f[1][4] == 0} == {0x15, 0x16}


@pytest.mark.parametrize('cancel', [False, True])
def test_stationary_check_trace_accepted_by_compiled_panda_hooks(cancel):
  from opendbc.safety.tests.test_hyundai_atc_c import run_compiled_probe
  rig, trace = CheckRig(steady_lamp), []
  for i in range(500):
    if i == 1:
      rig.request = rig.now
    if cancel and i == 60:  # during the first 0.9 s ON
      rig.cc.enabled = False
    frames = rig.step(respond=False)
    stamp, controls = rig.now // 1000, int(rig.cc.enabled)
    trace.append(f'{stamp} {controls} R 541 0 0 0000000000000000 1')
    for frame in frames:
      address, data, bus = frame
      trace.append(f'{stamp} {controls} T {address:x} {bus} 0 {data.hex()} 1')
      response = ack(frame)[1]
      rig.replies.append((0x778, response, 0, rig.now + 1_000_000))
      trace.append(f'{stamp + 1000} {controls} R 778 0 0 {response.hex()} 1')
  assert sum(is_on(f) for _, f in rig.history) == (1 if cancel else 4)
  assert rig.sender.test_result['verdict'] == ('canceled' if cancel else 'steady')
  result = run_compiled_probe('\n'.join(trace))
  assert f'HYUNDAI_ATC_REPLAY_EVENTS={len(trace)}' in result
