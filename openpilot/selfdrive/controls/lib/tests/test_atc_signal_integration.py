"""Real schemas, helper FSM and runtime adapters; no hardware or CAN sender."""

import ast
import importlib.util
from pathlib import Path
import sys
import types
import pytest
from openpilot.cereal import car, custom, log
from opendbc.car.hyundai.turn_signal_input import decode_turn_signal_input, from_parser
from openpilot.selfdrive.controls.lib.atc_signal import EpisodeProducer, RequestArbiter
from openpilot.selfdrive.controls.lib.atc_signal_integration import (
  navigation_candidate,
  model_candidate,
  control_candidate,
  service_fresh,
  NavigationSourceTracker,
  AtcDriverInput,
)

ROOT = Path(__file__).resolve().parents[4]


class MemoryParams:
  def get_bool(self, key):
    return False

  def get_int(self, key):
    return 0

  def get_float(self, key):
    return 0.0


class NoBluetooth:
  def __init__(self, *args):
    self.command = None

  def read(self, **kwargs):
    return self.command


def load_helper():
  modules = {}
  for name, values in {
    'openpilot.common.params': {'Params': MemoryParams},
    'openpilot.common.realtime': {'DT_MDL': 0.05, 'DT_CTRL': 0.01},
    'openpilot.selfdrive.carrot.bluetooth.model': {'CommandReader': NoBluetooth},
  }.items():
    modules[name] = types.ModuleType(name)
    modules[name].__dict__.update(values)
  path = ROOT / 'selfdrive/controls/lib/desire_helper.py'
  spec = importlib.util.spec_from_file_location('atc_test_actual_desire_helper', path)
  module = importlib.util.module_from_spec(spec)
  # Only platform I/O is replaced; the real helper and geometry execute.
  with pytest.MonkeyPatch.context() as monkeypatch:
    for name, stub in modules.items():
      monkeypatch.setitem(sys.modules, name, stub)
    spec.loader.exec_module(module)
  return module.DesireHelper


def wire_roundtrip(struct, values):
  msg = struct.new_message(**values)
  with struct.from_bytes(msg.to_bytes()) as parsed:
    return parsed.as_builder()


def model():
  return log.ModelDataV2.new_message(
    laneLines=[{'y': [y] * 33} for y in (-5.3, -1.8, 1.8, 5.3)],
    laneLineProbs=[1.0] * 4,
    roadEdges=[{'y': [y] * 33} for y in (-8.0, 8.0)],
    orientationRate={'z': [0.0] * 33},
    meta={'desireState': [0.0] * 8},
  )


class ReplaySM(dict):
  def __init__(self):
    super().__init__()
    self.seen, self.alive, self.valid, self.logMonoTime, self.recv_time = ({}, {}, {}, {}, {})

  def put(self, name, msg, now):
    self[name] = msg
    self.seen[name] = self.alive[name] = self.valid[name] = True
    self.logMonoTime[name], self.recv_time[name] = (now, now / 1000000000.0)


class Chain:
  def __init__(self, use_switches=False):
    self.use_switches = use_switches
    self.driver_input = AtcDriverInput()
    self.producer = EpisodeProducer(source_id=100, max_age_seconds=2.0)
    self.model_arbiter = RequestArbiter(max_age_seconds=2.0)
    self.control_arbiter = RequestArbiter(max_age_seconds=2.0)
    self.helper = load_helper()()
    self.sm = ReplaySM()
    self.now = 10000000000
    self.cs = car.CarState.new_message(canValid=True, vEgo=10.0, gearShifter='drive')
    self.radar = log.RadarState.new_message(leadLeft={'status': False}, leadRight={'status': False})

  def step(self, token='none', *, raw=0, enabled=True, car_changes=None, cmd='', arg='', cmd_index=0):
    self.now += 50000000
    for key, value in (car_changes or {}).items():
      setattr(self.cs, key, value)
    sample = decode_turn_signal_input(raw.to_bytes(8, 'little'), self.now, self.now, bus=0, can_valid=True)
    self.cs.turnSignalInput = vars(sample)
    self.sm.put('carState', wire_roundtrip(car.CarState, self.cs.to_dict()), self.now)
    cc = car.CarControl.new_message(enabled=enabled, latActive=enabled)
    self.sm.put('carControl', cc, self.now)
    request = navigation_candidate(
      self.producer, self.sm, self.now, atc_type=token, uses_nav_instruction=False, carrot_navi_active=False, legacy_source_ns=self.now, source_changed=False
    )
    nav = wire_roundtrip(custom.CarrotMan, {'atcType': token, 'atcSignal': request, 'carrotCmd': cmd, 'carrotArg': arg, 'carrotCmdIndex': cmd_index})
    self.sm.put('carrotMan', nav, self.now)
    md = model()
    self.helper.update(self.cs, md, enabled, 0.1, nav, self.radar,
                       driver_signal=self.driver_input.update(self.cs, self.now) if self.use_switches else None)
    md.meta.atcSignal = model_candidate(self.model_arbiter, self.helper, self.sm, self.now)
    self.sm.put('modelV2', wire_roundtrip(log.ModelDataV2, md.to_dict()), self.now)
    cc.atcSignal = control_candidate(self.control_arbiter, self.sm, cc, self.now)
    return wire_roundtrip(car.CarControl, cc.to_dict())

  def warm(self):
    for _ in range(30):
      self.step()
    for _ in range(3):
      self.step('turn left prepare')

  def active(self, token='turn left'):
    out = None
    for _ in range(4):
      out = self.step(token)
    return out


class TestAtcSignalIntegration:
  @pytest.mark.parametrize('owned', [False, True])
  def test_manual_one_touch_lamp_hold_matches_existing_helper(self, owned):
    legacy, enabled = Chain(), Chain(use_switches=True)
    for c in (legacy, enabled):
      c.step(car_changes={'vEgo': 20.0})
      if owned:
        c.cs.atcSignalStatus = {'enabled': True, 'phase': 'acknowledged', 'lastCommandMonoTime': c.now}
    for raw in [1 << 42] + [1 << 19] * 10:
      for c in (legacy, enabled):
        c.step(raw=raw, car_changes={'leftBlinker': True})
      for field in ('driver_blinker_state', 'lane_change_state', 'lane_change_direction', 'desire'):
        assert getattr(legacy.helper, field) == getattr(enabled.helper, field)
    assert enabled.helper.lane_change_state == log.LaneChangeState.laneChangeStarting

  def test_short_manual_press_survives_between_model_frames(self):
    c = Chain(use_switches=True)
    c.step(car_changes={'vEgo': 20.0})
    c.cs.atcSignalStatus = {'enabled': True, 'phase': 'acknowledged', 'lastCommandMonoTime': c.now}
    # card observed a press and release within one model interval. Modeld sees
    # only raw released + OEM lamp, and the retained actual press timestamp.
    press_ns = c.now + 10_000_000
    c.cs.leftBlinker = True
    c.cs.turnSignalInput = {'valid': True, 'sampleMonoTime': c.now + 50_000_000, 'leftLamp': 1,
                           'lastManualPressMonoTime': press_ns}
    assert c.driver_input.update(c.cs, c.now + 50_000_000) is None
    assert c.driver_input.update(c.cs, c.now + 100_000_000) is None
    c.cs.leftBlinker = False
    c.cs.turnSignalInput.sampleMonoTime = c.now + 100_000_000
    assert c.driver_input.update(c.cs, c.now + 100_000_000) == 0

  def test_automatic_feedback_suppression_lasts_until_public_lamp_hold_ends(self):
    c = Chain(use_switches=True)
    c.step()
    c.cs.atcSignalStatus = {'enabled': True, 'phase': 'acknowledged', 'lastCommandMonoTime': c.now}
    c.cs.leftBlinker = True
    assert c.driver_input.update(c.cs, c.now) == 0
    c.cs.atcSignalStatus.phase = 'idle'
    assert c.driver_input.update(c.cs, c.now) == 0
    c.cs.leftBlinker = False
    assert c.driver_input.update(c.cs, c.now) is None
    # An unrelated later OEM lamp request retains the original driver path.
    c.cs.leftBlinker = True
    assert c.driver_input.update(c.cs, c.now) is None

  def test_navigation_helper_control_sender_with_lamp_feedback_and_driver_override(self):
    from opendbc.car.hyundai.atc_blinker import PalisadeAtcBlinker
    sender, c = PalisadeAtcBlinker(), Chain(use_switches=True)
    replies, sent = [], []
    tokens = ['none'] * 30 + ['turn left prepare'] * 3 + ['turn left'] * 50
    for token in tokens:
      feedback = sender.status['txCount'] > 3
      cc = c.step(token, raw=(1 << 19) if feedback else 0, car_changes={'leftBlinker': feedback})
      frames = sender.update(cc, c.cs, replies, c.now)
      c.cs.atcSignalStatus = sender.status
      replies = []
      for address, data, bus in frames:
        assert (address, bus) == (0x770, 0)
        payload = b'\x50\x03' if data[1] == 0x10 else b'\x6f' + data[2:5]
        replies.append((0x778, (bytes([len(payload)]) + payload).ljust(8, b'\x00'), 0, c.now + 1))
      sent.extend(frames)
    assert sum(data[1] == 0x2f and data[4] == 3 for _, data, _ in sent) >= 3
    assert c.helper.driver_blinker_state == 0
    cc = c.step('turn left', raw=1 << 42)
    stopped = sender.update(cc, c.cs, replies, c.now)
    assert cc.atcSignal.direction == 0
    assert all(data[4] == 0 for _, data, _ in stopped)
    assert c.helper.driver_blinker_state == 1

  def test_opposite_torque_on_first_active_frame_cannot_resume(self):
    c = Chain()
    c.warm()
    stopped = c.step('turn left', car_changes={'steeringPressed': True, 'steeringTorque': -1.0})
    assert stopped.atcSignal.direction == 0
    c.cs.steeringPressed = False
    assert c.active().atcSignal.direction == 0

  def test_control_observes_torque_between_model_publications(self):
    c = Chain()
    c.warm()
    c.step('turn left')
    c.sm['carState'].steeringPressed = True
    c.sm['carState'].steeringTorque = -1.0
    result = control_candidate(c.control_arbiter, c.sm, c.sm['carControl'], c.now)
    assert result['direction'] == 0
    assert c.active().atcSignal.direction == 0

  def test_source_change_cancels_same_direction_until_new_idle_episode(self):
    c = Chain()
    c.warm()
    initial = c.active().atcSignal
    tracker = NavigationSourceTracker()
    args = {'uses_nav_instruction': False, 'carrot_navi_active': False, 'remote_id': 'A'}
    assert not tracker.changed(c.sm, **args)
    args['remote_id'] = 'B'
    changed = tracker.changed(c.sm, **args)
    request = navigation_candidate(
      c.producer, c.sm, c.now, atc_type='turn left', uses_nav_instruction=False, carrot_navi_active=False, legacy_source_ns=c.now, source_changed=changed
    )
    assert request['direction'] == 0
    assert request['episode'] == initial.episode
    assert c.active().atcSignal.direction == 0
    args['remote_id'] = 'A'
    assert tracker.changed(c.sm, **args)
    for _ in range(30):
      c.step()
    restarted = c.active()
    assert restarted.atcSignal.episode == initial.episode + 1
    assert restarted.atcSignal.direction == 1

  def test_fork_active_follows_actual_helper_torque_transition(self):
    c = Chain()
    for _ in range(30):
      c.step()
    for _ in range(3):
      c.step('fork right prepare')
    assert c.active('fork right').atcSignal.phase == 'fork_pending'
    result = c.step('fork right', car_changes={'steeringPressed': True, 'steeringTorque': -1.0})
    assert c.helper.lane_change_state == log.LaneChangeState.laneChangeStarting
    assert (result.atcSignal.direction, result.atcSignal.phase) == (2, 'fork_active')

  def test_fork_pending_is_signaled_before_desire_starts(self):
    for token, direction in [('fork left', 1), ('fork right', 2), ('atc left', 1), ('atc right', 2)]:
      c = Chain()
      for _ in range(30):
        c.step()
      for _ in range(3):
        c.step(token + ' prepare')
      out = c.active(token)
      assert c.helper.desire == log.Desire.none
      assert (out.atcSignal.direction, out.atcSignal.phase) == (direction, 'fork_pending')

  def test_source_switch_between_external_providers_is_detected(self):
    tracker, sm = (NavigationSourceTracker(), ReplaySM())
    assert not tracker.changed(sm, uses_nav_instruction=False, carrot_navi_active=False, remote_id='A')
    assert not tracker.changed(sm, uses_nav_instruction=False, carrot_navi_active=False, remote_id='A')
    assert tracker.changed(sm, uses_nav_instruction=False, carrot_navi_active=False, remote_id='B')
    sm['carrotNavi'] = custom.CarrotNaviState.new_message(sessionId='session1')
    assert tracker.changed(sm, uses_nav_instruction=False, carrot_navi_active=True, remote_id='B')
    assert not tracker.changed(sm, uses_nav_instruction=False, carrot_navi_active=True, remote_id='B')
    sm['carrotNavi'].sessionId = 'session2'
    assert tracker.changed(sm, uses_nav_instruction=False, carrot_navi_active=True, remote_id='B')
    assert tracker.changed(sm, uses_nav_instruction=True, carrot_navi_active=False, remote_id='B')

  def test_wire_request_drives_only_offline_diagnostic_preview_and_cleans_up(self):
    from opendbc.car.hyundai.atc_diagnostics import PalisadeAtcDiagnosticPlanner, Timing, Side, CanReply, Phase

    planner = PalisadeAtcDiagnosticPlanner(bus=0, timing=Timing(0.5, 10.0, 2))
    c = Chain()
    c.warm()

    def ack(payload, t):
      frame = (bytes([len(payload)]) + payload).ljust(8, b'\x00')
      return planner.receive(CanReply(1912, frame, 0, t), t)

    assert planner.initialize(1.0)[0].payload == bytes.fromhex('1003')
    ack(bytes.fromhex('5003'), 1.1)
    ack(bytes.fromhex('6fbc1500'), 1.2)
    ack(bytes.fromhex('6fbc1600'), 1.3)
    request = c.active().atcSignal
    side = {1: Side.LEFT, 2: Side.RIGHT}.get(request.direction)
    assert planner.request(side, 1.4)[0].payload == bytes.fromhex('2fbc1503')
    ack(bytes.fromhex('6fbc1503'), 1.5)
    assert planner.status.phase == Phase.ACKNOWLEDGED
    stopped = c.step('turn left', raw=1 << 33).atcSignal
    assert stopped.direction == 0
    assert planner.request(None, 1.6)[0].payload == bytes.fromhex('2fbc1500')
    ack(bytes.fromhex('6fbc1500'), 1.7)
    ack(bytes.fromhex('6fbc1600'), 1.8)
    assert planner.status.phase == Phase.READY
    assert c.active().atcSignal.direction == 0

  def test_turn_survives_actual_schemas_despite_lane_change_state_off(self):
    for token, expected in [('turn left', 1), ('turn right', 2)]:
      c = Chain()
      for _ in range(30):
        c.step()
      for _ in range(3):
        c.step(token + ' prepare')
      out = c.active(token)
      assert c.helper.lane_change_state == log.LaneChangeState.off
      assert out.atcSignal.direction == expected
      assert out.atcSignal.phase == 'turn'
      assert not (out.leftBlinker or out.rightBlinker)

  def test_driver_and_current_stop_inputs_latch_until_next_episode(self):
    cases = [
      {'raw': 1 << 42},
      {'raw': 1 << 41},
      {'raw': 1 << 33},
      {'enabled': False},
      {'car_changes': {'canValid': False}},
      {'car_changes': {'trailerConnected': True}},
      {'car_changes': {'steeringPressed': True, 'steeringTorque': -1.0}},
    ]
    for change in cases:
      c = Chain()
      c.warm()
      assert c.active().atcSignal.direction == 1
      assert c.step('turn left', **change).atcSignal.direction == 0
      c.cs.canValid, c.cs.trailerConnected, c.cs.steeringPressed = (True, False, False)
      assert c.active().atcSignal.direction == 0

  def test_startup_mid_turn_and_consumer_restart_do_not_resume(self):
    c = Chain()
    assert c.active().atcSignal.direction == 0
    c = Chain()
    c.warm()
    assert c.active().atcSignal.direction == 1
    c.control_arbiter = RequestArbiter(max_age_seconds=2.0)
    assert c.active().atcSignal.direction == 0
    c.model_arbiter = RequestArbiter(max_age_seconds=2.0)
    assert c.active().atcSignal.direction == 0

  def test_remote_lane_command_never_inherits_navigation_provenance(self):
    c = Chain()
    c.warm()
    for _ in range(4):
      out = c.step('turn left', cmd='LANECHANGE', arg='LEFT', cmd_index=1)
      assert out.atcSignal.direction == 0
      assert not c.helper.atc_signal_from_navigation

  def test_stale_model_and_stale_source_block_and_latch(self):
    c = Chain()
    c.warm()
    assert c.active().atcSignal.direction == 1
    c.sm.logMonoTime['modelV2'] -= 300000000
    out = control_candidate(c.control_arbiter, c.sm, c.sm['carControl'], c.now)
    assert out['direction'] == 0
    assert c.active().atcSignal.direction == 0
    c.sm.logMonoTime['modelV2'] = c.now + 1
    assert not service_fresh(c.sm, 'modelV2', c.now, 250000000)

  def test_new_publication_cannot_refresh_old_navigation_item(self):
    c = Chain()
    c.warm()
    nav = custom.CarrotNaviState.new_message()
    nav.guidanceCurrent.meta.receivedMonoTimeNanos = c.now - 2000000001
    c.sm.put('carrotNavi', nav, c.now)
    result = navigation_candidate(
      c.producer, c.sm, c.now, atc_type='turn left', uses_nav_instruction=False, carrot_navi_active=True, legacy_source_ns=c.now, source_changed=False
    )
    assert result['direction'] == 0
    assert not result['valid']

  def test_raw_adapter_never_turns_unknown_or_lamp_echo_into_driver_input(self):
    t = 1000000000
    for raw in (0, 1 << 19, 1 << 62, 1 << 19 | 1 << 62):
      v = decode_turn_signal_input(raw.to_bytes(8, 'little'), t, t, bus=0, can_valid=True)
      assert v.valid
      assert (v.leftSwitch, v.rightSwitch, v.hazardSwitch) == (0, 0, 0)
    for raw in (2 << 33, 3 << 19, 1 << 41 | 1 << 42):
      assert not decode_turn_signal_input(raw.to_bytes(8, 'little'), t, t, bus=0, can_valid=True).valid
    for bus, stamp, now, data in [(1, t, t, bytes(8)), (0, 0, t, bytes(8)), (0, t, t + 200000001, bytes(8)), (0, t, t - 1, bytes(8)), (0, t, t, bytes(7))]:
      assert not decode_turn_signal_input(data, stamp, now, bus=bus, can_valid=True).valid

  def test_real_parser_timestamp_adapter(self):
    cp = types.SimpleNamespace(
      ts_nanos={'CGW1': {'CF_Gway_TurnSigLh': 1000000000}},
      dat={1345: (1 << 42).to_bytes(8, 'little')},
      _last_update_nanos=1000000000,
      bus=0,
      can_valid=True,
      bus_timeout=False,
    )
    assert from_parser(cp).leftSwitch == 1
    cp._last_update_nanos += 200000001
    assert not from_parser(cp).valid

  def test_adapter_does_not_consume_can_valid_debounce(self):

    class Parser:
      ts_nanos = {'CGW1': {'CF_Gway_TurnSigLh': 1000000000}}
      dat = {1345: bytes(8)}
      bus, bus_timeout, _last_update_nanos = (0, False, 1000000000)

      @property
      def can_valid(self):
        raise AssertionError('Only the interface may consume can_valid debounce')

    assert from_parser(Parser()).valid

  def test_runtime_call_sites_use_tested_adapters(self):
    for rel, expected in [
      ('selfdrive/carrot/carrot_serv.py', 'navigation_candidate'),
      ('selfdrive/modeld/modeld.py', 'model_candidate'),
      ('selfdrive/controls/controlsd.py', 'control_candidate'),
    ]:
      tree = ast.parse((ROOT / rel).read_text(encoding='utf8'))
      assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id == expected) for n in ast.walk(tree)) == 1


@pytest.mark.parametrize('first', ['atc left', 'fork left'])
def test_same_direction_token_change_keeps_signal(first):
  # DesireHelper skips one frame on atc/fork -> turn; that must not end the episode.
  c = Chain(use_switches=True)
  c.warm()
  assert c.active(first).atcSignal.direction == 1
  results = [c.step('turn left').atcSignal.direction for _ in range(10)]
  assert results[-1] == 1


def test_helper_not_ready_beyond_grace_still_blocks():
  from openpilot.selfdrive.controls.lib.atc_signal import AtcRequest, Direction, RequestGates
  arb = RequestArbiter()
  gates = RequestGates(enabled=True, lateral_active=True, can_valid=True, model_fresh=True, navigation_fresh=True, driver_input_known=True)
  t = 10_000_000_000
  base = {'source_id': t - 10, 'episode': 1, 'source_mono_time': t, 'valid': True}
  arb.update(AtcRequest(direction=Direction.NONE, phase='prepare', **base), gates, now_ns=t)
  assert arb.update(AtcRequest(direction=Direction.LEFT, phase='turn', **base), gates, now_ns=t + 50_000_000).direction == Direction.LEFT
  not_ready = AtcRequest(direction=Direction.NONE, phase='prepare', reason='helper_not_ready', **base)
  for k in range(1, 10):
    arb.update(not_ready, gates, now_ns=t + 50_000_000 * (k + 1))
  assert arb.update(AtcRequest(direction=Direction.LEFT, phase='turn', **base), gates, now_ns=t + 600_000_000).direction == Direction.NONE


def test_opposite_manual_lever_ignores_automatic_lamp_feedback():
  c = Chain(use_switches=True)
  c.step(car_changes={'vEgo': 20.0})
  c.cs.atcSignalStatus = {'enabled': True, 'phase': 'acknowledged', 'lastCommandMonoTime': c.now}
  c.step(raw=1 << 19, car_changes={'leftBlinker': True})
  c.step(raw=(1 << 41) | (1 << 19), car_changes={'leftBlinker': True, 'rightBlinker': False})
  assert c.helper.driver_blinker_state == 2
  for _ in range(8):
    c.step(raw=(1 << 41) | (1 << 62), car_changes={'leftBlinker': True, 'rightBlinker': True})
  assert c.helper.driver_blinker_state == 2
