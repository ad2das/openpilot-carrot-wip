"""Stdlib-only tests; synthetic replies do not validate a physical vehicle.

Run directly on hosts without opendbc's compiled/car-schema dependencies.
"""
import importlib.util
from pathlib import Path
import sys
import unittest


SPEC = importlib.util.spec_from_file_location("_atc_diagnostics_unit", Path(__file__).resolve().parents[1] / "atc_diagnostics.py")
assert SPEC is not None and SPEC.loader is not None
atc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = atc
SPEC.loader.exec_module(atc)
Side, Phase, Operation = atc.Side, atc.Phase, atc.Operation


def sf(payload, padding=b"\x00"):
  return (bytes((len(payload),)) + payload).ljust(8, padding)


class TestAtcDiagnostics(unittest.TestCase):
  def planner(self, attempts=2, active=5.0):
    return atc.PalisadeAtcDiagnosticPlanner(bus=2, timing=atc.Timing(ack_timeout_s=1.0, max_active_s=active, release_attempts=attempts))

  def reply(self, planner, payload, now, **changes):
    values = dict(address=0x778, data=sf(payload), src=2, timestamp=now, service="can")
    values.update(changes)
    return planner.receive(atc.CanReply(**values), now)

  def ready(self, planner):
    self.assertEqual(planner.initialize(0)[0].data.hex(), "0210030000000000")
    self.assertEqual(self.reply(planner, b"\x50\x03", .1)[0].payload, bytes.fromhex("2fbc1500"))
    self.assertEqual(self.reply(planner, bytes.fromhex("6fbc1500"), .2)[0].payload, bytes.fromhex("2fbc1600"))
    self.assertEqual(self.reply(planner, bytes.fromhex("6fbc1600"), .3), ())
    self.assertEqual(planner.status.phase, Phase.READY)
    self.assertEqual(planner.status.uncertain_sides, frozenset())
    self.assertEqual(planner.status.release_acknowledged, frozenset(Side))

  def active(self, planner, side=Side.LEFT):
    self.ready(planner)
    frame = planner.request(side, .4)[0]
    self.assertEqual(frame.operation, Operation.ACQUIRE)
    self.reply(planner, bytes((0x6f, 0xbc, side.value, 3)), .5)
    self.assertEqual(planner.status.acknowledged_side, side)

  def test_explicit_parameters_and_unknown_initialization(self):
    with self.assertRaises(TypeError):
      atc.Timing()
    with self.assertRaises(TypeError):
      atc.PalisadeAtcDiagnosticPlanner(bus=2)
    for bus in (-1, 64, 128, 130, True):
      with self.subTest(bus=bus), self.assertRaises(ValueError):
        atc.PalisadeAtcDiagnosticPlanner(bus=bus, timing=atc.Timing(1, 5, 2))
    for args in ((0, 5, 2), (1, float("inf"), 2), (float("nan"), 5, 2), (1, 5, 0), (1, 5, 17), (1, 5, True)):
      with self.subTest(args=args), self.assertRaises(ValueError):
        atc.Timing(*args)
    p = self.planner()
    self.assertEqual(p.status.phase, Phase.UNKNOWN)
    self.assertEqual(p.status.uncertain_sides, frozenset(Side))
    self.assertEqual(p.status.release_acknowledged, frozenset())
    self.assertEqual(p.request(Side.LEFT, 0), ())
    self.assertEqual(p.tick(0), ())

  def test_initialization_does_not_replay_early_request(self):
    p = self.planner()
    p.request(Side.LEFT, 0)
    p.initialize(0)
    p.request(Side.LEFT, .01)
    self.reply(p, b"\x50\x03", .1)
    p.request(Side.RIGHT, .11)
    self.reply(p, bytes.fromhex("6fbc1500"), .2)
    self.assertEqual(self.reply(p, bytes.fromhex("6fbc1600"), .3), ())
    self.assertEqual(p.status.phase, Phase.READY)
    self.assertIsNone(p.status.acknowledged_side)

  def test_exact_preview_and_acknowledged_state(self):
    for side, wire in ((Side.LEFT, "042fbc1503000000"), (Side.RIGHT, "042fbc1603000000")):
      with self.subTest(side=side):
        p = self.planner()
        self.ready(p)
        frame = p.request(side, .4)[0]
        self.assertEqual((frame.address, frame.bus, frame.data.hex(), frame.attempt), (0x770, 2, wire, 1))
        self.assertIsNone(p.status.acknowledged_side)
        self.reply(p, bytes((0x6f, 0xbc, side.value, 3)), .5)
        self.assertEqual(p.status.phase, Phase.ACKNOWLEDGED)
        self.assertEqual(p.status.uncertain_sides, frozenset((side,)))
        self.assertEqual(p.request(side, .6), ())

  def test_only_real_correct_bus_response_can_acknowledge(self):
    p = self.planner()
    self.ready(p)
    p.request(Side.LEFT, .4)
    for changes in ({"src": 0}, {"src": 128}, {"src": 130}, {"src": 192}, {"src": 2.0}, {"address": 0x770},
                    {"address": 0x779}, {"address": float(0x778)}, {"service": "sendcan"}):
      self.assertEqual(self.reply(p, bytes.fromhex("6fbc1503"), .5, **changes), ())
      self.assertEqual(p.status.phase, Phase.ACQUIRING)
      self.assertIsNone(p.status.fault)
    self.reply(p, bytes.fromhex("6fbc1503"), .6)
    self.assertEqual(p.status.phase, Phase.ACKNOWLEDGED)

  def test_stale_and_future_reply_timestamps(self):
    p = self.planner()
    self.ready(p)
    p.request(Side.LEFT, .4)
    self.assertEqual(self.reply(p, bytes.fromhex("6fbc1503"), .5, timestamp=.39), ())
    self.assertEqual(p.status.phase, Phase.ACQUIRING)
    frames = self.reply(p, bytes.fromhex("6fbc1503"), .5, timestamp=.6)
    self.assertEqual(frames[0].operation, Operation.RELEASE)
    self.assertEqual(p.status.fault, "future_reply")

  def test_session_response_shapes_and_nonzero_padding(self):
    for payload in (bytes.fromhex("5003"), bytes.fromhex("5003003201f4")):
      p = self.planner()
      p.initialize(0)
      self.reply(p, payload, .1, data=sf(payload, b"\xaa"))
      self.assertEqual(p.status.phase, Phase.RELEASING)
      self.assertIsNone(p.status.fault)
    for payload in (bytes.fromhex("5002"), bytes.fromhex("500300"), bytes.fromhex("5003003201"), bytes.fromhex("6fbc1500")):
      p = self.planner()
      p.initialize(0)
      self.reply(p, payload, .1)
      self.assertEqual(p.status.fault, "unexpected_session_response")

  def test_nonfinite_reply_time_triggers_cleanup(self):
    for timestamp in (float("nan"), float("inf"), True):
      p = self.planner()
      self.ready(p)
      p.request(Side.LEFT, .4)
      self.assertEqual(self.reply(p, bytes.fromhex("6fbc1503"), .5, timestamp=timestamp)[0].operation, Operation.RELEASE)
      self.assertEqual(p.status.fault, "invalid_reply_time")

  def test_wrong_side_control_extra_or_missing_ack_bytes_fail(self):
    for payload in ("6fbc1603", "6fbc1500", "6fbc15", "6fbc150300", "2fbc1503"):
      with self.subTest(payload=payload):
        p = self.planner()
        self.ready(p)
        p.request(Side.LEFT, .4)
        frame = self.reply(p, bytes.fromhex(payload), .5)[0]
        self.assertEqual(frame.payload, bytes.fromhex("2fbc1500"))
        self.assertEqual(p.status.fault, "unexpected_control_response")

  def test_malformed_frames_fail_closed(self):
    cases = [b"", b"\x00", b"\x00" * 8, b"\x08" + b"x" * 7,
             bytes.fromhex("10046fbc15030000"), bytes.fromhex("046fbc15"), b"\x04" + b"x" * 8, bytearray(8)]
    for data in cases:
      with self.subTest(data=data):
        p = self.planner()
        self.ready(p)
        p.request(Side.LEFT, .4)
        frames = self.reply(p, b"", .5, data=data)
        self.assertEqual(frames[0].operation, Operation.RELEASE)
        self.assertEqual(p.status.fault, "malformed_single_frame")

  def test_negative_and_response_pending_do_not_extend_on(self):
    for nrc in (0x22, 0x31, 0x78):
      p = self.planner()
      self.ready(p)
      p.request(Side.LEFT, .4)
      frame = self.reply(p, bytes((0x7f, 0x2f, nrc)), .5)[0]
      self.assertEqual(frame.operation, Operation.RELEASE)
      self.assertEqual(p.status.fault, "negative_response")

  def test_session_timeout_cleanup_bounded_then_latched(self):
    p = self.planner()
    frames = list(p.initialize(0))
    for now in (1, 2, 3, 4, 5, 6, 100):
      frames.extend(p.tick(now))
    self.assertEqual([f.payload.hex() for f in frames], ["1003", "2fbc1500", "2fbc1500", "2fbc1600", "2fbc1600"])
    self.assertEqual([f.attempt for f in frames], [1, 1, 2, 1, 2])
    self.assertEqual(p.status.phase, Phase.FAULT)
    self.assertEqual(p.status.uncertain_sides, frozenset(Side))
    self.assertEqual(p.status.release_acknowledged, frozenset())
    self.assertEqual(p.request(Side.LEFT, 101), ())
    self.assertEqual(p.initialize(102), ())

  def test_acquire_timeout_never_retries_on_and_late_ack_cannot_match_cleanup(self):
    p = self.planner()
    self.ready(p)
    p.request(Side.LEFT, .4)
    frames = self.reply(p, bytes.fromhex("6fbc1503"), 1.4, timestamp=1.3)
    self.assertEqual(frames[0].payload, bytes.fromhex("2fbc1500"))
    self.assertIsNone(p.status.acknowledged_side)
    self.assertEqual(p.status.fault, "ack_timeout")
    self.reply(p, bytes.fromhex("6fbc1500"), 1.5)
    self.reply(p, bytes.fromhex("6fbc1600"), 1.6)
    self.assertEqual(p.status.phase, Phase.FAULT)
    self.assertEqual(p.status.uncertain_sides, frozenset())
    self.assertEqual(p.request(Side.LEFT, 1.7), ())

  def test_partial_release_does_not_claim_both_released(self):
    p = self.planner(attempts=1)
    self.active(p)
    p.request(None, .6)
    self.reply(p, bytes.fromhex("6fbc1500"), .7)
    self.assertEqual(p.tick(1.7), ())
    self.assertEqual(p.status.phase, Phase.FAULT)
    self.assertEqual(p.status.release_acknowledged, frozenset((Side.LEFT,)))
    self.assertEqual(p.status.uncertain_sides, frozenset((Side.RIGHT,)))

  def test_cleanup_retry_success_keeps_fault_latched(self):
    p = self.planner()
    self.active(p)
    p.request(None, .6)
    retry = p.tick(1.6)[0]
    self.assertEqual((retry.side, retry.attempt), (Side.LEFT, 2))
    self.reply(p, bytes.fromhex("6fbc1500"), 1.7)
    self.reply(p, bytes.fromhex("6fbc1600"), 1.8)
    self.assertEqual(p.status.phase, Phase.FAULT)
    self.assertEqual(p.status.uncertain_sides, frozenset())
    self.assertEqual(p.status.release_acknowledged, frozenset(Side))
    self.assertEqual(p.request(Side.RIGHT, 1.9), ())

  def test_switch_releases_prior_side_and_both_acks_before_acquire(self):
    for old, new in ((Side.LEFT, Side.RIGHT), (Side.RIGHT, Side.LEFT)):
      p = self.planner()
      self.active(p, old)
      frames = list(p.request(new, .6))
      self.assertEqual(frames[0].side, old)
      frames.extend(self.reply(p, bytes((0x6f, 0xbc, old.value, 0)), .7))
      self.assertTrue(all(f.operation is Operation.RELEASE for f in frames))
      frames.extend(self.reply(p, bytes((0x6f, 0xbc, new.value, 0)), .8))
      self.assertEqual([f.operation for f in frames], [Operation.RELEASE, Operation.RELEASE, Operation.ACQUIRE])
      self.assertEqual(frames[-1].side, new)

  def test_cancel_unacknowledged_acquire_and_cancel_queued_switch(self):
    p = self.planner()
    self.ready(p)
    p.request(Side.RIGHT, .4)
    self.assertEqual(p.request(Side.LEFT, .5)[0].side, Side.RIGHT)
    p.request(None, .6)
    self.reply(p, bytes.fromhex("6fbc1600"), .7)
    self.assertEqual(self.reply(p, bytes.fromhex("6fbc1500"), .8), ())
    self.assertEqual(p.status.phase, Phase.READY)
    self.assertIsNone(p.status.acknowledged_side)

  def test_changed_request_during_cleanup_discards_stale_queued_direction(self):
    for old, new in ((Side.LEFT, Side.RIGHT), (Side.RIGHT, Side.LEFT)):
      p = self.planner()
      self.active(p, old)
      p.request(new, .6)
      self.assertEqual(p.request(old, .65), ())
      self.reply(p, bytes((0x6f, 0xbc, old.value, 0)), .7)
      self.assertEqual(self.reply(p, bytes((0x6f, 0xbc, new.value, 0)), .8), ())
      self.assertEqual(p.status.phase, Phase.READY)
      self.assertIsNone(p.status.acknowledged_side)

  def test_release_error_discards_queued_opposite_and_bounds_retries(self):
    p = self.planner()
    self.active(p)
    frames = list(p.request(Side.RIGHT, .6))
    frames.extend(self.reply(p, bytes.fromhex("7f2f22"), .7))
    frames.extend(self.reply(p, bytes.fromhex("7f2f22"), .8))
    frames.extend(self.reply(p, bytes.fromhex("7f2f22"), .9))
    frames.extend(self.reply(p, bytes.fromhex("7f2f22"), 1.0))
    self.assertEqual(len(frames), 4)
    self.assertTrue(all(f.operation is Operation.RELEASE for f in frames))
    self.assertEqual(p.status.phase, Phase.FAULT)
    self.assertEqual(p.status.uncertain_sides, frozenset(Side))

  def test_active_limit_from_command_time_including_before_ack(self):
    for acknowledge in (False, True):
      p = self.planner(active=.2)
      self.ready(p)
      p.request(Side.LEFT, 1.0)
      if acknowledge:
        self.reply(p, bytes.fromhex("6fbc1503"), 1.1)
      self.assertEqual(p.tick(1.2)[0].operation, Operation.RELEASE)
      self.assertEqual(p.status.fault, "maximum_active_duration")

  def test_restart_does_not_restore_ack_ownership_or_request(self):
    p = self.planner()
    self.active(p)
    restarted = self.planner()
    self.assertEqual(restarted.status.phase, Phase.UNKNOWN)
    self.assertIsNone(restarted.status.acknowledged_side)
    self.assertEqual(restarted.request(Side.LEFT, 0), ())
    self.assertEqual(restarted.status.uncertain_sides, frozenset(Side))

  def test_time_and_request_api_validation(self):
    p = self.planner()
    p.tick(1)
    for now in (0, float("nan"), float("inf"), -1, True):
      with self.subTest(now=now), self.assertRaises(ValueError):
        p.tick(now)
    with self.assertRaises(TypeError):
      p.request("left", 2)


if __name__ == "__main__":
  unittest.main()
