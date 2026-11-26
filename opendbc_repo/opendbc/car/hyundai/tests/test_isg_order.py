"""K5 ISG restart order: a standstill release waits for the ISG-stopped engine to restart.

Routes 6d1 10-24 18:35 and 734 11-15 19:11: with the engine ISG auto-stopped under an openpilot hold,
openpilot released (StopReq 1->0, CRUSE_INFO_SET_2 still 1) before requesting the restart, the ISG gave
up ~0.65 s later and demanded a manual restart. The stock SCC requests the restart while it still holds.

Every SCC_CONTROL packet below is packed and decoded through a CANParser round-trip (test_stopping.send /
the local packer helper), so the assertions read wire values, not the internal dict.
"""
import math
from types import SimpleNamespace

import pytest

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car.hyundai import hyundaicanfd
from opendbc.car.hyundai.stopping import (CIS2_RESTART, ISG_AVH_HANDOFF_FRAMES, ISG_POST_FRAMES, STOP_LOWER_BAND,
                                          CanfdStopping, parse_isg_state, read_isg)
from opendbc.car.hyundai.tests.test_stopping import make_cs, send
from opendbc.car.hyundai.values import CAR

CIS2_OFF = 1
ISG_AUTO_STOP = 8
ISG_RESTARTING = 16
ISG_FAULT = 48

# ISG_STATUS (0x0B5) frames from route 000006d1--37af09f1aa segment 4, can src 0.
# t=288.9968: byte 15 = 8 (auto-stopped). t=289.6086: byte 15 = 48, byte 18 = 64 (fault).
ISG_HEX_AUTO_STOP = "c84e7600007f0e000000000000002f085000000014000000000000cf00000000"
ISG_HEX_FAULT = "2cd5b400007f0e000000000000002f305000400014000000000000cf00000000"


def at_rest(cs, speed=0.0):
  cs.out.vEgo = speed
  cs.out.vEgoRaw = speed
  cs.out.wheelSpeeds = SimpleNamespace(fl=speed, fr=speed, rl=speed, rr=speed)
  return cs


def isg_cs(*, auto_stop=True, speed=0.0, **out):
  """make_cs carrying the OEM restart byte and the ISG state under test."""
  cs = make_cs()
  cs.scc_control["CRUSE_INFO_SET_2"] = CIS2_OFF
  cs.isgAutoStop = auto_stop
  cs.isgState = ISG_AUTO_STOP if auto_stop else 0
  at_rest(cs, speed)
  for name, value in out.items():
    setattr(cs.out, name, value)
  return cs


def isg_off(controller):
  """The same controller with the ISG order disabled: the pre-task behaviour."""
  controller.isg_restart_order = lambda values, **kwargs: ""
  return controller


def base_cis2(camera):
  """CRUSE_INFO_SET_2 without the ISG order: the OEM template's 1, unset (0) on the legacy packet path."""
  return CIS2_OFF if camera else 0


def payload(camera, controller, cs, *, enabled=True, override=False, stopping=True, accel=-0.5,
            previous_value=-0.5):
  """The packed SCC_CONTROL bytes for one control tick, on both create_acc_control paths."""
  packer = CANPacker("hyundai_canfd_generated")
  can = SimpleNamespace(ECAN=0)
  hud = SimpleNamespace(leadDistanceBars=2, leadVisible=False)
  jerk = SimpleNamespace(carrot_cruise=0, jerk_u=2.0, jerk_l=1.0)
  if camera:
    msg, _ = hyundaicanfd.create_acc_control_scc2(packer, can, enabled, previous_value, accel, stopping, override,
                                                  30.0, hud, jerk, cs, controller)
  else:
    msg, _ = hyundaicanfd.create_acc_control(packer, can, enabled, previous_value, accel, stopping, override, 30.0,
                                             hud, 2.0, 1.0, cs, controller, accel_value_last=previous_value)
  return None if msg is None else msg[1]


def decode(packet):
  """Decode packed SCC_CONTROL bytes through a CANParser round-trip."""
  if packet is None:
    return None
  parser = CANParser("hyundai_canfd_generated", [("SCC_CONTROL", 50)], 0)
  parser.update([1_000_000_000, [(416, packet, 0)]])
  return parser.vl["SCC_CONTROL"]


def held(camera, cs):
  """One confirmed hold packet under an auto-stopped engine."""
  controller = CanfdStopping()
  cs.canfdSccHoldActive = True
  values = send(camera, controller, cs)
  assert controller.phase == "held"
  assert values["StopReq"] == 1
  return controller, values


def assert_hold(values):
  """The stock SCC's restart packet: keep the hold and request the engine restart."""
  assert values["ACCMode"] == 1
  assert values["StopReq"] == 1
  assert values["aReqValue"] == pytest.approx(0.0, abs=0.0051)
  assert values["aReqRaw"] == pytest.approx(0.0, abs=0.0051)
  assert values["AccelLimitBandLower"] == pytest.approx(STOP_LOWER_BAND)
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART


def isg_events(events, name):
  return [e for e in events if e.get("event") == "carrot_isg" and e.get("isg") == name]


# --- T1: a departure waits until the engine leaves auto-stop ----------------

@pytest.mark.parametrize("camera", [True, False])
def test_departure_holds_until_the_engine_restarts(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)

  for frame in range(1, 101):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5)), f"frame {frame} released early"

  cs.isgAutoStop = False
  cs.isgState = ISG_RESTARTING
  values = send(camera, controller, cs, stopping=False, accel=0.5)
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART, "the restart request must outlive the release"
  assert controller.isg_mode == "post"

  for _ in range(ISG_POST_FRAMES - 1):
    values = send(camera, controller, cs, stopping=False, accel=0.5)
    assert values["StopReq"] == 0
    assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART

  values = send(camera, controller, cs, stopping=False, accel=0.5)
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera), "the restart request must stop after its window"
  assert controller.isg_mode == ""


@pytest.mark.parametrize("camera", [True, False])
def test_moving_right_after_the_restart_drops_the_request(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  send(camera, controller, cs, stopping=False, accel=0.5)
  cs.isgAutoStop = False
  send(camera, controller, cs, stopping=False, accel=0.5)
  assert controller.isg_mode == "post"

  at_rest(cs, 0.2)
  values = send(camera, controller, cs, stopping=False, accel=0.5)
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)
  assert controller.isg_mode == ""


# --- T13: the same departure on the legacy (non-camera) packet path ---------

def test_legacy_packet_path_holds_the_departure():
  cs = isg_cs()
  controller, _ = held(False, cs)
  for frame in range(1, 31):
    values = send(False, controller, cs, stopping=False, accel=0.5)
    assert_hold(values), f"frame {frame} released early"
  assert values["ZEROS_7"] == 0

  cs.isgAutoStop = False
  values = send(False, controller, cs, stopping=False, accel=0.5)
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"


# --- T2: without an auto-stopped engine nothing changes --------------------

@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("variant", ["flag_false", "attribute_deleted"])
def test_no_auto_stop_keeps_the_plain_release(camera, variant):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.isgAutoStop = False
  if variant == "attribute_deleted":
    del cs.isgAutoStop
    assert not hasattr(cs, "isgAutoStop")

  for frame in range(1, 11):
    values = send(camera, controller, cs, stopping=False, accel=0.5)
    assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)
    assert values["StopReq"] == (1 if frame < 10 else 0), f"frame {frame}"
  assert controller.isg_mode == ""


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("auto_stop", [False, True])
def test_packets_match_the_controller_without_the_isg_order(camera, auto_stop):
  live, reference = isg_cs(auto_stop=auto_stop), isg_cs(auto_stop=auto_stop)
  a, b = CanfdStopping(), isg_off(CanfdStopping())
  for cs in (live, reference):
    cs.canfdSccHoldActive = True

  plan = [dict(stopping=True, accel=-0.5)] * 5 + [dict(stopping=False, accel=0.5)] * 20
  for frame, tick in enumerate(plan):
    left = payload(camera, a, live, **tick)
    right = payload(camera, b, reference, **tick)
    if not auto_stop:
      assert left == right, f"frame {frame}: the ISG order changed the packet"
    elif frame < 5:
      # The hold itself is untouched: the override only starts with the release.
      assert left == right, f"frame {frame}: the ISG order changed a hold packet"
    else:
      assert left != right, f"frame {frame}: the ISG order never fired"


# --- T3: a driver brake holds until the engine restarts (no time limit) ----

@pytest.mark.parametrize("camera", [True, False])
def test_brake_wait_holds_until_the_engine_restarts(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True

  # Route 722 held 6.7 s under the brake; 500 frames is 10 s and must all hold.
  for frame in range(1, 501):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame} released while the engine was off"

  cs.out.brakePressed = False
  for frame in range(10):
    assert_hold(send(camera, controller, cs, enabled=False)), f"brake-release frame {frame}"
  assert controller.isg_mode == "brake"

  cs.isgAutoStop = False
  values = send(camera, controller, cs, enabled=False)
  assert values["ACCMode"] == 0, "the wait must hand the packet back"
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART, "the request must outlive the release"
  assert controller.isg_mode == "post"

  for _ in range(ISG_POST_FRAMES - 1):
    assert send(camera, controller, cs, enabled=False)["CRUSE_INFO_SET_2"] == CIS2_RESTART
  values = send(camera, controller, cs, enabled=False)
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)
  assert controller.isg_mode == ""


@pytest.mark.parametrize("camera", [True, False])
def test_gear_change_during_a_brake_wait_keeps_the_request(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  for _ in range(20):
    assert_hold(send(camera, controller, cs, enabled=False))

  cs.out.gearShifter = "neutral"
  values = send(camera, controller, cs, enabled=False)
  assert values["ACCMode"] == 0, "a gear change hands the packet back"
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"

  for frame in range(300):
    assert send(camera, controller, cs, enabled=False)["CRUSE_INFO_SET_2"] == CIS2_RESTART, f"frame {frame}"

  cs.isgAutoStop = False
  for frame in range(ISG_POST_FRAMES):
    assert send(camera, controller, cs, enabled=False)["CRUSE_INFO_SET_2"] == CIS2_RESTART, f"post frame {frame}"
  assert send(camera, controller, cs, enabled=False)["CRUSE_INFO_SET_2"] == base_cis2(camera)


# --- T4: a brake during a departure wait switches to the brake wait ---------

@pytest.mark.parametrize("camera", [True, False])
def test_brake_during_a_departure_wait_switches_and_keeps_holding(camera, monkeypatch):
  events = []
  monkeypatch.setattr(hyundaicanfd.carlog, "warning", lambda message: events.append(message))
  cs = isg_cs()
  controller, _ = held(camera, cs)
  for _ in range(20):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5))
  assert controller.isg_mode == "depart"

  cs.out.brakePressed = True
  for frame in range(1, 201):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5)), f"frame {frame}"
  assert controller.isg_mode == "brake"
  switches = isg_events(events, "isg_wait_brake")
  assert len(switches) == 1, switches


# --- T5: the gas pedal hands the packet back but keeps the request ---------

@pytest.mark.parametrize("camera", [True, False])
def test_gas_pedal_hands_back_but_keeps_the_request(camera):
  live, reference = isg_cs(), isg_cs()
  a, b = CanfdStopping(), isg_off(CanfdStopping())
  for cs in (live, reference):
    cs.canfdSccHoldActive = True

  # Hold, then a departure wait, so gas interrupts a running wait.
  for _ in range(3):
    assert payload(camera, a, live, stopping=True, accel=-0.5) == \
           payload(camera, b, reference, stopping=True, accel=-0.5)
  for _ in range(3):
    assert payload(camera, a, live, stopping=False, accel=0.5) != \
           payload(camera, b, reference, stopping=False, accel=0.5)
  assert a.isg_mode == "depart"

  for cs in (live, reference):
    cs.out.gasPressed = True

  live_values = decode(payload(camera, a, live, stopping=False, accel=0.5, override=True))
  base_values = decode(payload(camera, b, reference, stopping=False, accel=0.5, override=True))
  assert a.isg_mode == "post"
  assert live_values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  # Everything but the request byte (and the checksum it feeds) is openpilot's own packet.
  for key in ("CRUSE_INFO_SET_2", "CHECKSUM"):
    live_values.pop(key)
    base_values.pop(key)
  assert live_values == base_values

  at_rest(live, 0.2)
  at_rest(reference, 0.2)
  assert decode(payload(camera, a, live, stopping=False, accel=0.5, override=True))["CRUSE_INFO_SET_2"] == base_cis2(camera)
  assert a.isg_mode == ""


# --- T6: everything else is untouched -------------------------------------

@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("case", ["moving_release", "no_previous_hold", "no_acc"])
def test_untouched_sequences_are_byte_identical(camera, case):
  speed = 0.2 if case == "moving_release" else 0.0
  live, reference = isg_cs(speed=speed), isg_cs(speed=speed)
  a, b = CanfdStopping(), isg_off(CanfdStopping())
  for cs in (live, reference):
    cs.canfdSccHoldActive = case != "no_previous_hold"

  tick = dict(stopping=False, accel=0.5, enabled=case != "no_acc")
  for frame in range(12):
    assert payload(camera, a, live, **tick) == payload(camera, b, reference, **tick), f"{case} frame {frame}"
  assert a.isg_mode == "", case


# --- T7: a long wait keeps holding and logs once ---------------------------

@pytest.mark.parametrize("camera", [True, False])
def test_long_departure_wait_holds_and_logs_once(camera, monkeypatch):
  events = []
  monkeypatch.setattr(hyundaicanfd.carlog, "warning", lambda message: events.append(message))
  cs = isg_cs()
  controller, _ = held(camera, cs)
  for frame in range(1, 401):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5)), f"frame {frame}"
  slow = isg_events(events, "isg_wait_slow")
  assert len(slow) == 1, slow


@pytest.mark.parametrize("camera", [True, False])
def test_long_brake_wait_holds_and_logs_once(camera, monkeypatch):
  events = []
  monkeypatch.setattr(hyundaicanfd.carlog, "warning", lambda message: events.append(message))
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  for frame in range(1, 401):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame}"
  slow = isg_events(events, "isg_wait_slow")
  assert len(slow) == 1, slow


# --- T8: a re-stop during the wait keeps requesting the restart -------------

@pytest.mark.parametrize("camera", [True, False])
def test_restop_during_the_wait_keeps_requesting_the_restart(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  for _ in range(20):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5))

  for _ in range(5):
    assert_hold(send(camera, controller, cs))
  assert controller.isg_mode == "depart"

  cs.isgAutoStop = False
  values = send(camera, controller, cs)
  assert values["StopReq"] == 1
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  for _ in range(ISG_POST_FRAMES - 1):
    values = send(camera, controller, cs)
    assert values["StopReq"] == 1
    assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  values = send(camera, controller, cs)
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)


# --- T9: guards block the ISG override --------------------------------------

DOES_NOT_BLOCK = ("nan_speed", "can_invalid", "parking_brake")


def guarded_cs(**out):
  cs = isg_cs()
  for name, value in out.items():
    if name == "speed":
      at_rest(cs, value)
    else:
      setattr(cs.out, name, value)
  return cs


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("guard", DOES_NOT_BLOCK)
def test_guards_give_no_hold_and_no_post(camera, guard):
  """A sensor/CAN/parking-brake fault must not hold the car, and must not leave a post request behind."""
  out = {"nan_speed": dict(speed=float("nan")), "can_invalid": dict(canValid=False),
         "parking_brake": dict(parkingBrake=True)}[guard]

  for with_guard in (False, True):
    cs = guarded_cs(**out) if with_guard else isg_cs()
    a, b = CanfdStopping(), isg_off(CanfdStopping())
    ticks = [dict(stopping=True, accel=-0.5)] * 2 + [dict(stopping=False, accel=0.5)] * 6
    left = [payload(camera, a, cs, **tick) for tick in ticks]
    right = [payload(camera, b, cs, **tick) for tick in ticks]
    if not with_guard:
      assert left != right, f"{guard}: the control sequence never fired"
      assert a.isg_mode == "depart", guard
    else:
      assert left == right, f"{guard}: the ISG order overrode a guarded frame"
      assert a.isg_mode == "", guard


@pytest.mark.parametrize("camera", [True, False])
def test_reverse_without_a_wait_is_untouched(camera):
  live, reference = isg_cs(), isg_cs()
  for cs in (live, reference):
    cs.out.gearShifter = "reverse"
    cs.canfdSccHoldActive = True
  a, b = CanfdStopping(), isg_off(CanfdStopping())
  for frame in range(12):
    assert payload(camera, a, live, stopping=False, accel=0.5) == \
           payload(camera, b, reference, stopping=False, accel=0.5), f"frame {frame}"
  assert a.isg_mode == ""


# --- T11: the state parser --------------------------------------------------

@pytest.mark.parametrize("state,fresh,expected", [
  (None, True, (-1, False)),
  (float("nan"), True, (-1, False)),
  (float("inf"), True, (-1, False)),
  (8.0, False, (-1, False)),
  (ISG_AUTO_STOP, True, (ISG_AUTO_STOP, True)),
  (ISG_RESTARTING, True, (ISG_RESTARTING, False)),
  (ISG_FAULT, True, (ISG_FAULT, False)),
  (0.0, True, (0, False)),
])
def test_parse_isg_state(state, fresh, expected):
  assert parse_isg_state(state, fresh) == expected


# --- T12: the real payloads through a CANParser -----------------------------

def test_read_isg_on_the_logged_frames():
  parser = CANParser("hyundai_canfd_generated", [("ISG_STATUS", 0)], 0)
  auto = bytes.fromhex(ISG_HEX_AUTO_STOP)
  fault = bytes.fromhex(ISG_HEX_FAULT)
  assert (len(auto), len(fault)) == (32, 32)
  assert (auto[15], auto[18]) == (ISG_AUTO_STOP, 0)
  assert (fault[15], fault[18]) == (ISG_FAULT, 64)

  t = 289_000_000_000
  parser.update([t, [(181, auto, 0)]])
  assert parser.vl["ISG_STATUS"]["ISG_STATE"] == ISG_AUTO_STOP
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (ISG_AUTO_STOP, True)

  parser.update([t + 608_600_000, [(181, fault, 0)]])
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (ISG_FAULT, False)
  assert parser.vl["ISG_STATUS"]["ISG_FAULT"] == 64

  # 0.6 s of other traffic only: the ISG frame is now stale.
  parser.update([t + 1_208_600_000, [(96, bytes(24), 0)]])
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (-1, False)
  assert read_isg(parser, None) == (-1, False)


@pytest.mark.parametrize("camera", [True, False])
def test_oem_autohold_after_openpilot_isg_hold_still_requests_restart(camera):
  """OEM AutoHold latching on driver brake during an openpilot ISG hold must not drop StopReq before CIS2=2."""
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  cs.out.brakeHoldActive = True
  for frame in range(1, 21):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame} dropped StopReq while engine off"
  assert controller.isg_mode == "brake"

  cs.isgAutoStop = False
  values = send(camera, controller, cs, enabled=False)
  assert values["ACCMode"] == 0
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"


@pytest.mark.parametrize("camera", [True, False])
def test_autohold_alone_takes_over_after_a_brake_wait(camera):
  """Foot off the brake under OEM AutoHold after 0.2 s of a brake wait: the stop is handed to AutoHold,
  StopReq is released and the restart request stays (route 737 1055-1169 s, 113 s under auto-stop)."""
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  cs.out.brakeHoldActive = True
  for frame in range(1, 13):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame} dropped StopReq under the brake"
  assert controller.isg_mode == "brake"
  assert controller.isg_frames >= ISG_AVH_HANDOFF_FRAMES

  cs.out.brakePressed = False
  values = send(camera, controller, cs, enabled=False)
  assert values["StopReq"] == 0, "the first foot-off frame must hand the stop to AutoHold"
  assert values["ACCMode"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART, "the restart request must survive the hand-off"
  assert controller.isg_mode == "post"

  for frame in range(20):
    values = send(camera, controller, cs, enabled=False)
    assert values["StopReq"] == 0, f"frame {frame} re-entered the wait under AutoHold"
    assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART, f"frame {frame} dropped the restart request"
    assert controller.isg_mode == "post"

  # The engine restarts under AutoHold: the request outlives it by its own window, then the batch ends.
  cs.isgAutoStop = False
  for _ in range(ISG_POST_FRAMES):
    values = send(camera, controller, cs, enabled=False)
    assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
    assert values["StopReq"] == 0
  values = send(camera, controller, cs, enabled=False)
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)
  assert controller.isg_mode == ""


@pytest.mark.parametrize("camera", [True, False])
def test_autohold_handoff_waits_for_the_request(camera):
  """A foot-off before 0.2 s of brake wait keeps asking for the restart until the request has held that long."""
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  cs.out.brakeHoldActive = True
  for frame in range(1, 4):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame} dropped StopReq under the brake"
  assert controller.isg_mode == "brake"
  assert controller.isg_frames == 3

  cs.out.brakePressed = False
  for frame in range(1, ISG_AVH_HANDOFF_FRAMES - 3 + 1):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame} handed off before 0.2 s"
  assert controller.isg_frames == ISG_AVH_HANDOFF_FRAMES
  assert controller.isg_mode == "brake"

  values = send(camera, controller, cs, enabled=False)
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"


@pytest.mark.parametrize("camera", [True, False])
def test_autohold_with_foot_on_brake_keeps_the_hold(camera):
  """As long as the foot stays on the brake, the wait has no time limit: 4 s of brake + AutoHold still holds."""
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  cs.out.brakeHoldActive = True
  for frame in range(1, 201):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame} dropped StopReq under the brake"
  assert controller.isg_mode == "brake"


@pytest.mark.parametrize("camera", [True, False])
def test_oem_autohold_without_prior_openpilot_hold_is_untouched(camera):
  """If OEM AutoHold was already holding (no prior StopReq=1), ISG order must not override the packet."""
  live, reference = isg_cs(brakeHoldActive=True), isg_cs(brakeHoldActive=True)
  a, b = CanfdStopping(), isg_off(CanfdStopping())
  for frame in range(10):
    assert payload(camera, a, live, stopping=True, accel=-0.5) == \
           payload(camera, b, reference, stopping=True, accel=-0.5), f"frame {frame}"
  assert a.isg_mode == ""


@pytest.mark.parametrize("camera", [True, False])
def test_gear_out_of_drive_from_hold_matches_stock(camera):
  """Shifting out of drive during an openpilot ISG hold is a plain driver release, as on the stock SCC:
  no post window, and CRUSE_INFO_SET_2 stays at the packet's own value instead of being forced to 2."""
  cs = isg_cs()
  controller, _ = held(camera, cs)

  cs.out.gearShifter = "neutral"
  for frame in range(6):
    values = send(camera, controller, cs, enabled=False)
    assert values["StopReq"] == 0, f"frame {frame} still requested the hold"
    assert values["CRUSE_INFO_SET_2"] == base_cis2(camera), f"frame {frame} forced CIS2"
    assert controller.isg_mode == "", f"frame {frame} entered post"


def test_late_isg_status_registration_recovers_after_startup_window():
  """If 0x0B5 arrives after the initial startup frame 126 window, _discover_canfd_isg_status still registers it."""
  from opendbc.car.hyundai.carstate import CarState

  parser = CANParser("hyundai_canfd_generated", [("ESP_STATUS", 50)], 0)
  parser.controls_ready = True
  cs = CarState.__new__(CarState)
  cs.CP = SimpleNamespace(carFingerprint=CAR.KIA_K5_DL3_24)
  cs.isg_status = None
  cs.controls_ready_count = 200

  # Bus without 0x0B5 (e.g. Palisade/EV): must stay None and add no extra tracked address.
  t = 1_000_000_000
  parser.update([t, [(96, bytes(24), 0)]])
  cs._discover_canfd_isg_status(parser)
  assert cs.isg_status is None
  assert 181 not in parser.addresses

  # Once 0x0B5 appears on bus 0, discovery registers it; the first size-correct frame then binds it.
  auto = bytes.fromhex(ISG_HEX_AUTO_STOP)
  parser.update([t + 10_000_000, [(181, auto, 0)]])
  cs._discover_canfd_isg_status(parser)
  assert 181 in parser.addresses
  assert parser.message_states[181].ignore_alive is True
  # The frame that registered 0x0B5 arrived before it was tracked (ts == 0), so read_isg must report unknown.
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (-1, False)

  parser.update([t + 20_000_000, [(181, auto, 0)]])
  cs._discover_canfd_isg_status(parser)
  assert cs.isg_status is not None
  assert read_isg(parser, cs.isg_status) == (ISG_AUTO_STOP, True)


@pytest.mark.parametrize(("fingerprint", "size"), [
  (CAR.KIA_K5_DL3_24, 32),
  (CAR.KIA_EV6, 32),
  (CAR.KIA_K5_DL3_24, 24),
])
def test_isg_status_only_on_k5(fingerprint, size):
  """Only the K5 DL3 ICE has a verified ISG_STATUS layout: another car's 0x0B5, or a short payload, stays untracked."""
  from opendbc.car.hyundai.carstate import CarState

  parser = CANParser("hyundai_canfd_generated", [("ESP_STATUS", 50)], 0)
  parser.controls_ready = True
  cs = CarState.__new__(CarState)
  cs.CP = SimpleNamespace(carFingerprint=fingerprint)
  cs.isg_status = None
  cs.controls_ready_count = 200

  parser.update([1_000_000_000, [(181, bytes.fromhex(ISG_HEX_AUTO_STOP)[:size], 0)]])
  cs._discover_canfd_isg_status(parser)
  parser.update([1_010_000_000, [(181, bytes.fromhex(ISG_HEX_AUTO_STOP)[:size], 0)]])
  cs._discover_canfd_isg_status(parser)
  expected = fingerprint == CAR.KIA_K5_DL3_24 and size == 32
  assert (cs.isg_status is not None) == expected


def test_isg_status_dropout_does_not_invalidate_can_valid():
  """A >100 ms 0x0B5 dropout (e.g. cranking brownout) must not trip can_valid; read_isg handles staleness at 500 ms."""
  from opendbc.car.hyundai.carstate import CarState

  # Use a 50 Hz message without checksum/counter (CRUISE_BUTTONS_ALT2 or valid ESP_STATUS via packer) to keep bus alive.
  packer = CANPacker("hyundai_canfd_generated")
  parser = CANParser("hyundai_canfd_generated", [("ESP_STATUS", 50)], 0)
  parser.controls_ready = True
  cs = CarState.__new__(CarState)
  cs.CP = SimpleNamespace(carFingerprint=CAR.KIA_K5_DL3_24)
  cs.isg_status = None
  cs.controls_ready_count = 126

  auto = bytes.fromhex(ISG_HEX_AUTO_STOP)
  t = 10_000_000_000
  esp_frame = packer.make_can_msg("ESP_STATUS", 0, {})[1]
  parser.update([t, [(96, esp_frame, 0), (181, auto, 0)]])
  cs._discover_canfd_isg_status(parser)
  assert parser.message_states[181].ignore_alive is True
  # The registering frame predates tracking: bind on the next size-correct 0x0B5 frame.
  parser.update([t + 5_000_000, [(181, auto, 0)]])
  cs._discover_canfd_isg_status(parser)
  assert cs.isg_status is not None

  # 1.2 s of 100 Hz 0x0B5 + 50 Hz valid ESP_STATUS
  for i in range(1, 121):
    t += 10_000_000
    frames = [(181, auto, 0)]
    if i % 2 == 0:
      frames.append((96, packer.make_can_msg("ESP_STATUS", 0, {})[1], 0))
    parser.update([t, frames])
  assert parser.can_valid is True
  assert read_isg(parser, cs.isg_status) == (ISG_AUTO_STOP, True)

  # Drop ONLY 0x0B5 for 600 ms while ESP_STATUS stays healthy: can_valid must stay True throughout,
  # while read_isg stays fresh <= 500 ms (relative to bus traffic) and cleanly falls back to (-1, False) past 500 ms.
  ts_last_isg = parser.ts_nanos["ISG_STATUS"]["ISG_STATE"]
  for i in range(1, 61):
    t += 10_000_000
    frames = [(96, packer.make_can_msg("ESP_STATUS", 0, {})[1], 0)] if i % 2 == 0 else []
    parser.update([t, frames])
    assert parser.can_valid is True, f"0x0B5 dropout tripped can_valid at {i * 10} ms"
    if parser._last_update_nanos - ts_last_isg <= 500_000_000:
      assert read_isg(parser, cs.isg_status) == (ISG_AUTO_STOP, True)
    else:
      assert read_isg(parser, cs.isg_status) == (-1, False)
  assert read_isg(parser, cs.isg_status) == (-1, False)


def test_monitor_fingerprint_frame_126_registers_isg_status_with_ignore_alive(monkeypatch):
  """Initial registration at controls_ready_count == 126 must also pass freq=nan so 0x0B5 never trips can_valid."""
  from opendbc.car import Bus
  from opendbc.car.hyundai import carstate as carstate_mod
  from opendbc.car.hyundai.carstate import CarState
  from opendbc.car.hyundai.values import HyundaiFlags

  class DummyParams:
    def get_bool(self, key):
      return key == "ControlsReady"

  monkeypatch.setattr(carstate_mod, "Params", DummyParams)

  pt_parser = CANParser("hyundai_canfd_generated", [("ESP_STATUS", 50)], 0)
  cam_parser = CANParser("hyundai_canfd_generated", [], 2)
  pt_parser.controls_ready = True
  pt_parser.update([1_000_000_000, [(181, bytes.fromhex(ISG_HEX_AUTO_STOP), 0)]])

  cs = CarState.__new__(CarState)
  cs.CP = SimpleNamespace(carFingerprint=CAR.KIA_K5_DL3_24, flags=HyundaiFlags.CANFD.value,
                          openpilotLongitudinalControl=True)
  cs.controls_ready_count = 125
  cs.isg_status = None
  cs.cruise_buttons_alt2 = None
  cs.trailer_status = None
  cs._update_camera_scc_hint = lambda *args, **kwargs: None
  cs._discover_canfd_camera_tx_messages = lambda *args, **kwargs: None

  cs.monitor_fingerprint({Bus.pt: pt_parser, Bus.cam: cam_parser}, canfd=True)
  assert cs.controls_ready_count == 126
  assert pt_parser.message_states[181].ignore_alive is True
  assert pt_parser.message_states[181].ignore_counter is True

  # The 126 registration lands after the frame it must read: the first size-correct 0x0B5 frame binds it.
  pt_parser.update([1_010_000_000, [(181, bytes.fromhex(ISG_HEX_AUTO_STOP), 0)]])
  cs._discover_canfd_isg_status(pt_parser)
  assert cs.isg_status is pt_parser.vl["ISG_STATUS"]


def test_read_isg_before_first_frame_is_unknown_even_at_low_timestamp():
  """ts == 0 (registered, never parsed) must read (-1, False) even when last_nonempty_nanos <= ISG_STALE_NS."""
  parser = CANParser("hyundai_canfd_generated", [("ESP_STATUS", 50)], 0)
  parser._add_message("ISG_STATUS", float("nan"), ignore_counter=True)
  parser.update([100_000_000, [(96, bytes(24), 0)]])
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (-1, False)


# --- W-ISG8: gas straight from a hold, non-finite planner values, a silent bus, soft-hold releases ---

@pytest.mark.parametrize("camera", [True, False])
def test_gas_straight_from_a_hold_keeps_the_request(camera):
  """A pedal press straight from a plain hold is a release like any other: it carries the restart request."""
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.gasPressed = True
  live = send(camera, controller, cs, stopping=False, accel=0.5, override=True)

  ref_cs = isg_cs()
  ref_controller, _ = held(camera, ref_cs)
  ref_cs.out.gasPressed = True
  reference = send(camera, isg_off(ref_controller), ref_cs, stopping=False, accel=0.5, override=True)

  assert live["StopReq"] == 0
  assert live["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"
  # Everything but the request byte (and the checksum it feeds) is openpilot's own packet.
  for key in ("CRUSE_INFO_SET_2", "CHECKSUM"):
    live.pop(key)
    reference.pop(key)
  assert live == reference

  at_rest(cs, 0.2)
  values = send(camera, controller, cs, stopping=False, accel=0.5, override=True)
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)
  assert controller.isg_mode == ""


@pytest.mark.parametrize("camera", [True, False])
def test_gas_from_a_hold_without_auto_stop_is_untouched(camera):
  """Without an auto-stopped engine the pedal press keeps the plain release packet, byte for byte."""
  cs = isg_cs(auto_stop=False)
  controller, _ = held(camera, cs)
  cs.out.gasPressed = True
  live = send(camera, controller, cs, stopping=False, accel=0.5, override=True)

  ref_cs = isg_cs(auto_stop=False)
  ref_controller, _ = held(camera, ref_cs)
  ref_cs.out.gasPressed = True
  reference = send(camera, isg_off(ref_controller), ref_cs, stopping=False, accel=0.5, override=True)

  assert live == reference
  assert controller.isg_mode == ""


@pytest.mark.parametrize("camera", [True, False])
def test_gas_from_a_soft_hold_keeps_the_request(camera):
  """Gas released from an armed soft hold (openpilot not enabled) also carries the restart request."""
  cs = isg_cs()
  cs.softHoldActive = 2
  cs.canfdSccHoldActive = True
  controller = CanfdStopping()
  for _ in range(3):
    values = send(camera, controller, cs, enabled=False, stopping=False, accel=0.0)
  assert values["StopReq"] == 1

  cs.softHoldActive = 0
  cs.out.gasPressed = True
  values = send(camera, controller, cs, enabled=False, stopping=False, accel=0.0)
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("what", ["accel", "aEgo", "previous_value"])
def test_nonfinite_planner_value_keeps_the_isg_hold(camera, what):
  """A NaN in the planner accel chain must not drop the guard while the wheels read a finite standstill."""
  cs = isg_cs()
  controller, _ = held(camera, cs)
  tick = dict(stopping=True, accel=-0.5)
  if what == "accel":
    tick["accel"] = float("nan")
  elif what == "aEgo":
    cs.out.aEgo = float("nan")
  else:
    tick["previous_value"] = float("nan")

  values = send(camera, controller, cs, **tick)
  assert_hold(values)
  assert all(math.isfinite(v) for v in values.values() if isinstance(v, float))
  assert controller.isg_mode == "brake"

  cs.isgAutoStop = False
  values = send(camera, controller, cs, **tick)
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"


def test_read_isg_goes_stale_when_the_whole_bus_is_silent():
  """Freshness follows the update clock: a silent bus cannot keep a forgotten ISG frame alive."""
  parser = CANParser("hyundai_canfd_generated", [("ESP_STATUS", 50)], 0)
  parser._add_message("ISG_STATUS", float("nan"), ignore_counter=True)
  t = 10_000_000_000
  parser.update([t, [(181, bytes.fromhex(ISG_HEX_AUTO_STOP), 0)]])
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (ISG_AUTO_STOP, True)
  parser.update([t + 400_000_000, []])
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (ISG_AUTO_STOP, True)
  parser.update([t + 600_000_000, []])
  assert read_isg(parser, parser.vl["ISG_STATUS"]) == (-1, False)


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("case", ["brake_tap", "cancel", "res_depart"])
def test_soft_hold_release_waits_for_the_restart(camera, case):
  """Leaving a soft hold waits for the engine restart before the packet hands back."""
  cs = isg_cs()
  cs.softHoldActive = 2
  cs.canfdSccHoldActive = True
  controller = CanfdStopping()
  tick = dict(enabled=False, stopping=False, accel=0.0)
  for _ in range(3):
    values = send(camera, controller, cs, **tick)
  assert values["StopReq"] == 1

  if case == "brake_tap":
    cs.out.brakePressed = True
    send(camera, controller, cs, **tick)
    cs.softHoldActive = 0
    for _ in range(5):
      values = send(camera, controller, cs, **tick)
  elif case == "cancel":
    cs.softHoldActive = 0
    for _ in range(5):
      values = send(camera, controller, cs, **tick)
  else:
    cs.softHoldActive = 0
    tick = dict(enabled=True, stopping=False, accel=0.5)
    for _ in range(12):
      values = send(camera, controller, cs, **tick)

  assert_hold(values)
  assert controller.isg_mode == ("depart" if case == "res_depart" else "brake")

  cs.isgAutoStop = False
  values = send(camera, controller, cs, **tick)
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"


@pytest.mark.parametrize("camera", [True, False])
def test_cancel_from_a_hold_waits_for_the_restart(camera):
  """Cancelling openpilot from a plain hold holds the standing car until the engine restarts."""
  cs = isg_cs()
  controller, _ = held(camera, cs)
  for frame in range(1, 11):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame}"
  assert controller.isg_mode == "brake"

  cs.isgAutoStop = False
  values = send(camera, controller, cs, enabled=False)
  assert values["ACCMode"] == 0
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == CIS2_RESTART
  assert controller.isg_mode == "post"


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("kind", ["stall", "surge"])
def test_rescue_under_auto_stop_keeps_the_hold(camera, kind):
  """A stall or surge rescue under an auto-stopped engine keeps StopReq and the restart request."""
  cs = isg_cs(speed=0.0)
  cs.canfdSccHoldActive = False
  controller = CanfdStopping()
  for _ in range(20):
    send(camera, controller, cs, stopping=True, accel=-0.5)
  if kind == "stall":
    at_rest(cs, 0.05)
    frames = 70
  else:
    at_rest(cs, 0.08)
    cs.out.aEgo = 0.2
    frames = 3

  rescue = None
  for _ in range(frames):
    values = send(camera, controller, cs, stopping=True, accel=-0.5)
    if controller.phase == "rescue":
      rescue = values
      break
  assert rescue is not None, f"{kind}: no rescue frame"
  assert_hold(rescue)
  assert controller.isg_mode == "depart"


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q"]))
