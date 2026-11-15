"""K5 ISG restart order: a standstill release waits for the ISG-stopped engine to restart.

Routes 6d1 10-24 18:35 and 734 11-15 19:11: with the engine ISG auto-stopped under an openpilot hold,
openpilot released (StopReq 1->0, CRUSE_INFO_SET_2 still 1) before requesting the restart, the ISG gave
up ~0.65 s later and demanded a manual restart. The stock SCC requests the restart while it still holds.

Every SCC_CONTROL packet below is packed and decoded through a CANParser round-trip (test_stopping.send /
the local packer helper), so the assertions read wire values, not the internal dict.
"""
from types import SimpleNamespace

import pytest

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car.hyundai import hyundaicanfd
from opendbc.car.hyundai.stopping import (CIS2_RESTART, ISG_BRAKE_WAIT_FRAMES, ISG_POST_FRAMES, STOP_LOWER_BAND,
                                          CanfdStopping, parse_isg_state, read_isg)
from opendbc.car.hyundai.tests.test_stopping import make_cs, send

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


# --- T3: a driver brake waits at most 1.0 s -------------------------------

@pytest.mark.parametrize("camera", [True, False])
def test_brake_release_waits_one_second_then_lets_go(camera, monkeypatch):
  events = []
  monkeypatch.setattr(hyundaicanfd.carlog, "warning", lambda message: events.append(message))
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True

  for frame in range(1, ISG_BRAKE_WAIT_FRAMES + 1):
    assert_hold(send(camera, controller, cs, enabled=False)), f"frame {frame} released during the brake wait"

  values = send(camera, controller, cs, enabled=False)
  assert values["StopReq"] == 0
  assert values["ACCMode"] == 0
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera), "a brake release gets no post phase"
  assert controller.isg_mode == "expired"

  for _ in range(10):
    assert send(camera, controller, cs, enabled=False)["StopReq"] == 0
  timeouts = isg_events(events, "isg_brake_timeout")
  assert len(timeouts) == 1, timeouts
  assert timeouts[0]["mode"] == "expired"


@pytest.mark.parametrize("camera", [True, False])
def test_brake_wait_ends_when_the_engine_restarts(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  for _ in range(10):
    assert_hold(send(camera, controller, cs, enabled=False))

  cs.isgAutoStop = False
  values = send(camera, controller, cs, enabled=False)
  assert values["StopReq"] == 0
  assert values["ACCMode"] == 0
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)
  assert controller.isg_mode == ""
  for _ in range(5):
    assert send(camera, controller, cs, enabled=False)["StopReq"] == 0
  assert controller.isg_mode == "", "a brake release must not open a post phase"


# --- T4: a brake during a departure wait restarts the clock ----------------

@pytest.mark.parametrize("camera", [True, False])
def test_brake_during_a_departure_wait_restarts_the_clock(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  for _ in range(20):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5))
  assert controller.isg_mode == "depart"

  cs.out.brakePressed = True
  for frame in range(1, ISG_BRAKE_WAIT_FRAMES + 1):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5)), f"frame {frame}"
  values = send(camera, controller, cs, stopping=False, accel=0.5)
  assert values["StopReq"] == 0
  assert values["CRUSE_INFO_SET_2"] == base_cis2(camera)
  assert controller.isg_mode == "expired"


# --- T5: the gas pedal keeps the plain release -----------------------------

@pytest.mark.parametrize("camera", [True, False])
def test_gas_pedal_keeps_the_plain_release(camera):
  live, reference = isg_cs(), isg_cs()
  a, b = CanfdStopping(), isg_off(CanfdStopping())
  for cs in (live, reference):
    cs.canfdSccHoldActive = True

  for _ in range(3):
    assert payload(camera, a, live, stopping=True, accel=-0.5) == \
           payload(camera, b, reference, stopping=True, accel=-0.5)
  for cs in (live, reference):
    cs.out.gasPressed = True
  for frame in range(5):
    assert payload(camera, a, live, stopping=False, accel=0.5, override=True) == \
           payload(camera, b, reference, stopping=False, accel=0.5, override=True), f"frame {frame}"
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


# --- T7: a long departure wait keeps holding ------------------------------

@pytest.mark.parametrize("camera", [True, False])
def test_long_departure_wait_holds_and_logs_once(camera, monkeypatch):
  events = []
  monkeypatch.setattr(hyundaicanfd.carlog, "warning", lambda message: events.append(message))
  cs = isg_cs()
  controller, _ = held(camera, cs)
  for frame in range(1, 401):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5)), f"frame {frame}"
  slow = isg_events(events, "isg_depart_slow")
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

GUARDS = {
  "nan_speed": dict(speed=float("nan")),
  "can_invalid": dict(canValid=False),
  "parking_brake": dict(parkingBrake=True),
  "reverse": dict(gearShifter="reverse"),
}


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("guard", list(GUARDS))
def test_guards_block_the_isg_override(camera, guard):
  def sequence(with_guard):
    cs = isg_cs(**GUARDS[guard]) if with_guard else isg_cs()
    if "speed" in GUARDS[guard] and with_guard:
      at_rest(cs, GUARDS[guard]["speed"])
    a, b = CanfdStopping(), isg_off(CanfdStopping())
    ticks = [dict(stopping=True, accel=-0.5)] * 2 + [dict(stopping=False, accel=0.5)] * 6
    left = [payload(camera, a, cs, **tick) for tick in ticks]
    right = [payload(camera, b, cs, **tick) for tick in ticks]
    return left, right, a.isg_mode

  live, off, mode = sequence(True)
  assert live == off, f"{guard}: the ISG order overrode a guarded frame"
  assert mode == "", guard

  live, off, mode = sequence(False)
  assert live != off, f"{guard}: the control sequence never fired"
  assert mode == "depart", guard


# --- T10: a brake timeout does not disarm the next stop --------------------

@pytest.mark.parametrize("camera", [True, False])
def test_expired_brake_wait_protects_a_new_hold(camera):
  cs = isg_cs()
  controller, _ = held(camera, cs)
  cs.out.brakePressed = True
  for _ in range(ISG_BRAKE_WAIT_FRAMES + 1):
    send(camera, controller, cs, enabled=False)
  assert controller.isg_mode == "expired"
  assert send(camera, controller, cs, enabled=False)["StopReq"] == 0

  cs.out.brakePressed = False
  for _ in range(12):
    assert send(camera, controller, cs)["StopReq"] == 1
  assert controller.isg_mode == "", "a new hold must clear the expired wait"

  for _ in range(10):
    assert_hold(send(camera, controller, cs, stopping=False, accel=0.5))
  assert controller.isg_mode == "depart"


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


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q"]))
