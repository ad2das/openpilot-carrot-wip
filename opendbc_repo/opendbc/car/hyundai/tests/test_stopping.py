import importlib.util
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.hyundai import carcontroller
from opendbc.car.hyundai.hyundaicanfd import create_acc_control, create_acc_control_scc2
from opendbc.car.hyundai.stopping import (CanfdStopping, StopPhase, DT, ENTRY_SPEED, RELEASE_CONFIRM_FRAMES, RESCUE_ACCEL,
                                          RESCUE_ACCEL_MIN, RESCUE_JERK, RESCUE_MAX_CYCLES, RESCUE_SPEED, STALL_RESCUE_MAX_T,
                                          STALL_T, STALL_ZERO_FRAMES, STOP_LOWER_BAND, STOP_SPEED, SURGE_ACCEL, SURGE_RISE,
                                          ZERO_CONFIRM_FRAMES)
from opendbc.car.hyundai.values import CAR, HyundaiFlags


def make_car_controller(monkeypatch, *, canfd=True, longitudinal=True):
  def get_bool(key):
    assert key != "CanfdStopRetry", "Removed stop retry setting must never be read"
    return False
  params = SimpleNamespace(get_bool=get_bool, get_int=lambda key: 0)
  monkeypatch.setattr(carcontroller, "Params", lambda: params)
  cp = structs.CarParams(carFingerprint=CAR.KIA_EV6, flags=int(HyundaiFlags.CANFD) if canfd else 0,
                         openpilotLongitudinalControl=longitudinal, stoppingDecelRate=0.8)
  return carcontroller.CarController({Bus.pt: "hyundai_canfd_generated"}, cp)


@pytest.mark.parametrize("canfd", [False, True])
@pytest.mark.parametrize("longitudinal", [False, True])
def test_controller_enables_canfd_stopping_by_vehicle_scope(monkeypatch, canfd, longitudinal):
  controller = make_car_controller(monkeypatch, canfd=canfd, longitudinal=longitudinal)
  assert isinstance(controller.canfd_stopping, CanfdStopping) == (canfd and longitudinal)


def step(controller, **overrides):
  args = dict(active=True, requested=True, speed=0.3, held=False, accel=-0.5,
              previous_value=-0.5, jerk_u=2.0, jerk_l=1.0)
  args.update(overrides)
  return controller.update(**args)


def test_normal_stop_keeps_zero_request_without_toggle():
  controller = CanfdStopping()
  for tick in range(150):
    cmd = step(controller, speed=max(0, 0.6 - tick * DT * 0.4), held=tick >= 80)
    assert (cmd.stop_req, cmd.raw, cmd.value, cmd.lower) == (1, 0., 0., STOP_LOWER_BAND)
  assert controller.phase == StopPhase.held


@pytest.mark.parametrize('speed', [.07, .3, .65])
def test_persistent_creep_never_releases_request(speed):
  controller = CanfdStopping()
  for _ in range(1500):  # exceed every old timeout, distance and retry limit
    cmd = step(controller, speed=speed)
    assert (cmd.stop_req, cmd.raw, cmd.value) == (1, 0., 0.)
  assert controller.phase == StopPhase.request


def test_motion_after_hold_changes_diagnostic_without_releasing():
  controller = CanfdStopping()
  step(controller, speed=.02, held=True)
  assert controller.phase == StopPhase.held
  for tick in range(300):
    cmd = step(controller, speed=.2 + tick * .005, held=True)
    assert (cmd.stop_req, cmd.raw, cmd.value) == (1, 0., 0.)
  assert controller.phase == StopPhase.request
  assert controller.reason == 'motion_after_hold'


def test_hold_flag_dropout_at_zero_does_not_release():
  controller = CanfdStopping()
  for _ in range(50):
    step(controller, speed=.02, held=True)
  for _ in range(50):
    assert step(controller, speed=.02, held=False).stop_req == 1
  assert controller.phase == StopPhase.held


def test_approach_preserves_braking_until_first_entry_then_zeros_immediately():
  controller = CanfdStopping()
  previous = -.3
  for _ in range(20):
    cmd = step(controller, speed=4., accel=-2., previous_value=previous)
    assert cmd.stop_req == 0
    assert cmd.raw == -2.
    assert cmd.value == pytest.approx(previous - DT)
    previous = cmd.value
  cmd = step(controller, speed=ENTRY_SPEED, accel=-2., previous_value=previous)
  assert (cmd.stop_req, cmd.raw, cmd.value) == (1, 0., 0.)
  for _ in range(100):
    assert step(controller, speed=ENTRY_SPEED + .3).stop_req == 1


@pytest.mark.parametrize('reason', ['inactive', 'departing'])
def test_cancel_resets_episode_and_restores_approach_gate(reason):
  controller = CanfdStopping()
  step(controller)
  assert step(controller, active=reason != 'inactive', requested=reason != 'departing') is None
  assert controller.phase == StopPhase.idle
  assert step(controller, speed=1.).stop_req == 0
  assert step(controller).stop_req == 1


def make_cs():
  return SimpleNamespace(
    scc_control={"ACC_ObjRelSpd": 0.0, "InfoDisplay": 5, "ZEROS_7": 1, "AccelLimitBandLower": 1.26},
    canfdSccHoldActive=False, softHoldActive=0, paddle_button_prev=0,
    out=SimpleNamespace(aEgo=0.0, vEgo=0.3, vEgoRaw=0.3,
                        wheelSpeeds=SimpleNamespace(fl=0.3, fr=0.3, rl=0.3, rr=0.3),
                        canValid=True, gearShifter="drive", brakePressed=False, gasPressed=False,
                        brakeHoldActive=False, parkingBrake=False, cruiseState=SimpleNamespace(available=True, standstill=True)),
  )


def send(camera, controller, CS, *, enabled=True, override=False, stopping=True, accel=-0.5,
         previous_value=-0.5, previous_target=-0.5, jerk_upper=2.0):
  packer = CANPacker("hyundai_canfd_generated")
  can = SimpleNamespace(ECAN=0)
  hud = SimpleNamespace(leadDistanceBars=2, leadVisible=False)
  jerk = SimpleNamespace(carrot_cruise=0, jerk_u=jerk_upper, jerk_l=1.0)
  if camera:
    msg, actual_value = create_acc_control_scc2(packer, can, enabled, previous_value, accel, stopping, override, 30.0, hud, jerk, CS, controller)
  else:
    msg, actual_value = create_acc_control(packer, can, enabled, previous_target, accel, stopping, override, 30.0, hud, jerk_upper, 1.0, CS,
                                         controller, accel_value_last=previous_value)
  if msg is None:
    return None
  parser = CANParser("hyundai_canfd_generated", [("SCC_CONTROL", 50)], 0)
  parser.update([1_000_000_000, [msg]])
  values = parser.vl["SCC_CONTROL"]
  assert values["aReqValue"] == pytest.approx(actual_value, abs=0.0051)  # CAN resolution is 0.01 m/s^2
  # Independent little-endian positions from the supplied OEM SCC definition.
  packed = int.from_bytes(msg[1], "little")
  if controller is not None:
    assert msg[1][7] == 0
    assert values["InfoDisplay"] == 0
  assert packed >> 74 & 7 == values["InfoDisplay"]
  assert packed >> 184 & 3 == values["StopReq"]
  assert ((packed >> 152) & 127) * 0.1 == pytest.approx(values["JerkUpperLimit"])
  assert (packed >> 176 & 63) * 0.02 == pytest.approx(values["AccelLimitBandLower"])
  return values


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("initial", [-1., -.3, .3])
@pytest.mark.parametrize("target", [-.5, -.8, -1.])
def test_packed_stop_zeros_both_requests_on_first_frame_and_keeps_zero(camera, initial, target):
  cs = make_cs()
  controller = CanfdStopping()
  previous = initial
  for tick in range(200):
    # Rebound beyond the entry threshold must not create another release.
    speed = .3 if tick < 40 else 1.2
    cs.out.vEgo = cs.out.vEgoRaw = speed
    cs.out.wheelSpeeds = SimpleNamespace(fl=speed, fr=speed, rl=speed, rr=speed)
    cs.out.aEgo = -.3 if tick < 40 else .5
    values = send(camera, controller, cs, accel=target, previous_value=previous, previous_target=target)
    assert values['StopReq'] == 1
    assert values['aReqRaw'] == values['aReqValue'] == 0.
    previous = values['aReqValue']


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("stock_info", [0, 4, 5])
def test_base_packet_builder_without_controller_preserves_legacy_fields(camera, stock_info):
  cs = make_cs()
  cs.scc_control["InfoDisplay"] = stock_info
  # Direct use of the base packet builder does not create a retry controller.
  for _ in range(200):
    values = send(camera, None, cs)
    assert values["StopReq"] == 1
    assert values["aReqRaw"] == pytest.approx(-0.5)
    assert values["aReqValue"] == pytest.approx(-0.5)
    assert values["AccelLimitBandLower"] == pytest.approx(0)
    assert values["InfoDisplay"] == (5 if camera and stock_info == 5 else 4)
    assert values["ZEROS_7"] == (1 if camera else 0)


@pytest.mark.parametrize("camera", [True, False])
def test_default_controller_keeps_stop_request_during_persistent_motion(camera, monkeypatch):
  controller = make_car_controller(monkeypatch)
  cs = make_cs()
  values = send(camera, controller.canfd_stopping, cs)
  assert values["StopReq"] == 1
  assert values["aReqRaw"] == pytest.approx(0.)
  assert values["InfoDisplay"] == values["ZEROS_7"] == 0
  assert values["AccelLimitBandLower"] == pytest.approx(STOP_LOWER_BAND)
  initial = controller.canfd_stopping
  for _ in range(170):
    send(camera, controller.canfd_stopping, cs)
  assert controller.canfd_stopping is initial
  assert initial.phase == StopPhase.request


@pytest.mark.parametrize("camera", [True, False])
def test_packed_scc_stop_acceleration_zero_display_fields_and_fixed_band(camera):
  controller = CanfdStopping()
  values = send(camera, controller, make_cs())
  assert values["ACCMode"] == 1
  assert values["StopReq"] == 1
  assert values["aReqRaw"] == pytest.approx(0.)
  assert values["aReqValue"] == pytest.approx(0.)
  assert values["InfoDisplay"] == values["ZEROS_7"] == 0
  assert values["AccelLimitBandLower"] == pytest.approx(STOP_LOWER_BAND)


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("a_ego", [float('nan'), float('inf')])
def test_invalid_measured_acceleration_blocks_stop_commands(camera, a_ego):
  cs = make_cs()
  cs.out.aEgo = a_ego
  controller = CanfdStopping()
  values = send(camera, controller, cs)
  assert values['ACCMode'] == values['StopReq'] == 0
  assert values['aReqRaw'] == values['aReqValue'] == 0.
  assert controller.phase == StopPhase.idle


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("block", ["brakePressed", "gasPressed", "brakeHoldActive", "parkingBrake", "canValid", "gearShifter",
                                   "unavailable", "disabled", "override", "nan_speed"])
def test_interlocks_and_motion_sources(camera, block):
  controller = CanfdStopping()
  cs = make_cs()
  for _ in range(40):
    send(camera, controller, cs)
  assert controller.phase == StopPhase.request
  if block in ("brakePressed", "gasPressed", "brakeHoldActive", "parkingBrake"):
    setattr(cs.out, block, True)
  elif block == "canValid":
    cs.out.canValid = False
  elif block == "gearShifter":
    cs.out.gearShifter = "reverse"
  elif block == "unavailable":
    cs.out.cruiseState.available = False
  elif block == "nan_speed":
    cs.out.vEgo = float("nan")
  values = send(camera, controller, cs, enabled=block != "disabled", override=block == "override")
  assert values["StopReq"] == 0
  assert values["aReqRaw"] == pytest.approx(0)
  assert values["aReqValue"] == pytest.approx(0)
  assert controller.phase == StopPhase.idle


@pytest.mark.parametrize("camera", [True, False])
def test_wheel_motion_overrides_zero_ego_for_initial_entry_but_never_retries(camera):
  controller = CanfdStopping()
  cs = make_cs()
  cs.out.vEgo = cs.out.vEgoRaw = 0.
  cs.canfdSccHoldActive = True
  cs.out.wheelSpeeds.fl = 1.
  assert send(camera, controller, cs)['StopReq'] == 0
  cs.out.wheelSpeeds.fl = .3
  assert send(camera, controller, cs)['StopReq'] == 1
  cs.out.wheelSpeeds.fl = 1.
  assert send(camera, controller, cs)['StopReq'] == 1
  assert controller.phase == StopPhase.request


@pytest.mark.parametrize("camera", [True, False])
def test_departure_releases_and_resumes_normal_acceleration_limiter(camera):
  controller = CanfdStopping()
  cs = make_cs()
  stopped = send(camera, controller, cs)
  assert stopped['aReqValue'] == 0.
  actual = dict(send(camera, controller, cs, stopping=False, accel=.5, previous_value=0., previous_target=0.))
  expected = dict(send(camera, None, cs, stopping=False, accel=.5, previous_value=0., previous_target=0.))
  assert actual['StopReq'] == expected['StopReq'] == 0
  assert actual['aReqRaw'] == expected['aReqRaw'] == .5
  assert actual['aReqValue'] == expected['aReqValue']
  assert actual['aReqValue'] == pytest.approx(.04 if camera else .1)
  assert controller.phase == StopPhase.idle


def test_missing_stock_message_resets_camera_experiment():
  cs = make_cs()
  controller = CanfdStopping()
  for _ in range(40):
    send(True, controller, cs)
  cs.scc_control = None
  assert send(True, controller, cs) is None
  assert controller.phase == StopPhase.idle


@pytest.mark.parametrize("camera", [True, False])
@pytest.mark.parametrize("brake_pressed", [False, True])
def test_soft_hold_zeros_requests_without_preparation(camera, brake_pressed):
  cs = make_cs()
  cs.softHoldActive = 1 if brake_pressed else 2
  cs.out.brakePressed = brake_pressed
  cs.out.vEgo = cs.out.vEgoRaw = 0.
  cs.out.wheelSpeeds = SimpleNamespace(fl=0., fr=0., rl=0., rr=0.)
  controller = CanfdStopping()
  previous = previous_target = 0.
  for _ in range(110):
    values = send(camera, controller, cs, enabled=False, stopping=False, accel=-2.,
                  previous_value=previous, previous_target=previous_target)
    assert values['ACCMode'] == values['StopReq'] == 1
    assert values['aReqRaw'] == 0.
    assert values['aReqValue'] == 0.
    previous, previous_target = values['aReqValue'], -2.
  assert controller.phase == StopPhase.held
  assert previous == 0.


@pytest.mark.parametrize('camera', [True, False])
@pytest.mark.parametrize('block', ['gasPressed', 'brakeHoldActive', 'parkingBrake', 'canValid', 'gearShifter', 'moving'])
def test_soft_hold_brake_exception_preserves_other_interlocks(camera, block):
  cs = make_cs()
  cs.softHoldActive = 1
  cs.out.brakePressed = True
  cs.out.vEgo = cs.out.vEgoRaw = 0.0
  cs.out.wheelSpeeds = SimpleNamespace(fl=0., fr=0., rl=0., rr=0.)
  controller = CanfdStopping()
  send(camera, controller, cs, enabled=False, stopping=False, accel=0.)
  assert controller.phase == StopPhase.request
  if block == 'canValid':
    cs.out.canValid = False
  elif block == 'gearShifter':
    cs.out.gearShifter = 'reverse'
  elif block == 'moving':
    cs.out.wheelSpeeds.fl = .11
  else:
    setattr(cs.out, block, True)
  values = send(camera, controller, cs, enabled=False, stopping=False, accel=0.)
  assert values['StopReq'] == 0
  assert values['aReqRaw'] == values['aReqValue'] == 0.
  assert controller.phase == StopPhase.idle


def test_confirmed_existing_hold_is_not_released():
  controller = CanfdStopping()
  command = step(controller, speed=0., held=True)
  assert command.stop_req == 1
  assert controller.phase == StopPhase.held


def _held_controller():
  controller = CanfdStopping()
  for _ in range(30):
    step(controller, speed=0., held=True)
  assert controller.phase == StopPhase.held
  return controller


def test_release_waits_for_confirm_time():
  controller = _held_controller()
  for _ in range(RELEASE_CONFIRM_FRAMES - 1):
    cmd = step(controller, speed=0., held=True, requested=False)
    assert (cmd.stop_req, cmd.raw, cmd.value, cmd.lower) == (1, 0., 0., STOP_LOWER_BAND)
    assert controller.phase == StopPhase.held
  assert controller.release_frames == RELEASE_CONFIRM_FRAMES - 1
  assert step(controller, speed=0., held=True, requested=False) is None
  assert controller.phase == StopPhase.idle


def test_flicker_never_drops_stop_req():
  controller = _held_controller()
  for requested in [False] * 9 + [True] * 2 + [False] * 9 + [True] + [False] * 9:
    cmd = step(controller, speed=0., held=True, requested=requested)
    assert cmd.stop_req == 1
  assert controller.phase == StopPhase.held


def test_inactive_releases_immediately():
  controller = _held_controller()
  assert step(controller, speed=0., held=True, active=False) is None
  assert controller.phase == StopPhase.idle


def test_moving_release_not_delayed():
  controller = CanfdStopping()
  step(controller, speed=0.5)
  assert controller.phase == StopPhase.request
  assert step(controller, speed=0.5, requested=False) is None
  assert controller.phase == StopPhase.idle


def test_approach_release_not_delayed():
  controller = CanfdStopping()
  step(controller, speed=1.0)
  assert controller.phase == StopPhase.approach
  assert step(controller, speed=1.0, requested=False) is None
  assert controller.phase == StopPhase.idle


@pytest.mark.parametrize('camera', [True, False])
@pytest.mark.parametrize('previous_upper', [.5, 1., 2., 5.])
@pytest.mark.parametrize('soft_hold', [False, True])
def test_stop_upper_increase_waits_one_packet_without_delaying_stop(camera, previous_upper, soft_hold):
  cs = make_cs()
  controller = CanfdStopping()
  first = send(camera, controller, cs, stopping=False, jerk_upper=previous_upper)
  assert first['StopReq'] == 0
  assert first['JerkUpperLimit'] == previous_upper
  if soft_hold:
    cs.softHoldActive = 2
    cs.out.vEgo = cs.out.vEgoRaw = 0.
    cs.out.wheelSpeeds = SimpleNamespace(fl=0., fr=0., rl=0., rr=0.)
  first = send(camera, controller, cs, stopping=not soft_hold, jerk_upper=2.)
  second = send(camera, controller, cs, stopping=not soft_hold, jerk_upper=2.)
  assert first['StopReq'] == second['StopReq'] == 1
  assert first['JerkUpperLimit'] == min(previous_upper, 2.)
  assert second['JerkUpperLimit'] == 2.
  assert first['aReqRaw'] == second['aReqRaw'] == 0.
  assert first['aReqValue'] == second['aReqValue'] == 0.
  assert first['AccelLimitBandLower'] == second['AccelLimitBandLower'] == STOP_LOWER_BAND


@pytest.mark.parametrize('camera', [True, False])
def test_initial_stop_without_previous_packet_uses_upper_one(camera):
  controller = CanfdStopping()
  cs = make_cs()
  assert send(camera, controller, cs)['JerkUpperLimit'] == 1.
  assert send(camera, controller, cs)['JerkUpperLimit'] == 2.


@pytest.mark.parametrize('camera', [True, False])
def test_upper_history_survives_departure_and_interlock_episode_resets(camera):
  cs = make_cs()
  controller = CanfdStopping()
  for _ in range(2):
    send(camera, controller, cs)
  send(camera, controller, cs, stopping=False, jerk_upper=.5)
  assert controller.phase == StopPhase.idle
  assert send(camera, controller, cs)['JerkUpperLimit'] == .5
  cs.out.canValid = False
  blocked = send(camera, controller, cs, stopping=False, jerk_upper=.5)
  assert blocked['StopReq'] == 0
  assert controller.phase == StopPhase.idle
  cs.out.canValid = True
  assert send(camera, controller, cs)['JerkUpperLimit'] == .5
  assert send(camera, controller, cs)['JerkUpperLimit'] == 2.


@pytest.mark.parametrize('camera', [True, False])
def test_upper_edge_uses_final_request_after_approach(camera):
  cs = make_cs()
  controller = CanfdStopping()
  cs.out.vEgo = cs.out.vEgoRaw = 1.
  cs.out.wheelSpeeds = SimpleNamespace(fl=1., fr=1., rl=1., rr=1.)
  approach = send(camera, controller, cs)
  assert approach['StopReq'] == 0
  assert approach['JerkUpperLimit'] == 2.
  cs.out.vEgo = cs.out.vEgoRaw = .3
  cs.out.wheelSpeeds = SimpleNamespace(fl=.3, fr=.3, rl=.3, rr=.3)
  request = send(camera, controller, cs)
  assert request['StopReq'] == 1
  assert request['JerkUpperLimit'] == 2.  # already sent before the real edge
  for _ in range(170):
    packet = send(camera, controller, cs)
    assert packet['JerkUpperLimit'] == 2.
    assert packet['StopReq'] == 1


def test_missing_camera_snapshot_does_not_consume_first_upper_packet():
  cs = make_cs()
  controller = CanfdStopping()
  send(True, controller, cs, stopping=False, jerk_upper=.5)
  snapshot = cs.scc_control
  cs.scc_control = None
  assert send(True, controller, cs) is None
  cs.scc_control = snapshot
  assert send(True, controller, cs)['JerkUpperLimit'] == .5
  assert send(True, controller, cs)['JerkUpperLimit'] == 2.


@pytest.mark.parametrize('camera', [True, False])
def test_upper_sequence_changes_no_other_decoded_packet_fields(camera):
  controller, reference = CanfdStopping(), CanfdStopping()
  reference.limit_scc_jerk_upper = lambda stop_req, upper: upper
  cs = make_cs()
  for tick in range(220):
    # Include ordinary PID frames and sustained stopping.
    args = dict(stopping=tick >= 5, jerk_upper=1. if tick < 5 else 2.,
                accel=-.5, previous_value=-.8, previous_target=-.8)
    actual = dict(send(camera, controller, cs, **args))
    expected = dict(send(camera, reference, cs, **args))
    actual.pop('CHECKSUM')
    expected.pop('CHECKSUM')
    actual.pop('JerkUpperLimit')
    expected.pop('JerkUpperLimit')
    assert actual == expected
    assert controller.phase == reference.phase


# --- ButterStop: pass the openpilot request through approach/request --------

def butter_cmd(cmd, expected):
  assert cmd.stop_req == expected[0]
  assert cmd.raw == pytest.approx(expected[1], abs=1e-9)
  assert cmd.value == pytest.approx(expected[2], abs=1e-9)
  assert cmd.lower == pytest.approx(expected[3], abs=1e-9)


def test_pass_accel_default_off():
  controller = CanfdStopping()
  assert controller.pass_accel is False

  cmd = step(controller, speed=0.3, accel=-0.2, previous_value=-0.2)
  assert controller.phase == StopPhase.request and controller.reason == "stop_requested"
  butter_cmd(cmd, (1, 0.0, 0.0, STOP_LOWER_BAND))

  controller = CanfdStopping()
  cmd = step(controller, speed=1.0, accel=-0.2, previous_value=-0.2)
  assert controller.phase == StopPhase.approach
  butter_cmd(cmd, (0, -0.5, -0.22, 0.0))

  controller = CanfdStopping()
  for _ in range(15):  # hold confirmed after 0.2 s at 50 Hz
    cmd = step(controller, speed=0.0, accel=-0.5, previous_value=-0.5)
  assert controller.phase == StopPhase.held
  butter_cmd(cmd, (1, 0.0, 0.0, STOP_LOWER_BAND))


def test_pass_accel_request_sends_decel_until_hold():
  # A zero entry while the car still rolls never reaches a hold; the request
  # phase keeps sending the openpilot decel until the ESC confirms it.
  for accel in (-0.2, -0.6):
    controller = CanfdStopping()
    controller.pass_accel = True
    butter_cmd(step(controller, speed=0.3, accel=accel, previous_value=accel), (1, accel, accel, 0.0))


def test_creep_guard_reaches_esc():
  # The relaxed -1.5 creep guard is jerk limited, not clipped to the -0.5 floor.
  controller = CanfdStopping()
  controller.pass_accel = True
  butter_cmd(step(controller, speed=0.3, accel=-1.5, previous_value=-0.5), (1, -1.5, -0.52, 0.0))


def test_avh_hold_switches_to_stock_zero():
  controller = CanfdStopping()
  controller.pass_accel = True
  butter_cmd(step(controller, speed=0.0, held=True, accel=-0.5, previous_value=-0.5), (1, 0.0, 0.0, STOP_LOWER_BAND))
  assert controller.phase == StopPhase.held


def test_slow_roll_without_avh_keeps_decel():
  # 6d1 3.5 case: 0.03 m/s is inside the stop region (held) but not truly stopped,
  # so the ESC never sees the zero entry and keeps getting the decel.
  controller = CanfdStopping()
  controller.pass_accel = True
  for _ in range(15):  # stopped_time reaches STOP_CONFIRM_TIME
    butter_cmd(step(controller, speed=0.03, held=False, accel=-0.5, previous_value=-0.5), (1, -0.5, -0.5, 0.0))
  assert controller.phase == StopPhase.held
  assert controller.zero_frames == 0


def test_true_zero_without_avh_falls_back_to_zero_after_1s():
  controller = CanfdStopping()
  controller.pass_accel = True
  for _ in range(ZERO_CONFIRM_FRAMES - 1):
    butter_cmd(step(controller, speed=0.0, held=False, accel=-0.5, previous_value=-0.5), (1, -0.5, -0.5, 0.0))
  butter_cmd(step(controller, speed=0.0, held=False, accel=-0.5, previous_value=-0.5), (1, 0.0, 0.0, STOP_LOWER_BAND))
  assert controller.zero_frames == ZERO_CONFIRM_FRAMES
  # Any motion restarts the climb-down wait.
  butter_cmd(step(controller, speed=0.05, held=False, accel=-0.5, previous_value=-0.5), (1, -0.5, -0.5, 0.0))
  assert controller.zero_frames == 0


def test_request_entry_never_carries_accel():
  for accel in (-0.05, -0.2, -0.44, -0.46, -1.5):
    controller = CanfdStopping()
    controller.pass_accel = False
    cmd = step(controller, speed=0.3, accel=accel, previous_value=accel)
    assert controller.phase == StopPhase.request
    assert (cmd.stop_req, cmd.raw, cmd.value, cmd.lower) == (1, 0.0, 0.0, STOP_LOWER_BAND)


def test_pass_accel_approach_uses_accel_not_floor():
  controller = CanfdStopping()
  controller.pass_accel = True
  cmd = step(controller, speed=1.0, accel=-0.2, previous_value=-0.2)
  assert controller.phase == StopPhase.approach
  butter_cmd(cmd, (0, -0.2, -0.2, 0.0))


def test_pass_accel_held_keeps_zero_hold():
  controller = CanfdStopping()
  controller.pass_accel = True
  held = []
  for _ in range(15):
    cmd = step(controller, speed=0.0, held=True, accel=-0.5, previous_value=0.0)
    if controller.phase == StopPhase.held:
      held.append(cmd)
  assert controller.phase == StopPhase.held
  assert len(held) >= 1
  for cmd in held:
    assert (cmd.stop_req, cmd.raw, cmd.value, cmd.lower) == (1, 0.0, 0.0, STOP_LOWER_BAND)


def butter_cs():
  cs = make_cs()
  cs.out.steeringTorque = 0.0
  cs.out.steeringPressed = False
  cs.out.steeringAngleDeg = 0.0
  cs.out.steerFaultTemporary = False
  cs.out.steerFaultPermanent = False
  cs.out.cruiseState.enabled = True
  cs.is_metric = False
  cs.modelV2 = None
  cs.radarState = None
  cs.lfahda_cluster = None
  cs.lfa = None
  cs.mdps = None
  cs.adrv_0x161 = None
  return cs


def test_controller_reads_butter_stop_param(monkeypatch):
  from opendbc.car.hyundai import hyundaican, hyundaicanfd
  seen = []

  def make_controller(butter_stop):
    def get_bool(key):
      assert key != "CanfdStopRetry", "Removed stop retry setting must never be read"
      return False

    def get_int(key):
      seen.append(key)
      return butter_stop if key == "ButterStop" else 0

    params = SimpleNamespace(get_bool=get_bool, get_int=get_int)
    monkeypatch.setattr(carcontroller, "Params", lambda: params)
    cp = structs.CarParams(carFingerprint=CAR.KIA_EV6, flags=int(HyundaiFlags.CANFD),
                           openpilotLongitudinalControl=True, stoppingDecelRate=0.8,
                           steerRatio=13.5, wheelbase=2.9)
    return carcontroller.CarController({Bus.pt: "hyundai_canfd_generated"}, cp)

  # The unrelated CAN emitters need a live bus; the parameter plumbing is the
  # subject here. Same separation the cluster-popup controller test uses.
  monkeypatch.setattr(hyundaicanfd, "create_acc_control", lambda *a, **kw: (None, 0.0))
  for name in ("create_steering_messages", "create_lfahda_cluster", "create_lfa_icon_non_camera_scc",
               "create_ccnc_messages", "create_fca_warning_light"):
    monkeypatch.setattr(hyundaicanfd, name, lambda *a, **kw: [])
  monkeypatch.setattr(hyundaican, "create_frt_radar_opt", lambda *a, **kw: [])

  controller = make_controller(1)
  assert isinstance(controller.canfd_stopping, CanfdStopping)
  assert controller.canfd_stopping.pass_accel is False  # constructor default
  monkeypatch.setattr(controller.hyundai_jerk, "make_jerk", lambda *a, **kw: None)
  monkeypatch.setattr(controller.hyundai_jerk, "check_carrot_cruise", lambda *a, **kw: None)
  assert controller.frame == 0  # frame % 50 == 0 runs the parameter refresh block
  controller.update(structs.CarControl().as_reader(), butter_cs(), 0)
  assert "ButterStop" in seen
  assert controller.canfd_stopping.pass_accel is True

  controller = make_controller(0)
  controller.canfd_stopping.pass_accel = True  # prove the block drives it both ways
  monkeypatch.setattr(controller.hyundai_jerk, "make_jerk", lambda *a, **kw: None)
  monkeypatch.setattr(controller.hyundai_jerk, "check_carrot_cruise", lambda *a, **kw: None)
  controller.update(structs.CarControl().as_reader(), butter_cs(), 0)
  assert controller.canfd_stopping.pass_accel is False


def test_pass_accel_approach_hard_keeps_floor():
  controller = CanfdStopping()
  controller.pass_accel = True
  cmd = step(controller, speed=1.0, accel=-0.6, previous_value=-0.6)
  assert controller.phase == StopPhase.approach
  butter_cmd(cmd, (0, -0.6, -0.6, 0.0))


# --- Creep rescue: the ESC lets go short of rest and the car re-accelerates --
# Route 6db 18:02: StopReq 1 with aReq -1.5 was ignored for 30 s while the car
# crawled 7.7 m. The same ESC follows StopReq 0 decel, so a re-acceleration
# under StopReq drops the request and brakes until rest, then asks again.

REPO_ROOT = Path(__file__).resolve().parents[5]
STOPPING_REL = "opendbc_repo/opendbc/car/hyundai/stopping.py"
BASE_REVS = ("rescue-base", "95590fa26221d5cd370687e35caf67014f8c02f1", "HEAD^",
             "pr-fork/carrot-wip", "origin/carrot-wip", "carrot-wip")


def load_base_stopping():
  """The pre-rescue controller (95590fa26 == fc31fe2ec for this file), from git."""
  for rev in BASE_REVS:
    try:
      source = subprocess.run(["git", "-C", str(REPO_ROOT), "-c", "safe.directory=*",
                               "show", f"{rev}:{STOPPING_REL}"],
                              check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, OSError):
      continue
    if not source:
      continue
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "base_stopping.py"
      path.write_text(source)
      spec = importlib.util.spec_from_file_location("rescue_base_stopping", path)
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
    return module
  pytest.skip("pre-rescue stopping module is unavailable in this checkout")


def creep_approach(controller, *, accel=-0.2, start=0.34, end=0.03, plateau=20, held=False):
  """The 6db profile up to the last still tick: v 0.34 -> 0.03, then a plateau."""
  previous = -0.5
  seen = []
  for i in range(int(round((start - end) / 0.01)) + 1):
    speed = round(start - 0.01 * i, 3)
    cmd = step(controller, speed=speed, held=held, accel=accel, previous_value=previous)
    seen.append((speed, cmd))
    previous = cmd.value
  for _ in range(plateau):
    cmd = step(controller, speed=end, held=held, accel=-0.51, previous_value=previous)
    seen.append((end, cmd))
    previous = cmd.value
  return seen, previous


def creep_rise(controller, previous, *, accel=-1.5, steps=14, base=0.03, held=False):
  """v climbs 0.035, 0.04, ... at the rate the car used in 6db."""
  out = []
  for i in range(1, steps + 1):
    speed = round(base + 0.005 * i, 4)
    cmd = step(controller, speed=speed, held=held, accel=accel, previous_value=previous)
    out.append((speed, cmd))
    previous = cmd.value
  return out, previous


def test_creep_rescue_drops_stop_req():
  controller = CanfdStopping()
  controller.pass_accel = True
  before, previous = creep_approach(controller)
  rising, _ = creep_rise(controller, previous)

  assert controller.phase == StopPhase.rescue
  assert controller.reason == "creep_rescue"
  trigger = next(cmd for speed, cmd in rising if speed >= RESCUE_SPEED)
  assert trigger.stop_req == 0
  assert trigger.raw == pytest.approx(RESCUE_ACCEL, abs=1e-9)

  for _, cmd in before:
    assert cmd.stop_req == 1
  for speed, cmd in rising:
    if speed < RESCUE_SPEED:
      assert cmd.stop_req == 1


def test_rescue_rerequests_after_rest():
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller)
  creep_rise(controller, previous)
  assert controller.phase == StopPhase.rescue

  previous = -1.0
  cmd = None
  tick = -1
  for tick in range(15):
    cmd = step(controller, speed=0.02, accel=-1.5, previous_value=previous)
    previous = cmd.value
    if cmd.stop_req == 1:
      break
  assert cmd.stop_req == 1
  assert tick <= 10  # 0.2 s of stopped_time at 50 Hz is 10 ticks
  assert controller.phase == StopPhase.held
  assert controller.reason == "rescue_stopped"
  assert controller.rescue_cycles == 1


def test_no_rescue_while_avh():
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller, held=False)
  for i in range(1, 21):
    speed = round(0.03 + 0.005 * i, 4)
    cmd = step(controller, speed=speed, held=True, accel=-1.5, previous_value=previous)
    previous = cmd.value
    assert cmd.stop_req == 1
    assert controller.phase != StopPhase.rescue


def test_normal_stop_never_rescues():
  base = load_base_stopping()
  controller = CanfdStopping()
  controller.pass_accel = True
  reference = base.CanfdStopping()
  reference.pass_accel = True

  speeds = [round(0.35 - 0.01 * i, 3) for i in range(36)]  # 0.35 -> 0.0
  speeds += [0.0] * 100                                    # 2 s at rest
  speeds += [0.02, 0.06, 0.02]                             # jitter near rest
  previous = -0.5
  for i, speed in enumerate(speeds):
    held = i >= 36 + 50  # AVH engages 1 s after rest
    cmd = step(controller, speed=speed, held=held, accel=-0.5, previous_value=previous)
    ref = step(reference, speed=speed, held=held, accel=-0.5, previous_value=previous)
    assert controller.phase != StopPhase.rescue
    assert (cmd.stop_req, cmd.raw, cmd.value, cmd.lower) == (ref.stop_req, ref.raw, ref.value, ref.lower)
    previous = cmd.value


def test_rescue_max_cycles_keeps_braking():
  controller = CanfdStopping()
  controller.pass_accel = True
  previous = -0.5
  for cycle in range(3):
    if cycle == 0:
      _, previous = creep_approach(controller)
      rising, previous = creep_rise(controller, previous)
    else:
      rising = []
      for i in range(1, 9):  # reset -> 0.10 at 0.01 per tick
        speed = round(0.02 + 0.01 * i, 4)
        cmd = step(controller, speed=speed, accel=-1.5, previous_value=previous)
        rising.append((speed, cmd))
        previous = cmd.value
    assert rising[-1][1].stop_req == 0
    assert controller.phase == StopPhase.rescue

    cmd = None
    for _ in range(100):  # 2 s held at rest (0.0: 0.02 m/s without AVH is a stall, see the stall rescue)
      cmd = step(controller, speed=0.0, accel=-1.5, previous_value=previous)
      previous = cmd.value

    if cycle < RESCUE_MAX_CYCLES:
      assert controller.rescue_cycles == cycle + 1
      assert controller.phase == StopPhase.held
      assert controller.reason == "rescue_stopped"
      assert cmd.stop_req == 1
    else:
      assert controller.rescue_cycles == RESCUE_MAX_CYCLES
      assert controller.phase == StopPhase.rescue
      assert cmd.stop_req == 0
      assert RESCUE_ACCEL_MIN <= cmd.raw <= RESCUE_ACCEL


def test_rescue_releases_when_not_requested():
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller)
  for speed in (0.10, 0.11):
    cmd = step(controller, speed=speed, accel=-1.5, previous_value=previous)
    previous = cmd.value
  assert controller.phase == StopPhase.rescue

  assert step(controller, requested=False, speed=0.12) is None
  assert controller.phase == StopPhase.idle


def test_rescue_without_pass_accel():
  controller = CanfdStopping()
  assert controller.pass_accel is False
  _, previous = creep_approach(controller)
  rising, _ = creep_rise(controller, previous)

  assert controller.phase == StopPhase.rescue
  trigger = next(cmd for speed, cmd in rising if speed >= RESCUE_SPEED)
  assert trigger.stop_req == 0
  assert trigger.raw == pytest.approx(RESCUE_ACCEL, abs=1e-9)


def test_rescue_ramp_reaches_braking_before_the_warning():
  """-0.51 -> -2.5 within 20 ticks (0.4 s): the driver warning needs 1 s of creep."""
  controller = CanfdStopping()
  value = -0.51  # the 6db entry: the creep plateau sat at -0.51
  reached = 0
  first_step = None
  for tick in range(1, 21):
    cmd = controller._rescue_command(RESCUE_ACCEL, value, 2.0, 1.0)
    assert cmd.stop_req == 0
    assert cmd.raw == pytest.approx(RESCUE_ACCEL, abs=1e-9)
    step_size = value - cmd.value
    assert step_size <= RESCUE_JERK * DT + 1e-9, f"tick {tick} stepped {step_size:.4f}"
    if first_step is None:
      first_step = step_size
    value = cmd.value
    if cmd.value <= RESCUE_ACCEL + 1e-9:
      reached = tick
      break
  # The stock 1.0 m/s^3 lower limit (passed here as jerk_l) would need 2 s.
  assert first_step == pytest.approx(RESCUE_JERK * DT, abs=1e-9)
  assert reached > 0, f"rescue aReq stuck at {value:.3f}"
  assert reached * DT <= 0.4 + 1e-9

  # The same cap holds on the real update() path, whatever jerk_l the caller passes.
  live = CanfdStopping()
  live.pass_accel = True
  _, previous = creep_approach(live)
  trigger = None
  for i in range(1, 15):
    speed = round(0.03 + 0.005 * i, 4)
    cmd = step(live, speed=speed, accel=-2.5, previous_value=previous, jerk_l=1.0)
    if live.phase == StopPhase.rescue:
      trigger = cmd
      break
    previous = cmd.value
  assert trigger is not None, "rescue never triggered"
  assert previous - trigger.value <= RESCUE_JERK * DT + 1e-9


def test_rescue_passes_a_harder_op_accel():
  """A harder request than -2.5 is passed down to the -3.5 floor, not clamped upward."""
  raw = {}
  for accel in (-3.0, -5.0):
    fresh = CanfdStopping()
    fresh.pass_accel = True
    _, previous = creep_approach(fresh)
    trigger = None
    for i in range(1, 15):
      speed = round(0.03 + 0.005 * i, 4)
      cmd = step(fresh, speed=speed, accel=accel, previous_value=previous)
      if fresh.phase == StopPhase.rescue:
        trigger = cmd
        break
      previous = cmd.value
    assert trigger is not None
    raw[accel] = trigger.raw

  assert raw[-3.0] == pytest.approx(-3.0, abs=1e-9)
  assert raw[-5.0] == pytest.approx(RESCUE_ACCEL_MIN, abs=1e-9)


def test_rescue_never_eases_with_a_soft_op_accel():
  """OP asking 0.0 or a positive accel in the creep still gets the full rescue braking."""
  for accel in (0.0, 0.6):
    controller = CanfdStopping()
    controller.pass_accel = True
    _, previous = creep_approach(controller)
    trigger = None
    for i in range(1, 15):
      speed = round(0.03 + 0.005 * i, 4)
      cmd = step(controller, speed=speed, accel=accel, previous_value=previous)
      if controller.phase == StopPhase.rescue:
        trigger = cmd
        break
      previous = cmd.value
    assert trigger is not None
    assert trigger.raw == pytest.approx(RESCUE_ACCEL, abs=1e-9)
    assert trigger.stop_req == 0


# --- Stall rescue: a stop that hangs just above rest without AVH ------------
# Route 6e1 7.00-7.14 min: the car came down to 0.03-0.05 m/s under StopReq 1 and
# aReq -0.51, never reached zero, so the ESC never set AVH, and idle creep walked
# it from 0.03 to 0.09 m/s over ~8 s (0.6 m) before RESCUE_SPEED fired.

STALL_TICKS = round(STALL_T / DT)


def stall_plateau(controller, previous, ticks, *, speed=0.04, held=False):
  """The 6e1 hang: StopReq 1 with the car neither at rest nor moving."""
  out = []
  for _ in range(ticks):
    cmd = step(controller, speed=speed, held=held, accel=-0.51, previous_value=previous)
    out.append(cmd)
    previous = cmd.value
  return out, previous


def stalled_controller():
  """A controller sitting in the first stall rescue, as the 6e1 trigger leaves it."""
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller, end=0.04, plateau=0)
  _, previous = stall_plateau(controller, previous, STALL_TICKS + 6)
  assert controller.phase == StopPhase.rescue
  assert controller.reason == "stall_rescue"
  return controller, previous


def test_stall_rescue_fires_after_stall_t():
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller, end=0.04, plateau=0)

  for _ in range(STALL_TICKS - 3):
    cmd = step(controller, speed=0.04, held=False, accel=-0.51, previous_value=previous)
    previous = cmd.value
    assert controller.phase != StopPhase.rescue
    assert cmd.stop_req == 1

  trigger = None
  for _ in range(6):
    cmd = step(controller, speed=0.04, held=False, accel=-0.51, previous_value=previous)
    previous = cmd.value
    if cmd.stop_req == 0:
      trigger = cmd
      break
    assert controller.phase != StopPhase.rescue
    assert cmd.stop_req == 1
  assert trigger is not None, "stall rescue never fired"
  assert trigger.raw == pytest.approx(RESCUE_ACCEL, abs=1e-9)
  assert controller.phase == StopPhase.rescue
  assert controller.reason == "stall_rescue"


def test_stall_rescue_ends_at_zero():
  controller, previous = stalled_controller()
  _, previous = stall_plateau(controller, previous, STALL_ZERO_FRAMES, speed=0.0)
  assert controller.phase == StopPhase.held
  assert controller.reason == "stall_stopped"
  assert controller.rescue_cycles == 1
  cmd = step(controller, speed=0.0, held=False, accel=-0.51, previous_value=previous)
  assert cmd.stop_req == 1


def test_stall_rescue_timeout():
  controller, previous = stalled_controller()
  _, previous = stall_plateau(controller, previous, round(STALL_RESCUE_MAX_T / DT) + 1, speed=0.03)
  assert controller.phase == StopPhase.held
  assert controller.reason == "stall_timeout"
  cmd = step(controller, speed=0.03, held=False, accel=-0.51, previous_value=previous)
  assert cmd.stop_req == 1


def test_no_stall_with_avh():
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller, end=0.04, plateau=0)
  cmds, _ = stall_plateau(controller, previous, 3 * STALL_TICKS, held=True)
  for cmd in cmds:
    assert cmd.stop_req == 1
  assert controller.phase == StopPhase.held


def test_no_stall_when_truly_stopped():
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller, end=0.04, plateau=0)
  cmds, _ = stall_plateau(controller, previous, 3 * STALL_TICKS, speed=0.0)
  for cmd in cmds:
    assert cmd.stop_req == 1
  assert controller.phase != StopPhase.rescue


def test_stall_respects_max_cycles():
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller, end=0.04, plateau=0)
  for cycle in range(RESCUE_MAX_CYCLES):
    rising, previous = stall_plateau(controller, previous, STALL_TICKS + 10)
    assert any(cmd.stop_req == 0 for cmd in rising), f"cycle {cycle} never triggered"
    _, previous = stall_plateau(controller, previous, STALL_ZERO_FRAMES, speed=0.0)
    assert controller.reason == "stall_stopped"
    assert controller.rescue_cycles == cycle + 1

  cmds, _ = stall_plateau(controller, previous, 3 * STALL_TICKS)
  for cmd in cmds:
    assert cmd.stop_req == 1
  assert controller.phase != StopPhase.rescue
  assert controller.rescue_cycles == RESCUE_MAX_CYCLES


def test_6e1_profile_stall_fires_before_creep_speed():
  assert STALL_TICKS <= 145, "the 6e1 plateau profile is too short for this STALL_T"
  controller = CanfdStopping()
  controller.pass_accel = True
  _, previous = creep_approach(controller, end=0.04, plateau=0)

  profile = [0.04] * 50 + [0.03] * 50 + [0.05] * 50
  profile += [round(0.05 + 0.001 * i, 4) for i in range(1, 41)]
  trigger = None
  for speed in profile:
    cmd = step(controller, speed=speed, held=False, accel=-0.51, previous_value=previous)
    previous = cmd.value
    if cmd.stop_req == 0:
      trigger = (speed, cmd)
      break
    assert controller.phase != StopPhase.rescue

  assert trigger is not None, "the 6e1 stall never triggered"
  speed, cmd = trigger
  assert controller.reason == "stall_rescue"
  assert speed <= 0.05
  assert cmd.raw == pytest.approx(RESCUE_ACCEL, abs=1e-9)


def test_no_stall_after_hold_dropout():
  """AVH held once, then the flag drops at 0.02 m/s: no stall rescue, even past 3 x STALL_T."""
  controller = CanfdStopping()
  controller.pass_accel = True
  for _ in range(50):
    step(controller, speed=.02, held=True)
  for _ in range(3 * STALL_TICKS):
    assert step(controller, speed=.02, held=False).stop_req == 1
  assert controller.phase == StopPhase.held


def surge_profile(controller, *, a_peak, held=False):
  """6e3 2.39 min: down to 0.04 m/s under StopReq 1, then the ESC lets go and idle creep surges."""
  creep_approach(controller, end=0.04, plateau=10, held=held)
  out = []
  previous = -0.5
  for speed, a_ego in ((0.05, 0.02), (0.05, 0.03), (0.065, a_peak), (0.08, a_peak), (0.09, a_peak), (0.10, a_peak)):
    cmd = step(controller, speed=speed, held=held, accel=-0.51, previous_value=previous, a_ego=a_ego)
    out.append((speed, a_ego, cmd))
    previous = cmd.value
  return out


def test_surge_rescue_fires_before_creep_speed():
  controller = CanfdStopping()
  controller.pass_accel = True
  out = surge_profile(controller, a_peak=0.12)
  first = next(i for i, (_, _, cmd) in enumerate(out) if cmd.stop_req == 0)
  speed, a_ego, cmd = out[first]
  assert speed == pytest.approx(0.065) and speed < RESCUE_SPEED
  assert a_ego >= SURGE_ACCEL
  assert cmd.raw == pytest.approx(RESCUE_ACCEL, abs=1e-9)
  assert controller.phase == StopPhase.rescue
  assert controller.reason == "surge_rescue"
  assert controller.rescue_kind == "creep"
  assert all(cmd.stop_req == 1 for _, _, cmd in out[:first])


def test_surge_rescue_rerequests_after_rest():
  controller = CanfdStopping()
  controller.pass_accel = True
  surge_profile(controller, a_peak=0.12)
  previous = -1.0
  for tick in range(15):
    cmd = step(controller, speed=0.0, accel=-1.5, previous_value=previous)
    previous = cmd.value
    if cmd.stop_req == 1:
      break
  assert cmd.stop_req == 1
  assert controller.reason == "rescue_stopped"
  assert controller.rescue_cycles == 1


def test_no_surge_rescue_below_surge_accel():
  """Stops that held peaked at +0.06 m/s^2 before AVH (routes 6d5..6e3): no rescue."""
  controller = CanfdStopping()
  controller.pass_accel = True
  out = surge_profile(controller, a_peak=0.06)
  assert all(cmd.stop_req == 1 for speed, _, cmd in out if speed < RESCUE_SPEED)


def test_no_surge_rescue_while_avh():
  controller = CanfdStopping()
  controller.pass_accel = True
  out = surge_profile(controller, a_peak=0.25, held=True)
  assert all(cmd.stop_req == 1 for _, _, cmd in out)
  assert controller.phase != StopPhase.rescue


def test_no_surge_rescue_before_stop_speed():
  """aEgo noise while the car is still above STOP_SPEED is the approach, not a surge."""
  controller = CanfdStopping()
  controller.pass_accel = True
  creep_approach(controller, end=STOP_SPEED + 0.02, plateau=10)
  cmd = step(controller, speed=STOP_SPEED + 0.02 + SURGE_RISE, accel=-0.51, a_ego=0.3)
  assert cmd.stop_req == 1
  assert controller.phase != StopPhase.rescue


def test_no_surge_rescue_without_rise():
  controller = CanfdStopping()
  controller.pass_accel = True
  creep_approach(controller, end=STOP_SPEED, plateau=10)
  cmd = step(controller, speed=STOP_SPEED + SURGE_RISE / 2, accel=-0.51, a_ego=0.3)
  assert cmd.stop_req == 1
  assert controller.phase != StopPhase.rescue


@pytest.mark.parametrize("camera", [True, False])
def test_caller_passes_measured_acceleration(camera):
  cs = make_cs()
  controller = CanfdStopping()
  controller.pass_accel = True
  for speed, a_ego in [(round(0.3 - 0.01 * i, 2), -0.2) for i in range(27)] + [(0.04, -0.05)] * 10 + [(0.065, 0.15)]:
    cs.out.vEgo = cs.out.vEgoRaw = speed
    cs.out.wheelSpeeds = SimpleNamespace(fl=speed, fr=speed, rl=speed, rr=speed)
    cs.out.aEgo = a_ego
    values = send(camera, controller, cs)
  assert controller.reason == "surge_rescue"
  assert values["StopReq"] == 0
