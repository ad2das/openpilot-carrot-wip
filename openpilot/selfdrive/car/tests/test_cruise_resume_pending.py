"""A pedal-release auto engage that the blinker or the steering interlock blocked
must be retried while the pedal stays released, instead of being decided on the
single frame where the press counter reads -1.

The helper keeps the real _prepare_brake_gas, _update_cruise_state, _cruise_control,
_gas_release and _brake_release; only the button / bluetooth path, the params store
and _auto_speed_up (pure set-speed shaping) are stubbed.
"""
from types import SimpleNamespace

import pytest

from openpilot.cereal import car
from openpilot.selfdrive.car.cruise import (AUTO_CRUISE_MAX_STEERING_ANGLE, RESUME_PENDING_AFTER_BLINKER_S,
                                            RESUME_PENDING_MAX_S, VCruiseCarrot)
from opendbc.car import structs

GearShifter = structs.CarState.GearShifter
DT = 0.01
V_EGO_KPH = 20
V_EGO = V_EGO_KPH / 3.6


class LogRecorder:
  def __init__(self):
    self.frames = []
    self.current = []

  def __call__(self, log):
    if len(log) == 0:
      return
    self.current.append(log)

  def flush(self):
    self.frames.append(list(self.current))
    self.current = []


def make_helper(disengage_on_accelerator=False):
  helper = VCruiseCarrot.__new__(VCruiseCarrot)
  rec = LogRecorder()
  helper._add_log = rec
  helper.params = SimpleNamespace(get_bool=lambda key, default=False, return_default=False: False,
                                  put_bool_nonblocking=lambda key, value: None)
  helper.CP = SimpleNamespace(pcmCruise=False, brand="hyundai")
  helper._auto_speed_up = lambda v: v

  # press counters start far from the release edge so frame 1 is not a release
  helper._gas_pressed_count = -100
  helper._gas_pressed_count_last = 0
  helper._gas_pressed_value = 0
  helper._gas_tok_timer = int(0.4 / DT)
  helper._gas_tok = False
  helper._gas_decel_count = 0
  helper._brake_pressed_count = -100
  helper._soft_hold_count = 0
  helper._soft_hold_active = 0
  helper.soft_hold_on_cancel = False

  helper.autoGasTokSpeed = 0
  helper.autoGasCancelSpeed = 0
  helper.autoGasSyncSpeed = 1
  helper.cruiseOnDist = 25.0
  helper.disengage_on_accelerator = disengage_on_accelerator
  helper.autoCruiseControl = 1
  helper.autoCruiseControl_cancel_timer = 0

  helper._cruise_ready = False
  helper._cruise_cancel_state = False
  helper._activate_cruise = 0
  helper._cruise_available = True
  helper._hold_interlock_active = False
  helper._steering_interlock_active = False
  helper._resume_pending = None
  helper._resume_pending_frames = 0
  helper._resume_blinker_off_frames = 0

  helper._lat_enabled = False
  helper._v_cruise_kph_at_brake = 0
  helper._paddle_decel_active = False
  helper._cancel_timer = 0
  helper._pause_auto_speed_up = True
  helper.carrot_cruise_active = False
  helper.enabled_last = False
  helper.cruise_state_available_last = False

  helper.xState = 0
  helper.aTarget = 0
  helper.desiredSpeed = 250
  helper.d_rel = 0.0
  helper.v_rel = 0.0
  helper.v_lead_kph = 0.0
  helper.model_v_kph = 0.0
  helper.nRoadLimitSpeed = 0
  helper.road_limit_kph = 0
  helper.nRoadLimitSpeed_last = 0
  helper.v_cruise_kph = float(V_EGO_KPH)
  helper.v_ego_kph_set = V_EGO_KPH

  CS = car.CarState(cruiseState={"standstill": False})
  CS.gasPressed = False
  CS.brakePressed = False
  CS.gas = 0.0
  CS.brake = 0.0
  CS.aEgo = 0.0
  CS.vEgo = V_EGO
  CS.vEgoCluster = V_EGO
  CS.steeringAngleDeg = 0.0
  CS.leftBlinker = False
  CS.rightBlinker = False
  CS.gearShifter = GearShifter.drive
  CS.brakeHoldActive = False
  CS.parkingBrake = False
  CC = car.CarControl(enabled=False)
  return helper, CS, CC, rec


def frame(helper, CS, CC, rec, vset=V_EGO_KPH, steer=0.0, hold=False, available=True, v_cruise=None):
  """One update_v_cruise frame: the interlock latches, then _prepare_brake_gas,
  then _update_cruise_state, exactly in the order update_v_cruise uses."""
  helper._steering_interlock_active = abs(steer) >= AUTO_CRUISE_MAX_STEERING_ANGLE
  helper._hold_interlock_active = hold
  helper._cruise_available = available
  helper.v_ego_kph_set = vset
  helper._activate_cruise = 0
  CS.steeringAngleDeg = steer
  CS.brakeHoldActive = hold
  if v_cruise is not None:
    helper.v_cruise_kph = v_cruise
  helper._prepare_brake_gas(CS, CC)
  helper._update_cruise_state(CS, CC, helper.v_cruise_kph)
  rec.flush()
  return rec.frames[-1]


def press_helper(helper, CS, CC, rec, brake=True, frames=100, vset=V_EGO_KPH, steer=0.0):
  if brake:
    CS.brakePressed = True
  else:
    CS.gasPressed = True
  for _ in range(frames):
    frame(helper, CS, CC, rec, vset=vset, steer=steer)
  CS.brakePressed = False
  CS.gasPressed = False


def test_u1_brake_release_blinker_then_retry():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)
  CS.leftBlinker = True

  logs = frame(helper, CS, CC, rec, steer=10.0)
  assert helper._activate_cruise == 0
  assert helper._resume_pending == 'brake'
  assert "Resume pending (brake, blinker)" in logs

  for _ in range(199):
    assert frame(helper, CS, CC, rec, steer=10.0) == []
  assert helper._resume_pending == 'brake'

  CS.leftBlinker = False
  logs = frame(helper, CS, CC, rec, steer=10.0)
  assert helper._activate_cruise == 1
  assert "Cruise on (speed)" in logs
  assert helper._resume_pending is None


def test_u2_brake_release_steering_interlock_then_retry():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)

  logs = frame(helper, CS, CC, rec, steer=120.0)
  assert helper._activate_cruise == 0
  assert "Cruise on (speed) > Steering angle interlock active" in logs
  assert helper._resume_pending == 'brake'

  for _ in range(99):
    frame(helper, CS, CC, rec, steer=120.0)
  assert helper._resume_pending == 'brake'

  logs = frame(helper, CS, CC, rec, steer=60.0)
  assert helper._activate_cruise == 1
  assert "Cruise on (speed)" in logs
  assert helper._resume_pending is None


@pytest.mark.parametrize("label,expected,kwargs", [
  ("gas", "pedal", {}), ("brake", "pedal", {}), ("cancel", "cancel", {}),
  ("gear", "gear", {}), ("hold", "interlock", {"hold": True}),
  ("enabled", "engaged", {}), ("unavailable", "interlock", {"available": False})])
def test_u3_each_drop_reason(label, expected, kwargs):
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)
  CS.leftBlinker = True
  frame(helper, CS, CC, rec, steer=10.0)
  assert helper._resume_pending == 'brake'

  if label == "gas":
    CS.gasPressed = True
  elif label == "brake":
    CS.brakePressed = True
  elif label == "cancel":
    helper._cruise_cancel_state = True
  elif label == "gear":
    CS.gearShifter = GearShifter.reverse
    helper.autoCruiseControl_cancel_timer = 20 * 100
  elif label == "enabled":
    CC.enabled = True

  logs = frame(helper, CS, CC, rec, steer=10.0, **kwargs)
  assert helper._resume_pending is None
  assert f"Resume pending dropped ({expected})" in logs

  # clear the drop condition and settle the press counters well past their
  # release edge, so only the absence of the latch can decide the next frame
  CS.gasPressed = False
  CS.brakePressed = False
  CS.gearShifter = GearShifter.drive
  CS.brakeHoldActive = False
  helper._cruise_cancel_state = False
  helper.autoCruiseControl_cancel_timer = 0
  CC.enabled = False
  helper._cruise_available = True
  helper._gas_pressed_count = -100
  helper._brake_pressed_count = -100
  CS.leftBlinker = False
  logs = frame(helper, CS, CC, rec, steer=0.0)
  assert helper._activate_cruise == 0
  assert helper._resume_pending is None


def test_u4_blinker_off_timeout():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)
  CS.leftBlinker = True
  frame(helper, CS, CC, rec, steer=120.0)
  assert helper._resume_pending == 'brake'

  CS.leftBlinker = False
  for _ in range(50):  # blinker off at release + 0.5 s, steering still blocked
    frame(helper, CS, CC, rec, steer=120.0)
  assert helper._resume_pending == 'brake'

  limit = int(RESUME_PENDING_AFTER_BLINKER_S / DT)
  dropped_at = None
  for i in range(limit + 2):
    logs = frame(helper, CS, CC, rec, steer=120.0)
    if "Resume pending dropped (timeout)" in logs:
      dropped_at = i
      break
  assert dropped_at is not None
  # the 51 frames above already counted as blinker-off frames; the drop lands one
  # frame past the 10.0 s of blinker-off time (the latch uses a strict >)
  assert abs((50 + dropped_at + 1) - (limit + 1)) <= 1
  assert helper._resume_pending is None

  logs = frame(helper, CS, CC, rec, steer=0.0)
  assert helper._activate_cruise == 0


def test_u4_blinker_off_timeout_cleared_in_time():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)
  CS.leftBlinker = True
  frame(helper, CS, CC, rec, steer=120.0)

  CS.leftBlinker = False
  for _ in range(50 + 940):  # 9.9 s of blinker-off frames
    frame(helper, CS, CC, rec, steer=120.0)
  assert helper._resume_pending == 'brake'

  logs = frame(helper, CS, CC, rec, steer=0.0)
  assert helper._activate_cruise == 1
  assert "Cruise on (speed)" in logs
  assert helper._resume_pending is None


def test_u5_hard_cap():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)
  CS.leftBlinker = True
  frame(helper, CS, CC, rec, steer=120.0)
  assert helper._resume_pending == 'brake'

  cap = int(RESUME_PENDING_MAX_S / DT)
  dropped_at = None
  for i in range(cap + 60):
    logs = frame(helper, CS, CC, rec, steer=120.0)
    if "Resume pending dropped (timeout)" in logs:
      dropped_at = i
      break
  assert dropped_at is not None
  assert abs((dropped_at + 1) - cap) <= 1  # frames after the release frame
  assert helper._resume_pending is None

  CS.leftBlinker = False
  logs = frame(helper, CS, CC, rec, steer=0.0)
  assert helper._activate_cruise == 0


def test_u6_gas_release_blinker_then_retry():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=False)
  CS.leftBlinker = True

  logs = frame(helper, CS, CC, rec, steer=10.0)
  assert helper._resume_pending == 'gas'
  assert "Resume pending (gas, blinker)" in logs
  assert helper._activate_cruise == 0

  CS.leftBlinker = False
  # the retry fires on the first frame the blinker is off (steering is clear)
  logs = frame(helper, CS, CC, rec, steer=10.0)
  assert helper._activate_cruise == 1
  assert "Cruise on (gas pressed)" in logs
  assert helper._resume_pending is None


def test_u6_gas_release_blinker_no_latch_with_disengage_on_accelerator():
  helper, CS, CC, rec = make_helper(disengage_on_accelerator=True)
  press_helper(helper, CS, CC, rec, brake=False)
  CS.leftBlinker = True

  frame(helper, CS, CC, rec, steer=10.0)
  assert helper._resume_pending is None


def test_u7_standstill_brake_release_steer_gate():
  helper, CS, CC, rec = make_helper()
  CS.vEgo = 0.0
  CS.vEgoCluster = 0.0
  # 50 brake frames stay below the 60-frame soft-hold threshold, so the release
  # frame takes the _brake_release branch (soft hold would swallow it).
  press_helper(helper, CS, CC, rec, brake=True, frames=50, vset=0, steer=0.0)
  assert helper._soft_hold_active == 0
  helper.d_rel = 10.0
  helper.desiredSpeed = 250

  logs = frame(helper, CS, CC, rec, vset=0, steer=30.0)
  assert helper._resume_pending == 'brake'
  assert "Resume pending (brake, steering)" in logs
  assert helper._activate_cruise == 0

  logs = frame(helper, CS, CC, rec, vset=0, steer=10.0)
  assert helper._activate_cruise == 1
  assert "Cruise on (lead car)" in logs
  assert helper._resume_pending is None


def test_u8_unblocked_release_unchanged():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)

  logs = frame(helper, CS, CC, rec, steer=0.0)
  assert helper._activate_cruise == 1
  assert "Cruise on (speed)" in logs
  assert helper._resume_pending is None
  assert helper._resume_pending_frames == 0


def test_u8_enabled_release_with_blinker_does_not_latch():
  helper, CS, CC, rec = make_helper()
  press_helper(helper, CS, CC, rec, brake=True)
  CC.enabled = True
  CS.leftBlinker = True

  frame(helper, CS, CC, rec, steer=10.0)
  assert helper._resume_pending is None


def test_disengaged_does_not_unconditionally_raise_v_cruise_kph():
  helper, CS, CC, rec = make_helper()
  helper.autoGasSyncSpeed = 0
  helper.v_cruise_kph = 60.0
  helper.v_ego_kph_set = 90
  CS.vEgo = 90 / 3.6
  CS.vEgoCluster = 90 / 3.6
  out = helper._update_cruise_state(CS, CC, helper.v_cruise_kph)
  assert out == 60.0
  assert helper.v_cruise_kph == 60.0


def test_soft_hold_on_cancel_not_wiped_by_fcw_or_desired_speed():
  helper, CS, CC, rec = make_helper()
  CS.cruiseState.available = True
  helper.soft_hold_on_cancel = True
  helper._cruise_cancel_state = True
  helper.cruiseOnDist = 20.0
  helper.d_rel = 6.0
  helper.v_rel = -0.5
  helper.desiredSpeed = 0.0
  CS.vEgo = 0.05  # below soft-hold arm threshold (< 0.1 m/s) but above fcw trigger (> 0.02 m/s)
  CS.vEgoCluster = 0.05
  press_helper(helper, CS, CC, rec, brake=True, frames=65, vset=1, steer=0.0)
  assert helper._soft_hold_active == 1

  # Release brake while still in cancel state: soft hold must engage and stay active
  # (not be clobbered by the same-tick fcw/desiredSpeed _cruise_control rejection).
  logs = frame(helper, CS, CC, rec, vset=1, steer=0.0)
  assert helper._soft_hold_active == 2
  assert helper._activate_cruise == 1
  assert helper._cruise_cancel_state is True
  assert "Cruise on (soft hold)" in logs

  # Next tick before selfdrived reports CC.enabled=True: soft hold must still not be wiped.
  frame(helper, CS, CC, rec, vset=1, steer=0.0)
  assert helper._soft_hold_active == 2
  assert helper._cruise_cancel_state is True


def test_soft_hold_survives_steering_interlock():
  """Soft hold is armed before the interlock check, as in 1456a8f00: at |steer| >= 70 the hold stays active
  and the rejected engage must not fall through to a latched brake-resume."""
  helper, CS, CC, rec = make_helper()
  CS.vEgo = CS.vEgoCluster = 0.0
  CS.cruiseState.available = True
  press_helper(helper, CS, CC, rec, brake=True, frames=100, vset=0, steer=80.0)
  assert helper._soft_hold_active == 1

  for _ in range(5):
    frame(helper, CS, CC, rec, vset=0, steer=80.0)
  assert helper._soft_hold_active == 2
  assert helper._activate_cruise == 0
  assert not any("Resume pending" in line for f in rec.frames[-6:] for line in f)


def test_safe_stop_fcw_engage_same_speed_lead():
  """Coasting manually at 72 km/h 40 m behind a same-speed lead is an unsafe gap for the v^2 stopping model,
  so the fcw engage fires as in 1456a8f00."""
  helper, CS, CC, rec = make_helper()
  CS.vEgo = CS.vEgoCluster = 20.0
  helper.d_rel, helper.v_rel = 40.0, 0.0
  logs = frame(helper, CS, CC, rec, vset=72)
  assert helper._activate_cruise == 1
  assert "Cruise on (fcw)" in logs
