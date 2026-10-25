"""ButterStop: the hold is confirmed only once the car is at rest.

CS.standstill on Hyundai is wheel speed <= 0.104 m/s, so it rises ~0.4 s before
the car stops. Confirming the hold on it ramped the deceleration to the hold
value while the car was still rolling, which is the step the K5 driver felt.
These tests pin the rest-floor behaviour: the taper value is kept until
STANDSTILL_V has held for STANDSTILL_T, and the creep guard is armed earlier.
"""
import pytest

from openpilot.selfdrive.controls.tests.test_butter_stop import make_control, stopping_inputs
from openpilot.selfdrive.controls.lib import longcontrol as lc
from openpilot.selfdrive.controls.lib.butter_stop import CREEP_ACCEL, HOLD_RAMP, STANDSTILL_T

LongCtrlState = lc.LongCtrlState

# One FLOOR_RELAX_JERK * DT_CTRL above the -0.3 the car was already doing.
REST_FLOOR = -0.292


def test_cs_standstill_does_not_step_to_hold_while_rolling(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.08, standstill=True)
  control.last_output_accel = -0.3
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  # Still rolling at 0.08 m/s: no hold ramp, the taper's last value is kept.
  assert accel == pytest.approx(REST_FLOOR)
  assert not control._butter_standstill_confirmed
  assert control._butter_near_rest


def test_rest_floor_holds_until_confirmed_then_hold_ramp(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.08, standstill=True)
  control.last_output_accel = -0.3
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  # Below STANDSTILL_V the taper is ineligible; the last taper value holds.
  cs.vEgo = 0.03
  for _ in range(int(STANDSTILL_T / lc.DT_CTRL) - 1):
    accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    assert accel == pytest.approx(REST_FLOOR)
    assert not control._butter_standstill_confirmed
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._butter_standstill_confirmed
  assert accel == pytest.approx(REST_FLOOR - HOLD_RAMP * lc.DT_CTRL)
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel == pytest.approx(REST_FLOOR - 2 * HOLD_RAMP * lc.DT_CTRL)


def test_cs_standstill_arms_creep_guard_before_confirm(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.08, standstill=True)
  control.last_output_accel = -0.3
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert not control._butter_standstill_confirmed
  # Creep starts before the hold is confirmed; the guard must still catch it.
  cs.vEgo = 0.2
  cs.standstill = False
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._butter_creep_latched
  assert accel == pytest.approx(CREEP_ACCEL)
