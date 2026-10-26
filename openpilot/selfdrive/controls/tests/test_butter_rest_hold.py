"""ButterStop: the hold is confirmed only once the car is at rest.

CS.standstill on Hyundai is wheel speed <= 0.104 m/s, so it rises ~0.4 s before
the car stops. Confirming the hold on it ramped the deceleration to the hold
value while the car was still rolling, which is the step the K5 driver felt.
These tests pin the rest-floor behaviour: below REST_RAMP_V the floor eases
down to the stop decel at the stop ramp rate (no HOLD_RAMP step) until
STANDSTILL_V has held for STANDSTILL_T, and the creep guard is armed earlier.
Ending on the taper's soft value (~-0.13) let the K5 ESC let go and surge at
0.04-0.05 m/s (routes 6da..6e3).
"""
import pytest

from openpilot.selfdrive.controls.tests.test_butter_stop import make_control, stopping_inputs
from openpilot.selfdrive.controls.lib import longcontrol as lc
from openpilot.selfdrive.controls.lib.butter_stop import (CREEP_ACCEL, HOLD_RAMP, REST_RAMP_V, STANDSTILL_T,
                                                          STANDSTILL_V, rest_ramp_floor)

LongCtrlState = lc.LongCtrlState


def test_cs_standstill_does_not_step_to_hold_while_rolling(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.08, standstill=True)
  control.last_output_accel = -0.3
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  # Still rolling at 0.08 m/s: no hold ramp; the rest ramp grows braking only at the stop ramp rate.
  assert accel == pytest.approx(-0.3 - control.CP.stoppingDecelRate * lc.DT_CTRL)
  assert accel > -0.3 - HOLD_RAMP * lc.DT_CTRL
  assert not control._butter_standstill_confirmed
  assert control._butter_near_rest


def test_rest_ramp_until_confirmed_then_hold_ramp(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.08, standstill=True)
  control.last_output_accel = -0.3
  previous, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  # Below STANDSTILL_V the floor is the stop decel; braking grows at the stop ramp rate.
  cs.vEgo = 0.03
  for _ in range(int(STANDSTILL_T / lc.DT_CTRL) - 1):
    accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    assert accel == pytest.approx(previous - control.CP.stoppingDecelRate * lc.DT_CTRL)
    assert not control._butter_standstill_confirmed
    previous = accel
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._butter_standstill_confirmed
  assert accel == pytest.approx(previous - HOLD_RAMP * lc.DT_CTRL)
  previous = accel
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel == pytest.approx(previous - HOLD_RAMP * lc.DT_CTRL)


def test_rest_ramp_floor_shape():
  assert rest_ramp_floor(REST_RAMP_V, -0.5) == pytest.approx(0.0)
  assert rest_ramp_floor(1.0, -0.5) == pytest.approx(0.0)
  assert rest_ramp_floor((REST_RAMP_V + STANDSTILL_V) / 2, -0.5) == pytest.approx(-0.25)
  assert rest_ramp_floor(STANDSTILL_V, -0.5) == pytest.approx(-0.5)
  assert rest_ramp_floor(0.0, -0.5) == pytest.approx(-0.5)
  # A non-negative stop decel setting never makes the floor positive.
  assert rest_ramp_floor(0.0, 0.2) == pytest.approx(0.0)


def roll_to_rest(control, *, v0=0.6, decel=0.15):
  """Stopping state, v falls at `decel` from v0 to just below STANDSTILL_V; the taper stays live."""
  control.long_control_state = LongCtrlState.stopping
  control.last_output_accel = -0.2
  out = []
  v = v0
  while v >= STANDSTILL_V - 0.005:
    cs, plan, radar = stopping_inputs(v_ego=v, standstill=v < 0.1)
    accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    out.append((v, accel, control.butter_floor_last))
    v -= decel * lc.DT_CTRL
  return out


def test_rest_ramp_reaches_stop_decel_without_a_step(monkeypatch):
  control, _ = make_control(monkeypatch)
  out = roll_to_rest(control)
  rate = control.CP.stoppingDecelRate * lc.DT_CTRL
  for (_, a0, _), (_, a1, _) in zip(out, out[1:]):
    assert a1 >= a0 - rate - 1e-9  # no step: braking grows at most at the stop ramp rate
  # The taper soft end (~-0.13) is gone: by STANDSTILL_V the request is close to the stop decel.
  at_rest = [a for v, a, _ in out if v < STANDSTILL_V]
  assert at_rest and max(at_rest) <= control.stopping_accel + 0.05
  # Above REST_RAMP_V the taper alone decides.
  for v, a, floor in out:
    if v > REST_RAMP_V and floor is not None:
      assert a >= floor - 1e-9


def test_rest_ramp_does_not_ease_when_the_car_creeps_back(monkeypatch):
  control, _ = make_control(monkeypatch)
  out = roll_to_rest(control)
  last = out[-1][1]
  # The ESC lets go: speed rises again below CREEP_V; the floor follows the lowest speed, not v.
  for v in (0.06, 0.09, 0.12):
    cs, plan, radar = stopping_inputs(v_ego=v)
    accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    assert accel <= last + 1e-9
    last = accel


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
