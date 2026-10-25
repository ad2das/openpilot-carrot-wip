"""Departure guard (W-DG): while a departing lead pulls away, the Hyundai speed-error
term must not cut the command below aTarget + DEPART_P_FLOOR for the first
DEPART_WINDOW_T seconds after leaving a stop from rest.

The unguarded output with Kp 1, Ki 0, Kf 1 is aTarget + (vTargetNow - vEgo); every
expected number below comes from that arithmetic (route 6db: v 2.06, vT 1.52,
aT 0.49 -> -0.05, floor 0.29).
"""
from types import SimpleNamespace

import numpy as np
import pytest

# test_butter_stop installs the Windows import shims (hardware, Params, solver) at
# module level, so it has to be imported before longcontrol itself.
from openpilot.selfdrive.controls.tests.test_butter_stop import PLAN_T, make_control
from openpilot.selfdrive.controls.lib import longcontrol as lc

LIMITS = (-3.5, 2.0)
PLAN_AGE = 0.05


def depart_inputs(v_ego, a_target=0.49, v_target=1.52, lead_status=True, d_rel=6.8, v_lead=2.24):
  cs = SimpleNamespace(vEgo=v_ego, aEgo=0.0, brakePressed=False, gasPressed=False,
                       softHoldActive=0, standstill=False,
                       cruiseState=SimpleNamespace(standstill=False), vCruise=0.0, carrotCruise=0)
  plan = SimpleNamespace(aTarget=a_target, vTargetNow=v_target, jTargetNow=0.0, shouldStop=False, fcw=False,
                         speeds=v_ego + max(a_target, 0.0) * PLAN_T,
                         longitudinalPlanSource="lead0", cruiseTarget=0.0,
                         cruiseCoastingTarget=0.0, cruiseCoastingPercent=0)
  lead = SimpleNamespace(status=lead_status, dRel=d_rel, vLead=v_lead, aLeadK=0.0)
  radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False),
                          leadCutInRisk=SimpleNamespace(status=False))
  return cs, plan, radar


def start(control):
  """Put the controller in stopping and leave the stop from rest."""
  control.long_control_state = lc.LongCtrlState.stopping
  cs, plan, radar = depart_inputs(0.0)
  control.update(True, cs, plan, LIMITS, PLAN_AGE, radar)


def _run(control, v_ego, **kwargs):
  cs, plan, radar = depart_inputs(v_ego, **kwargs)
  return control.update(True, cs, plan, LIMITS, PLAN_AGE, radar)[0]


def test_guard_limits_speed_error_cut(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  start(control)
  accel = _run(control, 2.06)
  # Unguarded the speed error is 1.52 - 2.06 = -0.54 -> -0.05; the guard holds 0.29.
  assert accel == pytest.approx(0.29, abs=1e-6)
  assert control.depart_guard_last
  assert control.pid.i == 0.0


def test_no_guard_when_lead_slower_and_one_shot(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  start(control)
  assert _run(control, 2.06, v_lead=1.9) == pytest.approx(-0.05, abs=1e-6)
  assert control._depart_t is None
  # The safety exit ends the guard for this departure, not just this frame.
  assert _run(control, 2.06, v_lead=2.24) == pytest.approx(-0.05, abs=1e-6)


def test_no_guard_when_lead_close(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  start(control)
  assert _run(control, 2.06, d_rel=3.5) == pytest.approx(-0.05, abs=1e-6)


def test_guard_without_lead(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  start(control)
  assert _run(control, 2.06, lead_status=False) == pytest.approx(0.29, abs=1e-6)


def test_window_end_relaxes_then_releases(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  start(control)
  outs = []
  for tick in range(1, 801):
    outs.append(_run(control, 2.06))
    if tick <= 350:
      assert outs[-1] == pytest.approx(0.29, abs=1e-6)
  # 4.0 s window at 100 Hz: the floor starts relaxing at tick 401.
  assert 0.15 < outs[499] < 0.25
  assert np.all(np.diff(outs[349:800]) <= 1e-9)
  assert outs[799] == pytest.approx(-0.05, abs=1e-6)
  assert control._depart_floor is None


def test_not_from_rest_no_guard(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  control.long_control_state = lc.LongCtrlState.stopping
  # Leaving stopping above DEPART_START_V is not a departure from rest: plain PID.
  assert _run(control, 1.0) == pytest.approx(1.01, abs=1e-6)
  assert _run(control, 2.06) == pytest.approx(-0.05, abs=1e-6)


def test_butter_stop_off_no_guard(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 0})
  start(control)
  assert _run(control, 2.06) == pytest.approx(-0.05, abs=1e-6)
  assert not control.depart_guard_last


def test_plan_decel_relaxes_floor(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  start(control)
  assert _run(control, 2.06) == pytest.approx(0.29, abs=1e-6)
  # The plan stops accelerating: the floor relaxes at DEPART_RELEASE_JERK = 1 m/s^3.
  assert _run(control, 2.06, a_target=-0.1) == pytest.approx(0.28, abs=1e-6)
  for _ in range(99):
    accel = _run(control, 2.06, a_target=-0.1)
  assert accel == pytest.approx(-0.64, abs=1e-6)
  assert control._depart_floor is None
