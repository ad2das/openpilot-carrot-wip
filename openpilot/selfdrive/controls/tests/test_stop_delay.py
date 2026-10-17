"""CP-SD stop-phase delay: pure weight scheduling and slew behaviour.

Tests the module-level pure function and constants only; the planner wiring is
covered by the harness H0 identity check.
"""
import numpy as np
import pytest

import openpilot.selfdrive.controls.lib.stop_delay as sd


def _w(v_ego, *, lead_status=True, d_rel=30.0, v_lead=0.0, a_lead=0.0,
       fcw=False, gas=False, brake=False):
  return sd.stop_phase_weight(v_ego, lead_status, d_rel, v_lead, a_lead, fcw, gas, brake)


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
  monkeypatch.setattr(sd, "STOP_DELAY_ENABLED", True)
  monkeypatch.setattr(sd, "DELAY_LO", 0.5)
  monkeypatch.setattr(sd, "V_LO", 2.0)
  monkeypatch.setattr(sd, "V_FADE", 1.0)
  monkeypatch.setattr(sd, "LEAD_V_MAX", 0.5)
  monkeypatch.setattr(sd, "LEAD_A_MIN", -0.5)
  monkeypatch.setattr(sd, "GAP_ABS", 3.0)
  monkeypatch.setattr(sd, "MARGIN_SD", 0.3)
  monkeypatch.setattr(sd, "A_REQ_MAX", 0.8)


def test_disabled_returns_zero(monkeypatch):
  monkeypatch.setattr(sd, "STOP_DELAY_ENABLED", False)
  assert _w(1.0) == 0.0


def test_w_zero_gates():
  assert _w(1.0, fcw=True) == 0.0
  assert _w(1.0, gas=True) == 0.0
  assert _w(1.0, brake=True) == 0.0
  assert _w(1.0, lead_status=False) == 0.0
  assert _w(1.0, v_lead=2.0) == 0.0            # moving lead
  assert _w(1.0, a_lead=-1.0) == 0.0           # braking lead
  assert _w(1.0, d_rel=3.0) == 0.0             # d_avail <= 0
  assert _w(3.5) == 0.0                        # above V_LO + V_FADE
  assert _w(0.0) == 0.0                        # standstill


def test_w_one_at_or_below_v_lo():
  assert _w(2.0) == pytest.approx(1.0)
  assert _w(1.0) == pytest.approx(1.0)
  assert _w(0.2) == pytest.approx(1.0)


def test_linear_fade():
  assert _w(2.5) == pytest.approx(0.5)
  assert _w(2.25) == pytest.approx(0.75)
  assert _w(2.999) < 0.01
  assert _w(3.001) == 0.0


def test_a_req_guard_blocks_high_speed():
  # Closer d_rel -> higher required decel -> base delay.
  assert _w(2.0, d_rel=5.5) == 0.0             # d_avail 2.2, a_req ~0.91 > 0.8
  assert _w(2.0, d_rel=10.0) == pytest.approx(1.0)


def test_weight_in_unit_range():
  for v in np.linspace(0.06, 4.0, 50):
    w = _w(float(v))
    assert 0.0 <= w <= 1.0


def _slew(cur, target, dt=0.05):
  """Mirror the planner slew: rise <= SLEW_UP*dt, drop instantly."""
  rise = sd.SLEW_UP * dt
  if target < cur:
    return target
  return min(target, cur + rise)


def test_slew_rise_limited_and_drop_instant():
  # Rise from 0.25 toward 0.5 is capped at SLEW_UP*DT_MDL per frame.
  cur, target = 0.25, 0.5
  step = _slew(cur, target)
  assert step == pytest.approx(cur + sd.SLEW_UP * 0.05)
  # A drop is instant.
  assert _slew(0.5, 0.25) == pytest.approx(0.25)
  # Repeated rise converges to the target.
  for _ in range(100):
    cur = _slew(cur, target)
  assert cur == pytest.approx(0.5)
