"""Pure-module tests for the far-lead smooth-approach relief (no acados)."""

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib import far_lead_approach as fla
from openpilot.selfdrive.controls.lib.far_lead_approach import ApproachState, approach_path_safe

TIMES = 10.0 * (np.arange(13) / 12.0) ** 2

BASE = {"dt": 0.05, "enabled": True, "lead_present": True, "v_ego": 31.5, "gap": 96.0, "v_lead": 21.6,
        "a_lead": 0.0, "t_follow": 2.2, "t_follow_danger": 1.4, "danger_factor": 0.8,
        "comfort_brake": 2.4, "stop_distance": 6.0}


def call(state=None, **overrides):
  kwargs = dict(BASE)
  kwargs.update(overrides)
  state = ApproachState() if state is None else state
  relief = state.relief(times=TIMES, **kwargs)
  return state, relief


def test_e1_like_state_gets_relief_and_ramps_to_full_scale():
  state, relief = call()
  assert relief[0] > 0.0
  tail = relief[5:]
  assert np.all(np.diff(tail) <= 1e-9), "tail must be non-increasing"
  assert relief[-1] == pytest.approx(0.0), "relief must reach zero at steady state"
  for _ in range(19):
    state, relief = call(state)
  assert state.scale == pytest.approx(1.0)


def test_gate_v_ego_below_minimum():
  state, relief = call(v_ego=14.0)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_v_lead_below_minimum():
  state, relief = call(v_lead=4.0)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_closing_speed_too_low():
  state, relief = call(v_lead=31.2)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_closing_speed_too_high():
  state, relief = call(v_lead=15.0)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_lead_decelerating_harder():
  state, relief = call(a_lead=-0.6)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_ttc_below_minimum():
  state, relief = call(v_lead=16.5, gap=55.0)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_distance_above_comfort_short():
  state, relief = call(gap=45.0)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_required_decel_above_maximum():
  state, relief = call(v_lead=16.5, gap=61.0)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_plan_still_violates_boundary_late():
  # Smaller gap: a_req larger, plan cannot recover the static boundary by 2 s.
  state, relief = call(gap=84.5)
  assert state.scale == 0.0 and not np.any(relief)


def test_gate_early_margin_worsened_by_more_than_half_metre():
  v, vl, tf, tfd, fac, cb, sd = 31.5, 21.6, 2.2, 1.4, 0.8, 2.4, 6.0
  gap = 96.0
  c = v - vl
  d = gap - (tf * vl + sd)
  a_req = c * c / (2 * d)
  a_s = float(np.clip(1.25 * a_req, 0.5, 1.6))
  v_p = np.maximum(vl, v - a_s * TIMES)
  x_p = v * TIMES - 0.5 * a_s * TIMES ** 2
  stop_t = c / a_s
  late = TIMES > stop_t
  x_p[late] = v * stop_t - 0.5 * a_s * stop_t ** 2 + vl * (TIMES[late] - stop_t)
  lead_x = gap + vl * TIMES
  lead_obstacle = lead_x + vl * vl / (2 * cb)
  danger = lead_obstacle - x_p - fac * fla.get_safe_obstacle_distance(v_p, tfd, cb, sd)
  physical = lead_x - x_p
  # The base plan passes; forcing its early minimum 0.6 below min(margin0, 3) must fail.
  assert approach_path_safe(TIMES, danger, physical, v_p, sd)
  danger_bad = danger.copy()
  danger_bad[1] = min(float(danger[0]), 3.0) - 0.6
  assert not approach_path_safe(TIMES, danger_bad, physical, v_p, sd)


def test_gate_physical_clearance_violation():
  v, sd = 31.5, 6.0
  lead_x = np.full_like(TIMES, 100.0)
  x_p = lead_x - (sd + 0.7 * v + 1.0) * np.ones_like(TIMES)
  danger = np.full_like(TIMES, 10.0)
  assert not approach_path_safe(TIMES, danger, lead_x - x_p, x_p, sd)


def test_gate_nan_input():
  state, relief = call(gap=float('nan'))
  assert state.scale == 0.0 and not np.any(relief)


def test_dropout_hold_keeps_scale_for_half_second_then_decays():
  state, _ = call()
  for _ in range(20):
    state, _ = call(state)
  assert state.scale == pytest.approx(1.0)
  for _ in range(10):
    state, relief = call(state, lead_present=False)
    assert not np.any(relief)
  assert state.scale > 0.0
  for _ in range(22):
    state, _ = call(state, lead_present=False)
  assert state.scale == 0.0


def test_disabled_switch_returns_zeros(monkeypatch):
  monkeypatch.setattr(fla, "FAR_LEAD_APPROACH", False)
  state, relief = call()
  assert state.scale == 0.0 and not np.any(relief)


def test_not_enabled_returns_zeros():
  state, relief = call(enabled=False)
  assert state.scale == 0.0 and not np.any(relief)
