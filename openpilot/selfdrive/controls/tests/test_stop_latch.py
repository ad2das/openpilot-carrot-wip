"""Separate stop entry/release speeds (shouldStop hysteresis)."""
import ast
from pathlib import Path

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import (
  STOP_ENTER_V, STOP_LATCH_V_MAX, STOP_RELEASE_RISE, get_accel_from_plan,
  plan_stop_speeds, stop_latch,
)


def _stateless(v_target, v_target_1sec, release_v):
  return v_target < release_v and v_target_1sec < release_v


@pytest.mark.parametrize('v_target', [0.05, 0.14, 0.22, 0.29, 0.35, 0.6])
@pytest.mark.parametrize('v_target_1sec', [0.05, 0.13, 0.2, 0.31, 0.5])
@pytest.mark.parametrize('prev', [False, True])
@pytest.mark.parametrize('v_ego', [0.0, 0.3, 0.9, 1.2, float('nan')])
def test_release_at_or_above_enter_is_stateless(v_target, v_target_1sec, prev, v_ego):
  release_v = 0.5
  assert release_v >= STOP_ENTER_V
  assert stop_latch(prev, v_target, v_target_1sec, v_ego, release_v) == \
    _stateless(v_target, v_target_1sec, release_v)


def test_enter_below_enter_speed():
  assert stop_latch(False, 0.22, 0.20, 0.3, 0.10) is True
  assert stop_latch(False, 0.35, 0.20, 0.3, 0.10) is False


def test_latched_flat_crawl_stays_latched():
  assert stop_latch(True, 0.14, 0.13, 0.3, 0.10) is True
  # A rise of exactly 0.01 is below STOP_RELEASE_RISE, so it stays latched.
  assert STOP_RELEASE_RISE == 0.02
  assert stop_latch(True, 0.14, 0.15, 0.3, 0.10) is True


def test_latched_starting_plan_releases():
  # Released at the same point as today's 0.10 stateless rule would release.
  assert stop_latch(True, 0.06, 0.12, 0.3, 0.10) is False
  assert stop_latch(True, 0.05, 0.09, 0.3, 0.10) is True
  assert stop_latch(True, 0.05, 0.31, 0.3, 0.10) is False
  # Above the latch speed cap (and NaN v_ego) never keeps a latched stop.
  assert STOP_LATCH_V_MAX == 1.0
  assert stop_latch(True, 0.14, 0.13, 1.2, 0.10) is False
  assert stop_latch(True, 0.14, 0.13, float('nan'), 0.10) is False


def test_plan_stop_speeds_matches_get_accel_from_plan_interp():
  rng = np.random.default_rng(7)
  t_idxs = np.array([0., .2, .45, .7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5])
  for _ in range(200):
    speeds = rng.uniform(0.0, 1.2, size=t_idxs.size)
    accels = rng.uniform(-1.0, 0.5, size=t_idxs.size)
    action_t = rng.uniform(0.0, 0.6)
    v_target, v_target_1sec = plan_stop_speeds(speeds, t_idxs, action_t)
    # get_accel_from_plan computes the same two interps identically.
    ref_target = np.interp(action_t, t_idxs, speeds)
    ref_target_1sec = np.interp(action_t + 1.0, t_idxs, speeds)
    assert abs(v_target - ref_target) < 1e-12
    assert abs(v_target_1sec - ref_target_1sec) < 1e-12
    _, should_stop, _, _ = get_accel_from_plan(speeds, accels, t_idxs, action_t=action_t, vEgoStopping=0.1)
    assert should_stop == (ref_target < 0.1 and ref_target_1sec < 0.1)


def test_plan_stop_speeds_length_mismatch_returns_zeros():
  assert plan_stop_speeds([0.1, 0.2], np.array([0., 1., 2.]), 0.1) == (0.0, 0.0)
  assert plan_stop_speeds([0.1], np.array([0., 1.]), 0.1) == (0.0, 0.0)


def test_reset_state_clears_the_latch():
  path = Path(__file__).resolve().parents[1] / 'lib/longitudinal_planner.py'
  tree = ast.parse(path.read_text(encoding='utf-8'))
  update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'update')
  reset_if = next(n for n in update.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'reset_state')
  assert any(isinstance(s, ast.Assign) and any(ast.unparse(t) == 'self._stop_latched' for t in s.targets)
             and ast.unparse(s.value) == 'False' for s in reset_if.body)


def test_stop_latch_uses_own_state_not_output_should_stop():
  path = Path(__file__).resolve().parents[1] / 'lib/longitudinal_planner.py'
  tree = ast.parse(path.read_text(encoding='utf-8'))
  update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'update')
  latch_assigns = [s for s in update.body if isinstance(s, ast.Assign)
                   and any(ast.unparse(t) == 'self._stop_latched' for t in s.targets)]
  assert latch_assigns, "latch assignment not found"
  assert any('stop_latch(self._stop_latched' in ast.unparse(s.value) for s in latch_assigns)
  assert all('self.output_should_stop' not in ast.unparse(s.value) for s in latch_assigns)
