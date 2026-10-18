"""LaneRecenterAssist: unit tests for the opt-in laneless return assist.

LRA only adds curvature when the model path already returns to the lane centre
monotonically over its 3 s horizon.  These are pure-function and single-frame
state tests, not closed-loop vehicle validation.
"""
import copy

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.lane_center_blend import (
  BOOST_S, GATE_OFF, LRA_ACC_MAX, LRA_JERK, R_MAX, R_MONO_TOL, R_OFF, R_ON, RET_MIN,
  LaneCenterBlend, recenter_profile, recenter_return_checks,
)
from openpilot.selfdrive.modeld.constants import ModelConstants

T = np.array(ModelConstants.T_IDXS)
V = 25.0
WIDTH = 3.5


def _geometry(r0, r_horizon=None, v=V, kappa=0.0, width=WIDTH):
  """Lane centre and a model path whose offset ramps from r0 to r_horizon."""
  x = v * T
  centre = kappa * x ** 2 / 2.0
  r_end = r0 if r_horizon is None else r_horizon
  t_end = 3.0
  frac = np.clip(T / t_end, 0.0, 1.0)
  r = r0 + (r_end - r0) * frac
  return x, centre - width / 2.0, centre + width / 2.0, x, centre + r, centre


def _update(blend, r0, r_horizon=None, *, recenter=True, v=V, kappa=0.0, lane_mode=False,
            lane_change=False, steering=False, l_prob=0.9, r_prob=0.9, width=WIDTH):
  ll_x, lll, rll, path_x, path_y, centre = _geometry(r0, r_horizon, v, kappa, width)
  return blend.update(False, lane_mode, v, l_prob, r_prob, ll_x, lll, rll,
                      T, path_x, path_y, lane_change, steering, recenter_enabled=recenter)


def test_profile_and_checks_match_sign_convention():
  ll_x, lll, rll, path_x, path_y, centre = _geometry(0.4, 0.1)
  t, r = recenter_profile(ll_x, (lll + rll) / 2.0, T, path_x, path_y)
  assert t is not None
  assert r[0] == pytest.approx(0.4)
  sign_ok, mono_ok, retract_ok = recenter_return_checks(t, r, float(r[0]))
  assert sign_ok and mono_ok and retract_ok

  # A positive r0 is a path right of centre; the correction must steer left.
  blend = LaneCenterBlend()
  _update(blend, 0.4, 0.1)
  assert blend.recenter_active
  assert blend.recenter_curvature < 0.0

  blend = LaneCenterBlend()
  _update(blend, -0.4, -0.1)
  assert blend.recenter_active
  assert blend.recenter_curvature > 0.0


def test_off_is_identical_and_exact_zero():
  for r0 in (0.0, 0.2, 0.5):
    blend = LaneCenterBlend(table=np.full(7, 0.1))
    before = blend.table.copy()
    out = _update(blend, r0, 0.0, recenter=False)
    assert out == 0.0
    assert blend.recenter_curvature == 0.0
    assert blend.recenter_active is False
    assert np.array_equal(blend.table, before)
    assert blend.gates == GATE_OFF

  # Turning the assist off after it was active steps it to exactly zero.
  blend = LaneCenterBlend()
  _update(blend, 0.4, 0.1)
  assert blend.recenter_curvature != 0.0
  _update(blend, 0.4, 0.1, recenter=False)
  assert blend.recenter_curvature == 0.0
  assert blend.recenter_k == 0.0


def test_hold_offset_does_not_activate():
  blend = LaneCenterBlend()
  for _ in range(60):
    _update(blend, 0.4, 0.4)
    assert blend.recenter_curvature == 0.0
    assert not blend.recenter_latched


def test_swinging_further_out_does_not_activate():
  blend = LaneCenterBlend()
  for _ in range(60):
    _update(blend, 0.4, 0.7)
    assert blend.recenter_curvature == 0.0


def test_sign_flip_inside_horizon_does_not_activate():
  x = V * T
  centre = np.zeros_like(x)
  path_y = np.where(T <= 1.5, 0.4, -0.3)
  blend = LaneCenterBlend()
  for _ in range(30):
    blend.update(False, False, V, 0.9, 0.9, x, centre - WIDTH / 2, centre + WIDTH / 2,
                 T, x, path_y, False, False, recenter_enabled=True)
    assert blend.recenter_curvature == 0.0


@pytest.mark.parametrize("r0", [0.10, 0.149])
def test_below_r_on_does_not_activate(r0):
  blend = LaneCenterBlend()
  for _ in range(30):
    _update(blend, r0, 0.0)
    assert blend.recenter_curvature == 0.0


@pytest.mark.parametrize("r0", [0.81, 1.0])
def test_above_r_max_does_not_activate(r0):
  blend = LaneCenterBlend()
  for _ in range(30):
    _update(blend, r0, 0.2)
    assert blend.recenter_curvature == 0.0


def test_gates_block_activation():
  cases = (
    {"lane_change": True},
    {"steering": True},
    {"v": 8.0},
    {"kappa": 0.002},
    {"l_prob": 0.5},
    {"r_prob": 0.5},
    {"width": 2.0},
    {"lane_mode": True},
  )
  for kwargs in cases:
    blend = LaneCenterBlend()
    for _ in range(30):
      _update(blend, 0.4, 0.1, **kwargs)
      assert blend.recenter_curvature == 0.0, kwargs
    assert not blend.recenter_latched, kwargs


def test_monotone_tolerance_boundary():
  t = np.array([0.0, 1.0, 2.0, 3.0])
  r0 = 0.4
  r = np.array([r0, r0 - 0.10, r0 - 0.10 + R_MONO_TOL, r0 - 0.25])
  sign_ok, mono_ok, retract_ok = recenter_return_checks(t, r, r0)
  assert sign_ok and mono_ok and retract_ok

  r_bad = np.array([r0, r0 - 0.10, r0 - 0.10 + R_MONO_TOL + 0.002, r0 - 0.25])
  _, mono_ok_bad, _ = recenter_return_checks(t, r_bad, r0)
  assert not mono_ok_bad


def test_insufficient_retraction_does_not_activate():
  blend = LaneCenterBlend()
  for _ in range(30):
    _update(blend, 0.20, 0.20 - RET_MIN + 0.01)
    assert blend.recenter_curvature == 0.0


def test_caps_hold():
  blend = LaneCenterBlend()
  prev = 0.0
  for v in (11.2, 20.0, 33.3):
    blend = LaneCenterBlend()
    prev = 0.0
    for _ in range(200):
      _update(blend, 0.8, 0.0, v=v)
      k = blend.recenter_curvature
      assert abs(k) * v ** 2 <= LRA_ACC_MAX + 1e-9
      assert abs(k - prev) * v ** 2 <= LRA_JERK * DT_MDL + 1e-9
      prev = k
    assert abs(blend.recenter_curvature) > 0.0


def test_boost_is_stronger_than_the_model_alone():
  blend = LaneCenterBlend()
  for _ in range(200):
    _update(blend, 0.4, 0.1)
  assert blend.recenter_active
  y_m = 0.4 + (0.1 - 0.4) * 1.5 / 3.0
  length = float(np.clip(V * 1.5, 15.0, 45.0))
  expected = 2.0 * ((BOOST_S - 1.0) * y_m) / length ** 2
  assert blend.recenter_curvature == pytest.approx(expected, rel=1e-6)


def test_turn_off_is_slewed_not_stepped():
  blend = LaneCenterBlend()
  for _ in range(200):
    _update(blend, 0.4, 0.1)
  active_k = blend.recenter_curvature
  assert abs(active_k) > 0.0

  slew = LRA_JERK / V ** 2 * DT_MDL
  prev = active_k
  steps = 0
  for _ in range(400):
    _update(blend, 0.05, 0.0)  # below R_OFF: latch releases
    k = blend.recenter_curvature
    assert abs(k - prev) <= slew + 1e-12
    assert abs(k) <= abs(prev) + 1e-12
    prev = k
    steps += 1
    if k == 0.0:
      break
  assert steps > 1
  assert blend.recenter_curvature == 0.0


def test_assist_runs_with_blend_disabled_and_leaves_blend_outputs_off():
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  for _ in range(30):
    _update(blend, 0.4, 0.1)
  assert blend.recenter_curvature != 0.0
  assert blend.curvature == 0.0
  assert blend.offset == 0.0
  assert blend.w == 0.0

  ref = LaneCenterBlend(table=np.full(7, 0.1))
  for _ in range(30):
    _update(ref, 0.4, 0.1, recenter=False)
  assert ref.curvature == blend.curvature == 0.0
  assert np.array_equal(ref.table, blend.table)


def test_blend_enabled_output_unchanged_by_assist_off():
  on = LaneCenterBlend()
  off = copy.deepcopy(on)
  x, lll, rll, path_x, path_y, centre = _geometry(0.4, 0.1)
  for _ in range(100):
    a = on.update(True, False, V, 0.9, 0.9, x, lll, rll, T, path_x, path_y, False, False,
                  recenter_enabled=False)
    b = off.update(True, False, V, 0.9, 0.9, x, lll, rll, T, path_x, path_y, False, False)
    assert a == b
  assert on.recenter_curvature == 0.0
