"""LaneRecenterAssist: unit tests for the opt-in laneless return assist.

LRA only adds curvature when the model path already returns to the lane centre
monotonically over its 3 s horizon.  These are pure-function and single-frame
state tests, not closed-loop vehicle validation.
"""
import copy
import importlib.util
import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.lane_center_blend import (
  BOOST_S, ENTER_HOLD_S, EXIT_HOLD_S, GATE_OFF, GATE_STEER, JUMP_HOLD_S, LRA_ACC_MAX, LRA_JERK,
  R_MONO_TOL, R_OFF, R_ON, R_REENTER_S, RET_MIN, STEER_BLOCK_HOLD_S, WIDTH_DELTA_MAX,
  LaneCenterBlend, dump_table, load_table, recenter_profile, recenter_return_checks,
  recenter_width_delta,
)
from openpilot.selfdrive.modeld.constants import ModelConstants

T = np.array(ModelConstants.T_IDXS)
V = 25.0
WIDTH = 3.5
SETTLE = int(math.ceil(ENTER_HOLD_S / DT_MDL)) + 2
BASE_COMMIT = '0f68e5624'


def _geometry(r0, r_horizon=None, v=V, kappa=0.0, width=WIDTH, widen=0.0, centre_shift=0.0):
  """Lane centre and a model path whose offset ramps from r0 to r_horizon.

  `widen` grows the right line over the first ~45 m (a taper/gore), and
  `centre_shift` moves the whole lane sideways (a lane-centre jump).
  """
  x = v * T
  centre = centre_shift + kappa * x ** 2 / 2.0
  r_end = r0 if r_horizon is None else r_horizon
  frac = np.clip(T / 3.0, 0.0, 1.0)
  r = r0 + (r_end - r0) * frac
  # Symmetric widening: the width changes without moving the lane centre, so
  # this test isolates the taper/gore guard from the sign/monotone releases.
  grow = widen * np.clip(x / 45.0, 0.0, 1.0)
  return x, centre - width / 2.0 - grow / 2.0, centre + width / 2.0 + grow / 2.0, x, centre + r, centre


def _update(blend, r0, r_horizon=None, *, recenter=True, v=V, kappa=0.0, lane_mode=False,
            lane_change=False, steering=False, l_prob=0.9, r_prob=0.9, width=WIDTH,
            widen=0.0, centre_shift=0.0, enabled=False):
  ll_x, lll, rll, path_x, path_y, centre = _geometry(r0, r_horizon, v, kappa, width, widen,
                                                     centre_shift)
  return blend.update(enabled, lane_mode, v, l_prob, r_prob, ll_x, lll, rll,
                      T, path_x, path_y, lane_change, steering, recenter_enabled=recenter)


def _settle(blend, r0, r_horizon=None, frames=SETTLE, **kwargs):
  out = 0.0
  for _ in range(frames):
    out = _update(blend, r0, r_horizon, **kwargs)
  return out


def _update_path(blend, r_curve, *, v=V, enabled=False, lane_mode=False, lane_change=False,
                 steering=False, l_prob=0.9, r_prob=0.9, width=WIDTH):
  """Drive the blend with an explicit r(t) profile instead of a linear ramp."""
  x = v * T
  centre = np.zeros_like(x)
  r = np.array([r_curve(float(t)) for t in T])
  lll = centre - width / 2.0
  rll = centre + width / 2.0
  return blend.update(enabled, lane_mode, v, l_prob, r_prob, x, lll, rll,
                      T, x, centre + r, lane_change, steering, recenter_enabled=True)


def test_profile_and_checks_match_sign_convention():
  ll_x, lll, rll, path_x, path_y, centre = _geometry(0.4, 0.1)
  t, r = recenter_profile(ll_x, (lll + rll) / 2.0, T, path_x, path_y)
  assert t is not None
  assert r[0] == pytest.approx(0.4)
  sign_ok, mono_ok, retract_ok = recenter_return_checks(t, r, float(r[0]))
  assert sign_ok and mono_ok and retract_ok

  # A positive r0 is a path right of centre; the correction must steer left.
  blend = LaneCenterBlend()
  _settle(blend, 0.4, 0.1)
  assert blend.recenter_active
  assert blend.recenter_curvature < 0.0

  blend = LaneCenterBlend()
  _settle(blend, -0.4, -0.1)
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
  _settle(blend, 0.4, 0.1)
  assert blend.recenter_curvature != 0.0
  _update(blend, 0.4, 0.1, recenter=False)
  assert blend.recenter_curvature == 0.0
  assert blend.recenter_k == 0.0


def test_enter_debounce_requires_the_full_hold():
  blend = LaneCenterBlend()
  hold_frames = int(round(ENTER_HOLD_S / DT_MDL))
  _update(blend, 0.4, 0.1)
  assert not blend.recenter_latched
  for _ in range(hold_frames - 2):
    _update(blend, 0.4, 0.1)
    assert not blend.recenter_latched
  _update(blend, 0.4, 0.1)
  assert blend.recenter_latched
  assert blend.recenter_curvature != 0.0

  # One failing frame restarts the count.
  blend = LaneCenterBlend()
  for _ in range(hold_frames - 1):
    _update(blend, 0.4, 0.1)
  _update(blend, 0.05, 0.0)  # resets the streak
  for _ in range(hold_frames - 1):
    _update(blend, 0.4, 0.1)
  assert not blend.recenter_latched
  _update(blend, 0.4, 0.1)
  assert blend.recenter_latched


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
  for _ in range(SETTLE):
    blend.update(False, False, V, 0.9, 0.9, x, centre - WIDTH / 2, centre + WIDTH / 2,
                 T, x, path_y, False, False, recenter_enabled=True)
    assert blend.recenter_curvature == 0.0


@pytest.mark.parametrize("r0", [0.10, 0.149])
def test_below_r_on_does_not_activate(r0):
  assert r0 < R_ON
  blend = LaneCenterBlend()
  _settle(blend, r0, 0.0, frames=60)
  assert blend.recenter_curvature == 0.0


@pytest.mark.parametrize("r0", [0.81, 1.0])
def test_above_r_max_does_not_activate(r0):
  blend = LaneCenterBlend()
  _settle(blend, r0, 0.2, frames=60)
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


def test_width_delta_helper():
  ll_x, lll, rll, path_x, path_y, centre = _geometry(0.0, 0.0)
  assert recenter_width_delta(ll_x, lll, rll, V) == pytest.approx(0.0)
  ll_x, lll, rll, path_x, path_y, centre = _geometry(0.0, 0.0, widen=0.8)
  x_l = float(np.clip(V * 1.5, 15.0, 45.0))
  expected = 0.8 * np.clip(x_l / 45.0, 0.0, 1.0)
  assert recenter_width_delta(ll_x, lll, rll, V) == pytest.approx(expected)
  assert abs(recenter_width_delta(ll_x, lll, rll, V)) > WIDTH_DELTA_MAX


def test_widening_lane_blocks_entry():
  blend = LaneCenterBlend()
  _settle(blend, 0.4, 0.0, frames=60, widen=0.8)
  assert blend.recenter_curvature == 0.0
  assert not blend.recenter_latched
  assert blend.recenter_width_delta > WIDTH_DELTA_MAX


@pytest.mark.parametrize("enabled", [False, True])
def test_widening_lane_releases_a_latched_assist(enabled):
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  _settle(blend, 0.4, 0.0, enabled=enabled)
  assert blend.recenter_latched
  before = blend.recenter_releases.get('width', 0)
  _settle(blend, 0.4, 0.0, frames=2, enabled=enabled, widen=0.8)
  assert not blend.recenter_latched
  assert blend.recenter_releases.get('width', 0) == before + 1
  assert blend.recenter_hold_s > 0.0


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
  _settle(blend, 0.20, 0.20 - RET_MIN + 0.01, frames=60)
  assert blend.recenter_curvature == 0.0


def test_caps_hold():
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
  _settle(blend, 0.4, 0.1, frames=200)
  assert blend.recenter_active
  y_m = 0.4 + (0.1 - 0.4) * 1.5 / 3.0
  length = float(np.clip(V * 1.5, 15.0, 45.0))
  expected = 2.0 * ((BOOST_S - 1.0) * y_m) / length ** 2
  assert blend.recenter_curvature == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("enabled", [False, True])
def test_inside_the_hysteresis_band_the_intent_drops_but_the_latch_holds(enabled):
  """R_ON..R_OFF is a dead band: no ACTIVE (that would leak) but no release."""
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  _settle(blend, 0.4, 0.0, enabled=enabled)
  assert blend.recenter_latched
  assert blend.recenter_active
  k_prev = blend.recenter_curvature

  r_in_band = (R_ON + R_OFF) / 2.0
  _settle(blend, r_in_band, 0.0, frames=1, enabled=enabled)
  assert blend.recenter_latched, 'the latch must survive between R_OFF and R_ON'
  assert blend.recenter_active is False
  assert abs(blend.recenter_curvature) <= abs(k_prev) + 1e-12
  assert 'r_off' not in blend.recenter_releases

  # Still latched a second later: this is hysteresis, not a release.
  _settle(blend, r_in_band, 0.0, frames=1, enabled=enabled)
  assert blend.recenter_latched
  assert abs(r_in_band) < R_ON and abs(r_in_band) > R_OFF


def test_turn_off_is_slewed_not_stepped():
  blend = LaneCenterBlend()
  _settle(blend, 0.4, 0.0, frames=200)
  active_k = blend.recenter_curvature
  assert abs(active_k) > 0.0

  slew = LRA_JERK / V ** 2 * DT_MDL
  prev = active_k
  steps = 0
  for _ in range(400):
    _update(blend, R_OFF * 0.5, 0.0)  # below R_OFF: latch releases
    k = blend.recenter_curvature
    assert abs(k - prev) <= slew + 1e-12
    assert abs(k) <= abs(prev) + 1e-12
    prev = k
    steps += 1
    if k == 0.0:
      break
  assert steps > 1
  assert blend.recenter_curvature == 0.0


@pytest.mark.parametrize("enabled", [False, True])
def test_steer_hold_then_early_release_gives_zero(enabled):
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  _settle(blend, 0.4, 0.0, enabled=enabled)
  assert blend.recenter_latched

  _settle(blend, 0.4, 0.0, frames=12, enabled=enabled, steering=True)  # 0.6 s hold
  assert blend.steer_block
  assert not blend.recenter_latched
  assert blend.recenter_active is False
  _settle(blend, 0.4, 0.0, frames=2, enabled=enabled, steering=True)
  assert blend.recenter_curvature == 0.0
  assert (blend.recenter_releases.get('steer', 0) +
          blend.recenter_releases.get('gate_steer', 0)) >= 1

  _settle(blend, 0.4, 0.0, frames=15, enabled=enabled)  # 0.75 s after release
  assert blend.steer_block
  assert not blend.recenter_latched
  assert blend.recenter_curvature == 0.0

  _settle(blend, 0.4, 0.0, frames=int(math.ceil(STEER_BLOCK_HOLD_S / DT_MDL)) + 25,
          enabled=enabled)
  assert not blend.steer_block
  assert blend.recenter_latched


@pytest.mark.parametrize("enabled", [False, True])
def test_centre_jump_gives_zero_for_its_hold(enabled):
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  _settle(blend, 0.4, 0.0, enabled=enabled)
  assert blend.recenter_latched

  _settle(blend, 0.4, 0.0, frames=2, enabled=enabled, centre_shift=0.4)  # centre jumps 0.4 m
  assert blend.jump_t > 0.0
  assert not blend.recenter_latched
  assert blend.recenter_active is False
  # The intent is already 0; the previous curvature only slews down.
  for _ in range(2):
    _settle(blend, 0.4, 0.0, frames=1, enabled=enabled, centre_shift=0.4)
    assert blend.recenter_active is False
    assert blend.recenter_curvature == 0.0

  frames = int(round(JUMP_HOLD_S / DT_MDL)) - 5
  _settle(blend, 0.4, 0.0, frames=frames, enabled=enabled)
  assert not blend.recenter_latched
  assert blend.recenter_curvature == 0.0

  _settle(blend, 0.4, 0.0, frames=SETTLE + 5, enabled=enabled)
  assert blend.recenter_latched
  assert blend.recenter_curvature != 0.0


@pytest.mark.parametrize("reason", ['mono', 'sign', 'retract'])
def test_every_release_starts_the_hold_off(reason):
  blend = LaneCenterBlend()
  _settle(blend, 0.4, 0.0)
  assert blend.recenter_latched

  # Soft reasons (mono/retract) only release after EXIT_HOLD_S of failure.
  soft_frames = int(round(EXIT_HOLD_S / DT_MDL)) + 1
  if reason == 'mono':
    _settle(blend, 0.4, 0.9, frames=soft_frames)   # |r| grows across the horizon
  elif reason == 'sign':
    x = V * T
    centre = np.zeros_like(x)
    path_y = np.where(T <= 1.5, 0.4, -0.3)
    blend.update(False, False, V, 0.9, 0.9, x, centre - WIDTH / 2, centre + WIDTH / 2,
                 T, x, path_y, False, False, recenter_enabled=True)
  else:
    _settle(blend, 0.2, 0.19, frames=soft_frames)

  assert not blend.recenter_latched
  assert blend.recenter_releases.get(reason, 0) == 1
  assert 0.0 < blend.recenter_hold_s <= R_REENTER_S

  # The hold-off blocks re-entry even though the entry condition holds again.
  _settle(blend, 0.4, 0.0, frames=max(1, int(round(R_REENTER_S / DT_MDL)) - 1))
  assert not blend.recenter_latched
  _settle(blend, 0.4, 0.0, frames=SETTLE)
  assert blend.recenter_latched


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("soft", ['mono', 'retract', 'rate'])
def test_a_soft_frame_stops_output_but_keeps_the_latch(soft, enabled):
  """R3-1: one failing soft frame zeroes the target at once, latch survives."""
  blend = LaneCenterBlend(table=np.full(7, 0.1)) if enabled else LaneCenterBlend()
  _settle(blend, 0.4, 0.0, enabled=enabled)
  assert blend.recenter_latched
  assert blend.recenter_curvature != 0.0
  k_before = blend.recenter_curvature
  releases_before = dict(blend.recenter_releases)

  if soft == 'mono':
    _settle(blend, 0.4, 0.9, frames=1, enabled=enabled)
  elif soft == 'retract':
    _settle(blend, 0.2, 0.19, frames=1, enabled=enabled)
  else:
    # Rate away: r at 1.5 s grows 0.02 m per frame (0.4 m/s after the filter)
    # while |r(3 s)| keeps shrinking, so every *other* check still passes and
    # rate_away is the only failing condition.
    r0 = [0.4]

    def r_curve(t):
      rise = 0.02 * np.clip(t / 1.5, 0.0, 1.0)
      fall = 0.1 * np.clip((t - 1.5) / 1.5, 0.0, 1.0)
      return r0[0] + rise - fall

    # EXIT_HOLD_S is 0.2 s (4 frames); stay strictly under it.
    for _ in range(int(round(EXIT_HOLD_S / DT_MDL)) - 1):
      r0[0] += 0.02
      _update_path(blend, r_curve, enabled=enabled)

  assert blend.recenter_latched, f'{soft}: a single soft frame must not release'
  assert blend.recenter_active is False, f'{soft}: intent must be False when target_k is 0'
  assert abs(blend.recenter_curvature) < abs(k_before), f'{soft}: output must slew down at once'
  assert blend.recenter_releases == releases_before


@pytest.mark.parametrize("enabled", [False, True])
def test_soft_failure_shorter_than_exit_hold_resumes_without_a_new_wait(enabled):
  blend = LaneCenterBlend(table=np.full(7, 0.1)) if enabled else LaneCenterBlend()
  _settle(blend, 0.4, 0.0, enabled=enabled)
  assert blend.recenter_latched

  # Fail mono for one frame less than EXIT_HOLD_S, then return to a good return.
  fail_frames = max(1, int(round(EXIT_HOLD_S / DT_MDL)) - 1)
  _settle(blend, 0.4, 0.9, frames=fail_frames, enabled=enabled)
  assert blend.recenter_latched

  _settle(blend, 0.4, 0.0, frames=1, enabled=enabled)
  # No new 0.2 s entry wait: the latch never dropped.
  assert blend.recenter_latched
  assert blend.recenter_active is True
  assert abs(blend.recenter_curvature) > 0.0


@pytest.mark.parametrize("enabled", [False, True])
def test_soft_failure_for_exit_hold_releases_with_the_hold_off(enabled):
  blend = LaneCenterBlend(table=np.full(7, 0.1)) if enabled else LaneCenterBlend()
  _settle(blend, 0.4, 0.0, enabled=enabled)
  assert blend.recenter_latched
  # EXIT_HOLD_S is reached after exactly this many consecutive failing frames.
  deadline = int(round(EXIT_HOLD_S / DT_MDL))
  _settle(blend, 0.4, 0.9, frames=deadline - 1, enabled=enabled)
  assert blend.recenter_latched
  _settle(blend, 0.4, 0.9, frames=1, enabled=enabled)
  assert not blend.recenter_latched
  assert blend.recenter_releases.get('mono', 0) == 1
  assert 0.0 < blend.recenter_hold_s <= R_REENTER_S


def test_above_r_max_release_starts_the_hold_off():
  blend = LaneCenterBlend()
  _settle(blend, 0.4, 0.0)
  assert blend.recenter_latched
  # Prime r_rate with the new geometry first, so the release is r_max and not
  # the rate criterion.
  _settle(blend, 0.85, 0.3, frames=3, recenter=False)
  blend.recenter_latched = True
  blend.recenter_hold_s = 0.0
  _settle(blend, 0.85, 0.3, frames=2)
  assert not blend.recenter_latched
  assert blend.recenter_releases.get('r_max', 0) == 1
  assert 0.0 < blend.recenter_hold_s <= R_REENTER_S


def test_assist_runs_with_blend_disabled_and_leaves_blend_outputs_off():
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  _settle(blend, 0.4, 0.1)
  assert blend.recenter_curvature != 0.0
  assert blend.curvature == 0.0
  assert blend.offset == 0.0
  assert blend.w == 0.0

  ref = LaneCenterBlend(table=np.full(7, 0.1))
  _settle(ref, 0.4, 0.1, recenter=False)
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


def _load_base(repo_root):
  """Load the 0f68e5624 LaneCenterBlend, or skip when git cannot provide it."""
  git = shutil.which('git')
  if git is None:
    pytest.skip('git not available')
  try:
    src = subprocess.run(
      [git, '-C', str(repo_root), 'show',
       f'{BASE_COMMIT}:openpilot/selfdrive/controls/lib/lane_center_blend.py'],
      check=True, capture_output=True, text=True).stdout
  except (subprocess.CalledProcessError, OSError) as exc:
    pytest.skip(f'base commit unavailable: {exc}')

  # The file only imports numpy and DT_MDL, both already imported here.
  work = tempfile.mkdtemp(prefix='lcb_base_')
  path = os.path.join(work, 'lane_center_blend_base.py')
  with open(path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(src)
  spec = importlib.util.spec_from_file_location('lane_center_blend_base', path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


BLEND_FIELDS = ('w', 'offset', 'curvature', 'gates', 'e', 'kappa_road', 'e_prev', 'r_prev',
                'r_rate', 'r_far', 'r_dev', 'lane_change_t', 'jump_t', 'avoid_clear_t',
                'steer_held_s', 'steer_block', 'steer_release_s', 'table', 'r_base')


def test_random_two_thousand_frames_match_the_base_blend_path():
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
  base = _load_base(repo_root)
  rng = np.random.default_rng(146)
  cur = LaneCenterBlend(table=np.full(7, 0.05))
  ref = base.LaneCenterBlend(table=np.full(7, 0.05))

  r0 = 0.0
  r_h = 0.0
  kappa = 0.0
  width = WIDTH
  for frame in range(2000):
    if frame % 20 == 0:
      r0 = float(rng.uniform(-0.6, 0.6))
      r_h = float(r0 + rng.uniform(-0.25, 0.05))
      kappa = float(rng.choice([0.0, 0.0, 0.0005, -0.0005, 0.003, -0.003, 0.025]))
      width = float(rng.uniform(2.2, 4.6))
    v = float(rng.uniform(4.0, 34.0))
    l_prob = float(rng.uniform(0.3, 1.0))
    r_prob = float(rng.uniform(0.3, 1.0))
    lane_mode = bool(rng.random() < 0.15)
    lane_change = bool(rng.random() < 0.10)
    steering = bool(rng.random() < 0.10)
    enabled = bool(rng.random() < 0.80)
    x = v * T
    centre = kappa * x ** 2 / 2.0
    frac = np.clip(T / 3.0, 0.0, 1.0)
    path_y = centre + r0 + (r_h - r0) * frac
    lll = centre - width / 2.0
    rll = centre + width / 2.0

    a = cur.update(enabled, lane_mode, v, l_prob, r_prob, x, lll, rll, T, x, path_y,
                   lane_change, steering, recenter_enabled=False)
    b = ref.update(enabled, lane_mode, v, l_prob, r_prob, x, lll, rll, T, x, path_y,
                   lane_change, steering)
    assert a == b, frame
    for name in BLEND_FIELDS:
      av, bv = getattr(cur, name), getattr(ref, name)
      if isinstance(av, np.ndarray):
        assert np.array_equal(av, bv), (frame, name)
      else:
        assert av == bv, (frame, name)

  assert cur.recenter_curvature == 0.0


def test_table_roundtrip_unchanged():
  table = np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.25, -0.05])
  assert np.allclose(load_table(dump_table(table)), table)
  assert np.array_equal(load_table(''), np.zeros(7))
  assert np.allclose(load_table('[0.9,0.9,0.9,0.9,0.9,0.9,0.9]'), np.full(7, 0.30))


def test_steer_gate_bit_is_used_in_both_branches():
  """The blend gate field stays GATE_OFF when the blend is off; the assist must
  still see the steer bit.  Latching then holding the wheel releases it."""
  assert GATE_STEER == 256
  for enabled in (False, True):
    blend = LaneCenterBlend(table=np.full(7, 0.1))
    _settle(blend, 0.4, 0.0, enabled=enabled)
    assert blend.recenter_latched
    _settle(blend, 0.4, 0.0, frames=12, enabled=enabled, steering=True)
    assert blend.steer_block
    assert not blend.recenter_latched
    assert blend.recenter_active is False
    if enabled:
      assert blend.gates & GATE_STEER
    else:
      assert blend.gates == GATE_OFF

  # And the steer bit alone blocks entry from a fresh state.
  for enabled in (False, True):
    blend = LaneCenterBlend()
    _settle(blend, 0.4, 0.0, frames=60, enabled=enabled, steering=True)
    assert not blend.recenter_latched
    assert blend.recenter_curvature == 0.0
