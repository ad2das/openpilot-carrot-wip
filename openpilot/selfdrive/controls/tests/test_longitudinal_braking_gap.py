import math

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib import longitudinal_braking_gap
from openpilot.selfdrive.controls.lib.longitudinal_braking_gap import BrakingGapState, braking_gap_reference


def update(state, **kwargs):
  args = {'enabled': True, 'track_id': 43, 'dt': .05, 'distance': 20., 'ego_speed': 15., 'ego_accel': -.5,
              'lead_speed': 14.5, 'lead_accel': -.5, 'relative_speed': -.5, 'base_tf': .6, 'stop_distance': 5., 'desired_distance': 17.}
  args.update(kwargs)
  return state.update(**args)


def test_distance_memory_survives_falling_speed_without_accumulating_each_frame(monkeypatch):
  # Pin the speed cap off: this test isolates the falling-speed memory mechanic;
  # the speed-proportional cap has dedicated coverage below.
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', float('inf'))
  state = BrakingGapState()
  for _ in range(20):
    update(state)
  captured = state.floor
  for i in range(150):
    update(state, ego_speed=max(2., 15. - i * .08), lead_speed=max(1.5, 14.5 - i * .08), desired_distance=10.)
  assert 10. < state.floor < captured <= 20.
  assert 0. < state.margin < 10.


@pytest.mark.parametrize('change', [{'enabled': False}, {'track_id': 44}, {'distance': 100.}, {'lead_accel': float('nan')}])
def test_lost_replaced_or_jumped_track_cannot_inherit_distance_memory(change):
  state = BrakingGapState()
  for _ in range(20):
    update(state)
  assert state.margin > 0.
  assert update(state, **change) == 0.
  assert state.floor is None


def test_opening_and_constant_speed_do_not_create_braking_memory():
  for change in [{'relative_speed': .5}, {'ego_accel': 0.}, {'lead_accel': 0.}]:
    state = BrakingGapState()
    for _ in range(40):
      assert update(state, **change) == 0.


def test_prediction_does_not_mutate_live_state_and_matches_repeated_advances():
  state = BrakingGapState()
  for _ in range(30):
    update(state)
  before = vars(state).copy()
  t = np.arange(21) * .05
  actual = state.margins(t, np.full(21, 10.), np.full(21, 9.5), np.full(21, 10.), .6, 5.)
  assert vars(state) == before
  expected = [state.margin]
  for _ in range(20):
    state._advance(.05, 11., 10., 10., 9.5, 5.)
    expected.append(state.margin)
  np.testing.assert_allclose(actual, expected)


def relief(state, **changes):
  args = {'dt': .05, 'enabled': True, 'base_tf': 1.1, 'ego_speeds': np.full(13, 20.), 'lead_speeds': np.full(13, 19.),
              'lead_accel': -1., 'physical_gaps': np.full(13, 40.), 'danger_margins': np.full(13, 10.),
              'times': np.linspace(0, 10, 13), 'stop_distance': 5.}
  args.update(changes)
  return state.comfort_relief(**args)


def test_comfort_allowance_is_bounded_and_danger_withdraws_it_immediately():
  state = BrakingGapState()
  state.age, state.distance, state.relative_speed = 2., 40., -1.
  for _ in range(20):
    result = relief(state)
  np.testing.assert_allclose(result, 8.)
  assert not relief(state, danger_margins=np.full(13, -1.)).any()
  assert state.relief == 0.


@pytest.mark.parametrize('change', [{'base_tf': .45}, {'enabled': False}, {'lead_speeds': np.zeros(13)},
                                    {'physical_gaps': np.full(13, 10.)}, {'lead_accel': 0.}])
def test_short_gap_stopped_lead_and_invalid_eligibility_never_get_relief(change):
  state = BrakingGapState()
  state.age, state.distance, state.relative_speed = 2., 40., -1.
  assert not relief(state, **change).any()


def test_slack_does_not_move_physical_obstacles_or_relax_another_obstacle():
  obstacles = np.array([[50., 30., 100.], [20., 30., 19.]])
  original = obstacles.copy()
  result = braking_gap_reference(obstacles, np.array([[-5., 0.], [-5., -5.]]), np.array([20., 20.]))
  np.testing.assert_array_equal(obstacles, original)
  np.testing.assert_array_equal(result, 0.)


def test_stopped_pair_retains_distance_until_departure():
  state = BrakingGapState()
  for _ in range(20):
    update(state)
  captured = state.floor
  for _ in range(600):
    update(state, distance=20., ego_speed=0., ego_accel=0., lead_speed=0., lead_accel=0.,
           relative_speed=0., desired_distance=5.)
  assert state.floor == captured
  held = state.margin
  for _ in range(400):
    update(state, distance=20., ego_speed=0., ego_accel=0., lead_speed=1., lead_accel=.5,
           relative_speed=1., desired_distance=5.)
  assert state.margin < held * .02


@pytest.mark.parametrize('field', ['danger_margins', 'physical_gaps', 'ego_speeds'])
def test_nonfinite_prediction_withdraws_comfort_relief(field):
  state = BrakingGapState()
  state.age, state.distance, state.relative_speed = 2., 40., -1.
  relief(state)
  assert state.relief > 0.
  assert not relief(state, **{field: np.full(13, np.nan)}).any()


def braking_pair(state, **kwargs):
  args = {'enabled': True, 'track_id': 43, 'dt': .05, 'distance': 20., 'ego_speed': 5., 'ego_accel': -.5,
              'lead_speed': 4.5, 'lead_accel': -.5, 'relative_speed': -.5, 'base_tf': 2.0,
              'stop_distance': 5.5, 'desired_distance': 17.}
  args.update(kwargs)
  return state.update(**args)


def capture(state):
  for _ in range(10):
    braking_pair(state)
  assert state.floor is not None
  return state.floor


def test_stopped_lead_releases_the_braking_gap_memory_over_about_one_second(monkeypatch):
  # Pin the speed cap off: this test isolates the stopped-lead release tau.
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', float('inf'))
  state = BrakingGapState()
  captured = capture(state)
  assert 16.0 < captured <= 18.0  # bounded capture: min(20, base+10, sd+2.5*v) = 18
  for _ in range(60):
    braking_pair(state, ego_speed=1.5, ego_accel=-.5, lead_speed=0., lead_accel=0., relative_speed=-1.5)
  base = 5.5 + 1.5 * 2.0
  assert state.floor - base <= 0.5


def test_old_eight_second_tau_would_not_release_within_three_seconds(monkeypatch):
  monkeypatch.setattr(longitudinal_braking_gap, 'STOPPED_LEAD_RELEASE_S', 8.0)
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', float('inf'))
  state = BrakingGapState()
  capture(state)
  for _ in range(60):
    braking_pair(state, ego_speed=1.5, ego_accel=-.5, lead_speed=0., lead_accel=0., relative_speed=-1.5)
  base = 5.5 + 1.5 * 2.0
  assert state.floor - base > 4.0


def test_moving_lead_keeps_the_eight_second_tau(monkeypatch):
  # Pin the speed cap off: this test isolates the moving-lead tau selection;
  # the speed-proportional cap has dedicated coverage.
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', float('inf'))
  state = BrakingGapState()
  capture(state)
  for _ in range(60):
    braking_pair(state, ego_speed=1.5, ego_accel=-.5, lead_speed=2., lead_accel=0., relative_speed=-.5)
  base = 5.5 + 1.5 * 2.0
  assert state.floor - base > 4.0


def test_stopped_pair_below_point_three_still_freezes():
  state = BrakingGapState()
  capture(state)
  before = state.floor
  for _ in range(600):
    braking_pair(state, ego_speed=0.2, ego_accel=-.2, lead_speed=0.2, lead_accel=0., relative_speed=0.)
  assert state.floor == before


@pytest.mark.parametrize('offset', [0.0, 3.0])
def test_floor_tracks_speed_cap(monkeypatch, offset):
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', 2.5)
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_CAP_OFFSET_V', offset)
  state = BrakingGapState()
  captured = capture(state)
  assert captured >= 5.5 + 5.0 * 2.0 - 1e-6  # the cap is never below the base distance
  for i in range(1, 81):
    ego = 5.0 - .05 * i  # 5 -> 1 m/s over 4 s
    lead = .8 - .00375 * i  # 0.8 -> 0.5 m/s, never below 0.3
    braking_pair(state, ego_speed=ego, lead_speed=lead, ego_accel=-.5, lead_accel=-.5,
                 relative_speed=lead - ego)
    base = 5.5 + ego * 2.0
    assert state.floor <= max(base, 5.5 + 2.5 * max(0.0, ego - offset)) + 1e-6
    if offset > 0.0 and ego <= 3.0:
      assert abs(state.floor - base) <= 1e-6


def test_stopped_pair_freeze_beats_cap(monkeypatch):
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', 2.5)
  state = BrakingGapState()
  state.key, state.distance, state.relative_speed, state.age = 43, 20., 0., 1.
  state.floor = 9.0
  for _ in range(100):
    braking_pair(state, ego_speed=0.2, ego_accel=-.2, lead_speed=0.2, lead_accel=0., relative_speed=0.)
  assert state.floor == 9.0


def test_margins_apply_cap_with_predicted_speeds(monkeypatch):
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', 2.5)
  # Offset stays 0.0 here: this test isolates the cap tracking against the
  # un-shifted capture bound; the offset has dedicated coverage above.
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_CAP_OFFSET_V', 0.0)
  state = BrakingGapState()
  capture(state)
  t = np.arange(81) * .05
  speeds = np.linspace(5., 1., 81)
  leads = np.linspace(.8, .5, 81)
  before = vars(state).copy()
  actual = state.margins(t, speeds, leads, np.full(81, 10.), 2.0, 5.5)
  assert vars(state) == before
  expected = [state.margin]
  floors = [state.floor]
  for i in range(1, 81):
    state._advance(.05, 5.5 + speeds[i] * 2.0, 10., speeds[i], leads[i], 5.5)
    expected.append(state.margin)
    floors.append(state.floor)
  np.testing.assert_allclose(actual, expected)
  for i, floor in enumerate(floors):
    base = 5.5 + speeds[i] * 2.0
    assert floor <= max(base, 5.5 + 2.5 * speeds[i]) + 1e-6


def test_releasing_path_still_uses_the_two_second_tau(monkeypatch):
  # Pin the speed cap off: this test isolates the releasing-path two-second tau.
  monkeypatch.setattr(longitudinal_braking_gap, 'FLOOR_SPEED_CAP_S', float('inf'))
  state = BrakingGapState()
  before = capture(state)
  base = 5.5 + 1.5 * 2.0
  for _ in range(20):
    braking_pair(state, ego_speed=1.5, ego_accel=.5, lead_speed=2.5, lead_accel=.5, relative_speed=-1.)
  expected = base + (before - base) * math.exp(-1.0 / 2.0)
  assert abs(state.floor - expected) < 0.05
