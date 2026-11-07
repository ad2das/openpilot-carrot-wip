"""W-TLT1: the pure two-tier state machine (CarrotTlvStopGo = 1).

No cereal, no params, no clock: every case here drives `TlvStopTiers.update`
with an explicit time and asserts the same values the parent derived from the
module spec. dt is 0.05 s unless a case says otherwise.
"""
import pytest

from openpilot.selfdrive.carrot.tlv_stop_tiers import (
  CUE_BRAKE_HOLD_S,
  LEAD_CLEAR_S,
  SOFT_DECEL,
  SOFT_MAX_MOVING_S,
  STOP_MAX_MPS,
  TIER_H_S,
  TierIn,
  TlvStopTiers,
)

DT = 0.05


def tier_in(t=0.0, v=20.0, x_end=None, stop_x=150.0, **kwargs):
  """Base tick: enabled, raw/conf/gated red, no lead/gas/brake/blinker, y 0, v0 = v."""
  values = dict(
    t=t, v_ego=v, gas=False, brake=False, blinker=False, steer_ok=True, lead=False,
    raw_red=True, conf=True, go=False, gated_red=True, model_stop=False,
    x_end=10.0 * v if x_end is None else x_end, v0=v, y_end=0.0, stop_x=stop_x,
    comfort_brake=2.4, cooldown=False, starting=False, enabled=True,
  )
  values.update(kwargs)
  return TierIn(**values)


def walk(state, t0, t1, make):
  """Ticks from t0 to t1 inclusive, one row per tick."""
  rows = []
  t = t0
  while t <= t1 + 1e-9:
    state.update(make(t))
    rows.append((round(t, 2), state.soft_active, state.committed, tuple(state.events)))
    t = round(t + DT, 4)
  return rows


def test_soft_starts_on_the_first_tick_and_caps_accel_without_committing():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 3.0, lambda t: tier_in(t, v=20.0, x_end=200.0, stop_x=150.0))
  started = [r for r in rows if r[1]]
  assert started and started[0][0] == 0.0
  assert started[0][3] == (('soft', 'start', 'confirmed_red'),)
  assert state.accel_cap() == -SOFT_DECEL == -0.5
  assert not any(r[2] for r in rows)  # no cue: tier 2 never commits


def test_cue_commit_needs_the_half_second_and_reports_cue():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 1.2, lambda t: tier_in(t, x_end=150.0))
  assert not any(r[2] for r in rows if r[0] <= 0.45)
  commit = next(r for r in rows if r[2])
  assert commit[0] == 0.50
  assert commit[3] == (('stop', 'commit', 'cue'),)


@pytest.mark.parametrize("v_ego,expected", [(8.3, False), (27.8, False), (8.34, True), (27.7, True)])
def test_soft_speed_band_is_thirty_to_one_hundred_kph(v_ego, expected):
  state = TlvStopTiers()
  state.update(tier_in(0.0, v=v_ego, stop_x=200.0))
  assert state.soft_active is expected


def test_graded_red_alone_starts_nothing():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 3.0, lambda t: tier_in(t, conf=False))
  assert not any(r[1] for r in rows)
  assert not any(r[2] for r in rows)


def test_soft_holds_through_the_red_gap_then_releases_and_can_restart():
  state = TlvStopTiers()
  state.update(tier_in(0.0))
  assert state.soft_active

  t = 0.0
  while True:
    t = round(t + DT, 4)
    state.update(tier_in(t, gated_red=False))
    assert state.soft_active or t > TIER_H_S  # the 2.45 s gap must hold
    if not state.soft_active:
      break
  assert t == pytest.approx(2.55)
  assert state.events == [('soft', 'release', 'red_end')]

  # The latch cleared on the tick with no gated red, so a red that returns
  # 0.1 s later starts a fresh episode.
  t = round(t + DT, 4)
  state.update(tier_in(t, gated_red=False))
  assert not state.soft_active
  state.update(tier_in(round(t + DT, 4), gated_red=True))
  assert state.soft_active
  assert state.events == [('soft', 'start', 'confirmed_red')]


def test_soft_times_out_on_the_moving_clock_and_does_not_restart_on_the_same_red():
  state = TlvStopTiers()
  t = 0.0
  state.update(tier_in(t))
  while state.soft_active:
    t = round(t + DT, 4)
    state.update(tier_in(t))
  assert t == pytest.approx(12.05)
  assert SOFT_MAX_MOVING_S == 12.0
  assert state.events == [('soft', 'release', 'timeout')]

  for _ in range(20):  # the same red continues: the latch holds
    t = round(t + DT, 4)
    state.update(tier_in(t))
    assert not state.soft_active

  for _ in range(52):  # 2.6 s without a gated red clears the latch
    t = round(t + DT, 4)
    state.update(tier_in(t, raw_red=False, conf=False, gated_red=False))
  state.update(tier_in(round(t + DT, 4)))
  assert state.soft_active


def test_one_model_stop_tick_resets_the_moving_clock():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 20.0, lambda t: tier_in(t, model_stop=(t == 1.0)))
  assert all(r[1] for r in rows)


def test_gas_lead_and_speed_each_release_soft_and_the_blinker_only_blocks_the_start():
  for reason, kwargs in (('gas', dict(gas=True)), ('lead', dict(lead=True))):
    state = TlvStopTiers()
    state.update(tier_in(0.0))
    assert state.soft_active
    state.update(tier_in(DT, **kwargs))
    assert not state.soft_active
    assert state.events == [('soft', 'release', reason)]

  state = TlvStopTiers()
  state.update(tier_in(0.0))
  state.update(tier_in(DT, v=8.0))
  assert not state.soft_active
  assert state.events == [('soft', 'release', 'slow')]

  state = TlvStopTiers()
  state.update(tier_in(0.0, blinker=True))
  assert not state.soft_active
  state.update(tier_in(DT))
  state.update(tier_in(2 * DT, blinker=True))
  assert state.soft_active  # a blinker after the start is not a release edge


def test_below_the_veto_band_the_stop_commits_without_a_cue():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 0.2, lambda t: tier_in(t, v=10.0, stop_x=60.0))
  assert rows[0][2] is True
  assert rows[0][3] == (('soft', 'start', 'confirmed_red'), ('stop', 'commit', 'slow_no_veto'))


def test_episode_start_veto_blocks_a_late_commit():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 5.0,
              lambda t: tier_in(t, v=15.0 if t < 1.0 else 10.0, stop_x=60.0))
  assert not any(r[2] for r in rows)


def test_above_the_veto_band_the_commit_waits_a_second_and_the_cue():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 2.0, lambda t: tier_in(t, v=24.0, x_end=168.0, stop_x=200.0))
  assert not any(r[2] for r in rows if r[0] < 1.0)
  commit = next(r for r in rows if r[2])
  assert commit[0] == pytest.approx(1.0)
  assert commit[3] == (('stop', 'commit', 'cue'),)


@pytest.mark.parametrize("v_ego,expected", [(27.7, True), (27.8, False)])
def test_stop_band_ends_at_one_hundred_kph(v_ego, expected):
  state = TlvStopTiers()
  rows = walk(state, 0.0, 2.0, lambda t: tier_in(t, v=v_ego, x_end=v_ego * 7.0, stop_x=200.0))
  assert any(r[2] for r in rows) is expected
  assert STOP_MAX_MPS == pytest.approx(100.0 / 3.6)


def test_committed_stop_holds_through_the_red_gap_and_releases_after_it():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 2.0, lambda t: tier_in(t, x_end=140.0))
  commit = next(r for r in rows if r[2])
  assert commit[0] == pytest.approx(0.5)

  def make(t):
    red = t <= 2.0
    return tier_in(t, x_end=140.0, raw_red=red, conf=red, gated_red=red)

  rows = walk(state, 2.05, 4.20, make)
  assert all(r[2] for r in rows)  # a 2.249 s gap holds
  for t in (4.25, 4.30, 4.35, 4.40, 4.45, 4.50, 4.55):
    state.update(make(t))
  assert state.committed is False
  assert ('stop', 'release', 'red_end') in state.events


def test_a_go_releases_both_tiers_and_no_recommit_follows():
  state = TlvStopTiers()
  walk(state, 0.0, 2.0, lambda t: tier_in(t, x_end=140.0))
  assert state.committed and state.soft_active

  def make(t):
    red = t <= 2.0
    return tier_in(t, x_end=140.0, raw_red=red, conf=red, gated_red=red, go=(2.0 <= t <= 2.5))

  rows = walk(state, 2.05, 5.0, make)
  assert rows[0][3] == (('soft', 'release', 'go'), ('stop', 'release', 'go'))
  assert not any(r[2] for r in rows)
  assert not any(r[1] for r in rows)


def test_the_cue_waits_out_the_brake_hold():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 4.0, lambda t: tier_in(t, x_end=140.0, brake=(t <= 0.5)))
  assert CUE_BRAKE_HOLD_S == 2.0
  commit = next(r for r in rows if r[2])
  # brake ends at 0.5, the hold clears at 2.5, the first valid tick is 2.55 and
  # the 0.5 s cue clock then lands the commit at 3.05.
  assert commit[0] == pytest.approx(3.05)
  assert commit[3] == (('stop', 'commit', 'cue'),)


def test_distance_window_tracks_speed_and_the_220_metre_floor():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 5.0, lambda t: tier_in(t, v=20.0, x_end=140.0, stop_x=230.0))
  assert not any(r[2] for r in rows)  # 230 m > hi 220 m at 20 m/s

  state = TlvStopTiers()
  rows = walk(state, 0.0, 5.0, lambda t: tier_in(t, v=24.0, x_end=168.0, stop_x=230.0))
  assert next(r for r in rows if r[2])[0] == pytest.approx(1.0)  # 230 m <= hi 252 m


def test_a_go_ends_the_red_so_no_commit_until_a_new_one():
  state = TlvStopTiers()

  def first(t):
    red = t < 1.0 - 1e-9
    return tier_in(t, x_end=200.0 if red else 140.0, raw_red=red, conf=red,
                   gated_red=red, go=not red)

  rows = walk(state, 0.0, 4.0, first)
  assert not any(r[2] for r in rows)  # red_newer is False after the go

  rows = walk(state, 4.05, 5.0, lambda t: tier_in(t, x_end=140.0))
  commit = next(r for r in rows if r[2])
  assert commit[0] == pytest.approx(4.05)  # first tick that meets age and cue
  assert ('stop', 'commit', 'cue') in commit[3]


def test_gas_through_a_held_red_blocks_both_tiers_until_that_red_ends():
  state = TlvStopTiers()

  def make(t):
    red = t < 5.0 or t >= 7.6
    return tier_in(t, v=10.0, x_end=100.0, stop_x=60.0, gas=(t <= 1.0),
                   raw_red=red, conf=red, gated_red=red)

  rows = walk(state, 0.0, 8.0, make)
  early = [r for r in rows if r[0] < 7.6]
  assert not any(r[1] for r in early)  # the gas holds tier 1 off
  assert not any(r[2] for r in early)  # and tier 2 commits nothing
  commits = [(r[0], e) for r in rows for e in r[3] if e[0] == 'stop' and e[1] == 'commit']
  assert len(commits) == 1
  assert commits[0][0] == pytest.approx(7.6)  # the fresh red after the gap
  assert commits[0][1] == ('stop', 'commit', 'slow_no_veto')
  assert state.committed


def test_a_lead_release_holds_the_recommit_for_one_second():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 3.0, lambda t: tier_in(t, v=10.0, x_end=100.0, stop_x=60.0, lead=(t == 1.0)))
  commits = [(r[0], e) for r in rows for e in r[3] if e[0] == 'stop' and e[1] == 'commit']
  assert [c[0] for c in commits] == pytest.approx([0.0, 2.05])
  assert [c[1][2] for c in commits] == ['slow_no_veto', 'slow_no_veto']
  releases = [(r[0], e[2]) for r in rows for e in r[3] if e[0] == 'stop' and e[1] == 'release']
  assert releases == [(1.0, 'lead')]  # one lead tick, one release, then the calm gap
  assert state.committed


def test_a_stray_lead_tick_only_delays_the_first_commit_by_the_clear_time():
  state = TlvStopTiers()
  rows = walk(state, 0.0, 3.0, lambda t: tier_in(t, v=10.0, x_end=100.0, stop_x=60.0, lead=(t <= 1.0)))
  commits = [(r[0], e) for r in rows for e in r[3] if e[0] == 'stop' and e[1] == 'commit']
  assert len(commits) == 1
  assert commits[0][0] == pytest.approx(1.0 + LEAD_CLEAR_S, abs=0.06)  # 000006d5 09:46:05.818 shape
  assert commits[0][1] == ('stop', 'commit', 'slow_no_veto')


def test_single_tick_lead_dropouts_never_commit():
  state = TlvStopTiers()

  def make(t):
    lead = t < 2.0 and t != 1.0 and t != 1.5  # a lead with two one-tick dropouts, gone from 2.0
    return tier_in(t, v=10.0, x_end=100.0, stop_x=60.0, lead=lead)

  rows = walk(state, 0.0, 3.0, make)
  commits = [(r[0], e) for r in rows for e in r[3] if e[0] == 'stop' and e[1] == 'commit']
  assert len(commits) == 1
  assert commits[0][0] == pytest.approx(1.95 + LEAD_CLEAR_S, abs=0.06)
  assert not any(e[0] == 'stop' and e[1] == 'release' for r in rows for e in r[3])


def test_gas_before_the_red_is_confirmed_does_not_block_the_commit():
  state = TlvStopTiers()

  def make(t):
    conf = t >= 1.0  # raw red from 0.0; the driver lifts the gas at 0.3, before the red is confirmed
    return tier_in(t, v=10.0, x_end=100.0, stop_x=60.0, gas=(t <= 0.3), conf=conf, gated_red=conf)

  rows = walk(state, 0.0, 2.0, make)
  commits = [(r[0], e) for r in rows for e in r[3] if e[0] == 'stop' and e[1] == 'commit']
  assert [c[0] for c in commits] == pytest.approx([1.0])  # 000006d5 09:46:03.6-04.7 shape
  assert commits[0][1] == ('stop', 'commit', 'slow_no_veto')
  assert state.gas_ep is False
