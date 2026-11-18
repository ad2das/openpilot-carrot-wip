"""W-TLD7: tier 1 floors at -0.5 until the driving model's own cue, then -1.0.

`cue` is the same signal tier 2 waits for: rx = x_end / (10 * max(v0, 1)) held
<= CUE_RX_MAX for CUE_MIN_S, with no brake inside CUE_BRAKE_HOLD_S and no gas.
Every case drives the pure state machine with dt = 0.05 s; x_end = 10 * v gives
rx 1.0 (no cue), x_end = 7 * v gives rx 0.7 (cue after 0.5 s).
"""
import pytest

from openpilot.selfdrive.carrot.signal_stop_prepare import PrepareAccel
from openpilot.selfdrive.carrot.tlv_stop_tiers import CUE_MIN_S, TierIn, TlvStopTiers

DT = 0.05
V = 18.38


def tier_in(t=0.0, v=V, rx=1.0, **kwargs):
  """Base tick: enabled, raw/conf/gated red, no lead/gas/brake/blinker, y 0, v0 = v."""
  values = dict(
    t=t, v_ego=v, gas=False, brake=False, blinker=False, steer_ok=True, lead=False,
    raw_red=True, conf=True, go=False, gated_red=True, model_stop=False,
    x_end=rx * 10.0 * v, v0=v, y_end=0.0, stop_x=150.0,
    comfort_brake=2.4, cooldown=False, starting=False, enabled=True,
  )
  values.update(kwargs)
  return TierIn(**values)


def soft(state):
  """Only tier-1 events; tier 2 may commit on the same tick and is not under test here."""
  return [e for e in state.events if e[0] == 'soft']


def test_far_start_is_not_firm():
  state = TlvStopTiers()
  state.update(tier_in(0.0, rx=1.0))
  assert state.soft_active
  assert state.soft_firm is False
  assert soft(state) == [('soft', 'start', 'confirmed_red')]


def test_cue_before_the_start_firms_the_start():
  """A cue that already holds when the red confirms makes tier 1 firm at once."""
  state = TlvStopTiers()
  t = 0.0
  while t <= 0.6 + 1e-9:
    state.update(tier_in(t, rx=0.7, raw_red=False, conf=False, gated_red=False))
    t = round(t + DT, 4)
  assert not state.soft_active
  state.update(tier_in(t, rx=0.7))
  assert state.soft_active
  assert state.soft_firm is True
  assert soft(state) == [('soft', 'start', 'confirmed_red')]


def test_cue_latches_firm_once_and_resets_the_moving_timer():
  state = TlvStopTiers()
  t = 0.0
  state.update(tier_in(t, rx=1.0))
  assert state.soft_active and state.soft_firm is False
  t = DT
  while t <= 1.0 + 1e-9:
    state.update(tier_in(t, rx=1.0))
    assert state.soft_firm is False
    t = round(t + DT, 4)
  t_drop = round(t, 4)
  firm_t = None
  while t <= t_drop + 1.0 + 1e-9:
    state.update(tier_in(t, rx=0.7))
    if soft(state):
      assert soft(state) == [('soft', 'firm', 'cue')]
      assert firm_t is None
      firm_t = t
      assert state.soft_moving_s == 0.0
      assert state.soft_firm is True
    t = round(t + DT, 4)
  assert firm_t is not None
  assert t_drop + CUE_MIN_S - 0.06 <= firm_t <= t_drop + CUE_MIN_S + 0.06
  state.update(tier_in(t, rx=1.0))
  assert state.soft_firm is True


def test_release_clears_the_firm_flag():
  state = TlvStopTiers()
  t = 0.0
  while t <= 0.6 + 1e-9:
    state.update(tier_in(t, rx=0.7))
    t = round(t + DT, 4)
  assert state.soft_firm is True
  state.update(tier_in(t, v=8.0, rx=0.7))
  assert soft(state) == [('soft', 'release', 'slow')]
  assert not state.soft_active
  assert state.soft_firm is False


def test_brake_blocks_the_cue_for_the_hold_window():
  state = TlvStopTiers()
  t = 0.0
  state.update(tier_in(t, rx=1.0))
  assert state.soft_active and state.soft_firm is False
  t = DT
  while t <= 1.0 + 1e-9:
    state.update(tier_in(t, rx=1.0))
    t = round(t + DT, 4)
  t_brake = round(t, 4)
  state.update(tier_in(t_brake, rx=0.7, brake=True))
  assert state.soft_firm is False
  t = round(t_brake + DT, 4)
  firm_t = None
  while t <= t_brake + 4.0 + 1e-9:
    state.update(tier_in(t, rx=0.7))
    if soft(state):
      assert soft(state) == [('soft', 'firm', 'cue')]
      firm_t = firm_t if firm_t is not None else t
    t = round(t + DT, 4)
  assert firm_t is not None
  assert firm_t >= t_brake + 2.0 + CUE_MIN_S


def test_prepare_accel_floor_follows_firm():
  def run(follow, firm):
    accel = PrepareAccel()
    out = 0.0
    for _ in range(200):
      out = accel.apply(0.0, out, DT, True, follow_target=follow, firm=firm)
    return out

  assert run(True, True) == pytest.approx(-1.0)
  assert run(True, False) == pytest.approx(-0.5)
  assert run(False, True) == pytest.approx(-0.5)
