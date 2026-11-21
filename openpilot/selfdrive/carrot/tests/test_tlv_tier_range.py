"""Unit tests for fused monocular TL range estimation and continuous
physics-based Tier-1 deceleration control.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.carrot.signal_stop_prepare import PrepareAccel
from openpilot.selfdrive.carrot.tlv.crops import FOCAL_REFERENCE_PX, REFERENCE_HEIGHT, REFERENCE_WIDTH, horizon_row
from openpilot.selfdrive.carrot.tlv.range_est import (
  RANGE_FAR_FALLBACK_M,
  box_width_range_m,
  elevation_grade_range_m,
  fused_tl_range_m,
  road_edges_mean_z,
)
from openpilot.selfdrive.carrot.tlv_stop_tiers import (
  SOFT_A_DEAD,
  SOFT_A_FULL,
  SOFT_DECEL,
  SOFT_FAST_NOCUE_S,
  SOFT_FAST_TIMEOUT_COOLDOWN_S,
  SOFT_FIRM_DECEL,
  SOFT_MAX_MOVING_S,
  SOFT_RANGE_HOLD_S,
  TierIn,
  TlvStopTiers,
)

DT = 0.05


def tier_in(t=0.0, v=18.39, rx=1.0, range_m=0.0, **kwargs):
  values = dict(
    t=t, v_ego=v, gas=False, brake=False, blinker=False, steer_ok=True, lead=False,
    raw_red=True, conf=True, go=False, gated_red=True, model_stop=False,
    x_end=rx * 10.0 * v, v0=v, y_end=0.0, stop_x=150.0,
    comfort_brake=2.4, cooldown=False, starting=False, enabled=True,
    range_m=range_m,
  )
  values.update(kwargs)
  return TierIn(**values)


# --- range_est tests ---------------------------------------------------------

def test_box_width_range_is_monotone_and_bounded():
  d_far = box_width_range_m(14.0)
  d_mid = box_width_range_m(22.0)
  d_near = box_width_range_m(45.0)
  assert 250.0 < d_far < 360.0
  assert 100.0 < d_mid < 160.0
  assert 40.0 < d_near < 70.0
  assert d_far > d_mid > d_near
  assert math.isnan(box_width_range_m(0.0))
  assert math.isnan(box_width_range_m(float("nan")))


def test_elevation_grade_range_compensates_pitch_and_road_slope():
  # Simulated Palisade camera calibration (pitch +0.068 rad, height 1.40 m).
  rpy = (0.0, 0.068, 0.007)
  hy = horizon_row(rpy, FOCAL_REFERENCE_PX, REFERENCE_WIDTH, REFERENCE_HEIGHT)
  h_tgt = 4.6 - 1.40  # 3.20 m above camera
  true_d = 150.0
  # Flat road: light sits at dy = focal * h_tgt / d above calibrated horizon.
  cy_flat = hy - FOCAL_REFERENCE_PX * (h_tgt / true_d)
  d_flat = elevation_grade_range_m(964.0, cy_flat, rpy, FOCAL_REFERENCE_PX,
                                   REFERENCE_WIDTH, REFERENCE_HEIGHT, h_cam_m=1.40)
  assert d_flat == pytest.approx(true_d, rel=0.03)

  # Downhill road (+2% -> z_road = +0.02 * x in NED calib frame): light sits lower in image.
  road_x = np.linspace(0.0, 192.0, 33)
  road_z = 0.02 * road_x
  cy_down = hy - FOCAL_REFERENCE_PX * ((h_tgt - 0.02 * true_d) / true_d)
  d_grade = elevation_grade_range_m(964.0, cy_down, rpy, FOCAL_REFERENCE_PX,
                                    REFERENCE_WIDTH, REFERENCE_HEIGHT, h_cam_m=1.40,
                                    road_x=road_x, road_z=road_z)
  assert d_grade == pytest.approx(true_d, rel=0.05)


def test_below_horizon_ray_returns_far_fallback():
  rpy = (0.0, 0.068, 0.0)
  hy = horizon_row(rpy, FOCAL_REFERENCE_PX, REFERENCE_WIDTH, REFERENCE_HEIGHT)
  # Box sitting slightly below the flat calibrated horizon (like 4c0 08:02:37 at 317 m).
  d_el = elevation_grade_range_m(964.0, hy + 12.0, rpy, FOCAL_REFERENCE_PX,
                                 REFERENCE_WIDTH, REFERENCE_HEIGHT, h_cam_m=1.40)
  assert d_el == pytest.approx(RANGE_FAR_FALLBACK_M)


def test_road_edges_mean_z_and_fused_range():
  xs = np.linspace(0.0, 190.0, 20)
  edges = [SimpleNamespace(x=xs, z=0.01 * xs), SimpleNamespace(x=xs, z=0.01 * xs)]
  rx, rz = road_edges_mean_z(edges)
  assert rx is not None and rz is not None
  assert rz[-1] == pytest.approx(1.9)

  rpy = (0.0, 0.068, 0.0)
  hy = horizon_row(rpy, FOCAL_REFERENCE_PX, REFERENCE_WIDTH, REFERENCE_HEIGHT)
  # 4c0-like far box (w=15 px, near horizon) vs 4ab-like near box (w=24 px, well above horizon).
  far_d = fused_tl_range_m((950.0, hy - 5.0, 965.0, hy + 2.0), rpy, FOCAL_REFERENCE_PX,
                           REFERENCE_WIDTH, REFERENCE_HEIGHT, h_cam_m=1.40, road_x=rx, road_z=rz)
  near_d = fused_tl_range_m((950.0, hy - 90.0, 974.0, hy - 80.0), rpy, FOCAL_REFERENCE_PX,
                            REFERENCE_WIDTH, REFERENCE_HEIGHT, h_cam_m=1.40, road_x=rx, road_z=rz)
  assert far_d > 240.0
  assert 80.0 < near_d < 140.0
  assert fused_tl_range_m((10.0, 10.0, 10.0, 20.0), rpy) == 0.0
  assert fused_tl_range_m((float("nan"), 0.0, 10.0, 10.0), rpy) == 0.0


# --- TlvStopTiers continuous range control tests -----------------------------

def test_far_red_coasts_without_braking_or_burning_timeout_clock():
  """66.2 km/h at 317 m: a_req = 0.56 <= 0.85 -> cap 0.0, moving timer stays 0."""
  state = TlvStopTiers()
  t = 0.0
  while t <= 15.0 + 1e-9:
    # Even past SOFT_MAX_MOVING_S (12 s), waiting far away must NOT time out.
    state.update(tier_in(t, v=66.2 / 3.6, range_m=300.0))
    assert state.soft_active is True
    assert state.accel_cap() == pytest.approx(0.0)
    assert state.soft_firm is False
    assert state.soft_moving_s == pytest.approx(0.0)
    t = round(t + DT, 4)


def test_near_fast_red_without_cue_caps_at_soft_decel():
  """4ab shape (81 km/h at 135 m, no model cue): -0.5 m/s^2 until the cue."""
  state = TlvStopTiers()
  state.update(tier_in(0.0, v=81.0 / 3.6, rx=1.0, range_m=135.0))
  assert state.soft_active is True
  assert state.accel_cap() == pytest.approx(-SOFT_DECEL)
  assert state.soft_firm is False


def test_continuous_ramp_and_non_latching_single_tick_spike():
  """A single noisy close tick inside a 1 s median window must not lock -1.0."""
  state = TlvStopTiers()
  v = 66.2 / 3.6
  t = 0.0
  for _ in range(10):
    state.update(tier_in(t, v=v, range_m=300.0))
    t = round(t + DT, 4)
  assert state.accel_cap() == pytest.approx(0.0)
  # One noisy outlier reading 120 m: rolling median is still 300 m -> cap stays 0.0.
  state.update(tier_in(t, v=v, range_m=120.0))
  assert state.accel_cap() == pytest.approx(0.0)
  assert state.soft_firm is False

  # Mid-range (a_req midway between SOFT_A_DEAD and SOFT_A_FULL) gives partial decel.
  a_mid = 0.5 * (SOFT_A_DEAD + SOFT_A_FULL)
  d_mid = (v * v) / (2.0 * a_mid) + v
  state2 = TlvStopTiers()
  state2.update(tier_in(0.0, v=v, range_m=d_mid))
  # a_req midway -> half of the no-cue -0.5 cap.
  assert state2.accel_cap() == pytest.approx(-0.25, abs=0.02)
  assert state2.soft_firm is False


def test_odometry_propagates_range_across_brief_dropout():
  """A 1 s box dropout at 300 m propagates to ~282 m instead of slamming fallback."""
  state = TlvStopTiers()
  v = 18.0
  state.update(tier_in(0.0, v=v, range_m=300.0))
  assert state.accel_cap() == pytest.approx(0.0)
  # 1.0 s of range_m=0.0 (detector miss while red is held): propagated range ~282 m -> still 0.0!
  t = DT
  while t <= 1.0 + 1e-9:
    state.update(tier_in(t, v=v, range_m=0.0))
    assert state.soft_range_m == pytest.approx(300.0 - v * t, abs=0.5)
    assert state.accel_cap() == pytest.approx(0.0)
    t = round(t + DT, 4)
  # After SOFT_RANGE_HOLD_S (2.5 s) without range, falls back safely to -SOFT_DECEL (-0.5).
  while t <= SOFT_RANGE_HOLD_S + 0.1:
    state.update(tier_in(t, v=v, range_m=0.0))
    t = round(t + DT, 4)
  assert state.soft_range_m is None
  assert state.accel_cap() == pytest.approx(-SOFT_DECEL)


def test_brake_pressed_blocks_tier1_start():
  """4c4 10:01:31.5 regression: active manual braking must not start tier 1."""
  state = TlvStopTiers()
  state.update(tier_in(0.0, v=15.0, brake=True, range_m=80.0))
  assert state.soft_active is False
  state.update(tier_in(DT, v=15.0, brake=False, range_m=80.0))
  assert state.soft_active is True


def test_braking_timeout_still_bounds_stuck_false_red():
  """Once actually braking (cap <= -0.15) with no model_stop/cue, 12 s timeout still fires."""
  state = TlvStopTiers()
  t = 0.0
  state.update(tier_in(t, v=18.0, range_m=100.0))
  assert state.accel_cap() == pytest.approx(-SOFT_DECEL)
  while state.soft_active and t < 20.0:
    t = round(t + DT, 4)
    state.update(tier_in(t, v=18.0, range_m=100.0))
  assert not state.soft_active
  assert t == pytest.approx(SOFT_MAX_MOVING_S + DT, abs=0.06)
  assert state.events == [('soft', 'release', 'timeout')]


def test_no_cue_cap_never_below_soft_decel_at_any_speed():
  """No cue: the range cap is scaled into [-0.5, 0.0] at every speed."""
  for v_kph in (40, 60, 81, 96):
    v = v_kph / 3.6
    state = TlvStopTiers()
    t = 0.0
    while t <= 2.0 + 1e-9:
      state.update(tier_in(t, v=v, rx=1.0, range_m=60.0))
      assert state.accel_cap() >= -SOFT_DECEL - 1e-9
      assert state.soft_firm is False
      t = round(t + DT, 4)


def test_cue_unlocks_firm():
  """The model cue alone deepens tier 1 to -1.0 m/s^2."""
  state = TlvStopTiers()
  v = 81.0 / 3.6
  t = 0.0
  while t <= 1.0 + 1e-9:
    state.update(tier_in(t, v=v, rx=1.0, range_m=135.0))
    assert state.soft_firm is False
    t = round(t + DT, 4)
  cue_t = None
  cue_events = []
  while t <= 1.6 + 1e-9:
    state.update(tier_in(t, v=v, rx=0.7, range_m=135.0))
    if cue_t is None and state.soft_firm:
      cue_t = t
      cue_events = [e for e in state.events if e[0] == 'soft']
    if cue_t is not None and t >= cue_t:
      assert state.accel_cap() == pytest.approx(-SOFT_FIRM_DECEL)
      assert state.soft_firm is True
    t = round(t + DT, 4)
  assert cue_t is not None
  assert cue_t == pytest.approx(1.0 + 0.5, abs=0.06)
  assert cue_events == [('soft', 'firm', 'cue')]


def test_fast_no_cue_times_out_after_short_braking():
  """96 km/h at 247 m without the cue: release after 4.5 s of braking, not 12 s."""
  state = TlvStopTiers()
  t = 0.0
  v = 96.0 / 3.6
  state.update(tier_in(t, v=v, rx=1.0, range_m=247.0))
  assert state.accel_cap() == pytest.approx(-SOFT_DECEL)
  while state.soft_active and t < 20.0:
    t = round(t + DT, 4)
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0))
  assert not state.soft_active
  assert t == pytest.approx(SOFT_FAST_NOCUE_S + DT, abs=0.06)
  assert state.events == [('soft', 'release', 'timeout')]


def test_fast_timeout_latches_start_speed():
  """The short timeout keys on the start speed, then falls back to 12 s below 80 km/h."""
  state = TlvStopTiers()
  t = 0.0
  v = 85.0 / 3.6
  state.update(tier_in(t, v=v, rx=1.0, range_m=120.0))
  while state.soft_active and t < 20.0:
    t = round(t + DT, 4)
    v = max(0.0, v - 0.5 * DT)
    state.update(tier_in(t, v=v, rx=1.0, range_m=120.0))
  assert not state.soft_active
  assert t == pytest.approx(SOFT_FAST_NOCUE_S + DT, abs=0.06)
  assert state.events == [('soft', 'release', 'timeout')]

  state2 = TlvStopTiers()
  t = 0.0
  v = 79.0 / 3.6
  state2.update(tier_in(t, v=v, rx=1.0, range_m=120.0))
  while state2.soft_active and t < 20.0:
    t = round(t + DT, 4)
    state2.update(tier_in(t, v=v, rx=1.0, range_m=120.0))
    if t < SOFT_MAX_MOVING_S - 0.1:
      assert state2.soft_active is True
      assert state2.events == []
  assert not state2.soft_active
  assert t == pytest.approx(SOFT_MAX_MOVING_S + DT, abs=0.06)
  assert state2.events == [('soft', 'release', 'timeout')]


def test_cue_before_fast_timeout_keeps_tier1():
  """A cue inside the 4.5 s window cancels the fast timeout."""
  state = TlvStopTiers()
  t = 0.0
  v = 96.0 / 3.6
  while t <= 3.0 + 1e-9:
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0))
    assert state.soft_active is True
    t = round(t + DT, 4)
  while t <= 6.0 + 1e-9:
    state.update(tier_in(t, v=v, rx=0.7, range_m=247.0))
    assert state.soft_active is True
    if t >= 3.6:
      assert state.accel_cap() == pytest.approx(-SOFT_FIRM_DECEL)
    t = round(t + DT, 4)
  assert [e for e in state.events if e[1] == 'release'] == []


def test_model_stop_blocks_fast_timeout():
  """A model-planned stop is not a false red: no fast timeout."""
  state = TlvStopTiers()
  t = 0.0
  v = 96.0 / 3.6
  while t <= 6.0 + 1e-9:
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0, model_stop=True))
    assert state.soft_active is True
    t = round(t + DT, 4)


def test_fast_timeout_cooldown_blocks_no_cue_restart():
  """The phantom cluster seen again after a 2.6 s gap may not brake twice without the cue."""
  state = TlvStopTiers()
  t = 0.0
  v = 96.0 / 3.6
  t_to = None
  while t < 20.0:
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0))
    if not state.soft_active:
      assert ('soft', 'release', 'timeout') in state.events
      t_to = t
      break
    t = round(t + DT, 4)
  assert t_to == pytest.approx(SOFT_FAST_NOCUE_S, abs=0.06)

  gap_end = round(t_to + 2.6, 4)
  while t < gap_end + 1e-9:
    t = round(t + DT, 4)
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0, raw_red=False, conf=False, gated_red=False))
    assert state.soft_active is False

  start_t = None
  while t <= t_to + 12.0 + 1e-9:
    t = round(t + DT, 4)
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0))
    if start_t is None:
      if ('soft', 'start', 'confirmed_red') in state.events:
        start_t = t
      elif t < t_to + SOFT_FAST_TIMEOUT_COOLDOWN_S - 0.06:
        assert state.soft_active is False
  assert start_t == pytest.approx(t_to + SOFT_FAST_TIMEOUT_COOLDOWN_S, abs=0.06)


def test_fast_timeout_cooldown_allows_cue_restart():
  """The cue bypasses the cooldown: -1.0 m/s^2 0.5 s after the red returns."""
  state = TlvStopTiers()
  t = 0.0
  v = 96.0 / 3.6
  t_to = None
  while t < 20.0:
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0))
    if not state.soft_active:
      assert ('soft', 'release', 'timeout') in state.events
      t_to = t
      break
    t = round(t + DT, 4)
  assert t_to == pytest.approx(SOFT_FAST_NOCUE_S, abs=0.06)

  gap_end = round(t_to + 2.6, 4)
  while t < gap_end + 1e-9:
    t = round(t + DT, 4)
    state.update(tier_in(t, v=v, rx=1.0, range_m=247.0, raw_red=False, conf=False, gated_red=False))
    assert state.soft_active is False

  t_red = None
  start_t = None
  while t <= gap_end + 3.0 + 1e-9:
    t = round(t + DT, 4)
    state.update(tier_in(t, v=v, rx=0.7, range_m=247.0))
    if t_red is None:
      t_red = t
    if start_t is None:
      if ('soft', 'start', 'confirmed_red') in state.events:
        start_t = t
        assert t < t_to + SOFT_FAST_TIMEOUT_COOLDOWN_S
        assert state.accel_cap() == pytest.approx(-SOFT_FIRM_DECEL)
        assert state.soft_firm is True
      else:
        assert state.soft_active is False
  assert start_t == pytest.approx(t_red + 0.5, abs=0.06)


# --- PrepareAccel smooth bidirectional floor tests ---------------------------

def test_prepare_accel_floor_override_ramps_smoothly_down_and_up():
  accel = PrepareAccel()
  out = 0.0
  # Ramp down to -0.6 at 0.25 m/s^3 (takes 2.4 s = 48 ticks).
  for _ in range(60):
    out = accel.apply(0.0, out, DT, True, follow_target=True, floor_override=-0.6)
  assert out == pytest.approx(-0.6, abs=1e-6)
  # Floor eases upward to -0.2: must ramp up at RAMP_UP (0.5 m/s^3 -> +0.025 per tick), never jump.
  prev = out
  for _ in range(4):
    out = accel.apply(0.0, prev, DT, True, follow_target=True, floor_override=-0.2)
    assert out - prev == pytest.approx(PrepareAccel.RAMP_UP * DT, abs=1e-6)
    prev = out
  for _ in range(40):
    out = accel.apply(0.0, out, DT, True, follow_target=True, floor_override=-0.2)
  assert out == pytest.approx(-0.2, abs=1e-6)


def test_carrot_planner_read_range_and_accel_cap_integration():
  import time
  from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner

  now = time.monotonic()
  dummy = SimpleNamespace(
    _tlv_soft_decel=SOFT_DECEL,
    coast_cap_active=False,
    tlv_soft_active=True,
    tlv_soft_cap=0.0,
    cruiseMaxVals0=1.6, cruiseMaxVals1=1.6, cruiseMaxVals2=1.6,
    cruiseMaxVals3=1.6, cruiseMaxVals4=1.6, cruiseMaxVals5=1.6, cruiseMaxVals6=1.6,
    myDrivingMode=SimpleNamespace(), myHighModeFactor=1.0, mySafeFactor=1.0,
  )
  # Far red (tlv_soft_cap == 0.0): MPC upper accel must coast (0.0), NOT clamp to -0.5.
  assert CarrotPlanner.get_carrot_accel(dummy, 18.0) == pytest.approx(0.0)
  dummy.tlv_soft_cap = -0.3
  assert CarrotPlanner.get_carrot_accel(dummy, 18.0) == pytest.approx(-0.3)
  dummy.tlv_soft_cap = -1.0
  assert CarrotPlanner.get_carrot_accel(dummy, 18.0) == pytest.approx(-SOFT_DECEL)

  # _read_tlv_range_m reads fresh src-1 range and rejects stale / non-TLV / NaN.
  class FakeSM:
    def __init__(self, msg, age=0.0):
      self._m = msg
      self.seen = {'carrotMan': True}
      self.alive = {'carrotMan': True}
      self.valid = {'carrotMan': True}
      self.recv_time = {'carrotMan': now - age}
    def __getitem__(self, k):
      return self._m

  assert CarrotPlanner._read_tlv_range_m(dummy, FakeSM(SimpleNamespace(trafficStateSrc=1, tlvRangeM=215.5))) == pytest.approx(215.5)
  assert CarrotPlanner._read_tlv_range_m(dummy, FakeSM(SimpleNamespace(trafficStateSrc=0, tlvRangeM=215.5))) == 0.0
  assert CarrotPlanner._read_tlv_range_m(dummy, FakeSM(SimpleNamespace(trafficStateSrc=1, tlvRangeM=float('nan')))) == 0.0
  assert CarrotPlanner._read_tlv_range_m(dummy, FakeSM(SimpleNamespace(trafficStateSrc=1, tlvRangeM=215.5), age=5.0)) == 0.0
