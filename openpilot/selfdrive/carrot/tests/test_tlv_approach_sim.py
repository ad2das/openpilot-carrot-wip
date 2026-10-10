"""Approach-simulation tests for the high-speed case (W74C).

The printed tables are the before/after evidence for the report; the detector
and classifier hit curves are documented ASSUMPTIONS in tlv/approach_sim.py
until W71's eval numbers exist. The simulation drives the REAL tracker, signal
state machine and ViewScheduler.
"""
import pytest

from openpilot.selfdrive.carrot.tlv import approach_sim as sim

SPEEDS = (60.0, 80.0, 100.0)
RATES = (1.0, 2.0, 5.0, 10.0)


def test_stop_need_matches_contract():
  assert sim.stop_need_m(60.0) == pytest.approx(55.6, abs=0.2)
  assert sim.stop_need_m(80.0) == pytest.approx(98.8, abs=0.2)
  assert sim.stop_need_m(100.0) == pytest.approx(154.3, abs=0.2)
  # v * latency is added on top of the braking distance.
  assert sim.stop_need_m(60.0, latency_s=1.0) - sim.stop_need_m(60.0) == pytest.approx(60.0 / 3.6, abs=0.01)


def test_hit_curves_bounded_and_monotonic():
  for curve in (sim.detector_hit_probability, sim.classifier_vote_probability):
    values = [curve(px) for px in (0.5, 2.0, 6.0, 12.0, 30.0)]
    assert all(0.0 < v < 1.0 for v in values)
    assert values == sorted(values)
    assert curve(100.0) > 0.9


def test_current_policy_table_and_high_speed_problem():
  results = sim.run_matrix(policies=("current",), speeds=SPEEDS, rates=RATES)
  print("\n" + sim.format_table(results))
  by = {(r.speed_kph, r.rate_hz): r for r in results}
  # The problem this task fixes: at high speed the alternation confirms red
  # far too late (80/100 km/h cannot stop), and only rarely/slowly at 60.
  assert by[(80.0, 5.0)].stop_margin_m is not None and by[(80.0, 5.0)].stop_margin_m < 0
  assert by[(100.0, 5.0)].stop_margin_m < 0
  assert by[(80.0, 10.0)].stop_margin_m < 0
  assert by[(100.0, 10.0)].stop_margin_m < 0
  # 60 km/h still confirms within its stopping need at 5/10 Hz.
  assert by[(60.0, 5.0)].stop_margin_m > 0


def test_speed_aware_table_improves_every_speed():
  current = {(r.speed_kph, r.rate_hz): r for r in sim.run_matrix(policies=("current",), speeds=SPEEDS, rates=RATES)}
  results = sim.run_matrix(policies=("speed_aware",), speeds=SPEEDS, rates=RATES)
  print("\n" + sim.format_table(results))
  by = {(r.speed_kph, r.rate_hz): r for r in results}
  for speed in SPEEDS:
    for rate in RATES:
      old, new = current[(speed, rate)], by[(speed, rate)]
      assert new.d_confirm_m is not None and old.d_confirm_m is not None
      assert new.d_confirm_m >= old.d_confirm_m
      if rate >= 5.0:
        # Classifier-time confirmation: clearly earlier than the detector belt.
        assert new.d_confirm_m > old.d_confirm_m + 10.0
        assert new.confirm_s <= 1.0
  # 60/80 km/h now stop comfortably at the governor's normal rates.
  assert by[(60.0, 5.0)].stop_margin_m > 0
  assert by[(80.0, 5.0)].stop_margin_m > 0
  assert by[(60.0, 10.0)].stop_margin_m > 0
  assert by[(80.0, 10.0)].stop_margin_m > 0
  # 100 km/h stays limited by geometry (need 154 m > 120 m first sight); the
  # graded redLikely is the early signal there.
  r = by[(80.0, 5.0)]
  assert r.red_likely_m is not None and r.red_likely_m > r.d_confirm_m
  r100 = by[(100.0, 10.0)]
  assert r100.d_confirm_m > 100.0 and r100.red_likely_m >= r100.d_confirm_m


def test_speed_aware_is_deterministic():
  first = sim.run_matrix(policies=("speed_aware",), speeds=SPEEDS, rates=RATES)
  second = sim.run_matrix(policies=("speed_aware",), speeds=SPEEDS, rates=RATES)
  assert [(r.d_confirm_m, r.confirm_s, r.red_likely_m) for r in first] == \
         [(r.d_confirm_m, r.confirm_s, r.red_likely_m) for r in second]


def test_unknown_policy_rejected():
  with pytest.raises(ValueError):
    sim.simulate_approach(60.0, 5.0, policy="nope")
