"""Fitted longitudinal plant: port fidelity + closed-loop smoke on both loops.

The plant in fitted_plant.py is a port of the W-PLANT2 scratch class. These
tests pin three things:

1. Port fidelity: every stored golden row (generated once by the scratch class
   over a fixed synthetic input, both cars) is reproduced within 1e-9.
2. Plant invariants: the standstill flag follows the reported speed against the
   car's own edge, and the true speed never goes negative.
3. Wiring: FarLeadLoop(fitted=...) and ButterStopLoop(fitted=...) run the real
   planner/MPC/LongControl to a stop. The b1 numbers (final gap, end-decel,
   crawl time) are printed next to the ideal plant for reference; they are not
   a pass criterion -- the plant is a distribution-level model and closed-loop
   validation against real stops is a separate task.

Existing suites (test_butter_stop.py, test_butter_stop_maneuvers.py) cover the
default path: with fitted=None no new line in either loop runs.
"""
import json
import os

import numpy as np
import pytest

from openpilot.selfdrive.test.longitudinal_maneuvers.fitted_plant import CARS, FittedPlant
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  FarLeadLoop, FixtureProfile, load_fixture,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_stop_maneuvers import (
  _CASES, ButterStopLoop,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_real_replay import _b1_metrics

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GOLDEN_PATH = os.path.join(DATA_DIR, "fitted_plant_golden.json")
TOL = 1e-9


def _golden():
  with open(GOLDEN_PATH, encoding="utf-8") as handle:
    return json.load(handle)


@pytest.mark.parametrize("car", sorted(CARS))
def test_golden_matches_source_class(car):
  golden = _golden()
  v0 = float(golden["v0"])
  dt = float(golden["dt"])
  n_steps = int(golden["n_steps"])
  every = int(golden["every"])
  u_first, u_knee, u_second = golden["u"]
  switch_v = float(golden["lcs_switch_v"])
  rows = golden["cars"][car]

  plant = FittedPlant(car)
  plant.reset(v0)
  lcs = 1
  got = []
  for k in range(n_steps):
    u = u_first if k < u_knee else u_second
    a_true, v_true, v_kf, a_kf, standstill = plant.step(u, lcs, dt)
    if v_true < switch_v:
      lcs = 2
    if k % every == 0:
      got.append((a_true, v_true, v_kf, a_kf, int(bool(standstill))))

  assert len(got) == len(rows)
  for i, (row, want) in enumerate(zip(got, rows, strict=True)):
    for name, g, w in zip(("a_true", "v_true", "v_kf", "a_kf", "standstill"), row, want,
                          strict=True):
      if name == "standstill":
        assert g == int(w), f"{car} step {i * every} {name}: {g} != {w}"
      else:
        assert abs(g - float(w)) <= TOL, f"{car} step {i * every} {name}: {g} != {w}"


@pytest.mark.parametrize("car", sorted(CARS))
def test_standstill_and_speed_floor(car):
  plant = FittedPlant(car)
  edge = CARS[car]["standstill_edge_speed"]
  plant.reset(0.0)
  assert plant.standstill, "a plant seeded at rest must report standstill"
  assert plant.v_kf == 0.0 and plant.a_kf == 0.0

  plant.reset(8.0, 0.5)
  assert not plant.standstill
  assert plant.v_kf == 8.0 and plant.a_kf == 0.5

  # Brake hard to rest; the flag must follow the reported speed, not v_true,
  # and it must flip exactly at the edge.
  plant.reset(8.0)
  seen_stop = False
  for _ in range(2000):
    _, v_true, v_kf, _, ss = plant.step(-1.5, 1, 0.01)
    assert v_true >= 0.0
    assert ss == (v_kf <= edge)
    if ss:
      seen_stop = True
      break
  assert seen_stop, f"{car}: never reached the standstill edge under 1.5 m/s^2 braking"

  # Creep torque is part of the model: once at rest the car creeps forward
  # again, so the only hard invariant left is that speed never goes negative.
  for _ in range(50):
    _, v_true, _, _, _ = plant.step(-1.5, 1, 0.01)
    assert v_true >= 0.0


def test_fitted_rejects_other_plants():
  plant = FittedPlant("palisade")
  assert plant.kind == CARS["palisade"]["kind"]
  with pytest.raises(ValueError):
    FittedPlant("civic")


def _kf_end_decel(run):
  """Mean |a_kf| over the 0.5 s before the first sample with v_kf < 0.05.

  Mirrors the real-car census (test_butter_real_replay.py), which averages
  |aEgo| over that window at the stop sample.
  """
  t = np.array(run["t"], dtype=float)
  v_kf = np.array(run["v_kf"], dtype=float)
  a_kf = np.array(run["a_kf"], dtype=float)
  stand = np.flatnonzero(v_kf < 0.05)
  if not stand.size:
    return None
  t_k = t[stand[0]]
  window = (t >= t_k - 0.5) & (t <= t_k)
  return float(np.mean(np.abs(a_kf[window]))) if window.any() else None


def test_b1_palisade_runs_to_standstill():
  """FarLeadLoop + fitted Palisade plant over the b1 fixture, ButterStop on and off."""
  fixture = load_fixture("b1")
  rows = {}
  for butter in (0, 1):
    header = dict(fixture["params"])
    header["ButterStop"] = butter
    for mode in ("fitted", "ideal"):
      kwargs = {"fitted": "palisade"} if mode == "fitted" else {}
      run = FarLeadLoop(header, FixtureProfile(fixture), **kwargs).run()
      gap = np.array(run["gap"], dtype=float)
      active = np.array(run["active"], dtype=bool)
      assert run["stopped"], f"B{butter} {mode}: run never stopped"
      assert np.all(gap[active] > 0.0), f"B{butter} {mode}: non-positive gap"
      applied = np.array(run["a_actual"], dtype=float)
      commanded = np.array(run["accel"], dtype=float)
      if mode == "fitted":
        # The fitted plant lags every command, so the applied accel must differ
        # from LongControl's output: proves the branch ran instead of falling
        # through to the identity path.
        assert np.any(np.abs(applied - commanded) > 1e-6), f"B{butter}: fitted plant did not act"
        assert "a_kf" in run and "v_kf" in run, "fitted run must expose the KF outputs"
        kf_decel = _kf_end_decel(run)
        assert kf_decel is not None and np.isfinite(kf_decel), \
          f"B{butter}: kf end-decel not finite ({kf_decel})"
      else:
        assert np.array_equal(applied, commanded), f"B{butter}: ideal path changed"
        assert "a_kf" not in run and "v_kf" not in run, "ideal run must not carry KF keys"
        kf_decel = None
      rows[(mode, butter)] = _b1_metrics(run)
      rows[(mode, butter)]["kf_end_decel"] = kf_decel

  for (mode, butter), m in sorted(rows.items()):
    print(f"b1 {mode:6s} B{butter}: final_gap_m={m['final_gap_m']} mean_accel_last_0.5s={m['mean_accel_last_0.5s']} "
          + f"crawl_time_s={m['crawl_time_s']} kf_end_decel={m['kf_end_decel']} stopped={m['stopped']}")


def test_full_run_extends_without_changing_prefix():
  """full_run=True appends steps; every sample the default run produced is identical."""
  fixture = load_fixture("b1")
  header = dict(fixture["params"])
  profile_duration = FixtureProfile(fixture).duration

  a = FarLeadLoop(header, FixtureProfile(fixture), fitted="palisade").run()
  b = FarLeadLoop(header, FixtureProfile(fixture), fitted="palisade", full_run=True).run()

  assert len(b["t"]) > len(a["t"]), \
    f"full_run did not extend the run ({len(b['t'])} <= {len(a['t'])} steps)"

  per_step = [k for k, v in a.items() if isinstance(v, list)]
  assert len(per_step) >= 10, f"unexpected per-step key set: {per_step}"
  for key in per_step:
    assert len(b[key]) >= len(a[key]), f"{key}: full run shorter ({len(b[key])} < {len(a[key])})"
    prefix = b[key][:len(a[key])]
    assert prefix == a[key], f"{key}: prefix diverged between default and full_run"

  # Explicitly pin the two stops' shared samples too, so a change that only
  # alters the shared tail is caught even if list equality above is relaxed.
  assert b["stopped"] == a["stopped"], "stopped flag changed"
  assert b["fcw_steps"] >= a["fcw_steps"], "fcw step count decreased"

  # The full run integrates to the fixture's last possible step.
  # max_frames = profile.n_steps * PLANNER_PERIOD, dt = DT_CTRL (test_far_lead_replay.py:444)
  assert b["t"][-1] >= profile_duration - 0.05, \
    f"full run ended at {b['t'][-1]:.2f} s, before the fixture end {profile_duration:.2f} s"


def test_butter_stop_loop_k5_runs_to_standstill():
  group, maneuver = _CASES[0]
  logs = ButterStopLoop(maneuver, butter=1, fitted="k5").run()
  t, v, gaps = logs[:, 0], logs[:, 1], logs[:, 4] - logs[:, 3]
  assert bool(np.any(v < 0.05)), f"{group}-{maneuver.title}: k5 loop never stopped"
  assert float(np.min(gaps)) > 0.0, f"{group}-{maneuver.title}: non-positive gap"
  assert np.all(np.diff(t) > 0.0)
