"""W-RRL2: e2eStop must never move the stop line forward once inside 10 m."""
import json
import os

import numpy as np

from openpilot.selfdrive.carrot.carrot_functions import STOP_REARM_MAX, e2e_stop_rearm
from openpilot.common.filter_simple import MyMovingAverage

DT_MDL = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "rearm_4bc_s30.json")


def test_far_phase_unchanged():
  # Above 10 m the model reading still replaces the dead-reckoned distance.
  assert e2e_stop_rearm(0.0, 30.0, False) == (30.0, False)
  assert e2e_stop_rearm(25.0, 28.0, False) == (28.0, False)
  # A reading inside the limit never pulls the stop line forward.
  assert e2e_stop_rearm(25.0, 8.0, False) == (25.0, False)


def test_near_latch():
  assert e2e_stop_rearm(9.9, 10.5, False) == (9.9, True)
  assert STOP_REARM_MAX == 10.0


def test_field_case_4bc_s30():
  # Field 4bc s30: 2.562 -> 10.397 m in one frame turned the plan to cruise.
  assert e2e_stop_rearm(2.562, 10.397, False) == (2.562, True)


def test_latched_stays_latched():
  assert e2e_stop_rearm(2.0, 30.0, True) == (2.0, True)
  assert e2e_stop_rearm(0.0, 30.0, True) == (0.0, True)


def _replay(frames, use_rearm):
  """Replay the real e2eStop stop-distance path on a logged 20 Hz plan series.

  Mirrors carrot_functions.py:370-390 (_stop_x_rl limiter), 510-512 (filters),
  1512-1521 (stop_dist + update) and 1564 (travel decay). Only frames whose logged
  xState is 3 run the stop branch; the first such frame seeds actual_stop_distance
  from the rate-limited reading, as the e2eStop entry does.
  """
  f1 = MyMovingAverage(3)
  f2 = MyMovingAverage(15)
  rl = None
  asd = 0.0
  near = False
  started = False
  out = []
  for fr in frames:
    x31 = fr["x31"]
    xstop = f2.process(f1.process(x31, median=True))
    v_ego = fr["v_ego"]
    if rl is None:
      rl = xstop
    else:
      max_close = v_ego * DT_MDL + 0.5
      rl = xstop if xstop > rl else max(rl - max_close, xstop)

    if fr["xstate"] == 3:
      if not started:
        started = True
        asd = rl
        near = False
      else:
        ratio = np.interp(v_ego * 3.6, [0, 100], [1.0, 0.7])
        stop_dist = rl * np.interp(rl, [0, 50], [1.0, ratio])
        if use_rearm:
          asd, near = e2e_stop_rearm(asd, stop_dist, near)
        elif stop_dist > 10.0:
          asd = stop_dist
      asd = max(0.0, asd - v_ego * DT_MDL)
    else:
      asd = 0.0
      rl = xstop
    out.append(dict(t=fr["t_rel"], v_ego=v_ego, asd=asd))
  return out


def _fixture():
  with open(FIXTURE, encoding="utf-8") as f:
    fx = json.load(f)
  return fx["frames"]


def test_real_log_replay():
  frames = _fixture()
  assert len(frames) > 200
  old = _replay(frames, use_rearm=False)
  new = _replay(frames, use_rearm=True)

  # (a) the OLD rule reproduces the field re-arm: a > 7 m jump near -5.17 s.
  jumps = [(old[i]["asd"] - old[i - 1]["asd"], old[i]["t"])
           for i in range(1, len(old)) if -6.5 <= old[i]["t"] <= -3.7]
  jump, t_jump = max(jumps)
  assert jump > 7.0, f"old rule did not re-arm: max jump {jump:.3f} m"
  assert abs(t_jump - (-5.17)) <= 0.1, f"old rule re-armed at {t_jump:+.2f} s, expected -5.17"

  # (b) with the new rule the stop distance never rises again in that window.
  rises = [(new[i]["t"], new[i]["asd"] - new[i - 1]["asd"])
           for i in range(1, len(new)) if -6.5 <= new[i]["t"] <= -3.7 and new[i]["asd"] > new[i - 1]["asd"]]
  assert not rises, f"new rule moved the stop line forward: {rises[:5]}"

  # (c) both rules agree until the old rule first re-arms from below 10 m.
  first_old_jump = next(i for i in range(1, len(old)) if old[i]["asd"] - old[i - 1]["asd"] > 1.0
                        and old[i - 1]["asd"] < STOP_REARM_MAX)
  for i in range(first_old_jump):
    assert abs(old[i]["asd"] - new[i]["asd"]) < 1e-9, f"rules differ before the re-arm at t={old[i]['t']:+.2f}"
