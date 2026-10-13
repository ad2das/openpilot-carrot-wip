"""LaneCenterBlend: learned lane-centre correction unit tests.

Tests 2-4 are a kinematic closed-loop simulation of a pure-pursuit model
driving a biased lane.  Run with -s to see the printed sim numbers.
"""
import copy
import json

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.lane_center_blend import (
  GATE_AVOID, GATE_CURVE, GATE_E_LARGE, GATE_INVALID, GATE_JUMP, GATE_LANE_CHANGE,
  GATE_LANE_MODE, GATE_OFF, GATE_PROB, GATE_SPEED, GATE_STEER, GATE_WIDTH,
  LaneCenterBlend, OFFSET_MAX, dump_table, load_table,
)
from openpilot.selfdrive.modeld.constants import ModelConstants

T = np.array(ModelConstants.T_IDXS)
DT = DT_MDL
V = 25.0
HALF_WIDTH = 1.75


def _lanes(p, psi, kr, v=V):
  x = v * T
  centre = -p - psi * x + kr * x ** 2 / 2.0
  return x, centre


def _model_path(p, psi, kr, b, v=V):
  x, centre = _lanes(p, psi, kr, v)
  lm = 1.8 * v
  path_y = centre + b - (centre[0] + b) * np.exp(-x / lm)
  a = (b - p) - psi * lm + kr * lm ** 2 / 2.0
  km = 2.0 * a / lm ** 2
  return path_y, km


class _Sim:
  def __init__(self, b0=0.0, b1=0.0, blend=None, enabled=True, kr_fn=None, bump_fn=None):
    self.p = 0.0
    self.psi = 0.0
    self.t = 0.0
    self.b0 = b0
    self.b1 = b1
    self.blend = blend if blend is not None else LaneCenterBlend()
    self.enabled = enabled
    self.kr_fn = kr_fn or (lambda t: 0.0)
    self.bump_fn = bump_fn or (lambda t: 0.0)
    self.p_hist = []
    self.kr_hist = []
    self.offset_hist = []
    self.avoid_frames = []

  def step(self):
    kr = self.kr_fn(self.t)
    b = self.b0 + self.b1 * kr + self.bump_fn(self.t)
    x, centre = _lanes(self.p, self.psi, kr)
    path_y, km = _model_path(self.p, self.psi, kr, b)
    dk = self.blend.update(self.enabled, False, V, 0.9, 0.9, x, centre - HALF_WIDTH, centre + HALF_WIDTH,
                           T, x, path_y, False, False)
    kc = km + dk
    self.psi += V * (kc - kr) * DT
    self.p += V * self.psi * DT
    self.t += DT
    self.p_hist.append(self.p)
    self.kr_hist.append(kr)
    self.offset_hist.append(self.blend.offset)
    if self.blend.gates & GATE_AVOID:
      self.avoid_frames.append(len(self.p_hist) - 1)

  def run(self, seconds):
    for _ in range(int(round(seconds / DT))):
      self.step()


def _mean_abs(hist, seconds):
  n = int(round(seconds / DT))
  return float(np.mean(np.abs(np.array(hist[-n:]))))


def _kr_wave(t):
  ph = t % 40.0
  if ph < 5.0:
    return 0.0
  if ph < 20.0:
    return 0.005
  if ph < 25.0:
    return 0.0
  return -0.005


def _bump(sign, t_start=2.0):
  def f(t):
    if t < t_start or t > t_start + 5.0:
      return 0.0
    if t < t_start + 1.0:
      return sign * 0.5 * (t - t_start)
    if t < t_start + 4.0:
      return sign * 0.5
    return sign * 0.5 * (t_start + 5.0 - t)
  return f


def _run_avoidance(sign):
  base = _Sim(b0=0.25)
  base.run(180.0)
  on = _Sim(b0=0.25, blend=copy.deepcopy(base.blend), enabled=True, bump_fn=_bump(sign))
  on.p, on.psi = base.p, base.psi
  on.run(9.0)
  off = _Sim(b0=0.25, blend=copy.deepcopy(base.blend), enabled=False, bump_fn=_bump(sign))
  off.p, off.psi = base.p, base.psi
  off.run(9.0)
  s = float(sign)
  peak_on = float(np.max(s * np.array(on.p_hist)))
  peak_off = float(np.max(s * np.array(off.p_hist)))
  table_change = float(np.max(np.abs(on.blend.table - base.blend.table)))
  print(f"bump {sign:+.1f}: peak_ON={peak_on:.4f} peak_OFF={peak_off:.4f} " +
        f"avoid_frames_ON={len(on.avoid_frames)} avoid_frames_OFF={len(off.avoid_frames)} " +
        f"table_change={table_change:.5f}")
  return base, on, off, peak_on, peak_off, table_change


def _synth(centre0=0.0, kr=0.0, width=3.5, v=V, nan_lll=False):
  x = v * T
  centre = centre0 + kr * x ** 2 / 2.0
  lll = centre - width / 2.0
  rll = centre + width / 2.0
  if nan_lll:
    lll = lll.copy()
    lll[5] = np.nan
  return x, lll, rll, centre.copy()


def _warm(centre0=0.0, frames=150):
  blend = LaneCenterBlend()
  x, lll, rll, centre = _synth(centre0=centre0)
  for _ in range(frames):
    blend.update(True, False, V, 0.9, 0.9, x, lll, rll, T, x, centre, False, False)
  return blend


def _one_update(warm_blend, centre0, **kw):
  blend = copy.deepcopy(warm_blend)
  x, lll, rll, centre = _synth(centre0=centre0, kr=kw.get("kr", 0.0), width=kw.get("width", 3.5),
                               v=kw.get("v", V), nan_lll=kw.get("nan_lll", False))
  blend.update(True, kw.get("lane_mode", False), kw.get("v", V),
               kw.get("l_prob", 0.9), kw.get("r_prob", 0.9), x, lll, rll, T, x, centre,
               kw.get("lane_change", False), kw.get("steering", False))
  return blend


def test_off_returns_exactly_zero_and_keeps_state():
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  table_before = blend.table.copy()
  x, centre = _lanes(0.0, 0.0, 0.0)
  out = blend.update(False, False, V, 0.9, 0.9, x, centre - HALF_WIDTH, centre + HALF_WIDTH, T, x, centre, False, False)
  assert out == 0.0
  assert np.array_equal(blend.table, table_before)
  assert blend.gates == GATE_OFF


def test_straight_bias_is_corrected_when_on():
  on = _Sim(b0=0.25)
  on.run(180.0)
  off = _Sim(b0=0.25, enabled=False)
  off.run(180.0)

  mean_on = _mean_abs(on.p_hist, 30.0)
  mean_off = _mean_abs(off.p_hist, 30.0)
  print(f"straight b0=+0.25: mean|p| last 30 s ON={mean_on:.4f} OFF={mean_off:.4f} " +
        f"max|offset|_ON={max(abs(v) for v in on.offset_hist):.4f}")
  assert mean_on < 0.10
  assert mean_off > 0.20
  assert max(abs(v) for v in on.offset_hist) <= OFFSET_MAX

  p = np.array(on.p_hist)
  i0 = None
  run_max = 0.0
  for i in range(len(p)):
    run_max = max(run_max, abs(p[i]))
    if i > 0 and run_max >= 0.05 and abs(p[i]) < 0.05:
      i0 = i
      break
  assert i0 is not None
  sign = 1.0 if p[i0] >= 0.0 else -1.0
  assert float(np.min(sign * p[i0:])) >= -0.05


def test_curve_bias_is_corrected_when_on():
  on = _Sim(b1=50.0, kr_fn=_kr_wave)
  on.run(900.0)
  off = _Sim(b1=50.0, kr_fn=_kr_wave, enabled=False)
  off.run(900.0)

  p_on = np.array(on.p_hist)
  p_off = np.array(off.p_hist)
  kr = np.array(on.kr_hist)
  n = int(round(120.0 / DT))
  mask = np.abs(kr[-n:]) > 0.0
  mean_on = float(np.mean(np.abs(p_on[-n:][mask])))
  mean_off = float(np.mean(np.abs(p_off[-n:][mask])))
  print(f"curves b1=+50: mean|p| in curves last 120 s ON={mean_on:.4f} OFF={mean_off:.4f} " +
        f"max|table|_ON={float(np.max(np.abs(on.blend.table))):.4f}")
  assert mean_on < 0.12
  assert mean_off > 0.20
  assert np.all(np.abs(on.blend.table) <= OFFSET_MAX + 1e-12)


def test_avoidance_is_preserved():
  base, on, off, peak_on, peak_off, table_change = _run_avoidance(+1)
  assert peak_on >= peak_off - 0.08
  assert table_change < 0.01
  assert len(on.avoid_frames) >= 1

  base, on, off, peak_on, peak_off, table_change = _run_avoidance(-1)
  assert peak_on >= peak_off - 0.08
  assert table_change < 0.01


def test_gates_single_update_after_warm():
  warm = _warm()
  assert warm.w == 1.0
  w_gated = 1.0 - 2.0 * DT

  for kwargs, bit in (
      ({"lane_mode": True}, GATE_LANE_MODE),
      ({"v": 5.0}, GATE_SPEED),
      ({"l_prob": 0.5}, GATE_PROB),
      ({"width": 2.0}, GATE_WIDTH),
      ({"lane_change": True}, GATE_LANE_CHANGE),
      ({"kr": 0.03}, GATE_CURVE),
      ({"nan_lll": True}, GATE_INVALID),
  ):
    blend = _one_update(warm, centre0=0.0, **kwargs)
    assert blend.gates & bit, kwargs
    assert blend.w == pytest.approx(w_gated), kwargs

  warm_steer = _warm(centre0=0.3)
  table_before = warm_steer.table.copy()
  blend = _one_update(warm_steer, centre0=0.3, steering=True)
  assert blend.gates == GATE_STEER
  assert blend.w == pytest.approx(1.0)
  assert np.array_equal(blend.table, table_before)
  learn_control = _one_update(warm_steer, centre0=0.3)
  assert not np.array_equal(learn_control.table, table_before)

  warm_e = _warm(centre0=0.8)
  table_before = warm_e.table.copy()
  blend = _one_update(warm_e, centre0=0.8)
  assert blend.gates == GATE_E_LARGE
  assert blend.w == pytest.approx(1.0)
  assert np.array_equal(blend.table, table_before)

  warm_jump = _warm(centre0=0.1)
  table_before = warm_jump.table.copy()
  blend = _one_update(warm_jump, centre0=0.5)
  assert blend.gates == GATE_JUMP
  assert blend.w == pytest.approx(1.0)
  assert np.array_equal(blend.table, table_before)


def test_lane_change_hold_three_seconds():
  blend = _warm()
  x, lll, rll, centre = _synth(centre0=0.0)
  blend.update(True, False, V, 0.9, 0.9, x, lll, rll, T, x, centre, True, False)
  assert blend.gates & GATE_LANE_CHANGE
  count = 0
  for _ in range(200):
    blend.update(True, False, V, 0.9, 0.9, x, lll, rll, T, x, centre, False, False)
    if not (blend.gates & GATE_LANE_CHANGE):
      break
    count += 1
  assert abs(count - 3.0 / DT) <= 1


def test_stale_table_decay():
  blend = LaneCenterBlend(table=np.full(7, 0.1))
  x, lll, rll, centre = _synth(centre0=0.0)
  for _ in range(int(round(70.0 / DT))):
    blend.update(True, False, V, 0.9, 0.9, x, lll, rll, T, x, centre, False, True)
  expected = 0.1 * np.exp(-(70.0 - 10.0) / 60.0)
  assert np.allclose(blend.table, np.full(7, expected), rtol=0.02)


def test_load_table_validation():
  table = np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.25, -0.05])
  assert np.allclose(load_table(dump_table(table)), table)

  zeros = np.zeros(7)
  assert np.array_equal(load_table(""), zeros)
  assert np.array_equal(load_table("x"), zeros)
  assert np.array_equal(load_table(json.dumps([0.0] * 6)), zeros)
  assert np.array_equal(load_table(json.dumps([0.0] * 6 + [float("nan")])), zeros)

  assert np.allclose(load_table(json.dumps([0.9] * 7)), np.full(7, OFFSET_MAX))


def test_lateral_acceleration_cap():
  blend = LaneCenterBlend(table=np.full(7, OFFSET_MAX))
  blend.w = 1.0
  v = 10.0
  x, lll, rll, centre = _synth(centre0=0.0, v=v)
  out = blend.update(True, False, v, 0.9, 0.9, x, lll, rll, T, x, centre, False, False)
  assert abs(out) * v ** 2 <= 0.25 + 1e-9


def _step_bump(sign, amp, dur, t_start=2.0):
  def f(t):
    if t < t_start or t > t_start + dur:
      return 0.0
    return sign * amp * min(1.0, t - t_start, t_start + dur - t)
  return f


@pytest.mark.parametrize("amp", [0.3, 0.5])
@pytest.mark.parametrize("sign", [+1, -1])
@pytest.mark.parametrize("dur", [5.0, 20.0])
def test_avoidance_either_side_of_learned_bias(amp, sign, dur):
  base = _Sim(b0=0.25)
  base.run(180.0)
  sims = []
  for enabled in (True, False):
    sim = _Sim(b0=0.25, blend=copy.deepcopy(base.blend), enabled=enabled, bump_fn=_step_bump(sign, amp, dur))
    sim.p, sim.psi = base.p, base.psi
    sim.run(dur + 6.0)
    sims.append(sim)
  on, off = sims
  peak_on = float(np.max(sign * np.array(on.p_hist)))
  peak_off = float(np.max(sign * np.array(off.p_hist)))
  table_change = float(np.max(np.abs(on.blend.table - base.blend.table)))
  print(f"amp={amp} sign={sign:+d} dur={dur}: peak_ON={peak_on:.4f} peak_OFF={peak_off:.4f} " +
        f"avoid_frames_ON={len(on.avoid_frames)} table_change={table_change:.5f}")
  assert len(on.avoid_frames) >= int(round(1.0 / DT))
  assert peak_on >= peak_off - 0.08
  assert table_change < (0.02 if dur <= 5.0 else 0.05)


@pytest.mark.parametrize("b0", [0.3, -0.3])
def test_large_straight_bias_still_corrected(b0):
  on = _Sim(b0=b0)
  on.run(300.0)
  mean_on = _mean_abs(on.p_hist, 30.0)
  print(f"straight b0={b0:+.2f}: mean|p| last 30 s ON={mean_on:.4f}")
  assert mean_on < 0.10
