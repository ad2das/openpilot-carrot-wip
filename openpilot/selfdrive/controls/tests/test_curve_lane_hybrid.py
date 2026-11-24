"""curve_lane_hybrid: masks, guards, gate and slew unit tests.

Every rule asserted here is copied from the validated offline reference
(.analysis/scratch/2026-11-20-lane-hybrid/W-SHD6/report_hybrid5.py and W-HYB1/report_hyb.py).
The reference is array-shaped; these tests pin the frame-by-frame equivalent.

Note the reference's own convention: metrics_route() hands simulate() curve_a =
max(|dc|, |kappa|) v^2 as a_lat, so both the entry test and the hold-band test read curve_a,
and a_road = |kappa| v^2 only gates the entry.
"""
import math

import numpy as np

from openpilot.selfdrive.controls.lib.curve_lane_hybrid import (
  CAP_A, CausalMedian, CurveLaneHybrid, GATE_HOLD_A, GATE_REENTRY_A, GATE_REENTRY_CS,
  GATE_ROAD_A, GUARD_BAD, GUARD_LW, GUARD_SHAPE, GUARD_W45, GUARD_REARM_S, HOLD_RELEASE_CS,
  HOLD_S, J_DOWN, J_UP, JUMP_C, SPIKE_OK_RUN, V_MIN, W_C_LO, W_C_SPAN, HybridFrame,
)

DT = 0.05          # 20 Hz modelV2
V = 25.0           # m/s, above V_MIN
V2 = V * V
K_CURVE = 0.002    # 1/m -> curve_a 1.25, a_road 1.25 (enters, above the hold band)
K_HOLD = 0.0005    # 1/m -> curve_a 0.3125 (hold band), a_road 0.3125 (still enters by itself
                   # only through the re-entry clause, which needs curve_a >= 0.5, so it cannot)


def frame(t=0.0, **kw):
  """One clean, eligible, curve-entering frame; kw overrides any field.

  Defaults: curve_a = a_road = 1.25, c_s 0.5 (car right of centre), k_lane 0.001 - k_model
  0.002 -> dk_rest -0.001, w_c 1 -> tgt -0.625 m/s^2 (pulls left).
  """
  d = dict(t=t, v=V, lat=True, sp=False, lane_change=False, lane_mode=False,
           c_s=0.5, c0=0.0, w0=3.2, w15=3.2, w30=3.2, lprob=0.95, rprob=0.95,
           std1=0.10, std2=0.10, spike_judge=True,
           w0_g=3.2, w45=3.2, lwl=3.5, lwr=3.5,
           ymod0=0.0, ymod10=0.0, ylane0=0.0, ylane10=0.0,
           k_lane=0.001, lane_status=0, lane_nan=False,
           dc=K_CURVE, kappa=K_CURVE, k_model=K_CURVE)
  d.update(kw)
  return HybridFrame(**d)


def hold_frame(t, c_s=0.5, dc=K_HOLD, kappa=K_HOLD):
  """A frame in the hold band: curve_a 0.3125 <= GATE_HOLD_A, a_road 0.3125 >= 0.3."""
  return frame(t, c_s=c_s, dc=dc, kappa=kappa, k_lane=dc, k_model=dc)


def run(frames, module=None):
  m = module or CurveLaneHybrid()
  return [m.update(f) for f in frames], m


def on_states(seq):
  mod = CurveLaneHybrid()
  out = []
  for fr in seq:
    mod.update(fr)
    out.append(mod.on)
  return out, mod


# --- sign convention ------------------------------------------------------------------------
def test_sign_car_right_centre_pulls_left():
  """c_s > 0 = car right of centre -> the restore is negative (leftward)."""
  a, m = run([frame(0.0), frame(DT)])
  assert m.on and m.eligible
  assert m.tgt < 0.0
  assert a[1] < 0.0
  assert m.curvature == a[1] / V2


def test_sign_car_left_centre_pulls_right():
  a, m = run([frame(0.0, c_s=-0.5, k_lane=0.003), frame(DT, c_s=-0.5, k_lane=0.003)])
  assert m.tgt > 0.0
  assert a[1] > 0.0


def test_dk_rest_one_sided():
  """A lane solution that pushes away from the centre contributes nothing."""
  # c_s > 0 (right of centre), k_lane above k_model -> dk > 0 -> wrong direction -> 0.
  a, m = run([frame(0.0), frame(DT, k_lane=0.003)])
  assert m.on and m.tgt == 0.0 and a[1] == 0.0


def test_zero_c_s_contributes_nothing():
  _a, m = run([frame(0.0), frame(DT, c_s=0.0)])
  assert m.on and m.tgt == 0.0


# --- cap / weights --------------------------------------------------------------------------
def test_cap_scales_with_v2():
  _a, m = run([frame(0.0), frame(DT, k_lane=-1.0)])
  # dk clipped to +-CAP_A / v^2 = 0.00128, one-sided and leftward -> |tgt| = CAP_A exactly.
  assert abs(m.tgt + CAP_A) < 1e-9


def test_cap_is_in_m_per_s2_not_curvature():
  """The cap is on the acceleration; curvature therefore shrinks as v^2."""
  _a, m = run([frame(0.0, v=20.0, kappa=0.004, dc=0.004),
               frame(DT, v=20.0, kappa=0.004, dc=0.004, k_lane=-1.0)])
  assert m.on
  assert abs(m.tgt + CAP_A) < 1e-9
  assert m.curvature == m.a_add / 20.0 ** 2


def test_w_c_ramp():
  for c_s in (0.05, 0.10, 0.175, 0.25, 0.5):
    _a, m = run([frame(0.0), frame(DT, c_s=c_s)])
    w_c = min(max((abs(c_s) - W_C_LO) / W_C_SPAN, 0.0), 1.0)
    got = m.tgt / (-0.001 * V2)
    assert abs(got - w_c) <= 1e-12, (c_s, got, w_c)


# --- slew -----------------------------------------------------------------------------------
def test_slew_rate_bounded():
  frames = [frame(DT * i) for i in range(40)]
  a, _m = run(frames)
  steps = np.diff(np.concatenate([[0.0], np.asarray(a)]))
  assert np.all(np.abs(steps) <= max(J_UP, J_DOWN) * DT + 1e-12)
  assert a[-1] < 0.0


def test_slew_uses_dt():
  seq = [frame(0.0), frame(0.08)]
  a, _m = run(seq)
  assert abs(a[1] - (-J_UP * 0.08)) < 1e-12


def test_saturation_exact():
  """Within one step of the target, a_add lands exactly on tgt."""
  f1 = frame(0.0, k_lane=0.002 - 0.00005)     # |tgt| = 5e-5 * 625 = 0.03125 > step 0.0135
  f2 = frame(DT, k_lane=0.002 - 0.00001)      # |tgt| = 0.00625 <= step
  a, m = run([f1, f2, f2])
  assert abs(m.tgt + 0.00625) < 1e-12
  assert a[1] == m.tgt                        # exact, not a step short


def test_slew_down_toward_zero_after_off():
  seq = [frame(DT * i) for i in range(4)]
  a, _m = run(seq)
  assert a[1] < 0.0 and a[3] < 0.0
  # a drop zeroes a_add on its own frame
  a2, m2 = run(seq + [frame(4 * DT, sp=True)])
  assert a2[-1] == 0.0 and m2.drop


# --- gate / hold ----------------------------------------------------------------------------
def test_hold_s_zero_releases_frame_after_latch():
  """HOLD_S 0: the latch frame keeps ON, the next frame turns it off (ht >= hold_s)."""
  seq = [frame(0.0), hold_frame(DT), hold_frame(2 * DT), hold_frame(3 * DT)]
  states, mod = on_states(seq)
  assert HOLD_S == 0.0
  assert states == [True, True, False, False]
  assert GATE_HOLD_A == 0.45
  assert mod.on is False


def test_hold_latch_needs_cs_above_latch():
  seq = [frame(0.0), hold_frame(DT, c_s=0.19)]
  states, _mod = on_states(seq)
  assert states == [True, False]


def test_hold_release_cs():
  """A latched hold ends at the first frame with |c_s| < HOLD_RELEASE_CS.

  With HOLD_S 0 the hold_s branch ends the hold on the same frame, so the observable
  difference is whether the latch was reached at all: at |c_s| 0.25 it latches (ON), at
  0.10 it never latches and the gate drops immediately.
  """
  seq = [frame(0.0), hold_frame(DT, c_s=0.25), hold_frame(2 * DT, c_s=HOLD_RELEASE_CS - 0.01)]
  states, _mod = on_states(seq)
  assert states == [True, True, False]
  seq2 = [frame(0.0), hold_frame(DT, c_s=0.10), hold_frame(2 * DT, c_s=0.10)]
  states2, _mod = on_states(seq2)
  assert states2 == [True, False, False]


def test_strong_curve_clears_latch():
  mod = CurveLaneHybrid()
  mod.update(frame(0.0))
  mod.update(hold_frame(DT))
  assert mod._latch is True
  mod.update(hold_frame(2 * DT, dc=0.004, kappa=0.004))
  assert mod._latch is False and mod.on is True


# --- drop -----------------------------------------------------------------------------------
def test_drop_zeroes_same_frame():
  for kw in (dict(sp=True), dict(lane_change=True), dict(lat=False)):
    a, m = run([frame(0.0), frame(DT), frame(2 * DT, **kw)])
    assert m.drop is True, kw
    assert m.on is False
    assert a[-1] == 0.0


def test_drop_zeroes_existing_ramp():
  seq = [frame(0.0), frame(DT), frame(2 * DT), frame(3 * DT, sp=True)]
  a, _m = run(seq)
  assert a[2] != 0.0
  assert a[3] == 0.0


def test_drop_resets_ramp():
  seq = [frame(0.0), frame(DT), frame(2 * DT), frame(3 * DT, sp=True), frame(4 * DT)]
  a, _m = run(seq)
  assert a[3] == 0.0
  # the frame after the drop walks from 0 again
  assert abs(a[4] - (-J_UP * DT)) < 1e-12


# --- time breaks ----------------------------------------------------------------------------
def test_dt_over_0_2_resets():
  seq = [frame(0.0), frame(DT), frame(DT + 0.3)]
  a, m = run(seq)
  assert m.cont is False
  assert a[1] != 0.0
  assert a[2] == 0.0                          # a_add zeroed by the break, no step this frame
  assert m.on is True                         # the gate re-enters on that same frame


def test_dt_over_0_1_is_no_step_but_continues():
  seq = [frame(0.0), frame(0.15)]
  a, m = run(seq)
  assert m.cont is True
  assert a[1] == 0.0


def test_dt_exactly_0_2_continues():
  _a, m = run([frame(0.0), frame(0.2)])
  assert m.cont is True


def test_duplicate_timestamp_breaks_chain():
  _a, m = run([frame(0.0), frame(0.0)])
  assert m.cont is False


# --- zc -------------------------------------------------------------------------------------
def test_zc_on_cs_sign_flip():
  seq = [frame(0.0), frame(DT), frame(2 * DT, c_s=-0.5, k_lane=0.003)]
  a, m = run(seq)
  assert m.zc is True
  assert a[2] > 0.0                            # walks toward the new target after the reset
  assert abs(a[2] - J_UP * DT) <= 1e-12


def test_no_zc_when_sign_holds():
  _a, m = run([frame(0.0), frame(DT)])
  assert m.zc is False


# --- guards ---------------------------------------------------------------------------------
def test_guard_shape_bit():
  _a, m = run([frame(0.0, ylane10=1.0, ymod0=0.0, ymod10=0.0)])
  assert m.guard_bits & GUARD_SHAPE and m.guard is True
  assert m.eligible is True                    # eligibility is computed without the guard


def test_guard_w45_bit():
  _a, m = run([frame(0.0, w45=4.5)])
  assert m.guard_bits & GUARD_W45
  _a, m = run([frame(0.0, w45=3.2 + 0.56)])
  assert m.guard_bits & GUARD_W45
  _a, m = run([frame(0.0, w45=3.2 + 0.5)])
  assert not m.guard_bits & GUARD_W45


def test_guard_w45_finite_required():
  _a, m = run([frame(0.0, w45=float('nan'))])
  assert m.guard_bits & GUARD_W45


def test_guard_lane_width_bit():
  _a, m = run([frame(0.0, lwl=3.5), frame(DT, lwl=4.2)])
  assert m.guard_bits & GUARD_LW


def test_guard_lane_width_window_expires():
  _a, m = run([frame(0.0, lwl=3.5), frame(0.6, lwl=4.2)])
  assert not m.guard_bits & GUARD_LW           # the old sample left the 0.5 s window


def test_guard_lane_width_needs_two_samples():
  _a, m = run([frame(0.0, lwr=4.4)])
  assert not m.guard_bits & GUARD_LW


def test_guard_bad_bits():
  for kw in (dict(k_lane=float('nan')), dict(lane_status=1), dict(lane_nan=True)):
    _a, m = run([frame(0.0, **kw)])
    assert m.guard_bits & GUARD_BAD, kw


def test_guard_rearm_half_second():
  """The raw guard fires at t 0.0; re-arm holds it through t 0.5 and clears at 0.55."""
  seq = [frame(DT * i, k_lane=float('nan') if i == 0 else 0.001) for i in range(15)]
  mod = CurveLaneHybrid()
  states = []
  for fr in seq:
    mod.update(fr)
    states.append((round(fr.t, 3), mod.guard))
  assert GUARD_REARM_S == 0.5
  for t, g in states:
    if t <= 0.5 + 1e-9:
      assert g is True, (t, states)
    else:
      assert g is False, (t, states)


def test_guard_blocks_entry_but_not_eligibility():
  _a, m = run([frame(0.0, w45=4.5)])
  assert m.eligible is True and m.guard is True and m.on is False and m.a_add == 0.0


# --- entry rules ----------------------------------------------------------------------------
def test_road_strength_blocks_entry():
  """curve_a >= 0.8 alone is not enough: a_road must reach GATE_ROAD_A."""
  f = frame(0.0, kappa=0.0002, dc=0.005)       # a_road 0.125, curve_a 3.125
  _a, m = run([f])
  assert m.eligible is True
  assert m.on is False


def test_road_strength_exact_boundary():
  _a, m = run([frame(0.0, kappa=GATE_ROAD_A / V2)])
  assert m.on is True


def test_reentry_clause_needs_cs():
  # curve_a = a_road = 0.625 in [GATE_REENTRY_A (0.45), GATE_ON_A (0.65))
  lo = dict(kappa=0.001, dc=0.001)
  _a, m = run([frame(0.0, c_s=0.15, **lo)])
  assert m.on is False
  _a, m = run([frame(0.0, c_s=GATE_REENTRY_CS, **lo)])
  assert m.on is True
  assert GATE_REENTRY_A == 0.45


def test_below_reentry_strength_never_enters():
  _a, m = run([frame(0.0, c_s=0.9, kappa=0.0007, dc=0.0007)])   # curve_a 0.4375
  assert m.on is False


def test_midspeed_narrow_right_curve_restores_left():
  """Regression for 000004e5 seg 10/12: 50 km/h right curve on 2.50 m narrow lane activates and pulls left."""
  v = 50.0 / 3.6
  kw = dict(v=v, c_s=0.22, w0=2.50, w15=2.54, w30=2.58, w0_g=2.50, w45=2.65,
            kappa=0.0045, dc=0.0045, k_model=0.0045, k_lane=0.0038)
  a, m = run([frame(0.0, **kw), frame(DT, **kw)])
  assert m.eligible is True
  assert m.on is True
  assert m.tgt < 0.0 and a[1] < 0.0


# --- masks ----------------------------------------------------------------------------------
def test_base_mask_fields():
  for kw in (dict(lat=False), dict(lane_mode=True), dict(lane_change=True), dict(sp=True),
             dict(v=V_MIN - 0.01), dict(c_s=float('nan')), dict(c0=float('nan'))):
    _a, m = run([frame(0.0, **kw)])
    assert m.eligible is False, kw


def test_strict_mask_fields():
  for kw in (dict(lprob=0.69), dict(rprob=0.69), dict(std1=0.26), dict(std2=0.26),
             dict(w0=2.44), dict(w0=4.21), dict(w15=2.4), dict(w30=4.3),
             dict(w30=3.7, w0=3.2)):          # |w30 - w0| = 0.5 > W_TAPER
    _a, m = run([frame(0.0, **kw)])
    assert m.eligible is False, kw


def test_k_model_must_be_finite():
  _a, m = run([frame(0.0, k_model=float('nan'))])
  assert m.eligible is False


def test_spike_ok_needs_ten_clean_frames():
  """The jump frame sets run 0; the tenth following clean frame restores eligibility."""
  mod = CurveLaneHybrid()
  elig = []
  for fr in (frame(0.0, c0=0.0), frame(DT, c0=JUMP_C + 0.01)):
    mod.update(fr)
    elig.append(mod.eligible)
  assert elig == [True, False]
  assert mod._spike_run == 0
  for i in range(2, 2 + SPIKE_OK_RUN + 1):
    mod.update(frame(DT * i, c0=JUMP_C + 0.01))
    elig.append(mod.eligible)
  assert elig[2:11] == [False] * 9            # run 1..9
  assert elig[11] is True                     # run 10


def test_spike_judge_false_never_judges():
  _a, m = run([frame(0.0, c0=0.0), frame(DT, c0=10.0, spike_judge=False)])
  assert m.eligible is True


def test_spike_run_survives_time_gap():
  """The offline run counter never resets on a time gap."""
  mod = CurveLaneHybrid()
  for fr in (frame(0.0), frame(DT), frame(10.0)):
    mod.update(fr)
  assert mod._spike_run == SPIKE_OK_RUN + 3


# --- CausalMedian ---------------------------------------------------------------------------
def test_causal_median_trailing_three():
  c = CausalMedian(3)
  assert c.update(0.0, 1.0) == 1.0
  assert c.update(DT, 2.0) == 1.5
  assert c.update(2 * DT, 3.0) == 2.0
  assert c.update(3 * DT, 4.0) == 3.0


def test_causal_median_removes_isolated_spike():
  c = CausalMedian(3)
  vals = [c.update(DT * i, x) for i, x in enumerate([0.0, 1.0, 0.0])]
  assert vals[-1] == 0.0


def test_causal_median_reset_on_time_gap():
  c = CausalMedian(3)
  c.update(0.0, 1.0)
  c.update(DT, 1.0)
  assert c.update(DT + 0.3, 5.0) == 5.0        # window cleared, only the new sample
  assert c.update(DT + 0.35, 7.0) == 6.0


def test_causal_median_reset_flag():
  c = CausalMedian(3)
  c.update(0.0, 1.0)
  c.update(DT, 1.0)
  assert c.update(2 * DT, 9.0, reset=True) == 9.0


def test_causal_median_ignores_nan():
  c = CausalMedian(3)
  c.update(0.0, 1.0)
  assert c.update(DT, float('nan')) == 1.0
  assert c.update(2 * DT, 3.0) == 2.0


def test_causal_median_empty_is_nan():
  c = CausalMedian(3)
  assert math.isnan(c.update(0.0, float('nan')))


def test_causal_median_duplicate_timestamp_clears():
  c = CausalMedian(3)
  c.update(0.0, 1.0)
  assert c.update(0.0, 2.0) == 2.0


def test_causal_median_even_count_midpoint():
  c = CausalMedian(4)
  for i, x in enumerate((1.0, 2.0)):
    last = c.update(DT * i, x)
  assert last == 1.5


# --- reset ----------------------------------------------------------------------------------
def test_reset_clears_state():
  m = CurveLaneHybrid()
  m.update(frame(0.0))
  m.update(frame(DT))
  assert m.a_add != 0.0
  m.reset()
  assert m.a_add == 0.0 and m.on is False and m.guard is False and m.cont is False
  assert m._spike_run == SPIKE_OK_RUN and m._guard_last_t is None
