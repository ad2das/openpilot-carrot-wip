"""W147 v2: the >= 82 km/h model stop cue, unlocked only by a corroborated TLV red.

The < 82 km/h branch must stay byte-identical; these tests pin both sides.
Frame counts follow the constants: TLV_HI_STOP_RED_S / DT_MDL = 10 frames,
TLV_HI_STOP_MODEL_S / DT_MDL = 6 frames, so both conditions hold at frame 10.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.carrot.carrot_functions import (
  CarrotPlanner, XState, DT_MDL,
  TLV_HI_STOP_Y_MAX_M, TLV_HI_STOP_MIN_SUPPORT, TLV_HI_STOP_RED_S, TLV_HI_STOP_MODEL_S,
)
from openpilot.common.filter_simple import MyMovingAverage

N = 70
RED_FRAMES = int(np.ceil(TLV_HI_STOP_RED_S / DT_MDL))      # 10
MODEL_FRAMES = int(np.ceil(TLV_HI_STOP_MODEL_S / DT_MDL))  # 6


def planner():
  p = CarrotPlanner.__new__(CarrotPlanner)
  p.vFilter = MyMovingAverage(10)
  p.xState = XState.e2eCruise
  p.model_stop_clear_since = None
  p.model_stop_sign_t = None
  p.stopSignCount = 0
  p.startSignCount = 0
  p.trafficState = None
  p.tlv_gas_cooldown_frames = 0
  p.last_model_stop_sign = False
  p.last_model_start_sign = False
  # __new__ skips __init__, so every attribute __init__ sets must be mirrored
  # here. The guard test below builds a real CarrotPlanner() instead, which is
  # what actually proves __init__ initialises the W147 state.
  p._hi_stop_carrot_man = None
  p.hi_stop_red_frames = 0
  p.hi_stop_red_gap = 0
  p.hi_stop_model_frames = 0
  p.hi_stop_support_ok = False
  return p


def man(state=1, src=1, support=TLV_HI_STOP_MIN_SUPPORT):
  return SimpleNamespace(trafficState=state, trafficStateSrc=src, tlvDetSupport=support)


def step(p, v_kph, y_last=0.0, model_x=50.0, d_rel=1000.0, v_cruise=60.0, a_ego=0.0,
         man_obj=None):
  """One planner frame. The model expression is satisfied by default."""
  p._hi_stop_carrot_man = man_obj
  v = np.full(N, 0.5)                       # model velocity: < 3.0 and < 0.7 * v[0]
  y = np.zeros(N)
  y[-1] = y_last
  p.check_model_stopping(v_cruise, v, v_kph / 3.6, a_ego, model_x, y, d_rel)
  return p.last_model_stop_sign


def run(p, frames, **kw):
  return [step(p, **kw) for _ in range(frames)]


# --- >= 82 km/h fires only when every condition holds -----------------------
def test_all_conditions_fire_above_82():
  p = planner()
  out = run(p, RED_FRAMES + 2, v_kph=85.0, man_obj=man())
  assert out[:RED_FRAMES - 1] == [False] * (RED_FRAMES - 1), out
  assert out[RED_FRAMES - 1] is True, "fires on the frame that completes 0.5 s"


def test_red_persistence_alone_is_not_enough_before_the_model_timer():
  # Support and red are satisfied at frame RED_FRAMES, but the model timer is
  # the same length here, so removing the model evidence must delay the fire.
  p = planner()
  run(p, RED_FRAMES, v_kph=85.0, man_obj=man())
  assert step(p, 85.0, model_x=900.0, man_obj=man()) is False
  out = run(p, MODEL_FRAMES + 1, v_kph=85.0, man_obj=man())
  assert out[:MODEL_FRAMES - 1] == [False] * (MODEL_FRAMES - 1), out
  assert out[MODEL_FRAMES - 1] is True


def test_below_82_is_the_old_branch_lateral_5():
  p = planner()
  assert run(p, 20, v_kph=81.9, y_last=6.0, man_obj=man(support=0)) == [False] * 20
  q = planner()
  assert run(q, 20, v_kph=81.9, y_last=4.0, man_obj=man(support=0))[-1] is True


def test_above_82_lateral_bound_is_10():
  p = planner()
  out = run(p, RED_FRAMES + 1, v_kph=85.0, y_last=6.0, man_obj=man())
  assert out[-1] is True, "6 m is inside the widened bound"
  assert run(planner(), RED_FRAMES + 1, v_kph=85.0,
             y_last=TLV_HI_STOP_Y_MAX_M + 0.1, man_obj=man()) == [False] * (RED_FRAMES + 1)


def test_support_one_never_unlocks():
  p = planner()
  out = run(p, 40, v_kph=85.0, man_obj=man(support=TLV_HI_STOP_MIN_SUPPORT - 1))
  assert out == [False] * 40


def test_absent_support_field_is_zero():
  p = planner()
  out = run(p, 40, v_kph=85.0, man_obj=SimpleNamespace(trafficState=1, trafficStateSrc=1))
  assert out == [False] * 40


def test_no_carrot_man_never_unlocks():
  p = planner()
  assert run(p, 40, v_kph=85.0, man_obj=None) == [False] * 40


def test_tlv_red_shorter_than_half_a_second_does_not_fire():
  p = planner()
  short = RED_FRAMES - 2
  assert run(p, short, v_kph=85.0, man_obj=man()) == [False] * short
  assert run(p, 30, v_kph=85.0, man_obj=man(state=0)) == [False] * 30


def test_model_expression_shorter_than_0_3_s_does_not_fire():
  p = planner()
  p._hi_stop_carrot_man = man()
  short = MODEL_FRAMES - 2
  assert run(p, short, v_kph=85.0, man_obj=man()) == [False] * short
  assert step(p, 85.0, model_x=900.0, man_obj=man()) is False, "the model timer must reset"
  out = run(p, MODEL_FRAMES, v_kph=85.0, man_obj=man())
  assert out[:MODEL_FRAMES - 1] == [False] * (MODEL_FRAMES - 1), out
  assert out[MODEL_FRAMES - 1] is True


def test_gas_cooldown_suppresses_the_unlock():
  p = planner()
  p.tlv_gas_cooldown_frames = 200
  assert run(p, 40, v_kph=85.0, man_obj=man()) == [False] * 40


def test_non_tlv_source_never_unlocks():
  p = planner()
  assert run(p, 40, v_kph=85.0, man_obj=man(src=0)) == [False] * 40


def test_graded_red_likely_is_not_red():
  p = planner()
  assert run(p, 40, v_kph=85.0, man_obj=man(state=4)) == [False] * 40


def test_one_frame_red_gap_is_tolerated():
  p = planner()
  run(p, RED_FRAMES, v_kph=85.0, man_obj=man())
  assert step(p, 85.0, man_obj=man(state=0)) is True
  assert run(p, 3, v_kph=85.0, man_obj=man())[-1] is True


def test_slow_frame_resets_the_unlock():
  p = planner()
  assert run(p, RED_FRAMES, v_kph=85.0, man_obj=man())[-1] is True
  step(p, 60.0, man_obj=man())
  out = run(p, RED_FRAMES, v_kph=85.0, man_obj=man())
  assert out[:RED_FRAMES - 1] == [False] * (RED_FRAMES - 1), out
  assert out[-1] is True


def test_e2e_cruise_braking_exclusion_still_applies_above_82():
  p = planner()
  assert run(p, 40, v_kph=85.0, a_ego=-2.0, man_obj=man()) == [False] * 40


def test_regen_v_cruise_zero_keeps_the_exclusion_off():
  p = planner()
  assert run(p, 40, v_kph=85.0, a_ego=-2.0, v_cruise=0.0, man_obj=man())[-1] is True


# --- the real constructor must initialise the unlock state -------------------
def test_real_constructor_initialises_hi_stop_state():
  """plannerd can restart at >= 82 km/h on a highway with src 1 / red: the
  first frame then reads hi_stop_support_ok before any slow frame reset it."""
  # __init__ reads MyDrivingMode; DrivingMode(0) is not a member and a fresh
  # test prefix holds no value, so seed the one the constructor requires.
  from openpilot.common.params import Params
  Params().put_int("MyDrivingMode", 3)
  p = CarrotPlanner()
  assert p.hi_stop_support_ok is False
  assert p.hi_stop_red_frames == 0
  assert p.hi_stop_red_gap == 0
  assert p.hi_stop_model_frames == 0
  p._hi_stop_carrot_man = man(support=3)
  v = np.full(N, 0.5)
  y = np.zeros(N)
  p.check_model_stopping(60.0, v, 90.0 / 3.6, 0.0, 50.0, y, 1000.0)
  assert p.last_model_stop_sign is False, "0.5 s of TLV red is not there yet"
  assert p.hi_stop_red_frames == 1
