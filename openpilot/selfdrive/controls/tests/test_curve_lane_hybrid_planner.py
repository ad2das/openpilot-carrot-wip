"""Planner-level test for the curve-lane hybrid wiring (SPEC_2B section 2).

Compiles the real lateral_planner.py class bodies (_HybridParams, HybridLanePath, LateralPlanner)
against stub MPC / LanePlanner objects and the REAL openpilot curve_lane_hybrid module, then drives
update() + publish() frame by frame. What is pinned here:

  - LaneRecenterAssist 0, or lane mode, does no hybrid MPC work at all (run count 0);
  - LRA 1 laneless on a curve publishes laneCenterBlendCurvature == lcb + recenter + hybrid, once;
  - the hybrid run never assigns the module-level LATERAL_ACCEL_COST / LATERAL_JERK_COST /
    STEERING_RATE_COST (the lateral_planner.py:92 global trap);
  - a run after a skipped frame resets the hybrid MPC to the zero state; every run seeds
    x0[3] = seed;
  - the four new LateralPlan fields exist in the schema and are written every frame;
  - a fault inside the hybrid is contained: P0 publishes exactly what it publishes with the
    hybrid switched off by LaneRecenterAssist 0, and the failure is counted (hybrid_errors).
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.curve_lane_hybrid import (
  CurveLaneHybrid, CausalMedian, HybridFrame, RUN_V_MIN, PREWARM_ROAD_A, PREWARM_CURVE_A,
)
from openpilot.selfdrive.controls.lib.lane_model_speed import LaneModelSpeedGuard

SOURCE = Path(__file__).parents[1] / 'lib/lateral_planner.py'

DT = 0.05              # DT_MDL: the rate the hybrid chain integrates on
CURVE_A = 0.006        # road centre line = CURVE_A * x^2, so the road curvature is 0.012 1/m
SPEED = 20.0           # m/s, above RUN_V_MIN with room to spare
CONTROL_N = 17         # drive_helpers.CONTROL_N: the published triple's length


class StubMpc:
  """Records runs/resets and fills x_sol from a fixed curvature and honest distances."""

  def __init__(self):
    self.curvature = 0.0
    self.runs = 0
    self.resets = []
    self.weights = None
    self.reset(x0=np.zeros(4))

  def reset(self, x0):
    self.resets.append(np.array(x0, dtype=float).copy())
    self.x_sol = np.zeros((33, 4))
    self.u_sol = np.zeros((32, 1))
    self.solution_status = 0
    self.cost = 0.0
    self.solve_time = 0.001
    self.x0_last = np.array(x0, dtype=float).copy()

  def set_weights(self, path, heading, lat_accel, lat_jerk, steering_rate):
    self.weights = (path, heading, lat_accel, lat_jerk, steering_rate)

  def run(self, x0, p, y, heading, yaw_rate):
    self.runs += 1
    self.x0_last = np.array(x0, dtype=float).copy()
    v = np.asarray(p)[:, 0]
    self.x_sol[:, 3] = self.curvature * v
    self.x_sol[:, 2] = 0.001
    self.x_sol[:, 0] = np.linspace(0.5, 16.0, 33)
    self.solution_status = 0
    self.solve_time = 0.001


class StubLanePlanner:
  """Lane geometry with a real sign convention: ll_x ascending, centre = amp * x^2."""

  def __init__(self):
    self.debugText = ''
    self.l_prob = 0.95
    self.r_prob = 0.95
    self.lane_width = 3.5
    self.lane_width_left = 3.5
    self.lane_width_right = 3.5
    self.offset_total = 0.0
    self.lanefull_mode = False
    self.lane_change_multiplier = 1.0
    self.d_prob_count = 0
    self.d_prob = 0.0
    self.curvature = 0.0
    self.lane_offset_filtered = SimpleNamespace(x=0.0)
    self.ll_x = np.linspace(0.0, 40.0, 33)
    self.ll_t = np.linspace(0.0, 2.0, 33)
    self.set_shape(CURVE_A)

  def set_shape(self, amp):
    """Set the road's quadratic centre-line coefficient; 0.0 is a perfectly straight road."""
    centre = amp * self.ll_x ** 2
    self.lll_y = centre - 1.75
    self.rll_y = centre + 1.75

  def parse_model(self, md):
    pass

  def get_d_path(self, cs, speed, t, path, curve_speed):
    return path, bool(self.lanefull_mode)


class StubLaneCenterBlend:
  def __init__(self, table=None):
    self.table = np.zeros(7)
    self.curvature = 0.010
    self.offset = 0.02
    self.w = 0.5
    self.r_dev = 0.0
    self.gates = 0
    self.recenter_curvature = 0.004
    self.recenter_active = True
    self.calls = 0

  def reset(self):
    pass

  def update(self, *args, **kwargs):
    self.calls += 1
    return self.curvature


class Message:
  def __init__(self, which):
    if which == 'lateralPlan':
      self.lateralPlan = SimpleNamespace(position=SimpleNamespace(x=[], y=[], z=[]),
                                        solverState=None)
    self.valid = True


def make_params(lra=0, path_offset=0):
  values = {
    'LatMpcPathCost': '200', 'LatMpcMotionCost': '7', 'LatMpcAccelCost': '120',
    'LatMpcJerkCost': '4', 'LatMpcSteeringRateCost': '7', 'LatMpcInputOffset': '4',
    'AdjustLaneOffset': '0', 'PathOffset': str(path_offset), 'SteerActuatorDelay': '13',
    'LatSmoothSec': '13', 'LaneCenterBlend': '1', 'LaneRecenterAssist': str(lra),
  }

  def _get(key, default=None):
    return values.get(key, default)

  return SimpleNamespace(
    get=_get,
    get_int=lambda key, default=0: int(float(values[key])) if key in values else default,
    get_float=lambda key, default=0.0: float(values[key]) if key in values else default,
    get_bool=lambda key, default=False: bool(values.get(key, default)),
    put_nonblocking=lambda *a, **k: None,
    put=lambda *a, **k: None,
  )


def _lag_curvature(cp, v_ego, psis, curvatures, steer_actuator_delay, distances):
  """Stand-in for drive_helpers.get_lag_adjusted_curvature: the leading curvature, unfiltered."""
  if len(curvatures) != CONTROL_N:
    return float('nan')
  return float(curvatures[0])


def build_planner(lra=0, path_offset=0):
  """Compile the production class bodies against the stubs and return (planner, namespace)."""
  tree = ast.parse(SOURCE.read_text(encoding='utf8'))
  wanted = {'_HybridParams', 'HybridLanePath', 'LateralPlanner'}
  body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
  assert {n.name for n in body} == wanted, 'lateral_planner.py lost a hybrid class body'
  namespace = {
    'np': np,
    'DT_MDL': DT,
    'TRAJECTORY_SIZE': 33,
    'MIN_SPEED': 1.0,
    'LAT_MPC_N': 32,
    'CONTROL_N': CONTROL_N,
    'LATERAL_ACCEL_COST': 0.0,
    'LATERAL_JERK_COST': 0.04,
    'STEERING_RATE_COST': 700.0,
    'HYBRID_CONTINUOUS': False,
    'HYBRID_RUN_ALWAYS': False,
    'Params': lambda: make_params(lra, path_offset),
    'LateralMpc': StubMpc,
    'LanePlanner': StubLanePlanner,
    'LaneModelSpeedGuard': LaneModelSpeedGuard,
    'LaneCenterBlend': StubLaneCenterBlend,
    'CurveLaneHybrid': CurveLaneHybrid,
    'CausalMedian': CausalMedian,
    'HybridFrame': HybridFrame,
    'RUN_V_MIN': RUN_V_MIN,
    'PREWARM_ROAD_A': PREWARM_ROAD_A,
    'PREWARM_CURVE_A': PREWARM_CURVE_A,
    'FIT_HORIZON_S': 2.0,
    'FIT_MIN_M': 20.0,
    'get_lag_adjusted_curvature': _lag_curvature,
    'dump_table': lambda table: '[]',
    'load_table': lambda text: None,
    'time': SimpleNamespace(monotonic=lambda: 0.0),
    'log': log,
    'messaging': SimpleNamespace(new_message=Message),
    'yaw_from_path_no_scipy': lambda *a, **k: (np.zeros(33), np.zeros(33)),
    'cloudlog': SimpleNamespace(exception=lambda *a, **k: None,
                                warning=lambda *a, **k: None,
                                error=lambda *a, **k: None),
  }
  exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), namespace)
  cp = SimpleNamespace(wheelbase=2.9, centerToFront=1.2, mass=2000, tireStiffnessRear=200000)
  return namespace['LateralPlanner'](cp), namespace


class _Msg(dict):
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e


def inputs(t, speed=SPEED, lane_speed=0.0, lat=True, y_off=0.0, dc=0.02, desire=0):
  x = np.linspace(0.0, 40.0, 33)
  model = SimpleNamespace(
    position=SimpleNamespace(x=x, y=np.full(33, y_off), z=np.zeros(33),
                            t=np.linspace(0.0, 2.0, 33)),
    orientation=SimpleNamespace(x=np.zeros(33), z=np.zeros(33)),
    orientationRate=SimpleNamespace(z=np.zeros(33)),
    velocity=SimpleNamespace(x=np.full(33, speed), y=np.zeros(33), z=np.zeros(33)),
    acceleration=SimpleNamespace(x=np.zeros(33)),
    action=SimpleNamespace(desiredCurvature=dc),
    laneLineStds=[0.1, 0.1, 0.1],
    meta=SimpleNamespace(desire=desire, laneChangeState=0, laneChangeDirection=0,
                         laneWidthLeft=3.5, laneWidthRight=3.5),
  )
  sm = {
    'modelV2': model,
    'carState': SimpleNamespace(vEgo=speed, useLaneLineSpeed=lane_speed, steeringPressed=False),
    'controlsState': SimpleNamespace(curvature=0.01, desiredCurvature=dc),
    'carControl': SimpleNamespace(latActive=lat),
    'carrotMan': SimpleNamespace(vTurnSpeed=200, atc_active=False),
    'logMonoTime': {'modelV2': int(round(t / 1e-9))},
  }
  sm['all_checks'] = lambda service_list: True
  sm['seen'] = {'carControl': True}
  return _Msg(sm)


def drive(planner, frames, start_t=1.0, shape=CURVE_A, **kwargs):
  """Run `frames` update+publish cycles on a road of the given shape; return the published plans."""
  planner.LP.set_shape(shape)
  published = []
  pm = SimpleNamespace(sent=[])
  pm.send = lambda name, msg: pm.sent.append((name, msg))
  for i in range(frames):
    sm = inputs(start_t + i * DT, **kwargs)
    planner.update(sm, SimpleNamespace(atc_active=False))
    planner.publish(sm, pm, SimpleNamespace())
    published.append(pm.sent[-1][1].lateralPlan)
  return published


def load_yaw_from_path():
  """Compile the production yaw_from_path_no_scipy (+ its smoothing helper) out of SOURCE."""
  tree = ast.parse(SOURCE.read_text(encoding='utf8'))
  wanted = {'smooth_moving_avg', 'yaw_from_path_no_scipy'}
  body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
  assert {n.name for n in body} == wanted, 'lateral_planner.py lost a yaw helper'
  ns = {'np': np}
  exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), ns)
  return ns['yaw_from_path_no_scipy']


def hybrid_mpc(planner):
  """The hybrid path's own MPC stub (never P0's planner.lat_mpc)."""
  return planner.hybrid_lane.mpc


def module_costs(namespace):
  """The three module-level cost globals, read straight out of the compiled namespace."""
  return tuple(float(namespace[name]) for name in
               ('LATERAL_ACCEL_COST', 'LATERAL_JERK_COST', 'STEERING_RATE_COST'))


# --- 1. no hybrid work when the switch is off or the car is in lane mode ---------------------

def test_assist_off_never_runs_the_hybrid_mpc():
  planner, _ = build_planner(lra=0)
  published = drive(planner, 40, lat=True)
  assert hybrid_mpc(planner).runs == 0, 'LaneRecenterAssist 0 must not run the hybrid MPC'
  for lp in published:
    assert lp.curveHybridCurvature == 0.0
    assert lp.curveHybridActive is False
    assert lp.curveHybridSolveTime == 0.0
    # rec_enabled False zeroes the recenter term too, exactly as before this change
    assert lp.laneCenterBlendCurvature == planner.lane_center_blend.curvature


def test_lane_mode_never_runs_the_hybrid_mpc():
  planner, _ = build_planner(lra=1)
  # the speed guard needs recovery_frames + 1 usable model frames before lane mode can engage
  drive(planner, 25, lane_speed=1.0, lat=True)
  assert planner.lanelines_active, 'the lane-mode branch must be exercised'
  runs_before = hybrid_mpc(planner).runs
  published = drive(planner, 20, lane_speed=1.0, lat=True, start_t=1.0 + 25 * DT)
  assert planner.lanelines_active
  assert hybrid_mpc(planner).runs == runs_before, 'lane mode is not a hybrid case'
  for lp in published:
    assert lp.curveHybridCurvature == 0.0
    assert lp.curveHybridActive is False


# --- 2. the published sum on a laneless curve ------------------------------------------------

def test_laneless_curve_adds_the_hybrid_term_once():
  planner, _ = build_planner(lra=1)
  published = drive(planner, 90, lane_speed=0.0, lat=True, y_off=0.5)
  assert hybrid_mpc(planner).runs > 10, 'the prewarm gate must open on a 0.012 1/m curve'
  assert math.isfinite(planner.hybrid_klag), 'the hybrid MPC must have produced a k_lane'

  blend = planner.lane_center_blend
  base = float(blend.curvature) + float(blend.recenter_curvature)
  hybrid = float(planner.hybrid_curvature)
  assert abs(hybrid) > 1e-6, 'a centring frame must carry a hybrid term'
  last = published[-1]
  assert last.curveHybridCurvature == hybrid
  # no double count: the published sum is base + hybrid exactly once
  assert last.laneCenterBlendCurvature == base + hybrid
  assert last.laneCenterBlendCurvature - hybrid == base
  assert last.curveHybridActive is True


def test_hybrid_term_is_zero_before_the_gate_opens():
  planner, _ = build_planner(lra=1)
  # straight road and no model curvature: a_road and curve_a stay under their prewarm bars
  published = drive(planner, 30, lane_speed=0.0, lat=True, y_off=0.5, shape=0.0, dc=0.0)
  assert hybrid_mpc(planner).runs == 0, 'no road and no model curvature must not prewarm'
  assert math.isnan(planner.hybrid_klag), 'nothing solved means no k_lane'
  for lp in published:
    assert lp.curveHybridCurvature == 0.0
    assert lp.curveHybridActive is False
    assert lp.laneCenterBlendCurvature == (planner.lane_center_blend.curvature +
                                          planner.lane_center_blend.recenter_curvature)


def test_lat_inactive_never_runs_the_hybrid_mpc():
  planner, _ = build_planner(lra=1)
  published = drive(planner, 60, lane_speed=0.0, lat=False, y_off=0.5)
  assert hybrid_mpc(planner).runs == 0, 'latActive False is not a hybrid case'
  for lp in published:
    assert lp.curveHybridCurvature == 0.0


def test_below_run_speed_never_runs_the_hybrid_mpc():
  planner, _ = build_planner(lra=1)
  published = drive(planner, 60, lane_speed=0.0, lat=True, y_off=0.5,
                    speed=RUN_V_MIN - 2.0)
  assert hybrid_mpc(planner).runs == 0, 'RUN_V_MIN is the hybrid run gate'
  for lp in published:
    assert lp.curveHybridCurvature == 0.0


# --- 3. the module-global cost trap ----------------------------------------------------------

def test_hybrid_run_leaves_the_module_costs_untouched():
  # P0 rewrites these globals from params inside update() every 10 frames (lateral_planner.py:92),
  # so the oracle is the same frame sequence with the hybrid switched off by LaneRecenterAssist 0:
  # identical P0 work, zero hybrid runs. Equal globals there means the hybrid path wrote neither.
  planner, namespace = build_planner(lra=1)
  quiet, quiet_ns = build_planner(lra=0)
  frames = 60
  for i in range(frames):
    for p in (planner, quiet):
      p.update(inputs(1.0 + i * DT, lane_speed=0.0, lat=True, y_off=0.5),
               SimpleNamespace(atc_active=False))
  mpc = hybrid_mpc(planner)
  assert mpc.runs > 10, 'the run must have happened for this test to mean anything'
  assert hybrid_mpc(quiet).runs == 0
  assert module_costs(namespace) == module_costs(quiet_ns), (
    'the hybrid path wrote a module-level cost global')
  # and the globals are exactly P0's own params-derived values
  assert module_costs(namespace) == pytest.approx((1.2, 0.04, 7.0))

  # V4's own weights reached the hybrid solver: accel 0, the store's path/motion/jerk/srate
  path_w, heading_w, accel_w, jerk_w, srate_w = mpc.weights
  assert accel_w == 0.0
  assert path_w == pytest.approx(2.0)      # LatMpcPathCost 200 * 0.01
  assert heading_w == pytest.approx(0.07)  # LatMpcMotionCost 7 * 0.01
  assert jerk_w == pytest.approx(0.04)     # LatMpcJerkCost 4 * 0.01
  assert srate_w == pytest.approx(7.0)     # LatMpcSteeringRateCost, not scaled


# --- 4. pre-warm skip then run resets the hybrid MPC -----------------------------------------

def test_skip_then_run_resets_the_hybrid_mpc():
  planner, _ = build_planner(lra=1)
  # straight road, no model curvature: no runs at all
  drive(planner, 8, lane_speed=0.0, lat=True, y_off=0.5, shape=0.0, dc=0.0)
  mpc = hybrid_mpc(planner)
  assert mpc.runs == 0
  assert planner.hybrid_ran is False

  # same planner, now a curve: the first run must reset from the zero state, exactly as P0 does
  resets_before = len(mpc.resets)
  drive(planner, 2, lane_speed=0.0, lat=True, y_off=0.5, start_t=1.0 + 8 * DT)
  assert mpc.runs >= 1
  assert len(mpc.resets) == resets_before + 1, 'a run after a skip resets the hybrid MPC'
  seed = max(SPEED, 1.0) * 0.02
  assert np.array_equal(mpc.resets[-1], np.zeros(4)), (
    'the reset is the zero state, matching P0 reset_mpc(np.zeros(4))')
  assert mpc.x0_last[3] == pytest.approx(seed), 'every run seeds x0[3] with the same expression'

  # a run that follows a run does not reset
  resets_mid = len(mpc.resets)
  drive(planner, 3, lane_speed=0.0, lat=True, y_off=0.5, start_t=1.0 + 10 * DT)
  assert len(mpc.resets) == resets_mid, 'consecutive runs keep the solver state'


# --- 4b. the path offset lands after the lane-mode yaw, as P0 does -----------------------------

def test_path_offset_added_after_lane_yaw():
  planner, namespace = build_planner(lra=1, path_offset=50)   # 50 * 0.01 = 0.5 m
  yaw_fn = load_yaw_from_path()
  namespace['yaw_from_path_no_scipy'] = yaw_fn   # the class bodies resolve it from this dict
  hybrid = planner.hybrid_lane
  assert planner.pathOffset == pytest.approx(0.5)

  # open the hybrid's own lane-mode gate: recovery_frames + 1 usable model frames
  for _ in range(int(1 / DT) + 5):
    hybrid.update_guard(SPEED, SPEED, SPEED, True)
  assert hybrid.laneless_only is False

  md = inputs(1.0)['modelV2']
  md.position.y = CURVE_A * md.position.x ** 2   # a real dy/dx for the yaw chain
  path_before = np.column_stack([np.array(md.position.x), np.array(md.position.y),
                                 np.array(md.position.z)])
  velocity_xyz = np.column_stack([md.velocity.x, md.velocity.y, md.velocity.z])
  v_plan = np.clip(np.linalg.norm(velocity_xyz, axis=1), 1.0, np.inf)

  triple = hybrid.step(md, SPEED, SPEED * 3.6, 0.01, 0.0, SPEED * 0.02,
                       (2.0, 0.07, 0.04, 7.0), False, planner.pathOffset)
  assert triple is not None
  assert hybrid.lanelines_active, 'the yaw branch must be the one exercised'

  expected_yaw, _ = yaw_fn(path_before, v_plan, smooth_window=5, clip_rate=2.0,
                           align_first_yaw=None)
  assert np.array_equal(hybrid.plan_yaw, expected_yaw), (
    'the yaw must come from the path before the offset, as P0 does')
  assert np.array_equal(hybrid.y_pts, path_before[:len(hybrid.y_pts), 1] + 0.5), (
    'y_pts must carry the offset')
  assert np.array_equal(hybrid.path_xyz[:, 1], path_before[:, 1] + 0.5)


# --- 5. the new capnp fields -----------------------------------------------------------------

def test_new_lateral_plan_fields_exist_and_are_populated():
  names = set(log.LateralPlan.schema.fields)   # a name -> schema mapping, not a list
  for name in ('curveHybridCurvature', 'curveHybridActive', 'curveHybridGuards',
               'curveHybridSolveTime'):
    assert name in names, f'{name} missing from LateralPlan'

  planner, _ = build_planner(lra=1)
  published = drive(planner, 40, lane_speed=0.0, lat=True, y_off=0.5)
  last = published[-1]
  assert last.curveHybridCurvature == pytest.approx(planner.hybrid_curvature)
  assert isinstance(last.curveHybridActive, bool)
  assert isinstance(last.curveHybridGuards, int)
  assert last.curveHybridSolveTime >= 0.0


# --- 6. exception containment (F2) -----------------------------------------------------------

def _p0_fields(lp):
  """Every published LateralPlan field except the five the hybrid family owns."""
  return {k: v for k, v in vars(lp).items()
          if k not in ('laneCenterBlendCurvature', 'curveHybridCurvature', 'curveHybridActive',
                       'curveHybridGuards', 'curveHybridSolveTime',
                       # the message stub's own members: not something publish() writes
                       'position', 'solverState')}


def test_hybrid_exception_is_contained_and_p0_is_unchanged(monkeypatch):
  planner, namespace = build_planner(lra=1)
  # warm the chain until the run gate is open, on a curve
  drive(planner, 30, lane_speed=0.0, lat=True, y_off=0.5)
  assert hybrid_mpc(planner).runs > 0, 'the hybrid must be running before the fault is injected'

  t_fault = 1.0 + 30 * DT

  def _boom(*args, **kwargs):
    raise RuntimeError('injected hybrid fault')

  monkeypatch.setattr(namespace['HybridLanePath'], 'step', _boom)
  calls = []
  for i in range(6):
    sm = inputs(t_fault + i * DT, lane_speed=0.0, lat=True, y_off=0.5)
    planner.update(sm, SimpleNamespace(atc_active=False))   # must not raise
    pm = SimpleNamespace(sent=[], send=lambda name, msg: pm.sent.append((name, msg)))
    planner.publish(sm, pm, SimpleNamespace())              # must not raise
    calls.append(pm.sent[-1][1].lateralPlan)

  assert planner.hybrid_errors == 6, 'every faulted frame must be counted'
  for lp in calls:
    assert lp.curveHybridCurvature == 0.0
    assert lp.curveHybridActive is False
    assert lp.curveHybridSolveTime == 0.0
    # the contained frame publishes P0 exactly: base blend + the (now zero) recenter term
    assert lp.laneCenterBlendCurvature == (planner.lane_center_blend.curvature +
                                           planner.lane_center_blend.recenter_curvature)

  # oracle: a second LRA-1 planner with the hybrid made inert exactly as the except branch leaves
  # it, so P0's own fields (the recenter pair included) must come out identical.
  quiet, quiet_ns = build_planner(lra=1)
  for i in range(30):
    quiet.update(inputs(1.0 + i * DT, lane_speed=0.0, lat=True, y_off=0.5),
                 SimpleNamespace(atc_active=False))
  assert hybrid_mpc(quiet).runs > 0, 'the oracle must warm up on the same running chain'

  def _inert(self, *args, **kwargs):
    self._reset_curve_hybrid()

  monkeypatch.setattr(quiet_ns['LateralPlanner'], '_update_curve_hybrid', _inert)
  oracle = []
  quiet_pm = SimpleNamespace(sent=[], send=lambda name, msg: quiet_pm.sent.append((name, msg)))
  for i in range(6):
    sm = inputs(t_fault + i * DT, lane_speed=0.0, lat=True, y_off=0.5)
    quiet.update(sm, SimpleNamespace(atc_active=False))
    quiet_pm.sent.clear()
    quiet.publish(sm, quiet_pm, SimpleNamespace())
    oracle.append(quiet_pm.sent[-1][1].lateralPlan)

  assert quiet.hybrid_errors == 0
  for lp, ref in zip(calls, oracle):
    assert _p0_fields(lp) == _p0_fields(ref), 'a hybrid fault changed a P0 published field'
