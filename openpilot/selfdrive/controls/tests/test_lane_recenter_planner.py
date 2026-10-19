"""Planner-level test: LateralPlanner.update() + publish() with LaneRecenterAssist.

The unit tests only exercised LaneCenterBlend directly, which is how a NameError
(`rec_enabled` used in publish() but bound in update()) survived CP-R1.  This
test compiles the real production class, runs update+publish for several frames,
and checks the published curvature contract:
  - laneless, assist on:  laneCenterBlendCurvature == blend + recenter
  - laneline or assist 0: laneCenterBlendCurvature == blend

Set LATERAL_PLANNER_SOURCE to a path to compile a different revision of
lateral_planner.py (used to show this test failing on 077c3c314).
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.controls.lib.lane_model_speed import LaneModelSpeedGuard

DEFAULT_SOURCE = Path(__file__).parents[1] / 'lib/lateral_planner.py'


class StubMpc:
  def reset(self, x0):
    self.x_sol = np.zeros((33, 4))
    self.u_sol = np.zeros((32, 1))
    self.solution_status = 0
    self.cost = 0.0
    self.solve_time = 0.0

  def set_weights(self, *args):
    pass

  def run(self, x0, p, y, heading, yaw_rate):
    self.input_speed = p[:, 0].copy()


class StubLaneCenterBlend:
  """Exposes the fields publish() reads; counts update() calls."""
  def __init__(self, table=None):
    self.table = np.zeros(7)
    self.curvature = 0.01
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


class StubLanePlanner:
  def __init__(self):
    self.debugText = ''
    self.l_prob = 0.9
    self.r_prob = 0.9
    self.lane_width = 3.5
    self.lane_width_left = 3.5
    self.lane_width_right = 3.5
    self.offset_total = 0.0
    self.ll_x = np.linspace(0, 300, 33)
    self.lll_y = np.full(33, -1.75)
    self.rll_y = np.full(33, 1.75)
    self.lanefull_mode = False

  def parse_model(self, md):
    pass

  def get_d_path(self, cs, speed, t, path, curve_speed):
    return path, self.lanefull_mode


class Message:
  def __init__(self, which):
    if which == 'lateralPlan':
      self.lateralPlan = SimpleNamespace(
        position=SimpleNamespace(x=[], y=[], z=[]),
        solverState=None,
      )
    self.valid = True


def make_planner(param_value=0, source=None):
  src = Path(source or os.environ.get('LATERAL_PLANNER_SOURCE', DEFAULT_SOURCE))
  tree = ast.parse(src.read_text(encoding='utf8'))
  params = SimpleNamespace(get_int=lambda _: param_value, get_float=lambda _: 1.0,
                           get=lambda _: None, put_nonblocking=lambda *_: None)
  namespace = {
    'np': np,
    'DT_MDL': 0.05,
    'TRAJECTORY_SIZE': 33,
    'MIN_SPEED': 1.0,
    'LAT_MPC_N': 32,
    'CONTROL_N': 32,
    'Params': lambda: params,
    'LateralMpc': StubMpc,
    'LanePlanner': StubLanePlanner,
    'LaneModelSpeedGuard': LaneModelSpeedGuard,
    'LaneCenterBlend': StubLaneCenterBlend,
    'dump_table': lambda table: '[]',
    'load_table': lambda text: None,
    'time': SimpleNamespace(monotonic=lambda: 0),
    'log': SimpleNamespace(Desire=SimpleNamespace(none=0), LaneChangeState=SimpleNamespace(off=0),
                           LateralPlan=SimpleNamespace(SolverState=SimpleNamespace(new_message=Message))),
    'messaging': SimpleNamespace(new_message=Message),
    'yaw_from_path_no_scipy': lambda *a, **k: (np.zeros(33), np.zeros(33)),
  }
  exec(
    compile(ast.Module(body=[n for n in tree.body
                             if isinstance(n, ast.ClassDef) and n.name == 'LateralPlanner'],
                       type_ignores=[]), str(src), 'exec'),
    namespace,
  )
  cp = SimpleNamespace(wheelbase=2.7, centerToFront=1.1, mass=1800, tireStiffnessRear=100000)
  return namespace['LateralPlanner'](cp)


class _Msg(dict):
  """Subscriptable message dict that the planner also reads as attributes."""
  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError as e:
      raise AttributeError(name) from e


def inputs(speed=30.0, start=30.0, end=30.0, lane_speed=1):
  model = SimpleNamespace(
    position=SimpleNamespace(x=np.linspace(0, 300, 33), y=np.zeros(33), z=np.zeros(33),
                            t=np.linspace(0, 10, 33)),
    orientation=SimpleNamespace(x=np.zeros(33), z=np.zeros(33)),
    orientationRate=SimpleNamespace(z=np.zeros(33)),
    velocity=SimpleNamespace(x=np.linspace(start, end, 33), y=np.zeros(33), z=np.zeros(33)),
    acceleration=SimpleNamespace(x=np.zeros(33)),
    meta=SimpleNamespace(desire=0, laneChangeState=0, laneChangeDirection=0,
                         laneWidthLeft=3.5, laneWidthRight=3.5),
  )
  sm = {
    'modelV2': model,
    'carState': SimpleNamespace(vEgo=speed, useLaneLineSpeed=lane_speed, steeringPressed=False),
    'controlsState': SimpleNamespace(curvature=0.0),
    'carrotMan': SimpleNamespace(vTurnSpeed=200),
    'logMonoTime': {'modelV2': 123},
  }
  sm['all_checks'] = lambda service_list: True
  return _Msg(sm)


class RecordingPm:
  def __init__(self):
    self.sent = []

  def send(self, name, msg):
    self.sent.append((name, msg))


def run_frames(planner, frames, lane_speed=1):
  pm = RecordingPm()
  published = []
  for _ in range(frames):
    sm = inputs(lane_speed=lane_speed)
    planner.update(sm, SimpleNamespace(atc_active=False))
    planner.publish(sm, pm, SimpleNamespace())
    name, msg = pm.sent[-1]
    assert name == 'lateralPlan'
    published.append(msg)
  return published


def _blend(planner):
  return planner.lane_center_blend


def test_assist_on_laneless_publishes_blend_plus_recenter():
  planner = make_planner(param_value=1)
  for _ in range(25):
    planner.update(inputs(lane_speed=0), SimpleNamespace(atc_active=False))
  assert not planner.lanelines_active, 'the laneless branch is the assist case'
  published = run_frames(planner, 3, lane_speed=0)
  for msg in published:
    lp = msg.lateralPlan
    blend = _blend(planner)
    assert lp.laneRecenterCurvature == blend.recenter_curvature
    assert lp.laneCenterBlendCurvature == blend.curvature + blend.recenter_curvature
    assert lp.laneRecenterActive is True


def test_assist_off_laneless_publishes_blend_only():
  planner = make_planner(param_value=0)
  for _ in range(25):
    planner.update(inputs(lane_speed=0), SimpleNamespace(atc_active=False))
  assert not planner.lanelines_active
  published = run_frames(planner, 3, lane_speed=0)
  for msg in published:
    lp = msg.lateralPlan
    blend = _blend(planner)
    assert lp.laneRecenterCurvature == 0.0
    assert lp.laneCenterBlendCurvature == blend.curvature
    assert lp.laneRecenterActive is False


def test_assist_on_laneline_publishes_blend_only():
  planner = make_planner(param_value=1)
  for _ in range(30):
    planner.update(inputs(lane_speed=1), SimpleNamespace(atc_active=False))
  assert planner.lanelines_active, 'the laneline branch must be exercised'
  published = run_frames(planner, 3, lane_speed=1)
  for msg in published:
    lp = msg.lateralPlan
    blend = _blend(planner)
    assert lp.laneRecenterCurvature == 0.0
    assert lp.laneCenterBlendCurvature == blend.curvature
    assert lp.laneRecenterActive is False
