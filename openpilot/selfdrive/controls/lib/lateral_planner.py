import time
import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc
from openpilot.selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import N as LAT_MPC_N
from openpilot.selfdrive.controls.lib.drive_helpers import (CONTROL_N, MIN_SPEED,
                                                             get_lag_adjusted_curvature)
# from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
import openpilot.cereal.messaging as messaging
from openpilot.cereal import log

from openpilot.common.params import Params
#from openpilot.selfdrive.controls.lib.lane_planner import LanePlanner
from openpilot.selfdrive.controls.lib.lane_planner_2 import LanePlanner
from openpilot.selfdrive.controls.lib.lane_model_speed import LaneModelSpeedGuard
from openpilot.selfdrive.controls.lib.lane_center_blend import (LaneCenterBlend, dump_table, load_table,
                                                                FIT_HORIZON_S, FIT_MIN_M)
from openpilot.selfdrive.controls.lib.curve_lane_hybrid import (CurveLaneHybrid, CausalMedian, HybridFrame,
                                                                RUN_V_MIN, PREWARM_ROAD_A, PREWARM_CURVE_A)
from collections import deque

TRAJECTORY_SIZE = 33
CAMERA_OFFSET = 0.04


PATH_COST = 1.0
LATERAL_MOTION_COST = 0.11
LATERAL_ACCEL_COST = 0.0
LATERAL_JERK_COST = 0.04
# Extreme steering rate is unpleasant, even
# when it does not cause bad jerk.
# TODO this cost should be lowered when low
# speed lateral control is stable on all cars
STEERING_RATE_COST = 700.0

# Curve-lane hybrid switches (SPEC_2B section 2). Not params: they exist so the Gate B harness can
# run the hybrid MPC every frame (continuous / always) without a second code path.
HYBRID_CONTINUOUS = False
HYBRID_RUN_ALWAYS = False


class _HybridParams:
  """The planner's params with the shadow V4 overrides: AdjustLaneOffset 0, LatMpcInputOffset 0.

  LanePlanner keeps self.params for its whole life and reads those two keys from it, so the
  overrides have to live on an object rather than on the shared store.
  """
  _INT_OVERRIDES = {'AdjustLaneOffset': 0}
  _FLOAT_OVERRIDES = {'LatMpcInputOffset': 0.0}

  def __init__(self, base):
    self._base = base

  def get(self, key, default=None):
    if key in self._INT_OVERRIDES:
      return str(self._INT_OVERRIDES[key])
    if key in self._FLOAT_OVERRIDES:
      return str(self._FLOAT_OVERRIDES[key])
    return self._base.get(key, default)

  def get_int(self, key, default=0):
    if key in self._INT_OVERRIDES:
      return self._INT_OVERRIDES[key]
    return self._base.get_int(key, default)

  def get_float(self, key, default=0.0):
    if key in self._FLOAT_OVERRIDES:
      return self._FLOAT_OVERRIDES[key]
    return self._base.get_float(key, default)

  def get_bool(self, key, default=False):
    return self._base.get_bool(key, default)

  def put(self, *a, **k):
    return self._base.put(*a, **k)

  def put_nonblocking(self, *a, **k):
    return self._base.put_nonblocking(*a, **k)


class HybridLanePath:
  """The shadow V4 planner: its own LanePlanner + LateralMpc, per-instance MPC weights.

  Reproduces W-SHD3/shadow_variants.py's V4 frame for frame on the published triple
  (W-SHD1/shadow_replay.py:373-389): lane mode forced (useLaneLineSpeed 1.0), AdjustLaneOffset 0,
  LatMpcInputOffset 0, LatMpcAccelCost 0, the route's path/motion/jerk/steering-rate costs,
  warm-up off (d_prob_count = int(1/DT_MDL) before every update) and x0[3] seeded each frame with
  max(carState.vEgo, MIN_SPEED) * controlsState.desiredCurvature.

  It never touches the module-level LATERAL_ACCEL_COST / LATERAL_JERK_COST / STEERING_RATE_COST:
  set_weights takes this instance's own copies, so a hybrid run cannot leak into P0's weights.
  The two lane-offset params sit on _HybridParams, its own view of the store, because LanePlanner
  keeps self.params for its whole life and reads them every frame.
  """

  def __init__(self, CP, params):
    self.CP = CP
    self.params = _HybridParams(params)
    self.factor1 = CP.wheelbase - CP.centerToFront
    self.factor2 = (CP.centerToFront * CP.mass) / (CP.wheelbase * CP.tireStiffnessRear)

    self.LP = LanePlanner()
    self.LP.params = self.params
    self.speed_guard = LaneModelSpeedGuard(recovery_frames=int(1 / DT_MDL))
    self.mpc = LateralMpc()
    self.x0 = np.zeros(4)
    self.mpc.reset(x0=self.x0)

    self.v_plan = np.zeros((TRAJECTORY_SIZE,))
    self.t_idxs = np.arange(TRAJECTORY_SIZE)
    self.path_xyz = np.zeros((TRAJECTORY_SIZE, 3))
    self.plan_yaw = np.zeros((TRAJECTORY_SIZE,))
    self.plan_yaw_rate = np.zeros((TRAJECTORY_SIZE,))
    self.y_pts = np.zeros((LAT_MPC_N + 1,))

    self.laneless_only = True
    self.useLaneLineMode = False
    self.lanelines_active = False

    self.ran = False          # did the MPC run on the previous frame
    self.status = -1
    self.nan = True
    self.solve_ms = 0.0

  def reset_mpc(self):
    """Drop the hybrid MPC state between runs:
    a zero-state reset exactly as P0's own (x0 = 0); step() seeds x0[3] before every solve.
    """
    self.x0 = np.zeros(4)
    self.mpc.reset(x0=self.x0)
    self.status = -1
    self.nan = True
    self.solve_ms = 0.0

  def update_guard(self, v_ego, model_v0, model_vend, model_active):
    """Advance the lane-model speed guard every frame, run or skip: V4's guard saw every frame."""
    if model_active:
      self.laneless_only = not self.speed_guard.update(v_ego, model_v0, model_vend)
    else:
      self.speed_guard.update(float('nan'), float('nan'), float('nan'))
      self.laneless_only = True
    return self.laneless_only

  def step(self, md, v_ego, speed_kph, measured_curvature, curve_speed, seed, weights,
           atc_active, path_offset):
    """Run one V4 update and return the published (curvatures, psis, distances) triple, or None.

    `weights` is (path, motion, jerk, steering_rate); the accel cost is V4's 0. update_guard()
    must have run for this frame already, so the guard's history covers skipped frames too.
    """
    self.path_xyz = np.column_stack([np.array(md.position.x), np.array(md.position.y),
                                     np.array(md.position.z)])
    self.t_idxs = np.array(md.position.t)
    self.plan_yaw = np.array(md.orientation.z)
    self.plan_yaw_rate = np.array(md.orientationRate.z)
    velocity_xyz = np.column_stack([md.velocity.x, md.velocity.y, md.velocity.z])
    self.v_plan = np.clip(np.linalg.norm(velocity_xyz, axis=1), MIN_SPEED, np.inf)

    # The shadow fed carState.useLaneLineSpeed = 1.0, so V4's apply gate threshold is 1.0 kph.
    if self.laneless_only:
      self.useLaneLineMode = False
    elif speed_kph >= 1.0 + 2:
      self.useLaneLineMode = True
    elif speed_kph < 1.0 - 2:
      self.useLaneLineMode = False

    self.LP.parse_model(md)
    self.LP.lane_change_multiplier = 0.0 if (md.meta.desire != log.Desire.none or atc_active) else 1.0
    self.LP.lanefull_mode = self.useLaneLineMode
    self.LP.lane_width_left = md.meta.laneWidthLeft
    self.LP.lane_width_right = md.meta.laneWidthRight
    self.LP.curvature = measured_curvature
    # warm-up off: the update's own `d_prob_count + 1 if d_prob > 0.3 else 0` then lands on
    # 21 > int(1/DT_MDL) = 20 on a qualifying frame, so lane mode is active from frame one.
    self.LP.d_prob_count = int(1 / DT_MDL)
    self.path_xyz, self.lanelines_active = self.LP.get_d_path(
      None, v_ego, self.t_idxs, self.path_xyz, curve_speed)
    if self.lanelines_active:
      self.plan_yaw, self.plan_yaw_rate = yaw_from_path_no_scipy(
        self.path_xyz, self.v_plan, smooth_window=5, clip_rate=2.0, align_first_yaw=None)
    self.path_xyz[:, 1] += path_offset

    self.y_pts = self.path_xyz[:LAT_MPC_N + 1, 1]
    heading_pts = self.plan_yaw[:LAT_MPC_N + 1]
    yaw_rate_pts = self.plan_yaw_rate[:LAT_MPC_N + 1]
    if not (len(self.v_plan) == TRAJECTORY_SIZE and len(self.y_pts) == LAT_MPC_N + 1
            and len(heading_pts) == LAT_MPC_N + 1 and len(yaw_rate_pts) == LAT_MPC_N + 1):
      # P0 would assert here; the hybrid path just reports that it did not run.
      self.ran = False
      return None

    self.x0[3] = seed
    lateral_factor = np.clip(self.factor1 - (self.factor2 * self.v_plan ** 2), 0.0, np.inf)
    p = np.column_stack([self.v_plan, lateral_factor])

    path_w, motion_w, jerk_w, srate_w = weights
    self.mpc.set_weights(path_w, motion_w, 0.0, jerk_w, srate_w)
    self.mpc.run(self.x0, p, self.y_pts, heading_pts, yaw_rate_pts)

    self.status = int(self.mpc.solution_status)
    self.nan = bool(np.isnan(self.mpc.x_sol[:, 3]).any())
    self.solve_ms = float(self.mpc.solve_time) * 1000.0
    self.ran = True

    # carry the state exactly as P0 does; the next frame re-seeds x0[3] before its solve
    self.x0[3] = np.interp(DT_MDL, self.t_idxs[:LAT_MPC_N + 1], self.mpc.x_sol[:, 3])

    x_sol = self.mpc.x_sol
    if len(self.v_plan) == TRAJECTORY_SIZE:
      curvatures = x_sol[:CONTROL_N, 3] / np.maximum(self.v_plan[:CONTROL_N], 6.0)
    else:
      curvatures = x_sol[:CONTROL_N, 3] / v_ego
    return (list(curvatures), list(x_sol[:CONTROL_N, 2]), list(x_sol[:CONTROL_N, 0]))


class LateralPlanner:
  def __init__(self, CP, debug=False):
    #self.DH = DesireHelper()

    # Vehicle model parameters used to calculate lateral movement of car
    self.factor1 = CP.wheelbase - CP.centerToFront
    self.factor2 = (CP.centerToFront * CP.mass) / (CP.wheelbase * CP.tireStiffnessRear)
    self.last_cloudlog_t = 0
    self.solution_invalid_cnt = 0

    self.path_xyz = np.zeros((TRAJECTORY_SIZE, 3))
    self.velocity_xyz = np.zeros((TRAJECTORY_SIZE, 3))
    self.v_plan = np.zeros((TRAJECTORY_SIZE,))
    self.x_sol = np.zeros((TRAJECTORY_SIZE, 4), dtype=np.float32)
    self.v_ego = MIN_SPEED
    self.l_lane_change_prob = 0.0
    self.r_lane_change_prob = 0.0

    self.debug_mode = debug

    self.params = Params()

    self.latDebugText = ""
    # lane_mode
    self.LP = LanePlanner()
    self.readParams = 0
    self.lanelines_active = False
    self.lanelines_active_tmp = False

    self.useLaneLineSpeedApply = self.params.get_int("UseLaneLineSpeed")
    self.pathOffset = float(self.params.get_int("PathOffset")) * 0.01
    self.useLaneLineMode = False
    self.plan_a = np.zeros((TRAJECTORY_SIZE, ))
    self.plan_yaw = np.zeros((TRAJECTORY_SIZE,))
    self.plan_yaw_rate = np.zeros((TRAJECTORY_SIZE,))
    self.t_idxs = np.arange(TRAJECTORY_SIZE)
    self.y_pts = np.zeros((TRAJECTORY_SIZE,))
    self.d_path_w_lines_xyz = np.zeros((TRAJECTORY_SIZE, 3))

    self.lat_mpc = LateralMpc()
    self.reset_mpc(np.zeros(4))
    self.curve_speed = 0
    self.lane_model_speed_guard = LaneModelSpeedGuard(recovery_frames=int(1 / DT_MDL))
    self.laneless_only = True

    self.lane_center_blend_enabled = self.params.get_int("LaneCenterBlend") > 0
    self.lane_center_blend = LaneCenterBlend(load_table(self.params.get("LaneCenterBlendMap")))
    self.lane_center_blend_saved = dump_table(self.lane_center_blend.table)
    self.lane_center_blend_save_frame = 0
    self.lane_recenter_assist_enabled = self.params.get_int("LaneRecenterAssist") > 0

    # Curve-lane hybrid (SPEC_2B section 2). Its own LanePlanner + LateralMpc + x0: nothing here is
    # shared with P0's plan, and the module-level cost globals are never assigned from this path.
    self.CP = CP
    self.hybrid_lane = HybridLanePath(CP, self.params)
    self.curve_hybrid = CurveLaneHybrid()
    self.cs_filter = CausalMedian()
    self.hybrid_frame = HybridFrame()
    self.hybrid_klag = float('nan')
    self.hybrid_status = -1
    self.hybrid_nan = True
    self.hybrid_ran = False
    self.hybrid_solve_ms = 0.0
    self.hybrid_curvature = 0.0
    self.hybrid_lagdelay = float('nan')
    self.hybrid_prev = False
    self.hybrid_errors = 0
    # V4's own cost set: path/motion from the store, accel 0 (never the module globals).
    self.hybrid_path_cost = self.params.get_float("LatMpcPathCost") * 0.01
    self.hybrid_motion_cost = self.params.get_float("LatMpcMotionCost") * 0.01
    self.hybrid_jerk_cost = self.params.get_float("LatMpcJerkCost") * 0.01
    self.hybrid_srate_cost = self.params.get_float("LatMpcSteeringRateCost")
    self.hybrid_steer_delay = self.params.get_float("SteerActuatorDelay") * 0.01
    self.hybrid_smooth_sec = self.params.get_float("LatSmoothSec") * 0.01

  def reset_mpc(self, x0=None):
    if x0 is None:
      x0 = np.zeros(4)
    self.x0 = x0
    self.lat_mpc.reset(x0=self.x0)

  def _hybrid_lagdelay(self, sm):
    """The shadow's lag rule (shadow_variants.py:586-590) plus SPEC_2B's CP fallback, in seconds.

    SteerActuatorDelay, or liveDelay.lateralDelay when the param is 0 and liveDelay has arrived, or
    the car's own CP.steerActuatorDelay when it is 0 and liveDelay has not arrived yet.
    """
    steer_delay = self.hybrid_steer_delay
    if steer_delay == 0.0:
      if bool(getattr(sm, 'seen', {}).get('liveDelay', False)):
        steer_delay = float(sm['liveDelay'].lateralDelay)
      else:
        steer_delay = float(self.CP.steerActuatorDelay)
    return steer_delay + self.hybrid_smooth_sec

  @staticmethod
  def _road_curvature(ll_x, centre, v_ego):
    """2 * quadratic fit coefficient over the fitted horizon (lane_center_blend.py:261-262).

    Fewer than three points in the window is 0.0, exactly as LaneCenterBlend and the offline
    t1_plant.kappa_road_of report it (t1_plant.py:70), not a missing measurement.
    """
    fit = ll_x <= max(FIT_HORIZON_S * v_ego, FIT_MIN_M)
    if np.count_nonzero(fit) < 3:
      return 0.0
    if (not np.all(np.isfinite(ll_x)) or not np.all(np.isfinite(centre))
            or float(np.ptp(ll_x[fit])) <= 0.0):
      # a repeated / non-finite x cannot carry a fit: the crash guard, not the missing-fit rule
      return float('nan')
    return float(2.0 * np.polyfit(ll_x[fit], centre[fit], 2)[0])

  def _hybrid_frame(self, sm, md, lane_change_active, model_path_x, model_path_y, v_ego_car):
    """This frame's HybridFrame: pure reads of modelV2, P0's LanePlanner lines and P0's blend.

    P0's LanePlanner has already parsed this modelV2 frame (update() runs parse_model, then
    get_d_path, then this), so ll_x / lll_y / rll_y / l_prob / r_prob are this frame's values and
    nothing here mutates planner state.
    """
    f = HybridFrame()
    lp = self.LP

    f.t = float(sm.logMonoTime['modelV2']) * 1e-9
    f.v = float(sm['carState'].vEgo)
    # carControl is in plannerd's SubMaster; before its first message there is no latActive yet
    f.lat = bool(sm['carControl'].latActive) if getattr(sm, 'seen', {}).get('carControl') else False
    f.sp = bool(sm['carState'].steeringPressed)
    f.lane_change = bool(lane_change_active)
    f.lane_mode = bool(self.lanelines_active or self.useLaneLineMode)

    ll_x = np.asarray(lp.ll_x, dtype=float)
    lll_y = np.asarray(lp.lll_y, dtype=float)
    rll_y = np.asarray(lp.rll_y, dtype=float)
    centre = 0.5 * (lll_y + rll_y)
    width_pts = rll_y - lll_y
    p_t = np.asarray(md.position.t, dtype=float)
    p_x = np.asarray(model_path_x, dtype=float)
    p_y = np.asarray(model_path_y, dtype=float)

    # c: model-path offset from the lane centre at the car, 0 s ahead (t1_plant.py:75-77 r_at)
    c = float(np.interp(0.0, p_t, p_y) - np.interp(np.interp(0.0, p_t, p_x), ll_x, centre))
    # a frame the gate dropped restarts the median window, so a jump cannot be smoothed across it
    f.c_s = float(self.cs_filter.update(f.t, c, reset=self.curve_hybrid.drop))
    f.c0 = float(np.interp(0.0, ll_x, centre))
    f.w0 = float(np.interp(0.0, ll_x, width_pts))
    f.w15 = float(np.interp(15.0, ll_x, width_pts))
    f.w30 = float(np.interp(30.0, ll_x, width_pts))
    f.lprob = float(lp.l_prob)
    f.rprob = float(lp.r_prob)
    if len(md.laneLineStds) > 2:
      f.std1 = float(md.laneLineStds[1])
      f.std2 = float(md.laneLineStds[2])
    # a jump is only judged against a previous frame of the same run (SPEC_2B section 1)
    f.spike_judge = bool(self.hybrid_prev)
    self.hybrid_prev = True

    f.w0_g = f.w0
    f.w45 = float(np.interp(45.0, ll_x, rll_y) - np.interp(45.0, ll_x, lll_y))
    f.lwl = float(md.meta.laneWidthLeft)
    f.lwr = float(md.meta.laneWidthRight)
    f.ymod0 = float(np.interp(0.0, p_t, np.asarray(md.position.y, dtype=float)))
    f.ymod10 = float(np.interp(1.0, p_t, np.asarray(md.position.y, dtype=float)))
    f.dc = float(md.action.desiredCurvature)
    f.kappa = self._road_curvature(ll_x, centre, v_ego_car)

    rec_enabled = (self.lane_recenter_assist_enabled and
                   not self.lanelines_active and not self.useLaneLineMode)
    recenter_curvature = float(self.lane_center_blend.recenter_curvature) if rec_enabled else 0.0
    f.k_model = f.dc + float(self.lane_center_blend.curvature) + recenter_curvature
    return f

  def _reset_curve_hybrid(self):
    """The inert hybrid state: no term, no history, no k_lane (rec_enabled False and error path)."""
    self.curve_hybrid.reset()
    self.cs_filter.reset()
    self.hybrid_curvature = 0.0
    self.hybrid_klag = float('nan')
    self.hybrid_status = -1
    self.hybrid_nan = True
    self.hybrid_lagdelay = float('nan')
    self.hybrid_lane.ran = False
    self.hybrid_prev = False
    self.hybrid_frame = HybridFrame()
    self.hybrid_ran = False
    self.hybrid_solve_ms = 0.0

  def _update_curve_hybrid(self, sm, md, model_path_x, model_path_y, v_ego_car,
                           measured_curvature, rec_enabled, lane_change_active):
    """One frame of the curve-lane hybrid (SPEC_2B section 2).

    A skip frame does no hybrid MPC work at all; the frame's masks still advance, because the chain
    needs them on either side of a skip. rec_enabled False makes the hybrid inert and forgetful.
    """
    self.hybrid_ran = False
    self.hybrid_solve_ms = 0.0
    if not rec_enabled:
      self._reset_curve_hybrid()
      return

    f = self.hybrid_frame = self._hybrid_frame(sm, md, lane_change_active, model_path_x,
                                               model_path_y, v_ego_car)
    self.hybrid_lagdelay = self._hybrid_lagdelay(sm)

    # V4's planner advanced this guard inside every update(); the hybrid must see the same frames
    # for its lane-mode gate to match, and advancing it costs one finite check.
    model_ok = (len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE
                and len(md.velocity.x) == TRAJECTORY_SIZE)
    if model_ok:
      self.hybrid_lane.update_guard(f.v, float(md.velocity.x[0]), float(md.velocity.x[-1]), True)
    else:
      self.hybrid_lane.update_guard(f.v, float('nan'), float('nan'), False)

    v2 = f.v * f.v
    a_road = abs(f.kappa) * v2
    curve_a = max(abs(f.dc), abs(f.kappa)) * v2
    run = bool(model_ok and
               (HYBRID_RUN_ALWAYS or
                (f.lat and f.v > RUN_V_MIN and
                 (HYBRID_CONTINUOUS or a_road >= PREWARM_ROAD_A or curve_a >= PREWARM_CURVE_A
                  or self.curve_hybrid.on))))
    seed = max(f.v, MIN_SPEED) * float(sm['controlsState'].desiredCurvature)

    if run:
      # A run after a skip starts from a zero-state reset; step() seeds x0[3] = seed as on every run.
      if not self.hybrid_lane.ran:
        self.hybrid_lane.reset_mpc()
      triple = self.hybrid_lane.step(
        md, v_ego_car, v_ego_car * 3.6, measured_curvature, self.curve_speed, seed,
        (self.hybrid_path_cost, self.hybrid_motion_cost, self.hybrid_jerk_cost,
         self.hybrid_srate_cost), bool(sm['carrotMan'].atc_active), self.pathOffset)
      self.hybrid_ran = triple is not None
      if self.hybrid_ran:
        self.hybrid_status = self.hybrid_lane.status
        self.hybrid_nan = self.hybrid_lane.nan
        self.hybrid_solve_ms = self.hybrid_lane.solve_ms
        t_idxs = self.hybrid_lane.t_idxs
        lane_y = self.hybrid_lane.path_xyz[:, 1]
        f.ylane0 = float(np.interp(0.0, t_idxs, lane_y))
        f.ylane10 = float(np.interp(1.0, t_idxs, lane_y))
        curvatures, psis, distances = triple
        self.hybrid_klag = float(get_lag_adjusted_curvature(
          self.CP, f.v, psis, curvatures, self.hybrid_lagdelay, distances))
      else:
        self.hybrid_status = -1
        self.hybrid_nan = True
        self.hybrid_klag = float('nan')
    else:
      # the hybrid MPC did not solve this frame; the next run resets it and re-seeds x0[3]
      self.hybrid_lane.ran = False
      self.hybrid_status = -1
      self.hybrid_nan = True
      self.hybrid_klag = float('nan')

    f.k_lane = self.hybrid_klag
    f.lane_status = self.hybrid_status
    f.lane_nan = self.hybrid_nan
    self.curve_hybrid.update(f)
    self.hybrid_curvature = float(self.curve_hybrid.curvature)

  def update(self, sm, carrot):
    global LATERAL_ACCEL_COST, LATERAL_JERK_COST, STEERING_RATE_COST
    self.readParams -= 1
    if self.readParams <= 0:
      self.readParams = 10
      self.useLaneLineSpeedApply = sm['carState'].useLaneLineSpeed
      self.pathOffset = float(self.params.get_int("PathOffset")) * 0.01
      self.lateralPathCost = self.params.get_float("LatMpcPathCost") * 0.01
      self.lateralMotionCost = self.params.get_float("LatMpcMotionCost") * 0.01
      LATERAL_ACCEL_COST = self.params.get_float("LatMpcAccelCost") * 0.01
      LATERAL_JERK_COST = self.params.get_float("LatMpcJerkCost") * 0.01
      STEERING_RATE_COST = self.params.get_float("LatMpcSteeringRateCost")
      blend_enabled = self.params.get_int("LaneCenterBlend") > 0
      if blend_enabled and not self.lane_center_blend_enabled:
        self.lane_center_blend.reset()
      self.lane_center_blend_enabled = blend_enabled
      self.lane_recenter_assist_enabled = self.params.get_int("LaneRecenterAssist") > 0
      self.hybrid_path_cost = self.params.get_float("LatMpcPathCost") * 0.01
      self.hybrid_motion_cost = self.params.get_float("LatMpcMotionCost") * 0.01
      self.hybrid_jerk_cost = self.params.get_float("LatMpcJerkCost") * 0.01
      self.hybrid_srate_cost = self.params.get_float("LatMpcSteeringRateCost")
      self.hybrid_steer_delay = self.params.get_float("SteerActuatorDelay") * 0.01
      self.hybrid_smooth_sec = self.params.get_float("LatSmoothSec") * 0.01

    # clip speed , lateral planning is not possible at 0 speed
    measured_curvature = sm['controlsState'].curvature
    v_ego_car = max(sm['carState'].vEgo, MIN_SPEED)
    speed_kph = v_ego_car * 3.6
    self.v_ego = v_ego_car
    self.curve_speed = sm['carrotMan'].vTurnSpeed

    # Parse model predictions
    md = sm['modelV2']
    model_active = False
    if (len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE and
        len(md.velocity.x) == TRAJECTORY_SIZE):
      model_active = True
      self.path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
      self.t_idxs = np.array(md.position.t)
      self.plan_yaw = np.array(md.orientation.z)
      self.plan_yaw_rate = np.array(md.orientationRate.z)
      self.velocity_xyz = np.column_stack([md.velocity.x, md.velocity.y, md.velocity.z])
      car_speed = np.linalg.norm(self.velocity_xyz, axis=1)# - get_speed_error(md, v_ego_car)
      self.v_plan = np.clip(car_speed, MIN_SPEED, np.inf)
      self.v_ego = self.v_plan[0]
      self.plan_a = np.array(md.acceleration.x)
      self.laneless_only = not self.lane_model_speed_guard.update(sm['carState'].vEgo, md.velocity.x[0], md.velocity.x[-1])
    else:
      self.lane_model_speed_guard.update(sm['carState'].vEgo, float('nan'), float('nan'))
      self.laneless_only = True

    # Parse model predictions
    self.LP.parse_model(md)
    #lane_change_prob = self.LP.l_lane_change_prob + self.LP.r_lane_change_prob
    #self.DH.update(sm['carState'], md, sm['carControl'].latActive, lane_change_prob, sm)

    if self.useLaneLineSpeedApply == 0 or self.laneless_only:
      self.useLaneLineMode = False
    elif speed_kph >= self.useLaneLineSpeedApply + 2:
      self.useLaneLineMode = True
    elif speed_kph < self.useLaneLineSpeedApply - 2:
      self.useLaneLineMode = False

    # Turn off lanes during lane change
    #if self.DH.desire == log.Desire.laneChangeRight or self.DH.desire == log.Desire.laneChangeLeft:
      
    if md.meta.desire != log.Desire.none or carrot.atc_active:
      self.LP.lane_change_multiplier = 0.0 #md.meta.laneChangeProb
    else:
      self.LP.lane_change_multiplier = 1.0

    # lanelines calculation?
    self.LP.lanefull_mode = self.useLaneLineMode
    self.LP.lane_width_left = md.meta.laneWidthLeft
    self.LP.lane_width_right = md.meta.laneWidthRight
    self.LP.curvature = measured_curvature
    model_path_x = self.path_xyz[:, 0].copy()
    model_path_y = self.path_xyz[:, 1].copy()
    self.path_xyz, self.lanelines_active = self.LP.get_d_path(sm['carState'], v_ego_car, self.t_idxs, self.path_xyz, self.curve_speed)

    if self.lanelines_active:
      self.plan_yaw, self.plan_yaw_rate = yaw_from_path_no_scipy(
        self.path_xyz, self.v_plan,
        smooth_window=5,
        clip_rate=2.0,
        align_first_yaw=None #md.orientation.z[0]  # 초기 정렬
      )
      
    lane_change_active = (md.meta.desire != log.Desire.none or bool(carrot.atc_active) or
                          md.meta.laneChangeState != log.LaneChangeState.off)
    rec_enabled = self.lane_recenter_assist_enabled and not self.lanelines_active and not self.useLaneLineMode
    self.lane_center_blend.update(self.lane_center_blend_enabled, self.lanelines_active, v_ego_car,
                                  self.LP.l_prob, self.LP.r_prob, self.LP.ll_x, self.LP.lll_y, self.LP.rll_y,
                                  self.t_idxs, model_path_x, model_path_y,
                                  lane_change_active, sm['carState'].steeringPressed,
                                  recenter_enabled=rec_enabled)
    self.lane_center_blend_save_frame += 1
    if self.lane_center_blend_save_frame >= int(30.0 / DT_MDL):
      self.lane_center_blend_save_frame = 0
      table_text = dump_table(self.lane_center_blend.table)
      if self.lane_center_blend_enabled and table_text != self.lane_center_blend_saved:
        self.params.put_nonblocking("LaneCenterBlendMap", table_text)
        self.lane_center_blend_saved = table_text

    self.latDebugText = self.LP.debugText
    #self.lanelines_active = True if self.LP.d_prob > 0.3 and self.LP.lanefull_mode else False

    # The hybrid is an add-on: a fault inside it must never stop P0's lateral plan, so it drops to
    # the inert state for this frame and counts the failure instead of propagating.
    try:
      self._update_curve_hybrid(sm, md, model_path_x, model_path_y, v_ego_car,
                                measured_curvature, rec_enabled, lane_change_active)
    except Exception:
      self.hybrid_errors += 1
      if self.hybrid_errors == 1 or self.hybrid_errors % 1000 == 0:
        cloudlog.exception('curveHybrid')
      self._reset_curve_hybrid()

    self.path_xyz[:, 1] += self.pathOffset

    self.lat_mpc.set_weights(self.lateralPathCost, self.lateralMotionCost,
                             LATERAL_ACCEL_COST, LATERAL_JERK_COST,
                             STEERING_RATE_COST)

    y_pts = self.path_xyz[:LAT_MPC_N+1, 1]
    heading_pts = self.plan_yaw[:LAT_MPC_N+1]
    yaw_rate_pts = self.plan_yaw_rate[:LAT_MPC_N+1]
    self.y_pts = y_pts

    assert len(y_pts) == LAT_MPC_N + 1
    assert len(heading_pts) == LAT_MPC_N + 1
    assert len(yaw_rate_pts) == LAT_MPC_N + 1
    lateral_factor = np.clip(self.factor1 - (self.factor2 * self.v_plan**2), 0.0, np.inf)
    p = np.column_stack([self.v_plan, lateral_factor])
    self.lat_mpc.run(self.x0,
                     p,
                     y_pts,
                     heading_pts,
                     yaw_rate_pts)
    # init state for next iteration
    # mpc.u_sol is the desired second derivative of psi given x0 curv state.
    # with x0[3] = measured_yaw_rate, this would be the actual desired yaw rate.
    # instead, interpolate x_sol so that x0[3] is the desired yaw rate for lat_control.
    self.x0[3] = np.interp(DT_MDL, self.t_idxs[:LAT_MPC_N + 1], self.lat_mpc.x_sol[:, 3])

    #  Check for infeasible MPC solution
    mpc_nans = np.isnan(self.lat_mpc.x_sol[:, 3]).any()
    t = time.monotonic()
    if mpc_nans or self.lat_mpc.solution_status != 0:
      self.reset_mpc()
      self.x0[3] = measured_curvature * self.v_ego
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.lat_mpc.cost > 1e6 or mpc_nans:
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0
  
    self.x_sol = self.lat_mpc.x_sol

  def publish(self, sm, pm, carrot):
    plan_solution_valid = self.solution_invalid_cnt < 2
    plan_send = messaging.new_message('lateralPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])
    if not plan_send.valid:
      #print("lateralPlan_valid=", sm.valid)
      #print("lateralPlan_alive=", sm.alive)
      #print("lateralPlan_freq_ok=", sm.freq_ok)
      #print(sm.avg_freq)
      pass

    lateralPlan = plan_send.lateralPlan
    lateralPlan.modelMonoTime = sm.logMonoTime['modelV2']
    lateralPlan.dPathPoints = self.y_pts.tolist()
    lateralPlan.psis = self.lat_mpc.x_sol[0:CONTROL_N, 2].tolist()
    lateralPlan.distances = self.lat_mpc.x_sol[0:CONTROL_N, 0].tolist()

    v_div = np.maximum(self.v_plan[:CONTROL_N], 6.0)
    if len(self.v_plan) == TRAJECTORY_SIZE:
      lateralPlan.curvatures = (self.lat_mpc.x_sol[0:CONTROL_N, 3] / v_div).tolist()
    else:
      lateralPlan.curvatures = (self.lat_mpc.x_sol[0:CONTROL_N, 3] / self.v_ego).tolist()

    v_div2 = max(self.v_ego, 6.0)
    lateralPlan.curvatureRates = [float(x.item() / v_div2) for x in self.lat_mpc.u_sol[0:CONTROL_N - 1]] + [0.0]

    lateralPlan.mpcSolutionValid = bool(plan_solution_valid)
    lateralPlan.solverExecutionTime = self.lat_mpc.solve_time
    if self.debug_mode:
      lateralPlan.solverCost = self.lat_mpc.cost
      lateralPlan.solverState = log.LateralPlan.SolverState.new_message()
      lateralPlan.solverState.x = self.lat_mpc.x_sol.tolist()
      lateralPlan.solverState.u = self.lat_mpc.u_sol.flatten().tolist()

    #lateralPlan.desire = self.DH.desire
    lateralPlan.useLaneLines = self.lanelines_active
    # DesireHelper runs in modeld in this branch. Mirror its published model
    # metadata into lateralPlan so compact HUD clients receive the same manual
    # and automatic lane-change state used by the on-road UI.
    lateralPlan.laneChangeState = sm['modelV2'].meta.laneChangeState
    lateralPlan.laneChangeDirection = sm['modelV2'].meta.laneChangeDirection
    lateralPlan.laneWidth = float(self.LP.lane_width)
    rec_enabled = (self.lane_recenter_assist_enabled and
                   not self.lanelines_active and not self.useLaneLineMode)
    recenter_curvature = float(self.lane_center_blend.recenter_curvature) if rec_enabled else 0.0
    lateralPlan.laneCenterBlendCurvature = (float(self.lane_center_blend.curvature) + recenter_curvature +
                                            self.hybrid_curvature)
    lateralPlan.laneCenterBlendOffset = float(self.lane_center_blend.offset)
    lateralPlan.laneCenterBlendWeight = float(self.lane_center_blend.w)
    lateralPlan.laneCenterBlendGates = int(self.lane_center_blend.gates)
    lateralPlan.laneCenterBlendAvoidDev = float(self.lane_center_blend.r_dev)
    lateralPlan.laneRecenterCurvature = recenter_curvature
    lateralPlan.laneRecenterActive = bool(rec_enabled and self.lane_center_blend.recenter_active)
    lateralPlan.curveHybridCurvature = self.hybrid_curvature
    lateralPlan.curveHybridActive = bool(rec_enabled and self.curve_hybrid.on)
    lateralPlan.curveHybridGuards = int(self.curve_hybrid.guard_bits)
    lateralPlan.curveHybridSolveTime = float(self.hybrid_solve_ms)

    #plan_send.lateralPlan.dPathWLinesX = [float(x) for x in self.d_path_w_lines_xyz[:, 0]]
    #plan_send.lateralPlan.dPathWLinesY = [float(y) for y in self.d_path_w_lines_xyz[:, 1]]
    #lateralPlan.laneWidthLeft = float(self.DH.lane_width_left)
    #lateralPlan.laneWidthRight = float(self.DH.lane_width_right)

    lateralPlan.position.x = self.x_sol[:, 0].tolist()
    lateralPlan.position.y = self.x_sol[:, 1].tolist()
    lateralPlan.position.z = self.path_xyz[:, 2].tolist()
    #lateralPlan.distances = self.lat_mpc.x_sol[0:CONTROL_N, 0].tolist()

    self.x_sol = self.lat_mpc.x_sol

    debugText = (
      f"{'lanemode' if self.lanelines_active else 'laneless'} | " +
      f"{self.LP.lane_width_left:.1f}m | " +
      f"{self.LP.lane_width:.1f}m | " +
      f"{self.LP.lane_width_right:.1f}m | " +
      f"{f'offset={self.LP.offset_total * 100.0:.1f}cm turn={np.clip(self.curve_speed, -200, 200):.0f}km/h' if self.lanelines_active else ''}"
    )

    lateralPlan.latDebugText = debugText
    #lateralPlan.latDebugText = self.latDebugText
    #lateralPlan.laneWidthLeft = float(self.DH.lane_width_left)
    #lateralPlan.laneWidthRight = float(self.DH.lane_width_right)
    #lateralPlan.distanceToRoadEdgeLeft = float(self.DH.distance_to_road_edge_left)
    #lateralPlan.distanceToRoadEdgeRight = float(self.DH.distance_to_road_edge_right)

    pm.send('lateralPlan', plan_send)


def smooth_moving_avg(arr, window=5):
  if window < 2:
    return arr
  if window % 2 == 0:
    window += 1
  pad = window // 2
  arr_pad = np.pad(arr, (pad, pad), mode='edge')
  kernel = np.ones(window) / window
  return np.convolve(arr_pad, kernel, mode='same')[pad:-pad]

def yaw_from_path_no_scipy(path_xyz, v_plan, smooth_window=5,
                           clip_rate=2.0, align_first_yaw=None):

  v0 = float(np.asarray(v_plan)[0]) if len(v_plan) else 0.0
  # 저속(≤6 m/s)에서는 창을 크게
  if v0 <= 6.0:
    smooth_window = max(smooth_window, 9)   # 9~11 권장
    
  N = path_xyz.shape[0]
  x = path_xyz[:, 0].astype(float)
  y = path_xyz[:, 1].astype(float)

  if N < 5:
    return np.zeros(N, np.float32), np.zeros(N, np.float32)

  # 1) s(호길이) 계산
  dx = np.diff(x)
  dy = np.diff(y)
  ds_seg = np.sqrt(dx*dx + dy*dy)
  ds_seg[ds_seg < 0.05] = 0.05
  s = np.zeros(N, float)
  s[1:] = np.cumsum(ds_seg)
  if s[-1] < 0.5:  # 총 호길이 < 0.5m면 미분 결과 의미가 약함
    return np.zeros(N, np.float32), np.zeros(N, np.float32)

  # 2) smoothing (이동평균)
  x_smooth = smooth_moving_avg(x, smooth_window)
  y_smooth = smooth_moving_avg(y, smooth_window)

  # 3) 1·2차 도함수(s축 미분)
  dx_ds  = np.gradient(x_smooth, s)
  dy_ds  = np.gradient(y_smooth, s)
  d2x_ds2 = np.gradient(dx_ds, s)
  d2y_ds2 = np.gradient(dy_ds, s)

  # 4) yaw = atan2(dy/ds, dx/ds)
  yaw = np.unwrap(np.arctan2(dy_ds, dx_ds))

  # 5) 곡률 kappa = ...
  denom = (dx_ds*dx_ds + dy_ds*dy_ds)**1.5
  denom[denom < 1e-9] = 1e-9
  kappa = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / denom

  # 6) yaw_rate = kappa * v
  v = np.asarray(v_plan, float)
  yaw_rate = kappa * v
  if v0 <= 6.0:
    # 이동평균으로 미세 요동 감쇄(창 5~7)
    yaw_rate = smooth_moving_avg(yaw_rate, window=7)

  # 7) 초기 yaw 정렬 (선택)
  if align_first_yaw is not None:
    bias = yaw[0] - float(align_first_yaw)
    yaw = yaw - bias

  # 8) 안정화
  yaw     = np.where(np.isfinite(yaw), yaw, 0.0)
  yaw_rate = np.where(np.isfinite(yaw_rate), yaw_rate, 0.0)
  yaw_rate = np.clip(yaw_rate, -abs(clip_rate), abs(clip_rate))

  return yaw.astype(np.float32), yaw_rate.astype(np.float32)
