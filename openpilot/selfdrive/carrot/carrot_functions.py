import time
from collections import deque
from enum import Enum

from openpilot.cereal import log
from openpilot.common.params import Params
import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.common.constants import CV
from openpilot.common.filter_simple import MyMovingAverage

try:
  from openpilot.common.swaglog import cloudlog
except ImportError:  # PC test environment has no zmq; the vehicle always imports it.
  class _NullCloudlog:
    @staticmethod
    def event(name, **kwargs):
      pass

  cloudlog = _NullCloudlog()
from openpilot.selfdrive.carrot.carrot_man_input import CARROT_MAN_TIMEOUT, get_carrot_man
from openpilot.selfdrive.carrot.t_follow import (
  get_t_follow_mode_factor, get_t_follow_mode_max, get_speed_t_follow_factor, get_lead_response_for_gap, ramp_t_follow,
  ramp_mode_t_follow_factor,
)
from openpilot.selfdrive.carrot.radar_motion.lane_change_gap import LaneChangeGapPlan, LaneChangeGapTracker
from openpilot.selfdrive.carrot.driving_mode import DrivingMode, DrivingModeDetector, get_mode_lead_response
from openpilot.selfdrive.carrot.traffic_stop import TrafficStopModelLeadMatcher, is_traffic_stop_entry_allowed
from openpilot.selfdrive.controls.radar_constants import RADAR_TO_CAMERA
from openpilot.selfdrive.selfdrived.events import Events

EventName = log.OnroadEvent.EventName
LaneChangeState = log.LaneChangeState

class XState(Enum):
  lead = 0
  cruise = 1
  e2eCruise = 2
  e2eStop = 3
  e2ePrepare = 4
  e2eStopped = 5
  e2eStopPrepare = 6

  def __str__(self):
    return self.name

def get_driving_mode_factors(driving_mode: DrivingMode, eco_factor: float = 0.9,
                             safe_factor: float = 0.8) -> tuple[float, float]:
  """Return the accel/comfort and following-time factors for a drive mode."""
  if driving_mode == DrivingMode.Eco:
    return eco_factor, get_t_follow_mode_factor(eco_factor)
  if driving_mode == DrivingMode.Safe:
    return safe_factor, get_t_follow_mode_factor(safe_factor)
  return 1.0, 1.0


def get_driving_mode_comfort_brake_factor(driving_mode: DrivingMode) -> float:
  """Keep Eco efficient while giving Safe a modestly earlier braking profile."""
  return 0.9 if driving_mode == DrivingMode.Safe else 1.0

class TrafficState(Enum):
  off = 0
  red = 1
  green = 2

  def __str__(self):
    return self.name

A_CRUISE_MAX_BP_CARROT = [0., 10 * CV.KPH_TO_MS, 40 * CV.KPH_TO_MS, 60 * CV.KPH_TO_MS, 80 * CV.KPH_TO_MS, 110 * CV.KPH_TO_MS, 140 * CV.KPH_TO_MS]

# W74D TLV-sourced policies (CONTRACT.md High-speed approach follow-ups).
TLV_STANDSTILL_MPS = 0.3
TLV_MOVE_END_MPS = 1.0
TLV_RED_HOLD_FRAMES = int(120.0 / DT_MDL)
TLV_GAS_COOLDOWN_FRAMES = int(10.0 / DT_MDL)
# W74D follow-up (Claude review): the model start sign must not release the
# coast cap; a moving run without any model stopSign is bounded by this limit
# so a stuck false red only coasts. TLV_LOG_MIN_INTERVAL_S rate-limits the TLV
# measurement events (at most one per kind).
TLV_COAST_MAX_MOVING_S = 12.0
TLV_COAST_MAX_MOVING_FRAMES = int(TLV_COAST_MAX_MOVING_S / DT_MDL)
TLV_LOG_MIN_INTERVAL_S = 2.0
TLV_ROUTE_LEFT_NEAR_M = 150.0
TLV_BLINKER_STOP_GAP_S = 0.5
# W74D (A4): a left blinker turned on only after stopping is allowed only in
# the leftmost two lanes, judged by the modelV2 left road edge near the car.
TLV_A4_LEFT_EDGE_MAX_M = 5.6
TLV_A4_EDGE_STD_MAX_M = 0.6
TLV_A4_APPROACH_WINDOW_FRAMES = int(3.0 / DT_MDL)
# W139c fast-road phantom gate: only at >= 80 km/h (W139b: every phantom was
# >= 87 km/h; real high-speed reds at >= 80 km/h were suppressed 0-0.4 s at
# most), open the TLV red coast cap only when the relevant track has detector
# support OR the model is signalling a stop. G-det|G-model, latched per episode.
TLV_FAST_GATE_MPS = 22.2
TLV_FAST_GATE_MIN_DET = 2
TLV_FAST_GATE_MODEL_S = 3.0
# W139c: the TLV red stream can flicker off for ~0.1-0.2 s (fusion latch decay)
# inside one real red. The gate must not treat that as an episode end: it resets
# the latched episode only after raw red has been inactive this long, so a
# mid-approach flicker never drops the coast cap or re-arms the gate.
TLV_FAST_GATE_RESET_S = 0.5
# W143: a TLV red->green/left start also needs the model stop cue to have been
# CLEAR for this long, so a momentary clear is not taken as the green. Measured
# non-terminal model-clear runs at standstill during real reds on 44c/44d/44e/
# 450: 3 runs, max 0.26 s. V1b: a 1.0 s guard removes all simulated false starts.
TLV_START_MODEL_CLEAR_S = 1.0
TLV_A4_STANDSTILL_MIN_FRAMES = int(1.0 / DT_MDL)
TLV_A4_MEDIAN_DISAGREE_M = 1.0
# W147 v2: the model stop cue is unlocked above 82 km/h only when the on-device
# TLV detector confirms the red. The < 82 km/h branch keeps its 5 m lateral
# bound byte-identically; only this branch widens it to 10 m, which is what
# W147 S1/S2 measured (every extra fire was a real red).
TLV_HI_STOP_Y_MAX_M = 10.0
TLV_HI_STOP_MIN_SUPPORT = 2          # W139c semantics: distinct dets in 2.5 s
TLV_HI_STOP_RED_S = 0.5              # TLV red continuously present this long
TLV_HI_STOP_MODEL_S = 0.3            # model expression continuously true
# Same flicker tolerance as the fast gate: the fusion latch can drop the red
# for ~0.1-0.2 s inside one episode, which must not restart the 0.5 s clock.
TLV_HI_STOP_RED_GAP_FRAMES = int(TLV_FAST_GATE_RESET_S / DT_MDL)

class CarrotPlanner:
  def __init__(self, signal_assist=None, start_guard=None, stop_prepare=None):
    # Per-device opt-in only; no flag means the original controller path.
    self.signal_assist = signal_assist
    self._signal_runtime = None
    if signal_assist is None:
      from openpilot.selfdrive.carrot.signal_assist_runtime import SignalAssistRuntime
      runtime = SignalAssistRuntime()
      if runtime.assist is not None:
        self._signal_runtime = runtime
        self.signal_assist = runtime.assist
    self.signal_decision = None
    from openpilot.selfdrive.carrot.signal_start_guard import SignalStartRuntime
    self._start_runtime = SignalStartRuntime() if start_guard is None else None
    self.start_guard = start_guard if start_guard is not None else self._start_runtime.guard
    self.start_decision = None
    from openpilot.selfdrive.carrot.signal_stop_prepare_runtime import SignalPrepareRuntime
    self._prepare_runtime = SignalPrepareRuntime() if stop_prepare is None else None
    self.stop_prepare = stop_prepare if stop_prepare is not None else self._prepare_runtime.prepare
    self.prepare_decision = None
    self.params = Params()
    self.params_count = 0
    self.frame = 0

    #self.log = ""

    #self.aChangeCost = 200
    #self.aChangeCostStart = 40
    #self.tFollowSpeedAdd = 0.0
    #self.tFollowSpeedAddM = 0.0
    #self.tFollowLeadCarSpeed = 0.0
    #self.tFollowLeadCarAccel = 0.0
    #self.lo_timer = 0
    #self.v_ego_prev = 0.0

    self.trafficState = TrafficState.off
    self.last_model_stop_sign = False
    self.xStopFilter = MyMovingAverage(3)
    self.xStopFilter2 = MyMovingAverage(15)
    self.vFilter = MyMovingAverage(10)
    #self.t_follow_prev = self.get_T_FOLLOW()
    self.stop_distance = 6.0
    self.fakeCruiseDistance = 0.0
    self.xState = XState.cruise
    self.xStop = 0.0
    self.actual_stop_distance = 0.0
    #self.debugLongText = ""
    self.stopping_count = 0
    self.traffic_starting_count = 0
    self.user_stop_distance = -1

    self.t_follow_last = 1.5
    self._tf_base_last = 1.5
    self.lane_change_active = False
    self.lane_change_gap = LaneChangeGapPlan()
    self._lane_change_tracker = LaneChangeGapTracker()
    self._lane_change_model_ns = 0

    self.startSignCount = 0
    self.stopSignCount = 0

    self.stop_distance = 6.0
    self.butter_stop = False
    self.trafficStopDistanceAdjust = 2.5 #params.get_float("TrafficStopDistanceAdjust") / 100.
    self.comfortBrake = 2.4
    self.comfort_brake = self.comfortBrake

    self.soft_hold_active = 0
    self.events = Events()
    self.myDrivingMode = DrivingMode(self.params.get_int("MyDrivingMode"))
    self.myDrivingMode_last = self.myDrivingMode
    self.myDrivingModeAuto = self.params.get_int("MyDrivingModeAuto")
    self.myDrivingMode_disable_auto = False
    self.myEcoModeFactor = 0.9
    self.mySafeModeFactor = 0.8
    self.myHighModeFactor = 1.2
    self.drivingModeDetector = DrivingModeDetector()
    self.mySafeFactor = 1.0
    self.myTFollowFactor = 1.0

    self.tFollowGap1 = 1.1
    self.tFollowGap2 = 1.3
    self.tFollowGap3 = 1.45
    self.tFollowGap4 = 1.6

    self.leadAccelResponse = 0
    self.leadAccelResponseBase = 0
    self.leadAccelResponseTF = [-1] * 4
    self.dynamicTFollowLC = 1.0
    self.speedTFFactor = 10
    self.tFollowDecelBoost = 0.0
    self.t_follow_danger = self.tFollowGap3
    self._tf_decel_extra = 0.0
    self.personality = 1

    self.cruiseMaxVals0 = 1.6
    self.cruiseMaxVals1 = 1.6
    self.cruiseMaxVals2 = 1.2
    self.cruiseMaxVals3 = 1.0
    self.cruiseMaxVals4 = 0.8
    self.cruiseMaxVals5 = 0.7
    self.cruiseMaxVals6 = 0.6

    self.aChangeCostStarting = 10.0

    self.trafficLightDetectMode = 2 # 0: None, 1:Stop, 2:Stop&Go
    self.trafficState_carrot = 0
    self.trafficState_carrot_src = 0
    self.carrot_stay_stop = False
    self.tlv_standstill_hold_frames = 0
    self.tlv_hold_run = False  # W74D B1: at most one 120 s hold per unknown run
    self.tlv_gas_cooldown_frames = 0
    self.left_blinker_pre_stop = False
    self.left_blinker_off_frames = 0
    self.a4_frame = 0
    self.a4_approach = deque(maxlen=TLV_A4_APPROACH_WINDOW_FRAMES + 5)
    self.a4_stop_frame = None
    self.a4_approach_median = None
    self.a4_standstill_dists = deque(maxlen=200)
    self.a4_standstill_ok_frames = 0
    self.a4_lane_ok = False

    self.eco_over_speed = 2
    self.eco_target_speed = 0
    
    self.autoNaviSpeedDecelRate = 1.5

    self.desireState = 0.0
    self.desireStateCount = 0
    self.jerk_factor = 1.0

    self.activeCarrot = 0
    self.xDistToTurn = 0
    self.atcType = ""
    self.atc_active = False

    self._stop_x_rl = None
    self._traffic_stop_model_lead_matcher = TrafficStopModelLeadMatcher()
    self.trafficStopModelLeadOffset = 0.0
    self.last_event_time = 0.0
    self.last_model_start_sign = False
    self.tlv_red_active = False
    self.coast_cap_active = False
    # W74D follow-up: moving-timeout and veto instrumentation state.
    self.tlv_coast_moving_frames = 0
    self.tlv_coast_stop_sign_seen = False
    self.tlv_coast_released = False
    self.tlv_veto_active = False
    self.tlv_veto_started = 0.0
    self.tlv_red_confirmed_at = None
    self.tlv_log_veto_t = -1e9
    self.tlv_log_clear_t = -1e9
    self.tlv_log_coast_t = -1e9
    # W147 v2: >= 82 km/h TLV-corroborated model stop state.
    self._hi_stop_carrot_man = None
    self.hi_stop_red_frames = 0
    self.hi_stop_red_gap = 0
    self.hi_stop_model_frames = 0
    # W139c fast-road phantom gate state.
    self.model_stop_sign_t = None
    self.tlv_fast_gate_open = False
    self._fast_gate_ep_start = 0.0
    self._fast_gate_started_slow = False
    self._fast_gate_applied = False
    self._fast_gate_opened_by = None
    self._raw_red_prev = False
    self._fast_gate_last_red_t = None
    # W143: monotonic time since the model stop cue was last False (None while
    # the model is signalling a stop).
    self.model_stop_clear_since = None
    # W144: one "green light" chime per held red episode when the model guard is
    # the only thing blocking a TLV green/left start.
    self.tlv_green_chime_done = False

  def _params_update(self):
    self.frame += 1
    self.params_count += 1
    if self.params_count % 10 == 0:
      myDrivingMode = DrivingMode(self.params.get_int("MyDrivingMode"))
      if myDrivingMode != self.myDrivingMode_last:
        self.myDrivingMode_disable_auto = True
      self.myDrivingMode_last = myDrivingMode
      
      self.myDrivingModeAuto = self.params.get_int("MyDrivingModeAuto")
      if self.myDrivingModeAuto > 0 and not self.myDrivingMode_disable_auto:
        self.myDrivingMode = self.drivingModeDetector.get_mode(self.myDrivingModeAuto)
      else:
        self.myDrivingMode = myDrivingMode

    if self.params_count == 10:
      self.myHighModeFactor = 1.2 #float(self.params.get_int("MyHighModeFactor")) / 100.
      self.trafficLightDetectMode = self.params.get_int("TrafficLightDetectMode") # 0: None, 1:Stop, 2:Stop&Go
    elif self.params_count == 20:
      self.tFollowGap1 = self.params.get_float("TFollowGap1") / 100.
      self.tFollowGap2 = self.params.get_float("TFollowGap2") / 100.
      self.tFollowGap3 = self.params.get_float("TFollowGap3") / 100.
      self.tFollowGap4 = self.params.get_float("TFollowGap4") / 100.
      self.leadAccelResponseBase = int(np.clip(self.params.get_int("LeadAccelResponse"), 0, 5))
      self.leadAccelResponseTF = [int(np.clip(self.params.get_int(f"LeadAccelResponseTF{i}"), -1, 5)) for i in range(1, 5)]
      self.dynamicTFollowLC = self.params.get_float("DynamicTFollowLC") / 100.
      self.speedTFFactor = int(np.clip(self.params.get_int("SpeedTFFactor"), 10, 30))
      self.tFollowDecelBoost = self.params.get_float("TFollowDecelBoost") / 100.
    elif self.params_count == 30:
      self.cruiseMaxVals0 = self.params.get_float("CruiseMaxVals0") / 100.
      self.cruiseMaxVals1 = self.params.get_float("CruiseMaxVals1") / 100.
      self.cruiseMaxVals2 = self.params.get_float("CruiseMaxVals2") / 100.
      self.cruiseMaxVals3 = self.params.get_float("CruiseMaxVals3") / 100.
      self.cruiseMaxVals4 = self.params.get_float("CruiseMaxVals4") / 100.
      self.cruiseMaxVals5 = self.params.get_float("CruiseMaxVals5") / 100.
      self.cruiseMaxVals6 = self.params.get_float("CruiseMaxVals6") / 100.
    elif self.params_count == 40:
      self.stop_distance = self.params.get_float("StopDistanceCarrot") / 100.
      try:
        self.butter_stop = int(self.params.get_int("ButterStop")) != 0
      except (TypeError, ValueError, AttributeError):
        self.butter_stop = False
      self.eco_over_speed = self.params.get_int("CruiseEcoControl")
      self.autoNaviSpeedDecelRate = float(self.params.get_int("AutoNaviSpeedDecelRate")) * 0.01
      self.aChangeCostStarting = self.params.get_float("AChangeCostStarting")
      self.trafficStopDistanceAdjust = self.params.get_float("TrafficStopDistanceAdjust") / 100.
    elif self.params_count >= 100:

      self.params_count = 0

  def get_carrot_accel(self, v_ego):
    cruiseMaxVals = [self.cruiseMaxVals0, self.cruiseMaxVals1, self.cruiseMaxVals2, self.cruiseMaxVals3, self.cruiseMaxVals4, self.cruiseMaxVals5, self.cruiseMaxVals6]
    factor = self.myHighModeFactor if self.myDrivingMode == DrivingMode.High else self.mySafeFactor
    accel = np.interp(v_ego, A_CRUISE_MAX_BP_CARROT, cruiseMaxVals) * factor
    if getattr(self, 'coast_cap_active', False):
      # Graded redLikely (mode 2, model not signalling a go): coast only, never brake.
      accel = min(accel, 0.0)
    return accel

  def _get_base_t_follow(self, personality, v_ego):
    if personality == log.LongitudinalPersonality.moreRelaxed:
      self.jerk_factor = 1.0
      tf_base = self.tFollowGap4
    elif personality == log.LongitudinalPersonality.relaxed:
      self.jerk_factor = 1.0
      tf_base = self.tFollowGap3
    elif personality == log.LongitudinalPersonality.standard:
      self.jerk_factor = 0.7
      tf_base = self.tFollowGap2
    elif personality == log.LongitudinalPersonality.aggressive:
      self.jerk_factor = 0.5
      tf_base = self.tFollowGap1
    else:
      raise NotImplementedError("Longitudinal personality not supported")
    return float(tf_base)

  def _apply_speed_t_follow_scale(self, tf_base, v_ego):
    return float(tf_base * get_speed_t_follow_factor(self.speedTFFactor, v_ego * CV.MS_TO_KPH))

  def _apply_decel_hold_and_boost_t_follow(self, tf_target, a_ego):
    # Hold only the unboosted baseline. Feeding the previous boosted target
    # back here would add the same margin on every planner cycle.
    previous_extra = getattr(self, "_tf_decel_extra", 0.0)
    previous_base = getattr(self, "_tf_decel_base", getattr(self, "_tf_applied", tf_target) - previous_extra)
    self._tf_decel_base = max(tf_target, previous_base) if a_ego <= -0.2 else tf_target
    requested_extra = float(np.interp(a_ego, [-2.5, -1.0, -0.3, 0.0], [0.50, 0.25, 0.06, 0.0])) * self.tFollowDecelBoost
    # Add braking margin promptly; release only the extra margin progressively
    # so its removal does not suddenly invite acceleration to a shorter gap.
    self._tf_decel_extra = max(requested_extra, previous_extra - 0.10 * DT_MDL)
    return float(self._tf_decel_base + self._tf_decel_extra)


  def _clip_t_follow(self, t_follow):
    # The selected, speed-scaled baseline can exceed every configured static gap.
    # Preserve the deceleration hold when speed (and its multiplier) falls.
    tf_max = max(getattr(self, '_tf_mode_target', t_follow), getattr(self, '_tf_decel_base', t_follow))
    tf_max = get_t_follow_mode_max(tf_max, 1.0, self._tf_decel_extra)
    return float(np.clip(t_follow, 0.3, max(0.3, tf_max)))

  def get_T_FOLLOW(self, personality=log.LongitudinalPersonality.standard, v_ego=0.0, a_ego=0.0):
    tf_base = self._get_base_t_follow(personality, v_ego)
    tf_target = self._apply_speed_t_follow_scale(tf_base, v_ego)
    self._tf_mode_factor = ramp_mode_t_follow_factor(
      self.myTFollowFactor, getattr(self, "_tf_mode_factor", self.myTFollowFactor), DT_MDL,
    )
    tf_mode_target = float(tf_target * self._tf_mode_factor)
    self._tf_mode_target = tf_mode_target
    tf_adjusted = self._apply_decel_hold_and_boost_t_follow(tf_mode_target, a_ego)
    tf_final = self._clip_t_follow(tf_adjusted)
    self._tf_applied = float(tf_final)
    # Apply the baseline increase ramp once per planner cycle.
    self._tf_base_last = ramp_t_follow(tf_final, getattr(self, '_tf_base_last', self.t_follow_last), self._tf_decel_extra, DT_MDL)
    self.t_follow_last = float(self._tf_base_last)
    # The MPC danger boundary uses the static gap and mode factor only.
    # SpeedTFFactor, the decel hold and the decel boost are comfort distance.
    self.t_follow_danger = float(np.clip(tf_base * self._tf_mode_factor, 0.3, max(0.3, self.t_follow_last)))
    return self.t_follow_last


  def _update_model_desire(self, sm):
    meta = sm['modelV2'].meta
    carState = sm['carState']
    self.lane_change_active = meta.laneChangeState in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)

    if meta.laneChangeState == LaneChangeState.laneChangeStarting:
      self.desireState = meta.desireState[3] if carState.leftBlinker else meta.desireState[4]
      self.desireStateCount += 1
    else:
      self.desireState = 0.0
      self.desireStateCount = 0

    self._update_lane_change_gap(sm)

  def _update_lane_change_gap(self, sm):
    state, model, radar = sm['carState'], sm['modelV2'], sm['radarState']
    signal = state.leftBlinker != state.rightBlinker
    direction = ((-1 if state.leftBlinker else 1) if signal else self._lane_change_tracker.direction) if self.lane_change_active else 0
    now_ns = int(sm.logMonoTime['modelV2'])
    valid = all(sm.valid[key] and sm.alive[key] for key in ('carState', 'modelV2', 'radarState'))
    valid = valid and abs(now_ns - int(sm.logMonoTime['radarState'])) <= 200_000_000
    if not valid or direction == 0:
      self._lane_change_tracker.reset()
      self.lane_change_gap = LaneChangeGapPlan(active=self.lane_change_active, reason='invalid-input' if self.lane_change_active else 'inactive')
      self._lane_change_model_ns = 0
      return
    if now_ns == self._lane_change_model_ns:
      return  # fast radar must not count the same model/pose twice
    self._lane_change_model_ns = now_ns
    pose = sm['livePose']
    angular = pose.angularVelocityDevice
    pose_valid = (sm.valid['livePose'] and sm.alive['livePose'] and pose.inputsOK and pose.sensorsOK and angular.valid
                  and abs(now_ns - int(sm.logMonoTime['livePose'])) <= 150_000_000)
    self.lane_change_gap = self._lane_change_tracker.update(
      now=now_ns * 1e-9, direction=direction, v_ego=float(state.vEgo),
      yaw_rate=float(angular.z) if pose_valid else float('nan'),
      path_t=tuple(model.position.t), path_x=tuple(model.position.x), path_y=tuple(model.position.y),
      primary=radar.leadOne, secondary=radar.leadTwo,
      blindspot=not signal or bool(state.leftBlindspot if direction == -1 else state.rightBlindspot), valid=valid,
    )


  def update_stop_dist(self, stop_x):
    stop_x = self.xStopFilter.process(stop_x, median = True)
    stop_x = self.xStopFilter2.process(stop_x)
    return stop_x

  def _reset_hi_stop_state(self) -> None:
    """Forget the >= 82 km/h unlock state once the car is slower than 82 km/h."""
    self.hi_stop_red_frames = 0
    self.hi_stop_red_gap = 0
    self.hi_stop_model_frames = 0
    self.hi_stop_support_ok = False

  def _check_model_stopping_hi(self, v_cruise, v, model_v, model_x, y, d_rel, a_ego) -> bool:
    """W147 v2: model stop cue at >= 82 km/h, unlocked only by a TLV red.

    Same expression as the < 82 km/h branch except the lateral bound, which is
    TLV_HI_STOP_Y_MAX_M, and it only counts when the on-device detector
    corroborates the red: carrotMan.trafficStateSrc == 1, trafficState == 1
    (red) for TLV_HI_STOP_RED_S, and carrotMan.tlvDetSupport >= 2 somewhere in
    that red episode (W139c semantics: latched until the red run ends; the
    field is absent -> 0, so a route without it never unlocks). The model
    expression itself must hold for TLV_HI_STOP_MODEL_S. Gas cooldown
    suppresses it, as it suppresses every TLV control effect. The W148
    expressway gate is upstream: it never lets a src-1 state reach carrotMan
    there.
    """
    expr = (model_x < d_rel - 3.0 and
            model_x < np.interp(v[0] * 3.6, [60, 80], [120.0, 150]) and
            ((model_v < 3.0) or (model_v < v[0] * 0.7)) and
            abs(y[-1]) < TLV_HI_STOP_Y_MAX_M)
    if expr and v_cruise != 0 and self.xState == XState.e2eCruise and a_ego < -1.0:
      expr = False

    # Model persistence is counted on the expression alone: the corroboration
    # gates the fire, not the evidence that the model saw the stop.
    self.hi_stop_model_frames = self.hi_stop_model_frames + 1 if expr else 0

    man = self._hi_stop_carrot_man
    red = False
    if man is not None:
      # Absent/old fields read as 0: a route that never logged the detector
      # state or the support can never unlock the >= 82 km/h branch.
      red = (int(getattr(man, 'trafficStateSrc', 0)) == 1
             and int(getattr(man, 'trafficState', 0)) == 1)
      if red:
        self.hi_stop_support_ok = self.hi_stop_support_ok or (
          int(getattr(man, 'tlvDetSupport', 0)) >= TLV_HI_STOP_MIN_SUPPORT)

    if red:
      self.hi_stop_red_frames += 1
      self.hi_stop_red_gap = 0
    elif self.hi_stop_red_frames > 0 and self.hi_stop_red_gap < TLV_HI_STOP_RED_GAP_FRAMES:
      # Tolerate the fusion latch's short drop inside one red episode: the run
      # keeps its length but stops growing, and the support latch survives.
      self.hi_stop_red_gap += 1
    else:
      self.hi_stop_red_frames = 0
      self.hi_stop_red_gap = 0
      self.hi_stop_support_ok = False

    red_ok = self.hi_stop_red_frames * DT_MDL >= TLV_HI_STOP_RED_S
    model_ok = self.hi_stop_model_frames * DT_MDL >= TLV_HI_STOP_MODEL_S
    return bool(expr and red_ok and model_ok and self.hi_stop_support_ok
                and self.tlv_gas_cooldown_frames == 0)

  def check_model_stopping(self, v_cruise, v, v_ego, a_ego, model_x, y, d_rel):
    v_ego_kph = v_ego * CV.MS_TO_KPH
    model_v = self.vFilter.process(v[-1])
    startSign = model_v > 5.0 or model_v > (v[0] + 2)

    # W147 v2: the >= 82 km/h unlock has no memory across a slow phase.
    if v_ego_kph < 82.0:
      self._reset_hi_stop_state()

    if v_ego_kph < 1.0:
      stopSign = model_x < 20.0 and model_v < 10.0
    elif v_ego_kph < 82.0:
      stopSign = (model_x < d_rel - 3.0 and
                  model_x < np.interp(v[0] * 3.6, [60, 80], [120.0, 150]) and
                  ((model_v < 3.0) or (model_v < v[0] * 0.7)) and
                  abs(y[-1]) < 5.0)
      # 정상 주행 중 감속하는 경우(카메라 감속 등)에는 오감지가 많음.
      # 회생 감속으로 v_cruise가 0인 경우에는 신호를 감지하도록 함.
      if v_cruise != 0 and (self.xState == XState.e2eCruise and a_ego < -1.0):
        stopSign = False
    else:
      stopSign = self._check_model_stopping_hi(v_cruise, v, model_v, model_x, y, d_rel, a_ego)

    self.last_model_stop_sign = bool(stopSign)
    self.last_model_start_sign = bool(startSign)
    # W143: track how long the model stop cue has been continuously clear.
    if stopSign:
      self.model_stop_clear_since = None
    elif self.model_stop_clear_since is None:
      self.model_stop_clear_since = time.monotonic()
    # W139c: remember when the model last signalled a stop (fast-road gate).
    if stopSign:
      self.model_stop_sign_t = time.monotonic()

    # self.stopSignCount = (
    #   self.stopSignCount + 1
    #   if (
    #     stopSign
    #     and (
    #       model_x > get_safe_obstacle_distance(
    #         v_ego,
    #         t_follow=0,
    #         comfort_brake=COMFORT_BRAKE,
    #         stop_distance=-1.0,
    #       )
    #     )
    #   )
    #   else 0
    # )
    self.stopSignCount = self.stopSignCount + 1 if stopSign else 0
    self.startSignCount = self.startSignCount + 1 if startSign and not stopSign else 0

    if self.stopSignCount * DT_MDL > 0.0:
      self.trafficState = TrafficState.red
    elif self.startSignCount * DT_MDL > 0.2:
      self.trafficState = TrafficState.green
    else:
      self.trafficState = TrafficState.off

  def _update_left_blinker_pre_stop(self, carstate, v_ego_kph: float) -> None:
    """W74D (A): a left blinker already on when the car came to a stop and
    stayed on since (>0.5 s off-gap disqualifies) may authorize a TLV red->left
    start. While moving, the current blinker state primes the next stop."""
    left_on = bool(carstate.leftBlinker)
    if v_ego_kph >= TLV_STANDSTILL_MPS * CV.MS_TO_KPH:
      self.left_blinker_pre_stop = left_on
      self.left_blinker_off_frames = 0
      return
    if left_on:
      self.left_blinker_off_frames = 0
    else:
      self.left_blinker_off_frames += 1
      if self.left_blinker_off_frames * DT_MDL > TLV_BLINKER_STOP_GAP_S:
        self.left_blinker_pre_stop = False

  @staticmethod
  def _engaged(sm) -> bool:
    try:
      return bool(sm['selfdriveState'].enabled)
    except Exception:
      return False

  def _update_tlv_gas_cooldown(self, carstate) -> None:
    """B3 gas override: gas pressed while a TLV effect is active (redLikely
    coast cap, standstill red hold, src-1 stay-stop) suppresses every TLV
    control effect for 10 s after release, in frame/DT_MDL timing. Ordinary
    non-TLV behavior continues."""
    effect_active = (
      self.coast_cap_active
      or self.tlv_standstill_hold_frames > 0
      or (self.carrot_stay_stop and self.trafficState_carrot_src == 1)
    )
    if carstate.gasPressed and (effect_active or self.tlv_gas_cooldown_frames > 0):
      self.tlv_gas_cooldown_frames = TLV_GAS_COOLDOWN_FRAMES
    elif not carstate.gasPressed and self.tlv_gas_cooldown_frames > 0:
      self.tlv_gas_cooldown_frames -= 1
    if self.tlv_gas_cooldown_frames > 0:
      self.coast_cap_active = False
      self.tlv_standstill_hold_frames = 0

  @staticmethod
  def _a4_left_edge_sample(model) -> tuple[float, float] | None:
    """W74D (A4): (|lateral y|, std) of the modelV2 left road edge near the car.

    No existing Carrot helper estimates the left edge (the only roadEdges
    consumer is the web vision overlay), so read modelV2 directly:
    roadEdges[0] (left edge), the point with x in 0-10 m nearest to 5 m, and
    roadEdgeStds[0] - one scalar per edge (modeld/fill_model_msg.py:156).
    """
    try:
      edge = model.roadEdges[0]
      xs = edge.x
      ys = edge.y
      best_idx = None
      best_dx = None
      for i, xv in enumerate(xs):
        x = float(xv)
        if 0.0 <= x <= 10.0:
          dx = abs(x - 5.0)
          if best_dx is None or dx < best_dx:
            best_dx = dx
            best_idx = i
      if best_idx is None or best_idx >= len(ys):
        return None
      dist = abs(float(ys[best_idx]))
      std = abs(float(model.roadEdgeStds[0]))
      if not (np.isfinite(dist) and np.isfinite(std)):
        return None
      return dist, std
    except Exception:
      return None

  def _update_a4_lane_tracking(self, model, v_ego_kph: float) -> None:
    """W74D (A4): the fresh-blinker (turned on after the stop) start way-in
    requires the left road edge (<= 5.6 m, std <= 0.6 m) to pass the median of
    the last 3 s before the stop and to keep passing for >= 1.0 s continuously
    at standstill; approach and standstill medians must agree within 1.0 m.
    Missing samples or a too-high std keep the condition False."""
    self.a4_frame += 1
    stopped = v_ego_kph < TLV_STANDSTILL_MPS * CV.MS_TO_KPH
    sample = self._a4_left_edge_sample(model) if model is not None else None
    if not stopped:
      self.a4_approach.append((self.a4_frame, sample))
      self.a4_stop_frame = None
      self.a4_approach_median = None
      self.a4_standstill_dists.clear()
      self.a4_standstill_ok_frames = 0
      self.a4_lane_ok = False
      return
    if self.a4_stop_frame is None:
      self.a4_stop_frame = self.a4_frame
      window = [s for f, s in self.a4_approach if f > self.a4_frame - TLV_A4_APPROACH_WINDOW_FRAMES]
      if window and all(s is not None for s in window) and max(s[1] for s in window) <= TLV_A4_EDGE_STD_MAX_M:
        self.a4_approach_median = float(np.median([s[0] for s in window]))
    if sample is not None:
      self.a4_standstill_dists.append(sample[0])
      if sample[1] <= TLV_A4_EDGE_STD_MAX_M and sample[0] <= TLV_A4_LEFT_EDGE_MAX_M:
        self.a4_standstill_ok_frames += 1
      else:
        self.a4_standstill_ok_frames = 0
    else:
      self.a4_standstill_ok_frames = 0
    if self.a4_approach_median is not None and self.a4_standstill_dists:
      stand_median = float(np.median(self.a4_standstill_dists))
      self.a4_lane_ok = (
        self.a4_approach_median <= TLV_A4_LEFT_EDGE_MAX_M
        and self.a4_standstill_ok_frames >= TLV_A4_STANDSTILL_MIN_FRAMES
        and abs(stand_median - self.a4_approach_median) <= TLV_A4_MEDIAN_DISAGREE_M
      )
    else:
      self.a4_lane_ok = False

  def _update_carrot_man(self, sm, v_ego_kph, v_cruise_kph):
    atc_active = False
    carstate = sm['carState']
    self._update_left_blinker_pre_stop(carstate, v_ego_kph)
    carrot_man = get_carrot_man(sm)
    # W147 v2: the >= 82 km/h unlock reads the fusion's own state + support.
    self._hi_stop_carrot_man = carrot_man
    if carrot_man is not None:
      state_in = int(carrot_man.trafficState)
      src_in = int(getattr(carrot_man, 'trafficStateSrc', 0))
      tlv_src = src_in == 1
      atc_turn_left = carrot_man.atcType in ["turn left", "atc left"]
      gas = bool(carstate.gasPressed)
      engaged = self._engaged(sm)
      v_stopped = v_ego_kph < TLV_STANDSTILL_MPS * CV.MS_TO_KPH
      v_moving_end = v_ego_kph > TLV_MOVE_END_MPS * CV.MS_TO_KPH
      cooldown = self.tlv_gas_cooldown_frames > 0
      # A start trigger produced by the on-device carrotTlv detector may start
      # the car only while the supercombo model is not signalling a stop, i.e.
      # both cues agree (ONE extra gate for tlv-originated red->green AND
      # red->left edges).
      tlv_start_disagrees = (
        tlv_src
        and state_in in (2, 3)
        and (getattr(self, 'last_model_stop_sign', False)
             or self.model_stop_clear_since is None
             or time.monotonic() - self.model_stop_clear_since < TLV_START_MODEL_CLEAR_S)
      )
      # W74D (A): a TLV red->left start also needs a nearby routed left
      # (carrotMan.atcType; carrotMan.xDistToTurn <= 150 m, int metres from
      # navInstruction.maneuverDistance) or a left blinker already on before
      # the stop; the right blinker vetoes it.
      route_left_near = atc_turn_left and carrot_man.xDistToTurn <= TLV_ROUTE_LEFT_NEAR_M
      # A4: a left blinker turned on only after the stop is allowed in the
      # leftmost two lanes (model-confident left edge) instead of pre-stop on.
      fresh_blinker_lane_ok = bool(carstate.leftBlinker) and self.a4_lane_ok
      left_start_allowed = (
        route_left_near or self.left_blinker_pre_stop or fresh_blinker_lane_ok
      ) and not bool(carstate.rightBlinker)
      trigger_start = self.carrot_stay_stop = False
      if atc_turn_left or carstate.leftBlinker:
        if self.trafficState_carrot == 1 and state_in == 3:  # red -> left triggered
          if tlv_src:
            trigger_start = (not tlv_start_disagrees) and left_start_allowed and not cooldown
          else:
            trigger_start = not tlv_start_disagrees
        elif state_in in [1, 2]:
          # TLV red/left keeps the stop only outside the gas cooldown;
          # non-TLV sources stay byte-identical to the pre-W74D behavior.
          self.carrot_stay_stop = not (tlv_src and cooldown)
      elif self.trafficState_carrot == 1 and state_in == 2:  # red -> green triggered
        if tlv_src:
          trigger_start = (not tlv_start_disagrees) and not cooldown
        else:
          trigger_start = not tlv_start_disagrees
      else:
        trigger_start = False
      # W74D instrument: exactly one cloudlog event per blocked TLV start edge
      # (rate-limited 2 s per kind). The episode ends when the held start fires
      # (cleared event, delay_s) or the edge decays (state leaves 2/3 or the
      # confirmed red ends).
      left_intent = bool(atc_turn_left or carstate.leftBlinker)
      blocked_edge = (
        tlv_start_disagrees and self.trafficState_carrot == 1
        and ((state_in == 3 and left_intent) or (state_in == 2 and not left_intent))
      )
      if self.tlv_veto_active:
        if trigger_start:
          self._log_tlv_start_veto_cleared()
          self.tlv_veto_active = False
        elif not blocked_edge:
          self.tlv_veto_active = False
      elif blocked_edge:
        self.tlv_veto_active = True
        self.tlv_veto_started = time.monotonic()
        self._log_tlv_start_veto(sm, state_in, v_ego_kph * CV.KPH_TO_MS, left_intent)
      # W144: the lamp turned green but only the model guard holds the start.
      # Raise the existing "traffic sign changed" alert (its chime) so the driver
      # knows, at most once per held red episode. This never starts the car.
      if (blocked_edge and v_stopped and engaged and not gas and not cooldown
          and not self.tlv_green_chime_done):
        self.add_event(EventName.trafficSignChanged)
        self.tlv_green_chime_done = True
        red_age = -1.0 if self.tlv_red_confirmed_at is None else time.monotonic() - self.tlv_red_confirmed_at
        cloudlog.event("tlv_green_chime", state_in=int(state_in),
                       v_ego=float(v_ego_kph * CV.KPH_TO_MS), red_age_s=float(red_age))
      if self.trafficState_carrot != 1 or gas or not engaged or v_moving_end:
        self.tlv_green_chime_done = False
      # B1: a confirmed TLV red at standstill survives an unknown (0) for up
      # to 120 s so a later green/left edge can still fire; unknown never
      # counts as go. Gas, disengage, > 1 m/s or expiry end the hold.
      if self.tlv_standstill_hold_frames > 0:
        if state_in != 0 or gas or not engaged or v_moving_end:
          self.tlv_standstill_hold_frames = 0
        else:
          self.tlv_standstill_hold_frames -= 1
      hold_armed = (
        self.trafficState_carrot == 1 and self.trafficState_carrot_src == 1
        and v_stopped and engaged and not gas and not cooldown
      )
      if state_in != 0:
        # Any non-zero TLV state ends the current unknown run.
        self.tlv_hold_run = False
      elif not self.tlv_hold_run:
        # First unknown frame of this run: arm at most one full 120 s hold.
        # The run never re-arms after the hold expires (spec: the hold ends
        # when 120 s elapsed) until a non-zero state starts a new run.
        self.tlv_hold_run = True
        if hold_armed and self.tlv_standstill_hold_frames == 0:
          self.tlv_standstill_hold_frames = TLV_RED_HOLD_FRAMES
      hold_skip_store = state_in == 0 and self.tlv_standstill_hold_frames > 0
      # Graded redLikely (4) must never overwrite the last confirmed state:
      # 1 -> 4 -> 2/3 would otherwise lose the red->green / red->left edge.
      # 4 is not red here, and it is not in (2, 3) for tlv_start_disagrees.
      if not tlv_start_disagrees and state_in != 4 and not hold_skip_store:
        self.trafficState_carrot = state_in
        self.trafficState_carrot_src = src_in
      # else: hold the previous value so the start edge can fire later once
      # the model's stop cue clears; a tlv trigger that decays away first
      # leaves normal (non-tlv) behavior untouched.
      if self.trafficState_carrot == 1 and self.trafficState_carrot_src == 1:
        if self.tlv_red_confirmed_at is None:
          self.tlv_red_confirmed_at = time.monotonic()
      else:
        self.tlv_red_confirmed_at = None

      if trigger_start:
        if self.soft_hold_active > 0:
          self.add_event(EventName.trafficSignChanged)
        elif self.xState in [XState.e2eStop, XState.e2eStopped]:
          self.xState = XState.e2eCruise
          self.traffic_starting_count = 10.0 / DT_MDL

      self.activeCarrot = carrot_man.activeCarrot
      self.xDistToTurn = carrot_man.xDistToTurn
      atc_active = self.activeCarrot > 1 and 0 < self.xDistToTurn < 100
      self.atcType = carrot_man.atcType

      v_cruise_kph = min(v_cruise_kph, carrot_man.desiredSpeed)
    else:
      self.trafficState_carrot = 0
      self.trafficState_carrot_src = 0
      self.tlv_standstill_hold_frames = 0
      self.tlv_hold_run = False
      self.carrot_stay_stop = False
      self.tlv_red_confirmed_at = None
      self.tlv_veto_active = False
      self.tlv_green_chime_done = False
      self.activeCarrot = 0
      self.xDistToTurn = 0
      self.atcType = ""

    return v_cruise_kph, atc_active

  def _read_tlv_red_active(self, sm) -> bool:
    """TLV red active from carrotMan: fresh TLV-sourced trafficState 1 or 4.

    State 4 is the graded redLikely, state 1 the confirmed red. The left-intent
    exception is already applied upstream by carrot_serv: redLeft/left with an
    active turn intent maps to "Left turn" (state 3), so state 1 here is a
    straight red for our path and needs no extra check. Read directly (not via
    get_carrot_man, which needs an active navigation speed) so the grade also
    works at a red light with no route set.
    """
    try:
      service = 'carrotMan'
      if not (sm.seen[service] and sm.alive[service] and sm.valid[service]):
        return False
      if time.monotonic() - sm.recv_time[service] > CARROT_MAN_TIMEOUT:
        return False
      carrot_man = sm[service]
      return (int(getattr(carrot_man, 'trafficStateSrc', 0)) == 1
              and int(getattr(carrot_man, 'trafficState', 0)) in (1, 4))
    except Exception:
      return False

  def _apply_fast_gate(self, raw_red: bool, sm, v_ego: float) -> bool:
    """W139c fast-road phantom gate: G-det OR G-model, only at >= 22.2 m/s.

    At high speed the TLV red coast cap is suppressed until the relevant track
    has >= 2 detector matches in the last 2.5 s (carrotMan.tlvDetSupport) OR the
    model signalled a stop within 3.0 s. The gate LATCHES open for the episode
    (until the raw TLV red goes inactive), so a support gap does not flicker it.
    An episode that starts below 22.2 m/s is never gated, even if speed rises.
    Below the threshold the cap is byte-identical to the ungated behaviour.
    """
    now = time.monotonic()
    if not raw_red:
      # Raw red inactive: hold the latched episode until red has been absent
      # TLV_FAST_GATE_RESET_S (measured from the last red frame), so a brief
      # stream flicker does not drop the cap or re-arm the gate (W139c).
      elapsed = None if self._fast_gate_last_red_t is None else now - self._fast_gate_last_red_t
      if elapsed is not None and elapsed >= TLV_FAST_GATE_RESET_S:
        self._log_fast_gate_episode_end()
        self.tlv_fast_gate_open = False
        self._fast_gate_applied = False
        self._fast_gate_ep_start = 0.0
        self._fast_gate_started_slow = False
        self._fast_gate_opened_by = None
        self._fast_gate_last_red_t = None
      self._raw_red_prev = False
      return False
    cont = (self._fast_gate_last_red_t is not None
            and (now - self._fast_gate_last_red_t) <= TLV_FAST_GATE_RESET_S)
    if not cont:
      # New red episode.
      self._fast_gate_ep_start = now
      self._fast_gate_started_slow = v_ego < TLV_FAST_GATE_MPS
      self._fast_gate_applied = False
      self._fast_gate_opened_by = None
      self.tlv_fast_gate_open = False
    # Continuation across a flicker keeps the latched state (started_slow, open,
    # opened_by) unchanged.
    self._raw_red_prev = True
    self._fast_gate_last_red_t = now
    if self._fast_gate_started_slow:
      # Episode began below the threshold: never gated.
      return True
    if self.tlv_fast_gate_open:
      return True
    det_ok = False
    try:
      det_ok = int(getattr(sm['carrotMan'], 'tlvDetSupport', 0)) >= TLV_FAST_GATE_MIN_DET
    except Exception:
      det_ok = False
    model_ok = self.model_stop_sign_t is not None and (now - self.model_stop_sign_t) <= TLV_FAST_GATE_MODEL_S
    if det_ok or model_ok:
      self.tlv_fast_gate_open = True
      self._fast_gate_opened_by = "det" if det_ok else "model"
    return self.tlv_fast_gate_open

  def _log_fast_gate_episode_end(self) -> None:
    """One cloudlog event per TLV red episode that the fast gate touched."""
    if self._fast_gate_applied and self.tlv_fast_gate_open:
      return
    duration = max(0.0, time.monotonic() - self._fast_gate_ep_start) if self._fast_gate_ep_start else 0.0
    opened = bool(self.tlv_fast_gate_open)
    cloudlog.event("tlv_fast_gate", opened=opened, opened_by=self._fast_gate_opened_by,
                   suppressed=not opened, duration_s=round(duration, 2),
                   started_slow=bool(self._fast_gate_started_slow))
    if opened:
      self._fast_gate_applied = True

  def _update_coast_cap(self, v_ego: float) -> None:
    """TLV red active (confirmed 1 or graded 4) -> coast (accel cap 0): mode 2
    only, gas cooldown respected. Never adds braking; mode 0/1 keep the pre-W74C
    behavior. The model start sign does NOT release the cap (it stays True while
    cruising toward a red the model has not noticed). A moving run longer than
    TLV_COAST_MAX_MOVING_S without any model stopSign releases the cap, latched
    until the TLV state leaves (1, 4); standstill keeps the cap so the car
    cannot creep into the red."""
    if not self.tlv_red_active:
      self.tlv_coast_moving_frames = 0
      self.tlv_coast_stop_sign_seen = False
      self.tlv_coast_released = False
    cap = (self.trafficLightDetectMode == 2 and self.tlv_red_active
           and self.tlv_gas_cooldown_frames == 0)
    if cap:
      if self.last_model_stop_sign:
        self.tlv_coast_stop_sign_seen = True
      if v_ego > TLV_MOVE_END_MPS:
        self.tlv_coast_moving_frames += 1
        if (self.tlv_coast_moving_frames > TLV_COAST_MAX_MOVING_FRAMES
            and not self.tlv_coast_stop_sign_seen):
          if not self.tlv_coast_released:
            self._log_tlv_coast_timeout(v_ego)
          self.tlv_coast_released = True
      else:
        self.tlv_coast_moving_frames = 0
        self.tlv_coast_stop_sign_seen = False
    else:
      self.tlv_coast_moving_frames = 0
      self.tlv_coast_stop_sign_seen = False
    self.coast_cap_active = cap and not self.tlv_coast_released

  def _tlv_red_age_s(self, now: float):
    if self.tlv_red_confirmed_at is None:
      return None
    return max(0.0, now - self.tlv_red_confirmed_at)

  def _log_tlv_start_veto(self, sm, state_in: int, v_ego: float, left_intent: bool) -> None:
    """W74D instrument: one event per blocked TLV start edge, rate-limited."""
    now = time.monotonic()
    if now - self.tlv_log_veto_t < TLV_LOG_MIN_INTERVAL_S:
      return
    self.tlv_log_veto_t = now
    model_x = model_v = lead_d_rel = None
    try:
      model = sm['modelV2']
      model_x = float(model.position.x[-1])
      model_v = float(model.velocity.x[-1])
    except Exception:
      pass
    try:
      lead = sm['radarState'].leadOne
      lead_d_rel = float(lead.dRel) if lead.status else None
    except Exception:
      pass
    cloudlog.event("tlv_start_veto", state_in=int(state_in), v_ego=float(v_ego),
                   model_x=model_x, model_v=model_v, lead_d_rel=lead_d_rel,
                   left_intent=bool(left_intent), red_age_s=self._tlv_red_age_s(now))

  def _log_tlv_start_veto_cleared(self) -> None:
    now = time.monotonic()
    if now - self.tlv_log_clear_t < TLV_LOG_MIN_INTERVAL_S:
      return
    self.tlv_log_clear_t = now
    cloudlog.event("tlv_start_veto_cleared", delay_s=max(0.0, now - self.tlv_veto_started))

  def _log_tlv_coast_timeout(self, v_ego: float) -> None:
    now = time.monotonic()
    if now - self.tlv_log_coast_t < TLV_LOG_MIN_INTERVAL_S:
      return
    self.tlv_log_coast_t = now
    cloudlog.event("tlv_coast_timeout", v_ego=float(v_ego),
                   moving_s=float(self.tlv_coast_moving_frames * DT_MDL),
                   red_age_s=self._tlv_red_age_s(now))

  def cruise_eco_control(self, v_ego_kph, v_cruise_kph):
    v_cruise_kph_apply = v_cruise_kph
    if self.eco_over_speed > 0:
      if self.eco_target_speed > 0:
        if self.eco_target_speed < v_cruise_kph:
          self.eco_target_speed = v_cruise_kph
        elif self.eco_target_speed > v_cruise_kph:
          self.eco_target_speed = 0
      elif self.eco_target_speed == 0 and v_ego_kph + 3 < v_cruise_kph and v_cruise_kph > 20.0:  # 주행중 속도가 떨어지면 다시 크루즈연비제어 시작.
        self.eco_target_speed = v_cruise_kph

      if self.eco_target_speed != 0:  ## 크루즈 연비 제어모드 작동중일때: 연비제어 종료지점
        if v_ego_kph > self.eco_target_speed: # 설정속도를 초과하면..
          self.eco_target_speed = 0
        else:
          v_cruise_kph_apply = self.eco_target_speed + self.eco_over_speed  # + 설정 속도로 설정함.
    else:
      self.eco_target_speed = 0

    return v_cruise_kph_apply

  def add_event(self, event_name):
    now = time.time()
    if now - self.last_event_time > 5.0:
      self.events.add(event_name)
      self.last_event_time = now

  def _update_driving_mode(self, sm):
    self.drivingModeDetector.update_data(
      sm['carState'], sm['radarState'].leadOne, valid=sm.all_checks(['carState', 'radarState']),
    )
    if self.myDrivingModeAuto > 0 and not self.myDrivingMode_disable_auto:
      self.myDrivingMode = self.drivingModeDetector.get_mode(self.myDrivingModeAuto)

  def update(self, sm, v_cruise_kph, mode, *, signal_observation=None, signal_context=None,
             start_observation=None, start_context=None, prepare_observation=None, prepare_context=None):
    self._params_update()
    self._update_driving_mode(sm)
    self.leadAccelResponse = get_mode_lead_response(get_lead_response_for_gap(
      self.leadAccelResponseBase, self.leadAccelResponseTF, sm['selfdriveState'].personality,
    ), self.myDrivingMode)
    self._update_model_desire(sm)

    self.events = Events()
    carstate = sm['carState']
    vCluRatio = carstate.vCluRatio
    #controlsState = sm['controlsState']
    radarstate = sm['radarState']
    model = sm['modelV2']

    #self.soft_hold_active = sm['carControl'].hudControl.softHoldActive # carrot 1
    self.soft_hold_active = sm['carState'].softHoldActive # carrot 2

    mode_comfort_brake = self.comfortBrake * get_driving_mode_comfort_brake_factor(self.myDrivingMode)
    self.comfort_brake = mode_comfort_brake

    v_ego = carstate.vEgo
    a_ego = carstate.aEgo
    v_ego_kph = v_ego * CV.MS_TO_KPH
    v_ego_cluster = carstate.vEgoCluster
    v_ego_cluster_kph = v_ego_cluster * CV.MS_TO_KPH

    self.mySafeFactor, self.myTFollowFactor = get_driving_mode_factors(
      self.myDrivingMode, self.myEcoModeFactor, self.mySafeModeFactor,
    )


    v_cruise_kph = self.cruise_eco_control(v_ego_cluster_kph, v_cruise_kph)
    self._update_tlv_gas_cooldown(carstate)
    try:
      model_v2 = sm['modelV2']
    except Exception:
      model_v2 = None
    self._update_a4_lane_tracking(model_v2, v_ego_kph)
    v_cruise_kph, atc_active = self._update_carrot_man(sm, v_ego_kph, v_cruise_kph)
    
    #if atc_active and not self.atc_active and self.xState not in [XState.e2eStop, XState.e2eStopped, XState.lead]:
    #  if self.atcType in ["turn left", "turn right", "atc left", "atc right"]:
    #    self.xState = XState.e2ePrepare
    self.atc_active = atc_active

    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if vCluRatio > 0.5:
      v_cruise *= vCluRatio

    x = model.position.x
    y = model.position.y
    v = model.velocity.x

    self.fakeCruiseDistance = 0.0
    lead_detected = radarstate.leadOne.status # & radarstate.leadOne.radar

    self.xStop = self.update_stop_dist(x[31])
    stop_model_x_raw = self.xStop
    if self._stop_x_rl is None:
      self._stop_x_rl = stop_model_x_raw
    else:
      max_close = v_ego * DT_MDL + 0.5
      if stop_model_x_raw > self._stop_x_rl:
        self._stop_x_rl = stop_model_x_raw
      else:
        self._stop_x_rl = max(self._stop_x_rl - max_close, stop_model_x_raw)

    stop_model_x = self._stop_x_rl
    stop_model_x_rl = self._stop_x_rl

    trafficState_last = self.trafficState
    #self.check_model_stopping(v, v_ego, self.xStop, y)
    self.check_model_stopping(v_cruise, v, v_ego, a_ego, x[-1], y, radarstate.leadOne.dRel if lead_detected else 1000)

    if self.myDrivingMode == DrivingMode.High or self.trafficLightDetectMode == 0:
      self.trafficState = TrafficState.off

    # TLV red active (W74D: graded redLikely 4 or confirmed red 1, CONTRACT.md
    # High-speed approach): mode 2 only, coast (accel cap 0). The model start
    # sign does not release the cap; a stuck false red is bounded by the moving
    # time limit inside _update_coast_cap.
    self.tlv_red_active = self._read_tlv_red_active(sm)
    # W139c fast-road phantom gate: only the TLV red cap is gated (>= 22.2 m/s
    # needs detector support or a recent model stop); everything else unchanged.
    if self.tlv_red_active:
      self.tlv_red_active = self._apply_fast_gate(True, sm, v_ego)
    else:
      self._apply_fast_gate(False, sm, v_ego)
    self._update_coast_cap(v_ego)

    start_hold = False
    if self.start_guard is not None:
      if start_context is None and self._start_runtime is not None:
        start_observation, start_context = self._start_runtime.read(sm)
      context = start_context or {}
      self.start_decision = self.start_guard.update(
        context.get('now', time.monotonic()), start_observation,
        enabled=bool(context.get('enabled', False)), valid=bool(context.get('valid', False)),
        drive=bool(context.get('drive', False)), gas=carstate.gasPressed, speed=float(v_ego),
        departure=self.trafficState == TrafficState.green or lead_detected or
                  self.xState not in (XState.e2eStop, XState.e2eStopped))
      start_hold = self.start_decision.hold
      if self.start_decision.warn:
        # Bypass unrelated navigation-event cooldown. refuse.wav plays once.
        self.events.add(EventName.audioRefuse)
      if self._start_runtime is not None:
        self._start_runtime.record(self.start_decision, v_ego, stop_model_x_rl)

    signal_hold = False
    if self.signal_assist is not None:
      if signal_context is None and self._signal_runtime is not None:
        signal_observation, signal_context = self._signal_runtime.read(sm)
      context = signal_context or {}
      self.signal_decision = self.signal_assist.update(
        context.get('now', time.monotonic()), signal_observation,
        enabled=bool(context.get('enabled', False)) and self.myDrivingMode != DrivingMode.High and self.trafficLightDetectMode != 0,
        valid=bool(context.get('valid', False)), drive=bool(context.get('drive', False)),
        gas=carstate.gasPressed, speed=float(v_ego), model_distance=float(stop_model_x_rl),
        model_y=float(y[-1]), comfort_brake=float(mode_comfort_brake), lead=lead_detected,
        entry_allowed=is_traffic_stop_entry_allowed(carstate.steeringAngleDeg) and self.traffic_starting_count == 0,
        turning=carstate.leftBlinker or carstate.rightBlinker)
      signal_hold = self.signal_decision.hold
      if self.signal_decision.red_sign:
        self.trafficState = TrafficState.red
      if self._signal_runtime is not None:
        self._signal_runtime.record(self.signal_decision, v_ego, stop_model_x_rl)

    #self.update_user_control()
    if carstate.gasPressed or carstate.brakePressed:
      self.user_stop_distance = -1

    if signal_hold or start_hold:
      # Keep the stop even if a lead appears/moves. Radar obstacles remain in MPC.
      # Green only removes this override; it never manufactures a go request.
      self.xState = XState.e2eStopped
      self.actual_stop_distance = 0.0
      stop_model_x = 0.0
      self.stopping_count = max(self.stopping_count, 0.5 / DT_MDL)
      v_cruise = 0.0
      if start_hold:
        self.trafficState = TrafficState.red
    elif self.soft_hold_active > 0:
      self.xState = XState.e2eStopped
      if trafficState_last in [TrafficState.off, TrafficState.red] and self.trafficState == TrafficState.green:
        self.add_event(EventName.trafficSignChanged)
    elif self.xState == XState.e2eStopped:
      if carstate.gasPressed:
        self.xState = XState.e2eCruise #XState.e2ePrepare
      elif lead_detected and (radarstate.leadOne.dRel - stop_model_x_raw) < 2.0:
        self.xState = XState.lead
      elif self.stopping_count == 0:
        if self.trafficState == TrafficState.green and not self.carrot_stay_stop and not carstate.leftBlinker and self.trafficLightDetectMode != 1:
          #self.xState = XState.e2ePrepare
          self.xState = XState.e2eCruise  # 실험모드를 거치지 않고 바로 출발.
          self.add_event(EventName.trafficSignGreen)
      self.stopping_count = max(0, self.stopping_count - 1)
      v_cruise = 0
    elif self.xState == XState.e2eStop:
      self.stopping_count = 0
      if carstate.gasPressed:  # Stop detecting traffic signal for 10 seconds
        #self.xState = XState.e2ePrepare
        self.xState = XState.e2eCruise
        self.traffic_starting_count = 10.0 / DT_MDL
      elif lead_detected and (radarstate.leadOne.dRel - stop_model_x_raw) < 2.0:
        self.xState = XState.lead
      else:
        if self.trafficState == TrafficState.green:
          self.add_event(EventName.trafficSignGreen)
          self.xState = XState.e2eCruise
        else:
          self.comfort_brake = min(mode_comfort_brake, self.comfortBrake * 0.9)
          #self.comfort_brake = COMFORT_BRAKE
          self.trafficStopAdjustRatio = np.interp(v_ego_kph, [0, 100], [1.0, 0.7])
          # 속도가 높을수록 먼 정지거리 추정값을 줄여 보정함.
          stop_dist = stop_model_x_rl * np.interp(stop_model_x_rl, [0, 50], [1.0, self.trafficStopAdjustRatio])
          if stop_dist > 10.0:  # 10m 이상일 때만 실제 정지거리를 갱신함.
            self.actual_stop_distance = stop_dist
          stop_model_x = 0
          self.fakeCruiseDistance = 0 if self.actual_stop_distance > 10.0 else 10.0
          if v_ego < 0.3:
            self.stopping_count = 0.5 / DT_MDL
            self.xState = XState.e2eStopped
    elif self.xState == XState.e2ePrepare:
      if lead_detected:
        self.xState = XState.lead
      elif self.atc_active:
        if carstate.gasPressed:
          self.xState = XState.e2eCruise
      elif v_ego_kph < 5.0 and self.trafficState != TrafficState.green:
        self.xState = XState.e2eStop
        self.actual_stop_distance = 5.0 #2.0
      elif v_ego_kph > 5.0: # and stop_model_x > 30.0:
        self.xState = XState.e2eCruise
    else: #XState.lead, XState.cruise, XState.e2eCruise
      self.traffic_starting_count = max(0, self.traffic_starting_count - 1)
      if lead_detected:
        self.xState = XState.lead
      elif self.trafficState == TrafficState.red and is_traffic_stop_entry_allowed(carstate.steeringAngleDeg) and self.traffic_starting_count == 0:
        self.add_event(EventName.trafficStopping)
        self.xState = XState.e2eStop
        self.actual_stop_distance = stop_model_x_rl
      else:
        self.xState = XState.e2eCruise

    if self.trafficState in [TrafficState.off, TrafficState.green] or self.xState not in [XState.e2eStop, XState.e2eStopped]:
      stop_model_x = 1000.0

    if self.user_stop_distance >= 0 and not start_hold:
      self.user_stop_distance = max(0, self.user_stop_distance - v_ego * DT_MDL)
      self.actual_stop_distance = self.user_stop_distance
      self.xState = XState.e2eStop if self.user_stop_distance > 0 else XState.e2eStopped

    if start_hold:
      mode = 'acc'  # The blended path otherwise ignores the traffic-stop obstacle.
    elif mode == 'acc':
      mode = 'blended' if self.xState in [XState.e2ePrepare] else 'acc'

    # Drive modes shape transient response through lead preview and preserve
    # steady-state spacing through tFollow.  Keep the MPC braking-distance model
    # independent of mode so its v^2 term does not create a large high-speed gap
    # jump (Safe/Eco max acceleration still uses mySafeFactor).
    self.actual_stop_distance = max(0, self.actual_stop_distance - (v_ego * DT_MDL))

    if stop_model_x == 1000.0:  # e2eCruise 또는 lead 상태
      self.actual_stop_distance = 0.0
    elif self.actual_stop_distance > 0:  # e2eStop 또는 e2eStopped 상태
      stop_model_x = 0.0

    stopping_active = self.xState not in [XState.e2eStop, XState.e2eStopped]
    if not stopping_active:
      self._stop_x_rl = stop_model_x_raw

    # self.debugLongText = (
    #   f"XState({str(self.xState)})," +
    #   f"stop_x={stop_x:.1f}," +
    #   f"stopDist={self.actual_stop_distance:.1f}," +
    #   f"Traffic={str(self.trafficState)}"
    # )
    stop_dist =  stop_model_x + self.actual_stop_distance
    stop_dist = max(stop_dist, 0.0)

    stopping_active = self.xState in [XState.e2eStop, XState.e2eStopped]
    model_lead = model.leadsV3[0] if stopping_active and not lead_detected and len(model.leadsV3) > 0 else None
    model_lead_x = float(model_lead.x[0]) - RADAR_TO_CAMERA if model_lead is not None and len(model_lead.x) > 0 else np.nan
    model_lead_v = float(model_lead.v[0]) if model_lead is not None and len(model_lead.v) > 0 else np.nan
    model_lead_x_std = float(model_lead.xStd[0]) if model_lead is not None and len(model_lead.xStd) > 0 else np.nan
    model_lead_y_std = float(model_lead.yStd[0]) if model_lead is not None and len(model_lead.yStd) > 0 else np.nan
    model_lead_v_std = float(model_lead.vStd[0]) if model_lead is not None and len(model_lead.vStd) > 0 else np.nan
    self.trafficStopModelLeadOffset = self._traffic_stop_model_lead_matcher.update(
      stop_active=stopping_active and stop_dist < 300.0,
      allow_confirmation=self.trafficState == TrafficState.red and v_ego > 0.3,
      active_lead=lead_detected,
      stop_distance=stop_dist,
      lead_probability=float(model_lead.prob) if model_lead is not None else np.nan,
      lead_distance=model_lead_x,
      lead_velocity=model_lead_v,
      lead_x_std=model_lead_x_std,
      lead_y_std=model_lead_y_std,
      lead_v_std=model_lead_v_std,
    )

    if stopping_active and stop_dist < 300.0:
      stop_dist_soft = max(stop_dist - 1.0, 0.0)
      v_soft = float(np.sqrt(max(0.0, 2.0 * self.comfort_brake * stop_dist_soft)))
      v_cruise = min(v_cruise, v_soft)

    self.v_cruise = v_cruise
    self.stop_dist = stop_dist
    self.mode = mode
    if self.stop_prepare is not None:
      if prepare_context is None and self._prepare_runtime is not None:
        prepare_observation, prepare_context = self._prepare_runtime.read(sm)
      context = prepare_context or {}
      self.prepare_decision = self.stop_prepare.update(
        context.get('now', time.monotonic()), prepare_observation,
        enabled=bool(context.get('enabled', False)) and self.myDrivingMode != DrivingMode.High and self.trafficLightDetectMode != 0,
        valid=bool(context.get('valid', False)), drive=bool(context.get('drive', False)),
        speed=float(v_ego), gas=carstate.gasPressed, brake=carstate.brakePressed,
        lead=lead_detected or bool(getattr(getattr(radarstate, 'leadTwo', None), 'status', False)),
        turning=bool(context.get('turning', True)) or carstate.leftBlinker or carstate.rightBlinker,
        model_stopping=self.xState in (XState.e2eStop, XState.e2eStopped, XState.e2ePrepare),
        entry_allowed=self.traffic_starting_count == 0 and self.user_stop_distance < 0 and mode == 'acc',
        geometry=context.get('geometry', lambda box: None))
      if self.prepare_decision.active:
        self.xState = XState.e2eStopPrepare
        if self.prepare_decision.entered:
          self.events.add(EventName.trafficSignChanged)
      if self._prepare_runtime is not None:
        self._prepare_runtime.record_prepare(self.prepare_decision)
    #return v_cruise, stop_dist, mode

    return v_cruise_kph
