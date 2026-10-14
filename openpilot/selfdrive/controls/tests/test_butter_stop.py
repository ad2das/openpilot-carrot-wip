"""ButterStop: pure taper math, eligibility, LongControl integration and
ButterStop=0 output identity against the pre-change controller.
"""
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Windows unit-test hosts need the same import shims as the other control
# tests; the tests below never construct the real Params backend.
if sys.platform == "win32":
  hardware_module = ModuleType("openpilot.system.hardware")
  hardware_module.PC = True
  sys.modules.setdefault("openpilot.system.hardware", hardware_module)

  # swaglog imports Paths from this submodule; the fake hardware package has
  # no __path__, so the real hw.py cannot be found through it.
  hw_module = ModuleType("openpilot.system.hardware.hw")

  class _WinPaths:
    @staticmethod
    def comma_home() -> str:
      return os.path.join(tempfile.gettempdir(), ".comma")

    @staticmethod
    def swaglog_root() -> str:
      return os.path.join(_WinPaths.comma_home(), "log")

    @staticmethod
    def log_root() -> str:
      return os.path.join(_WinPaths.comma_home(), "realdata")

    @staticmethod
    def persist_root() -> str:
      return os.path.join(_WinPaths.comma_home(), "persist")

    @staticmethod
    def stats_root() -> str:
      return os.path.join(_WinPaths.comma_home(), "stats")

    @staticmethod
    def swaglog_ipc() -> str:
      return "ipc:///tmp/logmessage"

    @staticmethod
    def download_cache_root() -> str:
      return os.path.join(tempfile.gettempdir(), "comma_download_cache") + "/"

  hw_module.Paths = _WinPaths
  sys.modules.setdefault("openpilot.system.hardware.hw", hw_module)

  params_module = ModuleType("openpilot.common.params")
  params_module.Params = object
  sys.modules.setdefault("openpilot.common.params", params_module)

import openpilot.selfdrive.controls.lib.longcontrol as lc
from openpilot.selfdrive.controls.lib.butter_stop import (
  A_END, CREEP_ACCEL, HOLD_RAMP, K_TAPER, MARGIN,
  MIN_FINAL_GAP_ABS, MIN_FINAL_GAP_SLACK, STANDSTILL_T, STANDSTILL_V,
  V_TAPER, butter_eligible, butter_floor, plan_stop_distance,
)
from openpilot.selfdrive.controls.lib.longcontrol import (
  CONTROL_N_T_IDX, LongControl, LongCtrlState,
)

PLAN_T = np.asarray(CONTROL_N_T_IDX, dtype=float)


class DictParams:
  def __init__(self, values):
    self.values = values

  def get_int(self, name):
    return int(self.values.get(name, 0))

  def get_float(self, name):
    return float(self.values.get(name, 0))

  def get_bool(self, name):
    return bool(self.values.get(name, False))

  def put_int(self, name, value):
    self.values[name] = value


def make_cp(brand="hyundai", starting_state=False):
  return SimpleNamespace(
    brand=brand, flags=0, openpilotLongitudinalControl=True,
    longitudinalTuning=SimpleNamespace(kpBP=[0.0], kpV=[1.0], kiBP=[0.0], kiV=[0.2], kf=1.0),
    startingState=starting_state, vEgoStarting=0.5,
    stopAccel=-2.0, stoppingDecelRate=0.8, startAccel=1.0,
  )


def make_control(monkeypatch, values=None, brand="hyundai", starting_state=False):
  stored = {"StoppingAccel": -50, "StopDistanceCarrot": 550, "ButterStop": 1,
            "LongTuningKpV": 100, "LongTuningKiV": 200, "LongTuningKf": 100}
  if values:
    stored.update(values)
  params = DictParams(stored)
  monkeypatch.setattr(lc, "Params", lambda: params)
  control = LongControl(make_cp(brand, starting_state))
  return control, params


def stopping_inputs(*, v_ego=1.0, standstill=False, plan_speeds=None, lead_status=False,
                    d_rel=30.0, a_lead=0.0, v_lead=0.0):
  cs = SimpleNamespace(vEgo=v_ego, aEgo=-0.5, brakePressed=False, gasPressed=False,
                       softHoldActive=0, standstill=standstill,
                       cruiseState=SimpleNamespace(standstill=False), vCruise=0.0, carrotCruise=0)
  speeds = plan_speeds if plan_speeds is not None else np.maximum(0.0, v_ego - 1.0 * PLAN_T)
  plan = SimpleNamespace(aTarget=-1.0, vTargetNow=0.0, jTargetNow=0.0, shouldStop=True, fcw=False,
                         speeds=speeds, longitudinalPlanSource="lead0", cruiseTarget=0.0,
                         cruiseCoastingTarget=0.0, cruiseCoastingPercent=0)
  lead = SimpleNamespace(status=lead_status, dRel=d_rel, aLeadK=a_lead, vLead=v_lead)
  radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False),
                          leadCutInRisk=SimpleNamespace(status=False))
  return cs, plan, radar


# ---------------------------------------------------------------------------
# butter_floor math
# ---------------------------------------------------------------------------

def test_plan_stop_distance_trapezoid_and_none():
  t = np.array([0.0, 1.0, 2.0, 3.0])
  v = np.array([4.0, 3.0, 2.0, 0.04])
  # 4->3 over 1s = 3.5, 3->2 = 2.5, 2->0.04 ~ 1.02
  assert plan_stop_distance(t, v) == pytest.approx(3.5 + 2.5 + 1.02)
  assert plan_stop_distance(t, np.array([4.0, 3.0, 2.0, 1.0])) is None
  assert plan_stop_distance(t, np.array([4.0, np.nan, 2.0, 0.0])) is None
  assert plan_stop_distance(t[:1], v[:1]) is None


def test_floor_is_more_negative_of_taper_and_kinematic_limit():
  v_ego = 1.0
  t = np.array([0.0, 0.5, 1.0])
  # Short plan (0.5 m of stop distance): the kinematic limit dominates.
  short = np.array([1.0, 0.5, 0.0])
  avail = plan_stop_distance(t, short) - MARGIN
  assert avail == pytest.approx(0.5 - MARGIN)
  expected = min(-(A_END + K_TAPER * v_ego), -(v_ego ** 2) / (2.0 * avail))
  assert expected == pytest.approx(-(v_ego ** 2) / (2.0 * avail))
  assert butter_floor(v_ego, t, short, SimpleNamespace(status=False), 5.5) == pytest.approx(expected)
  # Longer plan (1.5 m): the taper itself is the more negative limit.
  longer = np.array([3.0, 1.5, 0.0])
  avail_long = plan_stop_distance(t, longer) - MARGIN
  expected_long = min(-(A_END + K_TAPER * v_ego), -(v_ego ** 2) / (2.0 * avail_long))
  assert expected_long == pytest.approx(-(A_END + K_TAPER * v_ego))
  assert butter_floor(v_ego, t, longer, SimpleNamespace(status=False), 5.5) == pytest.approx(expected_long)


def test_floor_taper_gets_gentler_close_to_standstill():
  speeds = np.maximum(0.0, 3.0 - 2.0 * PLAN_T)  # stops within ~2.3 m
  floor_far = butter_floor(2.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5)
  floor_near = butter_floor(0.2, PLAN_T, speeds, SimpleNamespace(status=False), 5.5)
  assert floor_far < floor_near < 0.0


def test_floor_returns_none_without_a_planned_stop():
  speeds = np.full(len(PLAN_T), 5.0)
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5) is None


def test_floor_lead_bound_and_absolute_gap():
  speeds = np.maximum(0.0, 2.0 * (1.0 - PLAN_T / 2.0))
  lead = SimpleNamespace(status=True, dRel=10.0)
  bound = 10.0 - max(MIN_FINAL_GAP_ABS, 5.5 - MIN_FINAL_GAP_SLACK)
  avail = min(plan_stop_distance(PLAN_T, speeds) - MARGIN, bound)
  expected = min(-(A_END + K_TAPER * 1.0), -(1.0 ** 2) / (2.0 * avail))
  assert butter_floor(1.0, PLAN_T, speeds, lead, 5.5) == pytest.approx(expected)
  # dRel at the lead bound -> no floor at all.
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=True, dRel=5.5 - MIN_FINAL_GAP_SLACK), 5.5) is None
  # A small configured stop distance keeps the 3.0 m absolute minimum.
  strong = SimpleNamespace(status=True, dRel=6.0)
  assert butter_floor(1.0, PLAN_T, speeds, strong, 1.0) == pytest.approx(
    min(-(A_END + K_TAPER * 1.0), -(1.0 ** 2) / (2.0 * (6.0 - MIN_FINAL_GAP_ABS))))


def test_floor_invalid_inputs_and_stop_distance():
  speeds = np.maximum(0.0, 1.0 - PLAN_T)
  lead = SimpleNamespace(status=True, dRel=float("nan"))
  assert butter_floor(1.0, PLAN_T, speeds, lead, 5.5) is None
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=False), float("nan")) is None
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=False), -1.0) is None
  assert butter_floor(0.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5) is None
  assert butter_floor(1.0, PLAN_T, np.full(3, 0.0), SimpleNamespace(status=False), 5.5) is None


# ---------------------------------------------------------------------------
# eligibility gates
# ---------------------------------------------------------------------------

def eligible_kwargs(**overrides):
  kwargs = {"active": True, "v_ego": 1.0, "brake_pressed": False, "gas_pressed": False, "fcw": False,
                "t_since_plan": 0.05, "lead": SimpleNamespace(status=False)}
  kwargs.update(overrides)
  return kwargs


def test_eligibility_passes_when_all_gates_hold():
  assert butter_eligible(**eligible_kwargs())
  assert butter_eligible(**eligible_kwargs(lead=SimpleNamespace(
    status=True, dRel=12.0, aLeadK=-0.5, vLead=0.0)))


@pytest.mark.parametrize("gate", ["inactive", "too_slow", "too_fast", "brake", "gas", "fcw",
                                  "stale_negative", "stale_future", "lead_close", "lead_decel",
                                  "lead_closing", "nan_speed"])
def test_each_eligibility_gate_disables(gate):
  kwargs = eligible_kwargs()
  if gate == "inactive":
    kwargs["active"] = False
  elif gate == "too_slow":
    kwargs["v_ego"] = STANDSTILL_V
  elif gate == "too_fast":
    kwargs["v_ego"] = V_TAPER
  elif gate == "brake":
    kwargs["brake_pressed"] = True
  elif gate == "gas":
    kwargs["gas_pressed"] = True
  elif gate == "fcw":
    kwargs["fcw"] = True
  elif gate == "stale_negative":
    kwargs["t_since_plan"] = -0.01
  elif gate == "stale_future":
    kwargs["t_since_plan"] = 0.201
  elif gate == "lead_close":
    kwargs["lead"] = SimpleNamespace(status=True, dRel=2.9, aLeadK=0.0, vLead=0.0)
  elif gate == "lead_decel":
    kwargs["lead"] = SimpleNamespace(status=True, dRel=20.0, aLeadK=-1.6, vLead=0.0)
  elif gate == "lead_closing":
    kwargs["lead"] = SimpleNamespace(status=True, dRel=2.0, aLeadK=0.0, vLead=0.0)
    kwargs["v_ego"] = 2.0  # 2.0 m / 2.0 m/s = 1.0 s < 2.5 s
  elif gate == "nan_speed":
    kwargs["v_ego"] = float("nan")
  assert not butter_eligible(**kwargs)


# ---------------------------------------------------------------------------
# LongControl integration
# ---------------------------------------------------------------------------

def test_stopping_state_floor_limits_braking_and_does_not_accumulate(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  control.last_output_accel = -2.0
  cs, plan, radar = stopping_inputs(v_ego=1.0)
  floor = butter_floor(1.0, PLAN_T, plan.speeds, radar.leadOne, 5.5)
  for _ in range(10):
    accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    assert accel == pytest.approx(floor)
  # The stored value is the floored one, so the one-way ramp never drills below.
  assert control.last_output_accel == pytest.approx(floor)


def test_stopping_state_without_floor_keeps_original_ramp(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  control.last_output_accel = -0.2
  # Non-finite plan disables the floor; the original ramp still runs.
  cs, plan, radar = stopping_inputs(v_ego=1.0, plan_speeds=np.array([np.nan]))
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel == pytest.approx(-0.2 - control.CP.stoppingDecelRate * lc.DT_CTRL)


def test_standstill_latch_and_faster_hold_ramp(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  # CS.standstill confirms immediately.
  cs, plan, radar = stopping_inputs(v_ego=0.03, standstill=True)
  control.last_output_accel = 0.0
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._butter_standstill_confirmed
  assert accel == pytest.approx(-HOLD_RAMP * lc.DT_CTRL)
  # After confirmation the floor is gone even though a taper would exist. Gas
  # holds the creep guard latch off so the ramp rate itself is visible.
  cs.vEgo = 0.3
  cs.standstill = False
  cs.gasPressed = True
  control.last_output_accel = -0.4
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  taper_plan = np.maximum(0.0, 2.0 * (1.0 - PLAN_T / 2.0))
  floor_would_be = butter_floor(0.3, PLAN_T, taper_plan, radar.leadOne, 5.5)
  assert floor_would_be is not None and floor_would_be > -0.4
  assert not control._butter_creep_latched
  assert accel == pytest.approx(-0.4 - HOLD_RAMP * lc.DT_CTRL)


def test_standstill_confirm_requires_continuous_time(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.03, standstill=False)
  for _ in range(int(STANDSTILL_T / lc.DT_CTRL) - 1):
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    assert not control._butter_standstill_confirmed
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._butter_standstill_confirmed
  # A brief roll above standstill resets the unconfirmed timer.
  control2, _ = make_control(monkeypatch)
  control2.long_control_state = LongCtrlState.stopping
  for _ in range(15):
    control2.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  cs_roll = SimpleNamespace(**vars(cs))
  cs_roll.vEgo = 0.2
  control2.update(True, cs_roll, plan, (-3.5, 2.0), 0.05, radar)
  assert not control2._butter_standstill_confirmed


def test_soft_hold_path_is_unchanged_with_butter_on(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.03, standstill=True)
  cs.softHoldActive = 1
  for _ in range(5):
    accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    assert accel == pytest.approx(control.CP.stopAccel)


def test_creep_guard_fires_latches_and_logs_once(monkeypatch):
  events = []
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: events.append((a, k))))
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.03, standstill=True)
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._butter_standstill_confirmed
  # Creep appears after the standstill was confirmed.
  cs.vEgo = 0.2
  control.last_output_accel = -0.1
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel == pytest.approx(CREEP_ACCEL)
  assert control._butter_creep_latched and len(events) == 1
  # The latch persists for the rest of the episode, including below CREEP_V.
  cs.vEgo = 0.03
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel == pytest.approx(CREEP_ACCEL)
  assert len(events) == 1
  # Leaving stopping ends the episode; the guard resets.
  plan.shouldStop = False
  cs.cruiseState.standstill = False
  control.long_control_state = LongCtrlState.stopping
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert not control._butter_creep_latched
  assert control.long_control_state == LongCtrlState.pid


def test_creep_guard_does_not_fight_gas_before_latch(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=0.03, standstill=True)
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  control.last_output_accel = -0.1
  cs.vEgo = 0.2
  cs.gasPressed = True
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert not control._butter_creep_latched
  assert accel == pytest.approx(-0.1 - HOLD_RAMP * lc.DT_CTRL)


def test_pid_state_floor_limits_without_adding_braking(monkeypatch):
  control, _ = make_control(monkeypatch, brand="gm")
  control.long_control_state = LongCtrlState.pid
  cs, plan, radar = stopping_inputs(v_ego=1.0)
  cs.aEgo = 0.0
  plan.shouldStop = False
  plan.aTarget = -2.0
  plan.vTargetNow = 0.0
  floor = butter_floor(1.0, PLAN_T, plan.speeds, radar.leadOne, 5.5)
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control.long_control_state == LongCtrlState.pid
  assert accel == pytest.approx(floor)
  assert accel < 0.0
  # A softer plan whose PID output already sits above the floor is untouched.
  plan.vTargetNow = 0.5
  plan.aTarget = -0.05
  accel2, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel2 > floor


def test_butter_stop_defaults_off_when_param_missing(monkeypatch):
  monkeypatch.setattr(lc, "Params", lambda: DictParams({"StoppingAccel": -50}))
  control = lc.LongControl(make_cp())
  assert control.butter_stop is False


# ---------------------------------------------------------------------------
# ButterStop=0 identity against the pre-change controller (W84 code path)
# ---------------------------------------------------------------------------

def legacy_update(control, active, CS, long_plan, accel_limits, t_since_plan, radarState):
  """Verbatim pre-ButterStop LongControl.update body (80a20196f)."""
  soft_hold_active = CS.softHoldActive > 0
  a_target_ff = long_plan.aTarget
  v_target_now = long_plan.vTargetNow
  j_target_now = long_plan.jTargetNow
  should_stop = long_plan.shouldStop

  control.readParamCount += 1
  if control.readParamCount >= 100:
    control.readParamCount = 0
    control._refresh_stopping_accel()
  elif control.readParamCount == 10:
    control._refresh_longitudinal_tuning()

  control.pid.neg_limit = accel_limits[0]
  control.pid.pos_limit = accel_limits[1]

  control.long_control_state = lc.long_control_state_trans(control.CP, active, control.long_control_state, CS.vEgo,
                                                           should_stop, CS.brakePressed,
                                                           CS.cruiseState.standstill, CS.aEgo, control.stopping_accel, radarState)
  if active and soft_hold_active:
    control.long_control_state = LongCtrlState.stopping

  if control.long_control_state == LongCtrlState.off:
    control.reset()
    output_accel = 0.

  elif control.long_control_state == LongCtrlState.stopping:
    output_accel = control.last_output_accel

    if soft_hold_active:
      output_accel = control.CP.stopAccel
    # Restore the original one-way ramp. Do not unwind stronger braking.
    if output_accel > control.stopping_accel:
      output_accel = min(output_accel, 0.0)
      output_accel -= control.CP.stoppingDecelRate * lc.DT_CTRL
    control.reset()

  elif control.long_control_state == LongCtrlState.starting:
    output_accel = control.CP.startAccel
    control.reset()

  else:  # LongCtrlState.pid
    target = getattr(long_plan, 'cruiseCoastingTarget', 0.0)
    percent = getattr(long_plan, 'cruiseCoastingPercent', 0)
    relief = 0.0
    if (target > 0.0 and percent > 0 and control.CP.openpilotLongitudinalControl and
        0.0 <= t_since_plan <= lc.MAX_PLAN_AGE and
        long_plan.longitudinalPlanSource == 'cruise' and not long_plan.fcw and not should_stop and
        not CS.brakePressed and not CS.gasPressed and not CS.carrotCruise and
        abs(long_plan.cruiseTarget - CS.vCruise) < 0.001 and
        not CS.cruiseState.standstill and lc.no_coasting_lead(radarState) and
        np.isfinite(CS.aEgo) and np.isfinite(a_target_ff) and np.isfinite(v_target_now) and
        accel_limits[1] >= 0.0):
      relief = lc.coasting_relief(CS.vEgo, target, percent)
    if relief == 0.0:
      control.coasting.reset()
    if control.use_accel_pid:
      error = a_target_ff - CS.aEgo
    else:
      error = v_target_now - CS.vEgo
    previous_integral = control.pid.i
    output_accel = control.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff)
    if relief > 0.0 and output_accel < 0.0:
      control.pid.i = previous_integral
      output_accel = control.pid.update(error, speed=CS.vEgo, feedforward=a_target_ff, freeze_integrator=True)
      output_accel = control.coasting.apply(output_accel, relief, lc.DT_CTRL)
    else:
      control.coasting.reset()

  control.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
  return control.last_output_accel, a_target_ff, j_target_now


def _scenario_stream(frames=2000):
  """Deterministic stop / hold / soft-hold / release / cruise / off sequence."""
  v_profile = np.zeros(frames + 1)
  v_profile[0:100] = 22.0
  v_profile[100:701] = np.linspace(22.0, 0.0, 601)
  v_profile[701:1100] = 0.0
  v_profile[1100:1300] = np.linspace(0.0, 8.0, 200)
  v_profile[1300:1600] = 29.0
  v_profile[1600:1650] = np.linspace(29.0, 12.0, 50)
  v_profile[1650:1820] = np.linspace(12.0, 0.0, 170)
  v_profile[1820:1900] = 3.0
  v_profile[1900:1940] = 0.5
  v_profile[1940:] = 0.5
  return v_profile[:frames], 0.05


def _frame_inputs(frame, v_profile):
  dt = lc.DT_CTRL
  v_ego = float(v_profile[frame])
  v_next = float(v_profile[min(frame + 1, len(v_profile) - 1)])
  a_ego = (v_next - v_ego) / dt
  standstill = v_ego < 0.01
  should_stop = (100 <= frame < 1100) or (1650 <= frame < 1990)
  source = "cruise" if 1300 <= frame < 1600 else "lead0"
  cruise_target = 100.0 if source == "cruise" else 0.0
  coasting_target = 100.0 / 3.6 if source == "cruise" else 0.0
  coasting_percent = 5 if source == "cruise" else 0
  speeds = np.maximum(0.0, v_ego - 2.0 * PLAN_T)
  if standstill:
    speeds = np.zeros(len(PLAN_T))
  lead_status = should_stop
  lead = SimpleNamespace(status=lead_status, dRel=max(3.5, v_ego * 2.0), aLeadK=0.0, vLead=0.0,
                         radarTrackId=7 if lead_status else -1)
  cs = SimpleNamespace(
    vEgo=v_ego, aEgo=a_ego, brakePressed=1900 <= frame < 1940, gasPressed=1820 <= frame < 1900,
    softHoldActive=1 if 800 <= frame < 950 else 0, standstill=standstill,
    cruiseState=SimpleNamespace(standstill=v_ego == 0.0 and not should_stop),
    vCruise=cruise_target, carrotCruise=0)
  plan = SimpleNamespace(aTarget=-1.0 if should_stop else 0.2, vTargetNow=0.0 if should_stop else 27.78,
                         jTargetNow=0.0, shouldStop=should_stop, fcw=False, speeds=speeds,
                         longitudinalPlanSource=source, cruiseTarget=cruise_target,
                         cruiseCoastingTarget=coasting_target, cruiseCoastingPercent=coasting_percent)
  radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False),
                          leadCutInRisk=SimpleNamespace(status=False))
  active = not (1600 <= frame < 1650)
  return active, cs, plan, radar


@pytest.mark.parametrize("starting_state", [False, True])
def test_butter_off_replays_identically_to_pre_change_controller(monkeypatch, starting_state):
  control, _ = make_control(monkeypatch, values={"ButterStop": 0}, starting_state=starting_state)
  legacy, _ = make_control(monkeypatch, values={"ButterStop": 0}, starting_state=starting_state)
  assert not control.butter_stop
  v_profile, age = _scenario_stream()
  for frame in range(len(v_profile)):
    active, cs, plan, radar = _frame_inputs(frame, v_profile)
    expected = legacy_update(legacy, active, cs, plan, (-3.5, 2.0), age, radar)
    actual = control.update(active, cs, plan, (-3.5, 2.0), age, radar)
    assert actual[0] == expected[0] and actual[1] == expected[1] and actual[2] == expected[2]
    assert control.long_control_state == legacy.long_control_state
  assert control.long_control_state == legacy.long_control_state


def test_state_sequence_identical_on_vs_off_but_output_differs(monkeypatch):
  # The creep guard's one-time log event must not reach the real zmq logger.
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  off, _ = make_control(monkeypatch, values={"ButterStop": 0})
  on, _ = make_control(monkeypatch, values={"ButterStop": 1})
  v_profile, age = _scenario_stream()
  outputs_differ = False
  for frame in range(len(v_profile)):
    active, cs, plan, radar = _frame_inputs(frame, v_profile)
    off_out, _, _ = off.update(active, cs, plan, (-3.5, 2.0), age, radar)
    on_out, _, _ = on.update(active, cs, plan, (-3.5, 2.0), age, radar)
    assert on.long_control_state == off.long_control_state
    if on_out != off_out:
      outputs_differ = True
  assert outputs_differ, "ButterStop=1 must change the stopping output somewhere"
