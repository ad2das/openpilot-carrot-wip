"""ButterStop: pure taper math, eligibility, the lead-position filter, floor
slew, LongControl integration and ButterStop=0 output identity against the
pre-change controller.
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

# Stub casadi and the native solver for the long_mpc import; the tests only
# exercise the pure stop_distance shift, never the solver itself.
if "casadi" not in sys.modules:
  casadi_module = ModuleType("casadi")
  casadi_module.SX = lambda *a, **k: None
  casadi_module.vertcat = lambda *a, **k: None
  sys.modules.setdefault("casadi", casadi_module)

if "acados" not in sys.modules:
  acados_module = ModuleType("acados")
  acados_template_module = ModuleType("acados.acados_template")

  class _StubAcadosOcp:
    pass

  class _StubAcadosModel:
    pass

  acados_template_module.AcadosModel = _StubAcadosModel
  acados_template_module.AcadosOcp = _StubAcadosOcp
  acados_template_module.AcadosOcpSolver = None
  sys.modules.setdefault("acados", acados_module)
  sys.modules.setdefault("acados.acados_template", acados_template_module)

_generated_module = ModuleType("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code")
_generated_module.__path__ = []
_solver_module = ModuleType("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx")
_solver_module.AcadosOcpSolverCython = None
sys.modules.setdefault("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code", _generated_module)
sys.modules.setdefault("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx", _solver_module)

import openpilot.selfdrive.controls.lib.longcontrol as lc
from openpilot.selfdrive.controls.lib import butter_stop
from openpilot.selfdrive.controls.lib.butter_stop import (
  A_MAX, A_MIN, CREEP_ACCEL, D_EXTRA, FLOOR_RELAX_JERK, HOLD_RAMP, K_TAPER,
  LEAD_JUMP_M, LEAD_RAW_ALLOW, LEAD_SLACK, MARGIN, MIN_FINAL_GAP_ABS, STANDSTILL_T,
  STANDSTILL_V, V_TAPER, LeadPosFilter, butter_eligible, butter_floor,
  plan_stop_distance, slew_floor, solve_taper_a_end, taper_stop_distance,
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
# taper math
# ---------------------------------------------------------------------------

def _euler_taper_stop_distance(v0, a_end, k, dt=1e-4):
  """Numeric integration of a(v) = -(a_end + k*v) to standstill."""
  v = float(v0)
  s = 0.0
  while v > 0.0:
    a = -(a_end + k * v)
    s += v * dt
    v = max(0.0, v + a * dt)
  return s


def test_taper_stop_distance_matches_numeric_integration():
  for v in (0.2, 1.0, 2.0, 3.0):
    for a_end in (0.1, 0.25, 0.5, 1.0):
      for k in (0.0, 0.3, 0.4, 0.5):
        expected = _euler_taper_stop_distance(v, a_end, k)
        got = taper_stop_distance(v, a_end, k)
        assert got == pytest.approx(expected, rel=0.01), (v, a_end, k)


def test_taper_stop_distance_k_zero_is_constant_decel():
  assert taper_stop_distance(2.0, 0.5, 0.0) == pytest.approx(2.0 ** 2 / (2.0 * 0.5))


def test_taper_stop_distance_decreases_in_a_end():
  for v in (0.2, 1.0, 2.0, 3.0):
    for k in (0.0, 0.3, 0.4, 0.5):
      dists = [taper_stop_distance(v, a_end, k) for a_end in (0.1, 0.25, 0.5, 1.0)]
      for earlier, later in zip(dists, dists[1:], strict=False):
        assert later < earlier, (v, k)


def test_solve_taper_a_end_safety_property():
  for v in np.linspace(0.06, 2.99, 30):
    for avail in np.linspace(0.06, 20.0, 40):
      for k in (0.3, 0.4, 0.5):
        a_end = solve_taper_a_end(float(v), float(avail), k)
        if a_end is None:
          assert taper_stop_distance(float(v), A_MAX, k) > float(avail) + 1e-9
        else:
          assert A_MIN <= a_end <= A_MAX
          assert taper_stop_distance(float(v), a_end, k) <= float(avail) + 1e-9


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


def test_floor_uses_plan_distance_plus_extra():
  v_ego = 1.0
  t = np.array([0.0, 0.5, 1.0])
  short = np.array([3.0, 1.5, 0.0])
  dist = plan_stop_distance(t, short)
  avail = dist + D_EXTRA - MARGIN
  expected = -(solve_taper_a_end(v_ego, avail, K_TAPER) + K_TAPER * v_ego)
  assert butter_floor(v_ego, t, short, SimpleNamespace(status=False), 5.5) == pytest.approx(expected)


def test_floor_gets_gentler_close_to_standstill():
  speeds = np.maximum(0.0, 3.0 - 2.0 * PLAN_T)  # stops within ~2.3 m
  floor_far = butter_floor(2.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5)
  floor_near = butter_floor(0.2, PLAN_T, speeds, SimpleNamespace(status=False), 5.5)
  assert floor_far < floor_near < 0.0


def test_floor_returns_none_without_a_planned_stop():
  speeds = np.full(len(PLAN_T), 5.0)
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5) is None


def test_floor_returns_none_when_taper_cannot_fit():
  # 3 m/s comes to a stop in less than the A_MAX taper needs.
  speeds = np.array([3.0, 0.0])
  t = np.array([0.0, 0.1])
  assert butter_floor(3.0, t, speeds, SimpleNamespace(status=False), 5.5) is None


def test_floor_lead_bound_stationary_lead():
  speeds = np.maximum(0.0, 2.0 * (1.0 - PLAN_T / 2.0))
  lead = SimpleNamespace(status=True, dRel=10.0)
  bound = 10.0 - max(MIN_FINAL_GAP_ABS, 5.5 - LEAD_SLACK)
  avail = min(plan_stop_distance(PLAN_T, speeds) + D_EXTRA - MARGIN, bound)
  expected = -(solve_taper_a_end(1.0, avail, K_TAPER) + K_TAPER * 1.0)
  floor = butter_floor(1.0, PLAN_T, speeds, lead, 5.5)
  assert floor == pytest.approx(expected)
  taper = taper_stop_distance(1.0, -(floor + K_TAPER * 1.0), K_TAPER) if floor is not None else None
  assert taper <= bound + 1e-9
  # dRel inside the lead bound -> no floor at all.
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=True, dRel=5.5 - LEAD_SLACK), 5.5) is None
  # A small configured stop distance keeps the 3.0 m absolute minimum.
  strong = SimpleNamespace(status=True, dRel=6.0)
  strong_avail = 6.0 - MIN_FINAL_GAP_ABS
  assert butter_floor(1.0, PLAN_T, speeds, strong, 1.0) == pytest.approx(
    -(solve_taper_a_end(1.0, strong_avail, K_TAPER) + K_TAPER * 1.0))


def test_floor_lead_bound_safety_grid():
  rng = np.random.default_rng(1)
  for v in np.linspace(0.06, 2.99, 12):
    for d_rel in np.linspace(3.5, 15.0, 12):
      for stop_distance_orig in (3.0, 5.5, 7.0):
        speeds = np.maximum(0.0, v * (1.0 - PLAN_T / 3.0))
        lead = SimpleNamespace(status=True, dRel=float(d_rel))
        floor = butter_floor(float(v), PLAN_T, speeds, lead, stop_distance_orig)
        if floor is None:
          continue
        limit = d_rel - max(MIN_FINAL_GAP_ABS, stop_distance_orig - LEAD_SLACK)
        a_end = -(floor + K_TAPER * v)
        assert taper_stop_distance(float(v), a_end, K_TAPER) <= limit + 1e-9
  _ = rng


def test_floor_returns_none_when_lead_bound_leaves_no_avail():
  # avail below AVAIL_MIN when the lead bound eats the whole slack.
  speeds = np.maximum(0.0, 1.0 - PLAN_T)
  lead = SimpleNamespace(status=True, dRel=MIN_FINAL_GAP_ABS + 0.01)
  assert butter_floor(1.0, PLAN_T, speeds, lead, MIN_FINAL_GAP_ABS) is None


def test_floor_invalid_inputs_and_stop_distance():
  speeds = np.maximum(0.0, 1.0 - PLAN_T)
  lead = SimpleNamespace(status=True, dRel=float("nan"))
  assert butter_floor(1.0, PLAN_T, speeds, lead, 5.5) is None
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=False), float("nan")) is None
  assert butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=False), -1.0) is None
  assert butter_floor(0.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5) is None
  assert butter_floor(1.0, PLAN_T, np.full(3, 0.0), SimpleNamespace(status=False), 5.5) is None


def test_floor_d_rel_eff_overrides_raw_lead():
  # Raw lead is far (30 m) while the smoothed d_eff does not: the returned
  # floor must use d_eff, not the raw dRel, and be strictly tighter than the
  # plan-only floor. d_eff is chosen so the lead bound is the binding term.
  t = np.array([0.0, 0.5])
  speeds = np.array([2.0, 0.0])
  plan_avail = plan_stop_distance(t, speeds) + D_EXTRA - MARGIN
  lead = SimpleNamespace(status=True, dRel=30.0)
  floor_raw = butter_floor(1.0, t, speeds, lead, 5.5)
  assert floor_raw == pytest.approx(-(solve_taper_a_end(1.0, plan_avail, K_TAPER) + K_TAPER))
  lead_offset = max(MIN_FINAL_GAP_ABS, 5.5 - LEAD_SLACK)
  d_eff = plan_avail + lead_offset - 0.2  # lead bound tighter than the plan
  bound = d_eff - lead_offset
  avail = min(plan_avail, bound)
  assert avail == pytest.approx(bound)
  floor_eff = butter_floor(1.0, t, speeds, lead, 5.5, d_rel_eff=d_eff)
  assert floor_eff == pytest.approx(-(solve_taper_a_end(1.0, avail, K_TAPER) + K_TAPER))
  assert floor_eff < floor_raw
  assert taper_stop_distance(1.0, -(floor_eff + K_TAPER), K_TAPER) <= bound + 1e-9


def test_floor_d_rel_eff_safety_grid():
  for v in np.linspace(0.06, 2.99, 12):
    for d_eff in np.linspace(3.5, 15.0, 12):
      for stop_distance_orig in (3.0, 5.5, 7.0):
        speeds = np.maximum(0.0, v * (1.0 - PLAN_T / 3.0))
        lead = SimpleNamespace(status=True, dRel=3.0)
        floor = butter_floor(float(v), PLAN_T, speeds, lead, stop_distance_orig, d_rel_eff=float(d_eff))
        if floor is None:
          continue
        limit = d_eff - max(MIN_FINAL_GAP_ABS, stop_distance_orig - LEAD_SLACK)
        a_end = -(floor + K_TAPER * v)
        assert taper_stop_distance(float(v), a_end, K_TAPER) <= max(limit, 0.0) + 1e-9


def test_floor_diag_records_binding_term_and_a_end():
  speeds = np.maximum(0.0, 2.0 * (1.0 - PLAN_T / 2.0))
  diag = {}
  floor = butter_floor(1.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5, diag=diag)
  assert floor is not None
  assert diag["bound"] == "plan"
  assert diag["a_end"] == pytest.approx(-(floor + K_TAPER * 1.0))
  diag2 = {}
  lead = SimpleNamespace(status=True, dRel=6.0)
  assert butter_floor(1.0, PLAN_T, speeds, lead, 5.5, diag=diag2) is not None
  assert diag2["bound"] == "lead"
  # None result still clears the diag fields.
  diag3 = {"bound": "plan", "a_end": 0.5}
  assert butter_floor(0.0, PLAN_T, speeds, SimpleNamespace(status=False), 5.5, diag=diag3) is None
  assert diag3 == {"bound": None, "a_end": None}


# ---------------------------------------------------------------------------
# LeadPosFilter
# ---------------------------------------------------------------------------

def _filtered_sequence(noise, v_ego=2.0, d0=20.0, dt=0.05):
  """20 Hz filter updates (dt=0.05) over the given noise draws."""
  filt = LeadPosFilter()
  s_ego = 0.0
  devs = []
  worst = -1e9
  for n in noise:
    true_d = d0 - s_ego
    d_eff = filt.update(True, true_d + float(n), s_ego, dt=dt)
    assert d_eff is not None
    devs.append(d_eff - true_d)
    worst = max(worst, d_eff - true_d)
    s_ego += v_ego * dt
  return np.asarray(devs), worst


def test_lead_pos_filter_smooths_stationary_lead_noise():
  rng = np.random.default_rng(0)
  noise = rng.normal(0.0, 0.3, 100)  # 20 Hz over 5 s
  devs, worst = _filtered_sequence(noise)
  assert np.std(devs) <= 0.12
  assert worst <= 0.5 + 1e-9


def test_lead_pos_filter_closer_jump_drops_immediately():
  filt = LeadPosFilter()
  assert filt.update(True, 20.0, 0.0) == pytest.approx(20.0)
  # A 2 m closer jump is adopted on the same step (safe direction).
  assert filt.update(True, 18.0, 0.0) == pytest.approx(18.0)
  # A larger-than-jump move with a moving ego frame follows immediately too.
  assert filt.update(True, 18.0 - LEAD_JUMP_M - 0.1, 0.0) == pytest.approx(18.0 - LEAD_JUMP_M - 0.1)


def test_lead_pos_filter_farther_jump_needs_hold_time():
  dt = 0.05  # 20 Hz: 10 steps = the 0.5 s verification window
  filt = LeadPosFilter()
  assert filt.update(True, 20.0, 0.0, dt=dt) == pytest.approx(20.0)
  # 2 m farther: no reset before 0.5 s of continuous verification.
  for _ in range(9):
    assert filt.update(True, 22.0, 0.0, dt=dt) == pytest.approx(20.0)
  assert filt.update(True, 22.0, 0.0, dt=dt) == pytest.approx(22.0)
  # A single in-band sample in between resets the verification timer.
  filt2 = LeadPosFilter()
  filt2.update(True, 20.0, 0.0, dt=dt)
  for _ in range(5):
    filt2.update(True, 22.0, 0.0, dt=dt)
  filt2.update(True, 20.2, 0.0, dt=dt)
  for _ in range(9):
    filt2.update(True, 22.0, 0.0, dt=dt)
  assert filt2.update(True, 22.0, 0.0, dt=dt) == pytest.approx(22.0)


def test_lead_pos_filter_status_loss_resets():
  filt = LeadPosFilter()
  filt.update(True, 20.0, 0.0)
  filt.update(True, 19.0, 0.0)
  assert filt.update(False, 19.0, 0.0) is None
  # Reacquisition re-initializes from the raw position, not the old state.
  assert filt.update(True, 25.0, 0.0) == pytest.approx(25.0)
  assert filt.update(True, None, 0.0) is None


def test_lead_pos_filter_never_exceeds_raw_plus_allow():
  filt = LeadPosFilter()
  s_ego = 0.0
  # The true lead runs away while the raw reading includes a big closer bias:
  # d_eff must stay within LEAD_RAW_ALLOW of the raw value each step.
  for k in range(60):
    d_raw = 30.0 - 0.1 * k + (-0.4 if k % 2 else 0.0)
    d_eff = filt.update(True, d_raw, s_ego)
    assert d_eff is not None
    assert d_eff <= d_raw + LEAD_RAW_ALLOW + 1e-9
    s_ego += 1.0 * lc.DT_CTRL


def test_lead_pos_filter_nan_and_invalid_inputs_reset():
  filt = LeadPosFilter()
  filt.update(True, 20.0, 0.0)
  assert filt.update(True, float("nan"), 0.0) is None
  assert filt.update(True, 20.0, float("nan")) is None
  assert filt._x_f is None


# ---------------------------------------------------------------------------
# floor slew
# ---------------------------------------------------------------------------

def test_slew_floor_tightens_instantly_relaxes_rate_limited():
  dt = lc.DT_CTRL
  # First value adopted unchanged.
  assert slew_floor(None, -1.0) == pytest.approx(-1.0)
  # Tightening instantly.
  assert slew_floor(-1.0, -2.0) == pytest.approx(-2.0)
  # Relaxing only up to FLOOR_RELAX_JERK * dt per step.
  assert slew_floor(-2.0, -0.5) == pytest.approx(-2.0 + FLOOR_RELAX_JERK * dt)
  # None resets.
  assert slew_floor(-2.0, None) is None


def test_slew_floor_reads_constant_at_call_time(monkeypatch):
  monkeypatch.setattr(butter_stop, "FLOOR_RELAX_JERK", 0.0)
  assert slew_floor(-2.0, -1.0) == pytest.approx(-2.0)


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
  # Seeded activation: the floor starts at min(applied, raw) and never rises
  # faster than FLOOR_RELAX_JERK*DT, so appearing mid-braking cannot release.
  prev = -2.0
  for _ in range(10):
    accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    assert accel >= prev - 1e-9
    assert accel <= prev + FLOOR_RELAX_JERK * lc.DT_CTRL + 1e-9
    assert control.butter_floor_last <= floor + 1e-9
    prev = accel
  assert control.butter_floor_last > -2.0
  # The stored value is the floored one, so the one-way ramp never drills below.
  assert control.last_output_accel == pytest.approx(prev)


def test_floor_activation_seeds_slew_from_applied_accel(monkeypatch):
  # Floor appears while the car is braking at -1.0 with a raw floor of -0.3:
  # the seeded slew must relax the applied floor by at most
  # FLOOR_RELAX_JERK * DT_CTRL per step instead of adopting -0.3 instantly.
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  control.last_output_accel = -1.0
  cs, plan, radar = stopping_inputs(v_ego=1.0)
  monkeypatch.setattr(lc, "butter_floor", lambda *a, **k: -0.3)
  floor_values = []
  for _ in range(10):
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    floor_values.append(control.butter_floor_last)
  # Seeded at min(-1.0, -0.3) and rate-limited from there.
  assert floor_values[0] == pytest.approx(-1.0 + FLOOR_RELAX_JERK * lc.DT_CTRL)
  rises = np.diff([-1.0, *floor_values])
  assert np.all(rises <= FLOOR_RELAX_JERK * lc.DT_CTRL + 1e-9)
  assert floor_values[-1] < -0.3 + 1e-9


def test_stopping_state_without_floor_keeps_original_ramp(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  control.last_output_accel = -0.2
  # Non-finite plan disables the floor; the original ramp still runs.
  cs, plan, radar = stopping_inputs(v_ego=1.0, plan_speeds=np.array([np.nan]))
  accel, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel == pytest.approx(-0.2 - control.CP.stoppingDecelRate * lc.DT_CTRL)


def test_butter_floor_last_telemetry(monkeypatch):
  control, _ = make_control(monkeypatch)
  control.long_control_state = LongCtrlState.stopping
  cs, plan, radar = stopping_inputs(v_ego=1.0)
  assert control.butter_floor_last is None
  floor = butter_floor(1.0, PLAN_T, plan.speeds, radar.leadOne, 5.5)
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control.butter_floor_last == pytest.approx(floor)


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
  plan.vTargetNow = 1.0
  plan.aTarget = -0.05
  accel2, _, _ = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert accel2 > floor


def test_butter_stop_defaults_off_when_param_missing(monkeypatch):
  monkeypatch.setattr(lc, "Params", lambda: DictParams({"StoppingAccel": -50}))
  control = lc.LongControl(make_cp())
  assert control.butter_stop is False


# ---------------------------------------------------------------------------
# long_mpc stop-distance shift (call-time read through the module)
# ---------------------------------------------------------------------------

def _mpc_with_stub_solver(monkeypatch):
  import openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc as long_mpc

  class _StubSolver:
    def __init__(self, *args, **kwargs):
      pass

    def reset(self):
      pass

    def cost_set(self, *args, **kwargs):
      pass

    def set(self, *args, **kwargs):
      pass

  monkeypatch.setattr(long_mpc, "AcadosOcpSolverCython", _StubSolver)
  return long_mpc


def _mpc_carrot(monkeypatch, butter):
  carrot = SimpleNamespace(
    comfort_brake=2.4, stop_distance=6.0, butter_stop=butter, mode="acc",
    v_cruise=0.0, stop_dist=1000.0, trafficStopDistanceAdjust=0.0,
    trafficStopModelLeadOffset=0.0, autoNaviSpeedDecelRate=1.0,
    dynamicTFollowLC=0.0, lane_change_active=False, leadAccelResponse=0,
    get_T_FOLLOW=lambda *a, **k: 1.5, jerk_factor=1.0, aChangeCostStarting=10.0,
  )
  return carrot


def _run_mpc_capture_stop_distance(monkeypatch, butter, d_extra=None):
  long_mpc = _mpc_with_stub_solver(monkeypatch)
  captured = {}

  def _patched_run(self):
    for i in range(long_mpc.N + 1):
      self.solver.set(i, 'p', self.params[i])
    captured["stop_distance"] = float(self.params[0, 7])

  monkeypatch.setattr(long_mpc.LongitudinalMpc, "run", _patched_run)

  mpc = long_mpc.LongitudinalMpc()
  mpc.set_accel_limits(-3.5, 2.0)
  carrot = _mpc_carrot(monkeypatch, butter)
  if d_extra is not None:
    monkeypatch.setattr(butter_stop, "D_EXTRA", d_extra)
  lead = SimpleNamespace(status=False, radarTrackId=-1, radar=False, modelProb=0.0,
                         dRel=200.0, vLead=0.0, vRel=0.0, aLeadK=0.0, aLeadTau=0.5)
  radarstate = SimpleNamespace(leadOne=lead, leadTwo=lead)
  mpc.update(carrot, False, radarstate, 0.0, np.zeros(long_mpc.N + 1),
             np.zeros(long_mpc.N + 1), np.zeros(long_mpc.N + 1), np.zeros(long_mpc.N + 1),
             measured_a_ego=0.0)
  return captured["stop_distance"]


def test_long_mpc_stop_distance_shift_call_time(monkeypatch):
  off = _run_mpc_capture_stop_distance(monkeypatch, butter=0)
  assert off == pytest.approx(6.0)
  on = _run_mpc_capture_stop_distance(monkeypatch, butter=1)
  assert on == pytest.approx(6.0 + butter_stop.D_EXTRA)
  patched = _run_mpc_capture_stop_distance(monkeypatch, butter=1, d_extra=0.42)
  assert patched == pytest.approx(6.0 + 0.42)


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


# ---------------------------------------------------------------------------
# final-roll latch (W-ROLL2)
# ---------------------------------------------------------------------------

def _roll_plan():
  """Plan used by the final-roll fixtures: plan_stop_distance == 0.90 m."""
  t = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
  v = np.maximum(0.0, 0.6 - 0.2 * t)
  return t, v


def test_final_roll_limit_values():
  from openpilot.selfdrive.controls.lib.butter_stop import final_roll_limit
  assert final_roll_limit(0.6, 4.85, 10.0) == pytest.approx(10.51, abs=1e-9)
  assert final_roll_limit(0.3, 4.7, 0.0) == pytest.approx(0.24, abs=1e-9)
  assert final_roll_limit(0.6, 3.2, 5.0) == pytest.approx(5.2, abs=1e-9)


def test_butter_floor_roll_avail_only_raises_lead_budget():
  t, speeds = _roll_plan()
  assert plan_stop_distance(t, speeds) == pytest.approx(0.90, abs=1e-9)
  lead = SimpleNamespace(status=True, dRel=4.85, aLeadK=0.0)

  for d in (4.85, 4.55, 3.24):
    lead = SimpleNamespace(status=True, dRel=d, aLeadK=0.0)
    for v in (0.6, 0.3, 0.25):
      plain = butter_floor(v, t, speeds, lead, 6.0, d_rel_eff=d)
      garbage = butter_floor(v, t, speeds, lead, 6.0, d_rel_eff=d, roll_avail=-0.1)
      nan_roll = butter_floor(v, t, speeds, lead, 6.0, d_rel_eff=d, roll_avail=float("nan"))
      assert garbage == plain
      assert nan_roll == plain
      # No lead in view: roll_avail must not change the plan-bound floor.
      no_lead = SimpleNamespace(status=False, dRel=d, aLeadK=0.0)
      assert butter_floor(v, t, speeds, no_lead, 6.0, roll_avail=0.51) == \
             butter_floor(v, t, speeds, no_lead, 6.0)


def test_butter_floor_roll_avail_turns_none_into_floor():
  t, speeds = _roll_plan()
  lead = SimpleNamespace(status=True, dRel=4.85, aLeadK=0.0)
  assert butter_floor(0.6, t, speeds, lead, 6.0, d_rel_eff=4.85) is None
  floor = butter_floor(0.6, t, speeds, lead, 6.0, d_rel_eff=4.85, roll_avail=0.51)
  assert floor == pytest.approx(-0.469, abs=1e-3)
  assert floor == pytest.approx(-(solve_taper_a_end(0.6, 0.51, K_TAPER) + K_TAPER * 0.6), abs=1e-12)

  lead = SimpleNamespace(status=True, dRel=4.55, aLeadK=0.0)
  assert butter_floor(0.25, t, speeds, lead, 6.0, d_rel_eff=4.55) is None
  assert butter_floor(0.25, t, speeds, lead, 6.0, d_rel_eff=4.55, roll_avail=0.21) == \
         pytest.approx(-0.225, abs=1e-9)

  lead = SimpleNamespace(status=True, dRel=3.24, aLeadK=0.0)
  assert butter_floor(0.3, t, speeds, lead, 6.0, d_rel_eff=3.24, roll_avail=0.24) == \
         pytest.approx(-0.25, abs=1e-9)


def _roll_frames(control, v_sequence, radar_state):
  outs = []
  for v in v_sequence:
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=_roll_plan()[1],
                                      lead_status=radar_state["status"],
                                      d_rel=radar_state["d_rel"], v_lead=radar_state["v_lead"])
    radar.leadOne = SimpleNamespace(status=radar_state["status"], dRel=radar_state["d_rel"],
                                    aLeadK=0.0, vLead=radar_state["v_lead"])
    outs.append(control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0])
  return outs


def test_longcontrol_roll_latch(monkeypatch):
  from openpilot.selfdrive.controls.lib.butter_stop import (
    FINAL_ROLL_M, FINAL_ROLL_V, final_roll_limit,
  )
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  # Normal lead budget is 4.3 - 4.3 = 0 m, i.e. below AVAIL_MIN: only the
  # latch can make the floor exist here.
  lead = {"status": True, "d_rel": 4.3, "v_lead": 0.0}
  v_seq = [round(0.9 - 0.05 * i, 2) for i in range(17)]  # 0.9 .. 0.10

  latched_at = []
  prev = None
  for v in v_seq:
    _roll_frames(control, [v], lead)
    cur = control._butter_roll_limit
    if cur is not None and prev is None:
      latched_at.append(v)
      # Exactly the stock rolling distance plus FINAL_ROLL_M from this position.
      assert cur - control._butter_s_ego == pytest.approx(
        min(v * v / 1.0 + FINAL_ROLL_M, 4.3 - MIN_FINAL_GAP_ABS), abs=1e-9)
      assert cur == pytest.approx(final_roll_limit(v, 4.3, control._butter_s_ego), abs=1e-9)
    prev = cur
  assert latched_at == [FINAL_ROLL_V]
  assert control._butter_roll_limit is not None
  # Unchanged while the car keeps rolling down.
  held = control._butter_roll_limit
  _roll_frames(control, [0.3], lead)
  assert control._butter_roll_limit == pytest.approx(held, abs=1e-12)
  assert control.butter_floor_raw_last is not None

  # A stale latch above FINAL_ROLL_RESET_V is dropped.
  _roll_frames(control, [1.1], lead)
  assert control._butter_roll_limit is None
  # Relatching while the lead is stopped is allowed again.
  _roll_frames(control, [0.3], lead)
  assert control._butter_roll_limit is not None
  # Losing the lead invalidates the latch.
  _roll_frames(control, [0.3], {"status": False, "d_rel": 30.0, "v_lead": 0.0})
  assert control._butter_roll_limit is None
  # A moving lead invalidates it too.
  _roll_frames(control, [0.3], lead)
  assert control._butter_roll_limit is not None
  _roll_frames(control, [0.3], {"status": True, "d_rel": 4.3, "v_lead": 0.5})
  assert control._butter_roll_limit is None


def test_longcontrol_roll_latch_needs_stopped_lead(monkeypatch):
  """A lead that keeps rolling must never grant a limit, tick after tick."""
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  seen = []
  real = lc.butter_floor

  def spy(v, plan_t, speeds, lead, stop_d, d_rel_eff=None, diag=None, roll_avail=None):
    got = real(v, plan_t, speeds, lead, stop_d, d_rel_eff=d_rel_eff, diag=diag)
    seen.append((v, d_rel_eff, roll_avail, got))
    return got

  monkeypatch.setattr(lc, "butter_floor", spy)
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  moving = {"status": True, "d_rel": 4.3, "v_lead": 0.5}
  v_seq = [round(0.9 - 0.05 * i, 2) for i in range(17)]  # 0.9 .. 0.10
  for v in v_seq:
    _roll_frames(control, [v], moving)
    assert control._butter_roll_limit is None, "a moving lead must never latch"

  assert seen, "the floor was never consulted"
  t, speeds = _roll_plan()
  lead = SimpleNamespace(status=True, dRel=4.3, aLeadK=0.0, vLead=0.5)
  for v, d_eff, roll, got in seen:
    assert roll is None
    # The floor is exactly the latch-off floor for the same filtered distance.
    assert got == real(v, t, speeds, lead, control.stop_distance_orig, d_rel_eff=d_eff)
