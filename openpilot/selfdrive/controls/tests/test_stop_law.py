"""ButterStop stop law (stop_law.py) wired into LongControl on the Palisade:
takeover equality against a reference StopCtl, off-identity for every other
car / ButterStop=0 / outside the window, the release slew, the fixture replays
and the module constants.
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

import json, math, os
from types import SimpleNamespace
import numpy as np
import pytest
import openpilot.selfdrive.controls.lib.longcontrol as lc
from openpilot.selfdrive.controls.lib import butter_stop
from openpilot.selfdrive.controls.lib import stop_law as SL
from openpilot.selfdrive.controls.lib.longcontrol import CONTROL_N_T_IDX, LongControl, LongCtrlState
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.common.realtime import DT_CTRL
PLAN_T = np.asarray(CONTROL_N_T_IDX, dtype=float)
FIX = os.path.join(os.path.dirname(__file__), "stop_law_fixtures")
LIMITS = (-3.5, 2.0)
AGE = 0.05


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


def make_cp(fingerprint="HYUNDAI_PALISADE", flags=0):
  return SimpleNamespace(brand="hyundai", flags=flags, carFingerprint=fingerprint, openpilotLongitudinalControl=True,
                         longitudinalTuning=SimpleNamespace(kpBP=[0.0], kpV=[1.0], kiBP=[0.0], kiV=[0.2], kf=1.0),
                         startingState=False, vEgoStarting=0.5, stopAccel=-2.0, stoppingDecelRate=0.8, startAccel=1.0)


def make_control(monkeypatch, butter=1, stop_cm=550, **cp_kw):
  stored = {"StoppingAccel": -50, "StopDistanceCarrot": stop_cm, "ButterStop": butter,
            "LongTuningKpV": 100, "LongTuningKiV": 200, "LongTuningKf": 100}
  params = DictParams(stored)
  monkeypatch.setattr(lc, "Params", lambda: params)
  return LongControl(make_cp(**cp_kw)), params


class NominalPlant:
  """The brake response stop_law.py was designed on (G, C, TAU_B, ND): no creep, no ESC flag."""

  def __init__(self, v0, a0=-0.5):
    self.v, self.a, self.x = float(v0), float(a0), 0.0
    self.u_hist = [float(a0)] * (SL.ND + 1)

  def step(self, u):
    self.u_hist.append(float(u))
    self.a += (SL.G * self.u_hist[-1 - SL.ND] + SL.C - self.a) * (SL.DT / SL.TAU_B)
    self.v = max(0.0, self.v + self.a * SL.DT)
    self.x += self.v * SL.DT


def tick_inputs(plant, x_lead, v_lead=0.0, lead_status=True, a_target=-1.0, v_target_now=None,
                should_stop=None, gas=False, brake=False, a_ego=None):
  v = plant.v
  cs = SimpleNamespace(vEgo=v, aEgo=plant.a if a_ego is None else a_ego, brakePressed=brake, gasPressed=gas,
                       softHoldActive=0, standstill=v < 0.01, cruiseState=SimpleNamespace(standstill=False),
                       vCruise=0.0, carrotCruise=0)
  plan = SimpleNamespace(aTarget=a_target, vTargetNow=max(0.0, v - 0.1) if v_target_now is None else v_target_now,
                         jTargetNow=0.0, shouldStop=(v < 0.3) if should_stop is None else should_stop, fcw=False,
                         speeds=np.maximum(0.0, v - 1.0 * PLAN_T), longitudinalPlanSource="lead0", cruiseTarget=0.0,
                         cruiseCoastingTarget=0.0, cruiseCoastingPercent=0)
  lead = SimpleNamespace(status=lead_status, dRel=x_lead - plant.x, aLeadK=0.0, vLead=v_lead, radarTrackId=7)
  radar = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False), leadCutInRisk=SimpleNamespace(status=False))
  return cs, plan, radar


def run_stop(control, v0=3.4, gap0=4.8 + 12.0, n=1500, v_lead=0.0, script=None, ref=None):
  """Warm-up 30 ticks (plant frozen, a_target 0.0 so the law cannot start), then n closed-loop ticks.
  script(k, control, trace) -> dict of tick_inputs overrides (or None). ref: optional parallel StopCtl."""
  plant = NominalPlant(v0)
  x_lead = gap0
  trace = []
  for k in range(-30, n):
    kw = dict(v_lead=v_lead)
    if k < 0:
      kw["a_target"] = 0.0
    elif script is not None:
      kw.update(script(k, control, trace) or {})
    cs, plan, radar = tick_inputs(plant, x_lead, **kw)
    out = float(control.update(True, cs, plan, LIMITS, AGE, radar)[0])
    u_ref = None
    if ref is not None:
      if control.long_control_state == LongCtrlState.pid:
        u_ref = ref.update(not (cs.gasPressed or cs.brakePressed), cs.vEgo, cs.aEgo, radar.leadOne.status,
                           radar.leadOne.dRel if radar.leadOne.status else None,
                           radar.leadOne.vLead if radar.leadOne.status else float("nan"), plan.aTarget, float("inf"))
      else:
        ref.idle()
      ref.commit(float(control.last_output_accel))
    if k >= 0:
      trace.append(dict(k=k, v=plant.v, a=cs.aEgo, gap=x_lead - plant.x, out=out, lcs=control.long_control_state,
                        law=control.stop_law_last, u_ref=u_ref,
                        uh_ok=(control._stop_ctl is None or control._stop_ctl.uh[-1] == float(control.last_output_accel))))
    if k >= 0:
      plant.step(out)
      x_lead += kw.get("v_lead", v_lead) * DT_CTRL
  return trace, plant


def after_law50(make_kw):
  """Script factory. k50 = the first k at which the last 50 trace rows all had the law in charge.
  From k50 on it returns make_kw(k, control, trace). At the start of every tick k > k50 it records the
  controller state left by tick k-1 in state["post"] (so state["post"][0] is the state right after k50)."""
  state = {"k50": None, "post": []}

  def script(k, control, trace):
    if state["k50"] is None and len(trace) >= 50 and all(r["law"] for r in trace[-50:]):
      state["k50"] = k
    if state["k50"] is None:
      return None
    if k > state["k50"]:
      ctl = control._stop_ctl
      state["post"].append(dict(k=k - 1, why=None if ctl is None else ctl.why,
                                law_obj=None if ctl is None else ctl.law, ctl_none=ctl is None,
                                release=control._stop_release, lcs=control.long_control_state,
                                law_last=control.stop_law_last))
    return make_kw(k, control, trace)

  script.state = state
  return script


def silence_cloudlog(monkeypatch):
  events = []
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda name, **kw: events.append((name, kw))))
  return events


def sequences(trace):
  return ([r["out"] for r in trace], [r["lcs"] for r in trace], [r["v"] for r in trace], [r["gap"] for r in trace])


def stop_trace(monkeypatch, butter=1, **cp_kw):
  control, params = make_control(monkeypatch, butter=butter, **cp_kw)
  trace, plant = run_stop(control)
  return control, params, trace, plant


def test_t1_takeover_equals_reference(monkeypatch):
  events = silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch)
  ref = SL.StopCtl()
  trace, plant = run_stop(control, ref=ref)

  for row in trace:
    if row["u_ref"] is not None:
      assert row["law"] is True
      assert row["out"] == float(np.clip(row["u_ref"], *LIMITS))
    else:
      assert row["law"] is False
  assert all(r["uh_ok"] for r in trace)
  n_law = sum(r["law"] for r in trace)
  assert n_law >= 100
  assert control._stop_ctl.n_enter >= 1
  assert len([e for e in events if e[0] == "butterStopLaw"]) == 1
  min_gap = min(r["gap"] for r in trace)
  assert min_gap >= 3.0
  assert plant.v == 0.0
  first_law_v = next((r["v"] for r in trace if r["law"]), float("nan"))
  msg = "T1 final_gap=%.3f law_ticks=%d n_enter=%d min_gap=%.3f first_law_v=%.3f" % (
    trace[-1]["gap"], n_law, control._stop_ctl.n_enter, min_gap, first_law_v)
  print(msg)
  assert 4.3 <= trace[-1]["gap"] <= 5.5, msg


def test_t2_palisade_law_off_arm_differs(monkeypatch):
  silence_cloudlog(monkeypatch)
  on_control, params, on_trace, on_plant = stop_trace(monkeypatch)
  monkeypatch.setattr(lc, "STOP_LAW_ENABLED", False)
  try:
    off_control, off_params = make_control(monkeypatch)
    off_trace, off_plant = run_stop(off_control)
  finally:
    monkeypatch.setattr(lc, "STOP_LAW_ENABLED", True)
  assert off_control._stop_ctl is None
  assert any(a != b for a, b in zip([r["out"] for r in on_trace], [r["out"] for r in off_trace]))


def test_t3_non_palisade_identity(monkeypatch):
  silence_cloudlog(monkeypatch)
  ctrl_on, _ = make_control(monkeypatch, fingerprint="HYUNDAI_SONATA")
  trace_on, _ = run_stop(ctrl_on)
  monkeypatch.setattr(lc, "STOP_LAW_ENABLED", False)
  try:
    ctrl_off, _ = make_control(monkeypatch, fingerprint="HYUNDAI_SONATA")
    trace_off, _ = run_stop(ctrl_off)
  finally:
    monkeypatch.setattr(lc, "STOP_LAW_ENABLED", True)
  assert ctrl_on._stop_ctl is None
  assert ctrl_off._stop_ctl is None
  assert sequences(trace_on) == sequences(trace_off)


def test_t4_canfd_identity(monkeypatch):
  silence_cloudlog(monkeypatch)
  ctrl_on, _ = make_control(monkeypatch, flags=int(HyundaiFlags.CANFD))
  trace_on, _ = run_stop(ctrl_on)
  monkeypatch.setattr(lc, "STOP_LAW_ENABLED", False)
  try:
    ctrl_off, _ = make_control(monkeypatch, flags=int(HyundaiFlags.CANFD))
    trace_off, _ = run_stop(ctrl_off)
  finally:
    monkeypatch.setattr(lc, "STOP_LAW_ENABLED", True)
  assert ctrl_on._stop_ctl is None
  assert ctrl_off._stop_ctl is None
  assert sequences(trace_on) == sequences(trace_off)


def test_t5_butter_stop_off_identity(monkeypatch):
  silence_cloudlog(monkeypatch)
  ctrl_on, _ = make_control(monkeypatch, butter=0)
  trace_on, _ = run_stop(ctrl_on)
  monkeypatch.setattr(lc, "STOP_LAW_ENABLED", False)
  try:
    ctrl_off, _ = make_control(monkeypatch, butter=0)
    trace_off, _ = run_stop(ctrl_off)
  finally:
    monkeypatch.setattr(lc, "STOP_LAW_ENABLED", True)
  assert ctrl_on._stop_ctl is None
  assert ctrl_off._stop_ctl is None
  assert ctrl_on.stop_law is False
  assert sequences(trace_on) == sequences(trace_off)


def test_t6_moving_lead_never_enters(monkeypatch):
  silence_cloudlog(monkeypatch)
  ctrl_on, _ = make_control(monkeypatch)
  trace_on, _ = run_stop(ctrl_on, gap0=20.0, n=600, v_lead=3.0)
  monkeypatch.setattr(lc, "STOP_LAW_ENABLED", False)
  try:
    ctrl_off, _ = make_control(monkeypatch)
    trace_off, _ = run_stop(ctrl_off, gap0=20.0, n=600, v_lead=3.0)
  finally:
    monkeypatch.setattr(lc, "STOP_LAW_ENABLED", True)
  assert ctrl_on._stop_ctl.n_enter == 0
  assert sequences(trace_on) == sequences(trace_off)


def test_t7_release_slew_after_lead_go(monkeypatch):
  silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch)
  script = after_law50(lambda k, c, tr: dict(v_lead=2.0, a_target=0.0, v_target_now=tr[-1]["v"]))
  trace, plant = run_stop(control, script=script)
  out = [r["out"] for r in trace]
  k50 = script.state["k50"]
  assert k50 is not None
  post0 = script.state["post"][0]
  assert post0["k"] == k50
  assert trace[k50]["law"] is False
  assert post0["why"] == "lead_go"
  assert out[k50] > out[k50 - 1]
  for k in range(k50, k50 + 60):
    assert out[k] - out[k - 1] <= lc.STOP_LAW_RELEASE_JERK * DT_CTRL + 1e-12
  assert any(p["release"] is None and p["k"] <= k50 + 200 for p in script.state["post"])


def test_t8_harder_ordinary_command_passes_at_once(monkeypatch):
  silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch)
  script = after_law50(lambda k, c, tr: dict(v_lead=2.0, a_target=-3.0, v_target_now=0.0))
  trace, plant = run_stop(control, script=script)
  out = [r["out"] for r in trace]
  k50 = script.state["k50"]
  assert k50 is not None
  post0 = script.state["post"][0]
  assert post0["k"] == k50
  assert out[k50] < out[k50 - 1] - 0.5
  assert post0["release"] is None


def test_t9a_pid_to_stopping_ends_law(monkeypatch):
  events = silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch)
  script = after_law50(lambda k, c, tr: dict(should_stop=True, a_ego=-0.3) if k == script.state["k50"] else None)
  trace, plant = run_stop(control, script=script)
  k50 = script.state["k50"]
  assert k50 is not None
  post0 = script.state["post"][0]
  assert post0["k"] == k50
  assert post0["lcs"] == LongCtrlState.stopping
  assert post0["law_obj"] is None
  assert post0["release"] is None
  assert post0["law_last"] is False
  assert any(n == "butterStopLawDrop" and kw.get("why") == "state" for n, kw in events)


def test_t9b_brake_ends_law(monkeypatch):
  silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch)
  script = after_law50(lambda k, c, tr: dict(brake=True) if k == script.state["k50"] else None)
  trace, plant = run_stop(control, script=script)
  k50 = script.state["k50"]
  assert k50 is not None
  post0 = script.state["post"][0]
  assert post0["k"] == k50
  assert post0["why"] == "allow"
  assert post0["release"] is None
  assert post0["law_last"] is False


def test_t9c_butter_stop_off_mid_law(monkeypatch):
  silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch)

  def make_kw(k, c, tr):
    if k == script.state["k50"]:
      params.values["ButterStop"] = 0
      c._refresh_butter_stop()
    return dict(a_target=0.0, v_target_now=tr[-1]["v"])

  script = after_law50(make_kw)
  trace, plant = run_stop(control, script=script)
  out = [r["out"] for r in trace]
  k50 = script.state["k50"]
  assert k50 is not None
  post0 = script.state["post"][0]
  assert post0["k"] == k50
  assert post0["ctl_none"] is True
  for k in range(k50, k50 + 40):
    assert out[k] - out[k - 1] <= lc.STOP_LAW_RELEASE_JERK * DT_CTRL + 1e-12
  assert out[k50] > out[k50 - 1]


@pytest.fixture
def _c0_config(monkeypatch):
  """The deployed (c0) law's settings: no smoothing lag, no covariance cap, no guards, old th box.
  The recorded fixtures were taken from that law, so the replay tests pin it bit for bit."""
  for name, value in (("TAU_S", SL.DT), ("P_MAX", 1e9), ("GUARD", False), ("SLEW", False),
                      ("P0", (0.04, 0.09)), ("TH_LO", (-0.5, -1.0)), ("TH_HI", (0.8, 1.0))):
    monkeypatch.setattr(SL, name, value)
  # The L7 constants change the command these fixtures recorded; the replays pin the
  # deployed (c0) law, so they are pinned back to it.
  for name, value in (("U_MAX", 0.0), ("DEN_LOCK", 0.05)):
    monkeypatch.setattr(SL, name, value)
  monkeypatch.setitem(SL.CFG, "VL_IN", 1.0)


def test_t10_stoplaw_fixture(_c0_config):
  z = np.load(os.path.join(FIX, "stoplaw_fixture.npz"), allow_pickle=False)
  worst_u = worst_r = 0.0
  n_cases = int(z["n"])
  for i in range(n_cases):
    a0, x_t = (float(x) for x in z["c%d_meta" % i])
    law = SL.StopLaw(a0, [float(x) for x in z["c%d_uhist" % i]])
    io = z["c%d_io" % i]
    for k in range(io.shape[1]):
      v, a_m, x, u_exp, r_exp = (float(x) for x in io[:, k])
      u = law.step(v, a_m, x_t - x)
      worst_u = max(worst_u, abs(u - u_exp))
      worst_r = max(worst_r, abs(law.r - r_exp))
      law.commit(u)
  assert worst_u <= 1e-9
  assert worst_r <= 1e-9


_STOPCTL_N = int(np.load(os.path.join(FIX, "stopctl_fixture.npz"), allow_pickle=False)["n"])


@pytest.mark.parametrize("i", range(_STOPCTL_N))
def test_t11_stopctl_fixture(_c0_config, i):
  z = np.load(os.path.join(FIX, "stopctl_fixture.npz"), allow_pickle=False)
  ctl = SL.StopCtl(**dict(json.loads(str(z["c%d_cfg" % i])), TOL_LOCK=0.0))  # recorded with the D7 law at TOL_LOCK 0
  for u0 in z["c%d_warm" % i]:
    ctl.commit(float(u0))
  mism = 0
  worst = 0.0
  for row in z["c%d_io" % i]:
    allow, v, a, ls, d_rel, v_lead, a_tgt, u_plan, u_exp, u_app = (float(x) for x in row)
    u = ctl.update(bool(allow), v, a, bool(ls), d_rel, v_lead, a_tgt, u_plan)
    if math.isnan(u_exp):
      mism += u is not None
    elif u is None:
      mism += 1
    else:
      worst = max(worst, abs(u - u_exp))
    ctl.commit(u_app)
  assert mism == 0
  assert worst <= 1e-9


_STOPCTL_TOL_N = int(np.load(os.path.join(FIX, "stopctl_tol_fixture.npz"), allow_pickle=False)["n"])


def _replay_ctl(path, i, extra):
  z = np.load(path, allow_pickle=False)
  ctl = SL.StopCtl(**dict(json.loads(str(z["c%d_cfg" % i])), **extra))
  for u0 in z["c%d_warm" % i]:
    ctl.commit(float(u0))
  mism = 0
  worst = 0.0
  for row in z["c%d_io" % i]:
    allow, v, a, ls, d_rel, v_lead, a_tgt, u_plan, u_exp, u_app = (float(x) for x in row)
    u = ctl.update(bool(allow), v, a, bool(ls), d_rel, v_lead, a_tgt, u_plan)
    if math.isnan(u_exp):
      mism += u is not None
    elif u is None:
      mism += 1
    else:
      worst = max(worst, abs(u - u_exp))
    ctl.commit(u_app)
  return mism, worst


@pytest.mark.parametrize("i", range(_STOPCTL_TOL_N))
def test_t13a_stopctl_tol_fixture_at_default(_c0_config, i):
  # recorded with TWIN/PORT2/stop_law2.py at TOL_LOCK 0.3 (the law the sweep ran); stored cfg omits TOL_LOCK
  mism, worst = _replay_ctl(os.path.join(FIX, "stopctl_tol_fixture.npz"), i, {})
  assert mism == 0
  assert worst <= 1e-9


def test_t13b_tol_lock_default_and_teeth(_c0_config):
  assert SL.CFG["TOL_LOCK"] == 0.3
  worst = max(_replay_ctl(os.path.join(FIX, "stopctl_tol_fixture.npz"), i, {"TOL_LOCK": 0.0})[1] for i in range(_STOPCTL_TOL_N))
  assert worst > 1e-3


def test_t13c_locked_hold_is_softer_with_tol():
  law0 = SL.StopLaw(-0.7, [-0.7] * 64, 0.0)
  law3 = SL.StopLaw(-0.7, [-0.7] * 64, 0.3)
  for _ in range(40):
    u0 = law0.step(0.6, -0.7, 0.15)
    u3 = law3.step(0.6, -0.7, 0.15)
    assert law3.r >= law0.r - 1e-12
    law0.commit(u0)
    law3.commit(u3)
  assert law0.locked and law3.locked
  assert law3.r - law0.r > 0.5


def test_t12a_dt_matches_controls_tick():
  assert SL.DT == DT_CTRL


def test_t12b_lead_filter_constants_match_butter_stop():
  assert SL.CFG["TAU_LEAD"] == butter_stop.TAU_LEAD
  assert SL.CFG["LEAD_JUMP_M"] == butter_stop.LEAD_JUMP_M
  assert SL.CFG["LEAD_RAW_ALLOW"] == butter_stop.LEAD_RAW_ALLOW
  assert SL.CFG["JUMP_VERIFY_S"] == butter_stop.LeadPosFilter.JUMP_VERIFY_S


def test_t12c_lead_pos_filter_matches_butter_stop():
  lead_pos = SL.LeadPos(SL.CFG["TAU_LEAD"], SL.CFG["LEAD_JUMP_M"], SL.CFG["LEAD_RAW_ALLOW"], SL.CFG["JUMP_VERIFY_S"])
  lead_filter = butter_stop.LeadPosFilter()
  for k in range(400):
    s = 0.02 * k
    d = 20.0 - s + 0.05 * np.random.default_rng(3).standard_normal()
    if 100 <= k < 400:
      d -= 3.0
    if 200 <= k < 230:
      d += 3.0
    if 300 <= k < 400:
      d += 3.0
    a = lead_pos.update(d, s)
    b = lead_filter.update(True, d, s)
    assert abs(a - b) <= 1e-12


# --- spec_check_port.py P1-P9, one pytest function each (S = K = SL) ---
S = SL
K = SL
C = K.CFG
INF = float("inf")


def _warm(ctl, u=-0.8, n=30):
  for _ in range(n):
    ctl.commit(u)


def _base(**kw):
  d = dict(allow=True, v=3.0, a=-0.8, lead_status=True, d_rel=4.8 + 8.0, v_lead=0.0, a_target=-0.8, u_plan=INF)
  d.update(kw)
  return d


def _tick(ctl, **kw):
  return ctl.update(**_base(**kw))


def test_p1_entry_on_base_tick():
  c = K.StopCtl()
  _warm(c)
  u = _tick(c)
  assert u is not None


def test_p2_each_gate_alone_blocks_entry():
  for name, kw in [("v_lead>=VL_IN", dict(v_lead=C["VL_IN"])), ("v>V_ENTER", dict(v=C["V_ENTER"] + 0.01)),
                   ("v<=V_ENTER_MIN", dict(v=C["V_ENTER_MIN"], d_rel=4.8 + 0.6)),
                   ("d_rel<D_MIN", dict(d_rel=C["D_MIN"] - 0.01)), ("q<Q_ENTER", dict(d_rel=4.8 + 40.0)),
                   ("a_target>=0", dict(a_target=0.0)), ("allow False", dict(allow=False)),
                   ("lead_status False", dict(lead_status=False)), ("d_rel nan", dict(d_rel=float("nan")))]:
    c = K.StopCtl()
    _warm(c)
    r = _tick(c, **kw)
    assert r is None and c.law is None
  c = K.StopCtl()
  _warm(c, n=S.ND)
  r = _tick(c)
  assert r is None
  c = K.StopCtl()
  _warm(c, n=S.ND + 1)
  r = _tick(c)
  assert r is not None


def test_p3_drops_and_reasons():
  for name, kw, why in [("lead lost", dict(lead_status=False), "lead"), ("lead faster", dict(v_lead=C["VL_OUT"]), "lead_go"),
                        ("ego faster", dict(v=C["V_EXIT"] + 0.01), "fast"), ("allow False", dict(allow=False), "allow")]:
    c = K.StopCtl()
    _warm(c)
    _tick(c)
    c.commit(c.u_out)
    r = _tick(c, **kw)
    assert r is None and c.law is None and c.why == why
  c = K.StopCtl()
  _warm(c)
  _tick(c)
  c.commit(c.u_out)
  r = _tick(c, v_lead=C["VL_IN"] + 0.2)
  assert r is not None


def test_p4_flag_hold_freezes_command():
  c = K.StopCtl()
  _warm(c)
  _tick(c)
  c.commit(c.u_out)
  n_hold = 0
  r_at = None
  for k in range(80):
    u = _tick(c, v=0.7, a=-0.7, d_rel=4.8 + 0.4)
    c.commit(u)
    if c.frozen and r_at is None:
      r_at = (k, c.law.r, u)
    elif c.frozen:
      assert c.law.r == r_at[1] and u == r_at[2]
      n_hold += 1
  assert r_at[0] == 50


def test_p5_lead_guard():
  c = K.StopCtl()
  _warm(c)
  _tick(c)
  c.commit(c.u_out)
  u = _tick(c, v=2.0, d_rel=C["D_MIN"] + 0.2)
  g = -2.0 * 2.0 / (2 * 0.2)
  assert abs(u - g) < 1e-12 and c.n_guard == 1
  c = K.StopCtl()
  _warm(c)
  _tick(c)
  assert c.n_guard == 0


def test_p6_plan_floor():
  c = K.StopCtl(Q_PLAN=1.0)
  _warm(c)
  u = _tick(c, d_rel=4.8 + 5.0, u_plan=-3.0)
  assert u == -3.0 and c.n_plan == 1
  c = K.StopCtl(Q_PLAN=1.0)
  _warm(c)
  u = _tick(c, d_rel=4.8 + 5.0, u_plan=INF)
  assert c.n_plan == 0
  c = K.StopCtl(Q_PLAN=1.0)
  _warm(c)
  u = _tick(c, d_rel=4.8 + 12.0, u_plan=-3.0)
  assert u > -3.0 and c.n_plan == 0
  c = K.StopCtl()
  _warm(c)
  u = _tick(c, d_rel=4.8 + 5.0, u_plan=-3.0)
  assert u > -3.0


def test_p7_commit_feeds_applied_command_back():
  c = K.StopCtl()
  _warm(c)
  u = _tick(c)
  c.commit(-1.5)
  assert c.law.u_prev == -1.5 and c.law.uh[-1] == -1.5 and c.uh[-1] == -1.5 and c.u_out == -1.5
  u2 = _tick(c)
  assert abs(u2 - (-1.5)) <= S.DU_LIM * S.DT + 1e-12


def test_p8_first_command_is_continuous():
  c = K.StopCtl()
  _warm(c, u=-0.8)
  u = _tick(c)
  assert abs(u + 0.8) <= S.DU_LIM * S.DT + 1e-12


def test_p9_unknown_keyword_raises():
  with pytest.raises(KeyError):
    K.StopCtl(NOPE=1)


def test_gap_setting_map():
  assert SL.stop_law_gap(5.5) == 4.8 == SL.CFG["GAP"]
  assert SL.stop_law_gap(4.0) == 3.3
  assert SL.stop_law_gap(4.5) == 3.8
  assert SL.stop_law_gap(5.0) == 4.3
  assert SL.stop_law_gap(7.0) == 6.3
  assert SL.stop_law_gap(10.0) == 9.3
  for bad in (0.0, 3.99, 10.01, float("nan"), None, "x"):
    assert SL.stop_law_gap(bad) == SL.CFG["GAP"]
  assert SL.stop_law_gap(butter_stop.STOP_DISTANCE_DEFAULT) == SL.CFG["GAP"]


@pytest.mark.parametrize("stop_cm,expected", [(450, 3.8), (500, 4.3), (550, 4.8), (700, 6.3), (0, 4.8)])
def test_gap_wired_from_setting(monkeypatch, stop_cm, expected):
  control, _ = make_control(monkeypatch, stop_cm=stop_cm)
  assert control._stop_ctl.c["GAP"] == expected


def test_gap_change_waits_for_idle(monkeypatch):
  control, params = make_control(monkeypatch)
  ctl = control._stop_ctl
  sentinel = object()
  ctl.law = sentinel
  params.values["StopDistanceCarrot"] = 700
  control._refresh_butter_stop()
  assert control._stop_ctl is ctl
  assert ctl.law is sentinel
  assert ctl.c["GAP"] == 4.8
  ctl.law = None
  control._refresh_butter_stop()
  assert ctl.c["GAP"] == 6.3


@pytest.mark.parametrize("stop_cm", [450, 550, 700])
def test_t1_gap_tracks_setting(monkeypatch, stop_cm):
  events = silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch, stop_cm=stop_cm)
  gap = SL.stop_law_gap(stop_cm * 0.01)
  ref = SL.StopCtl(GAP=gap)
  trace, plant = run_stop(control, gap0=gap + 12.0, ref=ref)

  for row in trace:
    if row["u_ref"] is not None:
      assert row["law"] is True
      assert row["out"] == float(np.clip(row["u_ref"], *LIMITS))
    else:
      assert row["law"] is False
  assert all(r["uh_ok"] for r in trace)
  n_law = sum(r["law"] for r in trace)
  assert n_law >= 100
  assert control._stop_ctl.n_enter >= 1
  assert len([e for e in events if e[0] == "butterStopLaw"]) == 1
  min_gap = min(r["gap"] for r in trace)
  assert min_gap >= 3.0
  assert plant.v == 0.0
  first_law_v = next((r["v"] for r in trace if r["law"]), float("nan"))
  msg = "T1 cm=%d final_gap=%.3f law_ticks=%d n_enter=%d min_gap=%.3f first_law_v=%.3f" % (
    stop_cm, trace[-1]["gap"], n_law, control._stop_ctl.n_enter, min_gap, first_law_v)
  print(msg)
  assert gap - 0.5 <= trace[-1]["gap"] <= gap + 0.7, msg


# --- L6: the smoothing-lag regressors, the covariance cap, the two guard vetoes and the slew ---

def test_l6_s_frs_identity(monkeypatch):
  # S_frs: a linear plant built from the law's own filters. a_true = fb + (s-1)*fu + d*fw, and a_m is
  # a first-order lag of a_true with tau TAU_S starting at a0. With GUARD off the law must carry that
  # plant exactly: at the start of every tick a_m == fb_s + (s-1)*fu_s + d*fw_s, i.e. the regressors
  # the RLS solves with are the TAU_S-smoothed filters rather than the raw ones.
  monkeypatch.setattr(SL, "GUARD", False)
  worst = 0.0
  for s, d in ((1.1, -0.2), (0.9, 0.3)):
    a0 = -0.2
    law = SL.StopLaw(a0, [-0.3] * (SL.ND + 1))
    assert law.fu == 0.0 and law.fu_s == 0.0
    assert law.u_f == SL.G * law.uh[-1 - SL.ND]
    fb, fu, fw = a0, 0.0, 0.0                 # the raw regressors _adapt filters
    fb_s, fu_s, fw_s = a0, 0.0, 0.0           # their TAU_S smoothing
    a_m = a0
    for k in range(2000):
      # This tick's _adapt inputs, read from the state the law is about to see.
      u_d = SL.G * law.uh[-1 - SL.ND]
      wk = SL.dw(law.v_prev) if law.v_prev is not None else None
      v = max(0.0, 3.0 - 0.0005 * k)
      u = law.step(v, a_m, max(0.2, 12.0 - 0.005 * k))
      law.commit(u)
      if wk is None:
        continue                            # _adapt returns before touching a filter on tick 0
      fb += (u_d + SL.C - fb) * SL.BETA
      fu += (u_d - fu) * SL.BETA
      fw += (wk - fw) * SL.BETA
      fb_s += (fb - fb_s) * (SL.DT / SL.TAU_S)
      fu_s += (fu - fu_s) * (SL.DT / SL.TAU_S)
      fw_s += (fw - fw_s) * (SL.DT / SL.TAU_S)
      assert abs(law.fb - fb) <= 1e-12 and abs(law.fu - fu) <= 1e-12 and abs(law.fw - fw) <= 1e-12
      assert abs(law.fb_s - fb_s) <= 1e-12 and abs(law.fu_s - fu_s) <= 1e-12
      assert abs(law.fw_s - fw_s) <= 1e-12
      a_true = fb + (s - 1.0) * fu + d * fw
      a_m += (a_true - a_m) * (SL.DT / SL.TAU_S)
      worst = max(worst, abs(a_m - (law.fb_s + (s - 1.0) * law.fu_s + d * law.fw_s)))
  assert worst <= 1e-12


def _l6_cov_run(commands, v=1.0):
  law = SL.StopLaw(-0.1, [0.0] * (SL.ND + 1))
  bad = []
  for k, cmd in enumerate(commands):
    law.step(v, -0.1 + 0.05 * math.sin(0.1 * k), 8.0)
    law.commit(cmd)
    det = law.p00 * law.p11 - law.p01 ** 2
    if not (law.p00 >= 0.0 and law.p11 >= 0.0 and det >= -1e-15 and law.p00 + law.p11 <= SL.P_MAX + 1e-9):
      bad.append((k, law.p00, law.p11, det))
  return bad


@pytest.mark.parametrize("name,commands", [
  ("glide", [0.0] * 1000),
  ("impulse", [(-2.0 if (k // 30) % 2 == 0 else 0.0) for k in range(1000)]),
])
def test_l6_covariance_invariants(name, commands):
  bad = _l6_cov_run(commands)
  assert bad == [], "%s: %d violations, first %s" % (name, len(bad), bad[:3])


def _l6_adapt_state(kind, sign, locked=False):
  """One _adapt call on a prepared state; returns (before, after) tuples of th0/th1/p00/p01/p11."""
  law = SL.StopLaw(-0.5, [(-0.2 if kind == "db" else -1.0)] * (SL.ND + 1))
  law.v_prev = 1.0
  law.locked = locked
  if kind == "rel":
    # Command history is flat 0.0, so u_d == G*0.0 == 0.0 > u_f + FU_REL once u_f is negative.
    law.uh = [0.0] * (2 * SL.ND + 2)
    law.u_f = SL.G * -1.0
  law._adapt(-0.5)
  before = (law.th0, law.th1, law.p00, law.p01, law.p11)
  law._adapt(-0.5 + (0.5 if sign > 0 else -0.5))
  return before, (law.th0, law.th1, law.p00, law.p01, law.p11)


def test_l6_deadband_veto():
  # L7 A2: inside the deadband both res signs are vetoed while th1 >= 0 (a weak brake fakes a
  # -offset), and a +res may only correct a negative th1.
  assert abs(SL.G * -0.2) < SL.U_DB
  for sign in (+1, -1):
    before, after = _l6_adapt_state("db", sign)
    assert after == before
  law = SL.StopLaw(-0.5, [-0.2] * (SL.ND + 1))
  law.v_prev = 1.0
  law.th1 = -0.20
  law._adapt(-0.5)
  before = law.th1
  law._adapt(-0.0)                     # res > 0
  assert law.th1 > before


def test_l6_release_veto():
  # Inside the deadband the A2 guard vetoes both signs while th1 >= 0; the release guard outside
  # the deadband is test_l7_a4a.
  for locked in (False, True):
    for sign in (+1, -1):
      before, after = _l6_adapt_state("rel", sign, locked=locked)
      assert after == before


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_l6_box_slew_rate(seed):
  rnd = np.random.default_rng(seed)
  law = SL.StopLaw(-0.3, [-0.3] * (SL.ND + 1))
  up = law.u_prev
  t0, t1 = law.th0, law.th1
  viol = []
  for k in range(1500):
    u = law.step(max(0.0, 3.0 - 0.002 * k), float(rnd.uniform(-1.5, 0.5)), max(0.2, 15.0 - 0.01 * k))
    law.commit(u)
    if not (SL.TH_LO[0] <= law.th0 <= SL.TH_HI[0] and SL.TH_LO[1] <= law.th1 <= SL.TH_HI[1]):
      viol.append(("box", k, law.th0, law.th1))
    if abs(law.th0 - t0) > SL.TH0_SLEW * SL.DT + 1e-12 or abs(law.th1 - t1) > SL.TH1_SLEW * SL.DT + 1e-12:
      viol.append(("slew", k, law.th0 - t0, law.th1 - t1))
    if abs(u - up) > SL.DU_LIM * SL.DT + 1e-12:
      viol.append(("rate", k, u - up))
    t0, t1, up = law.th0, law.th1, u
  assert viol == [], "seed %d: %d violations, first %s" % (seed, len(viol), viol[:3])


def test_l6_constants():
  assert SL.TAU_S == 0.12
  assert SL.P_MAX == 0.60
  assert SL.GUARD is True
  assert SL.SLEW is True
  assert SL.U_DB == 0.40
  assert SL.FU_REL == 0.12
  assert SL.TH0_SLEW == 0.6
  assert SL.TH1_SLEW == 1.2
  assert SL.TH_LO == (-0.25, -0.40)
  assert SL.TH_HI == (0.40, 1.0)
  assert SL.U_MAX == -0.15
  assert SL.DEN_LOCK == 0.35
  assert SL.DEN_FLOOR == 0.05
  assert SL.P0 == (0.04, 0.09)
  assert SL.LAM == 0.98


def test_release_ramp_ends_at_zero_before_handover(monkeypatch):
  # The hand-back ramp owns the tick on which it reaches 0.0; the ordinary command takes over on the next tick.
  # Law off (no StopCtl), so only the hand-back block acts; the ordinary command here asks for +0.5 m/s^2.
  silence_cloudlog(monkeypatch)
  control, params = make_control(monkeypatch)
  control._stop_ctl = None
  plant = NominalPlant(1.0, a0=0.0)
  for _ in range(30):
    cs, plan, radar = tick_inputs(plant, 80.0, lead_status=False, a_target=0.0, v_target_now=plant.v, should_stop=False)
    control.update(True, cs, plan, LIMITS, AGE, radar)
  assert control.long_control_state == LongCtrlState.pid
  control._stop_release = -0.0197
  outs = []
  for _ in range(2):
    cs, plan, radar = tick_inputs(plant, 80.0, lead_status=False, a_target=0.5, v_target_now=plant.v + 0.5,
                                  should_stop=False)
    outs.append(float(control.update(True, cs, plan, LIMITS, AGE, radar)[0]))
    if len(outs) == 1:
      assert control._stop_release is None
  assert outs[0] == 0.0
  assert outs[0] - (-0.0197) <= lc.STOP_LAW_RELEASE_JERK * DT_CTRL + 1e-12
  assert outs[1] > 0.0


# --- creeping-lead and plan-go exits (Neon rulings 1120d + 1120e; field 000004d2--14bfd535b3--4) ---
def _kin_run(ctl, n, lead_x, v_lead_f, a_target_f, v0=3.0, a_cmd=-0.7):
  # Kinematic ego (constant decel), not closed loop: only the StopCtl gates are under test here.
  _warm(ctl)
  s, v, out = 0.0, v0, []
  for k in range(n):
    t = k * SL.DT
    u = ctl.update(True, v, a_cmd if v > 0.0 else 0.0, True, lead_x(t) - s, v_lead_f(t), a_target_f(t), INF)
    out.append((t, v, u, ctl.why if u is None else ""))
    ctl.commit(u if u is not None else -0.5)
    v = max(0.0, v + a_cmd * SL.DT)
    s += v * SL.DT
    if u is None and k > 0:
      break
  return out


def _prof(pts, n, x0=12.8):
  # Lead speed piecewise-linear through (t, v) points (held after the last), position integrated at DT.
  vl, xs, x = [], [], x0
  for k in range(n):
    t = k * SL.DT
    v = pts[-1][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
      if t0 <= t < t1:
        v = v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        break
    vl.append(v)
    xs.append(x)
    x += v * SL.DT
  idx = lambda t: min(int(round(t / SL.DT)), n - 1)
  return (lambda t: xs[idx(t)]), (lambda t: vl[idx(t)])


def _old_ctl():
  # 6833e0170 behaviour: neither new exit can fire and the re-entry lock releases on the next tick
  return SL.StopCtl(VL_CREEP=INF, PG_A=INF, VL_REIN=INF)


def test_p10_creeping_lead_ends_law_before_lock():
  for vl, t_max in ((1.0, 1.45), (0.8, 1.65)):
    lead_x = lambda t, vl=vl: 12.8 + max(0.0, t - 0.3) * vl
    v_lead = lambda t, vl=vl: 0.0 if t < 0.3 else vl
    tr = _kin_run(SL.StopCtl(), 400, lead_x, v_lead, lambda t: -0.5)
    t, v, u, why = tr[-1]
    assert u is None and why == "lead_go" and t <= t_max and v > SL.V_FLOOR
    old = _kin_run(_old_ctl(), 400, lead_x, v_lead, lambda t: -0.5)
    assert len(old) == 400 and all(r[2] is not None for r in old[1:])


def test_p11_vlead_noise_on_stopped_lead_changes_nothing():
  v_lead = lambda t: 1.2 if abs(t - 0.5) < 0.005 else (0.8 if 1.0 <= t < 1.5 else 0.0)
  new = _kin_run(SL.StopCtl(), 380, lambda t: 12.8, v_lead, lambda t: -0.5)
  old = _kin_run(_old_ctl(), 380, lambda t: 12.8, v_lead, lambda t: -0.5)
  assert len(new) == len(old) == 380
  assert all(a[2] == b[2] for a, b in zip(new, old))


def test_p12_plan_go_releases_the_held_command():
  for n_pos, on in ((19, True), (20, False)):
    c = SL.StopCtl()
    _warm(c)
    _tick(c)
    c.commit(c.u_out)
    for _ in range(80):
      c.commit(_tick(c, v=0.7, a=-0.7, d_rel=4.8 + 0.4))
    assert c.frozen
    for _ in range(n_pos):
      u = _tick(c, v=0.7, a=-0.7, d_rel=4.8 + 0.4, a_target=0.1)
      if u is not None:
        c.commit(u)
    assert (c.law is not None) == on
    if not on:
      assert c.why == "plan_go" and not c.creep_lock
  c = SL.StopCtl()
  _warm(c)
  _tick(c)
  c.commit(c.u_out)
  for _ in range(20):
    u = _tick(c, v=0.2, a=-0.5, d_rel=4.8 + 0.2, a_target=0.1)
    if u is not None:
      c.commit(u)
  assert c.law is None and c.why == "plan_go" and not c.frozen


def test_p13_stopped_lead_stop_unchanged():
  new = _kin_run(SL.StopCtl(), 500, lambda t: 9.0, lambda t: 0.0, lambda t: -0.6)
  old = _kin_run(_old_ctl(), 500, lambda t: 9.0, lambda t: 0.0, lambda t: -0.6)
  assert len(new) == len(old) == 500
  assert all(a[2] == b[2] for a, b in zip(new, old))


def test_p14_lead_still_slowing_never_creep_exits():
  # fleet 42a--7 / 479--3 / 4a6--6 / 4b7--1: entry just under VL_IN=1.0, the lead slows to ~0.65, lingers, then stops
  n = 450
  lead_x, v_lead = _prof([(0.0, 0.99), (1.2, 0.65), (1.8, 0.70), (2.4, 0.65), (3.0, 0.70), (3.2, 0.65), (4.5, 0.0)], n)
  new = _kin_run(SL.StopCtl(VL_IN=1.0), n, lead_x, v_lead, lambda t: -0.5)
  old = _kin_run(SL.StopCtl(VL_IN=1.0, VL_CREEP=INF, PG_A=INF, VL_REIN=INF), n, lead_x, v_lead, lambda t: -0.5)
  assert len(new) == len(old) == n
  assert all(a[2] == b[2] for a, b in zip(new, old))


def test_p15_lead_go_locks_reentry_until_the_lead_slows():
  c, o = SL.StopCtl(VL_IN=1.0), SL.StopCtl(VL_IN=1.0, VL_CREEP=INF, PG_A=INF, VL_REIN=INF)
  for ctl in (c, o):
    _warm(ctl)
    u = _tick(ctl)
    assert u is not None
    ctl.commit(u)
    assert _tick(ctl, v_lead=1.6) is None and ctl.why == "lead_go"
    ctl.commit(-0.5)
  assert c.creep_lock
  assert _tick(o, v_lead=0.8) is not None          # 6833e0170: re-enters on the next tick
  for _ in range(50):
    assert _tick(c, v_lead=0.8) is None           # locked while the lead stays above VL_REIN
    c.commit(-0.5)
  assert c.creep_lock
  assert _tick(c, v_lead=0.4) is not None and not c.creep_lock
  for release in ("idle", "fast", "lead"):
    c = SL.StopCtl()
    _warm(c)
    c.commit(_tick(c))
    assert _tick(c, v_lead=1.6) is None and c.creep_lock
    c.commit(-0.5)
    if release == "idle":
      c.idle()
    elif release == "fast":
      assert _tick(c, v=4.5, v_lead=0.8) is None
    else:
      assert _tick(c, lead_status=False, v_lead=0.8) is None
    assert not c.creep_lock


def test_p16_bounce_below_the_rise_never_creep_exits():
  # fleet 4a5--39: the lead slows 0.80 -> 0.58, bounces to 1.15 (rise 0.57 < DV_RISE) for 1 s, then stops
  n = 450
  lead_x, v_lead = _prof([(0.0, 0.80), (0.6, 0.58), (1.4, 1.15), (2.4, 1.15), (3.9, 0.0)], n)
  new = _kin_run(SL.StopCtl(VL_IN=1.0), n, lead_x, v_lead, lambda t: -0.5)
  old = _kin_run(SL.StopCtl(VL_IN=1.0, VL_CREEP=INF, PG_A=INF, VL_REIN=INF), n, lead_x, v_lead, lambda t: -0.5)
  assert len(new) == len(old) == n
  assert all(a[2] == b[2] for a, b in zip(new, old))


def test_p17_rise_from_a_slow_lead_ends_law():
  # field 4d1--9: the lead slows to 0.3 and pulls away at 1.3 (below VL_OUT): lead_go once, no re-entry
  n = 400
  lead_x, v_lead = _prof([(0.0, 0.9), (0.5, 0.3), (1.0, 1.3)], n)
  c = SL.StopCtl(VL_IN=1.0)
  tr = _kin_run(c, n, lead_x, v_lead, lambda t: -0.5)
  t, v, u, why = tr[-1]
  assert u is None and why == "lead_go" and 1.0 < t <= 2.5 and c.creep_lock
  old = _kin_run(SL.StopCtl(VL_IN=1.0, VL_CREEP=INF, PG_A=INF, VL_REIN=INF), n, lead_x, v_lead, lambda t: -0.5)
  assert len(old) == n and all(r[2] is not None for r in old[1:])


# --- L7: minimum brake, deadband/release guard polarity, one-axis forgetting, locked den floor ---

def test_l7_a1_law_never_releases_fully():
  # A1. The clip was U_MAX 0.0, so once the car reached the stop point with a released command and
  # the measured decel led the model, u_out clamped at exactly 0.000 and idle creep pushed the car
  # (W-LAWQ1 A-1 table: u_out 0.000 on 4d7--16, 991.0-991.7). The clip is now U_MAX = -0.15.
  assert SL.U_MAX == -0.15
  law = SL.StopLaw(0.0, [0.0] * (SL.ND + 1))       # entry with the brake released
  law.th0 = 0.0
  law.th1 = 0.0
  us = []
  for _ in range(60):
    u = law.step(0.02, -0.35, 0.02)                # at the stop point, measured decel ahead of the model
    law.commit(u)
    us.append(u)
  assert all(u < 0.0 for u in us), max(us)          # base's first tick is exactly 0.000
  # U_MAX is the clamp the rate limiter walks to, and it does get there
  assert us[-1] <= SL.U_MAX + 1e-12, us[-1]
  clamped = [k for k, u in enumerate(us) if abs(u - SL.U_MAX) < 1e-12]
  assert clamped, "U_MAX never bound"
  assert max(us) - min(us) <= SL.DU_LIM * SL.DT * (len(us) + 1) + 1e-9


def test_l7_a2_deadband_guard_polarity():
  # A2. The guard vetoed only res > 0, so a res < 0 at small |u| walked th1 to -0.479 (W-LAWQ1
  # 991.499). Now res < 0 is vetoed too, and a +res may only correct a negative th1.
  def run(res, th1_0, u_db=-0.2):
    law = SL.StopLaw(-0.5, [u_db] * (SL.ND + 1))
    law.v_prev = 1.0
    law.th1 = th1_0
    law._adapt(-0.5)
    before = law.th1
    law._adapt(-0.5 + res)
    return before, law.th1

  assert abs(SL.G * -0.2) < SL.U_DB
  # a res < 0 no longer moves th1 inside the deadband
  assert run(-0.5, 0.0) == (0.0, 0.0)
  # disconfirm the old polarity on the state that used to move it
  assert run(+0.5, 0.0) == (0.0, 0.0)
  # a negative th1 is still correctable by a +res
  b, a = run(+0.5, -0.20)
  assert a > b
  # the new floor holds
  assert SL.TH_LO == (-0.25, -0.40)
  law = SL.StopLaw(-1.0, [-0.2] * (SL.ND + 1))
  law.v_prev = 1.0
  for _ in range(600):
    law._adapt(-1.5)
  assert law.th1 >= -0.40 - 1e-12, law.th1


def _rel_prep(locked):
  """A law with the th1 axis excited, a command outside the deadband, and a fast release pending
  (u_d > u_f + FU_REL), so the release guard is the only thing deciding the update."""
  law = SL.StopLaw(-0.5, [-0.5] * (SL.ND + 1))
  law.v_prev = 1.0
  for k in range(40):                      # excite fw_s below W_V_HI
    u = law.step(max(0.2, 1.6 - 0.01 * k), -0.5, 6.0)
    law.commit(u)
  law.uh = [-0.5] * (2 * SL.ND + 2)        # u_d = G*-0.5, |u_d| > U_DB
  law.u_f = SL.G * -1.0                    # u_d > u_f + FU_REL
  law.locked = locked
  return law


def _rel_step(law, res):
  """One _adapt whose residual is exactly `res`, built from this law's own prediction."""
  p0, p1 = law.fu_s, law.fw_s
  return law._adapt(law.fb_s + p0 * law.th0 + p1 * law.th1 + res)


def test_l7_a4a_release_guard_runs_on_locked_ticks():
  # A4a. The guard skipped every locked tick, so a locked fast release with res > 0 moved th1
  # (-0.043 -> +0.231 on 4d6--6). The guard now runs while locked and vetoes that false +res; it
  # leaves the genuine res < 0 and the unlocked behaviour alone.
  law = _rel_prep(True)
  before = law.th1
  _rel_step(law, +0.30)
  assert law.th1 == before, (before, law.th1)     # base moves th1 up here
  law = _rel_prep(True)
  before = law.th1
  _rel_step(law, -0.30)
  assert law.th1 < before                         # a real -res still adapts
  law = _rel_prep(False)
  before = law.th1
  _rel_step(law, +0.30)
  assert law.th1 == before
  law = _rel_prep(False)
  before = law.th1
  _rel_step(law, -0.30)
  assert law.th1 == before                        # unlocked res < 0 keeps today's veto


def test_l7_a3_locked_den_floor():
  # A3. The locked branch floored den_l at DEN_FLOOR 0.05, so a_kin blew up once d_stop went
  # negative (-0.4^2/(2*0.05) = -1.6, and -1.40 after the J_FIRM slew). It floors at DEN_LOCK 0.35
  # now, so the locked hold stays at A_H; the pre-lock q path and the lead guard keep DEN_FLOOR.
  assert SL.DEN_LOCK == 0.35
  assert SL.DEN_FLOOR == 0.05
  assert -0.4 ** 2 / (2.0 * SL.DEN_LOCK) > SL.A_H
  assert -0.4 ** 2 / (2.0 * SL.DEN_FLOOR) < SL.A_H

  law = SL.StopLaw(-0.7, [-0.7] * (SL.ND + 1), 0.3)
  law.locked = True
  law.r_prev = SL.A_H
  r = None
  for _ in range(300):
    r = law._ref(0.4, -0.1)
  assert r == pytest.approx(SL.A_H), r            # base reached -1.40 here

  # the lead guard still floors at DEN_FLOOR through its own 0.05 clip and still binds
  ctl = SL.StopCtl(V_ENTER_MIN=0.0, NEED_PLAN_BRAKE=False)
  for _ in range(30):
    ctl.commit(-0.8)
  ctl.update(True, 1.5, -0.5, True, 6.0, 0.0, -1.0, INF)   # enter the law normally
  ctl.law = SL.StopLaw(-0.5, [-0.5] * 64)                  # fresh law: the guard is what binds
  u = ctl.update(True, 1.5, -0.5, True, 3.1, 0.0, -1.0, INF)
  assert ctl.n_guard == 1
  assert u is not None and u < SL.A_H, u


def test_p18_creeping_lead_matching_ego_speed_never_enters():
  """Regression for 000004e3 seg 7: a rolling lead matched closely by ego speed must not trigger StopLaw."""
  ctl = SL.StopCtl(GAP=4.3)
  _warm(ctl, u=-0.4)
  # Ego creeping at 1.22 -> 1.02 behind lead rolling at 0.92 -> 0.65 (closing speed 0.30..0.47 < DV_ENTER).
  for k in range(100):
    v = 1.22 - 0.002 * k
    vl = 0.92 if k < 20 else 0.68
    u = ctl.update(True, v, -0.3, True, 5.7 - 0.004 * k, vl, -0.45, INF)
    assert u is None, (k, v, vl, u)
    ctl.commit(-0.4)
  assert ctl.n_enter == 0

  # Behind a genuinely stopped lead (v_lead < VL_STILL), slow ego approach still enters normally.
  ctl2 = SL.StopCtl(GAP=4.3)
  _warm(ctl2, u=-0.4)
  u2 = ctl2.update(True, 1.15, -0.3, True, 5.5, 0.30, -0.45, INF)
  assert u2 is not None and ctl2.n_enter == 1


def test_p19_rolling_lead_at_high_closing_speed_never_slams():
  """Regression for 000004e5 seg 7: a lead still rolling at 0.6..0.95 m/s must not trigger stationary StopLaw."""
  ctl = SL.StopCtl(GAP=4.3)
  _warm(ctl, u=-1.06)
  # Ego approaching at 2.52 -> 1.70 m/s behind lead rolling at 0.95 -> 0.62 m/s (d=7.02 -> 5.98 m).
  for k in range(150):
    v = max(1.1, 2.52 - 0.008 * k)
    vl = max(0.55, 0.95 - 0.0025 * k)
    d = max(5.6, 7.02 - 0.009 * k)
    u = ctl.update(True, v, -1.1, True, d, vl, -0.90, INF)
    assert u is None, (k, v, vl, u)
    ctl.commit(-0.90)
  assert ctl.n_enter == 0

  # Once the lead genuinely slows below VL_IN (0.5 m/s), StopLaw still takes over immediately.
  u_stop = ctl.update(True, 2.1, -1.0, True, 6.8, 0.45, -0.90, INF)
  assert u_stop is not None and ctl.n_enter == 1


def test_release_after_lead_go_clears_faster_and_ramps_positive_accel(monkeypatch):
  """Regression for 000004e3 seg 7: departing lead release avoids dead-stop hold and 1-tick accel surge."""
  silence_cloudlog(monkeypatch)
  control, _ = make_control(monkeypatch)
  control._stop_ctl = None
  plant = NominalPlant(0.30, a0=-0.45)
  for _ in range(25):
    cs, plan, radar = tick_inputs(plant, 10.0, lead_status=True, v_lead=2.0, a_target=0.0,
                                  v_target_now=plant.v, should_stop=False)
    control.update(True, cs, plan, LIMITS, AGE, radar)
  assert control.long_control_state == LongCtrlState.pid
  control._stop_release = -1.62
  outs = [-1.62]
  for _ in range(120):
    cs, plan, radar = tick_inputs(plant, 8.0, lead_status=True, v_lead=3.0, a_target=1.40,
                                  v_target_now=plant.v + 0.8, should_stop=False)
    outs.append(float(control.update(True, cs, plan, LIMITS, AGE, radar)[0]))
  t_zero = next((i * DT_CTRL for i, u in enumerate(outs) if u >= 0.0), 99.0)
  assert t_zero <= 0.45, f"brake release too slow: {t_zero:.2f}s"
  diffs = np.diff(outs)
  max_jerk = float(np.max(diffs)) / DT_CTRL
  assert max_jerk <= lc.STOP_LAW_RELEASE_GO_JERK + 1e-6, f"1-tick accel surge: {max_jerk:.1f} m/s^3"
  # Positive ramp obeys STOP_LAW_GO_JERK once above zero.
  pos_idx = next(i for i, u in enumerate(outs) if u >= 0.0)
  pos_diffs = np.diff(outs[pos_idx:])
  assert float(np.max(pos_diffs)) <= lc.STOP_LAW_GO_JERK * DT_CTRL + 1e-9
  assert outs[-1] > 1.0, f"did not smoothly hand over to positive accel: {outs[-1]}"
