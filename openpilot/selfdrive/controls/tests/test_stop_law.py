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


def test_t10_stoplaw_fixture():
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
def test_t11_stopctl_fixture(i):
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
def test_t13a_stopctl_tol_fixture_at_default(i):
  # recorded with TWIN/PORT2/stop_law2.py at TOL_LOCK 0.3 (the law the sweep ran); stored cfg omits TOL_LOCK
  mism, worst = _replay_ctl(os.path.join(FIX, "stopctl_tol_fixture.npz"), i, {})
  assert mism == 0
  assert worst <= 1e-9


def test_t13b_tol_lock_default_and_teeth():
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
