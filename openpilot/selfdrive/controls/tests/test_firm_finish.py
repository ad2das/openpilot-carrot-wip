"""ButterStopFinal firm finish: the band gate, the cap ramp/release math, the
switch resolution, and the LongControl wiring (off-identity, the on-stream
floor, and the gas exit).

The firm finish is the classic-CAN Hyundai/Kia half of ButterStopFinal. F6 pins
which cars resolve it: Palisade with the switch on gets FIRM_A, CAN-FD and other
brands get 0. F7 pins off-identity: with the switch off the cap block is not
entered at all, and the stop stream is tick-for-tick the same as before. F8 pins
the floor the cap actually applies, F9 that a gas press drops it at once. The
FIRM_* constants are module values and nothing here tunes them.
"""
import math
import os
import sys
import tempfile
from types import ModuleType

import numpy as np
import pytest

# Same Windows import shims as test_butter_stop.py and test_early_req.py:
# these tests never construct the real Params backend or the solver.
if sys.platform == "win32":
  hardware_module = ModuleType("openpilot.system.hardware")
  hardware_module.PC = True
  sys.modules.setdefault("openpilot.system.hardware", hardware_module)

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
from openpilot.common.realtime import DT_CTRL
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.selfdrive.controls.lib.cruise_coasting import MAX_PLAN_AGE
from openpilot.selfdrive.controls.lib.butter_stop import EARLY_REQ_ENABLED, PROBE_V_MAX
from openpilot.selfdrive.controls.lib.longcontrol import (
  FIRM_A, FIRM_JERK, FIRM_V_HI, FIRM_V_LO, LongCtrlState,
  firm_cap_step, firm_gate,
)
from openpilot.selfdrive.controls.tests.test_butter_stop import (
  PLAN_T, DictParams, make_control, make_cp, stopping_inputs,
)

LEAD_D = 9.0


class _EventLog:
  def __init__(self):
    self.events = []

  def event(self, name, **kwargs):
    self.events.append((name, kwargs))

  def names(self, name):
    return [kw for ev, kw in self.events if ev == name]


def _control_with_cp(monkeypatch, values, cp):
  params = DictParams({"StoppingAccel": -50, "StopDistanceCarrot": 550, "ButterStop": 1,
                       "LongTuningKpV": 100, "LongTuningKiV": 200, "LongTuningKf": 100, **values})
  monkeypatch.setattr(lc, "Params", lambda: params)
  return lc.LongControl(cp), params


def _canfd_cp():
  cp = make_cp()
  cp.flags = int(HyundaiFlags.CANFD)
  return cp


# ---------------------------------------------------------------------------
# F1: the band gate and every single-input rejection
# ---------------------------------------------------------------------------

GATE_BASE = dict(firm_a=0.8, v_ego=1.0, a_target=-0.4, plan_speeds=[1.0, 0.5, 0.0],
                 t_since_plan=0.05, gas=False, brake=False, cruise_standstill=False)


def _gate(**overrides):
  args = dict(GATE_BASE)
  args.update(overrides)
  return firm_gate(args["firm_a"], args["v_ego"], args["a_target"], args["plan_speeds"],
                   args["t_since_plan"], args["gas"], args["brake"], args["cruise_standstill"])


def test_gate_passes_on_the_base_inputs():
  assert _gate() is True


@pytest.mark.parametrize("field,value", [
  ("cruise_standstill", True),
  ("gas", True),
  ("brake", True),
  ("v_ego", 2.01),
  ("v_ego", 0.29),
  ("a_target", -0.3),
  ("plan_speeds", [1.0, 0.5, 0.3]),
  ("a_target", float("nan")),
  ("t_since_plan", -0.01),
  ("t_since_plan", MAX_PLAN_AGE + 0.01),
  ("firm_a", 0.0),
  ("plan_speeds", []),
])
def test_gate_rejects_one_bad_input(field, value):
  assert _gate(**{field: value}) is False, field


@pytest.mark.parametrize("v_ego", [2.0, 0.3])
def test_gate_band_edges_are_inclusive(v_ego):
  assert _gate(v_ego=v_ego) is True


def test_gate_rejects_a_non_finite_plan_end():
  assert _gate(plan_speeds=[1.0, float("nan")]) is False


# ---------------------------------------------------------------------------
# F2-F5: the cap ramp and release math
# ---------------------------------------------------------------------------

def test_cap_builds_from_last_output_at_firm_jerk():
  cap = None
  values = []
  for _ in range(30):
    cap = firm_cap_step(cap, -0.4, 0.8, True, False)
    values.append(cap)
  assert values[0] == pytest.approx(-0.42, abs=1e-9)
  reached = [i for i, value in enumerate(values) if value <= -0.8 + 1e-9]
  assert reached[0] == 19, reached[:3]
  assert values[19] == pytest.approx(-0.8, abs=1e-9)
  assert values[29] == pytest.approx(-0.8, abs=1e-9)
  assert values[18] > -0.8 + 1e-9, values[18]


def test_cap_step_floors_a_small_last_output():
  # min(0.2, 0.0) - 0.02: a positive last output still yields braking at the jerk rate.
  assert firm_cap_step(None, 0.2, 0.8, True, False) == pytest.approx(-FIRM_JERK * DT_CTRL)


def test_cap_step_never_exceeds_firm_a():
  # -1.2 is already firmer than the cap: the cap stays at -firm_a, never deeper.
  assert firm_cap_step(None, -1.2, 0.8, True, False) == pytest.approx(-0.8)


def test_cap_release_reaches_none_at_40_calls():
  cap = -0.8
  first_none = None
  for call in range(1, 200):
    cap = firm_cap_step(cap, 0.0, 0.8, False, False)
    if cap is None:
      first_none = call
      break
  assert first_none == 40, first_none
  assert firm_cap_step(-0.8, 0.0, 0.8, False, True) is None
  assert firm_cap_step(None, 0.0, 0.8, False, False) is None


# ---------------------------------------------------------------------------
# F6: which cars resolve the firm finish from the ButterStopFinal switch
# ---------------------------------------------------------------------------

ON = {"ButterStop": 1, "ButterStopFinal": 1}
OFF = {"ButterStop": 1, "ButterStopFinal": 0}


def test_f6a_palisade_with_the_switch_on_gets_the_firm_finish(monkeypatch):
  control, _ = make_control(monkeypatch, values=ON)
  assert control.firm_a == pytest.approx(0.8)
  assert control.final_stop is True
  assert control.early_req is False, "the early hand-off is held off by EARLY_REQ_ENABLED"


def test_f6b_switch_off_is_zero(monkeypatch):
  control, _ = make_control(monkeypatch, values=OFF)
  assert control.firm_a == 0.0
  assert control.early_req is False


def test_f6c_butter_stop_off_is_zero(monkeypatch):
  """final_stop is only computed when ButterStop is on (the early return)."""
  control, _ = make_control(monkeypatch, values={"ButterStop": 0, "ButterStopFinal": 1})
  assert control.final_stop is False
  assert control.firm_a == 0.0
  assert control.early_req is False


def test_f6d_canfd_gets_the_probe_instead_of_the_firm_finish(monkeypatch):
  """CAN-FD Hyundai/Kia now drive the K5 measurement window, not the firm finish."""
  control, _ = _control_with_cp(monkeypatch, ON, _canfd_cp())
  assert control._early_req_capable is False
  assert control._final_window_capable is True
  assert control.firm_a == 0.0
  assert control.final_window is True, "CAN-FD Hyundai/Kia run the probe"


def test_f6e_other_brand_stays_off(monkeypatch):
  control, _ = make_control(monkeypatch, values=ON, brand="toyota")
  assert control._early_req_capable is False
  assert control._final_window_capable is False
  assert control.firm_a == 0.0
  assert control.final_window is False


def test_firm_a_is_the_module_constant(monkeypatch):
  """The decel is a module constant, not a param: nothing can retune it per car."""
  control, _ = make_control(monkeypatch, values=ON)
  assert control.firm_a == pytest.approx(FIRM_A)
  assert FIRM_A == pytest.approx(0.8)


def test_early_req_constant_is_off():
  assert EARLY_REQ_ENABLED is False


# ---------------------------------------------------------------------------
# The stop stream shared by F7, F8 and F9
# ---------------------------------------------------------------------------

def _build_stream(ticks=600, v_start=2.5, decel=0.5):
  dt = DT_CTRL
  out = []
  v = v_start
  for _ in range(ticks):
    out.append(max(v, 0.0))
    v = max(0.0, v - decel * dt)
  return out


def _tick_inputs(v):
  """Plan of a -0.5 m/s^2 stop: aTarget -0.4, shouldStop only below 0.3 m/s."""
  cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.5 * PLAN_T),
                                    lead_status=True, d_rel=LEAD_D)
  cs.aEgo = -0.5
  plan.aTarget = -0.4
  plan.shouldStop = bool(v < 0.3)
  return cs, plan, radar


def _probe_pid(control):
  """Record what the pid asked for, before the cap or the guard override it."""
  calls = []
  orig = control.pid.update

  def _wrapped(*args, **kwargs):
    out = orig(*args, **kwargs)
    calls.append(float(out))
    return out

  control.pid.update = _wrapped
  return calls


def _run_stream(monkeypatch, values, stream, gas_from=None):
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = make_control(monkeypatch, values=values)
  control.long_control_state = LongCtrlState.pid
  pid_calls = _probe_pid(control)

  rows = []
  for tick, v in enumerate(stream):
    cs, plan, radar = _tick_inputs(v)
    if gas_from is not None and tick >= gas_from:
      cs.gasPressed = True
    pid_calls.clear()
    integral_before = float(control.pid.i)
    output = float(control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0])
    rows.append({
      "tick": tick, "v": float(cs.vEgo), "output": output,
      "pid_wanted": pid_calls[-1] if pid_calls else float("nan"),
      "state": int(control.long_control_state), "integral_before": integral_before,
      "integral_after": float(control.pid.i),
      "cap": None if control._firm_cap is None else float(control._firm_cap),
      "gas": bool(cs.gasPressed),
    })
  return control, rows, events


def _run_stream_with_cp(monkeypatch, values, stream, cp):
  """Same stream, but with an explicit CP so the capability gate is exercised."""
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = _control_with_cp(monkeypatch, values, cp)
  control.long_control_state = LongCtrlState.pid
  rows = []
  for tick, v in enumerate(stream):
    cs, plan, radar = _tick_inputs(v)
    integral_before = float(control.pid.i)
    output = float(control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0])
    rows.append({"tick": tick, "v": float(cs.vEgo), "output": output,
                 "state": int(control.long_control_state), "integral_before": integral_before,
                 "integral_after": float(control.pid.i)})
  return control, rows, events


# ---------------------------------------------------------------------------
# F7: the switch off is invisible
# ---------------------------------------------------------------------------

def _assert_identical(left, right, label):
  assert len(left) == len(right)
  for a, b in zip(left, right, strict=True):
    assert a["output"] == b["output"], (label, a["tick"], a["output"], b["output"])
    assert a["state"] == b["state"], (label, a["tick"], a["state"], b["state"])
    assert a["integral_after"] == b["integral_after"], (label, a["tick"])


def test_switch_off_never_enters_the_block(monkeypatch):
  """With the switch off the block is not entered: neither helper is called."""
  calls = {"gate": 0, "step": 0}
  real_gate, real_step = lc.firm_gate, lc.firm_cap_step

  def spy_gate(*args, **kwargs):
    calls["gate"] += 1
    return real_gate(*args, **kwargs)

  def spy_step(*args, **kwargs):
    calls["step"] += 1
    return real_step(*args, **kwargs)

  monkeypatch.setattr(lc, "firm_gate", spy_gate)
  monkeypatch.setattr(lc, "firm_cap_step", spy_step)

  # Full-length stream: the gate needs plan_speeds[-1] < 0.3, i.e. v < ~1.55, so a
  # short high-speed slice would prove only the off arm and never enter the block.
  stream = _build_stream()
  _, rows_off, _ = _run_stream(monkeypatch, OFF, stream)
  assert rows_off, "no ticks ran"
  assert calls == {"gate": 0, "step": 0}, calls

  # Same spy, switch on: the block is entered, so the spy is proven live.
  _, rows_on, _ = _run_stream(monkeypatch, ON, stream)
  assert calls["gate"] > 0 and calls["step"] > 0, calls
  assert any(r["cap"] is not None for r in rows_on)


def test_switch_off_matches_a_controller_that_never_had_it(monkeypatch):
  """Off identity: a full stop is the same as ButterStop 1 with Final 0 on base."""
  stream = _build_stream()
  _, off_rows, _ = _run_stream(monkeypatch, OFF, stream)
  _, base_rows, _ = _run_stream(monkeypatch, {"ButterStop": 1, "ButterStopFinal": 0}, stream)
  _assert_identical(off_rows, base_rows, "off_vs_base")


def test_canfd_with_the_switch_on_does_not_enter_the_firm_block(monkeypatch):
  """The firm finish belongs to classic CAN: on CAN-FD the switch selects the probe.

  The probe owns the command below PROBE_V_MAX, so the arm-1 stream is no longer
  identical to arm 0 there. What this pins is that the firm block itself stays
  out of it: no gate call, no cap, and every tick above the probe's speed range
  is unchanged.
  """
  calls = {"gate": 0, "step": 0}
  real_gate, real_step = lc.firm_gate, lc.firm_cap_step

  def spy_gate(*args, **kwargs):
    calls["gate"] += 1
    return real_gate(*args, **kwargs)

  def spy_step(*args, **kwargs):
    calls["step"] += 1
    return real_step(*args, **kwargs)

  monkeypatch.setattr(lc, "firm_gate", spy_gate)
  monkeypatch.setattr(lc, "firm_cap_step", spy_step)
  stream = _build_stream()
  _, on_rows, _ = _run_stream_with_cp(monkeypatch, ON, stream, _canfd_cp())
  _, off_rows, _ = _run_stream_with_cp(monkeypatch, OFF, stream, _canfd_cp())
  assert calls == {"gate": 0, "step": 0}, calls
  above = [(a, b) for a, b in zip(on_rows, off_rows, strict=True) if a["v"] > PROBE_V_MAX]
  assert above, "the stream must spend ticks above the probe's speed range"
  for a, b in above:
    assert a["output"] == b["output"], (a["tick"], a["v"], a["output"], b["output"])


# ---------------------------------------------------------------------------
# F8: the cap is the floor it promises, and only below the band top
# ---------------------------------------------------------------------------

def test_switch_on_holds_the_floor_through_the_band(monkeypatch):
  stream = _build_stream()
  control, rows, events = _run_stream(monkeypatch, ON, stream)

  assert control.firm_a == pytest.approx(0.8)

  first_gated = next((r["tick"] for r in rows if r["cap"] is not None), None)
  assert first_gated is not None, "the firm finish never engaged"

  band = [r for r in rows
          if r["state"] == int(LongCtrlState.pid) and FIRM_V_LO <= r["v"] <= FIRM_V_HI
          and r["tick"] >= first_gated + 20]
  assert len(band) >= 20, len(band)
  for row in band:
    assert row["output"] <= -0.8 + 1e-9, (row["tick"], row["v"], row["output"])

  # Exact binding test: the command that left update() is the cap itself.
  binding = [r for r in rows if r["cap"] is not None and abs(r["output"] - r["cap"]) <= 1e-12]
  assert binding, "the cap never bound the command"
  for row in binding:
    assert row["integral_after"] == row["integral_before"], (row["tick"], row["v"])
  cut = [r for r in binding if math.isfinite(r["pid_wanted"]) and r["pid_wanted"] > r["cap"] + 1e-12]
  assert cut, "the cap never cut a weaker pid command"
  for row in cut:
    assert row["integral_after"] == row["integral_before"], (row["tick"], row["v"], row["pid_wanted"])

  assert len(events.names("butterStopFirm")) == 1, events.events


def test_before_the_band_top_the_output_matches_the_switch_off(monkeypatch):
  stream = _build_stream()
  _, off_rows, _ = _run_stream(monkeypatch, OFF, stream)
  _, on_rows, _ = _run_stream(monkeypatch, ON, stream)
  above = [(a, b) for a, b in zip(on_rows, off_rows, strict=True) if a["v"] > FIRM_V_HI]
  assert above, "the stream never runs above the band top"
  for a, b in above:
    assert a["output"] == b["output"], (a["tick"], a["v"], a["output"], b["output"])
    assert a["state"] == b["state"], (a["tick"], a["v"])
    assert a["integral_after"] == b["integral_after"], (a["tick"], a["v"])


# ---------------------------------------------------------------------------
# F9: a gas press drops the cap at once
# ---------------------------------------------------------------------------

def test_gas_press_drops_the_cap(monkeypatch):
  stream = _build_stream()
  _, probe_rows, _ = _run_stream(monkeypatch, ON, stream)
  gated = [r["tick"] for r in probe_rows if r["cap"] is not None]
  assert len(gated) >= 30, len(gated)
  gas_from = gated[29]

  _, rows, _ = _run_stream(monkeypatch, ON, stream, gas_from=gas_from)
  assert rows[gas_from]["gas"] is True
  assert rows[gas_from]["cap"] is None, (rows[gas_from]["tick"], rows[gas_from]["cap"])
  for row in rows[gas_from:]:
    assert row["gas"] is True
    assert row["cap"] is None, (row["tick"], row["cap"])
