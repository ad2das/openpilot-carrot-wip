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
import bisect
import itertools
import math
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace

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
  FIRM_A, FIRM_JERK, FIRM_TRACK_ATGT, FIRM_TRACK_MAX, FIRM_TRACK_MIN, FIRM_V_HI, FIRM_V_LO,
  LongCtrlState, firm_cap_step, firm_gate, firm_track_step, firm_track_target,
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
  ("a_target", -0.2),
  ("plan_speeds", [1.0, 0.5, 0.5]),
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


def test_gate_opens_on_a_creeping_lead_stop_plan():
  # Field 4b7 s1: behind a stopped lead the plan creeps (aTarget -0.27, plan end 0.45 m/s
  # at v 1.0); the old -0.3 / 0.3 gate never opened and the stop crawled.
  assert _gate(a_target=-0.27, plan_speeds=[1.0, 0.7, 0.45]) is True
  assert _gate(a_target=-0.21, plan_speeds=[1.0, 0.49]) is True


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
  calls = {"gate": 0, "step": 0, "track": 0}
  real_gate, real_step, real_track = lc.firm_gate, lc.firm_cap_step, lc.firm_track_step

  def spy_gate(*args, **kwargs):
    calls["gate"] += 1
    return real_gate(*args, **kwargs)

  def spy_step(*args, **kwargs):
    calls["step"] += 1
    return real_step(*args, **kwargs)

  def spy_track(*args, **kwargs):
    calls["track"] += 1
    return real_track(*args, **kwargs)

  monkeypatch.setattr(lc, "firm_gate", spy_gate)
  monkeypatch.setattr(lc, "firm_cap_step", spy_step)
  monkeypatch.setattr(lc, "firm_track_step", spy_track)

  # Full-length stream: the gate needs plan_speeds[-1] < 0.5, i.e. v < ~1.75, so a
  # short high-speed slice would prove only the off arm and never enter the block.
  stream = _build_stream()
  _, rows_off, _ = _run_stream(monkeypatch, OFF, stream)
  assert rows_off, "no ticks ran"
  assert calls == {"gate": 0, "step": 0, "track": 0}, calls

  # Same spy, switch on: the block is entered, so the spy is proven live.
  _, rows_on, _ = _run_stream(monkeypatch, ON, stream)
  assert calls["gate"] > 0 and calls["track"] > 0, calls
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
  calls = {"gate": 0, "step": 0, "track": 0}
  real_gate, real_step, real_track = lc.firm_gate, lc.firm_cap_step, lc.firm_track_step

  def spy_gate(*args, **kwargs):
    calls["gate"] += 1
    return real_gate(*args, **kwargs)

  def spy_step(*args, **kwargs):
    calls["step"] += 1
    return real_step(*args, **kwargs)

  def spy_track(*args, **kwargs):
    calls["track"] += 1
    return real_track(*args, **kwargs)

  monkeypatch.setattr(lc, "firm_gate", spy_gate)
  monkeypatch.setattr(lc, "firm_cap_step", spy_step)
  monkeypatch.setattr(lc, "firm_track_step", spy_track)
  stream = _build_stream()
  _, on_rows, _ = _run_stream_with_cp(monkeypatch, ON, stream, _canfd_cp())
  _, off_rows, _ = _run_stream_with_cp(monkeypatch, OFF, stream, _canfd_cp())
  assert calls == {"gate": 0, "step": 0, "track": 0}, calls
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


# ---------------------------------------------------------------------------
# F10: the decel tracker inside the band (firm_track_target / firm_track_step)
# ---------------------------------------------------------------------------

def test_t1_track_target_units():
  """The target is -FIRM_A when the car decelerates at FIRM_A, and saturates."""
  cases = {(-2.0): -0.70, (-1.1): -0.70, (-1.0): -0.70, (-0.8): -0.80,
           (-0.6): -1.00, (-0.55): -1.05, (-0.4): -1.20, 0.0: -1.20, 0.5: -1.20}
  for a_ego, expected in cases.items():
    assert firm_track_target(0.8, a_ego) == pytest.approx(expected, abs=1e-9), a_ego


@pytest.mark.parametrize("cap,last,a_ego,ease,expected", [
  (None, -1.3, -1.1, True, -1.28),
  (None, -1.3, -1.1, False, -1.28),
  (None, -0.4, -0.5, True, -0.42),
  (None, 0.3, 0.0, True, -0.02),
  (-0.8, 0.0, -0.8, True, -0.80),
  (None, -0.8, float("nan"), True, -0.80),
])
def test_t2_track_step_one_call(cap, last, a_ego, ease, expected):
  assert firm_track_step(cap, last, 0.8, a_ego, ease) == pytest.approx(expected, abs=1e-9)


def test_t3a_ease_ramp_reaches_firm_track_min_at_30_calls():
  cap, calls = None, 0
  while True:
    cap = firm_track_step(cap, -1.3, 0.8, -1.1, True)
    calls += 1
    if cap >= -FIRM_TRACK_MIN - 1e-12 or calls > 1000:
      break
  assert calls == 30, calls
  assert cap == pytest.approx(-FIRM_TRACK_MIN, abs=1e-9)
  for _ in range(20):
    cap = firm_track_step(cap, -1.3, 0.8, -1.1, True)
    assert cap == pytest.approx(-FIRM_TRACK_MIN, abs=1e-9)


def test_t3b_lead_ramp_reaches_the_target_at_35_calls():
  cap, calls = None, 0
  while True:
    cap = firm_track_step(cap, -0.4, 0.8, -0.5, False)
    calls += 1
    if cap <= firm_track_target(0.8, -0.5) + 1e-12 or calls > 1000:
      break
  assert calls == 35, calls
  assert cap == pytest.approx(-1.1, abs=1e-9)


def test_t3c_without_ease_the_cap_never_rises_above_firm_a():
  cap = -1.3
  for _ in range(200):
    cap = firm_track_step(cap, -1.3, 0.8, -1.1, False)
    assert cap <= -0.8 + 1e-12, cap


# ---------------------------------------------------------------------------
# F11: closed loop against the fitted plants (copy of the parent's spec_orch)
# ---------------------------------------------------------------------------

def _table(points):
  xs = [p[0] for p in points]

  def _f(v):
    i = min(max(bisect.bisect_right(xs, v), 1), len(points) - 1)
    (x0, y0), (x1, y1) = points[i - 1], points[i]
    return y0 + (y1 - y0) * (v - x0) / (x1 - x0)
  return _f


P4B5 = [(0.0, -0.80), (1.13, -0.80), (1.38, -0.98), (1.64, -1.13), (1.92, -1.28), (2.20, -1.35),
        (2.50, -1.41), (3.06, -1.48), (3.54, -1.46), (9, -1.46)]
P4B1 = [(0.0, -0.80), (1.42, -0.80), (1.62, -0.83), (1.88, -0.90), (2.15, -0.98), (2.45, -1.11),
        (2.71, -1.17), (2.96, -1.22), (9, -1.22)]
PSLOW = [(0.0, -0.80), (9, -0.80)]

GRID = list(itertools.product((0.6, 0.75, 0.82, 0.9, 1.0), (0.6, 0.9, 1.1), (0.2, 0.3, 0.5)))
SCENARIOS = [("4b5 no lead", _table(P4B5), 3.54, -0.85, False),
             ("4b1 no lead", _table(P4B1), 2.96, -1.11, False),
             ("4b1 lead", _table(P4B1), 2.96, -1.11, True),
             ("4b5 lead", _table(P4B5), 3.54, -0.85, True),
             ("slow lead", _table(PSLOW), 1.06, -0.68, True)]

# Printed values of the parent's spec_orch.py (raw/spec_orch.txt). Every fitted
# row is (g, tau, d, old a@0.4, new a@0.4, old a@0.8, new a@0.8, old x_end, new x_end);
# every grid row is (old a@0.4 min, old a@0.4 max, new a@0.4 min, new a@0.4 max,
# extra min, extra max, new u min, new u max).
SPEC_EXPECTED = {
  "4b5 no lead": {
    "fitted": [(0.8, 0.7, 0.5, -0.832, -0.682, -0.983, -0.824, 6.011, 6.124),
               (0.85, 0.8, 0.3, -0.855, -0.727, -0.984, -0.825, 5.880, 6.017),
               (0.6, 1.1, 0.3, -0.590, -0.603, -0.675, -0.619, 7.491, 7.630)],
    "grid": (-1.172, -0.505, -1.043, -0.586, 0.033, 0.199, -1.283, -0.700),
  },
  "4b1 no lead": {
    "fitted": [(0.8, 0.7, 0.5, -0.679, -0.691, -0.727, -0.667, 4.583, 4.661),
               (0.85, 0.8, 0.3, -0.725, -0.728, -0.770, -0.720, 4.445, 4.532),
               (0.6, 1.1, 0.3, -0.528, -0.601, -0.574, -0.607, 5.280, 5.249)],
    "grid": (-0.938, -0.483, -0.864, -0.594, -0.274, 0.118, -1.023, -0.700),
  },
  "4b1 lead": {
    "fitted": [(0.8, 0.7, 0.5, -0.679, -0.689, -0.727, -0.727, 4.583, 4.580),
               (0.85, 0.8, 0.3, -0.725, -0.731, -0.770, -0.770, 4.445, 4.445),
               (0.6, 1.1, 0.3, -0.528, -0.601, -0.574, -0.611, 5.280, 5.214)],
    "grid": (-0.938, -0.483, -0.938, -0.594, -0.285, 0.000, -1.019, -0.800),
  },
  "4b5 lead": {
    "fitted": [(0.8, 0.7, 0.5, -0.832, -0.832, -0.983, -0.983, 6.011, 6.011),
               (0.85, 0.8, 0.3, -0.855, -0.855, -0.984, -0.984, 5.880, 5.880),
               (0.6, 1.1, 0.3, -0.590, -0.618, -0.675, -0.678, 7.491, 7.481)],
    "grid": (-1.172, -0.505, -1.172, -0.572, -0.069, 0.000, -1.300, -0.800),
  },
  "slow lead": {
    "fitted": [(0.8, 0.7, 0.5, -0.649, -0.699, -0.663, -0.663, 0.786, 0.773),
               (0.85, 0.8, 0.3, -0.680, -0.733, -0.680, -0.687, 0.765, 0.745),
               (0.6, 1.1, 0.3, -0.554, -0.602, -0.618, -0.622, 0.853, 0.830)],
    "grid": (-0.774, -0.508, -0.821, -0.579, -0.056, -0.008, -1.035, -0.800),
  },
}


def _firm_cap_step_old(cap, last_output, firm_a):
  return lc.firm_cap_step(cap, last_output, firm_a, True, False)


def _closed_loop(new, pid_of_v, v0, a0, g, tau, delay, lead):
  """Pure loop at DT_CTRL. pid_of_v = the logged command as a function of speed.

  aEgo seen by the controller = plant accel through a 0.15 s first-order filter.
  """
  v, a = v0, a0
  am = a0
  last = pid_of_v(v0)
  cap = None
  buf = [last] * (int(round(delay / DT_CTRL)) + 1)
  x = 0.0
  t = 0.0
  out = {}
  while v > FIRM_V_LO and t < 30.0:
    pid = pid_of_v(v)
    gate = FIRM_V_LO <= v <= FIRM_V_HI
    if gate:
      if new:
        cap = firm_track_step(cap, last, FIRM_A, am, ease_ok=not lead)
        u = cap if not lead else min(pid, cap)
      else:
        cap = _firm_cap_step_old(cap, last, FIRM_A)
        u = min(pid, cap)
    else:
      u = pid
    last = u
    buf.append(u)
    ud = buf.pop(0)
    a += (g * ud - a) * DT_CTRL / tau
    am += (a - am) * DT_CTRL / 0.15
    vn = v + a * DT_CTRL
    for vc in (2.0, 0.8, 0.4):
      if v >= vc > vn:
        out[f"a{vc}"] = a
        out[f"x{vc}"] = x
    out["umin"] = min(out.get("umin", 0.0), u) if gate else out.get("umin", 0.0)
    out["umax"] = max(out.get("umax", -9.0), u) if gate else out.get("umax", -9.0)
    out["all_min"] = min(out.get("all_min", 0.0), u)
    out["all_max"] = max(out.get("all_max", -9.0), u)
    x += v * DT_CTRL
    v = vn
    t += DT_CTRL
  out["x_end"] = x
  return out


@pytest.mark.parametrize("name", [s[0] for s in SCENARIOS])
def test_t4_closed_loop_matches_the_spec(name):
  pid, v0, a0, lead = next((p, v, a, l) for n, p, v, a, l in SCENARIOS if n == name)
  expected = SPEC_EXPECTED[name]

  for g, tau, d, oa4, na4, oa8, na8, ox, nx in expected["fitted"]:
    o = _closed_loop(False, pid, v0, a0, g, tau, d, lead)
    n = _closed_loop(True, pid, v0, a0, g, tau, d, lead)
    assert o["a0.4"] == pytest.approx(oa4, abs=1e-3), (name, g, "old a@0.4", o["a0.4"])
    assert n["a0.4"] == pytest.approx(na4, abs=1e-3), (name, g, "new a@0.4", n["a0.4"])
    assert o["a0.8"] == pytest.approx(oa8, abs=1e-3), (name, g, "old a@0.8", o["a0.8"])
    assert n["a0.8"] == pytest.approx(na8, abs=1e-3), (name, g, "new a@0.8", n["a0.8"])
    assert o["x_end"] == pytest.approx(ox, abs=1e-3), (name, g, "old x_end", o["x_end"])
    assert n["x_end"] == pytest.approx(nx, abs=1e-3), (name, g, "new x_end", n["x_end"])
    for key in ("umin", "umax"):
      assert -3.5 - 1e-9 <= n[key] <= 2.0 + 1e-9, (name, g, key, n[key])

  pairs = [(_closed_loop(False, pid, v0, a0, *p, lead), _closed_loop(True, pid, v0, a0, *p, lead))
           for p in GRID]
  extra = [n["x_end"] - o["x_end"] for o, n in pairs]
  new_a4 = [n["a0.4"] for _, n in pairs]
  old_a4 = [o["a0.4"] for o, _ in pairs]
  for o, n in pairs:
    for key in ("all_min", "all_max", "umin", "umax"):
      assert -3.5 - 1e-9 <= n[key] <= 2.0 + 1e-9, (name, key, n[key])
  for a_ego in (-5.0, -2.0, -1.1, -0.8, -0.55, -0.4, 0.0, 5.0):
    assert -FIRM_TRACK_MAX - 1e-12 <= firm_track_target(FIRM_A, a_ego) <= -FIRM_TRACK_MIN + 1e-12
  gm = expected["grid"]
  assert min(old_a4) == pytest.approx(gm[0], abs=1e-3), (name, "old a@0.4 min", min(old_a4))
  assert max(old_a4) == pytest.approx(gm[1], abs=1e-3), (name, "old a@0.4 max", max(old_a4))
  assert min(new_a4) == pytest.approx(gm[2], abs=1e-3), (name, "new a@0.4 min", min(new_a4))
  assert max(new_a4) == pytest.approx(gm[3], abs=1e-3), (name, "new a@0.4 max", max(new_a4))
  assert min(extra) == pytest.approx(gm[4], abs=1e-3), (name, "extra min", min(extra))
  assert max(extra) == pytest.approx(gm[5], abs=1e-3), (name, "extra max", max(extra))
  for o, n in pairs:
    for key in ("umin", "umax"):
      assert -3.5 - 1e-9 <= n[key] <= 2.0 + 1e-9, (name, key, n[key])
    assert -1.2 - 1e-9 <= firm_track_target(FIRM_A, -1e9) <= -FIRM_TRACK_MIN + 1e-9

  if lead:
    for o, n in pairs:
      assert n["x_end"] <= o["x_end"] + 1e-9, (name, "FAIL_lead", n["x_end"] - o["x_end"])
  else:
    for o, n in pairs:
      assert n["x_end"] - o["x_end"] <= 0.25, (name, "no-lead extra", n["x_end"] - o["x_end"])


def test_t5_band_hit_counts():
  """Plants whose arrival decel lands in the W-PJU0 best bin [-0.85, -0.65]."""
  expected = {"4b5 no lead": (16, 26), "4b1 no lead": (24, 35)}
  for name, (want_old, want_new) in expected.items():
    pid, v0, a0, lead = next((p, v, a, l) for n, p, v, a, l in SCENARIOS if n == name)
    old = sum(1 for p in GRID if -0.85 <= _closed_loop(False, pid, v0, a0, *p, lead)["a0.4"] <= -0.65)
    new = sum(1 for p in GRID if -0.85 <= _closed_loop(True, pid, v0, a0, *p, lead)["a0.4"] <= -0.65)
    assert old == want_old, (name, "old", old)
    assert new == want_new, (name, "new", new)


# ---------------------------------------------------------------------------
# F12: the tracker through the real update()
# ---------------------------------------------------------------------------

def _tick_inputs_track(v, lead, a_ego, a_target, fcw):
  cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.5 * PLAN_T),
                                    lead_status=lead, d_rel=LEAD_D)
  cs.aEgo = a_ego
  plan.aTarget = a_target
  # The plan decelerates at 0.5 m/s^2, so its desired speed right now is v - 0.5*dt.
  # Leaving vTargetNow at 0 makes the pid ask ~-2.4 m/s^2 in the band: the cap would
  # then have to climb ~90 ticks from that last output, not the 60 this test allows.
  plan.vTargetNow = max(0.0, v - 0.5 * DT_CTRL)
  plan.fcw = fcw
  plan.shouldStop = bool(v < 0.3)
  return cs, plan, radar


def _run_track_stream(monkeypatch, lead, a_ego, a_target, fcw, radar_override=None):
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = make_control(monkeypatch, values=ON)
  control.long_control_state = LongCtrlState.pid

  rows = []
  for tick, v in enumerate(_build_stream()):
    cs, plan, radar = _tick_inputs_track(v, lead, a_ego, a_target, fcw)
    # The override is a deliberately bare object; long_control_state_trans reads
    # radarState.leadOne without getattr once shouldStop is true (v < 0.3), so it
    # is only valid while the gate can run. Below the band bottom use the real one.
    if radar_override is not None and v >= FIRM_V_LO:
      radar = radar_override
    integral_before = float(control.pid.i)
    output = float(control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0])
    rows.append({"tick": tick, "v": float(cs.vEgo), "output": output,
                 "integral_before": integral_before, "integral_after": float(control.pid.i),
                 "cap": None if control._firm_cap is None else float(control._firm_cap),
                 "state": int(control.long_control_state)})
  return control, rows, events


def _settled(rows, min_after=60):
  """Pid-band ticks at least `min_after` ticks after the first gated tick."""
  first = next((r["tick"] for r in rows if r["cap"] is not None), None)
  assert first is not None, "the firm finish never engaged"
  return [r for r in rows
          if r["state"] == int(LongCtrlState.pid) and FIRM_V_LO <= r["v"] <= FIRM_V_HI
          and r["tick"] >= first + min_after]


def test_t6a_no_lead_no_fcw_eases_to_firm_track_min(monkeypatch):
  control, rows, events = _run_track_stream(monkeypatch, False, -1.1, -0.4, False)
  firm = events.names("butterStopFirm")
  assert len(firm) == 1, events.events
  assert firm[0]["ease"] is True, firm
  ticks = _settled(rows)
  assert len(ticks) >= 60, len(ticks)
  for row in ticks:
    assert row["output"] == pytest.approx(-FIRM_TRACK_MIN, abs=1e-9), (row["tick"], row["v"], row["output"])
    assert row["integral_after"] == row["integral_before"], (row["tick"], row["v"])


def test_t6b_fcw_never_eases(monkeypatch):
  control, rows, events = _run_track_stream(monkeypatch, False, -1.1, -0.4, True)
  firm = events.names("butterStopFirm")
  assert len(firm) == 1, events.events
  assert firm[0]["ease"] is False, firm
  ticks = _settled(rows)
  assert len(ticks) >= 60, len(ticks)
  for row in ticks:
    assert row["output"] <= -0.8 + 1e-9, (row["tick"], row["v"], row["output"])


def test_t6c_a_hard_plan_never_eases(monkeypatch):
  _, rows, events = _run_track_stream(monkeypatch, False, -1.1, -1.3, False)
  assert events.names("butterStopFirm")[0]["ease"] is False, events.events
  ticks = _settled(rows)
  assert len(ticks) >= 60, len(ticks)
  for row in ticks:
    assert row["output"] <= -0.8 + 1e-9, (row["tick"], row["v"], row["output"])


def test_t6d_a_missing_lead_object_counts_as_a_lead(monkeypatch):
  _, rows, events = _run_track_stream(monkeypatch, False, -1.1, -0.4, False,
                                      radar_override=SimpleNamespace())
  assert events.names("butterStopFirm")[0]["ease"] is False, events.events
  ticks = _settled(rows)
  assert len(ticks) >= 60, len(ticks)
  for row in ticks:
    assert row["output"] <= -0.8 + 1e-9, (row["tick"], row["v"], row["output"])


def test_t6e_with_a_lead_the_tracker_adds_braking(monkeypatch):
  _, rows, events = _run_track_stream(monkeypatch, True, -0.5, -0.4, False)
  assert events.names("butterStopFirm")[0]["ease"] is False, events.events
  ticks = _settled(rows)
  assert len(ticks) >= 60, len(ticks)
  for row in ticks:
    assert row["output"] <= -1.1 + 1e-9, (row["tick"], row["v"], row["output"])
