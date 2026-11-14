"""The ButterStop final stop on classic-CAN Hyundai/Kia: the early hand-off gate, its
off-identity against the pre-change state machine, and the LongControl wiring
behind the log event.

The shipped module constant EARLY_REQ_ENABLED is False (W-SS1: a StopReq sent
before the ESC flag did not stop the release), so on the car this half of the
switch is silent and the ButterStop final stop drives the firm finish instead. The tests
below are the early hand-off's own contract: they turn the constant back on and
pin its behaviour, so re-enabling it is a one-line change with coverage.
With the final stop off the controller must still be identical to the previous one: T1
compares long_control_state_trans against the pre-change machine over a full
grid, and T4 compares every tick of a replayed stop against it as well. T5/T6
pin that the other half of the switch (the final window) is the complement:
shut on classic CAN, open on CAN-FD.
"""
import itertools
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Same Windows import shims as test_butter_stop.py and test_final_stop.py:
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
from opendbc.car.hyundai.values import HyundaiFlags
from openpilot.selfdrive.controls.lib.butter_stop import (
  EARLY_A_WEAK, EARLY_V, MIN_FINAL_GAP_ABS, PROBE_V_MAX, early_req_gap_ok,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, long_control_state_trans
from openpilot.selfdrive.controls.tests.test_butter_stop import (
  PLAN_T, DictParams, make_control, make_cp, stopping_inputs,
)

STATES = (LongCtrlState.stopping, LongCtrlState.starting, LongCtrlState.pid)


def base_long_control_state_trans(CP, active, long_control_state, v_ego,
                                  should_stop, brake_pressed, cruise_standstill, a_ego, stopping_accel, radarState):
  """The pre-change machine (130328d85) specialized to final_hold=False, the off-identity oracle.

  The only textual difference from that commit is the hand-off condition: its
  `and not final_hold` guards are dropped because this oracle is only ever
  compared against calls that pass final_hold=False, where they evaluate true.
  """
  stopping_condition = should_stop
  stopping_accel = stopping_accel if stopping_accel < 0.0 else -0.5
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        leadOne = radarState.leadOne
        fcw_stop = leadOne.status and leadOne.dRel < 4.0
        if (fcw_stop) or (a_ego > stopping_accel):
          long_control_state = LongCtrlState.stopping
        elif long_control_state == LongCtrlState.starting:
          long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state


def _radar(lead_status=False, d_rel=30.0):
  lead = SimpleNamespace(status=lead_status, dRel=d_rel, aLeadK=0.0, vLead=0.0)
  return SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False),
                         leadCutInRisk=SimpleNamespace(status=False))


class _EventLog:
  def __init__(self):
    self.events = []

  def event(self, name, **kwargs):
    self.events.append((name, kwargs))

  def names(self, name):
    return [kw for ev, kw in self.events if ev == name]


# ---------------------------------------------------------------------------
# T3: the gap gate
# ---------------------------------------------------------------------------

def test_gap_gate_needs_a_lead():
  assert early_req_gap_ok(0.8, False, 8.0) is False


def test_gap_gate_rejects_a_non_finite_gap():
  assert early_req_gap_ok(0.8, True, float("inf")) is False
  assert early_req_gap_ok(0.8, True, float("nan")) is False


def test_gap_gate_spec_numbers():
  # need = 0.8^2 / (2 * 0.25) = 1.28 m
  assert early_req_gap_ok(0.8, True, 8.0) is True
  assert early_req_gap_ok(0.8, True, 4.0) is False   # 4.0 - 1.28 = 2.72 < 3.0
  assert early_req_gap_ok(0.8, True, 6.72) is True   # 6.72 - 1.28 = 5.44 >= 3.0


def test_gap_gate_boundary_is_the_literal_constant():
  need = EARLY_V ** 2 / (2.0 * EARLY_A_WEAK)
  assert need == pytest.approx(1.28, abs=1e-9)
  edge = need + MIN_FINAL_GAP_ABS
  assert early_req_gap_ok(EARLY_V, True, edge - 1e-6) is False
  assert early_req_gap_ok(EARLY_V, True, edge + 1e-6) is True


def test_gap_gate_holds_at_rest():
  assert early_req_gap_ok(0.0, True, MIN_FINAL_GAP_ABS) is True
  assert early_req_gap_ok(-0.2, True, MIN_FINAL_GAP_ABS) is True
  assert early_req_gap_ok(0.0, True, MIN_FINAL_GAP_ABS - 0.01) is False


# ---------------------------------------------------------------------------
# T1: off-identity of the state machine
# ---------------------------------------------------------------------------

def test_off_grid_is_identical_to_the_pre_change_machine():
  vs = (0.3, 0.6, 0.8, 1.2)
  a_egos = (-1.0, -0.6, -0.3, 0.0)
  bools = (True, False)
  leads = (None, 3.5, 8.0)
  cp = make_cp()
  for v, a_ego, should_stop, state, d_rel, active in itertools.product(
      vs, a_egos, bools, STATES, leads, bools):
    radar = _radar(d_rel is not None, d_rel if d_rel is not None else 30.0)
    args = (cp, active, state, v, should_stop, False, False, a_ego, -0.5, radar)
    base = base_long_control_state_trans(*args)
    assert long_control_state_trans(*args, final_hold=False, early_handoff=False) == base, args[1:]
    assert long_control_state_trans(*args, early_handoff=False) == base, args[1:]
    assert long_control_state_trans(*args) == base, args[1:]


def test_off_is_the_default_argument():
  cp = make_cp()
  for v, a_ego, should_stop, state, d_rel in itertools.product(
      (0.3, 0.8, 1.2), (-0.6, 0.0), (True, False), STATES, (None, 8.0)):
    radar = _radar(d_rel is not None, d_rel if d_rel is not None else 30.0)
    args = (cp, True, state, v, should_stop, False, False, a_ego, -0.5, radar)
    assert long_control_state_trans(*args, early_handoff=False) == long_control_state_trans(*args)


# ---------------------------------------------------------------------------
# T2: the early hand-off itself
# ---------------------------------------------------------------------------

def test_early_handoff_holds_the_stop_that_the_stock_path_leaves_in_pid():
  """v 0.8, a_ego -0.6, stopping_accel -0.5: stock keeps pid, the probe hands off."""
  cp = make_cp()
  radar = _radar(True, 8.0)
  args = (cp, True, LongCtrlState.pid, 0.8, True, False, False, -0.6, -0.5, radar)
  assert long_control_state_trans(*args) == LongCtrlState.pid
  assert long_control_state_trans(*args, early_handoff=True) == LongCtrlState.stopping


def test_early_handoff_is_ignored_outside_the_pid_handoff_branch():
  """The flag only feeds the pid/starting hand-off; every other transition is unchanged."""
  cp = make_cp()
  radar = _radar(True, 8.0)
  for state, active, should_stop, v in itertools.product(
      (LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.pid, LongCtrlState.starting),
      (True, False), (True, False), (0.0, 0.8, 2.0)):
    args = (cp, active, state, v, should_stop, False, False, -0.6, -0.5, radar)
    if state == LongCtrlState.pid and active and should_stop:
      continue
    assert long_control_state_trans(*args, early_handoff=True) == base_long_control_state_trans(*args), args[1:]


def test_early_handoff_flag_is_the_state_machine_contract():
  """The flag is unconditional in the state machine; update() is what gates it.

  final_hold precedence is enforced where the flag is computed (update()), not
  here, so this test pins the state-machine half of that contract and the
  LongControl test below pins the precedence itself.
  """
  cp = make_cp()
  radar = _radar(True, 8.0)
  args = (cp, True, LongCtrlState.pid, 0.8, True, False, False, -0.6, -0.5, radar)
  assert long_control_state_trans(*args, early_handoff=True) == LongCtrlState.stopping
  assert long_control_state_trans(*args, final_hold=True, early_handoff=False) == LongCtrlState.pid


# ---------------------------------------------------------------------------
# T4: LongControl wiring
# ---------------------------------------------------------------------------

def _stop_stream(v_start=1.2, decel=0.6, ticks=260, relax_at=None):
  """Speed falls to rest at `decel`; aEgo relaxes above stopping_accel after relax_at."""
  dt = 0.01
  v = v_start
  out = []
  for tick in range(ticks):
    a_ego = -decel
    if relax_at is not None and v <= relax_at:
      a_ego = -0.2
    out.append((max(v, 0.0), a_ego))
    v = max(0.0, v - decel * dt)
  return out


def _run_stop(monkeypatch, values, stream, lead_d_rel=9.0):
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = make_control(monkeypatch, values=values)
  control.long_control_state = LongCtrlState.pid
  states, first_early_tick = [], None
  ticks = []
  for tick, (v, a_ego) in enumerate(stream):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=lead_d_rel)
    cs.aEgo = a_ego
    previous = control.long_control_state
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    states.append(control.long_control_state)
    ticks.append((tick, previous, v, a_ego, radar))
    if first_early_tick is None and control.long_control_state == LongCtrlState.stopping:
      first_early_tick = tick
  return control, states, first_early_tick, events, ticks


def test_update_hands_off_at_early_v_and_logs_once(monkeypatch):
  stream = _stop_stream()
  monkeypatch.setattr(lc, "EARLY_REQ_ENABLED", True)
  control, states, fire_tick, events, _ = _run_stop(
    monkeypatch, {"ButterStop": 1}, stream)

  expected_tick = next(i for i, (v, _) in enumerate(stream) if v <= EARLY_V)
  assert fire_tick == expected_tick, (fire_tick, expected_tick)
  assert states[fire_tick] == LongCtrlState.stopping
  assert all(s == LongCtrlState.stopping for s in states[fire_tick:])

  logged = events.names("butterStopEarlyReq")
  assert len(logged) == 1, logged
  vEgo, dRel, aEgo = logged[0]["vEgo"], logged[0]["dRel"], logged[0]["aEgo"]
  assert vEgo == pytest.approx(stream[fire_tick][0])
  assert dRel == pytest.approx(9.0)
  assert aEgo == pytest.approx(stream[fire_tick][1])


def test_update_with_the_toggle_off_follows_the_stock_path_every_tick(monkeypatch):
  """Constant -0.6 m/s^2 never satisfies a_ego > stopping_accel: stock stays pid."""
  monkeypatch.setattr(lc, "FINAL_STOP_ENABLED", False)
  stream = _stop_stream(decel=0.6, ticks=180)
  control, states, fire_tick, events, ticks = _run_stop(
    monkeypatch, {"ButterStop": 1}, stream)

  assert control.final_stop is False, "the off arm must resolve the final stop off"
  assert control.early_req is False
  assert fire_tick is None, fire_tick
  assert set(states) == {LongCtrlState.pid}
  assert events.names("butterStopEarlyReq") == []
  for tick, previous, v, a_ego, radar in ticks:
    expected = base_long_control_state_trans(control.CP, True, previous, v, True, False, False,
                                             a_ego, control.stopping_accel, radar)
    assert states[tick] == expected, (tick, v, a_ego)


def test_update_with_the_toggle_off_fires_on_the_stock_accel_rule(monkeypatch):
  """Once aEgo relaxes above stopping_accel the stock hand-off runs, later than 0.8 m/s."""
  monkeypatch.setattr(lc, "FINAL_STOP_ENABLED", False)
  stream = _stop_stream(decel=0.6, ticks=260, relax_at=0.3)
  control, states, fire_tick, events, ticks = _run_stop(
    monkeypatch, {"ButterStop": 1}, stream)

  assert control.final_stop is False, "the off arm must resolve the final stop off"
  expected_tick = next(i for i, (v, a) in enumerate(stream) if a > control.stopping_accel)
  assert fire_tick == expected_tick, (fire_tick, expected_tick)
  assert stream[fire_tick][0] < EARLY_V
  assert events.names("butterStopEarlyReq") == []
  for tick, previous, v, a_ego, radar in ticks:
    expected = base_long_control_state_trans(control.CP, True, previous, v, True, False, False,
                                             a_ego, control.stopping_accel, radar)
    assert states[tick] == expected, (tick, v, a_ego)


def test_update_does_not_fire_without_a_lead(monkeypatch):
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  monkeypatch.setattr(lc, "EARLY_REQ_ENABLED", True)
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  control.long_control_state = LongCtrlState.pid
  states = []
  for v, a_ego in _stop_stream(ticks=180):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=False, d_rel=9.0)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    states.append(control.long_control_state)
  assert set(states) == {LongCtrlState.pid}
  assert events.names("butterStopEarlyReq") == []


def test_update_waits_in_the_weak_braking_gap_until_the_need_shrinks(monkeypatch):
  """4.2 m to the lead blocks the 0.8 m/s hand-off; the gate opens once need shrinks.

  At 0.80 m/s weak braking needs 1.28 m and leaves 2.92 m < 3.0, so the probe
  must stay silent; the need falls with the square of speed, so the hand-off
  arrives at the first speed where need + MIN_FINAL_GAP_ABS fits in 4.2 m. The
  4.2 m distance is deliberately above the 4 m fcw radius so the stock hand-off
  cannot mask the blocked probe.
  """
  stream = _stop_stream(ticks=200)
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  monkeypatch.setattr(lc, "EARLY_REQ_ENABLED", True)
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  control.long_control_state = LongCtrlState.pid

  states = []
  for v, a_ego in stream:
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=4.2)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    states.append(control.long_control_state)

  v_seen_at_early = next(i for i, (v, _) in enumerate(stream) if v <= EARLY_V)
  assert states[v_seen_at_early] == LongCtrlState.pid, "the probe must not fire inside the gap"

  fire_tick = next(i for i, s in enumerate(states) if s == LongCtrlState.stopping)
  # The contract is the gate itself, not an independently rounded speed bound.
  assert fire_tick > v_seen_at_early, "the hand-off must come later than the 0.8 m/s crossing"
  assert not early_req_gap_ok(stream[fire_tick - 1][0], True, 4.2), "gate was already open one tick earlier"
  assert early_req_gap_ok(stream[fire_tick][0], True, 4.2), "hand-off fired with the gate closed"
  assert 0.0 < stream[fire_tick][0] <= EARLY_V
  assert len(events.names("butterStopEarlyReq")) == 1


def test_update_requires_butter_stop_on(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 0})
  assert control.early_req is False


def test_early_req_defaults_off(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  assert control.final_stop, "the final stop itself is on with ButterStop"
  assert control.early_req is False, "the early hand-off stays off by constant"


def test_update_relatches_the_log_above_walking_speed(monkeypatch):
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  monkeypatch.setattr(lc, "EARLY_REQ_ENABLED", True)
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})

  def frame(v, a_ego):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=9.0)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)

  control.long_control_state = LongCtrlState.pid
  frame(0.5, -0.6)
  assert len(events.names("butterStopEarlyReq")) == 1
  # Back to a state where the probe can fire, and moving again: the latch clears.
  control.long_control_state = LongCtrlState.pid
  frame(1.4, -0.6)
  assert control._early_logged is False
  control.long_control_state = LongCtrlState.pid
  frame(0.5, -0.6)
  assert len(events.names("butterStopEarlyReq")) == 2


# ---------------------------------------------------------------------------
# T6: the switch is per car; CAN-FD gets the window, never the early hand-off
# ---------------------------------------------------------------------------

def _control_with_cp(monkeypatch, values, cp):
  params = DictParams({"StoppingAccel": -50, "StopDistanceCarrot": 550, "ButterStop": 1,
                       "LongTuningKpV": 100, "LongTuningKiV": 200, "LongTuningKf": 100, **values})
  monkeypatch.setattr(lc, "Params", lambda: params)
  return lc.LongControl(cp), params


def test_canfd_never_hands_off_early_and_opens_the_window_instead(monkeypatch):
  """T6: a CAN-FD car (K5) drives the window; the early hand-off never fires."""
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  cp = make_cp()
  cp.flags = int(HyundaiFlags.CANFD)
  control, _ = _control_with_cp(monkeypatch, {"ButterStop": 1}, cp)

  assert control.final_stop is True, "the switch itself is on"
  assert control.final_window is True, "CAN-FD drives the final window"
  assert control.early_req is False, "CAN-FD must never hand off early"
  assert control._early_req_capable is False

  control.long_control_state = LongCtrlState.pid
  states = []
  open_ticks = []
  # aEgo relaxes only below the probe's entry speed, so the stock hand-off
  # cannot take pid before the window opens.
  for v, a_ego in _stop_stream(ticks=200, relax_at=PROBE_V_MAX - 0.05):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=9.0)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    states.append(control.long_control_state)
    if control._final_t is not None:
      open_ticks.append((v, control._final_t))

  assert events.names("butterStopEarlyReq") == []
  assert open_ticks, "the window must open on a CAN-FD car"
  entry_v = control._final_v0
  assert entry_v <= PROBE_V_MAX + 1e-9, entry_v
  assert len(events.names("butterStopFinal")) == 1


def test_non_hyundai_brands_never_hand_off_early(monkeypatch):
  """The early hand-off is classic-CAN Hyundai/Kia only; other brands run the window."""
  for brand in ("toyota", "gm", "tesla", "ford"):
    events = _EventLog()
    monkeypatch.setattr(lc, "cloudlog", events)
    control, _ = make_control(monkeypatch, values={"ButterStop": 1}, brand=brand)
    assert control.final_stop is True, brand
    assert control.early_req is False, brand
    assert control.final_window is False, brand
    assert control._early_req_capable is False, brand

    control.long_control_state = LongCtrlState.pid
    states = []
    for v, a_ego in _stop_stream(ticks=120):
      cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                        lead_status=True, d_rel=9.0)
      cs.aEgo = a_ego
      control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
      states.append(control.long_control_state)
    assert events.names("butterStopEarlyReq") == [], brand
    assert set(states) == {LongCtrlState.pid}, brand


def test_classic_can_hyundai_is_capable(monkeypatch):
  """Control for the two tests above: flags without CANFD leave the probe live."""
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  monkeypatch.setattr(lc, "EARLY_REQ_ENABLED", True)
  cp = make_cp()
  cp.flags = 0
  control, _ = _control_with_cp(monkeypatch, {"ButterStop": 1}, cp)
  assert control._early_req_capable is True

  control.long_control_state = LongCtrlState.pid
  for v, a_ego in _stop_stream(ticks=200):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=9.0)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert len(events.names("butterStopEarlyReq")) == 1


def test_other_hyundai_flags_do_not_block_the_probe(monkeypatch):
  """Only the CANFD bit matters; a different Hyundai flag leaves the probe live."""
  cp = make_cp()
  cp.flags = int(HyundaiFlags.MANDO_RADAR)
  control, _ = _control_with_cp(monkeypatch, {"ButterStop": 1}, cp)
  assert control._early_req_capable is True

  cp.flags = int(HyundaiFlags.MANDO_RADAR | HyundaiFlags.CANFD)
  control, _ = _control_with_cp(monkeypatch, {"ButterStop": 1}, cp)
  assert control._early_req_capable is False


# ---------------------------------------------------------------------------
# E-new: with the shipped constant the hand-off never fires, and the firm
# finish takes the switch's place
# ---------------------------------------------------------------------------

def test_shipped_constant_never_hands_off(monkeypatch):
  """The final stop on a classic-CAN car, constant at its shipped False."""
  assert lc.EARLY_REQ_ENABLED is False, "the module constant is the shipped value"
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  assert control.final_stop is True, "the switch is on"
  assert control.early_req is False, "the hand-off is held off by the constant"
  assert control.firm_a == pytest.approx(0.8), "the firm finish takes its place"

  control.long_control_state = LongCtrlState.pid
  states = []
  for v, a_ego in _stop_stream(ticks=200, relax_at=0.55):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=9.0)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    states.append(control.long_control_state)

  assert events.names("butterStopEarlyReq") == [], "no early hand-off event"
  # The hand-off is what used to force stopping; the stock accel rule still can.
  expected = next((i for i, (v, a) in enumerate(_stop_stream(ticks=200, relax_at=0.55))
                   if a > control.stopping_accel), None)
  first_stopping = next((i for i, s in enumerate(states) if s == LongCtrlState.stopping), None)
  assert first_stopping == expected, (first_stopping, expected)


# ---------------------------------------------------------------------------
# T5: classic-CAN Hyundai/Kia never open the final window
# ---------------------------------------------------------------------------

def _run_final_stop(monkeypatch, values, lead_d_rel=9.0):
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = make_control(monkeypatch, values=values)
  control.long_control_state = LongCtrlState.pid
  open_ticks = []
  for tick, (v, a_ego) in enumerate(_stop_stream(ticks=260, relax_at=0.55)):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=lead_d_rel)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    if control._final_t is not None:
      open_ticks.append(tick)
  return control, events, control.long_control_state, open_ticks


def test_t5_classic_can_hyundai_never_opens_the_window(monkeypatch):
  """T5: on classic-CAN Hyundai/Kia the switch selects the early hand-off only.

  The window is the complement, so with the same parameters it must stay shut:
  no butterStopFinal event, _final_t never set. _run_final_stop builds a
  classic-CAN CP, which is exactly the configuration under test here.
  """
  monkeypatch.setattr(lc, "EARLY_REQ_ENABLED", True)
  control, events, state, open_ticks = _run_final_stop(
    monkeypatch, {"ButterStop": 1})
  assert control.final_stop and control.early_req
  assert not control.final_window, "classic CAN must not open the window"
  assert state == LongCtrlState.stopping
  assert open_ticks == [], "the final window must not open on classic-CAN Hyundai"
  assert control._final_t is None
  assert events.names("butterStopFinal") == []
  assert len(events.names("butterStopEarlyReq")) == 1


def test_t5_canfd_opens_the_window_and_never_hands_off_early(monkeypatch):
  """T5 control: the other half of the switch, on a CAN-FD car."""
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  cp = make_cp()
  cp.flags = int(HyundaiFlags.CANFD)
  control, _ = _control_with_cp(monkeypatch, {"ButterStop": 1}, cp)
  assert control.final_stop and control.final_window
  assert not control.early_req

  control.long_control_state = LongCtrlState.pid
  open_ticks = []
  # Same reason as T6: aEgo relaxes only below the probe's entry speed.
  for tick, (v, a_ego) in enumerate(_stop_stream(ticks=260, relax_at=PROBE_V_MAX - 0.05)):
    cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=np.maximum(0.0, v - 0.6 * PLAN_T),
                                      lead_status=True, d_rel=9.0)
    cs.aEgo = a_ego
    control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
    if control._final_t is not None:
      open_ticks.append(tick)
  assert open_ticks, "the final window must open on a CAN-FD car"
  assert control._final_done
  assert len(events.names("butterStopFinal")) == 1
  assert events.names("butterStopEarlyReq") == []
