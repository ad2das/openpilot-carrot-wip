"""ButterStopFinal: the constant-jerk final-stop profile, the final_hold gate in
the state machine, and the LongControl window that drives the last part of a
stop directly instead of handing it to the car.

ButterStopFinal=0 must be byte-identical to the pre-change controller: the G0
grid compares long_control_state_trans against a verbatim copy of the base
function, and the LongControl replay compares the full output against the
pre-change update() body.
"""
import itertools
import os
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# Same Windows import shims as test_butter_stop.py: these tests never construct
# the real Params backend or the solver.
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

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib import longcontrol as lc
from openpilot.selfdrive.controls.lib.butter_stop import (
  FINAL_A0_DIST, FINAL_A0_MAX, FINAL_A0_MIN, FINAL_A_END, FINAL_FB_V, FINAL_HOLD_T,
  FINAL_KV, FINAL_U_MAX, FINAL_U_MIN, FINAL_V_MAX, MIN_FINAL_GAP_ABS, STANDSTILL_V,
  STOCK_ROLL_DECEL, final_a0_dist, final_profile,
)
from openpilot.selfdrive.controls.lib.longcontrol import (
  CONTROL_N_T_IDX, LongCtrlState, long_control_state_trans,
)
from openpilot.selfdrive.controls.tests.test_butter_stop import (
  make_control, make_cp, stopping_inputs,
)

STATES = (LongCtrlState.off, LongCtrlState.stopping, LongCtrlState.starting, LongCtrlState.pid)
PLAN_T = np.asarray(CONTROL_N_T_IDX, dtype=float)


def base_long_control_state_trans(CP, active, long_control_state, v_ego,
                                  should_stop, brake_pressed, cruise_standstill, a_ego, stopping_accel, radarState):
  """Verbatim pre-change long_control_state_trans (9bbecd92b), the G0 oracle."""
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
        if a_ego > stopping_accel or fcw_stop:
          long_control_state = LongCtrlState.stopping
        elif long_control_state == LongCtrlState.starting:
          long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state


def _lead(status=False, d_rel=30.0):
  return SimpleNamespace(status=status, dRel=d_rel)


def _radar(status=False, d_rel=30.0):
  return SimpleNamespace(leadOne=_lead(status, d_rel))


# ---------------------------------------------------------------------------
# final_profile
# ---------------------------------------------------------------------------

def test_final_profile_matches_closed_form():
  a_ref, v_ref, T = final_profile(0.0, 0.5, 0.5)
  assert T == pytest.approx(1.6666666666666667, abs=1e-4)
  assert a_ref == pytest.approx(-0.5)
  assert v_ref == pytest.approx(0.5)

  a_end, v_end, T_end = final_profile(T, 0.5, 0.5)
  assert T_end == pytest.approx(T, abs=1e-12)
  assert a_end == pytest.approx(-FINAL_A_END)
  assert v_end == pytest.approx(0.0)

  # Beyond T the profile holds the end decel and zero speed.
  a_late, v_late, _ = final_profile(T + 5.0, 0.5, 0.5)
  assert a_late == pytest.approx(-FINAL_A_END)
  assert v_late == pytest.approx(0.0)


def test_final_profile_distance_matches_numeric_integration():
  dt = 1e-3
  for v0, a0 in ((0.5, 0.5), (0.4, 0.25), (0.6, 0.8), (0.3, 0.35)):
    _, _, T = final_profile(0.0, v0, a0)
    t = np.arange(0.0, T, dt)
    v = np.array([final_profile(float(x), v0, a0)[1] for x in t])
    distance = float(np.sum(0.5 * (v[1:] + v[:-1]) * dt) + 0.5 * v[-1] * (T - t[-1]))
    # Closed form: v0*T - a0*T^2/2 + j*T^3/6 with j = (a0 - FINAL_A_END)/T.
    j = (a0 - FINAL_A_END) / T
    closed = v0 * T - 0.5 * a0 * T * T + j * T ** 3 / 6.0
    assert distance == pytest.approx(closed, abs=2e-3), (v0, a0, distance, closed)


def test_final_profile_spec_distances():
  _, _, T = final_profile(0.0, 0.5, 0.5)
  assert T == pytest.approx(1.6667, abs=1e-4)
  j = (0.5 - FINAL_A_END) / T
  distance = 0.5 * T - 0.5 * 0.5 * T * T + j * T ** 3 / 6.0
  assert distance == pytest.approx(0.324, abs=2e-3)
  _, _, T2 = final_profile(0.0, 0.4, 0.25)
  assert T2 == pytest.approx(2.2857, abs=1e-4)


def test_final_profile_decel_is_monotone_and_bounded():
  v0, a0 = 0.5, 0.55
  _, _, T = final_profile(0.0, v0, a0)
  previous = -a0
  for x in np.linspace(0.0, T, 200):
    a_ref, _, _ = final_profile(float(x), v0, a0)
    assert a_ref >= previous - 1e-12, x
    assert -a0 - 1e-9 <= a_ref <= -FINAL_A_END + 1e-9, (x, a_ref)
    previous = a_ref


# ---------------------------------------------------------------------------
# state machine
# ---------------------------------------------------------------------------

def test_g0_grid_final_hold_false_equals_base():
  vs = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0)
  a_egos = (-1.0, -0.6, -0.5, -0.4, -0.2, 0.0)
  bools = (True, False)
  for starting_state in bools:
    cp = make_cp("hyundai", starting_state)
    for v, a_ego, should_stop, brake, standstill, state, active, fcw in itertools.product(
        vs, a_egos, bools, bools, bools, STATES, bools, bools):
      radar = _radar(fcw, 3.9 if fcw else 30.0)
      args = (cp, active, state, v, should_stop, brake, standstill, a_ego, -0.5, radar)
      base = base_long_control_state_trans(*args)
      new = long_control_state_trans(*args, final_hold=False)
      assert new == base, (starting_state, v, a_ego, should_stop, brake, standstill,
                           state, active, fcw, int(base), int(new))


def test_g0_default_arguments_keep_base_behaviour():
  cp = make_cp()
  for v, a_ego, state, fcw in itertools.product((0.0, 0.3, 1.0), (-0.6, -0.2), STATES, (True, False)):
    args = (cp, True, state, v, True, False, False, a_ego, -0.5, _radar(fcw, 3.9 if fcw else 30.0))
    assert long_control_state_trans(*args) == base_long_control_state_trans(*args)


def test_final_hold_keeps_pid_past_the_speed_handoff():
  cp = make_cp()
  # a_ego -0.2 > stopping_accel -0.5: the base machine hands off, the window does not.
  args = (cp, True, LongCtrlState.pid, 0.3, True, False, False, -0.2, -0.5, _radar())
  assert long_control_state_trans(*args) == LongCtrlState.stopping
  assert long_control_state_trans(*args, final_hold=True) == LongCtrlState.pid


def test_final_hold_stays_pid_inside_the_lead_gap():
  """Inside the window a lead between the gap and the fcw radius keeps pid."""
  cp = make_cp()
  args = (cp, True, LongCtrlState.pid, 0.3, True, False, False, -0.2, -0.5, _radar(True, 3.9))
  assert long_control_state_trans(*args) == LongCtrlState.stopping, "base hands off at 3.9 m"
  assert long_control_state_trans(*args, final_hold=True) == LongCtrlState.pid


def test_final_hold_never_delays_a_starting_exit():
  cp = make_cp()
  args = (cp, True, LongCtrlState.starting, 1.0, True, False, False, -0.2, -0.5, _radar())
  assert long_control_state_trans(*args, final_hold=True) == LongCtrlState.stopping


# ---------------------------------------------------------------------------
# LongControl integration
# ---------------------------------------------------------------------------

def _stop_frame(control, v, a_ego=-0.35, speeds=None, lead_status=False, d_rel=30.0):
  cs, plan, radar = stopping_inputs(v_ego=v, plan_speeds=speeds)
  cs.aEgo = a_ego
  radar.leadOne = SimpleNamespace(status=lead_status, dRel=d_rel, aLeadK=0.0, vLead=0.0)
  return control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)


def _decaying_speeds(v):
  return np.maximum(0.0, v - 0.6 * PLAN_T)


def test_toggle_off_matches_base_state_machine_and_never_opens_a_window(monkeypatch):
  """ButterStopFinal=0: the window never opens and every tick equals the base transition."""
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 0})
  assert not control.final_stop

  from openpilot.selfdrive.controls.tests.test_butter_stop import _frame_inputs, _scenario_stream
  v_profile, age = _scenario_stream()
  for frame in range(len(v_profile)):
    active, cs, plan, radar = _frame_inputs(frame, v_profile)
    previous = control.long_control_state
    expected = base_long_control_state_trans(control.CP, active, previous, cs.vEgo,
                                             plan.shouldStop, cs.brakePressed,
                                             cs.cruiseState.standstill, cs.aEgo,
                                             control.stopping_accel, radar)
    if active and cs.softHoldActive > 0:
      expected = LongCtrlState.stopping
    control.update(active, cs, plan, (-3.5, 2.0), age, radar)
    assert control._final_t is None, frame
    assert control.long_control_state == expected, frame


def test_toggle_on_changes_the_command_of_the_same_frames(monkeypatch):
  """The window is live: at identical inputs the toggle-on run commands differently."""
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  off, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 0})
  on, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  off.long_control_state = LongCtrlState.pid
  on.long_control_state = LongCtrlState.pid

  v = 0.55
  outs_off, outs_on = [], []
  for _ in range(120):
    speeds = _decaying_speeds(v)
    outs_off.append(_stop_frame(off, v, a_ego=-0.32, speeds=speeds)[0])
    outs_on.append(_stop_frame(on, v, a_ego=-0.32, speeds=speeds)[0])
    v = max(0.0, v - 0.32 * DT_CTRL)
  assert off._final_t is None
  assert on._final_t is not None or on._final_done, "the toggle-on run must enter the window"
  assert outs_off != outs_on, "ButterStopFinal=1 must change the final-stop command"


def test_window_opens_only_below_final_v_max(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  assert control.final_stop

  above = FINAL_V_MAX + 0.2
  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, above, speeds=_decaying_speeds(above))
  assert control._final_t is None, "window must not open above FINAL_V_MAX"
  assert control.long_control_state == LongCtrlState.stopping

  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, FINAL_V_MAX - 0.1, speeds=_decaying_speeds(FINAL_V_MAX - 0.1))
  assert control._final_t is not None
  assert control.long_control_state == LongCtrlState.pid


def test_window_entry_captures_clipped_decel_and_speed(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = -2.0  # far stronger than FINAL_A0_MAX

  _stop_frame(control, 0.5, a_ego=-0.3, speeds=_decaying_speeds(0.5))
  assert control._final_t is not None
  assert control._final_v0 == pytest.approx(0.5)
  assert control._final_a0 == pytest.approx(FINAL_A0_MAX)


def test_window_output_stays_inside_final_clip(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid

  v = 0.5
  outs = []
  for _ in range(200):
    out = _stop_frame(control, v, a_ego=-0.3, speeds=_decaying_speeds(v))[0]
    outs.append(out)
    assert FINAL_U_MIN - 1e-9 <= out <= FINAL_U_MAX + 1e-9, (v, out)
    v = max(0.0, v - 0.35 * DT_CTRL)
  assert control._final_t is not None or control._final_done


def test_window_exits_at_standstill_speed(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid

  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5))
  assert control._final_t is not None
  _stop_frame(control, STANDSTILL_V - 0.01, speeds=np.zeros(len(PLAN_T)))
  assert control._final_t is None
  assert control._final_done


def test_window_exits_on_timeout(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5))
  assert control._final_t is not None

  _, _, T = final_profile(0.0, control._final_v0, control._final_a0)
  # Stay rolling above STANDSTILL_V so only the timeout can end the window.
  ticks = int(round((T + FINAL_HOLD_T) / DT_CTRL)) + 5
  v = 0.5
  for _ in range(ticks):
    _stop_frame(control, v, speeds=_decaying_speeds(v))
  assert control._final_t is None
  assert control._final_done


def test_window_survives_a_lead_inside_the_fcw_radius(monkeypatch):
  """A lead between MIN_FINAL_GAP_ABS and 4.0 m no longer ends the window."""
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5), lead_status=True, d_rel=30.0)
  assert control._final_t is not None

  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5), lead_status=True,
              d_rel=MIN_FINAL_GAP_ABS + 0.5)
  assert control._final_t is not None, "3.5 m is inside the fcw radius but outside the gap"
  assert not control._final_done


def test_window_exits_close_lead_and_hands_off_the_same_tick(monkeypatch):
  """Below MIN_FINAL_GAP_ABS the window ends and the ordinary fcw hand-off runs."""
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5), lead_status=True, d_rel=30.0)
  assert control._final_t is not None

  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5), lead_status=True, d_rel=2.9)
  assert control._final_t is None
  assert control._final_done
  assert control.long_control_state == LongCtrlState.stopping


def test_window_exits_when_should_stop_clears(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5))
  assert control._final_t is not None

  cs, plan, radar = stopping_inputs(v_ego=0.5, plan_speeds=_decaying_speeds(0.5))
  plan.shouldStop = False
  plan.aTarget = 0.5
  plan.vTargetNow = 2.0
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._final_t is None
  assert control._final_done


def test_window_never_reopens_in_the_same_stop(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid

  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5))
  assert control._final_t is not None
  _stop_frame(control, STANDSTILL_V - 0.01, speeds=np.zeros(len(PLAN_T)))
  assert control._final_done

  # Back to the same stop: no second window until the car is rolling again.
  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5))
  assert control._final_t is None, "a stop gets one window only"

  # Rolling above 1 m/s clears the latch for the next stop.
  _stop_frame(control, 1.5, speeds=np.full(len(PLAN_T), 1.0))
  assert not control._final_done
  control.long_control_state = LongCtrlState.pid
  _stop_frame(control, 0.5, speeds=_decaying_speeds(0.5))
  assert control._final_t is not None


def test_window_distance_guard_limits_braking(monkeypatch):
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid

  # A plan that stops within 0.1 m and no lead: the guard demands v^2/(2*0.1).
  speeds = np.array([0.5 - 5.0 * x for x in PLAN_T])
  speeds = np.maximum(0.0, speeds)
  control.long_control_state = LongCtrlState.pid
  cs, plan, radar = stopping_inputs(v_ego=0.5, plan_speeds=speeds)
  cs.aEgo = -0.3
  out = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0]
  assert out <= -1.25 + 1e-6, out
  assert out >= FINAL_U_MIN - 1e-9


def test_window_distance_guard_uses_lead_gap(monkeypatch):
  """The lead term of the guard, exercised directly on the window helper.

  In update() a lead closer than 4.0 m trips fcw_stop and blocks the window
  outright, so the smallest lead budget reachable inside the window is
  4.0 - MIN_FINAL_GAP_ABS = 1.0 m, i.e. 0.125 m/s^2 at FINAL_V_MAX -- weaker
  than the profile itself. The lead term therefore cannot bind at entry; it
  exists so a lead that closes mid-window still caps the command. This test
  drives the helper with that smaller budget to prove the term works.
  """
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  control._final_t = 0.0
  control._final_v0 = 0.5
  control._final_a0 = 0.5

  far_plan = np.full(len(PLAN_T), 3.0)  # no planned stop: only the lead can bind
  cs, plan, _ = stopping_inputs(v_ego=0.5, plan_speeds=far_plan)
  radar = SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=3.1))
  out = control._final_window_accel(cs, plan, radar)
  assert out <= -1.25 + 1e-6, out
  assert out >= FINAL_U_MIN - 1e-9

  # A lead beyond the fcw radius with a large budget leaves the profile alone.
  control._final_t = 0.0
  radar_far = SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=30.0))
  out_far = control._final_window_accel(cs, plan, radar_far)
  assert out_far == pytest.approx(-0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# final_a0_dist: the stock-distance-closed entry decel
# ---------------------------------------------------------------------------

def test_final_a0_dist_matches_the_closed_form():
  assert final_a0_dist() == pytest.approx(0.65497, abs=1e-4)
  assert FINAL_A0_DIST == pytest.approx(0.65497, abs=1e-4)
  # Independent of the end decel only through its own argument.
  assert final_a0_dist(a_end=0.10, a_stock=STOCK_ROLL_DECEL) == pytest.approx(FINAL_A0_DIST)
  # With no end decel the profile is a pure linear ramp: D = 2 v0^2 / (3 a0),
  # so a0 = 2/3 for a stock decel of 0.5.
  assert final_a0_dist(a_end=0.0, a_stock=0.5) == pytest.approx(2.0 / 3.0, abs=1e-6)


@pytest.mark.parametrize("v0", [0.3, 0.5, 0.6])
def test_final_a0_dist_profile_travels_the_stock_distance(v0):
  """With a0 = FINAL_A0_DIST the profile covers exactly v0^2/(2*STOCK_ROLL_DECEL)."""
  _, _, T = final_profile(0.0, v0, FINAL_A0_DIST)
  steps = 20000
  dt = T / steps
  travelled = sum(final_profile(i * dt, v0, FINAL_A0_DIST)[1] for i in range(steps)) * dt
  stock = v0 * v0 / (2.0 * STOCK_ROLL_DECEL)
  assert travelled == pytest.approx(stock, abs=1e-3), (travelled, stock)


def test_window_entry_never_brakes_softer_than_final_a0_dist(monkeypatch):
  """A gentle entry is raised to FINAL_A0_DIST so the stop does not run long."""
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = -FINAL_A0_MIN  # gentler than the stock-distance decel
  cs, plan, radar = stopping_inputs(v_ego=0.5, plan_speeds=_decaying_speeds(0.5))
  cs.aEgo = -FINAL_A0_MIN
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._final_t is not None
  assert control._final_a0 == pytest.approx(FINAL_A0_DIST)

  # A stronger entry is clamped to FINAL_A0_MAX. aEgo must still clear the
  # stopping threshold for the window to open, so it is the last command, not
  # the measured decel, that carries the harder value.
  control._final_t = None
  control._final_done = False
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = -FINAL_A0_MAX
  cs, plan, radar = stopping_inputs(v_ego=0.5, plan_speeds=_decaying_speeds(0.5))
  cs.aEgo = -0.4
  control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)
  assert control._final_t is not None
  assert control._final_a0 == pytest.approx(FINAL_A0_MAX)


def test_butter_stop_off_forces_final_stop_false(monkeypatch):
  control, params = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  assert control.final_stop
  params.put_int("ButterStop", 0)
  control._refresh_butter_stop()
  assert not control.butter_stop
  assert not control.final_stop
  params.put_int("ButterStop", 1)
  control._refresh_butter_stop()
  assert control.final_stop


def test_final_stop_defaults_off(monkeypatch):
  control, _ = make_control(monkeypatch, values={"ButterStop": 1})
  assert not control.final_stop, "ButterStopFinal must default to off"


def test_window_profile_drives_the_command(monkeypatch):
  """While the window is open the command follows final_profile, not the PID."""
  monkeypatch.setattr(lc, "cloudlog", SimpleNamespace(event=lambda *a, **k: None))
  control, _ = make_control(monkeypatch, values={"ButterStop": 1, "ButterStopFinal": 1})
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = -0.4

  # No planned stop and no lead, so only the profile can shape the command.
  cruise_plan = np.full(len(PLAN_T), 3.0)
  _stop_frame(control, 0.5, a_ego=-0.32, speeds=cruise_plan)
  assert control._final_t is not None
  v0, a0 = control._final_v0, control._final_a0
  out = _stop_frame(control, 0.5, a_ego=-0.32, speeds=cruise_plan)[0]
  # One tick already elapsed: the reference at t=DT_CTRL plus the low-speed
  # velocity feedback on the measured (unmoving) vEgo.
  a_ref, v_ref, _ = final_profile(DT_CTRL, v0, a0)
  expected = a_ref + FINAL_KV * (v_ref - 0.5) if v_ref >= FINAL_FB_V else a_ref
  assert out == pytest.approx(max(FINAL_U_MIN, min(FINAL_U_MAX, expected)), abs=1e-6)
