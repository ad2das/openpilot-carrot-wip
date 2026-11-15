"""K5 stop replay: the real logs' five window spans through the real LongControl.

The fixture (data/k5_stop_windows.csv) is cut from the W-K5R1 cache by
2026-11-15-k5/W-K5F1/make_fixture.py, one row per carState event of the K5 drive
(52002533): t, v, aEgo, should_stop, lead_on, lead_dRel, lcs_logged, u_logged.

The spans:
  734 558.70-559.60  the window opened at 558.838 and the under-response guard killed it 0.30 s
                     later (559.132, v 0.293); the car then crawled 3.2 s and had to re-stop.
  733 160.60-161.50  the same, and that stop crawled 15.7 s.
  733 298.30-298.60  pid handed off to stopping at v 0.507, above the old 0.45 ceiling, so no
                     window ever opened and the stop crawled 1.7 s.
  735 212.70-212.95  the same at v 0.489, and that stop crawled 12.2 s.
  734 403.80-406.20  the window that worked: opened at 403.893, ran to v<0.05 at 406.112.

R0 replays the BASE exit rule from the logged columns only (constants copied into this
file) and must reproduce the logged open/exit ticks before any of the new-code assertions
mean anything: if the fixture or the reading of it is wrong, R0 fails first.

R1-R4 drive the same rows through the real LongControl, one update() per carState, and
assert what the final stop does now.
"""
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

# Same Windows import shims as test_final_stop.py: these tests never construct the
# real Params backend or the solver.
if sys.platform == "win32":
  hardware_module = ModuleType("openpilot.system.hardware")
  hardware_module.PC = True
  sys.modules.setdefault("openpilot.system.hardware", hardware_module)

  hw_module = ModuleType("openpilot.system.hardware.hw")

  class _WinPaths:
    @staticmethod
    def comma_home() -> str:
      return os.path.join(os.environ.get("TEMP", "C:\\Windows\\Temp"), ".comma")

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
      return os.path.join(os.environ.get("TEMP", "C:\\Windows\\Temp"), "comma_download_cache") + "/"

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
  PROBE_FIRM_MAX, PROBE_GUARD_T, STANDSTILL_V, probe_weak,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.tests.test_butter_stop import stopping_inputs
from openpilot.selfdrive.controls.tests.test_final_stop import _decaying_speeds, _window_control

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "k5_stop_windows.csv")
COLUMNS = ("t", "v", "aEgo", "should_stop", "lead_on", "lead_dRel", "lcs_logged", "u_logged")

# The spans as cut by make_fixture.py, and what the log's own events say happened in them.
SPANS = (
  ("734_558", 558.70, 559.60),
  ("733_160", 160.60, 161.50),
  ("733_298", 298.30, 298.60),
  ("735_212", 212.70, 212.95),
  ("734_403", 403.80, 406.20),
)
LOGGED_OPEN = {"734_558": 558.838, "733_160": 160.739, "734_403": 403.893}
LOGGED_GUARD = {"734_558": 559.132, "733_160": 161.037}
LOGGED_HANDOFF = {"733_298": 298.443, "735_212": 212.808}
LOGGED_STANDSTILL_EXIT = {"734_403": 406.112}
TOL = 0.02

# The base rule at d42590275, copied: the replay has to prove the logs reproduce the old
# behaviour before anything about the new one can be believed.
BASE_PROBE_V_MAX = 0.45
BASE_PROBE_MAX_T = 3.0
BASE_GUARD_T = 0.3
BASE_GUARD_V = 0.10
BASE_STANDSTILL_V = 0.05
BASE_MIN_FINAL_GAP_ABS = 3.0
BASE_STOPPING_ACCEL = -0.5
LCS_PID = 1


def _rows():
  if not os.path.exists(DATA):
    pytest.fail("FAIL_NO_FIXTURE %s (run 2026-11-15-k5/W-K5F1/make_fixture.py)" % DATA)
  out = []
  with open(DATA, encoding="utf-8") as handle:
    header = handle.readline().strip().split(",")
    assert tuple(header) == COLUMNS, header
    for line in handle:
      if not line.strip():
        continue
      values = line.strip().split(",")
      assert len(values) == len(COLUMNS), line
      out.append(dict(zip(COLUMNS, (float(x) for x in values), strict=True)))
  return out


def _span(name):
  t0, t1 = next((a, b) for n, a, b in SPANS if n == name)
  rows = [r for r in _rows() if t0 - 1e-9 <= r["t"] <= t1 + 1e-9]
  assert len(rows) >= 20, (name, len(rows))
  return rows


def _base_replay(rows):
  """The pre-change exit rule over the logged columns: (open_t, exit_t, reason)."""
  open_t = None
  weak_t = 0.0
  for row in rows:
    if open_t is None:
      lead_close = bool(row["lead_on"]) and row["lead_dRel"] < BASE_MIN_FINAL_GAP_ABS
      if (row["should_stop"] == 1 and int(row["lcs_logged"]) == LCS_PID
          and row["aEgo"] > BASE_STOPPING_ACCEL and row["v"] <= BASE_PROBE_V_MAX
          and not lead_close):
        open_t = row["t"]
      continue
    weak = row["v"] >= BASE_GUARD_V and row["aEgo"] > probe_weak(row["v"])
    weak_t = weak_t + DT_CTRL if weak else 0.0
    terms = []
    if row["should_stop"] != 1:
      terms.append("should_stop")
    if int(row["lcs_logged"]) != LCS_PID:
      terms.append("left_pid")
    if bool(row["lead_on"]) and row["lead_dRel"] < BASE_MIN_FINAL_GAP_ABS:
      terms.append("lead_close")
    if row["v"] < BASE_STANDSTILL_V:
      terms.append("standstill")
    if row["t"] - open_t >= BASE_PROBE_MAX_T:
      terms.append("timeout")
    if weak_t >= BASE_GUARD_T - 1e-9:
      terms.append("guard")
    if terms:
      return open_t, row["t"], terms[0]
  return open_t, None, None


class _EventLog:
  def __init__(self):
    self.events = []

  def event(self, name, **kwargs):
    self.events.append(dict(kwargs, event=name))

  def named(self, name):
    return [e for e in self.events if e["event"] == name]


def _replay(monkeypatch, name):
  """One span through the real LongControl, one update() per logged carState."""
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = _window_control(monkeypatch, values={"ButterStop": 1})
  rows = _span(name)
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = rows[0]["u_logged"]
  trace = []
  for row in rows:
    cs, plan, radar = stopping_inputs(v_ego=row["v"], plan_speeds=_decaying_speeds(row["v"]))
    cs.aEgo = row["aEgo"]
    plan.shouldStop = row["should_stop"] == 1
    radar.leadOne = SimpleNamespace(status=bool(row["lead_on"]), dRel=row["lead_dRel"],
                                    aLeadK=0.0, vLead=0.0)
    out = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0]
    trace.append({"t": row["t"], "v": row["v"], "out": float(out),
                  "open": control._final_t is not None, "done": bool(control._final_done),
                  "extra": float(control._probe_extra), "weak_t": float(control._probe_weak_t),
                  "state": int(control.long_control_state),
                  "logged_u": row["u_logged"]})
  return control, trace, events


def _first_open(trace):
  for row in trace:
    if row["open"]:
      return row
  return None


# ---------------------------------------------------------------------------
# R0: the fixture reproduces the base behaviour
# ---------------------------------------------------------------------------

def test_r0_the_fixture_reproduces_the_base_rule():
  """The old exit rule on the logged columns gives the logged opens and exits.

  This is the guard for the fixture itself and for how the replay reads it: the base
  0.45 m/s ceiling, the base 3 s timeout and the base hand-back guard, in base source
  order. If these numbers drift, the fixture was cut wrong and every R1-R4 number below
  is meaningless.
  """
  for name, expected in LOGGED_OPEN.items():
    open_t, exit_t, reason = _base_replay(_span(name))
    assert open_t is not None, ("FAIL_REPLAY_BASE no open", name)
    assert abs(open_t - expected) <= TOL, ("FAIL_REPLAY_BASE open", name, open_t, expected)
  for name, expected in LOGGED_GUARD.items():
    open_t, exit_t, reason = _base_replay(_span(name))
    assert reason == "guard", ("FAIL_REPLAY_BASE exit reason", name, reason)
    assert abs(exit_t - expected) <= TOL, ("FAIL_REPLAY_BASE guard exit", name, exit_t, expected)
  open_t, exit_t, reason = _base_replay(_span("734_403"))
  assert reason == "standstill", ("FAIL_REPLAY_BASE exit reason", "734_403", reason)
  assert abs(exit_t - LOGGED_STANDSTILL_EXIT["734_403"]) <= TOL, ("FAIL_REPLAY_BASE standstill exit",
                                                                 exit_t)
  # The two stops that never opened: pid handed off above the base ceiling.
  for name in LOGGED_HANDOFF:
    open_t, exit_t, reason = _base_replay(_span(name))
    assert open_t is None, ("FAIL_REPLAY_BASE opened above the ceiling", name, open_t)


# ---------------------------------------------------------------------------
# R1/R2: the weak windows stay open instead of handing the stop back
# ---------------------------------------------------------------------------

def test_r1_734_window_opens_and_never_hands_back(monkeypatch):
  _, trace, events = _replay(monkeypatch, "734_558")
  first = _first_open(trace)
  assert first is not None, "the window must open on the 734 stop"
  assert abs(first["t"] - LOGGED_OPEN["734_558"]) <= TOL, (first["t"], LOGGED_OPEN["734_558"])

  open_index = trace.index(first)
  for row in trace[open_index:]:
    assert row["open"], ("window closed while rolling", row["t"], row["v"])
    assert row["state"] == int(LongCtrlState.pid), ("left pid", row["t"], row["state"])
  assert not trace[-1]["done"], "the window must still be open at the end of the span"

  guards = events.named("butterStopProbeGuard")
  assert len(guards) == 1, [(g.get("t"), g.get("vEgo")) for g in guards]
  assert guards[0]["firm"] is True, guards[0]

  after = [row for row in trace if row["t"] > LOGGED_GUARD["734_558"] + 0.008]
  assert after, "the span must reach past the logged guard"
  for row in after:
    assert row["extra"] < 0.0, ("the car stayed weak, so the extra must be building",
                                row["t"], row["extra"])


def test_r2_733_window_survives_the_logged_trip(monkeypatch):
  _, trace, events = _replay(monkeypatch, "733_160")
  first = _first_open(trace)
  assert first is not None, "the window must open on the 733 stop"
  assert abs(first["t"] - LOGGED_OPEN["733_160"]) <= TOL, (first["t"], LOGGED_OPEN["733_160"])

  at_trip = next(row for row in trace if row["t"] >= LOGGED_GUARD["733_160"])
  assert events.named("butterStopProbeGuard"), "the guard must have logged by the trip time"
  assert at_trip["extra"] < 0.0, ("the extra must be building at the logged trip",
                                  at_trip["t"], at_trip["extra"])

  open_index = trace.index(first)
  for row in trace[open_index:]:
    assert row["open"], ("window closed while rolling", row["t"], row["v"])
    assert row["state"] == int(LongCtrlState.pid), ("left pid", row["t"], row["state"])
  assert not trace[-1]["done"], "the window must still be open at the end of the span"
  # The span has to reach well past the logged trip for the assertion above to mean
  # anything: the cut ends at the last logged row, ~6 ms short of 161.50.
  assert trace[-1]["t"] - LOGGED_GUARD["733_160"] >= 0.40, trace[-1]["t"]


# ---------------------------------------------------------------------------
# R3: the stops that never opened now open in pid before the logged hand-off
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LOGGED_HANDOFF))
def test_r3_handoff_above_the_old_ceiling_now_opens_the_window(monkeypatch, name):
  control, trace, _ = _replay(monkeypatch, name)
  first = _first_open(trace)
  assert first is not None, ("the window must open on %s" % name)
  assert first["t"] <= LOGGED_HANDOFF[name] + 1e-9, (first["t"], LOGGED_HANDOFF[name])
  assert first["state"] == int(LongCtrlState.pid), first["state"]
  assert first["v"] > BASE_PROBE_V_MAX, ("this span exists because the hand-off is above the old "
                                        "ceiling", first["v"])
  assert control._final_v0 <= 1.0 + 1e-9, control._final_v0

  # The row the log has leaving pid is still pid here.
  row = min(trace, key=lambda r: abs(r["t"] - LOGGED_HANDOFF[name]))
  assert row["state"] == int(LongCtrlState.pid), (row["t"], row["state"])


# ---------------------------------------------------------------------------
# R4: the window that ran to rest is untouched
# ---------------------------------------------------------------------------

def test_r4_734_good_window_is_unchanged(monkeypatch):
  _, trace, events = _replay(monkeypatch, "734_403")
  first = _first_open(trace)
  assert first is not None, "the window must open on the good 734 stop"
  assert abs(first["t"] - LOGGED_OPEN["734_403"]) <= TOL, (first["t"], LOGGED_OPEN["734_403"])

  for row in trace:
    assert row["extra"] == 0.0, ("no firm-up on a window the car answers", row["t"], row["extra"])
  assert not events.named("butterStopProbeGuard"), [e.get("t") for e in events.named(
    "butterStopProbeGuard")]

  exit_index = next(i for i, row in enumerate(trace) if row["v"] < STANDSTILL_V)
  exit_row = trace[exit_index]
  assert abs(exit_row["t"] - LOGGED_STANDSTILL_EXIT["734_403"]) <= TOL, (exit_row["t"],
                                                                       LOGGED_STANDSTILL_EXIT[
                                                                         "734_403"])
  open_index = trace.index(first)
  for row in trace[open_index:exit_index]:
    assert row["open"], ("window closed early", row["t"], row["v"])
  for row in trace[exit_index:]:
    assert not row["open"], ("window still open at v<%.2f" % STANDSTILL_V, row["t"], row["v"])
  assert trace[-1]["done"], "the stop keeps its one window"


def test_the_extra_never_exceeds_the_constant(monkeypatch):
  """Across every span the firm-up stays within PROBE_FIRM_MAX and only ever builds."""
  for name in ("734_558", "733_160"):
    _, trace, _ = _replay(monkeypatch, name)
    extras = [row["extra"] for row in trace]
    assert min(extras) >= -PROBE_FIRM_MAX - 1e-9, (name, min(extras))
    for previous, current in zip(extras, extras[1:], strict=False):
      assert current <= previous + 1e-12, (name, previous, current)
    assert PROBE_GUARD_T == pytest.approx(0.3)
