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
  734 403.80-406.20  the window that worked: opened at 403.893 and handed the stop off at
                     v<0.05 at 406.112.

The second fixture (data/k5_stop_zero_737.csv, cut by 2026-11-16-k5-restop/W-K5Z1/make_fixture_737.py
from route 00000737--2d13190285) adds the columns the wheels-stopped rule needs (vEgoRaw and the
four wheel speeds) on the two stops of build c33bd6f1a where the old window handed a still-rolling
car to StopReq 1:
  737_92   90.70-96.60  opened 90.840, handed off 92.514 with every wheel at 0.04-0.05, crept 3.5 s,
                        rolled back to 0.159 m/s, -2.5 rescue, true rest 96.25
  737_188 186.45-193.10 opened 186.577, handed off 188.858, crept 3.6 s, rest 192.829

R0 replays the BASE exit rule from the logged columns only (constants copied into this
file) and must reproduce the logged open/exit ticks before any of the new-code assertions
mean anything: if the fixture or the reading of it is wrong, R0 fails first.

R1-R4 drive the 734/733 rows through the real LongControl, one update() per carState, and
assert what the final stop does now. R5-R7 are validate-first on the 737 fixture and then pin
the zero rule: the window holds through the logged hand-off, stays pid, and closes on the
fixture's own 10-row wheels-stopped run.
"""
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
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
  MIN_FINAL_GAP_ABS, PROBE_FIRM_MAX, PROBE_GUARD_T, PROBE_GUARD_V, PROBE_LOW_MAX, PROBE_ZERO_T,
  PROBE_ZERO_V, STANDSTILL_V, probe_weak, rest_speed,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.tests.test_butter_stop import stopping_inputs
from openpilot.selfdrive.controls.tests.test_final_stop import _decaying_speeds, _window_control

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "k5_stop_windows.csv")
COLUMNS = ("t", "v", "aEgo", "should_stop", "lead_on", "lead_dRel", "lcs_logged", "u_logged")

# Route 00000737--2d13190285 (build c33bd6f1a), cut by 2026-11-16-k5-restop/W-K5Z1/make_fixture_737.py.
# It is the route the zero rule is for: both stops handed off while the wheels still turned, the car
# crept for 3.5 s / 3.6 s under StopReq 1, and stop 1 rolled back to 0.159 m/s before a -2.5 rescue.
DATA_737 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "k5_stop_zero_737.csv")
COLUMNS_737 = ("t", "v", "vraw", "ws_fl", "ws_fr", "ws_rl", "ws_rr", "aEgo", "should_stop",
               "lead_on", "lead_dRel", "lcs_logged", "u_logged")
SPANS_737 = (
  ("737_92", 90.70, 96.60),
  ("737_188", 186.45, 193.10),
)
LOGGED_OPEN_737 = {"737_92": 90.840, "737_188": 186.577}
LOGGED_HANDOFF_737 = {"737_92": 92.514, "737_188": 188.858}
LOGGED_REST_737 = {"737_92": 96.250, "737_188": 192.829}   # 10th consecutive rest_speed <= 0.01 row
REST_ROWS = 10
REST_V = 0.01

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
REPLAY_TOL = 0.005   # how closely the replay reproduces the logged command above PROBE_GUARD_V

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
                  "low": float(control._probe_low_extra),
                  "state": int(control.long_control_state),
                  "logged_u": row["u_logged"]})
  return control, trace, events


def _first_open(trace):
  for row in trace:
    if row["open"]:
      return row
  return None


# ---------------------------------------------------------------------------
# Route 737: the fixture with wheel speeds, and the zero rule's own replay
# ---------------------------------------------------------------------------

def _rows_737():
  if not os.path.exists(DATA_737):
    pytest.fail("FAIL_NO_FIXTURE_737 %s (run W-K5Z1/make_fixture_737.py)" % DATA_737)
  out = []
  with open(DATA_737, encoding="utf-8") as handle:
    header = handle.readline().strip().split(",")
    assert tuple(header) == COLUMNS_737, header
    for line in handle:
      if not line.strip():
        continue
      values = line.strip().split(",")
      assert len(values) == len(COLUMNS_737), line
      out.append(dict(zip(COLUMNS_737, (float(x) for x in values), strict=True)))
  return out


def _span_737(name):
  t0, t1 = next((a, b) for n, a, b in SPANS_737 if n == name)
  rows = [r for r in _rows_737() if t0 - 1e-9 <= r["t"] <= t1 + 1e-9]
  assert len(rows) >= 20, (name, len(rows))
  return rows


def _row_rest(row):
  """rest_speed of a 737 fixture row, from its own columns."""
  return max(abs(row["v"]), abs(row["vraw"]),
             abs(row["ws_fl"]), abs(row["ws_fr"]), abs(row["ws_rl"]), abs(row["ws_rr"]))


def _replay_737(monkeypatch, name):
  """One 737 span through the real LongControl with vEgoRaw and the wheel speeds fed too."""
  events = _EventLog()
  monkeypatch.setattr(lc, "cloudlog", events)
  control, _ = _window_control(monkeypatch, values={"ButterStop": 1})
  rows = _span_737(name)
  assert np.isfinite(rows[0]["u_logged"]), (name, rows[0]["t"])
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = rows[0]["u_logged"]
  trace = []
  for row in rows:
    cs, plan, radar = stopping_inputs(v_ego=row["v"], plan_speeds=_decaying_speeds(row["v"]))
    cs.aEgo = row["aEgo"]
    cs.vEgoRaw = row["vraw"]
    cs.wheelSpeeds = SimpleNamespace(fl=row["ws_fl"], fr=row["ws_fr"],
                                     rl=row["ws_rl"], rr=row["ws_rr"])
    plan.shouldStop = row["should_stop"] == 1
    radar.leadOne = SimpleNamespace(status=bool(row["lead_on"]), dRel=row["lead_dRel"],
                                    aLeadK=0.0, vLead=0.0)
    out = control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0]
    rest = _row_rest(row)
    # The fixture columns must agree with what the controller itself computes from the CS it gets.
    assert rest == pytest.approx(rest_speed(cs), abs=1e-9), (name, row["t"], rest,
                                                             rest_speed(cs))
    trace.append({"t": row["t"], "v": row["v"], "row": row, "rest": rest,
                  "out": float(out),
                  "open": control._final_t is not None, "done": bool(control._final_done),
                  "extra": float(control._probe_extra), "low": float(control._probe_low_extra),
                  "zero_t": float(control._probe_zero_t),
                  "state": int(control.long_control_state),
                  "logged_u": row["u_logged"]})
  return control, trace, events


def _rest_run_start(trace):
  """Time of the first row of the first REST_ROWS-long rest_speed <= REST_V run."""
  run = 0
  start = None
  for row in trace:
    if row["rest"] <= REST_V:
      run += 1
      if run == 1:
        start = row["t"]
      if run >= REST_ROWS:
        return start
    else:
      run = 0
      start = None
  return None


def _rest_run_t(trace):
  """Time of the row that completes the first REST_ROWS-long rest_speed <= REST_V run."""
  run = 0
  for row in trace:
    run = run + 1 if row["rest"] <= REST_V else 0
    if run >= REST_ROWS:
      return row["t"]
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
# R4: the window that ran to rest now holds past the old exit
# ---------------------------------------------------------------------------

def test_r4_734_good_window_now_holds_to_zero(monkeypatch):
  """The good 734 window is identical while it rolls, then keeps the request past 406.112.

  That fixture has no wheel speeds, so rest_speed is vEgo alone and the window cannot see the
  wheels stop; it holds (and the near-rest extra builds) to the end of the cut instead of handing
  the stop back at the old vEgo < 0.05 exit.
  """
  _, trace, events = _replay(monkeypatch, "734_403")
  first = _first_open(trace)
  assert first is not None, "the window must open on the good 734 stop"
  assert abs(first["t"] - LOGGED_OPEN["734_403"]) <= TOL, (first["t"], LOGGED_OPEN["734_403"])

  for row in trace:
    assert row["extra"] == 0.0, ("no firm-up on a window the car answers", row["t"], row["extra"])
  assert not events.named("butterStopProbeGuard"), [e.get("t") for e in events.named(
    "butterStopProbeGuard")]

  # Identical to the logged command while the car still rolls at or above the guard speed. The
  # check starts one row after the open: that fixture's u_logged is the last command *before* each
  # carState (W-K5F1/make_fixture.py holds the previous value), so the window's own first tick has
  # no command of its own in the log to compare against.
  open_index = trace.index(first)
  mismatched = [(row["t"], row["out"], row["logged_u"]) for row in trace[open_index + 1:]
                if row["v"] >= PROBE_GUARD_V and row["state"] == int(LongCtrlState.pid)
                and abs(row["out"] - row["logged_u"]) > 0.005]
  assert not mismatched, ("FAIL_REPLAY_734 the command changed above the guard", mismatched[:8])

  # The window holds past the logged exit and the near-rest extra builds behind it.
  past = [row for row in trace if row["t"] > LOGGED_STANDSTILL_EXIT["734_403"]]
  assert past, ("the cut has to reach past the logged exit", trace[-1]["t"])
  for row in past:
    assert row["open"], ("window closed at the old vEgo<%.2f exit" % STANDSTILL_V,
                         row["t"], row["v"])
  assert trace[-1]["low"] < 0.0, "the near-rest extra must be building at the end of the cut"
  assert not trace[-1]["done"], "the window is still open at the end of the cut"


# ---------------------------------------------------------------------------
# R5-R7: route 737, where the hand-off left every wheel still turning
# ---------------------------------------------------------------------------

def test_r5_737_fixture_reproduces_c33bd6f1a_above_guard_v(monkeypatch):
  """Validate-first: at v >= 0.10 the new code must still reproduce the logged command.

  The zero rule cannot change anything up there (the near-rest extra only builds below
  PROBE_GUARD_V and the exit only fires at PROBE_ZERO_V), so any disagreement means this
  replay or the fixture is reading the log wrong, not that the behaviour changed.

  Only rows where the log itself was still in pid are compared. After its hand-off the old
  build ran the stopping state, and span 737_92 rolls back to 0.159 m/s inside that state
  (R_737_go416b.txt:245) -- those rows are the behaviour this change removes, not a fixture
  error, and the window has no logged command of its own to reproduce there.
  """
  for name in sorted(LOGGED_OPEN_737):
    control, trace, _ = _replay_737(monkeypatch, name)
    first = _first_open(trace)
    expected_open = min(trace, key=lambda r: abs(r["t"] - LOGGED_OPEN_737[name]))
    assert first is not None, ("FAIL_REPLAY_737 no window", name)
    assert abs(first["t"] - expected_open["t"]) <= 0.011, (name, first["t"],
                                                         LOGGED_OPEN_737[name])
    compared = 0
    mismatched = []
    for row in trace:
      if not row["open"] or row["v"] < PROBE_GUARD_V:
        continue
      if row["state"] != int(LongCtrlState.pid):
        continue
      if int(row["row"]["lcs_logged"]) != int(LongCtrlState.pid):
        continue
      compared += 1
      if abs(row["out"] - row["logged_u"]) > REPLAY_TOL:
        mismatched.append((round(row["t"], 3), round(row["out"], 4),
                           round(row["logged_u"], 4)))
    assert compared >= 90, ("FAIL_REPLAY_737 too few comparable rows", name, compared)
    assert not mismatched, ("FAIL_REPLAY_737", name, len(mismatched), mismatched[:8])
  assert control.final_window, "the 737 spans run the CAN-FD window"


def test_r6_737_window_holds_past_the_logged_hand_off(monkeypatch):
  """The window is still open at the hand-off, stays pid, and closes on the wheels-stopped run."""
  for name in sorted(LOGGED_OPEN_737):
    control, trace, events = _replay_737(monkeypatch, name)
    first = _first_open(trace)
    assert first is not None, ("the window must open on %s" % name)

    # The row the log leaves pid on is still open and still pid here.
    hand = min(trace, key=lambda r: abs(r["t"] - LOGGED_HANDOFF_737[name]))
    assert hand["open"], ("the window must still hold the logged hand-off", name, hand["t"])
    assert hand["state"] == int(LongCtrlState.pid), (name, hand["t"], hand["state"])
    assert hand["zero_t"] < PROBE_ZERO_T - 1e-9, (name, hand["t"], hand["zero_t"])
    # The whole reason for the change: at the logged hand-off the wheels were still turning.
    assert hand["rest"] > PROBE_ZERO_V, (name, hand["t"], hand["rest"])

    # It closes on the fixture's own 10-row rest run, not at the old vEgo exit: that run completes
    # REST_ROWS rows into a continuous rest_speed <= REST_V stretch, and the clock reaches
    # PROBE_ZERO_T on exactly that row (0.10 s = 10 rows at DT_CTRL).
    run_start = _rest_run_start(trace)
    run_t = _rest_run_t(trace)
    assert run_t is not None, (name, "no rest run in the span")
    assert abs(run_t - LOGGED_REST_737[name]) <= 0.011, (name, run_t, LOGGED_REST_737[name])
    close_row = next(row for row in trace if row["t"] >= run_t - 1e-9)
    assert not close_row["open"], ("the window must close on the rest run row", name,
                                   close_row["t"], close_row["rest"])
    assert close_row["done"], (name, close_row["t"])
    assert close_row["zero_t"] >= PROBE_ZERO_T - 1e-9, (name, close_row["t"], close_row["zero_t"])
    # Every row from the run's start to the row before the close is still open and still pid.
    inside = [row for row in trace if run_start - 1e-9 <= row["t"] < run_t - 1e-9]
    assert len(inside) == REST_ROWS - 1, (name, len(inside))
    for row in inside:
      assert row["open"], (name, row["t"], row["rest"])
      assert row["state"] == int(LongCtrlState.pid), ("left pid inside the window", name,
                                                      row["t"], row["state"])

    # Nothing before the run closed it: the window held from the open through the crawl.
    open_rows = [row for row in trace if row["open"]]
    assert open_rows[0]["t"] == first["t"], (name, open_rows[0]["t"], first["t"])
    assert open_rows[-1]["t"] < run_t + 1e-9, (name, open_rows[-1]["t"], run_t)
    for row in open_rows:
      assert row["state"] == int(LongCtrlState.pid), ("left pid inside the window", name,
                                                      row["t"], row["state"])

    ends = events.named("butterStopFinalEnd")
    assert len(ends) == 1, [(e.get("t"), e.get("reason")) for e in ends]
    assert ends[0]["reason"] == "wheels_stopped", ends[0]
    assert not trace[-1]["open"], "the window is closed by the end of the span"
    assert trace[-1]["done"]
    assert control.final_window, "the 737 spans must run the CAN-FD window"


def test_r7_737_command_firms_below_guard_v(monkeypatch):
  """Near rest the command is the probe table plus the near-rest extra, and never weaker."""
  for name in sorted(LOGGED_OPEN_737):
    _, trace, _ = _replay_737(monkeypatch, name)
    hand_t = LOGGED_HANDOFF_737[name]

    # These two stops have no lead at all before the hand-off; the only lead rows are 32 m away
    # and after both windows have closed, so nothing here is a lead decision.
    lead_rows = [row for row in trace if row["row"]["lead_on"] > 0]
    for row in lead_rows:
      assert row["row"]["lead_dRel"] > MIN_FINAL_GAP_ABS, (name, row["t"],
                                                          row["row"]["lead_dRel"])
    assert not [row for row in trace
                if row["row"]["lead_on"] > 0 and row["t"] <= hand_t + 1e-9], name

    near = [row for row in trace
            if row["open"] and row["rest"] < PROBE_GUARD_V and row["extra"] == 0.0]
    assert near, (name, "no near-rest open rows")
    for row in near:
      assert -0.30 - 1e-9 <= row["out"] <= -0.14 + 1e-9, (name, row["t"], row["rest"], row["out"])

    # Up to the logged hand-off the request only ever firms up.
    pre = [row for row in near if row["t"] <= hand_t + 1e-9]
    assert len(pre) >= 20, (name, len(pre))
    for previous, current in zip(pre, pre[1:], strict=False):
      assert current["out"] <= previous["out"] + 1e-9, (name, previous["t"], previous["out"],
                                                       current["t"], current["out"])

    # Where the log still had pid, this window never asks for notably more than the log's command.
    # The +0.005 is the replay's own reproduction error, not slack on the rule: one probe step
    # (~0.004 m/s^2) is what R5 measures as the row-to-row pairing jitter of u_logged.
    stronger = [(row["t"], row["out"], row["logged_u"]) for row in trace
                if row["open"] and int(row["row"]["lcs_logged"]) == 1
                and row["out"] > row["logged_u"] + REPLAY_TOL]
    assert not stronger, (name, stronger[:8])
    assert PROBE_LOW_MAX == pytest.approx(0.16)


def test_the_extra_never_exceeds_the_constant(monkeypatch):
  """Across every span the firm-up stays within PROBE_FIRM_MAX and only ever builds."""
  for name in ("734_558", "733_160"):
    _, trace, _ = _replay(monkeypatch, name)
    extras = [row["extra"] for row in trace]
    assert min(extras) >= -PROBE_FIRM_MAX - 1e-9, (name, min(extras))
    for previous, current in zip(extras, extras[1:], strict=False):
      assert current <= previous + 1e-12, (name, previous, current)
    assert PROBE_GUARD_T == pytest.approx(0.3)
