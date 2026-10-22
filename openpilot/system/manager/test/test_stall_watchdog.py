import ast
import time
from pathlib import Path

from openpilot.system.manager.stall_watchdog import StallWatchdog


def _busy_spin(seconds):
  end = time.monotonic() + seconds
  x = 0
  while time.monotonic() < end:
    x += 1
  return x


def test_busy_main_thread_stack_is_captured(tmp_path):
  logs = []
  fault = tmp_path / "fh.log"
  wd = StallWatchdog(stall_s=0.3, check_s=0.05, repeat_s=100.0, log=logs.append, fault_path=str(fault))
  wd.start()
  wd.kick()
  _busy_spin(1.0)
  stalls = [m for m in logs if m["event"] == "manager_stall"]
  assert len(stalls) == 1
  main = [v for k, v in stalls[0]["stacks"].items() if k.endswith("*")]
  assert len(main) == 1 and "_busy_spin" in main[0]
  wd.kick()
  ends = [m for m in logs if m["event"] == "manager_stall_end"]
  assert len(ends) == 1 and ends[0]["reports"] == 1 and ends[0]["stalled_s"] >= 0.9
  # faulthandler fires at 2 * stall_s = 0.6 s after the first kick
  text = fault.read_text()
  assert "Thread" in text and "_busy_spin" in text


def test_repeat_and_cap_with_injected_clock(tmp_path):
  logs, now = [], [0.0]
  wd = StallWatchdog(stall_s=5, repeat_s=30, max_reports=3, clock=lambda: now[0], log=logs.append,
                     fault_path=str(tmp_path / "x" / "fh.log"))
  wd.kick()
  for t in (0.5, 4.9, 5, 20, 35, 65, 95, 200):
    wd._check(t)
  assert [round(m["stalled_s"]) for m in logs] == [5, 35, 65]
  now[0] = 210.0
  wd.kick()
  wd.kick()
  assert logs[-1] == {"event": "manager_stall_end", "stalled_s": 210.0, "reports": 3} and len(logs) == 4


def test_unwritable_fault_path_does_not_raise(tmp_path):
  blocker = tmp_path / "file"
  blocker.write_text("x")
  logs = []
  wd = StallWatchdog(stall_s=1, clock=lambda: 10.0, log=logs.append, fault_path=str(blocker / "sub" / "fh.log"))
  wd.start()
  wd._last = 0.0
  wd._check(10.0)
  assert len(logs) == 1


def test_manager_loop_kicks_first():
  src = (Path(__file__).parents[1] / "manager.py").read_text()
  fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "manager_thread")
  loop = next(n for n in ast.walk(fn) if isinstance(n, ast.While))
  first = loop.body[0]
  assert ast.unparse(first) == "stall_watchdog.kick()"
  assert ast.unparse(fn).count("StallWatchdog(") == 1
