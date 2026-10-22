"""Record where the manager main loop hangs. Instrumentation only: never kills or restarts anything.

Route 46d (2026-10-22): managerState stopped at 31.4 s while the manager process stayed alive
and busy, and nothing showed where the loop was stuck.
"""
import faulthandler
import os
import sys
import threading
import time
import traceback

STALL_S = 5.0          # a loop iteration longer than this is a stall
REPEAT_S = 30.0        # re-report while still stalled
MAX_REPORTS = 6        # per stall
CHECK_S = 1.0
STACK_CAP = 16000      # chars per report
FAULT_DIR = "/data/media/0/carrot_manager_stall"
FAULT_MAX_BYTES = 1_000_000


def _cloudlog_error(msg: dict) -> None:
  from openpilot.common.swaglog import cloudlog
  cloudlog.error(msg)


class StallWatchdog:
  def __init__(self, *, stall_s=STALL_S, repeat_s=REPEAT_S, max_reports=MAX_REPORTS, check_s=CHECK_S,
               clock=time.monotonic, log=None, fault_path=os.path.join(FAULT_DIR, "faulthandler.log"), main_ident=None):
    self.stall_s = stall_s
    self.repeat_s = repeat_s
    self.max_reports = max_reports
    self.check_s = check_s
    self.clock = clock
    self.log = log or _cloudlog_error
    self.fault_path = fault_path
    self.main_ident = main_ident if main_ident is not None else threading.get_ident()
    self._last = clock()
    self._reports = 0
    self._last_report = None
    self._fh = None
    self._thread = None

  def start(self) -> None:
    try:
      os.makedirs(os.path.dirname(self.fault_path), exist_ok=True)
      if os.path.exists(self.fault_path) and os.path.getsize(self.fault_path) > FAULT_MAX_BYTES:
        open(self.fault_path, "w").close()
      self._fh = open(self.fault_path, "a")
    except OSError:
      self._fh = None
    self._thread = threading.Thread(target=self._run, name="manager_stall_watchdog", daemon=True)
    self._thread.start()

  def kick(self) -> None:
    now = self.clock()
    if self._reports:
      self._safe_log({"event": "manager_stall_end", "stalled_s": round(now - self._last, 2), "reports": self._reports})
      self._reports = 0
      self._last_report = None
    self._last = now
    if self._fh is not None:
      try:
        faulthandler.dump_traceback_later(self.stall_s * 2, repeat=False, file=self._fh)
      except Exception:
        pass

  def _check(self, now: float) -> None:
    stalled = now - self._last
    if stalled < self.stall_s or self._reports >= self.max_reports:
      return
    if self._last_report is not None and now - self._last_report < self.repeat_s:
      return
    self._reports += 1
    self._last_report = now
    self._safe_log({"event": "manager_stall", "stalled_s": round(stalled, 2), "report": self._reports,
                    "stacks": self._stacks()})

  def _stacks(self) -> dict[str, str]:
    own = threading.get_ident()
    names = {t.ident: t.name for t in threading.enumerate()}
    frames = sys._current_frames()
    order = sorted((i for i in frames if i != own), key=lambda i: i != self.main_ident)
    out, total = {}, 0
    for ident in order:
      text = "".join(traceback.format_stack(frames[ident]))
      key = f"{names.get(ident, '?')}({ident}){'*' if ident == self.main_ident else ''}"
      text = text[-max(0, STACK_CAP - total):] if total + len(text) > STACK_CAP else text
      if not text:
        break
      out[key] = text
      total += len(text)
    return out

  def _safe_log(self, msg: dict) -> None:
    try:
      self.log(msg)
    except Exception:
      pass

  def _run(self) -> None:
    while True:
      time.sleep(self.check_s)
      try:
        self._check(self.clock())
      except Exception:
        pass
