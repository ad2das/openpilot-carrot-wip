"""Self-throttle watchdog for tlvd.

Suspends inference when the rest of the stack shows stress so the detector can
never add load on top of an already struggling system. Pure logic (no messaging
imports); tlvd feeds it the latest modelV2 / deviceState values.

Suspend triggers (per task spec):
  - modelV2.frameDropPerc rises > 1.0 above its rolling baseline,
  - modelV2 model execution time p95 rises > 10% above baseline,
  - deviceState.thermalStatus >= yellow.
Baseline = rolling window of samples observed while tlvd was NOT inferring.
Resume after 30 s clean with exponential backoff (30 -> 60 -> 120 -> 240 s) when
suspensions repeat within the re-arm window.
"""
from __future__ import annotations

from collections import deque

import numpy as np

STATUS_BASELINE = "baseline"
STATUS_RUNNING = "running"
STATUS_SUSPENDED = "suspended"

THERMAL_YELLOW = 1


class SuspendWatchdog:
  def __init__(self, *, baseline_min_s: float = 10.0, baseline_min_samples: int = 50, window: int = 600,
               frame_drop_delta: float = 1.0, exec_p95_ratio: float = 1.10,
               clean_resume_s: float = 30.0, rearm_window_s: float = 60.0, max_clean_resume_s: float = 240.0,
               thermal_suspend: int = THERMAL_YELLOW) -> None:
    self.baseline_min_s = float(baseline_min_s)
    self.baseline_min_samples = int(baseline_min_samples)
    self.frame_drop_delta = float(frame_drop_delta)
    self.exec_p95_ratio = float(exec_p95_ratio)
    self.clean_resume_s = float(clean_resume_s)
    self.rearm_window_s = float(rearm_window_s)
    self.max_clean_resume_s = float(max_clean_resume_s)
    self.thermal_suspend = int(thermal_suspend)

    self._window = int(window)
    self._fd_live: deque[float] = deque(maxlen=self._window)
    self._exec_live: deque[float] = deque(maxlen=self._window)
    self._fd_base: deque[float] = deque(maxlen=self._window)
    self._exec_base: deque[float] = deque(maxlen=self._window)

    self._first_sample_t: float | None = None
    self._last_sample_t: float | None = None
    self._samples = 0
    self._thermal = 0

    self._inferring = False
    self._status = STATUS_BASELINE
    self._reason = STATUS_BASELINE
    self._clean_since: float | None = None
    self._clean_required = self.clean_resume_s
    self._last_resume_t: float | None = None
    self._suspends = 0

  @property
  def inferring(self) -> bool:
    return self._inferring

  @property
  def status(self) -> str:
    return self._status

  @property
  def reason(self) -> str:
    return self._reason

  @property
  def clean_required_s(self) -> float:
    return self._clean_required

  def update(self, now: float, *, frame_drop: float | None = None, exec_time: float | None = None,
             thermal: int | None = None) -> bool:
    """Feed one loop tick; returns whether inference may run."""
    if frame_drop is not None:
      value = float(frame_drop)
      if np.isfinite(value):
        self._fd_live.append(value)
        if not self._inferring:
          self._fd_base.append(value)
    if exec_time is not None:
      value = float(exec_time)
      if np.isfinite(value) and value > 0.0:
        self._exec_live.append(value)
        if not self._inferring:
          self._exec_base.append(value)
    if thermal is not None:
      self._thermal = int(thermal)
    if frame_drop is not None or exec_time is not None or thermal is not None:
      self._last_sample_t = float(now)
      if self._first_sample_t is None:
        self._first_sample_t = float(now)
      self._samples += 1

    if self._inferring:
      reason = self._trigger_reason()
      if reason is not None:
        self._suspend(now, reason)
    elif self._status == STATUS_BASELINE:
      if self._baseline_ready():
        self._arm(now)
    elif self._status == STATUS_SUSPENDED:
      if self._trigger_reason() is None:
        if self._clean_since is None:
          self._clean_since = float(now)
        if now - self._clean_since >= self._clean_required:
          self._arm(now)
      else:
        self._clean_since = None
    return self._inferring

  def _baseline_ready(self) -> bool:
    if self._samples < self.baseline_min_samples or self._first_sample_t is None or self._last_sample_t is None:
      return False
    return (self._last_sample_t - self._first_sample_t) >= self.baseline_min_s

  def _arm(self, now: float) -> None:
    self._last_resume_t = float(now)
    self._inferring = True
    self._status = STATUS_RUNNING
    self._reason = ""
    self._clean_since = None

  def _suspend(self, now: float, reason: str) -> None:
    # Backoff: re-suspending within the re-arm window of the last resume doubles
    # the clean requirement for the next resume (30 -> 60 -> 120 -> 240 s).
    if self._suspends > 0 and self._last_resume_t is not None and (now - self._last_resume_t) <= self.rearm_window_s:
      self._clean_required = min(self._clean_required * 2.0, self.max_clean_resume_s)
    else:
      self._clean_required = self.clean_resume_s
    self._suspends += 1
    self._inferring = False
    self._status = STATUS_SUSPENDED
    self._reason = reason
    self._clean_since = None
    # Re-baseline from the now interference-free stream before the next resume.
    self._fd_base.clear()
    self._exec_base.clear()

  def _trigger_reason(self) -> str | None:
    if self._thermal >= self.thermal_suspend:
      return f"thermal({self._thermal})"
    if len(self._fd_live) >= 20 and len(self._fd_base) >= 20:
      live = _recent_mean(self._fd_live, 50)
      base = float(np.mean(np.fromiter(self._fd_base, dtype=np.float64, count=len(self._fd_base))))
      if live > base + self.frame_drop_delta:
        return "model_frame_drop"
    if len(self._exec_live) >= 20 and len(self._exec_base) >= 20:
      live_p95 = _recent_p95(self._exec_live, 150)
      base_p95 = self._p95(self._exec_base)
      if base_p95 > 0.0 and live_p95 > base_p95 * self.exec_p95_ratio:
        return "model_exec_time"
    return None

  @staticmethod
  def _p95(values: deque[float]) -> float:
    if not values:
      return 0.0
    return float(np.percentile(np.fromiter(values, dtype=np.float64, count=len(values)), 95.0))


def _recent(values: deque[float], n: int) -> list[float]:
  count = min(len(values), n)
  return [values[i] for i in range(len(values) - count, len(values))]


def _recent_mean(values: deque[float], n: int) -> float:
  recent = _recent(values, n)
  return float(np.mean(recent)) if recent else 0.0


def _recent_p95(values: deque[float], n: int) -> float:
  recent = _recent(values, n)
  return float(np.percentile(recent, 95.0)) if recent else 0.0
