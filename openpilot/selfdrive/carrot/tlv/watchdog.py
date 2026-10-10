"""Self-throttle watchdog for tlvd.

Suspends inference when the rest of the stack shows stress so the detector can
never add load on top of an already struggling system. Pure logic (no messaging
imports); tlvd feeds it the latest modelV2 / deviceState values.

Suspend triggers (per task spec):
  - modelV2.frameDropPerc rises > 1.0 above its rolling baseline,
  - modelV2 model execution time p95 rises > 10% above baseline,
  - deviceState.thermalStatus >= yellow,
  - controlsState logMonoTime delta p99 exceeds the baseline + 2 ms (loop jitter),
  - roadCameraState frameId skip rate exceeds the baseline skip rate (frame gaps).
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
               controls_slack_s: float = 0.002, camera_gap_slack_per_s: float = 0.5, camera_min_span_s: float = 3.0,
               camera_window_s: float = 10.0,
               clean_resume_s: float = 30.0, rearm_window_s: float = 60.0, max_clean_resume_s: float = 240.0,
               thermal_suspend: int = THERMAL_YELLOW) -> None:
    self.baseline_min_s = float(baseline_min_s)
    self.baseline_min_samples = int(baseline_min_samples)
    self.frame_drop_delta = float(frame_drop_delta)
    self.exec_p95_ratio = float(exec_p95_ratio)
    self.controls_slack_s = float(controls_slack_s)
    self.camera_gap_slack_per_s = float(camera_gap_slack_per_s)
    self.camera_min_span_s = float(camera_min_span_s)
    self.camera_window_s = float(camera_window_s)
    self.clean_resume_s = float(clean_resume_s)
    self.rearm_window_s = float(rearm_window_s)
    self.max_clean_resume_s = float(max_clean_resume_s)
    self.thermal_suspend = int(thermal_suspend)

    self._window = int(window)
    self._fd_live: deque[float] = deque(maxlen=self._window)
    self._exec_live: deque[float] = deque(maxlen=self._window)
    self._fd_base: deque[float] = deque(maxlen=self._window)
    self._exec_base: deque[float] = deque(maxlen=self._window)
    self._ctl_live: deque[float] = deque(maxlen=self._window)
    self._ctl_base: deque[float] = deque(maxlen=self._window)
    self._gap_live: deque[float] = deque(maxlen=self._window)
    self._gap_base: deque[float] = deque(maxlen=self._window)
    self._fid_samples: deque[tuple[float, int]] = deque(maxlen=self._window)
    self._last_fid: int | None = None

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
             thermal: int | None = None, controls_interval: float | None = None,
             camera_frame_id: int | None = None) -> bool:
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
    if controls_interval is not None:
      value = float(controls_interval)
      if np.isfinite(value) and value > 0.0:
        self._ctl_live.append(value)
        if not self._inferring:
          self._ctl_base.append(value)
    if camera_frame_id is not None:
      fid = int(camera_frame_id)
      if self._last_fid is not None and fid >= self._last_fid:
        self._fid_samples.append((float(now), fid - self._last_fid - 1))
        self._prune_fid(float(now))
        rate = self._camera_gap_rate()
        if rate is not None:
          (self._gap_live if self._inferring else self._gap_base).append(rate)
      self._last_fid = fid
    if (frame_drop is not None or exec_time is not None or thermal is not None
        or controls_interval is not None or camera_frame_id is not None):
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
    if len(self._ctl_live) >= 20 and len(self._ctl_base) >= 20:
      live_p99 = _recent_p99(self._ctl_live, 150)
      base_p99 = self._p99(self._ctl_base)
      if base_p99 > 0.0 and live_p99 > base_p99 + self.controls_slack_s:
        return "controls_jitter"
    if len(self._gap_live) >= 20 and len(self._gap_base) >= 20:
      live_gap = _recent_mean(self._gap_live, 50)
      base_gap = float(np.mean(np.fromiter(self._gap_base, dtype=np.float64, count=len(self._gap_base))))
      if live_gap > base_gap + self.camera_gap_slack_per_s:
        return "camera_gaps"
    return None

  def _prune_fid(self, now: float) -> None:
    # Keep the rate local to camera_window_s so a new gap pattern shows up even
    # after a long clean run.
    while len(self._fid_samples) > 2 and self._fid_samples[1][0] < now - self.camera_window_s:
      self._fid_samples.popleft()

  def _camera_gap_rate(self) -> float | None:
    """Skipped road-camera frameIds per second over the retained window."""
    if len(self._fid_samples) < 20:
      return None
    t0 = self._fid_samples[0][0]
    t1 = self._fid_samples[-1][0]
    span = t1 - t0
    if span < self.camera_min_span_s:
      return None
    skipped = sum(skip for _, skip in self._fid_samples)
    return skipped / span

  @staticmethod
  def _p95(values: deque[float]) -> float:
    if not values:
      return 0.0
    return float(np.percentile(np.fromiter(values, dtype=np.float64, count=len(values)), 95.0))

  @staticmethod
  def _p99(values: deque[float]) -> float:
    if not values:
      return 0.0
    return float(np.percentile(np.fromiter(values, dtype=np.float64, count=len(values)), 99.0))


def _recent(values: deque[float], n: int) -> list[float]:
  count = min(len(values), n)
  return [values[i] for i in range(len(values) - count, len(values))]


def _recent_mean(values: deque[float], n: int) -> float:
  recent = _recent(values, n)
  return float(np.mean(recent)) if recent else 0.0


def _recent_p95(values: deque[float], n: int) -> float:
  recent = _recent(values, n)
  return float(np.percentile(recent, 95.0)) if recent else 0.0


def _recent_p99(values: deque[float], n: int) -> float:
  recent = _recent(values, n)
  return float(np.percentile(recent, 99.0)) if recent else 0.0
