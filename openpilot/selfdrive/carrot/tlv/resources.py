"""Resource governors for tlvd, per RESOURCE_BUDGET.md.

Pure logic with injectable clocks and readers so everything is unit-testable on
a PC. tlvd wires these to the real /proc files and os.times(); the classes here
never import messaging or touch the device by themselves.

  DutyCycleGovernor  - at most `budget` of one core averaged over a 2 s window,
                       measured with os.times(); paces frames at 1..10 Hz.
  CpuContentionGuard - /proc/stat cores 1-3 idle < 25% for 3 s -> cpu_contention.
  MemoryGuard        - RSS soft/hard limits and MemAvailable floor decisions.

The rules: if tlvd cannot meet a limit, it gets slower or stops. It never takes
resources from anything else.
"""
from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass

DEFAULT_CORES = (1, 2, 3)
MB = 1024 * 1024


def os_cpu_seconds() -> float:
  """Process CPU time (user+sys, all threads) in seconds; the budget metric."""
  times = os.times()
  return float(times[0]) + float(times[1])


def default_page_size() -> int:
  try:
    return int(os.sysconf("SC_PAGE_SIZE"))
  except (AttributeError, ValueError, OSError):
    return 4096


def read_text(path: str) -> str | None:
  try:
    with open(path) as f:
      return f.read()
  except OSError:
    return None


def write_oom_score_adj(value: int = 1000, path: str = "/proc/self/oom_score_adj") -> bool:
  """Ask the kernel to OOM-kill tlvd before anything else. Best effort."""
  try:
    with open(path, "w") as f:
      f.write(str(int(value)))
    return True
  except OSError:
    return False


def drop_caches(value: int = 1, path: str = "/proc/sys/vm/drop_caches") -> bool:
  """Best-effort page-cache drop (needs root; silently ignored otherwise)."""
  try:
    with open(path, "w") as f:
      f.write(str(int(value)))
    return True
  except OSError:
    return False


# ---------------------------------------------------------------------------
# Duty-cycle governor
# ---------------------------------------------------------------------------

class DutyCycleGovernor:
  """Caps tlvd at `budget` of one core and paces frames to respect it.

  Measured with os.times() (user+sys, all threads). The next frame is allowed
  only after `period_s`; the period adapts to the worst recent frame CPU cost
  (period = cost / budget) and is additionally pulled down whenever the actual
  duty over the rolling `window_s` window exceeds the budget. Rate is clamped
  to [min_hz, max_hz] and starts at start_hz.
  """

  def __init__(self, *, budget: float = 0.40, window_s: float = 2.0, min_hz: float = 1.0, max_hz: float = 10.0,
               start_hz: float = 5.0, cpu_clock=os_cpu_seconds, clock=time.monotonic) -> None:
    self.budget = float(budget)
    self.window_s = float(window_s)
    self.min_hz = float(min_hz)
    self.max_hz = float(max_hz)
    self._cpu_clock = cpu_clock
    self._clock = clock
    self._rate_hz = self._clamp_rate(start_hz)
    self._frames: deque[float] = deque(maxlen=10)
    self._samples: deque[tuple[float, float]] = deque()
    self._frame_start: tuple[float, float] | None = None
    self._next_allowed = 0.0

  @property
  def rate_hz(self) -> float:
    return self._rate_hz

  @property
  def period_s(self) -> float:
    return 1.0 / self._rate_hz

  @property
  def window_duty(self) -> float | None:
    """CPU seconds per wall second over the retained window, None until it spans."""
    if len(self._samples) < 2:
      return None
    t0, c0 = self._samples[0]
    t1, c1 = self._samples[-1]
    span = t1 - t0
    if span < max(0.5, 0.25 * self.window_s):
      return None
    return max(0.0, (c1 - c0) / span)

  def ready(self, now: float | None = None) -> bool:
    if now is None:
      now = self._clock()
    return now >= self._next_allowed

  def begin_frame(self, now: float | None = None) -> None:
    if now is None:
      now = self._clock()
    self._frame_start = (float(now), self._cpu_clock())

  def end_frame(self, now: float | None = None) -> None:
    if self._frame_start is None:
      return
    if now is None:
      now = self._clock()
    t0, c0 = self._frame_start
    self._frame_start = None
    now = float(now)
    cost = max(0.0, self._cpu_clock() - c0)
    self._frames.append(cost)
    self._samples.append((now, c0 + cost))
    # Keep one sample just before the window edge so the duty spans the window.
    while len(self._samples) > 1 and self._samples[1][0] < now - self.window_s:
      self._samples.popleft()
    self._recompute()
    self._next_allowed = now + self.period_s

  def _recompute(self) -> None:
    worst = max(self._frames) if self._frames else 0.0
    rate = self.max_hz if worst <= 1e-9 else self.budget / worst
    duty = self.window_duty
    if duty is not None and duty > self.budget:
      # Direct feedback: pull the rate down while the measured 2 s duty is over.
      rate = min(rate, self._rate_hz * self.budget / duty)
    self._rate_hz = self._clamp_rate(rate)

  def _clamp_rate(self, rate: float) -> float:
    return max(self.min_hz, min(self.max_hz, float(rate)))


# ---------------------------------------------------------------------------
# CPU credit budget (W99): strict long-run budget, cost-aware frame kinds
# ---------------------------------------------------------------------------

class CpuCreditBudget:
  """Strict long-run CPU budget with per-frame-kind cost estimation.

  Every tick charges ALL process CPU (os.times(), every thread: recorder writer,
  SubMaster, publish, inference - not just the model call) against a credit that
  refills at `budget` CPU-seconds per wall second and is capped at `cap_s`.
  A frame of kind K may start when credit >= estimate(K), where the estimate is
  a max of an EMA and 0.8x of the worst of the last 5 samples for that kind, so
  a changed frame cost is absorbed after a few frames.

  Guarantee: long-run process CPU <= budget; for any wall window W the process
  CPU used is <= cap_s + budget*W + one frame's estimation error (at most one
  under-estimated frame can start before the deficit is measured).

  There is NO minimum frame rate: when the budget is exhausted, frames are
  skipped (tlvd counts them as skipped_budget, not as droppedFrames).
  """

  CPU_BUDGET = 0.40
  CREDIT_CAP_S = 2.0
  CREDIT_START_S = 0.4
  MAX_FRAME_HZ = 10.0
  COST_EMA_ALPHA = 0.3
  COST_INIT = {"detect": 0.5, "classify": 0.05}
  DUTY_WINDOW_S = 10.0

  def __init__(self, *, budget: float = CPU_BUDGET, cap_s: float = CREDIT_CAP_S,
               cpu_clock=os_cpu_seconds, clock=time.monotonic) -> None:
    self.budget = float(budget)
    self.cap_s = float(cap_s)
    self._cpu_clock = cpu_clock
    self._clock = clock
    self._credit = float(self.CREDIT_START_S)
    # Baseline at CREATION (W127): tlvd builds the budget only after both
    # model warm-ups, so the one-time compile CPU (~160 s, route 00000443)
    # can never be charged to the first frames; credit starts at the normal
    # CREDIT_START_S with nothing used.
    self._last_now: float | None = float(self._clock())
    self._last_cpu: float | None = float(self._cpu_clock())
    self._samples: deque[tuple[float, float]] = deque()
    self._ema: dict[str, float] = dict(self.COST_INIT)
    self._costs: dict[str, deque[float]] = {kind: deque(maxlen=5) for kind in self.COST_INIT}
    self._frame: tuple[str, float] | None = None
    self._last_frame_start: float | None = None

  # -- accounting -----------------------------------------------------------

  def tick(self, now: float | None = None) -> None:
    """Refill by wall time, charge all process CPU since the previous tick."""
    if now is None:
      now = self._clock()
    now = float(now)
    cpu_now = float(self._cpu_clock())
    if self._last_now is not None:
      self._credit += self.budget * (now - self._last_now) - (cpu_now - self._last_cpu)
    self._credit = min(self._credit, self.cap_s)
    self._last_now, self._last_cpu = now, cpu_now
    self._samples.append((now, cpu_now))
    # Keep one sample just before the window edge so the duty spans the window.
    while len(self._samples) > 1 and self._samples[1][0] < now - self.DUTY_WINDOW_S:
      self._samples.popleft()

  # -- scheduling -----------------------------------------------------------

  def estimate(self, kind: str) -> float:
    """Seconds of CPU a frame of `kind` is expected to cost (COST_INIT first)."""
    costs = self._costs.get(kind)
    if not costs:
      return float(self.COST_INIT.get(kind, self.COST_INIT["classify"]))
    default = float(self.COST_INIT.get(kind, max(self.COST_INIT.values())))
    return max(self._ema.get(kind, default), 0.8 * max(costs))

  def can_run(self, kind: str, now: float | None = None) -> bool:
    """True when a frame of `kind` may start now without breaking the budget."""
    if now is None:
      now = self._clock()
    now = float(now)
    self.tick(now)
    if self._last_frame_start is not None and now - self._last_frame_start < 1.0 / self.MAX_FRAME_HZ:
      return False
    if self._credit >= self.estimate(kind):
      return True
    # A detector costlier than the cap still runs from a full bucket; credit
    # then goes negative, so the long-run average still holds.
    return kind == "detect" and self._credit >= self.cap_s - 1e-6

  def begin(self, kind: str, now: float | None = None) -> None:
    if now is None:
      now = self._clock()
    now = float(now)
    self.tick(now)
    self._frame = (str(kind), float(self._cpu_clock()))
    self._last_frame_start = now

  def end(self, now: float | None = None) -> None:
    frame = self._frame
    if frame is None:
      return
    if now is None:
      now = self._clock()
    now = float(now)
    kind, cpu0 = frame
    self._frame = None
    cost = max(0.0, float(self._cpu_clock()) - cpu0)
    costs = self._costs.setdefault(kind, deque(maxlen=5))
    costs.append(cost)
    default = float(self.COST_INIT.get(kind, cost))
    self._ema[kind] = self.COST_EMA_ALPHA * cost + (1.0 - self.COST_EMA_ALPHA) * self._ema.get(kind, default)
    self.tick(now)

  # -- introspection --------------------------------------------------------

  @property
  def credit(self) -> float:
    return self._credit

  @property
  def duty(self) -> float | None:
    """Process CPU seconds per wall second over DUTY_WINDOW_S, None until spanned."""
    if len(self._samples) < 2:
      return None
    t0, c0 = self._samples[0]
    t1, c1 = self._samples[-1]
    span = t1 - t0
    if span < 2.0:
      return None
    return max(0.0, (c1 - c0) / span)

  def estimate_ms(self, kind: str) -> float:
    return 1000.0 * self.estimate(kind)


# ---------------------------------------------------------------------------
# Detector throttle (classifier priority)
# ---------------------------------------------------------------------------

class DetectorThrottle:
  """Detector cadence while the lamp classifier carries the confirmation load.

  Base interval 0.2 s (5 Hz); halved to 0.4 s while tracked lights exist, so
  the governor budget goes to the classifier first and the detector runs at a
  lower rate instead of classification being dropped (CONTRACT.md).
  """

  def __init__(self, *, base_interval_s: float = 0.2, tracks_interval_s: float = 0.4) -> None:
    self.base_interval_s = float(base_interval_s)
    self.tracks_interval_s = float(tracks_interval_s)
    self._last = -float("inf")

  def due(self, now: float, *, has_tracks: bool = False) -> bool:
    interval = self.tracks_interval_s if has_tracks else self.base_interval_s
    return now - self._last >= interval

  def mark(self, now: float) -> None:
    self._last = float(now)


def detector_due(throttle: DetectorThrottle, now: float, *, classifier_present: bool,
                 has_tracks: bool) -> bool:
  """Classifier first: with a classifier only the detector is throttled.

  Without a classifier every frame keeps its detector run (current behavior).
  """
  if not classifier_present:
    return True
  return throttle.due(now, has_tracks=has_tracks)


# ---------------------------------------------------------------------------
# /proc/stat cores guard
# ---------------------------------------------------------------------------

def parse_proc_stat_totals(text: str, cores: tuple[int, ...] = DEFAULT_CORES) -> tuple[float, float] | None:
  """(idle, total) jiffies summed over `cores` from a /proc/stat snapshot.

  `idle` = idle+iowait, `total` = user+nice+system+idle+iowait+irq+softirq+steal
  (guest times are already folded into user/nice). None when no core matched.
  """
  want = {f"cpu{c}" for c in cores}
  idle = 0.0
  total = 0.0
  found = 0
  for line in text.splitlines():
    name, _, rest = line.partition(" ")
    if name not in want:
      continue
    fields = rest.split()
    if len(fields) < 8:
      continue
    values = [int(v) for v in fields[:8]]
    idle += values[3] + values[4]
    total += sum(values)
    found += 1
  if found != len(want) or total <= 0:
    return None
  return idle, total


class CpuContentionGuard:
  """Suspends tlvd when cores 1-3 lack idle headroom for the rest of the stack."""

  def __init__(self, *, cores: tuple[int, ...] = DEFAULT_CORES, min_idle: float = 0.25, hold_s: float = 3.0) -> None:
    self.cores = tuple(cores)
    self.min_idle = float(min_idle)
    self.hold_s = float(hold_s)
    self._prev: tuple[float, float] | None = None
    self._low_since: float | None = None
    self._reason: str | None = None

  @property
  def reason(self) -> str | None:
    return self._reason

  def update(self, now: float, stat_text: str | None) -> str | None:
    """Feed one /proc/stat snapshot; returns the active reason or None.

    The reason stays latched while idle recovers? No: it clears as soon as a
    window with enough idle is seen again.
    """
    parsed = parse_proc_stat_totals(stat_text, self.cores) if stat_text else None
    if parsed is None:
      return self._reason
    idle, total = parsed
    prev = self._prev
    self._prev = (idle, total)
    if prev is None:
      return self._reason
    d_idle = idle - prev[0]
    d_total = total - prev[1]
    if d_total <= 0.0:
      return self._reason
    fraction = d_idle / d_total
    if fraction < self.min_idle:
      if self._low_since is None:
        self._low_since = float(now)
      if now - self._low_since >= self.hold_s:
        self._reason = "cpu_contention"
    else:
      self._low_since = None
      self._reason = None
    return self._reason


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

def parse_statm_rss_bytes(text: str, page_size: int | None = None) -> int:
  """Resident set size from a /proc/self/statm line (field 2, in pages)."""
  fields = text.split()
  if len(fields) < 2:
    raise ValueError(f"short statm: {text!r}")
  return int(fields[1]) * int(page_size if page_size is not None else default_page_size())


def parse_meminfo_available_bytes(text: str) -> int | None:
  for line in text.splitlines():
    if line.startswith("MemAvailable:"):
      parts = line.split()
      if len(parts) >= 2:
        try:
          return int(parts[1]) * 1024
        except ValueError:
          return None
  return None


def read_rss_bytes(path: str = "/proc/self/statm", page_size: int | None = None) -> int | None:
  text = read_text(path)
  if text is None:
    return None
  try:
    return parse_statm_rss_bytes(text, page_size)
  except ValueError:
    return None


def read_mem_available_bytes(path: str = "/proc/meminfo") -> int | None:
  text = read_text(path)
  return parse_meminfo_available_bytes(text) if text is not None else None


@dataclass(frozen=True)
class MemoryDecision:
  """One memory-guard verdict: what tlvd should do right now."""
  suspend_reason: str | None = None
  run_gc: bool = False
  drop_caches: bool = False
  exit_reason: str | None = None


class MemoryGuard:
  """RSS soft/hard caps and the MemAvailable floor, with hysteresis.

  soft -> suspend rss_soft:<rss_mb>MB, gc + cache drop once per crossing;
  hard -> exit_reason rss_hard (caller publishes error, sets CarrotTlvDisabled,
  exits 0 so the manager does not restart-loop);
  MemAvailable < floor -> suspend mem_low:<avail_mb>MB while it stays below.
  The MB values are in the reason so the field log shows the actual footprint
  (W127: route 00000443 suspended at 246 MB with no size in the reason).
  """

  # Field route 00000443 (C3X, tinygrad CPU, r8s50a40 + r9 lamp): steady RSS 246 MB
  # after both models compiled; the old 220 MB soft cap suspended tlvd forever.
  # MemAvailable >= 600 MB (mem_available_min) remains the system floor.
  def __init__(self, *, soft_bytes: int = 360 * MB, hard_bytes: int = 440 * MB, mem_available_min: int = 600 * MB,
               soft_rearm_bytes: int | None = None, mem_rearm_bytes: int | None = None) -> None:
    self.soft_bytes = int(soft_bytes)
    self.hard_bytes = int(hard_bytes)
    self.mem_available_min = int(mem_available_min)
    self.soft_rearm_bytes = int(soft_rearm_bytes if soft_rearm_bytes is not None else self.soft_bytes - 10 * MB)
    self.mem_rearm_bytes = int(mem_rearm_bytes if mem_rearm_bytes is not None else self.mem_available_min + 50 * MB)
    self._soft_active = False
    self._mem_active = False

  def update(self, now: float, rss_bytes: int, mem_available_bytes: int | None) -> MemoryDecision:
    del now  # decisions are level-based; the signature keeps room for time-based tuning
    if rss_bytes >= self.hard_bytes:
      return MemoryDecision(exit_reason="rss_hard")
    suspend: str | None = None
    run_gc = False
    drop = False
    if rss_bytes >= self.soft_bytes:
      if not self._soft_active:
        self._soft_active = True
        run_gc = True
        drop = True
      suspend = f"rss_soft:{rss_bytes / MB:.0f}MB"
    elif rss_bytes < self.soft_rearm_bytes:
      self._soft_active = False
    elif self._soft_active:
      suspend = f"rss_soft:{rss_bytes / MB:.0f}MB"
    if mem_available_bytes is not None:
      if mem_available_bytes < self.mem_available_min:
        self._mem_active = True
      elif mem_available_bytes >= self.mem_rearm_bytes:
        self._mem_active = False
      if self._mem_active:
        suspend = suspend or f"mem_low:{mem_available_bytes / MB:.0f}MB"
    return MemoryDecision(suspend_reason=suspend, run_gc=run_gc, drop_caches=drop)
