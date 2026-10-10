"""Unit tests for the tlvd resource governors (fake clocks, recorded /proc text)."""
import pytest

from openpilot.selfdrive.carrot.tlv import resources


class FakeClocks:
  def __init__(self) -> None:
    self.t = 0.0
    self.cpu = 0.0

  def clock(self) -> float:
    return self.t

  def cpu_clock(self) -> float:
    return self.cpu


def _governor(**kwargs):
  clocks = FakeClocks()
  kwargs.setdefault("clock", clocks.clock)
  kwargs.setdefault("cpu_clock", clocks.cpu_clock)
  return resources.DutyCycleGovernor(**kwargs), clocks


def _run_frame(gov, clocks, cost: float, work: float = 0.05, step: float = 0.005) -> None:
  while not gov.ready(clocks.t):
    clocks.t += step
  gov.begin_frame()
  clocks.cpu += cost
  clocks.t += work
  gov.end_frame()


# ---------------------------------------------------------------------------
# DutyCycleGovernor
# ---------------------------------------------------------------------------

def test_governor_starts_at_5hz_and_stays_in_bounds():
  gov, clocks = _governor()
  assert gov.rate_hz == pytest.approx(5.0)
  assert gov.period_s == pytest.approx(0.2)
  _run_frame(gov, clocks, 0.001)  # very cheap -> max 10 Hz
  assert gov.rate_hz == pytest.approx(10.0)
  for _ in range(10):
    _run_frame(gov, clocks, 0.5)  # very expensive -> floor 1 Hz
  assert gov.rate_hz == pytest.approx(1.0)


def test_governor_rate_from_frame_cost():
  gov, clocks = _governor()
  _run_frame(gov, clocks, 0.06)
  # period = worst frame cost / budget -> 0.06 / 0.4 = 0.15 s (6.67 Hz)
  assert gov.rate_hz == pytest.approx(0.4 / 0.06)
  assert gov.period_s == pytest.approx(0.06 / 0.4)


def test_governor_paces_next_frame():
  gov, clocks = _governor()
  _run_frame(gov, clocks, 0.06)
  assert not gov.ready(clocks.t)
  assert not gov.ready(clocks.t + 0.5 * gov.period_s)
  assert gov.ready(clocks.t + gov.period_s)


def test_governor_window_duty_and_feedback():
  gov, clocks = _governor()
  for _ in range(6):  # 0.05 CPU per 0.1 s wall -> 50% duty over the 0.5 s span
    gov.begin_frame()
    clocks.cpu += 0.05
    clocks.t += 0.1
    gov.end_frame()
  assert gov.window_duty == pytest.approx(0.5)
  # The per-frame target would be 8 Hz (0.4/0.05); the measured over-budget duty
  # must pull the rate down instead.
  assert gov.rate_hz < 8.0


def test_governor_window_duty_needs_span():
  gov, clocks = _governor()
  gov.begin_frame()
  clocks.cpu += 0.05
  clocks.t += 0.1
  gov.end_frame()
  assert gov.window_duty is None


def test_governor_simulation_meets_40_percent_cap():
  gov, clocks = _governor()
  sim_end = 60.0
  while clocks.t < sim_end:
    if gov.ready(clocks.t):
      _run_frame(gov, clocks, 0.06)
    else:
      clocks.t += 0.005
  duty = clocks.cpu / clocks.t
  assert duty <= 0.40 + 0.01, f"simulated duty {duty:.3f} over budget"
  assert duty > 0.10, "governor should keep making progress"


def test_governor_end_frame_without_begin_is_noop():
  gov, clocks = _governor()
  before = gov.rate_hz
  gov.end_frame()
  assert gov.rate_hz == before


# ---------------------------------------------------------------------------
# /proc/stat parsing + cores guard
# ---------------------------------------------------------------------------

# Recorded /proc/stat snapshot (C3X-class 8-core device; real field layout:
# user nice system idle iowait irq softirq steal guest guest_nice).
RECORDED_STAT = "\n".join([
  "cpu  120481 250 89321 1903445 8123 0 4501 0 0 0",
  "cpu0 15500 40 11000 240000 1200 0 600 0 0 0",
  "cpu1 15100 30 11200 238000 1000 0 550 0 0 0",
  "cpu2 14900 35 11050 239500 900 0 560 0 0 0",
  "cpu3 15300 30 11100 238900 950 0 540 0 0 0",
  "cpu4 14800 35 11250 238500 1300 0 560 0 0 0",
  "cpu5 14950 30 11150 238700 900 0 580 0 0 0",
  "cpu6 14800 25 11250 239200 950 0 540 0 0 0",
  "cpu7 15131 25 11321 240645 923 0 571 0 0 0",
  "intr 1234567 0 0 0",
  "ctxt 987654",
  "btime 1700000000",
  "processes 12345",
  "procs_running 2",
  "procs_blocked 0",
  "softirq 12345 0 0 0 0 0 0 0 0 0 0",
]) + "\n"


def _stat_text(core_totals_idle: dict) -> str:
  """Snapshot text: {core: (total_jiffies, idle_jiffies)}; idle includes iowait."""
  lines = ["cpu 0 0 0 0 0 0 0 0 0 0"]
  for i in range(8):
    total, idle = core_totals_idle.get(i, (0, 0))
    busy = total - idle
    lines.append(f"cpu{i} {busy} 0 0 {idle} 0 0 0 0 0 0")
  lines.append("intr 0")
  return "\n".join(lines) + "\n"


def test_parse_proc_stat_totals_recorded_sample():
  parsed = resources.parse_proc_stat_totals(RECORDED_STAT, (1, 2, 3))
  assert parsed is not None
  idle, total = parsed
  # cpu1..3 idle+iowait = (238000+1000)+(239500+900)+(238900+950)
  assert idle == pytest.approx(719250)
  # sum of the first 8 fields of cpu1..3 (guest fields are not counted twice)
  assert total == pytest.approx(799645)


def test_parse_proc_stat_totals_requires_all_cores():
  partial = RECORDED_STAT.replace("cpu3 15300 30 11100 238900 950 0 540 0 0 0\n", "")
  assert resources.parse_proc_stat_totals(partial, (1, 2, 3)) is None
  assert resources.parse_proc_stat_totals("garbage", (1, 2, 3)) is None


def test_cpu_guard_suspends_after_3s_of_low_idle():
  guard = resources.CpuContentionGuard(cores=(1, 2, 3), min_idle=0.25, hold_s=3.0)
  idle_total = {(1): (1000, 900), 2: (1000, 900), 3: (1000, 900)}
  t = 0.0
  assert guard.update(t, _stat_text(idle_total)) is None  # first sample: baseline only
  for _ in range(5):  # 360/400 busy -> 10% idle
    t += 1.0
    idle_total = {c: (idle_total[c][0] + 400, idle_total[c][1] + 40) for c in idle_total}
    reason = guard.update(t, _stat_text(idle_total))
  assert reason == "cpu_contention"
  assert guard.reason == "cpu_contention"


def test_cpu_guard_needs_continuous_3s():
  guard = resources.CpuContentionGuard(cores=(1, 2, 3), min_idle=0.25, hold_s=3.0)
  state = {1: (1000, 900), 2: (1000, 900), 3: (1000, 900)}
  guard.update(0.0, _stat_text(state))
  # low, low, recover, low, low -> never 3 s in a row
  for step, idle_delta in enumerate([40, 40, 380, 40, 40]):
    state = {c: (state[c][0] + 400, state[c][1] + idle_delta) for c in state}
    assert guard.update(1.0 + step, _stat_text(state)) is None


def test_cpu_guard_recovers_with_idle():
  guard = resources.CpuContentionGuard(cores=(1, 2, 3), min_idle=0.25, hold_s=3.0)
  state = {1: (1000, 900), 2: (1000, 900), 3: (1000, 900)}
  guard.update(0.0, _stat_text(state))
  for step in range(4):  # 10% idle from t=1 on -> suspended once 3 s are continuous
    state = {c: (state[c][0] + 400, state[c][1] + 40) for c in state}
    guard.update(1.0 + step, _stat_text(state))
  assert guard.reason == "cpu_contention"
  state = {c: (state[c][0] + 400, state[c][1] + 380) for c in state}
  assert guard.update(6.0, _stat_text(state)) is None


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------

RECORDED_MEMINFO = "\n".join([
  "MemTotal:        7908000 kB",
  "MemFree:         1234567 kB",
  "MemAvailable:    3456789 kB",
  "Buffers:          123456 kB",
  "Cached:          2345678 kB",
]) + "\n"


def test_parse_statm_and_meminfo():
  assert resources.parse_statm_rss_bytes("12345 6789 100 1 0 200 0\n", 4096) == 6789 * 4096
  with pytest.raises(ValueError):
    resources.parse_statm_rss_bytes("12345")
  assert resources.parse_meminfo_available_bytes(RECORDED_MEMINFO) == 3456789 * 1024
  assert resources.parse_meminfo_available_bytes("MemTotal: 1 kB\n") is None


MB = 1024 * 1024


def test_memory_guard_soft_limit_gc_once():
  guard = resources.MemoryGuard(soft_bytes=220 * MB, hard_bytes=300 * MB, mem_available_min=600 * MB)
  decision = guard.update(0.0, 100 * MB, 4000 * MB)
  assert decision == resources.MemoryDecision()
  decision = guard.update(1.0, 225 * MB, 4000 * MB)
  assert decision.suspend_reason == "rss_soft"
  assert decision.run_gc and decision.drop_caches
  decision = guard.update(2.0, 228 * MB, 4000 * MB)
  assert decision.suspend_reason == "rss_soft"
  assert not decision.run_gc  # gc once per crossing
  # must fall below the re-arm level before the next crossing counts
  assert guard.update(3.0, 205 * MB, 4000 * MB).suspend_reason is None
  decision = guard.update(4.0, 221 * MB, 4000 * MB)
  assert decision.run_gc and decision.drop_caches


def test_memory_guard_hard_limit_exits():
  guard = resources.MemoryGuard()
  decision = guard.update(0.0, 305 * MB, 4000 * MB)
  assert decision.exit_reason == "rss_hard"
  assert decision.suspend_reason is None


def test_memory_guard_mem_low_hysteresis():
  guard = resources.MemoryGuard()
  assert guard.update(0.0, 100 * MB, 550 * MB).suspend_reason == "mem_low"
  assert guard.update(1.0, 100 * MB, 620 * MB).suspend_reason == "mem_low"  # below re-arm 650
  assert guard.update(2.0, 100 * MB, 660 * MB).suspend_reason is None
  assert guard.update(3.0, 100 * MB, 500 * MB).suspend_reason == "mem_low"


def test_memory_guard_soft_beats_mem_low():
  guard = resources.MemoryGuard()
  decision = guard.update(0.0, 230 * MB, 500 * MB)
  assert decision.suspend_reason == "rss_soft"


def test_write_oom_score_adj(tmp_path):
  target = tmp_path / "oom_score_adj"
  assert resources.write_oom_score_adj(1000, str(target))
  assert target.read_text() == "1000"
  assert not resources.write_oom_score_adj(1000, str(tmp_path / "missing" / "oom_score_adj"))
