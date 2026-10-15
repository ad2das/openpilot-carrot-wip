"""Unit tests for the tlvd self-throttle watchdog (injected clock, pure logic)."""
import inspect
import random

import pytest

from openpilot.selfdrive.carrot.tlv.watchdog import STATUS_BASELINE, STATUS_RUNNING, STATUS_SUSPENDED, SuspendWatchdog

DT = 0.05
BASE_FD = 0.2
BASE_EXEC = 0.030
BASE_LAG_MS = 20.0   # route 447 baseline p99 22.5 ms
LAG_JITTER_MS = 1.5


def feed(wd: SuspendWatchdog, t: float, seconds: float, *, frame_drop=None, exec_time=None, thermal=None) -> float:
  steps = int(round(seconds / DT))
  for _ in range(steps):
    t += DT
    wd.update(t, frame_drop=frame_drop, exec_time=exec_time, thermal=thermal)
  return t


def armed_watchdog() -> tuple[SuspendWatchdog, float]:
  wd = SuspendWatchdog(baseline_min_s=1.0, baseline_min_samples=40, clean_resume_s=30.0)
  t = 100.0
  t = feed(wd, t, 2.5, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  assert wd.inferring and wd.status == STATUS_RUNNING
  return wd, t


def test_baseline_arms_after_min_samples_and_span():
  wd = SuspendWatchdog(baseline_min_s=1.0, baseline_min_samples=40)
  t = feed(wd, 0.0, 1.5, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  assert not wd.inferring and wd.status == STATUS_BASELINE
  t = feed(wd, t, 1.2, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  assert wd.inferring and wd.status == STATUS_RUNNING


def test_frame_drop_suspends_and_30s_clean_resumes():
  wd, t = armed_watchdog()
  t = feed(wd, t, 4.0, frame_drop=1.5, exec_time=BASE_EXEC, thermal=0)
  assert not wd.inferring
  assert wd.status == STATUS_SUSPENDED
  assert wd.reason == "model_frame_drop"
  # The recent window still holds the bad samples; resume needs ~30 s clean.
  t = feed(wd, t, 25.0, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  assert not wd.inferring
  t = feed(wd, t, 12.0, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  assert wd.inferring and wd.status == STATUS_RUNNING


def test_exec_p95_suspend():
  wd, t = armed_watchdog()
  t = feed(wd, t, 4.0, frame_drop=BASE_FD, exec_time=BASE_EXEC * 1.35, thermal=0)
  assert not wd.inferring
  assert wd.reason == "model_exec_time"


def test_thermal_suspend_and_recovery():
  wd, t = armed_watchdog()
  wd.update(t + DT, thermal=1)
  assert not wd.inferring and wd.reason == "thermal(1)"
  t = feed(wd, t, 29.0, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  assert not wd.inferring
  t = feed(wd, t, 2.5, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  assert wd.inferring


def test_exponential_backoff_on_quick_resuspend():
  wd, t = armed_watchdog()
  required = []
  for _ in range(4):
    wd.update(t + DT, thermal=1)
    assert not wd.inferring
    required.append(wd.clean_required_s)
    # Resume: feed clean samples until the watchdog re-arms.
    for _ in range(int(300 / DT)):
      t += DT
      if wd.update(t, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0):
        break
    assert wd.inferring
  assert required == [30.0, 60.0, 120.0, 240.0]


def test_backoff_resets_after_long_run():
  wd, t = armed_watchdog()
  wd.update(t + DT, thermal=1)
  assert wd.clean_required_s == 30.0
  for _ in range(int(300 / DT)):
    t += DT
    if wd.update(t, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0):
      break
  assert wd.inferring
  # Run clean for longer than the re-arm window, then suspend again.
  t = feed(wd, t, 70.0, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0)
  wd.update(t + DT, thermal=1)
  assert wd.clean_required_s == 30.0


# -- lag trigger (route 447: per-message carState.cumLagMs, not tlvd gaps) ------

def armed_with_lag(base_lag_ms: float = BASE_LAG_MS):
  """Arm the watchdog from a baseline of intrinsic per-message lag samples."""
  wd = SuspendWatchdog(baseline_min_s=1.0, baseline_min_samples=40, clean_resume_s=30.0)
  t = 100.0
  rng = random.Random(7)
  for _ in range(int(2.5 / DT)):
    t += DT
    wd.update(t, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0,
              controls_lag_ms=base_lag_ms + rng.uniform(-LAG_JITTER_MS, LAG_JITTER_MS))
  assert wd.inferring
  return wd, t, rng


def feed_lag(wd, t, lag_ms, seconds, rng=None):
  steps = int(round(seconds / DT))
  for _ in range(steps):
    t += DT
    value = lag_ms + (rng.uniform(-LAG_JITTER_MS, LAG_JITTER_MS) if rng is not None else 0.0)
    wd.update(t, controls_lag_ms=value)
  return t


def stress_until_suspended(wd, t, rng, lag_ms=30.0, max_seconds=10.0):
  """Feed stressed lag until the watchdog suspends, then stop (a real system is
  clean once tlvd stops inferring, so no further stress pollutes the baseline)."""
  for _ in range(int(max_seconds / DT)):
    t += DT
    wd.update(t, controls_lag_ms=lag_ms + rng.uniform(-LAG_JITTER_MS, LAG_JITTER_MS))
    if not wd.inferring:
      return t
  return t


def resume_until_armed(wd, t, rng, base_lag_ms=BASE_LAG_MS):
  for _ in range(int((wd.clean_required_s + 3.0) / DT)):
    t += DT
    wd.update(t, controls_lag_ms=base_lag_ms + rng.uniform(-LAG_JITTER_MS, LAG_JITTER_MS))
    if wd.inferring:
      return t
  return t


def test_lag_regression_own_sampling_gaps_do_not_suspend():
  # The watchdog must never suspend on tlvd's own blocking: feed 30 s of baseline
  # lag, then keep the same lag distribution while calling update only every
  # 0.7 s (tlvd's detect block). The old inter-arrival metric suspended here.
  wd = SuspendWatchdog(baseline_min_s=1.0, baseline_min_samples=40, clean_resume_s=30.0)
  t = 100.0
  rng = random.Random(6)
  for _ in range(int(30.0 / DT)):
    t += DT
    wd.update(t, controls_lag_ms=BASE_LAG_MS + rng.uniform(-LAG_JITTER_MS, LAG_JITTER_MS))
  assert wd.inferring
  for _ in range(int(30.0 / 0.7)):
    t += 0.7
    wd.update(t, controls_lag_ms=BASE_LAG_MS + rng.uniform(-LAG_JITTER_MS, LAG_JITTER_MS))
  assert wd.inferring and wd.status == STATUS_RUNNING


def test_lag_p99_over_slack_suspends_with_controls_jitter():
  wd, t, rng = armed_with_lag()
  assert wd.inferring
  # live p99 ~33 ms vs baseline p99 ~21.5 ms: > baseline + 4 ms slack.
  t = feed_lag(wd, t, 30.0, 5.0, rng=rng)
  assert not wd.inferring
  assert wd.reason == "controls_jitter"


def test_lag_steady_does_not_suspend():
  wd, t, rng = armed_with_lag()
  t = feed_lag(wd, t, BASE_LAG_MS, 8.0, rng=rng)
  assert wd.inferring


def test_controls_jitter_resumes_after_clean_seconds():
  # Addendum: a genuine controls_jitter suspension must still resume. The old
  # _suspend kept the live stress samples and every resume was blocked forever.
  wd, t, rng = armed_with_lag()
  t = stress_until_suspended(wd, t, rng)
  assert not wd.inferring and wd.reason == "controls_jitter"
  assert wd.clean_required_s == 30.0
  # No further stress: clean baseline lag must resume it after clean_resume_s.
  t = resume_until_armed(wd, t, rng)
  assert wd.inferring and wd.status == STATUS_RUNNING


def test_quick_controls_jitter_resuspend_doubles_clean_requirement():
  wd, t, rng = armed_with_lag()
  required = []
  for _ in range(4):
    t = stress_until_suspended(wd, t, rng)
    assert not wd.inferring and wd.reason == "controls_jitter"
    required.append(wd.clean_required_s)
    t = resume_until_armed(wd, t, rng)
    assert wd.inferring
  assert required == [30.0, 60.0, 120.0, 240.0]


def test_camera_gap_trigger_is_gone():
  init_params = inspect.signature(SuspendWatchdog.__init__).parameters
  assert not any("camera" in name for name in init_params)
  assert "controls_lag_slack_ms" in init_params
  update_params = inspect.signature(SuspendWatchdog.update).parameters
  assert "camera_frame_id" not in update_params
  assert "controls_interval" not in update_params
  assert "controls_lag_ms" in update_params
  # No camera-gap attribute or old slack arg survives anywhere on the class.
  assert not any("camera" in name for name in dir(SuspendWatchdog))
  with pytest.raises(TypeError):
    SuspendWatchdog(camera_gap_slack_per_s=0.5)
