"""Unit tests for the tlvd self-throttle watchdog (injected clock, pure logic)."""
from openpilot.selfdrive.carrot.tlv.watchdog import STATUS_BASELINE, STATUS_RUNNING, STATUS_SUSPENDED, SuspendWatchdog

DT = 0.05
BASE_FD = 0.2
BASE_EXEC = 0.030


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


def armed_with_controls(controls_interval: float = 0.010, camera_step: int = 2, ticks: int = 90):
  wd = SuspendWatchdog(baseline_min_s=0.5, baseline_min_samples=50, clean_resume_s=30.0,
                       camera_min_span_s=1.0, camera_window_s=2.0)
  t = 100.0
  fid = 0
  for _ in range(ticks):
    t += DT
    fid += camera_step
    wd.update(t, frame_drop=BASE_FD, exec_time=BASE_EXEC, thermal=0,
              controls_interval=controls_interval, camera_frame_id=fid)
  return wd, t, fid


def feed_live(wd, t, fid, ticks, *, controls_interval=0.010, camera_step=2):
  for _ in range(ticks):
    t += DT
    fid += camera_step
    wd.update(t, controls_interval=controls_interval, camera_frame_id=fid)
  return t, fid


def test_controls_jitter_suspends():
  wd, t, fid = armed_with_controls(controls_interval=0.010)
  assert wd.inferring
  t, fid = feed_live(wd, t, fid, 40, controls_interval=0.013)
  assert not wd.inferring
  assert wd.reason == "controls_jitter"


def test_controls_steady_interval_does_not_suspend():
  wd, t, fid = armed_with_controls(controls_interval=0.010)
  t, fid = feed_live(wd, t, fid, 60, controls_interval=0.010)
  assert wd.inferring


def test_camera_gap_rate_suspends():
  wd, t, fid = armed_with_controls(camera_step=2)
  assert wd.inferring
  t, fid = feed_live(wd, t, fid, 60, camera_step=2)
  assert wd.inferring  # steady stream (same as baseline) must not suspend
  # now add real skips: every 5th observation skips an extra frameId -> 24/s vs 20/s
  for i in range(60):
    t += DT
    fid += 3 if i % 5 == 0 else 2
    wd.update(t, camera_frame_id=fid)
  assert not wd.inferring
  assert wd.reason == "camera_gaps"


def test_camera_steady_stream_does_not_suspend():
  wd, t, fid = armed_with_controls(camera_step=2)
  t, fid = feed_live(wd, t, fid, 80, camera_step=2)
  assert wd.inferring
