"""Tests for the onroad-while-parked DSP probe (openpilot/selfdrive/carrot/tlv/dsp_probe.py).

No real DSP exists on the PC: every step is exercised through its injectable
dependencies (path/runner/loader/adapter fakes), subprocess use is mocked and
the parked gate, lost-step latch and UI notice sequence run on a fake clock and
a minimal fake SubMaster, so no real time passes.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import dsp_probe


def _load_json(path):
  with open(path, encoding="utf-8") as f:
    return json.load(f)


class FakeParams:
  def __init__(self, values) -> None:
    self._values = dict(values)

  def get_int(self, key):
    value = self._values.get(key)
    return None if value is None else int(value)

  def get(self, key):
    return self._values.get(key)


@pytest.fixture
def probe_root(tmp_path, monkeypatch):
  root = tmp_path / "dsp_probe"
  monkeypatch.setattr(dsp_probe, "PROBE_ROOT", str(root))
  monkeypatch.setattr(dsp_probe, "current_commit_short", lambda: "abc1234")
  return root


# ---------------------------------------------------------------------------
# fake clock / SubMaster / subprocess infrastructure
# ---------------------------------------------------------------------------

class FakeClock:
  def __init__(self, start: float = 0.0) -> None:
    self.now = float(start)

  def __call__(self) -> float:
    return self.now

  def advance(self, dt: float) -> None:
    self.now += dt


class FakeSM:
  """Minimal SubMaster: carState + selfdriveState with optional staleness."""

  def __init__(self, clock, *, gear="park", standstill=True, vego=0.0,
               enabled=False, fresh=True) -> None:
    self.clock = clock
    self.gear = gear
    self.standstill = standstill
    self.vego = vego
    self.enabled = enabled
    self.fresh = fresh
    self.updates = 0
    base = clock() if fresh else clock() - 10.0
    self.seen = dict.fromkeys(dsp_probe.PARK_SERVICES, True)
    self.recv_time = dict.fromkeys(dsp_probe.PARK_SERVICES, base)

  def update(self, timeout=0) -> None:
    self.updates += 1
    if self.fresh:
      now = self.clock()
      for service in dsp_probe.PARK_SERVICES:
        self.recv_time[service] = now

  def __getitem__(self, key):
    if key == "carState":
      return SimpleNamespace(gearShifter=self.gear, standstill=self.standstill, vEgo=self.vego)
    if key == dsp_probe.SD_SERVICE:
      return SimpleNamespace(enabled=self.enabled)
    raise KeyError(key)


class NoticeLog:
  """Captures every notice publish; None entries are key removals."""

  def __init__(self) -> None:
    self.items: list = []

  def __call__(self, payload) -> None:
    self.items.append(payload)

  @property
  def states(self):
    return [item.get("state") if isinstance(item, dict) else None for item in self.items]

  def of_state(self, state):
    return [item for item in self.items if isinstance(item, dict) and item.get("state") == state]


class FakeProc:
  def __init__(self, poll_fn=None, pid=4242):
    self._poll_fn = poll_fn or (lambda: None)
    self.pid = pid
    self.killed = False
    self.returncode = None

  def poll(self):
    if self.killed:
      return self.returncode
    code = self._poll_fn()
    if code is not None:
      self.returncode = code
    return code

  def kill(self):
    self.killed = True
    self.returncode = -9

  def wait(self, timeout=None):
    return self.returncode


class PowerCut(BaseException):
  """Abrupt ignition-off: no signal handler runs and nothing is cleaned up."""


def _power_cut(*args, **kwargs):
  raise PowerCut()


def _advancing_sleep(clock):
  def sleep(dt):
    clock.advance(dt)
  return sleep


def _run_paths(tmp_path, name="abc1234"):
  result = str(tmp_path / f"{name}.json")
  return result, dsp_probe.attempt_path(result), dsp_probe.progress_path(result)


# ---------------------------------------------------------------------------
# enable condition / result path
# ---------------------------------------------------------------------------

def test_should_run_truth_table(probe_root):
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is True  # onroad; PARKED is gated by run_probe
  assert dsp_probe.should_run(False, params) is False  # manager not started
  assert dsp_probe.should_run(True, FakeParams({"TrafficLightDetectMode": 0})) is False
  assert dsp_probe.should_run(True, FakeParams({})) is False  # missing -> off

  probe_root.mkdir(parents=True, exist_ok=True)
  (probe_root / "abc1234.json").write_text("{}", encoding="utf-8")
  assert dsp_probe.should_run(True, params) is False  # latched for this commit


def test_should_run_false_when_latched(probe_root, monkeypatch):
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", True)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is False


def _install_fake_signal_color_module(monkeypatch, requested):
  module = types.ModuleType("openpilot.selfdrive.modeld.signal_color_shadow")
  module.requested = lambda: requested
  monkeypatch.setitem(sys.modules, "openpilot.selfdrive.modeld.signal_color_shadow", module)


def test_should_run_disabled_when_signal_color_requested(probe_root, monkeypatch):
  _install_fake_signal_color_module(monkeypatch, True)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is False


def test_should_run_unchanged_when_signal_color_not_requested(probe_root, monkeypatch):
  _install_fake_signal_color_module(monkeypatch, False)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is True


def test_result_and_latch_paths_use_short_commit(probe_root):
  result = dsp_probe.probe_result_path()
  assert result == os.path.join(str(probe_root), "abc1234.json")
  assert dsp_probe.attempt_path(result) == os.path.join(str(probe_root), "abc1234.attempt")
  assert dsp_probe.progress_path(result) == os.path.join(str(probe_root), "abc1234.progress")


def test_main_skips_when_result_exists(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  (probe_root / "abc1234.json").write_text("{}", encoding="utf-8")
  called = []
  monkeypatch.setattr(dsp_probe, "run_probe", lambda *a, **k: called.append(1))
  dsp_probe.main()
  assert not called


def test_current_commit_short_prefers_env(monkeypatch):
  monkeypatch.setenv("GIT_COMMIT", "0123456789abcdef")
  assert dsp_probe.current_commit_short() == "0123456"


# ---------------------------------------------------------------------------
# atomic write
# ---------------------------------------------------------------------------

def test_write_result_atomic(tmp_path):
  path = tmp_path / "r.json"
  dsp_probe.write_result_atomic(str(path), {"ok": True, "n": 1})
  assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True, "n": 1}
  assert not (tmp_path / "r.json.tmp").exists()


def test_write_result_atomic_cleans_tmp_on_failure(tmp_path):
  path = tmp_path / "r.json"
  path.write_text('{"old": true}', encoding="utf-8")
  with pytest.raises(TypeError):
    dsp_probe.write_result_atomic(str(path), {"bad": object()})
  assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
  assert not (tmp_path / "r.json.tmp").exists()


# ---------------------------------------------------------------------------
# step a: presence + clang
# ---------------------------------------------------------------------------

def _ok_run(returncode=0, stderr=b""):
  return lambda cmd, capture_output=True, timeout=None: SimpleNamespace(returncode=returncode, stderr=stderr)


def test_step_presence_all_ok(tmp_path):
  calls = []

  def fake_run(cmd, capture_output=True, timeout=None):
    calls.append(cmd)
    return SimpleNamespace(returncode=0, stderr=b"")

  result = dsp_probe.step_presence(path_status=lambda p: "ok", which=lambda n: "/usr/bin/clang",
                                   run=fake_run, workdir=str(tmp_path))
  assert result["status"] == "pass"
  assert all(value == "ok" for value in result["paths"].values())
  assert result["clang"]["status"] == "pass"
  assert "--target=hexagon" in calls[0]
  assert "-mhvx-length=128b" in calls[0]


def test_step_presence_missing_clang(tmp_path):
  result = dsp_probe.step_presence(path_status=lambda p: "ok", which=lambda n: None, workdir=str(tmp_path))
  assert result["status"] == "fail"
  assert result["clang"]["status"] == "fail"
  assert "not found" in result["clang"]["error"]


def test_step_presence_clang_failure_truncates_stderr(tmp_path):
  result = dsp_probe.step_presence(path_status=lambda p: "ok", which=lambda n: "/clang",
                                   run=_ok_run(returncode=1, stderr=b"e" * 5000), workdir=str(tmp_path))
  assert result["status"] == "fail"
  assert result["clang"]["returncode"] == 1
  assert len(result["clang"]["error"]) <= 2048


def test_step_presence_unreadable_path(tmp_path):
  def path_status(path):
    return "ok" if path != "/dev/ion" else "not readable"

  result = dsp_probe.step_presence(path_status=path_status, which=lambda n: None, workdir=str(tmp_path))
  assert result["status"] == "fail"
  assert result["paths"]["/dev/ion"] == "not readable"


def test_step_presence_clang_timeout(tmp_path):
  def fake_run(cmd, capture_output=True, timeout=None):
    raise subprocess.TimeoutExpired(cmd, timeout or 1.0)

  result = dsp_probe.step_presence(path_status=lambda p: "ok", which=lambda n: "/clang",
                                   run=fake_run, workdir=str(tmp_path))
  assert result["status"] == "fail"
  assert "timed out" in result["clang"]["error"]


# ---------------------------------------------------------------------------
# step b: tinygrad DSP smoke
# ---------------------------------------------------------------------------

class _NumpyTensor:
  def __init__(self, data, dtype=None, device=None):
    arr = np.asarray(data)
    self._arr = arr.astype(dtype) if dtype is not None else arr

  def __add__(self, other):
    return _NumpyTensor(self._arr + other._arr)

  def matmul(self, other):
    return _NumpyTensor(self._arr @ other._arr)

  def numpy(self):
    return self._arr


class _FakeDtypes:
  uint8 = np.uint8


def _fake_loader(default="DSP", opened=()):
  return lambda: (_NumpyTensor, _FakeDtypes, default, lambda: list(opened))


def test_step_smoke_pass(tmp_path):
  result = dsp_probe.step_smoke(load_tensors=_fake_loader(), runs=10)
  assert result["status"] == "pass"
  assert result["runs"] == 10
  assert result["first_ms"] >= 0.0
  assert result["steady_median_ms"] >= 0.0


def test_step_smoke_load_failure():
  def loader():
    raise ImportError("no tinygrad")

  result = dsp_probe.step_smoke(load_tensors=loader)
  assert result["status"] == "fail"
  assert "no tinygrad" in result["error"]


def test_step_smoke_wrong_default_device():
  result = dsp_probe.step_smoke(load_tensors=_fake_loader(default="QCOM"))
  assert result["status"] == "fail"
  assert "QCOM" in result["error"]


def test_step_smoke_rejects_opened_accelerators():
  result = dsp_probe.step_smoke(load_tensors=_fake_loader(opened=("DSP", "QCOM:0")))
  assert result["status"] == "fail"
  assert "QCOM" in result["error"]


def test_smoke_ops_detects_wrong_values():
  class BadTensor(_NumpyTensor):
    def __add__(self, other):
      return _NumpyTensor(np.zeros_like(self._arr))

  ok, message = dsp_probe._smoke_ops(BadTensor, _FakeDtypes, "DSP")
  assert ok is False
  assert "mismatch" in message


# ---------------------------------------------------------------------------
# step c: detector on DEV=DSP with a CPU reference
# ---------------------------------------------------------------------------

class _FakeAdapter:
  def __init__(self, model_path, device, *, outputs=None, opened=()):
    self.device = device
    self._outputs = outputs if outputs is not None else [np.zeros((1, 4), dtype=np.float32)]
    self._opened = opened

  def run(self, model_input):
    return [np.asarray(value, dtype=np.float32) for value in self._outputs]

  def opened_devices(self):
    return list(self._opened)


def _fake_adapter_factory(outputs_by_device=None, opened=("DSP", "CPU")):
  def factory(model_path, device):
    outputs = None if outputs_by_device is None else outputs_by_device.get(device)
    return _FakeAdapter(model_path, device, outputs=outputs, opened=opened)
  return factory


def test_step_detector_no_model_path():
  result = dsp_probe.step_detector(path_getter=lambda: "")
  assert result["status"] == "fail"
  assert "not set" in result["error"]


def test_step_detector_model_not_found(tmp_path):
  result = dsp_probe.step_detector(model_path=str(tmp_path / "nope.onnx"))
  assert result["status"] == "fail"
  assert "not found" in result["error"]


def test_step_detector_pass(tmp_path):
  model = tmp_path / "m.onnx"
  model.write_bytes(b"x")
  result = dsp_probe.step_detector(model_path=str(model), adapter_factory=_fake_adapter_factory(), runs=20)
  assert result["status"] == "pass"
  assert result["runs"] == 20
  assert result["max_abs_diff"] == 0.0
  assert result["compile_s"] >= 0.0
  assert result["steady_median_ms"] >= 0.0
  assert result["steady_p90_ms"] >= 0.0


def test_step_detector_reports_cpu_diff(tmp_path):
  model = tmp_path / "m.onnx"
  model.write_bytes(b"x")
  outputs = {
    "DSP": [np.array([[1.0, 2.0]], dtype=np.float32)],
    "CPU": [np.array([[1.0, 2.5]], dtype=np.float32)],
  }
  result = dsp_probe.step_detector(model_path=str(model),
                                   adapter_factory=_fake_adapter_factory(outputs), runs=20)
  assert result["status"] == "pass"
  assert result["max_abs_diff"] == pytest.approx(0.5)


def test_step_detector_rejects_qcom(tmp_path):
  model = tmp_path / "m.onnx"
  model.write_bytes(b"x")
  result = dsp_probe.step_detector(model_path=str(model),
                                   adapter_factory=_fake_adapter_factory(opened=("DSP", "QCOM:0")))
  assert result["status"] == "fail"
  assert "QCOM" in result["error"]


def test_step_detector_adapter_exception(tmp_path):
  model = tmp_path / "m.onnx"
  model.write_bytes(b"x")

  def factory(_path, _device):
    raise RuntimeError("boom")

  result = dsp_probe.step_detector(model_path=str(model), adapter_factory=factory)
  assert result["status"] == "fail"
  assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# per-step timeout + orchestration
# ---------------------------------------------------------------------------

def test_timed_timeout():
  result = dsp_probe._timed("slow", lambda: time.sleep(5.0), 0.05)
  assert result["status"] == "timeout"
  assert result["name"] == "slow"
  assert "exceeded" in result["error"]


def test_timed_captures_failure():
  def boom():
    raise ValueError("nope")

  result = dsp_probe._timed("boom", boom, 1.0)
  assert result["status"] == "fail"
  assert "ValueError" in result["error"]
  assert result["duration_s"] >= 0.0


def test_collect_steps_skips_dependents_after_failure():
  calls = []

  def timed(name, fn, timeout):
    calls.append(name)
    return {"name": name, "status": "fail" if name == "presence" else "pass"}

  steps = dsp_probe.collect_steps(timed=timed)
  assert [step["status"] for step in steps.values()] == ["fail", "skipped", "skipped"]
  assert calls == ["presence"]


def test_collect_steps_runs_all_when_passing():
  order = []

  def timed(name, fn, timeout):
    order.append(name)
    return {"name": name, "status": "pass"}

  steps = dsp_probe.collect_steps(timed=timed)
  assert order == ["presence", "smoke", "detector"]
  assert all(step["status"] == "pass" for step in steps.values())


def test_collect_steps_writes_progress(tmp_path):
  progress = str(tmp_path / "c.progress")
  seen = []
  order = []

  def timed(name, fn, timeout):
    order.append(name)
    seen.append(dsp_probe.read_progress_step(progress))
    return {"name": name, "status": "pass"}

  dsp_probe.collect_steps(timed=timed, progress_file=progress)
  assert order == ["presence", "smoke", "detector"]
  assert seen == ["presence", "smoke", "detector"]


def test_build_result_payload():
  steps = {"presence": {"status": "pass"}, "smoke": {"status": "fail"}, "detector": {"status": "skipped"}}
  payload = dsp_probe.build_result_payload(steps, commit="abc", device_type="tici",
                                           agnos_version="19.8", timestamp="t0")
  assert payload["ok"] is False
  assert payload["commit"] == "abc"
  assert payload["device_type"] == "tici"
  assert payload["agnos_version"] == "19.8"
  payload = dsp_probe.build_result_payload({"a": {"status": "pass"}}, commit="c", device_type=None,
                                           agnos_version=None, timestamp="t", error="x" * 5000)
  assert payload["ok"] is True
  assert len(payload["error"]) <= 2048


def test_parse_argv():
  assert dsp_probe._parse_argv(["--child", "--result", "x.json"]) == {"child": True, "result": "x.json"}
  assert dsp_probe._parse_argv([]) == {"child": False, "result": None}
  assert dsp_probe._parse_argv(["--result"]) == {"child": False, "result": None}


def test_child_main_writes_result(tmp_path, monkeypatch):
  monkeypatch.setenv("GIT_COMMIT", "abc1234deadbeef")
  monkeypatch.setattr(dsp_probe, "collect_steps",
                      lambda **kwargs: {"presence": {"name": "presence", "status": "pass"}})
  path = tmp_path / "c.json"
  dsp_probe._child_main(str(path))
  payload = json.loads(path.read_text(encoding="utf-8"))
  assert payload["ok"] is True
  assert payload["commit"] == "abc1234"
  assert payload["steps"]["presence"]["status"] == "pass"


# ---------------------------------------------------------------------------
# parked gate
# ---------------------------------------------------------------------------

def test_parked_requires_all_conditions():
  clock = FakeClock(100.0)
  now = clock()
  assert dsp_probe._parked(FakeSM(clock), now) is True
  for kwargs in ({"gear": "drive"}, {"standstill": False}, {"vego": 0.5}, {"enabled": True}):
    assert dsp_probe._parked(FakeSM(clock, **kwargs), now) is False


def test_parked_requires_fresh_messages():
  clock = FakeClock(100.0)
  stale = FakeSM(clock, fresh=False)
  assert dsp_probe._parked(stale, clock()) is False
  fresh = FakeSM(clock)
  assert dsp_probe._parked(fresh, clock()) is True
  clock.advance(2.0)  # no update(): messages older than PARK_FRESH_S
  assert dsp_probe._parked(fresh, clock()) is False
  fresh.update(0)
  assert dsp_probe._parked(fresh, clock()) is True


def test_wait_for_parked_holds_full_five_seconds():
  clock = FakeClock()
  sm = FakeSM(clock)
  sleeps = []

  def sleep(dt):
    sleeps.append(dt)
    clock.advance(dt)

  dsp_probe.wait_for_parked(sm, clock=clock, sleep=sleep)
  assert clock() == pytest.approx(5.0)
  assert len(sleeps) == 10


def test_wait_for_parked_resets_after_interruption():
  clock = FakeClock()
  sm = FakeSM(clock)

  def sleep(dt):
    clock.advance(dt)
    if clock() == pytest.approx(2.0):
      sm.standstill = False
    if clock() == pytest.approx(3.0):
      sm.standstill = True

  dsp_probe.wait_for_parked(sm, clock=clock, sleep=sleep)
  assert clock() == pytest.approx(8.0)


def test_wait_for_parked_arming_notice_and_deadline():
  clock = FakeClock()
  sm = FakeSM(clock)
  logs = NoticeLog()

  def sleep(dt):
    clock.advance(dt)

  dsp_probe.wait_for_parked(sm, clock=clock, sleep=sleep, notice=logs)
  arming = logs.of_state("arming")
  assert arming, "arming notice must be refreshed while parked"
  assert logs.states[0] == "arming"
  assert all(item["deadline"] == pytest.approx(5.0) for item in arming)
  assert arming[0]["ts"] >= dsp_probe.ARMING_SHOW_S
  assert len(arming) == 10  # one refresh per gate period while holding


def test_brief_park_writes_no_notice():
  clock = FakeClock()
  sm = FakeSM(clock)
  logs = NoticeLog()

  def sleep(dt):
    clock.advance(dt)
    if clock() == pytest.approx(0.5):
      sm.gear = "drive"
    if clock() == pytest.approx(1.0):
      sm.gear = "park"

  dsp_probe.wait_for_parked(sm, clock=clock, sleep=sleep, notice=logs)
  assert clock() == pytest.approx(6.0)
  assert logs.items[0] is None  # drop before ARMING_SHOW_S removes the key, no flash
  assert not logs.of_state("aborted")
  arming = logs.of_state("arming")
  assert arming and arming[0]["ts"] == pytest.approx(1.5)


def test_parked_drop_after_arming_marks_aborted():
  clock = FakeClock()
  sm = FakeSM(clock)
  logs = NoticeLog()

  def sleep(dt):
    clock.advance(dt)
    if clock() == pytest.approx(2.0):
      sm.gear = "drive"
    if clock() == pytest.approx(3.0):
      sm.gear = "park"

  dsp_probe.wait_for_parked(sm, clock=clock, sleep=sleep, notice=logs)
  aborted = logs.of_state("aborted")
  assert len(aborted) == 1
  assert aborted[0]["ts"] == pytest.approx(2.0)
  assert clock() == pytest.approx(8.0)


def test_monitor_child_detects_exit_first():
  clock = FakeClock()
  sm = FakeSM(clock)
  outcome, detail = dsp_probe.monitor_child(FakeProc(poll_fn=lambda: 7), sm, clock=clock,
                                            sleep=_advancing_sleep(clock))
  assert (outcome, detail) == ("exited", 7)


def test_monitor_child_kills_within_one_poll_interval_of_unpark():
  clock = FakeClock()
  sm = FakeSM(clock)
  flip = {}

  def sleep(dt):
    clock.advance(dt)
    if not flip:
      flip["t"] = clock()
      sm.gear = "drive"

  outcome, detail = dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=sleep,
                                            rss_reader=lambda p: None)
  assert (outcome, detail) == ("unparked", None)
  assert clock() - flip["t"] <= dsp_probe.RUN_POLL_S


def test_monitor_child_total_timeout():
  clock = FakeClock()
  sm = FakeSM(clock)
  outcome, detail = dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=_advancing_sleep(clock),
                                            total_timeout_s=1.0, rss_reader=lambda p: None)
  assert (outcome, detail) == ("timeout", None)
  # 0.1 s poll steps accumulate in binary floating point, so allow one step.
  assert 1.0 <= clock() <= 1.0 + dsp_probe.RUN_POLL_S + 1e-6


def test_monitor_child_rss_limit():
  clock = FakeClock()
  sm = FakeSM(clock)
  outcome, detail = dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=_advancing_sleep(clock),
                                            rss_reader=lambda p: 999, rss_limit=500)
  assert (outcome, detail) == ("rss", 999)


def test_monitor_child_running_notice_tracks_progress(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  progress = str(tmp_path / "c.progress")
  logs = NoticeLog()
  written = []

  def sleep(dt):
    clock.advance(dt)
    now = clock()
    for threshold, step in ((0.3, "presence"), (1.0, "smoke"), (1.5, "detector")):
      if now >= threshold and step not in written:
        written.append(step)
        dsp_probe.write_progress(progress, step)

  dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=sleep, total_timeout_s=2.0,
                          rss_reader=lambda p: None, progress_file=progress, notice=logs)
  running = logs.of_state("running")
  assert running, "running notice must be refreshed while the child runs"
  assert all(item["steps"] == 3 for item in running)
  assert [item["step"] for item in running] == [1, 1, 2, 3]


# ---------------------------------------------------------------------------
# parent: gate loop, latch restore, result guarantee
# ---------------------------------------------------------------------------

def test_run_probe_clean_finish_deletes_latch_and_posts_done(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  pre_state = {"count": 1, "lost_steps": []}
  dsp_probe.write_attempt_state(attempt, pre_state)
  dsp_probe.write_progress(progress, "smoke")
  logs = NoticeLog()
  seen_attempts = []

  def factory(cmd, **kwargs):
    ticks = {"n": 0}

    def poll():
      ticks["n"] += 1
      seen_attempts.append(dsp_probe.read_attempt_state(attempt))
      if ticks["n"] >= 3:
        dsp_probe.write_result_atomic(result, {"ok": True, "steps": {}})
        return 0
      return None

    return FakeProc(poll_fn=poll)

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=pre_state,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock),
                               rss_reader=lambda p: None, meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "done"
  assert _load_json(result)["ok"] is True
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)
  assert seen_attempts[0] == {"count": 2, "lost_steps": []}
  assert logs.states[0] == "arming"
  assert "running" in logs.states
  assert logs.states[-1] == "done"
  assert not logs.of_state("failed")


def test_run_probe_unparked_kills_restores_and_aborts(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  pre_state = {"count": 2, "lost_steps": ["smoke"]}
  dsp_probe.write_attempt_state(attempt, pre_state)
  dsp_probe.write_progress(progress, "smoke")
  logs = NoticeLog()
  spawns = []
  capture = {"restored": None}
  phase = {"unparked": False, "reparked": False}

  def factory(cmd, **kwargs):
    spawns.append(dsp_probe.read_attempt_state(attempt))
    index = len(spawns)
    if index == 1:
      flipped = {"done": False}

      def poll():
        if not flipped["done"]:
          flipped["done"] = True
          sm.gear = "drive"
          phase["unparked"] = True
        return None

      return FakeProc(poll_fn=poll)

    def poll_finish():
      dsp_probe.write_result_atomic(result, {"ok": True, "steps": {}})
      return 0

    return FakeProc(poll_fn=poll_finish)

  def sleep(dt):
    clock.advance(dt)
    if phase["unparked"] and not phase["reparked"]:
      # first gate iteration after the abort: latch restored, no result, progress gone
      capture["restored"] = dsp_probe.read_attempt_state(attempt)
      capture["progress_exists"] = os.path.exists(progress)
      capture["result_exists"] = os.path.exists(result)
      phase["reparked"] = True
      sm.gear = "park"

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=pre_state,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=sleep, rss_reader=lambda p: None,
                               meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "done"
  assert capture["restored"] == pre_state
  assert capture["progress_exists"] is False
  assert capture["result_exists"] is False
  aborted = logs.of_state("aborted")
  assert len(aborted) == 1
  assert aborted[0]["ts"] == pytest.approx(5.0)
  assert logs.states[-1] == "done"
  # the first spawn saw the incremented latch, the clean finish removed it
  assert spawns[0] == {"count": 3, "lost_steps": ["smoke"]}
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)


def test_run_probe_timeout_writes_child_timeout_and_posts_failed(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  logs = NoticeLog()
  procs = []

  def factory(cmd, **kwargs):
    proc = FakeProc()
    procs.append(proc)
    return proc

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock),
                               rss_reader=lambda p: None, total_timeout_s=1.0,
                               meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "timeout"
  assert procs[0].killed is True
  payload = _load_json(result)
  assert payload["error"] == "child_timeout"
  assert payload["commit"] == "abc1234"
  assert payload["ok"] is False
  assert not (tmp_path / "abc1234.json.tmp").exists()
  assert len(logs.of_state("failed")) == 1
  assert "done" not in logs.states
  assert not os.path.exists(attempt)


def test_run_probe_rss_limit_writes_result(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  logs = NoticeLog()

  def factory(cmd, **kwargs):
    return FakeProc()

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock),
                               rss_reader=lambda p: dsp_probe.CHILD_RSS_LIMIT_BYTES + 1,
                               meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "rss"
  payload = _load_json(result)
  assert payload["error"] == "child_rss_limit"
  assert len(logs.of_state("failed")) == 1


def test_run_probe_child_exit_without_result(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)

  def factory(cmd, **kwargs):
    return FakeProc(poll_fn=lambda: 3)

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock), rss_reader=lambda p: None,
                               meta={"commit": "abc1234"})
  assert status["status"] == "exited"
  payload = _load_json(result)
  assert payload["error"] == "child_exit_3"
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)


# ---------------------------------------------------------------------------
# attempt / progress roundtrip + lost-step latch
# ---------------------------------------------------------------------------

def test_attempt_progress_roundtrip(tmp_path):
  attempt = str(tmp_path / "c.attempt")
  progress = str(tmp_path / "c.progress")
  assert dsp_probe.read_attempt_state(attempt) is None
  assert dsp_probe.read_progress_step(progress) is None
  dsp_probe.write_attempt_state(attempt, {"count": 2, "lost_steps": ["smoke", "detector"]})
  assert dsp_probe.read_attempt_state(attempt) == {"count": 2, "lost_steps": ["smoke", "detector"]}
  with open(attempt, "w", encoding="utf-8") as f:
    f.write("{not json")
  assert dsp_probe.read_attempt_state(attempt) is None
  dsp_probe.write_progress(progress, "smoke")
  assert dsp_probe.read_progress_step(progress) == "smoke"
  with open(progress, "w", encoding="utf-8") as f:
    f.write("[]")
  assert dsp_probe.read_progress_step(progress) is None


def test_latched_step_requires_same_step_three_times():
  assert dsp_probe._latched_step(["smoke", "smoke"]) is None
  assert dsp_probe._latched_step(["smoke", "smoke", "smoke"]) == "smoke"
  assert dsp_probe._latched_step(["smoke", "detector", "smoke"]) is None
  assert dsp_probe._latched_step(["presence", "smoke", "detector"]) is None
  assert dsp_probe._latched_step(["detector", "smoke", "detector", "smoke", "detector"]) == "detector"


def test_main_latches_after_three_same_step_losses(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "abc1234.json")
  dsp_probe.write_attempt_state(dsp_probe.attempt_path(result), {"count": 2, "lost_steps": ["smoke", "smoke"]})
  dsp_probe.write_progress(dsp_probe.progress_path(result), "smoke")
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))

  dsp_probe.main()

  payload = _load_json(result)
  assert payload["error"] == "lost_in_step:smoke"
  assert payload["ok"] is False
  assert dsp_probe._probe_disabled_by_lost_attempts is True
  assert not os.path.exists(dsp_probe.attempt_path(result))
  assert not os.path.exists(dsp_probe.progress_path(result))
  os.unlink(result)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is False  # latched, not the result file


def test_main_does_not_latch_on_mixed_losses(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "abc1234.json")
  dsp_probe.write_attempt_state(dsp_probe.attempt_path(result), {"count": 3, "lost_steps": ["smoke", "detector"]})
  dsp_probe.write_progress(dsp_probe.progress_path(result), "smoke")
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: None)
  calls = []
  fake_sm = object()
  monkeypatch.setattr(dsp_probe, "_make_submaster", lambda: fake_sm)
  monkeypatch.setattr(dsp_probe, "run_probe", lambda *a, **k: calls.append(k) or {"status": "done"})

  dsp_probe.main()

  assert len(calls) == 1
  assert calls[0]["sm"] is fake_sm
  assert calls[0]["notice"] is not None
  assert dsp_probe._probe_disabled_by_lost_attempts is False
  assert not os.path.exists(result)


def test_main_retries_messaging_then_runs_probe(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  captured = {}
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: captured.__setitem__(sig, fn))
  monkeypatch.setattr(dsp_probe.time, "sleep", lambda _s: None)
  attempts = {"n": 0}
  fake_sm = object()

  def flaky_submaster():
    attempts["n"] += 1
    if attempts["n"] < 3:
      raise RuntimeError("messaging not up")
    return fake_sm

  monkeypatch.setattr(dsp_probe, "_make_submaster", flaky_submaster)
  calls = []
  monkeypatch.setattr(dsp_probe, "run_probe", lambda *a, **k: calls.append(k) or {"status": "done"})

  dsp_probe.main()

  assert attempts["n"] == 3
  assert calls and calls[0]["sm"] is fake_sm
  assert set(captured) == {signal.SIGINT, signal.SIGTERM}


def test_main_consumes_progress_once_without_false_latch(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "abc1234.json")
  attempt = dsp_probe.attempt_path(result)
  progress = dsp_probe.progress_path(result)
  dsp_probe.write_attempt_state(attempt, {"count": 1, "lost_steps": []})
  dsp_probe.write_progress(progress, "smoke")
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: None)
  monkeypatch.setattr(dsp_probe.time, "sleep", lambda _s: None)
  monkeypatch.setattr(dsp_probe, "_make_submaster", lambda: object())
  monkeypatch.setattr(dsp_probe, "run_probe", _power_cut)

  for _boot in range(2):
    with pytest.raises(PowerCut):
      dsp_probe.main()

  assert dsp_probe.read_attempt_state(attempt) == {"count": 1, "lost_steps": ["smoke"]}
  assert not os.path.exists(progress)
  assert not os.path.exists(result)
  assert dsp_probe._probe_disabled_by_lost_attempts is False


def test_main_still_latches_three_separate_lost_runs(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "abc1234.json")
  attempt = dsp_probe.attempt_path(result)
  progress = dsp_probe.progress_path(result)
  dsp_probe.write_attempt_state(attempt, {"count": 1, "lost_steps": []})
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: None)
  monkeypatch.setattr(dsp_probe.time, "sleep", lambda _s: None)
  monkeypatch.setattr(dsp_probe, "_make_submaster", lambda: object())
  monkeypatch.setattr(dsp_probe, "run_probe", _power_cut)

  for _boot in range(2):
    dsp_probe.write_progress(progress, "smoke")
    with pytest.raises(PowerCut):
      dsp_probe.main()

  dsp_probe.write_progress(progress, "smoke")
  dsp_probe.main()

  payload = _load_json(result)
  assert payload["error"] == "lost_in_step:smoke"
  assert dsp_probe._probe_disabled_by_lost_attempts is True
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)


# ---------------------------------------------------------------------------
# stop signal handling (orphan protection, DEFECT 1)
# ---------------------------------------------------------------------------

def test_stop_handler_kills_child_restores_and_clears_notice(monkeypatch):
  killed = []
  restored = []
  logs = NoticeLog()

  class P:
    def kill(self):
      killed.append(True)

    def wait(self, timeout=None):
      return 0

  monkeypatch.setattr(dsp_probe, "_RUNNING_CHILD", P())
  handler = dsp_probe._make_stop_handler(lambda: restored.append(1), logs)
  with pytest.raises(SystemExit) as excinfo:
    handler(signal.SIGTERM, None)
  assert excinfo.value.code == 0
  assert killed == [True]
  assert restored == [1]
  assert logs.items == [None]


def test_stop_handler_without_child_is_clean(monkeypatch):
  monkeypatch.setattr(dsp_probe, "_RUNNING_CHILD", None)
  restored = []
  handler = dsp_probe._make_stop_handler(lambda: restored.append(1))
  with pytest.raises(SystemExit):
    handler(signal.SIGINT, None)
  assert restored == [1]


def test_install_signal_handlers_registers_both(monkeypatch):
  captured = {}
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: captured.__setitem__(sig, fn))
  dsp_probe._install_signal_handlers(lambda: None)
  assert set(captured) == {signal.SIGINT, signal.SIGTERM}


# ---------------------------------------------------------------------------
# child preexec (PDEATHSIG)
# ---------------------------------------------------------------------------

def test_default_popen_passes_preexec_on_linux(monkeypatch):
  captured = {}

  def fake_popen(cmd, **kwargs):
    captured["cmd"] = list(cmd)
    captured.update(kwargs)
    return object()

  monkeypatch.setattr(dsp_probe, "IS_LINUX", True)
  monkeypatch.setattr(dsp_probe, "_POPEN", fake_popen)
  dsp_probe._default_popen(["x", "y"], stdout=None)
  assert captured["cmd"] == ["x", "y"]
  assert captured["preexec_fn"] is dsp_probe._child_preexec


def test_default_popen_skips_preexec_off_linux(monkeypatch):
  captured = {}

  def fake_popen(cmd, **kwargs):
    captured.update(kwargs)
    return object()

  monkeypatch.setattr(dsp_probe, "IS_LINUX", False)
  monkeypatch.setattr(dsp_probe, "_POPEN", fake_popen)
  dsp_probe._default_popen(["x"])
  assert "preexec_fn" not in captured


def test_default_popen_honors_explicit_preexec(monkeypatch):
  captured = {}

  def fake_popen(cmd, **kwargs):
    captured.update(kwargs)
    return object()

  monkeypatch.setattr(dsp_probe, "IS_LINUX", True)
  monkeypatch.setattr(dsp_probe, "_POPEN", fake_popen)
  dsp_probe._default_popen(["x"], preexec_fn=None)
  assert captured["preexec_fn"] is None


def test_child_preexec_never_raises():
  dsp_probe._child_preexec()  # best-effort on every platform


# ---------------------------------------------------------------------------
# UI notice sink
# ---------------------------------------------------------------------------

def test_make_notice_sink_puts_and_removes():
  calls = []

  class FakeParamsObj:
    def put(self, key, value):
      calls.append(("put", key, value))

    def remove(self, key):
      calls.append(("remove", key))

  sink = dsp_probe.make_notice_sink(params_factory=FakeParamsObj)
  sink({"state": "arming", "ts": 1.0, "deadline": 5.0})
  sink(None)
  assert calls == [("put", "CarrotTlvDspNotice", {"state": "arming", "ts": 1.0, "deadline": 5.0}),
                   ("remove", "CarrotTlvDspNotice")]


def test_make_notice_sink_failure_is_silent():
  def boom():
    raise RuntimeError("no params")

  sink = dsp_probe.make_notice_sink(params_factory=boom)
  sink({"state": "done", "ts": 1.0})
  sink(None)

  class ExplodingParams:
    def put(self, key, value):
      raise RuntimeError("put failed")

    def remove(self, key):
      raise RuntimeError("remove failed")

  sink2 = dsp_probe.make_notice_sink(params_factory=ExplodingParams)
  sink2({"state": "done", "ts": 1.0})
  sink2(None)


def test_publish_outcome_notice_maps_ok(tmp_path):
  logs = NoticeLog()
  result = str(tmp_path / "r.json")
  dsp_probe.write_result_atomic(result, dsp_probe.build_result_payload(
    {"a": {"status": "pass"}}, commit="c", device_type=None, agnos_version=None, timestamp="t"))
  dsp_probe._publish_outcome_notice(result, 1.0, logs)
  dsp_probe.write_result_atomic(result, dsp_probe.build_result_payload(
    {}, commit="c", device_type=None, agnos_version=None, timestamp="t", error="x"))
  dsp_probe._publish_outcome_notice(result, 2.0, logs)
  assert logs.states == ["done", "failed"]
  assert logs.items[0]["ts"] == 1.0
  dsp_probe._publish_outcome_notice(str(tmp_path / "missing.json"), 3.0, logs)
  assert logs.states[-1] == "failed"


def test_step_index_from_progress(tmp_path):
  progress = str(tmp_path / "p")
  assert dsp_probe._step_index_from_progress(None, 2) == 2
  assert dsp_probe._step_index_from_progress(progress, 1) == 1  # missing -> last known
  dsp_probe.write_progress(progress, "detector")
  assert dsp_probe._step_index_from_progress(progress, 1) == 3
  dsp_probe.write_progress(progress, "smoke")
  assert dsp_probe._step_index_from_progress(progress, 1) == 2
