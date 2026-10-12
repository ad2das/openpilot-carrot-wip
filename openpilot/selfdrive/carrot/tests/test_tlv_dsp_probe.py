"""Tests for the offroad DSP probe (openpilot/selfdrive/carrot/tlv/dsp_probe.py).

No real DSP exists on the PC: every step is exercised through its injectable
dependencies (path/runner/loader/adapter fakes) and subprocess use is mocked.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import dsp_probe


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
# enable condition / result path
# ---------------------------------------------------------------------------

def test_should_run_truth_table(probe_root):
  params = FakeParams({"CarrotTlvMode": 1, "TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(False, params) is True
  assert dsp_probe.should_run(True, params) is False  # onroad
  assert dsp_probe.should_run(False, FakeParams({"CarrotTlvMode": 0, "TrafficLightDetectMode": 1})) is False
  assert dsp_probe.should_run(False, FakeParams({"CarrotTlvMode": 1, "TrafficLightDetectMode": 0})) is False
  assert dsp_probe.should_run(False, FakeParams({})) is False  # missing -> off

  probe_root.mkdir(parents=True, exist_ok=True)
  (probe_root / "abc1234.json").write_text("{}", encoding="utf-8")
  assert dsp_probe.should_run(False, params) is False  # latched for this commit


def test_probe_result_path_uses_short_commit(probe_root):
  assert dsp_probe.probe_result_path() == os.path.join(str(probe_root), "abc1234.json")


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


# ---------------------------------------------------------------------------
# parent: hard total timeout + result guarantee
# ---------------------------------------------------------------------------

class _FakeProc:
  def __init__(self, *, timeout=False, returncode=0):
    self._timeout = timeout
    self.returncode = returncode
    self.killed = False

  def wait(self, timeout=None):
    if self._timeout and not self.killed:
      raise subprocess.TimeoutExpired("cmd", timeout or 1.0)
    return self.returncode

  def kill(self):
    self.killed = True
    self.returncode = -9


def test_run_probe_skips_when_result_exists(tmp_path):
  path = tmp_path / "abc.json"
  path.write_text("{}", encoding="utf-8")
  called = []
  result = dsp_probe.run_probe(str(path), popen_factory=lambda *a, **k: called.append(1))
  assert result["status"] == "already-done"
  assert not called


def test_run_probe_kills_child_on_total_timeout(tmp_path):
  path = tmp_path / "abc.json"
  procs = []

  def factory(cmd, **kwargs):
    proc = _FakeProc(timeout=True)
    procs.append(proc)
    return proc

  result = dsp_probe.run_probe(str(path), popen_factory=factory, total_timeout_s=1.0,
                               meta={"commit": "abc", "device_type": "pc", "agnos_version": None})
  assert result["status"] == "killed"
  assert procs and procs[0].killed is True
  payload = json.loads(path.read_text(encoding="utf-8"))
  assert payload["error"] == "child_timeout"
  assert payload["commit"] == "abc"
  assert payload["ok"] is False
  assert not (tmp_path / "abc.json.tmp").exists()


def test_run_probe_keeps_child_result(tmp_path):
  path = tmp_path / "abc.json"

  def factory(cmd, **kwargs):
    path.write_text('{"ok": true}', encoding="utf-8")
    return _FakeProc(returncode=0)

  result = dsp_probe.run_probe(str(path), popen_factory=factory, total_timeout_s=5.0)
  assert result["status"] == "done"
  assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_run_probe_writes_failure_when_child_exits_without_result(tmp_path):
  path = tmp_path / "abc.json"

  def factory(cmd, **kwargs):
    return _FakeProc(returncode=3)

  result = dsp_probe.run_probe(str(path), popen_factory=factory, total_timeout_s=5.0,
                               meta={"commit": "abc"})
  assert result["status"] == "done"
  payload = json.loads(path.read_text(encoding="utf-8"))
  assert payload["error"] == "child_exit_3"
