"""Tests for the bundled r8s50a40 traffic-light models and their default paths."""

import hashlib
import os
import shutil
import sys
import types

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import backends, dsp_probe, resources, tlvd

DETECTOR_OUTPUT_SHAPES = ((1, 64, 40, 80), (1, 1, 40, 80), (1, 64, 20, 40), (1, 1, 20, 40), (1, 64, 10, 20), (1, 1, 10, 20))


class FakeParams:
  def __init__(self, values=None) -> None:
    self._values = dict(values or {})

  def get_int(self, key):
    value = self._values.get(key)
    return None if value is None else int(value)

  def get(self, key):
    return self._values.get(key)


def _sha256(path: str) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _install_backend_capture(monkeypatch) -> list:
  calls: list = []

  class FakeBackend:
    def warmup(self) -> None:
      pass

    def close(self) -> None:
      pass

  def make_backend(name, model_path="", decoder=None):
    calls.append(("backend", name, model_path))
    return FakeBackend(), None

  def make_classifier(name, model_path=""):
    calls.append(("classifier", name, model_path))
    return FakeBackend()

  monkeypatch.setattr(backends, "make_backend", make_backend)
  monkeypatch.setattr(backends, "make_classifier", make_classifier)
  return calls


def test_bundled_models_exist_and_match_hashes():
  assert os.path.isfile(backends.BUNDLED_DETECTOR)
  assert os.path.isfile(backends.BUNDLED_LAMP)
  assert _sha256(backends.BUNDLED_DETECTOR) == backends.BUNDLED_DETECTOR_SHA256
  assert _sha256(backends.BUNDLED_LAMP) == backends.BUNDLED_LAMP_SHA256


def test_tlvd_empty_params_resolve_to_bundled_paths(monkeypatch):
  calls = _install_backend_capture(monkeypatch)
  runner = tlvd.TlvRunner(FakeParams({}))
  assert runner.backend is not None
  assert ("backend", "onnx", backends.BUNDLED_DETECTOR) in calls
  assert ("classifier", "onnx", backends.BUNDLED_LAMP) in calls


def test_tlvd_explicit_params_override_bundled_paths(monkeypatch):
  calls = _install_backend_capture(monkeypatch)
  params = FakeParams(
    {
      "CarrotTlvModelPath": "D:/models/custom_detector.onnx",
      "CarrotTlvLampModelPath": "D:/models/custom_lamp.onnx",
    }
  )
  runner = tlvd.TlvRunner(params)
  assert runner.backend is not None
  assert ("backend", "onnx", "D:/models/custom_detector.onnx") in calls
  assert ("classifier", "onnx", "D:/models/custom_lamp.onnx") in calls


def _install_warmup_capture(monkeypatch, *, detector_raises=None):
  events: list = []

  class FakeDetector:
    def __init__(self):
      self.warmups = 0

    def warmup(self):
      events.append("detector_warmup")
      self.warmups += 1
      if detector_raises is not None:
        raise detector_raises

    def close(self):
      pass

  class FakeClassifier:
    def warmup(self):
      events.append("classifier_warmup")

    def close(self):
      pass

  detector = FakeDetector()

  def make_backend(name, model_path="", decoder=None):
    return detector, None

  def make_classifier(name, model_path=""):
    return FakeClassifier()

  monkeypatch.setattr(backends, "make_backend", make_backend)
  monkeypatch.setattr(backends, "make_classifier", make_classifier)
  return events, detector


def test_detector_warmup_runs_before_budget_creation(monkeypatch):
  # W127: both warm-ups must complete BEFORE CpuCreditBudget() is constructed,
  # so the one-time compile CPU (route 00000443) is not charged to the frames.
  events, detector = _install_warmup_capture(monkeypatch)
  real_budget_cls = resources.CpuCreditBudget

  class RecordingBudget(real_budget_cls):
    def __init__(self, **kwargs):
      if kwargs == {}:
        events.append("budget")
      super().__init__(**kwargs)

  monkeypatch.setattr(resources, "CpuCreditBudget", RecordingBudget)
  tlvd.TlvRunner(FakeParams({}))

  assert events.index("classifier_warmup") < events.index("budget")
  assert events.index("detector_warmup") < events.index("budget")
  assert detector.warmups == 1


def test_detector_warmup_failure_sets_backend_error(monkeypatch):
  events, _detector = _install_warmup_capture(monkeypatch, detector_raises=RuntimeError("boom"))
  runner = tlvd.TlvRunner(FakeParams({}))
  assert "detector warmup failed" in runner.backend_error
  # A warm-up failure must block inference exactly like a backend error.
  assert runner._can_infer() is False


def _install_fake_params_module(monkeypatch, value) -> None:
  class FakeParamsModule:
    def get(self, key):
      return value

  module = types.ModuleType("openpilot.common.params")
  module.Params = FakeParamsModule
  monkeypatch.setitem(sys.modules, "openpilot.common.params", module)


def test_dsp_probe_model_path_falls_back_to_bundled(monkeypatch):
  _install_fake_params_module(monkeypatch, None)
  assert dsp_probe._model_path_from_params() == backends.BUNDLED_DETECTOR
  _install_fake_params_module(monkeypatch, b"")
  assert dsp_probe._model_path_from_params() == backends.BUNDLED_DETECTOR
  _install_fake_params_module(monkeypatch, b"  /data/models/explicit.onnx  ")
  assert dsp_probe._model_path_from_params() == "/data/models/explicit.onnx"


def _ensure_cc(tmp_path, monkeypatch) -> bool:
  """tinygrad compiles CPU kernels with clang (CC env); reuse the ziglang
  wrapper used by test_tlv_backends when clang is not on PATH."""
  if shutil.which("clang"):
    return True
  try:
    import ziglang  # noqa: F401
  except ImportError:
    return False
  driver = tmp_path / "zigcc_driver.py"
  driver.write_text(
    "\n".join(
      [
        "import subprocess, sys",
        "args = [a.replace('-none-unknown-elf', '-linux-gnu') for a in sys.argv[1:]]",
        "args.append('-g0')",
        "sys.exit(subprocess.run([sys.executable, '-m', 'ziglang', 'cc', *args],",
        "                        input=sys.stdin.buffer.read()).returncode)",
      ]
    )
    + "\n",
    encoding="utf-8",
  )
  wrapper = tmp_path / "zigcc.cmd"
  wrapper.write_text(f'@"{sys.executable}" "{driver}" %*\n', encoding="ascii")
  monkeypatch.setenv("CC", str(wrapper))
  return True


def test_tinygrad_loads_bundled_detector_with_expected_shapes(tmp_path, monkeypatch):
  try:
    import tinygrad
  except ImportError:
    repo = backends._tinygrad_repo_path()
    if repo not in sys.path:
      sys.path.append(repo)
    try:
      import tinygrad  # noqa: F401
    except ImportError:
      pytest.skip("tinygrad is not available")
  if not _ensure_cc(tmp_path, monkeypatch):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")

  backend, _decoder = backends.make_backend("tinygrad", backends.BUNDLED_DETECTOR)
  try:
    assert backend.name == "tinygrad"
    inputs = np.random.default_rng(0).integers(0, 256, size=(1, 3, 320, 640), dtype=np.uint8)
    outs = backend.infer(inputs)
    assert [tuple(np.asarray(out).shape) for out in outs] == list(DETECTOR_OUTPUT_SHAPES)
  finally:
    backend.close()


def test_tinygrad_loads_bundled_lamp_matches_ort(tmp_path, monkeypatch):
  try:
    import tinygrad
  except ImportError:
    repo = backends._tinygrad_repo_path()
    if repo not in sys.path:
      sys.path.append(repo)
    try:
      import tinygrad  # noqa: F401
    except ImportError:
      pytest.skip("tinygrad is not available")
  if not _ensure_cc(tmp_path, monkeypatch):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")

  inputs = np.random.default_rng(0).integers(0, 256, (4, 3, 32, 64), dtype=np.uint8)
  tg = backends.make_classifier("tinygrad", backends.BUNDLED_LAMP)
  try:
    out = tg.classify(inputs)
    assert out is not None
    assert np.asarray(out).shape == (4, 8)

    try:
      import onnxruntime  # noqa: F401
    except ImportError:
      return
    ort = backends.make_classifier("onnx", backends.BUNDLED_LAMP)
    try:
      ort_out = ort.classify(inputs)
      assert ort_out is not None
      assert np.allclose(np.asarray(out), np.asarray(ort_out), atol=1e-3)
    finally:
      ort.close()
  finally:
    tg.close()
