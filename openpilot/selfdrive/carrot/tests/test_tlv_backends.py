"""Tests for TLV inference backends: CPU-only refusal guards and ONNX stub model."""
import os
import shutil
import sys
import time
import types

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import backends, crops
from openpilot.selfdrive.carrot.tlv.decode import PlaceholderDecoder


def test_gpu_env_reason_detects_gpu_hints():
  assert backends.gpu_env_reason({}) is None
  assert backends.gpu_env_reason({"PATH": "/usr/bin", "HOME": "/root"}) is None
  assert backends.gpu_env_reason({"GPU": "0"}) is None
  assert backends.gpu_env_reason({"QCOM": "1"}) is not None
  assert backends.gpu_env_reason({"GPU": "1"}) is not None
  assert backends.gpu_env_reason({"CL": "1"}) is not None
  assert backends.gpu_env_reason({"CUDA_VISIBLE_DEVICES": "0"}) is not None
  assert backends.gpu_env_reason({"OPENCL_VENDOR_PATH": "/etc"}) is not None
  assert backends.gpu_env_reason({"SOME_VAR": "use CUDA here"}) is not None
  assert backends.gpu_env_reason({"CLOUD_MODE": "1"}) is None


def test_null_backend_and_factory():
  backend, decoder = backends.make_backend("null", "")
  assert backend.name == "null"
  assert backend.infer(np.zeros((1, 3, 320, 640), dtype=np.uint8)) == []
  assert isinstance(decoder, PlaceholderDecoder)
  with pytest.raises(backends.BackendError):
    backends.make_backend("banana", "")


def test_dsp_backend_refuses_a_missing_or_custom_model():
  # WSNPE13: dsp is a real backend now; it only runs the bundled detector DLC.
  with pytest.raises(backends.BackendError, match="bundled detector"):
    backends.make_backend("dsp", "")


def test_onnx_backend_refuses_bad_configuration(tmp_path):
  with pytest.raises(backends.BackendError, match="not set"):
    backends.make_backend("onnx", "")
  missing = tmp_path / "missing.onnx"
  with pytest.raises(backends.BackendError, match="model not found"):
    backends.make_backend("onnx", str(missing))


def test_onnx_backend_refuses_gpu_environment(tmp_path, monkeypatch):
  onnx = pytest.importorskip("onnx")
  path = _tiny_onnx_model(onnx, tmp_path)
  monkeypatch.setenv("QCOM", "1")
  with pytest.raises(backends.BackendError, match="GPU-capable"):
    backends.make_backend("onnx", str(path))


def _tiny_onnx_model(onnx, tmp_path):
  from onnx import TensorProto, helper
  img = helper.make_tensor_value_info("img", TensorProto.UINT8, [1, 3, 320, 640])
  boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4])
  scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1])
  classes = helper.make_tensor_value_info("classes", TensorProto.INT64, [1])
  nodes = [
    helper.make_node("Constant", [], ["boxes"], value=helper.make_tensor("boxes_v", TensorProto.FLOAT, [1, 4], [100.0, 50.0, 140.0, 90.0])),
    helper.make_node("Constant", [], ["scores"], value=helper.make_tensor("scores_v", TensorProto.FLOAT, [1], [0.9])),
    helper.make_node("Constant", [], ["classes"], value=helper.make_tensor("classes_v", TensorProto.INT64, [1], [2])),
  ]
  graph = helper.make_graph(nodes, "tlv_stub", [img], [boxes, scores, classes])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
  model.ir_version = 10  # onnxruntime 1.30 refuses the onnx-default IR 14
  path = tmp_path / "stub.onnx"
  onnx.save(model, str(path))
  return path


def test_onnx_cpu_backend_infers_and_decodes(tmp_path):
  onnx = pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")
  path = _tiny_onnx_model(onnx, tmp_path)
  backend, decoder = backends.make_backend("onnx", str(path))
  raw = backend.infer(np.zeros((1, 3, 320, 640), dtype=np.uint8))
  dets = decoder.decode(raw, "far")
  assert len(dets) == 1
  assert dets[0].cls == 2 and dets[0].score == pytest.approx(0.9)
  backend.close()


def test_onnx_stub_end_to_end_maps_to_full_frame(tmp_path):
  onnx = pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")
  path = _tiny_onnx_model(onnx, tmp_path)
  backend, decoder = backends.make_backend("onnx", str(path))
  raw = backend.infer(np.zeros((1, 3, 320, 640), dtype=np.uint8))
  det = decoder.decode(raw, "far")[0]
  geom = crops.far_geometry(600.0, 1928, 1208, 2648.0)
  x1, y1, x2, y2 = crops.map_model_box_to_full((det.x1, det.y1, det.x2, det.y2), geom)
  assert (x1, y1, x2, y2) == (744.0, 410.0, 784.0, 450.0)
  backend.close()


def test_placeholder_decoder_contract():
  decoder = PlaceholderDecoder(score_threshold=0.25)
  boxes = np.array([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
  scores = np.array([0.9, 0.1], dtype=np.float32)
  classes = np.array([0, 2], dtype=np.int64)
  dets = decoder.decode((boxes, scores, classes), "near")
  assert len(dets) == 1 and dets[0].cls == 0
  dets = decoder.decode({"boxes": boxes, "scores": scores, "classes": classes}, "near")
  assert len(dets) == 1
  assert decoder.decode(None, "near") == []
  assert decoder.decode((np.zeros((2, 5)), scores, classes), "near") == []
  bad = decoder.decode((np.array([[30.0, 20.0, 10.0, 60.0]]), np.array([0.9]), np.array([2])), "far")
  assert bad == []


# -- detector warm-up hooks (W127) -------------------------------------------

def test_null_backend_warmup_is_noop():
  backends.NullBackend("").warmup()


def test_onnx_backend_warmup_before_infer(tmp_path):
  onnx = pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")
  path = _tiny_onnx_model(onnx, tmp_path)
  backend, _decoder = backends.make_backend("onnx", str(path))
  try:
    backend.warmup()  # first-run allocations happen here, not on the road
    raw = backend.infer(np.zeros((1, 3, 320, 640), dtype=np.uint8))
    assert raw is not None
  finally:
    backend.close()


def _tiny_conv_onnx_model(onnx, tmp_path):
  """A tiny Conv+ReLU ONNX (1,3,8,8) for CPU parity checks."""
  from onnx import TensorProto, helper
  x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
  y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
  rng = np.random.default_rng(1)
  w = helper.make_tensor("w", TensorProto.FLOAT, [3, 3, 3, 3],
                         rng.standard_normal(81).astype(np.float32).tobytes(), raw=True)
  b = helper.make_tensor("b", TensorProto.FLOAT, [3],
                         np.array([0.5, -1.0, 2.0], np.float32).tobytes(), raw=True)
  nodes = [
    helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    helper.make_node("Relu", ["c"], ["output"]),
  ]
  graph = helper.make_graph(nodes, "tlv_conv", [x], [y], [w, b])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
  model.ir_version = 10  # onnxruntime 1.30 refuses the onnx-default IR 14
  path = tmp_path / "conv.onnx"
  onnx.save(model, str(path))
  return path


def _ensure_cc(monkeypatch, tmp_path) -> bool:
  """tinygrad compiles CPU kernels with clang (CC env). Use a ziglang wrapper
  when clang is not on PATH; return False when no clang-compatible driver exists."""
  if shutil.which("clang"):
    return True
  try:
    import ziglang  # noqa: F401
  except ImportError:
    return False
  driver = tmp_path / "zigcc_driver.py"
  driver.write_text(
    "\n".join([
      "import subprocess, sys",
      "args = [a.replace('-none-unknown-elf', '-linux-gnu') for a in sys.argv[1:]]",
      "args.append('-g0')",
      "sys.exit(subprocess.run([sys.executable, '-m', 'ziglang', 'cc', *args],",
      "                        input=sys.stdin.buffer.read()).returncode)",
    ]) + "\n",
    encoding="utf-8",
  )
  wrapper = tmp_path / "zigcc.cmd"
  wrapper.write_text(f'@"{sys.executable}" "{driver}" %*\n', encoding="ascii")
  monkeypatch.setenv("CC", str(wrapper))
  return True


def test_tinygrad_backend_refuses_bad_configuration():
  with pytest.raises(backends.BackendError, match="not set"):
    backends.make_backend("tinygrad", "")
  with pytest.raises(backends.BackendError, match="model not found"):
    backends.make_backend("tinygrad", "Z:/definitely/missing.onnx")


def test_tinygrad_backend_refuses_gpu_environment(tmp_path, monkeypatch):
  onnx = pytest.importorskip("onnx")
  path = _tiny_conv_onnx_model(onnx, tmp_path)
  monkeypatch.setenv("QCOM", "1")
  with pytest.raises(backends.BackendError, match="GPU-capable"):
    backends.make_backend("tinygrad", str(path))


def test_tinygrad_backend_pins_cpu_with_dev_unset(tmp_path, monkeypatch):
  """DEV unset + a GPU that would win Device.DEFAULT auto-selection: the
  backend must not probe at all and must never open a non-CPU device."""
  onnx = pytest.importorskip("onnx")
  if not _ensure_cc(monkeypatch, tmp_path):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")
  from tinygrad import device as tgdevice
  from tinygrad.helpers import DEV as tg_dev
  path = _tiny_conv_onnx_model(onnx, tmp_path)

  monkeypatch.delenv("DEV", raising=False)
  tg_dev.value = ""  # fresh-process state: no DEV env, no resolved target
  probes = []

  def fake_available_devices():
    probes.append(1)
    yield "CUDA"  # first in ALL_DEVICES on this class of machine (device.py:14)

  monkeypatch.setattr(tgdevice.Device, "get_available_devices", fake_available_devices)
  monkeypatch.delitem(tgdevice.Device.__dict__, "_select_device", raising=False)
  opened_before = set(map(str, tgdevice.Device._opened_devices))

  backend, decoder = backends.make_backend("tinygrad", str(path))
  assert backend.device_name in backends.TINYGRAD_CPU_DEVICES
  assert probes == []  # Device.DEFAULT probing never ran
  assert os.environ.get("DEV") == "CPU"
  assert str(tgdevice.Device.DEFAULT).split(":")[0] in backends.TINYGRAD_CPU_DEVICES
  for dev in set(map(str, tgdevice.Device._opened_devices)) - opened_before:
    base = str(dev).split(":")[0]
    assert base in backends.TINYGRAD_CPU_DEVICES or base in backends.TINYGRAD_HOST_DEVICES, dev
  backend.close()


def test_tinygrad_backend_overrides_inherited_gpu_dev(tmp_path, monkeypatch):
  """An inherited DEV=QCOM (parent env / earlier import) must be overridden,
  not obeyed, and must never reach tinygrad device selection."""
  onnx = pytest.importorskip("onnx")
  if not _ensure_cc(monkeypatch, tmp_path):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")
  from tinygrad.helpers import DEV as tg_dev
  path = _tiny_conv_onnx_model(onnx, tmp_path)

  monkeypatch.setenv("DEV", "QCOM")
  tg_dev.value = "QCOM"
  backend, decoder = backends.make_backend("tinygrad", str(path))
  assert backend.device_name in backends.TINYGRAD_CPU_DEVICES
  assert os.environ.get("DEV") == "CPU"
  assert str(tg_dev.device).split(":")[0] in backends.TINYGRAD_CPU_DEVICES
  backend.close()


def test_resolve_backend_name_defaults():
  from openpilot.selfdrive.carrot.tlv import tlvd
  # WSNPE13: "" resolves to the Hexagon DSP detector on TICI and to onnxruntime
  # on PC; an explicit "onnx" always means onnxruntime.
  assert tlvd.resolve_backend_name("", False) == "onnx"
  assert tlvd.resolve_backend_name("onnx", False) == "onnx"
  assert tlvd.resolve_backend_name("", True) == "dsp"
  assert tlvd.resolve_backend_name("onnx", True) == "onnx"
  assert tlvd.resolve_backend_name("tinygrad", True) == "tinygrad"
  assert tlvd.resolve_backend_name("null", True) == "null"
  assert tlvd.resolve_backend_name("dsp", False) == "dsp"


def test_tinygrad_cpu_backend_matches_onnxruntime(tmp_path, monkeypatch):
  onnx = pytest.importorskip("onnx")
  ort = pytest.importorskip("onnxruntime")
  if not _ensure_cc(monkeypatch, tmp_path):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")
  path = _tiny_conv_onnx_model(onnx, tmp_path)
  x = np.random.default_rng(3).standard_normal((1, 3, 8, 8)).astype(np.float32)

  so = ort.SessionOptions()
  so.intra_op_num_threads = 1
  sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
  ref = sess.run(None, {"input": x})[0]

  backend, decoder = backends.make_backend("tinygrad", str(path))
  assert backend.name == "tinygrad"
  assert backend.device_name in backends.TINYGRAD_CPU_DEVICES
  assert backend.max_threads == 2

  got = backend.infer(x)[0]
  assert got.shape == ref.shape
  max_diff = float(np.max(np.abs(got - ref)))
  assert max_diff < 1e-4, f"tinygrad vs onnxruntime max abs diff {max_diff}"

  times = []
  for _ in range(20):
    t0 = time.perf_counter()
    backend.infer(x)
    times.append((time.perf_counter() - t0) * 1000.0)
  mean_ms, p95_ms, min_ms = float(np.mean(times)), float(np.percentile(times, 95)), float(np.min(times))
  print(f"\ntinygrad-cpu: max_diff={max_diff:.2e} per-inference mean={mean_ms:.2f}ms p95={p95_ms:.2f}ms min={min_ms:.2f}ms")
  backend.close()


def _tiny_classifier_onnx_model(onnx, tmp_path):
  """Tiny input-dependent lamp classifier: (4,3,32,64) uint8 -> logits (4,8).

  Same pattern as the real crop model: DequantizeLinear first, then a 1x1 conv
  and GlobalAveragePool. A zero crop still yields logits[:, 0] == 2.0 (bias),
  but the non-zero weights keep the graph input-dependent so the tinygrad JIT
  actually compiles kernels (a Constant-only graph defeats TinyJit capture).
  """
  from onnx import TensorProto, helper
  img = helper.make_tensor_value_info("img", TensorProto.UINT8, [4, 3, 32, 64])
  logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [4, 8])
  rng = np.random.default_rng(11)
  w = (rng.standard_normal((8, 3, 1, 1)) * 0.1).astype(np.float32)
  b = np.zeros(8, np.float32)
  b[0] = 2.0
  nodes = [
    helper.make_node("DequantizeLinear", ["img", "scale", "zp"], ["xf"]),
    helper.make_node("Conv", ["xf", "w", "b"], ["c"], kernel_shape=[1, 1]),
    helper.make_node("GlobalAveragePool", ["c"], ["gap"]),
    helper.make_node("Reshape", ["gap", "lshape"], ["logits"]),
  ]
  inits = [
    helper.make_tensor("scale", TensorProto.FLOAT, [1], np.array([0.5], np.float32).tobytes(), raw=True),
    helper.make_tensor("zp", TensorProto.UINT8, [1], np.array([0], np.uint8).tobytes(), raw=True),
    helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.tobytes(), raw=True),
    helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.tobytes(), raw=True),
    helper.make_tensor("lshape", TensorProto.INT64, [2], np.array([4, 8], np.int64).tobytes(), raw=True),
  ]
  graph = helper.make_graph(nodes, "tlv_classifier", [img], [logits], inits)
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
  model.ir_version = 10  # onnxruntime 1.30 refuses the onnx-default IR 14
  path = tmp_path / "classifier.onnx"
  onnx.save(model, str(path))
  return path


def test_make_classifier_falls_back_to_null_without_model():
  null = backends.make_classifier("", "")
  assert null.enabled is False
  assert null.classify(np.zeros(backends.LAMP_INPUT_SHAPE, dtype=np.uint8)) is None
  assert backends.make_classifier("onnx", "").enabled is False
  assert backends.make_classifier("tinygrad", "").enabled is False


def test_make_classifier_rejects_unknown_and_maps_dsp_to_onnx():
  with pytest.raises(backends.BackendError):
    backends.make_classifier("banana", "some.onnx")
  # WSNPE13: the lamp classifier stays on onnx CPU for the dsp backend; the
  # model check still applies, so a fake path is rejected the onnx way.
  with pytest.raises(backends.BackendError, match="model not found"):
    backends.make_classifier("dsp", "some.onnx")


def test_classifier_errors_name_the_lamp_param():
  with pytest.raises(backends.BackendError, match="CarrotTlvLampModelPath"):
    backends.OnnxCpuClassifier("")
  with pytest.raises(backends.BackendError, match="model not found"):
    backends.make_classifier("onnx", "Z:/definitely/missing.onnx")


def test_stub_classifier_votes_requested_class():
  stub = backends.StubClassifier(cls=2, prob=0.9)
  assert stub.enabled is True
  logits = stub.classify(np.zeros((4, 3, 32, 64), dtype=np.uint8))
  assert logits.shape == (4, 8) and logits.dtype == np.float32
  probs = np.exp(logits - logits.max(axis=1, keepdims=True))
  probs /= probs.sum(axis=1, keepdims=True)
  assert np.argmax(probs, axis=1).tolist() == [2, 2, 2, 2]
  assert probs[0, 2] == pytest.approx(0.9, abs=0.01)


def test_classifier_refuses_gpu_environment(tmp_path, monkeypatch):
  onnx = pytest.importorskip("onnx")
  path = _tiny_classifier_onnx_model(onnx, tmp_path)
  monkeypatch.setenv("QCOM", "1")
  with pytest.raises(backends.BackendError, match="GPU-capable"):
    backends.make_classifier("onnx", str(path))


def test_onnx_classifier_end_to_end(tmp_path):
  onnx = pytest.importorskip("onnx")
  pytest.importorskip("onnxruntime")
  path = _tiny_classifier_onnx_model(onnx, tmp_path)
  classifier = backends.make_classifier("onnx", str(path))
  assert classifier.enabled and classifier.name == "onnx"
  classifier.warmup()
  out = classifier.classify(np.zeros(backends.LAMP_INPUT_SHAPE, dtype=np.uint8))
  assert out.shape == (4, 8)
  assert float(out[0, 0]) == pytest.approx(2.0)
  classifier.close()


def test_tinygrad_classifier_end_to_end(tmp_path, monkeypatch):
  onnx = pytest.importorskip("onnx")
  if not _ensure_cc(monkeypatch, tmp_path):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")
  path = _tiny_classifier_onnx_model(onnx, tmp_path)
  classifier = backends.make_classifier("tinygrad", str(path))
  out = classifier.classify(np.zeros(backends.LAMP_INPUT_SHAPE, dtype=np.uint8))
  assert out.shape == (4, 8)
  classifier.close()


# -- detector warm-up runs the JIT capture too (W128) -------------------------

def test_tinygrad_detector_warmup_runs_the_capture_call(tmp_path, monkeypatch):
  """Construction must leave the detector JIT at cnt==2 (run + capture), so
  the first real infer() is the replay path, not the one-time capture."""
  onnx = pytest.importorskip("onnx")
  if not _ensure_cc(monkeypatch, tmp_path):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")
  path = _tiny_conv_onnx_model(onnx, tmp_path)
  backend, _decoder = backends.make_backend("tinygrad", str(path))
  try:
    key = (1, 3, 8, 8)
    jit = backend._jits[key]
    assert jit.cnt == 2, "1st call runs, 2nd captures (engine/jit.py)"
    assert jit.captured is not None
    backend.warmup()  # public hook stays a no-op once warmed
    assert jit.cnt == 2
    backend.infer(np.zeros(key, dtype=np.float32))  # 3rd call: replay
    assert jit.cnt == 3
  finally:
    backend.close()


def test_tinygrad_classifier_warmup_is_two_calls_total(tmp_path, monkeypatch):
  """The classifier must not add a third pointless call: constructor run +
  capture only, and the public warmup() is a no-op afterwards."""
  onnx = pytest.importorskip("onnx")
  if not _ensure_cc(monkeypatch, tmp_path):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")
  path = _tiny_classifier_onnx_model(onnx, tmp_path)
  classifier = backends.make_classifier("tinygrad", str(path))
  try:
    jit = classifier._jits[tuple(backends.LAMP_INPUT_SHAPE)]
    assert jit.cnt == 2
    classifier.warmup()
    assert jit.cnt == 2
    classifier.classify(np.zeros(backends.LAMP_INPUT_SHAPE, dtype=np.uint8))
    assert jit.cnt == 3
  finally:
    classifier.close()


# -- W132: onnxruntime session options + TICI fallback ------------------------

class _FakeOrtSessionOptions:
  def __init__(self):
    self.intra_op_num_threads = 0
    self.inter_op_num_threads = 0
    self.execution_mode = None
    self.graph_optimization_level = None
    self.entries = {}

  def add_session_config_entry(self, key, value):
    self.entries[key] = value


class _FakeOrtSession:
  def __init__(self, path, sess_options=None, providers=None):
    self.sess_options = sess_options
    self.providers = list(providers or [])

  def get_providers(self):
    return self.providers

  def get_inputs(self):
    return [types.SimpleNamespace(name="img", shape=[1, 3, 320, 640])]

  def get_outputs(self):
    return [types.SimpleNamespace(name="out")]

  def run(self, out_names, feed):
    return [np.zeros((1, 1), dtype=np.float32)]


def _install_fake_ort(monkeypatch):
  import types as _types
  fake = _types.ModuleType("onnxruntime")
  fake.SessionOptions = _FakeOrtSessionOptions
  fake.ExecutionMode = _types.SimpleNamespace(ORT_SEQUENTIAL=1, ORT_PARALLEL=0)
  fake.GraphOptimizationLevel = _types.SimpleNamespace(ORT_ENABLE_ALL=99)
  fake.InferenceSession = _FakeOrtSession
  fake.get_available_providers = lambda: ["CPUExecutionProvider"]
  monkeypatch.setitem(sys.modules, "onnxruntime", fake)
  return fake


def test_onnx_backend_session_options_single_thread_no_spinning(tmp_path, monkeypatch):
  _install_fake_ort(monkeypatch)
  model = tmp_path / "m.onnx"
  model.write_bytes(b"x")
  backend = backends.OnnxCpuBackend(str(model))
  opts = backend._session.sess_options
  assert opts.intra_op_num_threads == 1
  assert opts.inter_op_num_threads == 1
  assert opts.execution_mode == 1
  assert opts.graph_optimization_level == 99
  assert opts.entries["session.intra_op.allow_spinning"] == "0"
  assert opts.entries["session.inter_op.allow_spinning"] == "0"


def test_onnx_classifier_uses_single_thread_options(tmp_path, monkeypatch):
  _install_fake_ort(monkeypatch)
  model = tmp_path / "lamp.onnx"
  model.write_bytes(b"x")
  classifier = backends.OnnxCpuClassifier(str(model))
  opts = classifier._session.sess_options
  assert opts.intra_op_num_threads == 1 and opts.inter_op_num_threads == 1
  assert opts.entries["session.intra_op.allow_spinning"] == "0"


class _RunnerFakeParams:
  def __init__(self, values):
    self._values = dict(values)

  def get(self, key, block=False, return_default=False):
    return self._values.get(key)

  def get_int(self, key, block=False):
    value = self._values.get(key, 0)
    return 0 if value is None else int(value)

  def get_bool(self, key, block=False):
    return bool(self._values.get(key, False))

  def put(self, key, value, block=True):
    self._values[key] = value


def _runner_with_fake_backends(monkeypatch, tici, onnx_ok, tinygrad_ok, onnx_classifier_ok=True):
  """Build a TlvRunner with stubbed backend factories; returns (runner, error)."""
  from openpilot.selfdrive.carrot.tlv import tlvd
  monkeypatch.setattr(tlvd, "_host_is_tici", lambda: tici)
  calls = []

  def fake_make_backend(name, model_path="", decoder=None):
    calls.append(name)
    if name == "onnx" and not onnx_ok:
      raise backends.BackendError("onnxruntime is not available: boom")
    if name == "tinygrad" and not tinygrad_ok:
      raise backends.BackendError("tinygrad CPU warmup failed: boom")
    return types.SimpleNamespace(name=name, warmup=lambda: None, output_names=()), None

  def fake_make_classifier(name, model_path=""):
    if name == "onnx" and (not onnx_ok or not onnx_classifier_ok):
      raise backends.BackendError("onnxruntime lamp classifier failed: boom")
    if name == "tinygrad" and not tinygrad_ok:
      raise backends.BackendError("boom")
    return types.SimpleNamespace(name=name, enabled=True, classify=lambda x: None, warmup=lambda: None)

  monkeypatch.setattr(backends, "make_backend", fake_make_backend)
  monkeypatch.setattr(backends, "make_classifier", fake_make_classifier)
  params = _RunnerFakeParams({"CarrotTlvBackend": "onnx", "CarrotTlvModelPath": "x", "CarrotTlvLampModelPath": "y"})
  try:
    runner = tlvd.TlvRunner(params, recorder_root=None)
    return runner, calls
  except Exception as exc:  # noqa: BLE001
    return exc, calls


def test_runner_falls_back_to_tinygrad_for_both_models_on_tici(monkeypatch):
  import types as _t
  runner, calls = _runner_with_fake_backends(monkeypatch, tici=True, onnx_ok=False, tinygrad_ok=True)
  assert not isinstance(runner, Exception), runner
  assert runner.runtime_name == "tinygrad"
  assert runner.backend.name == "tinygrad"
  assert runner.classifier.name == "tinygrad"
  assert runner.backend_error == ""


def test_runner_no_fallback_on_pc(monkeypatch):
  runner, calls = _runner_with_fake_backends(monkeypatch, tici=False, onnx_ok=False, tinygrad_ok=True)
  assert not isinstance(runner, Exception), runner
  assert runner.backend is None
  assert "onnxruntime" in runner.backend_error
  assert calls == ["onnx"]  # never tried tinygrad


def test_runner_tici_onnx_classifier_failure_rebuilds_both_with_tinygrad(monkeypatch):
  # The single-class detector needs the lamp classifier for colour, so on TICI
  # an ORT classifier failure must fall back to tinygrad for BOTH models, not
  # leave tlvd detector-only.
  runner, calls = _runner_with_fake_backends(monkeypatch, tici=True, onnx_ok=True,
                                             tinygrad_ok=True, onnx_classifier_ok=False)
  assert not isinstance(runner, Exception), runner
  assert runner.runtime_name == "tinygrad"
  assert runner.backend.name == "tinygrad"
  assert runner.classifier.name == "tinygrad"
  # onnx backend built once (classifier failed), then both rebuilt as tinygrad.
  assert calls == ["onnx", "tinygrad"]


def test_runner_pc_onnx_classifier_failure_stays_detector_only(monkeypatch):
  # PC keeps the old swallow: a failed onnx classifier leaves the onnx detector
  # running detector-only (no tinygrad fallback).
  runner, calls = _runner_with_fake_backends(monkeypatch, tici=False, onnx_ok=True,
                                             tinygrad_ok=True, onnx_classifier_ok=False)
  assert not isinstance(runner, Exception), runner
  assert runner.runtime_name == "onnx"
  assert runner.backend.name == "onnx"
  assert runner.classifier.enabled is False
  assert runner.classifier_error
  assert calls == ["onnx"]


def test_runner_uses_onnx_when_available_on_tici(monkeypatch):
  runner, calls = _runner_with_fake_backends(monkeypatch, tici=True, onnx_ok=True, tinygrad_ok=True)
  assert runner.runtime_name == "onnx"
  assert runner.backend.name == "onnx" and runner.classifier.name == "onnx"


def test_runner_reports_backend_error_when_both_fail_on_tici(monkeypatch):
  runner, calls = _runner_with_fake_backends(monkeypatch, tici=True, onnx_ok=False, tinygrad_ok=False)
  assert runner.backend is None
  assert "fallback failed" in runner.backend_error
  assert calls == ["onnx", "tinygrad"]
