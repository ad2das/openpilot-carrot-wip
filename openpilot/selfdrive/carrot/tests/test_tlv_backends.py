"""Tests for TLV inference backends: CPU-only refusal guards and ONNX stub model."""
import os
import shutil
import sys
import time

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


def test_dsp_backend_is_a_clear_stub():
  with pytest.raises(NotImplementedError, match="W70"):
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
  assert tlvd.resolve_backend_name("", False) == "onnx"
  assert tlvd.resolve_backend_name("onnx", False) == "onnx"
  assert tlvd.resolve_backend_name("", True) == "tinygrad"
  assert tlvd.resolve_backend_name("onnx", True) == "tinygrad"
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


def test_make_classifier_rejects_unknown_and_dsp_model():
  with pytest.raises(backends.BackendError):
    backends.make_classifier("banana", "some.onnx")
  with pytest.raises(NotImplementedError, match="W70"):
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
