"""Tests for TLV inference backends: CPU-only refusal guards and ONNX stub model."""
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
