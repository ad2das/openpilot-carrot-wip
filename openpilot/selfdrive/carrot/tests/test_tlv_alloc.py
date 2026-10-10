"""Hot-path allocation proof for the TLV pipeline (RESOURCE_BUDGET.md: Memory).

Runs 2000 synthetic NV12 frames through the full PC pipeline
(FramePreprocessor -> backend -> decoder -> tracker -> state machine) and asserts
that net traced-memory growth after warm-up is < 256 KB. Two backends:
  - NullBackend (pure crop/convert path),
  - tinygrad CPU on a tiny ONNX stub (same ziglang CC trick as test_tlv_backends).
The measured numbers are printed for the record.
"""
from __future__ import annotations

import gc
import shutil
import tracemalloc

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import backends, crops, pipeline

W, H = 1928, 1208
FRAMES = 2000
WARMUP = 100            # null backend: nothing lazily constructed per frame
WARMUP_TINYGRAD = 2000  # tinygrad lazily builds caches over the first ~1600 frames
TRACE_WARMUP = 500      # frames traced before the baseline (tracemalloc itself amortizes here)
MAX_GROWTH_BYTES = 256 * 1024


def _synthetic_frame(stride_pad: int = 64):
  stride = W + stride_pad
  rng = np.random.default_rng(11)
  y = rng.integers(0, 255, size=(H, stride), dtype=np.uint8)
  uv = rng.integers(0, 255, size=(H // 2, stride), dtype=np.uint8)
  buf = np.concatenate([y.reshape(-1), uv.reshape(-1)]).tobytes()
  return buf, stride, stride * H


def _drive(pipe: pipeline.FramePipeline, data, stride: int, uv_offset: int, frames: int, t0: float, start: int = 0) -> None:
  far = crops.far_geometry(600.0, W, H, 2648.0)
  near = crops.near_geometry(600.0, W, H, 2648.0)
  for i in range(start, start + frames):
    far_view = i % 2 == 0
    geom = far if far_view else near
    pipe.process(data, W, H, stride, uv_offset, geom, view="far" if far_view else "near",
                 image_width=W, path_x=(), path_y=(), left_intent=False, now=t0 + i)


def _measure(pipe: pipeline.FramePipeline, data, stride: int, uv_offset: int, warmup: int = WARMUP) -> tuple[float, float]:
  _drive(pipe, data, stride, uv_offset, warmup, 0.0)
  gc.collect()
  tracemalloc.start()
  # tracemalloc amortizes its own tracing tables over the first few hundred
  # frames; run those before taking the baseline so the number is the pipeline.
  _drive(pipe, data, stride, uv_offset, TRACE_WARMUP, 0.0, start=warmup)
  gc.collect()
  before = tracemalloc.get_traced_memory()[0]
  _drive(pipe, data, stride, uv_offset, FRAMES, 0.0, start=warmup + TRACE_WARMUP)
  gc.collect()  # count retained memory, not uncollected cyclic garbage
  current, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  gc.collect()
  return current - before, peak - before


def test_null_backend_no_allocation_growth():
  data, stride, uv_offset = _synthetic_frame()
  backend, decoder = backends.make_backend("null", "")
  pipe = pipeline.FramePipeline(backend, decoder)
  growth, peak = _measure(pipe, data, stride, uv_offset)
  print(f"\nalloc[null]: frames={FRAMES} net_growth={growth / 1024:.1f} KB peak_delta={peak / 1024:.1f} KB")
  assert growth < MAX_GROWTH_BYTES, f"net growth {growth / 1024:.1f} KB >= 256 KB"


def _ensure_cc(monkeypatch, tmp_path) -> bool:
  if shutil.which("clang"):
    return True
  try:
    import ziglang  # noqa: F401
  except ImportError:
    return False
  import sys
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


def _stub_conv_onnx(onnx, tmp_path):
  """Tiny contract-shaped model: uint8 input -> dequant -> conv stride 8 -> relu
  -> maxpool -> reshape (1,4) -> +offset = one box; constant score/class."""
  from onnx import TensorProto, helper
  rng = np.random.default_rng(1)
  w = (rng.standard_normal((4, 3, 3, 3)) * 0.1).astype(np.float32)
  b = np.array([0.1, -0.2, 0.3, 0.0], np.float32)
  img = helper.make_tensor_value_info("img", TensorProto.UINT8, [1, 3, 320, 640])
  boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 4])
  scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1])
  classes = helper.make_tensor_value_info("classes", TensorProto.INT64, [1])
  nodes = [
    helper.make_node("DequantizeLinear", ["img", "scale", "zp"], ["xf"]),
    helper.make_node("Conv", ["xf", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[8, 8]),
    helper.make_node("Relu", ["c"], ["r"]),
    helper.make_node("MaxPool", ["r"], ["p"], kernel_shape=[40, 80], strides=[40, 80]),
    helper.make_node("Reshape", ["p", "shape4"], ["flat"]),
    helper.make_node("Add", ["flat", "box_add"], ["boxes"]),
    helper.make_node("Constant", [], ["scores"], value=helper.make_tensor("scores_v", TensorProto.FLOAT, [1], [0.9])),
    helper.make_node("Constant", [], ["classes"], value=helper.make_tensor("classes_v", TensorProto.INT64, [1], [2])),
  ]
  graph = helper.make_graph(nodes, "tlv_stub_conv", [img], [boxes, scores, classes], [
    helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.tobytes(), raw=True),
    helper.make_tensor("b", TensorProto.FLOAT, [4], b.tobytes(), raw=True),
    helper.make_tensor("scale", TensorProto.FLOAT, [1], np.array([0.5], np.float32).tobytes(), raw=True),
    helper.make_tensor("zp", TensorProto.UINT8, [1], np.array([0], np.uint8).tobytes(), raw=True),
    helper.make_tensor("shape4", TensorProto.INT64, [2], np.array([1, 4], np.int64).tobytes(), raw=True),
    helper.make_tensor("box_add", TensorProto.FLOAT, [4], np.array([100.0, 50.0, 140.0, 90.0], np.float32).tobytes(), raw=True),
  ])
  model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
  model.ir_version = 10  # onnxruntime 1.30 refuses the onnx-default IR 14
  path = tmp_path / "tlv_stub_conv.onnx"
  onnx.save(model, str(path))
  return path


def test_tinygrad_cpu_backend_no_allocation_growth(tmp_path, monkeypatch):
  onnx = pytest.importorskip("onnx")
  if not _ensure_cc(monkeypatch, tmp_path):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")
  # The backend pins DEV to CPU itself (backends.py); delenv is defence in
  # depth so no inherited value can reach this allocation-only test.
  monkeypatch.delenv("DEV", raising=False)
  path = _stub_conv_onnx(onnx, tmp_path)
  backend, decoder = backends.make_backend("tinygrad", str(path))
  assert backend.device_name in backends.TINYGRAD_CPU_DEVICES
  pipe = pipeline.FramePipeline(backend, decoder)
  data, stride, uv_offset = _synthetic_frame()
  growth, peak = _measure(pipe, data, stride, uv_offset, warmup=WARMUP_TINYGRAD)
  print(f"\nalloc[tinygrad-cpu]: frames={FRAMES} net_growth={growth / 1024:.1f} KB peak_delta={peak / 1024:.1f} KB tracker_frames={pipe.frame_idx}")
  backend.close()
  assert pipe.frame_idx == WARMUP_TINYGRAD + TRACE_WARMUP + FRAMES
  assert growth < MAX_GROWTH_BYTES, f"net growth {growth / 1024:.1f} KB >= 256 KB"
