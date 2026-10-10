#!/usr/bin/env python3
"""PC-only soak runner for the TLV pipeline (RESOURCE_BUDGET.md tasks 4/6/10).

Runs the full PC pipeline (synthetic NV12 -> FramePipeline -> tinygrad CPU or
null backend) under the real DutyCycleGovernor until --minutes or --max-frames
is reached (whichever first), samples RSS every --sample-s seconds and prints a
JSON summary: start/end/max RSS, growth, and the measured process CPU versus
the 40%-of-one-core governor target.

Developer tool for the Windows workstation: it forces the tinygrad CPU backend
(never the GPU) and sets CC to a ziglang wrapper when no clang is available
(same technique as the tests). Requires numpy, and psutil for RSS.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

from openpilot.selfdrive.carrot.tlv import backends, crops, pipeline, resources

W, H = 1928, 1208
CPU_BUDGET_PCT = 40.0


def _synthetic_frame(stride_pad: int = 64):
  stride = W + stride_pad
  rng = np.random.default_rng(11)
  y = rng.integers(0, 255, size=(H, stride), dtype=np.uint8)
  uv = rng.integers(0, 255, size=(H // 2, stride), dtype=np.uint8)
  buf = np.concatenate([y.reshape(-1), uv.reshape(-1)]).tobytes()
  return buf, stride, stride * H


def _ensure_cc(tmp_dir: str) -> None:
  """tinygrad's CPU kernels need a clang-compatible CC (ziglang fallback)."""
  if shutil.which("clang"):
    return
  try:
    import ziglang  # noqa: F401
  except ImportError as exc:
    raise SystemExit("no clang on PATH and no ziglang installed for tinygrad CPU kernels") from exc
  driver = os.path.join(tmp_dir, "zigcc_driver.py")
  with open(driver, "w") as f:
    f.write("\n".join([
      "import subprocess, sys",
      "args = [a.replace('-none-unknown-elf', '-linux-gnu') for a in sys.argv[1:]]",
      "args.append('-g0')",
      "sys.exit(subprocess.run([sys.executable, '-m', 'ziglang', 'cc', *args],",
      "                        input=sys.stdin.buffer.read()).returncode)",
    ]) + "\n")
  wrapper = os.path.join(tmp_dir, "zigcc.cmd")
  with open(wrapper, "w") as f:
    f.write(f'@"{sys.executable}" "{driver}" %*\n')
  os.environ["CC"] = wrapper


def _stub_conv_onnx(path: str) -> None:
  import onnx
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
  model.ir_version = 10
  onnx.save(model, path)


def _rss_mb(proc) -> float:
  return proc.memory_info().rss / resources.MB


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description="PC soak for the carrot TLV pipeline (CPU only)")
  parser.add_argument("--minutes", type=float, default=30.0)
  parser.add_argument("--max-frames", type=int, default=50000)
  parser.add_argument("--sample-s", type=float, default=10.0)
  parser.add_argument("--backend", default="tinygrad", choices=("tinygrad", "null"))
  parser.add_argument("--model", default=None, help="ONNX model path; a tiny stub conv is generated when omitted")
  parser.add_argument("--warmup-frames", type=int, default=2000, help="pipeline warm-up before the measured soak")
  args = parser.parse_args(argv)

  import psutil
  proc = psutil.Process(os.getpid())

  tmp_dir = tempfile.mkdtemp(prefix="tlv_soak_")
  model_path = args.model
  if args.backend == "tinygrad":
    _ensure_cc(tmp_dir)
    if model_path is None:
      model_path = os.path.join(tmp_dir, "stub_conv.onnx")
      _stub_onnx_import = __import__("onnx")  # fail fast with a clear error
      del _stub_onnx_import
      _stub_conv_onnx(model_path)
  backend, decoder = backends.make_backend(args.backend, model_path or "")
  pipe = pipeline.FramePipeline(backend, decoder)
  data, stride, uv_offset = _synthetic_frame()
  far = crops.far_geometry(600.0, W, H, 2648.0)
  near = crops.near_geometry(600.0, W, H, 2648.0)

  def drive_one(i: int) -> None:
    far_view = i % 2 == 0
    pipe.process(data, W, H, stride, uv_offset, far if far_view else near,
                 view="far" if far_view else "near", image_width=W, now=time.monotonic())

  governor = resources.DutyCycleGovernor()

  for i in range(args.warmup_frames):
    if governor.ready():
      governor.begin_frame()
      try:
        drive_one(i)
      finally:
        governor.end_frame()
    else:
      time.sleep(0.001)

  frames = 0
  samples = []
  errors = 0
  cpu_start = resources.os_cpu_seconds()
  t_measure = time.monotonic()
  rss_start = _rss_mb(proc)
  next_sample = t_measure
  deadline = t_measure + args.minutes * 60.0

  while True:
    now = time.monotonic()
    if now >= deadline or frames >= args.max_frames:
      break
    if governor.ready(now):
      governor.begin_frame(now)
      try:
        drive_one(args.warmup_frames + frames)
        frames += 1
      except Exception:
        errors += 1
      finally:
        governor.end_frame()
    else:
      time.sleep(min(0.005, max(0.0, deadline - now)))
    if now >= next_sample:
      next_sample = now + args.sample_s
      samples.append({
        "t": round(now - t_measure, 1),
        "rss_mb": round(_rss_mb(proc), 1),
        "cpu_s": resources.os_cpu_seconds(),
      })

  t_end = time.monotonic()
  cpu_end = resources.os_cpu_seconds()
  wall = t_end - t_measure
  rss_end = _rss_mb(proc)
  rss_series = [s["rss_mb"] for s in samples] or [rss_end]
  cpu_series = samples + [{"t": round(wall, 1), "rss_mb": rss_end, "cpu_s": cpu_end}]
  window_pcts = []
  for a, b in zip(cpu_series, cpu_series[1:], strict=False):
    dt = b["t"] - a["t"]
    if dt > 0:
      window_pcts.append(100.0 * (b["cpu_s"] - a["cpu_s"]) / dt)
  summary = {
    "backend": args.backend,
    "model": model_path,
    "warmup_frames": args.warmup_frames,
    "frames": frames,
    "errors": errors,
    "wall_s": round(wall, 1),
    "governor_rate_hz_end": round(governor.rate_hz, 3),
    "governor_duty_end": None if governor.window_duty is None else round(governor.window_duty, 4),
    "rss_start_mb": round(rss_start, 1),
    "rss_end_mb": round(rss_end, 1),
    "rss_max_mb": round(max(rss_series), 1),
    "rss_growth_mb": round(rss_end - rss_start, 1),
    "cpu_pct_avg": round(100.0 * (cpu_end - cpu_start) / wall, 2) if wall > 0 else None,
    "cpu_pct_max_window": round(max(window_pcts), 2) if window_pcts else None,
    "cpu_budget_pct": CPU_BUDGET_PCT,
    "cpu_within_budget": bool(window_pcts) and max(window_pcts) <= CPU_BUDGET_PCT + 1.0,
    "samples": samples,
  }
  print(json.dumps(summary, indent=2))
  backend.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
