#!/usr/bin/env python3
"""ARM64 runtime check for the tlvd inference backends (W132).

Runs on the free ubuntu-24.04-arm GitHub runner (and any arm64 host). It imports
ONLY the tlv backend code path (no cereal/msgq), so it needs just numpy plus the
bundled onnxruntime wheel.

What it does:
  (a) builds the "onnx" detector + lamp exactly as tlvd does (bundled models,
      the single-thread / no-spinning session options);
  (b) builds the "tinygrad" detector + lamp forced onto a CPU device;
  (c) on 8 seeded random uint8 inputs (np.random.default_rng(0)) plus the zero
      input: detector raw outputs ORT vs tinygrad max|delta| <= 1e-4, decoded
      box lists equal, lamp logits max|delta| <= 1e-4 and argmax equal;
  (d) CPU-time mean (os.times user+sys over 50 runs, 1 thread) for ORT and
      tinygrad detector + lamp, and prints the ARM ratio and the predicted C3X
      ORT detect time.

Exit 0 only if every comparison passes. Any import/parity failure exits 1 so
the workflow fails loudly.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# Importing the package path works both from the repo root (PYTHONPATH=$PWD)
# and installed; the tlv modules are pure numpy + tinygrad, no cereal/msgq.
from openpilot.selfdrive.carrot.tlv import backends

DETECTOR_INPUT_SHAPE = (1, 3, 320, 640)
LAMP_INPUT_SHAPE = tuple(backends.LAMP_INPUT_SHAPE)
N_INPUTS = 8
PARITY_TOL = 1e-4
TIMING_RUNS = 50
# W131 measured 752 ms per detect on the C3X with tinygrad; the ratio of
# tinygrad/ORT on ARM scales that prediction.
W126_TINYGRAD_DETECT_MS = 752.0


def _seeded_inputs(shape, count: int) -> list[np.ndarray]:
  rng = np.random.default_rng(0)
  inputs = [np.zeros(shape, dtype=np.uint8)]
  for _ in range(count):
    inputs.append(rng.integers(0, 256, size=shape, dtype=np.uint8))
  return inputs


def _decode(dets_raw, view: str) -> list:
  """Decode raw outputs through the real decoder, normalizing to tuples."""
  out = []
  for det in dets_raw:
    out.append((int(det.cls), round(float(det.score), 4), round(float(det.x1), 3),
                round(float(det.y1), 3), round(float(det.x2), 3), round(float(det.y2), 3)))
  return out


def compare(onnx_det_raw, tiny_det_raw, onnx_lamp_logits, tiny_lamp_logits) -> tuple[bool, str]:
  """Pure parity comparison on already-computed outputs (unit-tested on fakes).

  Returns (ok, message). Detector: per-level max|delta| <= PARITY_TOL.
  Lamp: max|delta| <= PARITY_TOL and identical argmax per row.
  """
  if len(onnx_det_raw) != len(tiny_det_raw):
    return False, f"detector output count {len(onnx_det_raw)} != {len(tiny_det_raw)}"
  worst = 0.0
  for a, b in zip(onnx_det_raw, tiny_det_raw):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
      return False, f"detector shape {a.shape} != {b.shape}"
    worst = max(worst, float(np.max(np.abs(a - b))))
  if worst > PARITY_TOL:
    return False, f"detector max|delta| {worst:.3e} > {PARITY_TOL:.0e}"

  oa = np.asarray(onnx_lamp_logits, dtype=np.float64)
  ta = np.asarray(tiny_lamp_logits, dtype=np.float64)
  if oa.shape != ta.shape:
    return False, f"lamp shape {oa.shape} != {ta.shape}"
  lamp_delta = float(np.max(np.abs(oa - ta)))
  if lamp_delta > PARITY_TOL:
    return False, f"lamp max|delta| {lamp_delta:.3e} > {PARITY_TOL:.0e}"
  if not np.array_equal(np.argmax(oa, axis=-1), np.argmax(ta, axis=-1)):
    return False, "lamp argmax differs"

  return True, f"detector max|delta| {worst:.3e}, lamp max|delta| {lamp_delta:.3e}, argmax equal"


def _cpu_mean_ms(fn, runs: int) -> float:
  """Mean per-call CPU time (user+sys) in ms over `runs` calls."""
  t0 = os.times()
  for _ in range(runs):
    fn()
  t1 = os.times()
  cpu = (t1.user - t0.user) + (t1.system - t0.system)
  return 1000.0 * cpu / runs


def _wall_p50_p95_ms(fn, runs: int) -> tuple[float, float]:
  samples = []
  for _ in range(runs):
    start = time.perf_counter()
    fn()
    samples.append((time.perf_counter() - start) * 1000.0)
  arr = np.asarray(samples, dtype=np.float64)
  return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def main() -> int:
  try:
    import onnxruntime as ort
  except ImportError as exc:
    print(f"arm64_runtime_check: onnxruntime import failed: {exc!r}", file=sys.stderr)
    return 1
  print(f"arm64_runtime_check: onnxruntime {ort.__version__} at {ort.__file__}")

  onnx_det, _onnx_dec = backends.make_backend("onnx", backends.BUNDLED_DETECTOR)
  onnx_cls = backends.make_classifier("onnx", backends.BUNDLED_LAMP)
  tiny_det, _tiny_dec = backends.make_backend("tinygrad", backends.BUNDLED_DETECTOR)
  tiny_cls = backends.make_classifier("tinygrad", backends.BUNDLED_LAMP)

  det_inputs = _seeded_inputs(DETECTOR_INPUT_SHAPE, N_INPUTS)
  lamp_inputs = _seeded_inputs(LAMP_INPUT_SHAPE, N_INPUTS)

  ok = True
  for i, x in enumerate(det_inputs):
    o = onnx_det.infer(x)
    t = tiny_det.infer(x)
    passed, msg = compare(o, t, onnx_cls.classify(lamp_inputs[i]), tiny_cls.classify(lamp_inputs[i]))
    print(f"arm64_runtime_check: input {i} parity: {'OK' if passed else 'FAIL'} ({msg})")
    ok = ok and passed
    if not passed:
      print("arm64_runtime_check: PARITY FAILED", file=sys.stderr)
      return 1

  x_det = det_inputs[0]
  x_lamp = lamp_inputs[0]
  ort_det_ms = _cpu_mean_ms(lambda: onnx_det.infer(x_det), TIMING_RUNS)
  tiny_det_ms = _cpu_mean_ms(lambda: tiny_det.infer(x_det), TIMING_RUNS)
  ort_lamp_ms = _cpu_mean_ms(lambda: onnx_cls.classify(x_lamp), TIMING_RUNS)
  tiny_lamp_ms = _cpu_mean_ms(lambda: tiny_cls.classify(x_lamp), TIMING_RUNS)
  ort_det_p50, ort_det_p95 = _wall_p50_p95_ms(lambda: onnx_det.infer(x_det), TIMING_RUNS)

  ratio = (tiny_det_ms / ort_det_ms) if ort_det_ms > 0.0 else float("inf")
  predicted = (W126_TINYGRAD_DETECT_MS / ratio) if ratio > 0.0 else float("inf")
  print(f"arm64_runtime_check: ORT detect CPU mean {ort_det_ms:.2f} ms (wall p50 {ort_det_p50:.2f} / p95 {ort_det_p95:.2f})")
  print(f"arm64_runtime_check: tinygrad detect CPU mean {tiny_det_ms:.2f} ms")
  print(f"arm64_runtime_check: ORT lamp CPU mean {ort_lamp_ms:.2f} ms ; tinygrad lamp {tiny_lamp_ms:.2f} ms")
  print(f"arm64_runtime_check: ARM ratio tinygrad/ORT = {ratio:.2f}")
  print(f"arm64_runtime_check: predicted C3X ORT detect ms = {predicted:.1f}")

  onnx_det.close(); onnx_cls.close(); tiny_det.close(); tiny_cls.close()
  print("arm64_runtime_check: PASS" if ok else "arm64_runtime_check: FAIL")
  return 0 if ok else 1


if __name__ == "__main__":
  raise SystemExit(main())
