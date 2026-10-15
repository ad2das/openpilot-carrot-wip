"""Unit test for the ARM64 runtime check's pure compare logic (W132).

Runs the compare() function on synthetic ORT/tinygrad outputs, so the parity
rules (detector max|delta|, lamp max|delta| + argmax) are exercised without an
arm64 host or the real models.
"""
import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tlv" / "tools" / "arm64_runtime_check.py"
_spec = importlib.util.spec_from_file_location("arm64_runtime_check", _MODULE_PATH)
armcheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(armcheck)


def _det(level, value=0.0):
  return [np.full((1, 4), value, np.float32) for _ in range(level)]


def test_compare_passes_on_identical_outputs():
  det = _det(6, 0.5)
  lamp = np.eye(8, dtype=np.float32)[:4]
  ok, msg = armcheck.compare(det, det, lamp, lamp)
  assert ok, msg


def test_compare_passes_within_tolerance():
  det = _det(6, 0.5)
  det2 = [a + 5e-5 for a in det]
  lamp = np.eye(8, dtype=np.float32)[:4]
  ok, _ = armcheck.compare(det, det2, lamp, lamp + 5e-5)
  assert ok


def test_compare_fails_on_large_detector_delta():
  det = _det(6, 0.5)
  det2 = [a + 1e-2 for a in det]
  lamp = np.eye(8, dtype=np.float32)[:4]
  ok, msg = armcheck.compare(det, det2, lamp, lamp)
  assert not ok and "detector max|delta|" in msg


def test_compare_fails_on_lamp_argmax_mismatch():
  det = _det(6, 0.5)
  lamp = np.eye(8, dtype=np.float32)[:4]
  lamp2 = lamp.copy()
  # Same max value within tolerance but a different argmax (near-tie).
  lamp2[0, 0] = np.float32(1.0 + 5e-5)
  lamp2[0, 1] = np.float32(1.0)
  lamp[0, 0] = np.float32(1.0)
  lamp[0, 1] = np.float32(1.0 + 5e-5)
  ok, msg = armcheck.compare(det, det, lamp, lamp2)
  assert not ok and "argmax" in msg


def test_compare_fails_on_output_count_mismatch():
  ok, msg = armcheck.compare(_det(6), _det(5), np.zeros((4, 8), np.float32), np.zeros((4, 8), np.float32))
  assert not ok and "count" in msg
