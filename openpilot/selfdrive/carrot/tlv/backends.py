"""Inference backends for tlvd.

CPU-only by construction. The detector refuses to run when the environment or
runtime hints at a GPU (QCOM/GPU/CL/CUDA) per CONTRACT.md; the Hexagon DSP
backend is a stub until W70 selects the runtime.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from openpilot.selfdrive.carrot.tlv.decode import Decoder, PlaceholderDecoder

GPU_ENV_KEYS = ("QCOM", "GPU", "CL", "CUDA", "OPENCL", "ROCM", "VULKAN", "TINY_GPU", "TINYGPU")
GPU_ENV_PREFIXES = ("QCOM", "CUDA", "OPENCL", "ROCM", "VULKAN", "GPU_", "TINYGPU")
# These tokens in the key name are a GPU hint regardless of the value.
GPU_STRICT_KEY_TOKENS = ("QCOM", "CUDA", "OPENCL", "ROCM", "VULKAN")
GPU_VALUE_TOKENS = ("QCOM", "CUDA", "OPENCL", "VULKAN", "ROCM")
_FALSY = ("", "0", "false", "no", "off")


class BackendError(RuntimeError):
  pass


def gpu_env_reason(environ=None) -> str | None:
  """Human reason when the environment would route work to a GPU, else None."""
  env = os.environ if environ is None else environ
  for key, value in env.items():
    text = str(value).strip()
    if not text:
      continue
    upper_key = str(key).upper()
    key_hit = upper_key in GPU_ENV_KEYS or any(upper_key.startswith(p) for p in GPU_ENV_PREFIXES)
    if key_hit:
      if any(token in upper_key for token in GPU_STRICT_KEY_TOKENS) or text.lower() not in _FALSY:
        return f"{key}={text}"
    if any(token in text.upper() for token in GPU_VALUE_TOKENS):
      return f"{key}={text}"
  return None


class NullBackend:
  """Pipeline-only backend: measures crop/convert cost without a model."""

  name = "null"

  def __init__(self, model_path: str = "") -> None:
    self.model_path = model_path

  def infer(self, model_input):
    return []

  def close(self) -> None:
    pass


class OnnxCpuBackend:
  """onnxruntime CPU-only backend (intra_op 2 threads, inter_op 1)."""

  name = "onnx"

  def __init__(self, model_path: str, intra_op_threads: int = 2, inter_op_threads: int = 1) -> None:
    reason = gpu_env_reason()
    if reason is not None:
      raise BackendError(f"refusing GPU-capable environment: {reason}")
    if not model_path:
      raise BackendError("CarrotTlvModelPath is not set")
    if not os.path.isfile(model_path):
      raise BackendError(f"model not found: {model_path}")
    try:
      import onnxruntime as ort
    except ImportError as exc:
      raise BackendError(f"onnxruntime is not available: {exc}") from exc
    if "CPUExecutionProvider" not in ort.get_available_providers():
      raise BackendError("onnxruntime has no CPUExecutionProvider")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(intra_op_threads)
    options.inter_op_num_threads = int(inter_op_threads)
    self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
    if self._session.get_providers() != ["CPUExecutionProvider"]:
      raise BackendError(f"unexpected execution providers: {self._session.get_providers()}")
    self._input_name = self._session.get_inputs()[0].name
    self.model_path = model_path

  def infer(self, model_input):
    return self._session.run(None, {self._input_name: model_input})

  def close(self) -> None:
    self._session = None


TINYGRAD_CPU_DEVICES = ("CPU", "CLANG", "LLVM")
TINYGRAD_GPU_DEVICES = ("QCOM", "GPU", "CL", "CUDA", "METAL", "AMD", "NV", "HIP", "DSP", "WEBGPU", "REMOTE")
TINYGRAD_THREAD_CAP = 2


def _tinygrad_repo_path() -> str:
  from pathlib import Path
  return str(Path(__file__).resolve().parents[4] / "tinygrad_repo")


def _import_tinygrad():
  """Import the repo's vendored tinygrad (modeld's ONNX runner) if not installed."""
  try:
    __import__("tinygrad")
  except ImportError:
    repo = _tinygrad_repo_path()
    if repo not in sys.path:
      sys.path.append(repo)
    __import__("tinygrad")
  from tinygrad import Device, Tensor
  from tinygrad.engine.jit import TinyJit
  from tinygrad.nn.onnx import OnnxRunner
  return Device, Tensor, TinyJit, OnnxRunner


class TinygradCpuBackend:
  """tinygrad backend for the C3X, where onnxruntime is not a device dependency.

  Runs selfdrive/modeld's tinygrad ONNX runner forced onto a CPU backend
  (CPU/CLANG/LLVM) with <= 2 kernel threads, and refuses to run when only
  GPU-ish devices (QCOM/GPU/CL/CUDA/...) exist. The JIT compiles once here,
  off the tlvd hot path (lazily per input shape when dims are dynamic)."""

  name = "tinygrad"

  def __init__(self, model_path: str, max_threads: int = TINYGRAD_THREAD_CAP) -> None:
    reason = gpu_env_reason()
    if reason is not None:
      raise BackendError(f"refusing GPU-capable environment: {reason}")
    if not model_path:
      raise BackendError("CarrotTlvModelPath is not set")
    if not os.path.isfile(model_path):
      raise BackendError(f"model not found: {model_path}")
    Device, Tensor, TinyJit, OnnxRunner = _import_tinygrad()
    self.device_name = self._pick_cpu_device(Device)
    self.max_threads = max(1, min(int(max_threads), TINYGRAD_THREAD_CAP))
    try:
      from tinygrad.helpers import CPU_COUNT
      CPU_COUNT.value = self.max_threads
    except Exception:
      pass
    runner = OnnxRunner(model_path)
    runner.to(self.device_name)
    self._runner = runner
    self._Tensor = Tensor
    self._TinyJit = TinyJit
    self._input_name = next(iter(runner.graph_inputs))
    self._input_spec = runner.graph_inputs[self._input_name]
    self._output_names = tuple(runner.graph_outputs)
    self._jits: dict[tuple, object] = {}
    self.model_path = model_path
    self._warmup()

  @staticmethod
  def _pick_cpu_device(Device) -> str:
    for name in TINYGRAD_CPU_DEVICES:
      try:
        Device[name]
        return name
      except Exception:
        continue
    raise BackendError("tinygrad has no CPU backend (GPU-only environment)")

  def _assert_cpu(self, tensor) -> None:
    base = str(tensor.device).split(":")[0]
    if base in TINYGRAD_GPU_DEVICES or base not in TINYGRAD_CPU_DEVICES:
      raise BackendError(f"tinygrad resolved to non-CPU device {base!r}")

  def _run_raw(self, x):
    outs = self._runner({self._input_name: x})
    return tuple(outs[name] for name in self._output_names)

  def _make_tensor(self, model_input):
    dtype = None
    try:
      dtype = self._input_spec.dtype.np
    except Exception:
      dtype = None
    data = np.asarray(model_input)
    if dtype is not None:
      data = data.astype(dtype, copy=False)
    return self._Tensor(data, device=self.device_name)

  def _call(self, model_input):
    x = self._make_tensor(model_input)
    self._assert_cpu(x)
    key = tuple(x.shape)
    jit = self._jits.get(key)
    if jit is None:
      jit = self._TinyJit(self._run_raw, prune=True)
      self._jits[key] = jit
    result = []
    for out in jit(x):
      self._assert_cpu(out)
      result.append(out.numpy())
    return result

  def _warmup(self) -> None:
    shape = tuple(self._input_spec.shape)
    if not all(isinstance(d, int) for d in shape):
      return  # dynamic dims: the first real frame compiles once
    try:
      self._call(np.zeros(shape, dtype=np.float32))
    except BackendError:
      raise
    except Exception as exc:
      raise BackendError(f"tinygrad CPU warmup failed: {exc}") from exc

  def infer(self, model_input):
    return self._call(model_input)

  def close(self) -> None:
    self._jits.clear()
    self._runner = None


class DspBackend:
  """Hexagon DSP backend placeholder; W70 decides the runtime."""

  name = "dsp"

  def __init__(self, model_path: str = "") -> None:
    raise NotImplementedError("DSP backend is not implemented yet; W70 decides the Hexagon runtime")
def make_backend(name: str, model_path: str = "", decoder: Decoder | None = None):
  key = (name or "").strip().lower()
  if key in ("", "null", "none"):
    return NullBackend(model_path), decoder or PlaceholderDecoder()
  if key == "onnx":
    return OnnxCpuBackend(model_path), decoder or PlaceholderDecoder()
  if key == "tinygrad":
    return TinygradCpuBackend(model_path), decoder or PlaceholderDecoder()
  if key == "dsp":
    return DspBackend(model_path), decoder or PlaceholderDecoder()
  raise BackendError(f"unknown backend: {name!r}")
