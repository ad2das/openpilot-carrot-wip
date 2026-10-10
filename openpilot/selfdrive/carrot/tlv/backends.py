"""Inference backends for tlvd.

CPU-only by construction. The detector refuses to run when the environment or
runtime hints at a GPU (QCOM/GPU/CL/CUDA) per CONTRACT.md; the Hexagon DSP
backend is a stub until W70 selects the runtime.
"""
from __future__ import annotations

import os

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
  if key == "dsp":
    return DspBackend(model_path), decoder or PlaceholderDecoder()
  raise BackendError(f"unknown backend: {name!r}")
