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

# Bundled R3b traffic-light models: shipped so CarrotTlvMode works with no
# model path params. Absolute paths resolved next to this module; the SHA256
# values are the verified artifacts, checked by the unit tests only (tlvd never
# hashes a model at startup).
_BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BUNDLED_DETECTOR = os.path.join(_BUNDLED_DIR, "tl_detector_r3b.onnx")
BUNDLED_LAMP = os.path.join(_BUNDLED_DIR, "tl_lamp_r3b.onnx")
BUNDLED_DETECTOR_SHA256 = "d106b3f9f9d100ecdd71bd743683eac21f7b06b8e5ff2abd96841e934fb8db09"
BUNDLED_LAMP_SHA256 = "467f7a1f25c5de1dd1e50280ca23f0d8abc3e9d72857af9bb044646f5b6f737f"


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

  def __init__(self, model_path: str, intra_op_threads: int = 2, inter_op_threads: int = 1,
               param_name: str = "CarrotTlvModelPath") -> None:
    reason = gpu_env_reason()
    if reason is not None:
      raise BackendError(f"refusing GPU-capable environment: {reason}")
    if not model_path:
      raise BackendError(f"{param_name} is not set")
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
# Host-side plumbing devices tinygrad may open without an accelerator
# (tinygrad_repo/tinygrad/device.py:25 allow-list).
TINYGRAD_HOST_DEVICES = ("NPY", "DISK", "TINYFS", "PYTHON")
TINYGRAD_THREAD_CAP = 2
# tinygrad DTypes have no numpy accessor (dtypes.uchar.np does not exist), so
# map the ONNX input dtype names onto numpy for the input cast.
TINYGRAD_NP_DTYPES = {
  "bool": np.bool_,
  "uchar": np.uint8,
  "char": np.int8,
  "ushort": np.uint16,
  "short": np.int16,
  "uint": np.uint32,
  "int": np.int32,
  "ulong": np.uint64,
  "long": np.int64,
  "half": np.float16,
  "float": np.float32,
  "double": np.float64,
}


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
  # The submodules below can read Device.DEFAULT at import time; an unset or
  # stale-empty DEV would probe and open the first available device
  # (device.py:42,45-51). Pin the context before importing them.
  _force_cpu_dev_context()
  from tinygrad import Device, Tensor
  from tinygrad.engine.jit import TinyJit
  from tinygrad.nn.onnx import OnnxRunner
  return Device, Tensor, TinyJit, OnnxRunner


def _force_cpu_dev_env() -> None:
  """Set DEV=CPU in the environment before tinygrad (re)imports it.

  device.py:42 falls back to _select_device when DEV is empty; that probe walks
  ALL_DEVICES (device.py:14: CUDA/QCOM before CPU), opens the first device that
  exists and writes it into os.environ (device.py:51) - on TICI the QCOM GPU
  next to modeld. Unconditional: an inherited QCOM/CUDA value must never reach
  tinygrad device selection."""
  os.environ["DEV"] = "CPU"


def _force_cpu_dev_context() -> None:
  """Pin DEV=CPU for an already-imported tinygrad (env is only read at import)."""
  try:
    from tinygrad.helpers import DEV as tinygrad_dev
    tinygrad_dev.value = "CPU"
  except Exception as exc:
    raise BackendError(f"cannot pin tinygrad DEV to CPU: {exc}") from exc


def _assert_only_cpu_devices_opened(Device) -> None:
  """Raise unless every device opened so far is a CPU renderer or host staging.

  tinygrad records each instantiated device in _opened_devices
  (tinygrad_repo/tinygrad/device.py:18,34)."""
  for dev in getattr(Device, "_opened_devices", ()):
    base = str(dev).split(":")[0]
    if base in TINYGRAD_GPU_DEVICES or (base not in TINYGRAD_CPU_DEVICES and base not in TINYGRAD_HOST_DEVICES):
      raise BackendError(f"tinygrad opened a non-CPU device: {dev}")


class TinygradCpuBackend:
  """tinygrad backend for the C3X, where onnxruntime is not a device dependency.

  Runs selfdrive/modeld's tinygrad ONNX runner forced onto a CPU backend
  (CPU/CLANG/LLVM) with <= 2 kernel threads, and refuses to run when only
  GPU-ish devices (QCOM/GPU/CL/CUDA/...) exist. The JIT compiles once here,
  off the tlvd hot path (lazily per input shape when dims are dynamic)."""

  name = "tinygrad"

  def __init__(self, model_path: str, max_threads: int = TINYGRAD_THREAD_CAP,
               param_name: str = "CarrotTlvModelPath") -> None:
    # CPU default FIRST: OnnxRunner loads weights on Device.DEFAULT, which with
    # DEV unset (or inherited QCOM/CUDA) auto-opens and writes the first
    # available device (tinygrad_repo/tinygrad/device.py:14,45-51).
    _force_cpu_dev_env()
    reason = gpu_env_reason()
    if reason is not None:
      raise BackendError(f"refusing GPU-capable environment: {reason}")
    if not model_path:
      raise BackendError(f"{param_name} is not set")
    if not os.path.isfile(model_path):
      raise BackendError(f"model not found: {model_path}")
    Device, Tensor, TinyJit, OnnxRunner = _import_tinygrad()
    # A tinygrad imported earlier in the process does not re-read the env:
    # pin its DEV context and verify the resolution before any weight tensor.
    _force_cpu_dev_context()
    default = str(Device.DEFAULT).split(":")[0]
    if default not in TINYGRAD_CPU_DEVICES:
      raise BackendError(f"tinygrad default device resolved to {default!r}, not CPU")
    self.max_threads = max(1, min(int(max_threads), TINYGRAD_THREAD_CAP))
    # CPU_COUNT must be lowered before the CPU device is created: the renderers
    # capture it as a class attribute at import time
    # (tinygrad_repo/tinygrad/renderer/cstyle.py:252 global_max = (CPU_COUNT.value, 0, 0)).
    self._cap_tinygrad_threads()
    self.device_name = self._pick_cpu_device(Device)
    runner = OnnxRunner(model_path)
    runner.to(self.device_name)
    # Weight streaming above runs on Device.DEFAULT; no GPU may have opened.
    _assert_only_cpu_devices_opened(Device)
    self._runner = runner
    self._Tensor = Tensor
    self._TinyJit = TinyJit
    self._input_name = next(iter(runner.graph_inputs))
    self._input_spec = runner.graph_inputs[self._input_name]
    self._output_names = tuple(runner.graph_outputs)
    self._jits: dict[tuple, object] = {}
    self.model_path = model_path
    self._warmup()
    _assert_only_cpu_devices_opened(Device)

  @staticmethod
  def _cap_tinygrad_threads() -> None:
    try:
      from tinygrad.helpers import CPU_COUNT
      CPU_COUNT.value = TINYGRAD_THREAD_CAP
    except Exception:
      pass

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
    spec_dtype = getattr(self._input_spec, "dtype", None)
    dtype = TINYGRAD_NP_DTYPES.get(str(spec_dtype).rsplit(".", 1)[-1]) if spec_dtype is not None else None
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


# ---------------------------------------------------------------------------
# Lamp classifier (CONTRACT.md High-speed approach detect-then-track)
# ---------------------------------------------------------------------------

# Fixed batch so the tinygrad JIT compiles once at startup; the pipeline pads
# unused slots and consumes only the first `count` rows.
LAMP_INPUT_SHAPE = (4, 3, 32, 64)


class NullClassifier:
  """No classifier model configured: pipeline keeps the detector-only behavior."""

  name = "null"
  enabled = False

  def classify(self, crops_input):
    return None

  def warmup(self) -> None:
    pass

  def close(self) -> None:
    pass


class StubClassifier:
  """Deterministic classifier for tests and the PC soak: fixed class vote."""

  name = "stub"
  enabled = True

  def __init__(self, cls: int = 0, prob: float = 0.9) -> None:
    import math
    self.cls = int(cls)
    self.prob = float(min(max(prob, 1e-3), 1.0 - 1e-6))
    self._logit = math.log(7.0 * self.prob / (1.0 - self.prob))

  def classify(self, crops_input):
    n = int(np.asarray(crops_input).shape[0])
    logits = np.zeros((n, 8), dtype=np.float32)
    logits[:, self.cls] = self._logit
    return logits

  def warmup(self) -> None:
    pass

  def close(self) -> None:
    pass


class OnnxCpuClassifier(OnnxCpuBackend):
  """onnxruntime CPU lamp classifier: same guards/threads, logits output."""

  name = "onnx"
  enabled = True

  def __init__(self, model_path: str) -> None:
    super().__init__(model_path, param_name="CarrotTlvLampModelPath")

  def classify(self, crops_input):
    outs = self.infer(crops_input)
    return outs[0] if outs else None

  def warmup(self) -> None:
    self.classify(np.zeros(LAMP_INPUT_SHAPE, dtype=np.uint8))


class TinygradCpuClassifier(TinygradCpuBackend):
  """tinygrad CPU lamp classifier (fixed batch 4, compiled once at startup)."""

  name = "tinygrad"
  enabled = True

  def __init__(self, model_path: str) -> None:
    super().__init__(model_path, param_name="CarrotTlvLampModelPath")

  def classify(self, crops_input):
    outs = self.infer(crops_input)
    return outs[0] if outs else None

  def warmup(self) -> None:
    self.classify(np.zeros(LAMP_INPUT_SHAPE, dtype=np.uint8))


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


def make_classifier(name: str, model_path: str = ""):
  """Lamp-classifier factory. No/empty path -> NullClassifier (detector-only)."""
  key = (name or "").strip().lower()
  if key in ("", "null", "none") or not model_path:
    return NullClassifier()
  if key == "onnx":
    return OnnxCpuClassifier(model_path)
  if key == "tinygrad":
    return TinygradCpuClassifier(model_path)
  if key == "dsp":
    raise NotImplementedError("DSP lamp classifier is not implemented yet; W70 decides the Hexagon runtime")
  raise BackendError(f"unknown classifier backend: {name!r}")
