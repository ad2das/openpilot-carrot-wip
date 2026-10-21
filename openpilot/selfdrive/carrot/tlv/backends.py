"""Inference backends for tlvd.

CPU-only by construction. The detector refuses to run when the environment or
runtime hints at a GPU (QCOM/GPU/CL/CUDA) per CONTRACT.md; the Hexagon DSP
detector runs through a persistent SNPE server process (WSNPE13) and falls back
to the onnx CPU backend on any failure.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from openpilot.selfdrive.carrot.tlv.decode import Decoder, PlaceholderDecoder, make_decoder

GPU_ENV_KEYS = ("QCOM", "GPU", "CL", "CUDA", "OPENCL", "ROCM", "VULKAN", "TINY_GPU", "TINYGPU")
GPU_ENV_PREFIXES = ("QCOM", "CUDA", "OPENCL", "ROCM", "VULKAN", "GPU_", "TINYGPU")
# These tokens in the key name are a GPU hint regardless of the value.
GPU_STRICT_KEY_TOKENS = ("QCOM", "CUDA", "OPENCL", "ROCM", "VULKAN")
GPU_VALUE_TOKENS = ("QCOM", "CUDA", "OPENCL", "VULKAN", "ROCM")
_FALSY = ("", "0", "false", "no", "off")

# Bundled traffic-light models: shipped so the TL vision works with no model
# path params. The detector is r8s50a40 (single-class, 40-epoch r8s50a); the
# lamp is the r9 device-crop retrain. Absolute paths resolved next to this
# module; the SHA256 values are the verified artifacts, checked by the unit
# tests only (tlvd never hashes a model at startup).
_BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BUNDLED_DETECTOR = os.path.join(_BUNDLED_DIR, "tl_detector_r8s50a40.onnx")
BUNDLED_LAMP = os.path.join(_BUNDLED_DIR, "tl_lamp_r9.onnx")
BUNDLED_DETECTOR_SHA256 = "fc895608b547b0224080e0ee9f183bbfa9a5b441ce5ba381e722f4cc28939162"
BUNDLED_LAMP_SHA256 = "665d3649c0392a6d306a822dfe905070e1149cea68f3e6c5b0cddfbd3dc38be7"


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
  output_names: tuple[str, ...] = ()

  def __init__(self, model_path: str = "") -> None:
    self.model_path = model_path

  def infer(self, model_input):
    return []

  def warmup(self) -> None:
    pass

  def close(self) -> None:
    pass


class OnnxCpuBackend:
  """onnxruntime CPU-only backend (1 intra-op thread, 1 inter-op thread).

  Threads are pinned to 1 with spinning disabled so the detector adds only one
  core's worth of work next to modeld/controlsd on the C3X; graph
  optimization is enabled for a one-time compile.
  """

  name = "onnx"

  def __init__(self, model_path: str, intra_op_threads: int = 1, inter_op_threads: int = 1,
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
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # No busy-wait spinning: a spin loop burns a core next to modeld even when
    # the detector is idle between frames.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    self._session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
    if self._session.get_providers() != ["CPUExecutionProvider"]:
      raise BackendError(f"unexpected execution providers: {self._session.get_providers()}")
    self._input_name = self._session.get_inputs()[0].name
    self.output_names = tuple(o.name for o in self._session.get_outputs())
    self.model_path = model_path

  def infer(self, model_input):
    return self._session.run(None, {self._input_name: model_input})

  def warmup(self) -> None:
    """One inference on a zero uint8 input of the model's input shape.

    Mirrors the classifier warmup: runtimes may allocate lazy kernels/pools on
    the first call (field route 00000443: the first detect frame took 12.2 s
    and blew the CPU credit); doing it here keeps the run loop clean.
    """
    shape = self._session.get_inputs()[0].shape
    if not all(isinstance(d, int) for d in shape):
      return  # dynamic dims: the first real frame compiles once
    self.infer(np.zeros(tuple(shape), dtype=np.uint8))

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
    self.output_names = self._output_names
    self._jits: dict[tuple, object] = {}
    self.model_path = model_path
    self._warmed = False
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

  def _spec_np_dtype(self):
    """Numpy zeros dtype for warm-up, mapped exactly like _make_tensor maps it."""
    spec_dtype = getattr(self._input_spec, "dtype", None)
    if spec_dtype is None:
      return np.float32
    return TINYGRAD_NP_DTYPES.get(str(spec_dtype).rsplit(".", 1)[-1], np.float32)

  def _warmup(self) -> None:
    shape = tuple(self._input_spec.shape)
    if not all(isinstance(d, int) for d in shape):
      return  # dynamic dims: the first real frame compiles once
    np_dtype = self._spec_np_dtype()
    try:
      # TinyJit: 1st call runs, 2nd captures, 3rd+ replays (engine/jit.py); warm both so the first real frame replays.
      self._call(np.zeros(shape, dtype=np_dtype))
      self._call(np.zeros(shape, dtype=np_dtype))
    except BackendError:
      raise
    except Exception as exc:
      raise BackendError(f"tinygrad CPU warmup failed: {exc}") from exc
    self._warmed = True

  def warmup(self) -> None:
    """Public warm-up hook; the constructor already ran the run+capture pair."""
    if not getattr(self, "_warmed", False):
      self._warmup()

  def infer(self, model_input):
    return self._call(model_input)

  def close(self) -> None:
    self._jits.clear()
    self._runner = None


# ---------------------------------------------------------------------------
# Hexagon DSP detector (WSNPE13): a persistent SNPE server process
# ---------------------------------------------------------------------------

# The r8s50a40 detector as a quantized DLC, run by third_party/snpe_tlv. The DLC
# takes NHWC float 0..255 and returns one "heads" tensor; its one output is
# split into six NCHW heads by the heads.json offsets (the probe's
# load_heads/split_heads, REPORT-WSNPE11 T-D: 31/31 dets equal to onnxruntime).
DSP_DLC = os.path.join(_BUNDLED_DIR, "tl_detector_r8s50a40_q.dlc")
DSP_HEADS = os.path.join(_BUNDLED_DIR, "tl_detector_r8s50a40_heads.json")
DSP_HEAD_ORDER = ("p3_box", "p3_cls", "p4_box", "p4_cls", "p5_box", "p5_cls")
DSP_IN_SHAPE = (1, 3, 320, 640)
DSP_IN_BYTES = 320 * 640 * 3
DSP_OUT_FLOATS = 273000
DSP_OUT_BYTES = DSP_OUT_FLOATS * 4
SNPE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "third_party", "snpe_tlv")
DSP_SERVER_SRC = os.path.join(SNPE_ROOT, "src", "tlv_snpe_server.cc")
DSP_CACHE_ROOT = "/data/carrot_tlv_snpe"
DSP_READY_TIMEOUT_S = 20.0
# The server inherits tlvd's SCHED_IDLE/nice 19, so a busy C3X can delay a
# 16 ms frame; one miss retires the DSP for the whole drive, so allow 2 s.
DSP_FRAME_TIMEOUT_S = 2.0
DSP_BUILD_TIMEOUT_S = 150.0
# The probe's build flags verbatim (dsp_probe.build_argv, WSNPE3 build pass
# 17.6 s): SNPE 1.41 exports the pre-cxx11 libstdc++ ABI, so every zdl:: API
# taking a std::string fails to link without the ABI flag.
_DSP_COMPILE_FLAGS = ("-std=c++17", "-O2", "-D_GLIBCXX_USE_CXX11_ABI=0")
_DSP_LINK_FLAGS = ("-lSNPE",)
_POSIX = os.name == "posix"


def dsp_build_argv(clangxx: str, src: str, include: str, lib: str, out_bin: str) -> list[str]:
  return [clangxx, *_DSP_COMPILE_FLAGS, "-I", include, src, "-L", lib, *_DSP_LINK_FLAGS,
          f"-Wl,-rpath,{lib}", "-o", out_bin]


def _dsp_server_hash() -> str:
  """Cache key: the server source bytes plus the path-free compile flags."""
  import hashlib
  with open(DSP_SERVER_SRC, "rb") as f:
    digest = hashlib.sha256(f.read())
  digest.update(b"\0")
  digest.update("\0".join(_DSP_COMPILE_FLAGS + _DSP_LINK_FLAGS).encode())
  return digest.hexdigest()[:12]


def dsp_server_path() -> str:
  return os.path.join(DSP_CACHE_ROOT, _dsp_server_hash(), "tlv_snpe_server")


def _dsp_shm_dir() -> str:
  """The parent's shm directory: /dev/shm on the device, the temp dir on a PC."""
  if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
    return "/dev/shm"
  return tempfile.gettempdir()


def dsp_server_env() -> dict:
  """The probe's runtime env (dsp_probe.snpe_env): the bundled SNPE libraries
  and the Hexagon skels."""
  env = dict(os.environ)
  lib = os.path.join(SNPE_ROOT, "larch64")
  old = env.get("LD_LIBRARY_PATH", "")
  env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:" + lib + (":" + old if old else "")
  env["ADSP_LIBRARY_PATH"] = os.path.join(SNPE_ROOT, "dsp") + "/"
  return env


def ensure_dsp_server(run=None, which=None) -> str:
  """Path of the compiled SNPE server, compiling it once when absent.

  The cache key covers the source bytes and the flags, so a source edit yields
  a new path; the binary is compiled into `<path>.tmp` and renamed, so a
  concurrent or interrupted build never leaves a truncated executable."""
  run = run or subprocess.run
  which = which or shutil.which
  out_bin = dsp_server_path()
  if os.path.isfile(out_bin) and os.access(out_bin, os.X_OK):
    return out_bin
  clangxx = which("clang++")
  if not clangxx:
    raise BackendError("clang++ not found on PATH")
  try:
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
  except OSError as exc:
    raise BackendError(f"cannot create {os.path.dirname(out_bin)}: {exc!r}") from exc
  tmp_bin = out_bin + ".tmp"
  argv = dsp_build_argv(clangxx, DSP_SERVER_SRC, os.path.join(SNPE_ROOT, "include"),
                        os.path.join(SNPE_ROOT, "larch64"), tmp_bin)
  try:
    proc = run(argv, capture_output=True, timeout=DSP_BUILD_TIMEOUT_S)
  except subprocess.TimeoutExpired as exc:
    raise BackendError(f"SNPE server build timed out after {DSP_BUILD_TIMEOUT_S:.0f} s") from exc
  except Exception as exc:  # noqa: BLE001 - any spawn failure is a build failure
    raise BackendError(f"SNPE server build failed: {exc!r}") from exc
  rc = int(getattr(proc, "returncode", 1))
  if rc != 0 or not os.path.isfile(tmp_bin):
    stderr = getattr(proc, "stderr", b"") or b""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    raise BackendError(f"SNPE server build failed (rc={rc}): {text.strip()[-400:]}")
  os.replace(tmp_bin, out_bin)
  return out_bin


def _remove_stale_dsp_shm(shm_dir: str, own_pid: int) -> list[str]:
  """Unlink tlv_snpe_<pid>_in/_out files whose tlvd is gone.

  A SIGKILLed tlvd cannot unlink its two files (1.7 MB of RAM-backed /dev/shm
  each run); the next tlvd removes them. The own pid and live pids are kept.
  (The server sets PR_SET_PDEATHSIG itself, so no preexec_fn is needed here: a
  preexec_fn is unsafe in tlvd, which already runs threads.)"""
  removed: list[str] = []
  try:
    names = os.listdir(shm_dir)
  except OSError:
    return removed
  for name in names:
    parts = name.split("_")
    if len(parts) != 4 or parts[:2] != ["tlv", "snpe"] or parts[3] not in ("in", "out"):
      continue
    try:
      pid = int(parts[2])
    except ValueError:
      continue
    if pid == own_pid:
      continue
    try:
      os.kill(pid, 0)
      continue  # alive (or not ours to signal): keep
    except ProcessLookupError:
      pass
    except OSError:
      continue
    try:
      os.unlink(os.path.join(shm_dir, name))
      removed.append(name)
    except OSError:
      pass
  return removed


class DspBackend:
  """Hexagon DSP detector through a persistent SNPE server process.

  tlvd and the server exchange pixels and heads through two mmap'ed files in
  /dev/shm (the server checks their sizes; tlvd checks the READY line), and
  frames are handed over with one `RUN <seq>` line each. Every DSP failure -
  build, init, timeout, EOF, ERR, a wrong seq - switches this backend to the
  onnx CPU detector for good, so the TL vision is never worse than the W132
  default. The server's CPU is charged to tlvd's budget through
  child_cpu_seconds().
  """

  name = "dsp"
  output_names: tuple[str, ...] = DSP_HEAD_ORDER

  def __init__(self, model_path: str, *, runtime: str = "dsp", server_cmd: str | None = None,
               command_prefix=(), fallback_factory=None) -> None:
    if os.path.realpath(model_path or "") != os.path.realpath(BUNDLED_DETECTOR):
      # A custom onnx has no DLC; refusing keeps the fallback honest.
      raise BackendError(f"the dsp backend only runs the bundled detector: {model_path!r}")
    reason = gpu_env_reason()
    if reason is not None:
      raise BackendError(f"refusing GPU-capable environment: {reason}")
    self.model_path = model_path
    self.runtime = runtime
    self.server_cmd = server_cmd
    self.command_prefix = tuple(command_prefix)
    self._fallback_factory = fallback_factory or (lambda: OnnxCpuBackend(model_path))
    self._fallback = None
    self._active = "dsp"
    self.fallback_reason = ""
    self._proc = None
    self._rbuf = b""
    self._seq = 0
    self._reaped_cpu = 0.0
    self._reader = None
    with open(DSP_HEADS, encoding="utf-8") as f:
      spec = json.load(f)
    self._heads_total = int(spec["total"])
    self._heads = {name: (int(h["offset"]), int(h["count"]), tuple(h["dims"]))
                   for name, h in spec["heads"].items()}
    if self._heads_total != DSP_OUT_FLOATS or set(DSP_HEAD_ORDER) - set(self._heads):
      raise BackendError(f"heads spec mismatch: {DSP_HEADS}")
    shm_dir = _dsp_shm_dir()
    _remove_stale_dsp_shm(shm_dir, os.getpid())
    self._in_path = os.path.join(shm_dir, f"tlv_snpe_{os.getpid()}_in")
    self._out_path = os.path.join(shm_dir, f"tlv_snpe_{os.getpid()}_out")
    self._create_shm(self._in_path, DSP_IN_BYTES)
    self._create_shm(self._out_path, DSP_OUT_BYTES)
    self._in_map = np.memmap(self._in_path, dtype=np.uint8, mode="r+", shape=(DSP_IN_BYTES,))
    self._out_map = np.memmap(self._out_path, dtype=np.float32, mode="r+", shape=(DSP_OUT_FLOATS,))

  @staticmethod
  def _create_shm(path: str, size: int) -> None:
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o600)
    try:
      os.ftruncate(fd, size)
    finally:
      os.close(fd)

  # -- lifecycle ------------------------------------------------------------

  @property
  def active(self) -> str:
    return self._active

  def _log_path(self) -> str:
    try:
      os.makedirs(DSP_CACHE_ROOT, exist_ok=True)
      return os.path.join(DSP_CACHE_ROOT, "server.log")
    except OSError:
      return os.path.join(tempfile.gettempdir(), "carrot_tlv_snpe_server.log")

  def _start_server(self) -> None:
    server = self.server_cmd or ensure_dsp_server()
    cmd = [*self.command_prefix, server, "--dlc", DSP_DLC, "--runtime", self.runtime,
           "--in", self._in_path, "--out", self._out_path]
    log = open(self._log_path(), "ab")
    try:
      self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=log, env=dsp_server_env())
    finally:
      log.close()
    self._rbuf = b""

  def _readline(self, timeout: float) -> str | None:
    """One stdout line, or None on EOF/timeout. select on POSIX; a reader
    thread where pipes are not selectable (the PC test host)."""
    if not _POSIX:
      return self._readline_threaded(timeout)
    import select
    deadline = time.monotonic() + timeout
    while b"\n" not in self._rbuf:
      remain = deadline - time.monotonic()
      if remain <= 0:
        return None
      try:
        ready, _, _ = select.select([self._proc.stdout.fileno()], [], [], remain)
      except (OSError, ValueError):
        return None
      if not ready:
        return None
      chunk = os.read(self._proc.stdout.fileno(), 4096)
      if not chunk:
        return None
      self._rbuf += chunk
    line, _, self._rbuf = self._rbuf.partition(b"\n")
    return line.decode("utf-8", "replace")

  def _readline_threaded(self, timeout: float) -> str | None:
    import queue
    import threading
    if self._reader is None:
      lines: queue.Queue = queue.Queue()
      stdout = self._proc.stdout

      def _pump() -> None:
        for raw in iter(stdout.readline, b""):
          lines.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        lines.put(None)

      threading.Thread(target=_pump, daemon=True).start()
      self._reader = lines
    try:
      return self._reader.get(timeout=timeout)
    except queue.Empty:
      return None

  def _reap(self, graceful: bool = False) -> None:
    """Stop the server and fold its CPU into the reaped total.

    Failure paths kill it outright (a stalled server must not delay the
    fallback frame); close() first closes stdin and gives it a second to exit."""
    proc, self._proc = self._proc, None
    self._reader = None
    if proc is None:
      return
    cpu = self._proc_cpu(proc.pid)
    try:
      if proc.stdin is not None:
        proc.stdin.close()
    except OSError:
      pass
    if graceful and proc.poll() is None:
      try:
        proc.wait(timeout=1.0)
      except subprocess.TimeoutExpired:
        pass
    if proc.poll() is None:
      try:
        proc.kill()
      except OSError:
        pass
      try:
        proc.wait(timeout=1.0)
      except subprocess.TimeoutExpired:
        pass
    if proc.stdout is not None:
      try:
        proc.stdout.close()
      except OSError:
        pass
    self._reaped_cpu += cpu

  @staticmethod
  def _proc_cpu(pid: int) -> float:
    """utime+stime of `pid` from /proc (0.0 where /proc is absent)."""
    try:
      with open(f"/proc/{int(pid)}/stat", encoding="utf-8", errors="replace") as f:
        fields = f.read().rsplit(")", 1)[1].split()
      return (float(fields[11]) + float(fields[12])) / float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError, AttributeError):
      return 0.0

  def child_cpu_seconds(self) -> float:
    """Total CPU of the live server plus every server already reaped."""
    live = self._proc_cpu(self._proc.pid) if self._proc is not None else 0.0
    return self._reaped_cpu + live

  def _switch_to_fallback(self, reason: str) -> None:
    self._reap()
    self._active = "cpu_fallback"
    self.fallback_reason = str(reason)
    fallback = self._fallback_factory()
    fallback.warmup()
    self._fallback = fallback

  def warmup(self) -> None:
    try:
      self._start_server()
      line = self._readline(DSP_READY_TIMEOUT_S)
      if line is None:
        raise BackendError("no READY line")
      parts = line.split()
      if parts[:1] != ["READY"]:
        raise BackendError(line.strip() or "empty READY line")
      if parts[1:] != [str(DSP_IN_BYTES), str(DSP_OUT_BYTES)]:
        raise BackendError(f"READY sizes {parts[1:]}")
      # One zero frame proves the whole loop end to end; a failure here already
      # switched to the warm fallback, so it is not raised again.
      self.infer(np.zeros(DSP_IN_SHAPE, dtype=np.uint8))
    except Exception as exc:  # noqa: BLE001 - every DSP failure falls back
      print(f"tlvd: dsp backend unavailable ({exc}); using the onnx CPU detector", flush=True)
      try:
        self._switch_to_fallback(exc)
      except Exception as fallback_exc:  # noqa: BLE001
        raise BackendError(f"dsp failed ({exc}) and the fallback too ({fallback_exc!r})") from fallback_exc

  # -- inference ------------------------------------------------------------

  def infer(self, model_input):
    if self._active != "dsp" or self._fallback is not None:
      if self._fallback is None:
        raise BackendError(f"dsp failed ({self.fallback_reason}) and no fallback detector is available")
      return self._fallback.infer(model_input)
    data = np.asarray(model_input)
    if data.shape != DSP_IN_SHAPE or data.dtype != np.uint8:
      # A caller bug, not a DSP failure: never silently degrade the backend.
      raise ValueError(f"dsp model_input must be uint8 {DSP_IN_SHAPE}, got {data.dtype} {data.shape}")
    self._in_map[:] = data.reshape(-1)
    self._seq += 1
    seq = self._seq
    try:
      self._proc.stdin.write(f"RUN {seq}\n".encode())
      self._proc.stdin.flush()
      line = self._readline(DSP_FRAME_TIMEOUT_S)
      if line is None:
        raise BackendError("no reply (timeout or EOF)")
      parts = line.split(None, 2)
      if not parts or parts[0] != "OK" or len(parts) < 2 or parts[1] != str(seq):
        raise BackendError(line.strip() or "empty reply")
    except Exception as exc:  # noqa: BLE001 - one bad reply retires the DSP path
      print(f"tlvd: dsp frame failed ({exc}); using the onnx CPU detector", flush=True)
      self._switch_to_fallback(exc)
      return self._fallback.infer(model_input)
    flat = np.array(self._out_map)
    return [flat[off:off + count].reshape(dims) for off, count, dims in
            (self._heads[name] for name in DSP_HEAD_ORDER)]

  def close(self) -> None:
    self._reap(graceful=True)
    for path in (self._in_path, self._out_path):
      try:
        os.unlink(path)
      except OSError:
        pass
    if self._fallback is not None:
      self._fallback.close()
      self._fallback = None


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
    """No-op when the constructor already ran the run+capture pair: the
    classifier must not pay a third pointless JIT call."""
    if not getattr(self, "_warmed", False):
      self._warmup()


def make_backend(name: str, model_path: str = "", decoder: Decoder | None = None):
  key = (name or "").strip().lower()
  if key in ("", "null", "none"):
    backend = NullBackend(model_path)
  elif key == "onnx":
    backend = OnnxCpuBackend(model_path)
  elif key == "tinygrad":
    backend = TinygradCpuBackend(model_path)
  elif key == "dsp":
    backend = DspBackend(model_path)
  else:
    raise BackendError(f"unknown backend: {name!r}")
  if decoder is not None:
    return backend, decoder
  return backend, make_decoder(getattr(backend, "output_names", ()))


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
    return OnnxCpuClassifier(model_path)
  raise BackendError(f"unknown classifier backend: {name!r}")
