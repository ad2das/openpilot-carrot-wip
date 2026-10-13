"""Onroad-while-parked one-time Hexagon DSP probe for the carrot TLV detector.

Runs as the manager process "tlv_dsp_probe" while the car is started
(onroad). The manager starts it when started AND CarrotTlvMode >= 1 AND
TrafficLightDetectMode >= 1 and the result file for the current git commit does
not exist; after the result is written the process exits 0 and the manager does
not restart it. There is no offroad window (the car cuts OBD power at ignition
off), so the probe runs while PARKED during a normal drive.

Parent gate loop (2 Hz, tiny idle cost while driving): the parent idles on
short sleeps and only checks

  PARKED = carState.gearShifter == park AND carState.standstill AND
           vEgo < 0.1 AND not selfdriveState.enabled,
           with both messages fresh (< 1 s old).

The child is spawned only after PARKED held continuously for 5 s. While the
child runs, the parent re-checks every 0.1 s: the first non-PARKED sample (or
stale message) SIGKILLs the child immediately, writes no result and restores
the attempt latch, then goes back to waiting. Every 0.5 s it reads
/proc/<child>/statm; over 300 MB RSS the child is killed and the result is
written with error="child_rss_limit". The 180 s hard total timeout and the
per-step timeouts are unchanged; a child that exits without a result gets a
child_exit_<code> result.

Orphan protection: the child is spawned with a Linux PR_SET_PDEATHSIG=SIGKILL
preexec hook (best-effort), and main() installs SIGINT/SIGTERM handlers that
kill the running child (waiting up to 2 s) before exiting 0 WITHOUT writing a
result - a manager stop is not a probe failure. All results are written
atomically (tmp + fsync + rename), so a kill never leaves a partial file.

UI notice: while it works the parent publishes CarrotTlvDspNotice in
params_memory (arming -> running step 1..3 -> done/failed, aborted when PARKED
drops) for the onroad HUD capsule; contract in ui/onroad/tlv_dsp_notice.py.
Every notice write is best-effort and never affects the probe outcome.

Lost-step latch: the child writes <commit>.progress (step name + monotonic
start, atomic) before each step. At startup an existing <commit>.attempt means
the previous run was lost (power cut or device reset): the .progress step is
appended to its "lost_steps" list and consumed immediately (the updated
.attempt is persisted and .progress removed before the latch check), so a cut
without a signal cannot count the same loss twice. A 3rd loss of the SAME
step writes the final result error="lost_in_step:<step>" and sets
_probe_disabled_by_lost_attempts. Losses spread over different steps (random
power cuts) never latch. A clean finish deletes .attempt and .progress; a
clean kill (unparked or signal) restores .attempt to its pre-spawn content.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from openpilot.selfdrive.carrot.tlv import backends, resources

PROBE_ROOT = "/data/media/0/carrot_tlv_rec/dsp_probe"
TOTAL_TIMEOUT_S = 180.0
STEP_TIMEOUTS = {"presence": 30.0, "smoke": 60.0, "detector": 90.0}
PRESENCE_PATHS = ("/dev/adsprpc-smd", "/dsp/cdsp/fastrpc_shell_3", "/dev/ion")
CLANG_FLAGS = ("--target=hexagon", "-mcpu=hexagonv65", "-mhvx=v65", "-mhvx-length=128b", "-shared", "-fPIC")
CLANG_SOURCE = b"int tlv_dsp_probe_add(int a, int b) { return a + b; }\n"
CLANG_TIMEOUT_S = 20.0
MAX_ERROR_CHARS = 2048
STEADY_RUNS_SMOKE = 10
STEADY_RUNS_DETECTOR = 20
DETECTOR_INPUT_SHAPE = (1, 3, 320, 640)
DSP_DEVICE = "DSP"
# Anything accelerators-specific that is NOT the Hexagon DSP is forbidden here
# (same guard spirit as tlvd's DEV=CPU pin; backends.TINYGRAD_GPU_DEVICES
# includes DSP itself, which this probe is allowed to open).
FORBIDDEN_DEVICES = tuple(d for d in backends.TINYGRAD_GPU_DEVICES if d != DSP_DEVICE)

MB = 1024 * 1024
# Parked gate (item values are the contract with the manager-side review).
SD_SERVICE = "selfdriveState"  # this tree publishes selfdriveState (not controlsState)
PARK_SERVICES = ("carState", SD_SERVICE)
PARK_FRESH_S = 1.0
PARK_HOLD_S = 5.0
GATE_HZ = 2.0
GATE_PERIOD_S = 1.0 / GATE_HZ
RUN_POLL_S = 0.1
RSS_POLL_S = 0.5
CHILD_RSS_LIMIT_BYTES = 300 * MB
LOST_STEP_LIMIT = 3
# On-screen notice contract with ui/onroad/tlv_dsp_notice.py (params_memory).
NOTICE_KEY = "CarrotTlvDspNotice"
NOTICE_PERIOD_S = 0.5
ARMING_SHOW_S = 0.5

_FALLBACK_COMMIT: str | None = None
IS_LINUX = sys.platform.startswith("linux")
PR_SET_PDEATHSIG = 1
_POPEN = subprocess.Popen
_SIGNAL = signal.signal
# The child currently being waited on, reachable from the stop-signal handler.
_RUNNING_CHILD = None
_probe_disabled_by_lost_attempts = False


# ---------------------------------------------------------------------------
# identity / filesystem helpers
# ---------------------------------------------------------------------------

def _truncate(text, limit: int = MAX_ERROR_CHARS):
  if text is None:
    return None
  return str(text)[:limit]


def _timestamp_iso() -> str:
  return datetime.now().astimezone().isoformat(timespec="seconds")


def _agnos_version():
  try:
    with open("/VERSION", encoding="utf-8", errors="replace") as f:
      value = f.read(200).strip()
    return value or None
  except OSError:
    return None


def _device_type():
  try:
    from openpilot.system.hardware import PC, TICI
    if TICI:
      return "tici"
    if PC:
      return "pc"
  except Exception:
    pass
  return None


def _basedir() -> str:
  try:
    from openpilot.common.basedir import BASEDIR
    return str(BASEDIR)
  except Exception:
    return str(Path(__file__).resolve().parents[4])


def _read_git_head_sha() -> str:
  """Current commit id from .git/HEAD, cheap enough for the manager loop.

  Two small file reads (HEAD + the ref file); "" when packed refs / detached
  layouts hide it, in which case the cached subprocess fallback is used."""
  git_dir = os.path.join(_basedir(), ".git")
  try:
    with open(os.path.join(git_dir, "HEAD"), encoding="utf-8", errors="replace") as f:
      head = f.read(200).strip()
  except OSError:
    return ""
  if head.startswith("gitdir:"):
    git_dir = head.split(":", 1)[1].strip()
    try:
      with open(os.path.join(git_dir, "HEAD"), encoding="utf-8", errors="replace") as f:
        head = f.read(200).strip()
    except OSError:
      return ""
  if head.startswith("ref:"):
    ref_path = os.path.join(git_dir, *head.split(":", 1)[1].strip().split("/"))
    try:
      with open(ref_path, encoding="utf-8", errors="replace") as f:
        return f.read(200).strip()
    except OSError:
      return ""  # packed refs: fall back to git
  return head


def _fallback_commit_short() -> str:
  global _FALLBACK_COMMIT
  if _FALLBACK_COMMIT is None:
    try:
      done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_basedir(),
                            capture_output=True, text=True, timeout=5.0)
      _FALLBACK_COMMIT = done.stdout.strip() or "unknown"
    except Exception:
      _FALLBACK_COMMIT = "unknown"
  return _FALLBACK_COMMIT


def current_commit_short() -> str:
  env = os.getenv("GIT_COMMIT", "").strip()
  if env:
    return env[:7]
  sha = _read_git_head_sha()
  if sha:
    return sha[:7]
  return _fallback_commit_short()


def probe_result_path() -> str:
  return os.path.join(PROBE_ROOT, f"{current_commit_short()}.json")


def attempt_path(result_path: str) -> str:
  return os.path.splitext(result_path)[0] + ".attempt"


def progress_path(result_path: str) -> str:
  return os.path.splitext(result_path)[0] + ".progress"


def _signal_color_requested() -> bool:
  # The opt-in signal-color observer owns the camera on /data. The TLV probe
  # must never run alongside it; missing module/file means not requested.
  try:
    from openpilot.selfdrive.modeld.signal_color_shadow import requested
    return requested()
  except Exception:
    return False


def should_run(started: bool, params) -> bool:
  """Manager enable condition: onroad, TLV enabled, no result, not latched."""
  if not started or _probe_disabled_by_lost_attempts or _signal_color_requested():
    return False
  try:
    if params.get_int("CarrotTlvMode") < 1 or params.get_int("TrafficLightDetectMode") < 1:
      return False
  except Exception:
    return False
  return not os.path.exists(probe_result_path())


def _write_atomic(path: str, write_body) -> None:
  """tmp + fsync + rename; never leaves a partial file behind."""
  directory = os.path.dirname(path)
  if directory:
    os.makedirs(directory, exist_ok=True)
  tmp = f"{path}.tmp"
  try:
    with open(tmp, "w", encoding="utf-8") as f:
      write_body(f)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
  except Exception:
    try:
      os.unlink(tmp)
    except OSError:
      pass
    raise


def write_result_atomic(path: str, payload: dict) -> None:
  _write_atomic(path, lambda f: json.dump(payload, f, ensure_ascii=False, indent=1, sort_keys=True))


def read_attempt_state(path: str) -> dict | None:
  """{"count": n, "lost_steps": [...]} or None when absent/unreadable."""
  try:
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    if not isinstance(data, dict):
      return None
    count = max(0, int(data.get("count", 0)))
    lost = [str(step) for step in (data.get("lost_steps") or [])]
    return {"count": count, "lost_steps": lost}
  except (OSError, ValueError, TypeError):
    return None


def write_attempt_state(path: str, state: dict) -> None:
  payload = {"count": int(state.get("count", 0)),
             "lost_steps": [str(step) for step in state.get("lost_steps", [])]}
  _write_atomic(path, lambda f: json.dump(payload, f, sort_keys=True))


def write_progress(path: str, step: str) -> None:
  payload = {"step": str(step), "t_mono": time.monotonic()}
  _write_atomic(path, lambda f: json.dump(payload, f, sort_keys=True))


def read_progress_step(path: str) -> str | None:
  try:
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    step = data.get("step") if isinstance(data, dict) else None
    return str(step) if step else None
  except (OSError, ValueError, AttributeError):
    return None


def _remove_file(path: str) -> None:
  try:
    os.unlink(path)
  except OSError:
    pass


def restore_attempt_state(attempt_file: str, pre_state: dict | None) -> None:
  """Clean kill: put the attempt latch back to its pre-spawn content."""
  try:
    if pre_state is None:
      _remove_file(attempt_file)
    else:
      write_attempt_state(attempt_file, pre_state)
  except Exception:
    pass


def _cleanup_latch(attempt_file: str, progress_file: str) -> None:
  _remove_file(attempt_file)
  _remove_file(progress_file)


def _latched_step(lost_steps: list[str]) -> str | None:
  """The first step that was lost LOST_STEP_LIMIT times, else None."""
  counts: dict[str, int] = {}
  for step in lost_steps:
    counts[step] = counts.get(step, 0) + 1
    if counts[step] >= LOST_STEP_LIMIT:
      return step
  return None


def _result_payload(error=None, meta: dict | None = None) -> dict:
  meta = meta or {}
  return build_result_payload(
    {},
    commit=meta.get("commit") or current_commit_short(),
    device_type=meta.get("device_type", _device_type()),
    agnos_version=meta.get("agnos_version", _agnos_version()),
    timestamp=_timestamp_iso(),
    error=error,
  )


# ---------------------------------------------------------------------------
# cpu policy (same as tlvd)
# ---------------------------------------------------------------------------

def apply_cpu_policy() -> list[str]:
  """Cores 1-3 + SCHED_IDLE + nice19 on POSIX; silently skipped on Windows."""
  applied: list[str] = []
  try:
    os.sched_setaffinity(0, {1, 2, 3})
    applied.append("affinity=1,2,3")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    applied.append("sched=SCHED_IDLE")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    os.nice(19)
    applied.append("nice=19")
  except (AttributeError, OSError, ValueError):
    pass
  return applied


# ---------------------------------------------------------------------------
# step runner with per-step timeout
# ---------------------------------------------------------------------------

def _timed(name: str, fn, timeout_s: float) -> dict:
  box: dict = {}

  def run() -> None:
    started = time.perf_counter()
    try:
      result = fn()
      box["out"] = result if isinstance(result, dict) else {"status": "fail", "error": f"{name}: no result"}
    except Exception as exc:
      box["out"] = {"status": "fail", "error": _truncate(repr(exc))}
    box["duration_s"] = time.perf_counter() - started

  thread = threading.Thread(target=run, name=f"tlv-dsp-{name}", daemon=True)
  thread.start()
  thread.join(timeout_s)
  out = box.get("out")
  if out is None:
    return {"name": name, "status": "timeout", "error": f"step exceeded {timeout_s:.0f} s"}
  out.setdefault("name", name)
  out["duration_s"] = round(float(box.get("duration_s", 0.0)), 3)
  return out


# ---------------------------------------------------------------------------
# step a: DSP presence + clang toolchain
# ---------------------------------------------------------------------------

def _path_status(path: str) -> str:
  if not os.path.exists(path):
    return "missing"
  if not os.access(path, os.R_OK):
    return "not readable"
  return "ok"


def step_presence(*, path_status=None, which=None, run=None, workdir=None) -> dict:
  path_status = path_status or _path_status
  which = which or shutil.which
  run = run or subprocess.run

  paths = {path: path_status(path) for path in PRESENCE_PATHS}
  ok = all(status == "ok" for status in paths.values())
  clang: dict = {"path": which("clang") or "missing"}
  if clang["path"] == "missing":
    clang["status"] = "fail"
    clang["error"] = "clang not found on PATH"
    ok = False
  else:
    tmp = workdir or tempfile.mkdtemp(prefix="tlv_dsp_probe_")
    try:
      src = os.path.join(tmp, "probe.c")
      out = os.path.join(tmp, "probe.so")
      with open(src, "wb") as f:
        f.write(CLANG_SOURCE)
      try:
        proc = run([clang["path"], *CLANG_FLAGS, "-o", out, src],
                   capture_output=True, timeout=CLANG_TIMEOUT_S)
        clang["returncode"] = int(getattr(proc, "returncode", 1))
        stderr = getattr(proc, "stderr", b"") or b""
        stderr_text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
        if clang["returncode"] == 0:
          clang["status"] = "pass"
        else:
          clang["status"] = "fail"
          clang["error"] = _truncate(stderr_text or "clang returned nonzero")
      except subprocess.TimeoutExpired:
        clang["status"] = "fail"
        clang["error"] = f"clang timed out after {CLANG_TIMEOUT_S:.0f} s"
      except Exception as exc:
        clang["status"] = "fail"
        clang["error"] = _truncate(repr(exc))
      if clang.get("status") != "pass":
        ok = False
    finally:
      if workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)

  return {
    "status": "pass" if ok else "fail",
    "paths": paths,
    "clang": clang,
    "error": None if ok else "DSP presence or clang toolchain check failed",
  }


# ---------------------------------------------------------------------------
# step b: tinygrad DSP smoke
# ---------------------------------------------------------------------------

def _smoke_ops(Tensor, dtypes, device: str) -> tuple[bool, str]:
  a = Tensor([[1, 2], [3, 4]], dtype=dtypes.uint8, device=device)
  b = Tensor([[1, 0], [0, 1]], dtype=dtypes.uint8, device=device)
  elementwise = (a + b).numpy().tolist()
  matmul = a.matmul(b).numpy().tolist()
  if elementwise != [[2, 2], [3, 5]]:
    return False, f"elementwise mismatch: {elementwise}"
  if matmul != [[1, 2], [3, 4]]:
    return False, f"matmul mismatch: {matmul}"
  return True, ""


def _load_dsp_tensors():
  """(Tensor, dtypes, default device, opened-devices getter) on DEV=DSP."""
  os.environ["DEV"] = DSP_DEVICE
  try:
    import tinygrad
  except ImportError:
    repo = backends._tinygrad_repo_path()
    if repo not in sys.path:
      sys.path.append(repo)
    import tinygrad  # noqa: F401
  from tinygrad import Device, Tensor, dtypes
  return Tensor, dtypes, str(Device.DEFAULT), lambda: [str(d) for d in getattr(Device, "_opened_devices", ())]


def _forbidden_opened(opened) -> list[str]:
  bases = {str(device).split(":")[0] for device in opened}
  return sorted(bases & set(FORBIDDEN_DEVICES))


def step_smoke(*, load_tensors=None, runs: int = STEADY_RUNS_SMOKE) -> dict:
  load = load_tensors or _load_dsp_tensors
  try:
    Tensor, dtypes, default_device, opened = load()
  except Exception as exc:
    return {"status": "fail", "error": _truncate(f"tinygrad DSP load failed: {exc!r}")}
  base = str(default_device).split(":")[0]
  if base != DSP_DEVICE:
    return {"status": "fail", "default_device": base, "error": f"tinygrad default device is {base!r}, expected DSP"}
  try:
    started = time.perf_counter()
    ok, message = _smoke_ops(Tensor, dtypes, DSP_DEVICE)
    first_ms = (time.perf_counter() - started) * 1000.0
    if not ok:
      return {"status": "fail", "default_device": base, "error": message}
    times = []
    for _ in range(int(runs)):
      started = time.perf_counter()
      ok, message = _smoke_ops(Tensor, dtypes, DSP_DEVICE)
      times.append((time.perf_counter() - started) * 1000.0)
      if not ok:
        return {"status": "fail", "default_device": base, "error": message}
  except Exception as exc:
    return {"status": "fail", "default_device": base, "error": _truncate(repr(exc))}
  forbidden = _forbidden_opened(opened())
  if forbidden:
    return {"status": "fail", "default_device": base, "error": f"opened non-DSP accelerators: {forbidden}"}
  times.sort()
  return {
    "status": "pass",
    "default_device": base,
    "first_ms": round(first_ms, 3),
    "steady_median_ms": round(statistics.median(times), 3),
    "runs": int(runs),
  }


# ---------------------------------------------------------------------------
# step c: the tlvd ONNX detector on DEV=DSP (+ one CPU reference run)
# ---------------------------------------------------------------------------

def _fixed_detector_input() -> np.ndarray:
  rng = np.random.default_rng(7)
  return rng.integers(0, 256, size=DETECTOR_INPUT_SHAPE, dtype=np.uint8)


def _model_path_from_params() -> str:
  from openpilot.common.params import Params
  value = Params().get("CarrotTlvModelPath")
  if isinstance(value, bytes):
    value = value.decode("utf-8", "replace")
  return str(value or "").strip() or backends.BUNDLED_DETECTOR


class _OnnxAdapter:
  """OnnxRunner pinned to one explicit device; no implicit default search."""

  def __init__(self, model_path: str, device: str) -> None:
    os.environ["DEV"] = DSP_DEVICE
    try:
      import tinygrad
    except ImportError:
      repo = backends._tinygrad_repo_path()
      if repo not in sys.path:
        sys.path.append(repo)
      import tinygrad  # noqa: F401
    from tinygrad import Device, Tensor
    from tinygrad.nn.onnx import OnnxRunner
    self._device = str(device)
    self._Device = Device
    self._Tensor = Tensor
    self._runner = OnnxRunner(model_path)
    self._runner.to(self._device)
    self._input_name = next(iter(self._runner.graph_inputs))
    self._input_spec = self._runner.graph_inputs[self._input_name]
    self._output_names = tuple(self._runner.graph_outputs)

  def run(self, model_input) -> list[np.ndarray]:
    dtype = None
    spec_dtype = getattr(self._input_spec, "dtype", None)
    if spec_dtype is not None:
      dtype = backends.TINYGRAD_NP_DTYPES.get(str(spec_dtype).rsplit(".", 1)[-1])
    data = np.asarray(model_input)
    if dtype is not None:
      data = data.astype(dtype, copy=False)
    x = self._Tensor(data, device=self._device)
    outs = self._runner({self._input_name: x})
    return [np.asarray(outs[name].numpy()) for name in self._output_names]

  def opened_devices(self) -> list[str]:
    return [str(d) for d in getattr(self._Device, "_opened_devices", ())]


def step_detector(*, model_path=None, adapter_factory=None, path_getter=None,
                  runs: int = STEADY_RUNS_DETECTOR) -> dict:
  path_getter = path_getter or _model_path_from_params
  factory = adapter_factory or _OnnxAdapter
  try:
    path = model_path if model_path is not None else path_getter()
  except Exception as exc:
    return {"status": "fail", "error": _truncate(f"cannot read CarrotTlvModelPath: {exc!r}")}
  if not path:
    return {"status": "fail", "error": "CarrotTlvModelPath is not set"}
  if not os.path.isfile(path):
    return {"status": "fail", "error": f"model not found: {path}"}

  model_input = _fixed_detector_input()
  try:
    dsp = factory(path, DSP_DEVICE)
    started = time.perf_counter()
    dsp_out = dsp.run(model_input)
    compile_s = time.perf_counter() - started
    times = []
    for _ in range(int(runs)):
      started = time.perf_counter()
      dsp.run(model_input)
      times.append((time.perf_counter() - started) * 1000.0)
    cpu = factory(path, "CPU")
    cpu_out = cpu.run(model_input)
    opened = list(dsp.opened_devices()) + list(cpu.opened_devices())
  except Exception as exc:
    return {"status": "fail", "error": _truncate(repr(exc))}

  forbidden = _forbidden_opened(opened)
  if forbidden:
    return {"status": "fail", "error": f"opened non-DSP accelerators: {forbidden}"}
  if len(dsp_out) != len(cpu_out):
    return {"status": "fail", "error": f"output count mismatch: dsp={len(dsp_out)} cpu={len(cpu_out)}"}
  diff = 0.0
  for dsp_value, cpu_value in zip(dsp_out, cpu_out, strict=True):
    if dsp_value.shape != cpu_value.shape:
      return {"status": "fail", "error": f"output shape mismatch: {dsp_value.shape} vs {cpu_value.shape}"}
    if dsp_value.size:
      diff = max(diff, float(np.max(np.abs(dsp_value.astype(np.float64) - cpu_value.astype(np.float64)))))
  times.sort()
  return {
    "status": "pass",
    "compile_s": round(compile_s, 3),
    "steady_median_ms": round(statistics.median(times), 3),
    "steady_p90_ms": round(times[min(len(times) - 1, int(len(times) * 0.9))], 3),
    "max_abs_diff": diff,
    "runs": int(runs),
  }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

STEP_SPECS = (
  ("presence", step_presence, STEP_TIMEOUTS["presence"]),
  ("smoke", step_smoke, STEP_TIMEOUTS["smoke"]),
  ("detector", step_detector, STEP_TIMEOUTS["detector"]),
)


def collect_steps(*, timed=None, specs=STEP_SPECS, progress_file: str | None = None) -> dict:
  timed = timed or _timed
  steps: dict = {}
  ok_so_far = True
  for name, fn, timeout_s in specs:
    if not ok_so_far:
      steps[name] = {"name": name, "status": "skipped"}
      continue
    if progress_file:
      try:
        write_progress(progress_file, name)
      except Exception:
        pass
    result = timed(name, fn, timeout_s)
    result.setdefault("name", name)
    steps[name] = result
    ok_so_far = result.get("status") == "pass"
  return steps


def build_result_payload(steps: dict, *, commit: str, device_type, agnos_version,
                         timestamp: str, error=None) -> dict:
  return {
    "commit": commit,
    "timestamp": timestamp,
    "device_type": device_type,
    "agnos_version": agnos_version,
    "ok": bool(steps) and all(step.get("status") == "pass" for step in steps.values()),
    "error": _truncate(error) if error else None,
    "steps": steps,
  }


# ---------------------------------------------------------------------------
# orphan protection: PDEATHSIG + stop signals
# ---------------------------------------------------------------------------

def _child_preexec() -> None:
  """Runs in the forked child: die when the parent dies (PR_SET_PDEATHSIG=SIGKILL).

  Best-effort: a missing libc or a failing prctl is silently ignored."""
  try:
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
  except Exception:
    pass


def _default_popen(cmd, **kwargs):
  """Default Popen factory: the child is coupled to this process on Linux."""
  if IS_LINUX and "preexec_fn" not in kwargs:
    kwargs["preexec_fn"] = _child_preexec
  return _POPEN(cmd, **kwargs)


def _kill_proc(proc) -> None:
  try:
    proc.kill()
  except Exception:
    pass
  try:
    proc.wait(timeout=2.0)
  except Exception:
    pass


def _kill_running_child(wait_s: float = 2.0) -> None:
  proc = _RUNNING_CHILD
  if proc is None:
    return
  try:
    proc.kill()
  except Exception:
    pass
  try:
    proc.wait(timeout=wait_s)
  except Exception:
    pass


def _make_stop_handler(restore_attempt, notice=None):
  """SIGINT/SIGTERM: kill the child, restore the attempt latch, drop the
  on-screen notice, exit clean."""

  def handler(_signum, _frame):
    _kill_running_child()
    restore_attempt()
    if notice is not None:
      notice(None)
    raise SystemExit(0)

  return handler


def _install_signal_handlers(restore_attempt, notice=None) -> None:
  handler = _make_stop_handler(restore_attempt, notice)
  for sig in (signal.SIGINT, signal.SIGTERM):
    try:
      _SIGNAL(sig, handler)
    except (ValueError, OSError):
      pass


# ---------------------------------------------------------------------------
# on-screen notice (params_memory) - contract in ui/onroad/tlv_dsp_notice.py
# ---------------------------------------------------------------------------

def make_notice_sink(params_factory=None):
  """Best-effort CarrotTlvDspNotice publisher: sink(dict) puts, sink(None) removes."""
  try:
    if params_factory is None:
      from openpilot.common.params import Params
      params = Params("/dev/shm/params")
    else:
      params = params_factory()
  except Exception:
    return lambda payload: None

  def sink(payload) -> None:
    try:
      if payload is None:
        params.remove(NOTICE_KEY)
      else:
        params.put(NOTICE_KEY, payload)
    except Exception:
      pass

  return sink


def _notice_payload(state: str, now: float, **fields) -> dict:
  payload = {"state": state, "ts": now}
  payload.update(fields)
  return payload


_STEP_INDEX = {name: index for index, (name, _fn, _timeout) in enumerate(STEP_SPECS, start=1)}


def _step_index_from_progress(progress_file: str | None, last: int = 1) -> int:
  if not progress_file:
    return last
  step = read_progress_step(progress_file)
  return _STEP_INDEX.get(step or "", last)


def _publish_outcome_notice(result_path: str, now: float, notice) -> None:
  """done when the written result is ok, otherwise failed (best-effort)."""
  if notice is None:
    return
  ok = False
  try:
    with open(result_path, encoding="utf-8") as f:
      payload = json.load(f)
    ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
  except (OSError, ValueError):
    ok = False
  notice(_notice_payload("done" if ok else "failed", now))


# ---------------------------------------------------------------------------
# parked gate loop
# ---------------------------------------------------------------------------

def _make_submaster():
  from openpilot.cereal import messaging
  return messaging.SubMaster(list(PARK_SERVICES), conflate=True)


def _parked(sm, now: float) -> bool:
  """PARKED per the probe contract: park gear, standstill, slow, disengaged, fresh."""
  try:
    for service in PARK_SERVICES:
      if not sm.seen[service]:
        return False
      if now - sm.recv_time[service] > PARK_FRESH_S:
        return False
    car = sm["carState"]
    sd = sm[SD_SERVICE]
    return (str(car.gearShifter) == "park"
            and bool(car.standstill)
            and float(car.vEgo) < 0.1
            and not bool(sd.enabled))
  except Exception:
    return False


def wait_for_parked(sm, *, clock=time.monotonic, sleep=time.sleep,
                    park_hold_s: float = PARK_HOLD_S, notice=None) -> None:
  """Block at GATE_HZ until PARKED held continuously for park_hold_s.

  While arming, refreshes the on-screen notice at 2 Hz once PARKED has held
  ARMING_SHOW_S (a brief P does not flash it). If PARKED drops before the
  child spawns, the notice is marked aborted only when it had been shown;
  otherwise the key is removed."""
  since: float | None = None
  shown = False
  while True:
    sm.update(0)
    now = clock()
    if _parked(sm, now):
      if since is None:
        since = now
      held = now - since
      if notice is not None and held >= ARMING_SHOW_S:
        notice(_notice_payload("arming", now, deadline=since + park_hold_s))
        shown = True
      if held >= park_hold_s:
        return
    else:
      if since is not None and notice is not None:
        if shown:
          notice(_notice_payload("aborted", now))
        else:
          notice(None)
      since = None
      shown = False
    sleep(GATE_PERIOD_S)


def _child_rss_bytes(proc):
  pid = getattr(proc, "pid", None)
  if pid is None:
    return None
  return resources.read_rss_bytes(f"/proc/{int(pid)}/statm")


def monitor_child(proc, sm, *, clock=time.monotonic, sleep=time.sleep,
                  total_timeout_s: float = TOTAL_TIMEOUT_S, poll_s: float = RUN_POLL_S,
                  rss_poll_s: float = RSS_POLL_S, rss_limit: int = CHILD_RSS_LIMIT_BYTES,
                  rss_reader=None, progress_file: str | None = None,
                  notice=None) -> tuple[str, object]:
  """Watch a running child: ("exited"|"unparked"|"timeout"|"rss", detail).

  While the child runs, the notice sink (when given) is refreshed at 2 Hz with
  the child's current step from its .progress file."""
  rss_reader = rss_reader or _child_rss_bytes
  spawn_t = clock()
  next_rss = spawn_t
  next_notice = spawn_t
  step = 1
  while True:
    code = proc.poll()
    if code is not None:
      return ("exited", code)
    sm.update(0)
    now = clock()
    if now - spawn_t >= total_timeout_s:
      return ("timeout", None)
    if not _parked(sm, now):
      return ("unparked", None)
    if notice is not None and now >= next_notice:
      next_notice = now + NOTICE_PERIOD_S
      step = _step_index_from_progress(progress_file, step)
      notice(_notice_payload("running", now, step=step, steps=len(STEP_SPECS)))
    if now >= next_rss:
      next_rss = now + rss_poll_s
      rss = rss_reader(proc)
      if rss is not None and rss > rss_limit:
        return ("rss", rss)
    sleep(poll_s)


def run_probe(result_path: str, *, sm, attempt_file: str, pre_state: dict | None,
              progress_file: str, total_timeout_s: float = TOTAL_TIMEOUT_S,
              popen_factory=None, child_cmd=None, clock=time.monotonic,
              sleep=time.sleep, rss_reader=None, meta: dict | None = None,
              notice=None) -> dict:
  """Gate loop: spawn the child only while PARKED and watch PARKED + RSS.

  A non-PARKED sample (or a stale message) kills the child without a result
  and restores the attempt latch (notice aborted). A cleanly finished child
  leaves its result file; any other outcome gets a parent-written failure
  result (notice failed). The notice writes are best-effort.
  """
  global _RUNNING_CHILD
  factory = popen_factory or _default_popen
  cmd = child_cmd or _default_child_cmd(result_path)
  while True:
    wait_for_parked(sm, clock=clock, sleep=sleep, notice=notice)
    state = pre_state or {"count": 0, "lost_steps": []}
    write_attempt_state(attempt_file, {"count": int(state.get("count", 0)) + 1,
                                       "lost_steps": list(state.get("lost_steps", []))})
    proc = factory(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _RUNNING_CHILD = proc
    try:
      outcome, detail = monitor_child(proc, sm, clock=clock, sleep=sleep,
                                      total_timeout_s=total_timeout_s, rss_reader=rss_reader,
                                      progress_file=progress_file, notice=notice)
    finally:
      _RUNNING_CHILD = None

    if outcome == "unparked":
      if notice is not None:
        notice(_notice_payload("aborted", clock()))
      _kill_proc(proc)
      # Ignition/park stop: not a probe failure, do not consume the attempt.
      restore_attempt_state(attempt_file, pre_state)
      _remove_file(progress_file)
      continue

    _kill_proc(proc)
    if outcome == "exited" and os.path.exists(result_path):
      _cleanup_latch(attempt_file, progress_file)
      _publish_outcome_notice(result_path, clock(), notice)
      return {"status": "done"}
    if outcome == "timeout":
      error = "child_timeout"
    elif outcome == "rss":
      error = "child_rss_limit"
    else:
      error = f"child_exit_{detail}"
    try:
      write_result_atomic(result_path, _result_payload(error=error, meta=meta))
    except Exception as exc:
      print(f"dsp_probe: cannot write result: {exc!r}", flush=True)
    _publish_outcome_notice(result_path, clock(), notice)
    _cleanup_latch(attempt_file, progress_file)
    return {"status": outcome}


def _default_child_cmd(result_path: str) -> list[str]:
  return [sys.executable, "-m", "openpilot.selfdrive.carrot.tlv.dsp_probe", "--child", "--result", result_path]


def _child_main(result_path: str) -> None:
  payload = build_result_payload(
    collect_steps(progress_file=progress_path(result_path)),
    commit=current_commit_short(),
    device_type=_device_type(),
    agnos_version=_agnos_version(),
    timestamp=_timestamp_iso(),
  )
  write_result_atomic(result_path, payload)


def _parse_argv(argv: list[str]) -> dict:
  args = {"child": False, "result": None}
  if "--child" in argv:
    args["child"] = True
  if "--result" in argv:
    index = argv.index("--result")
    if index + 1 < len(argv):
      args["result"] = argv[index + 1]
  return args


def main() -> None:
  global _probe_disabled_by_lost_attempts
  result_path = probe_result_path()
  if os.path.exists(result_path):
    print(f"dsp_probe: result already exists for this commit: {result_path}", flush=True)
    return

  attempt_file = attempt_path(result_path)
  progress_file = progress_path(result_path)
  notice = make_notice_sink()
  pre_state = read_attempt_state(attempt_file)
  if pre_state is not None:
    step = read_progress_step(progress_file)
    if step:
      pre_state["lost_steps"].append(step)
      # Consume the lost run before the latch check: a power cut without a
      # signal must not count the same .progress step again on the next boot.
      try:
        write_attempt_state(attempt_file, pre_state)
        _remove_file(progress_file)
      except Exception as exc:
        print(f"dsp_probe: cannot persist lost-step accounting: {exc!r}", flush=True)
    latched = _latched_step(pre_state["lost_steps"])
    if latched is not None:
      _probe_disabled_by_lost_attempts = True
      try:
        write_result_atomic(result_path, _result_payload(error=f"lost_in_step:{latched}"))
      except Exception as exc:
        print(f"dsp_probe: cannot write result: {exc!r}", flush=True)
      _publish_outcome_notice(result_path, time.monotonic(), notice)
      _cleanup_latch(attempt_file, progress_file)
      print(f"dsp_probe: step {latched!r} lost {LOST_STEP_LIMIT} times; disabling the probe", flush=True)
      return

  def _restore_pre_spawn() -> None:
    restore_attempt_state(attempt_file, pre_state)
    _remove_file(progress_file)

  _install_signal_handlers(_restore_pre_spawn, notice)
  print(f"dsp_probe: start cpu-policy={apply_cpu_policy()}", flush=True)
  resources.write_oom_score_adj(1000)

  sm = None
  warned = False
  while sm is None:
    # Messaging may not be up yet right after startup: retry cheaply instead of
    # exiting (a clean exit while should_run stays true would restart-loop).
    try:
      sm = _make_submaster()
    except Exception as exc:
      if not warned:
        warned = True
        print(f"dsp_probe: awaiting messaging: {exc!r}", flush=True)
      time.sleep(GATE_PERIOD_S)

  try:
    status = run_probe(result_path, sm=sm, attempt_file=attempt_file, pre_state=pre_state,
                       progress_file=progress_file, notice=notice)
    print(f"dsp_probe: {status.get('status')} -> {result_path}", flush=True)
  except SystemExit:
    raise  # manager stop: no failure result, latch restored by the handler
  except Exception as exc:
    try:
      write_result_atomic(result_path, _result_payload(error=_truncate(repr(exc))))
    except Exception:
      pass
    _publish_outcome_notice(result_path, time.monotonic(), notice)
    _cleanup_latch(attempt_file, progress_file)
    print(f"dsp_probe: failed: {exc!r}", flush=True)


if __name__ == "__main__":
  _args = _parse_argv(sys.argv[1:])
  if _args["child"] and _args["result"]:
    _child_main(_args["result"])
  else:
    main()
