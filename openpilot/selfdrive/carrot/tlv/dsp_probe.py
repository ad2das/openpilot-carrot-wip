"""Onroad one-time Hexagon DSP probe for the carrot TLV detector.

Runs as the manager process "tlv_dsp_probe" while the car is started
(onroad). The manager starts it when started AND TrafficLightDetectMode >= 1
(W129: no separate vision toggle) and the result file snpe1.json does not
exist; after the result is written the process exits 0 and the manager does not
restart it. The probe runs ONCE in total, not once per commit. There is no offroad window (the car cuts OBD power at ignition
off), so the probe works during ordinary STOPS of a normal drive.

The probe benchmarks SNPE, not tinygrad: presence checks the DSP nodes, the
SNPE bundle under third_party/snpe_tlv and clang++; build compiles
src/tlv_snpe_bench.cc on the device; snpe_cpu and snpe_dsp run comma's
quantized DM model on both SNPE runtimes and compare their dumps.

Steps resume across stops: snpe1.steps.json stores each step's result as soon
as it finishes, and a final status (pass/fail/timeout) is never re-run. The
built runner and the dumps persist under bin/snpe1/, so a cpu run from one red
light still diffs against a dsp run from the next. The result JSON is written
only once every step is final; steps.json and bin/snpe1 are then removed.

Parent gate loop (2 Hz, tiny idle cost while driving): the parent idles on
short sleeps and only checks

  STOPPED = carState.standstill AND vEgo < 0.1,
            with both messages fresh (< 1 s old); the gear and
            selfdriveState.enabled are irrelevant.

The child is spawned only after STOPPED held continuously for 3 s. While the
child runs, the parent re-checks every 0.1 s: the first non-STOPPED sample (or
stale message) SIGKILLs the child immediately, writes no result, restores the
attempt latch and goes back to waiting - whatever the child already persisted
in steps.json is kept. Every 0.5 s it reads /proc/<child>/statm; over 300 MB
RSS the child is killed and the result is written with error="child_rss_limit".
The 360 s cap is per child spawn (paused while driving); a child that exits 0
with steps still pending just waits for the next stop, while a nonzero exit
gets a child_exit_<code> result.

Orphan protection: the child is spawned with a Linux PR_SET_PDEATHSIG=SIGKILL
preexec hook (best-effort), and main() installs SIGINT/SIGTERM handlers that
kill the running child (waiting up to 2 s) before exiting 0 WITHOUT writing a
result - a manager stop is not a probe failure. All results are written
atomically (tmp + fsync + rename), so a kill never leaves a partial file.

UI notice: while it works the parent publishes CarrotTlvDspNotice in
params_memory (arming -> running step 1..4 -> done/failed, aborted when STOPPED
drops) for the onroad HUD capsule; contract in ui/onroad/tlv_dsp_notice.py.
Every notice write is best-effort and never affects the probe outcome.

Lost-step latch: the child writes snpe1.progress (step name + monotonic
start, atomic) before each step. At startup an existing snpe1.attempt means
the previous run was lost (power cut or device reset): the .progress step is
appended to its "lost_steps" list and consumed immediately (the updated
.attempt is persisted and .progress removed before the latch check), so a cut
without a signal cannot count the same loss twice. A 3rd loss of the SAME
step writes the final result error="lost_in_step:<step>" and sets
_probe_disabled_by_lost_attempts. Losses spread over different steps (random
power cuts) never latch. A clean finish deletes .attempt and .progress; a
clean kill (stop ended or signal) restores .attempt to its pre-spawn content.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from openpilot.selfdrive.carrot.tlv import resources

PROBE_ROOT = "/data/media/0/carrot_tlv_rec/dsp_probe"
# One probe run in total (not one per commit): every path below hangs off this
# key, so a later openpilot update does not re-run the benchmark.
PROBE_KEY = "snpe1"
TOTAL_TIMEOUT_S = 360.0
STEP_TIMEOUTS = {"presence": 15.0, "build": 150.0, "snpe_cpu": 60.0, "snpe_dsp": 90.0}
BUILD_TIMEOUT_S = 140.0
RUNNER_TIMEOUT_S = {"cpu": 50.0, "dsp": 80.0}
SNPE_RUNS = {"cpu": 10, "dsp": 20}
CDSPRPC_DIRS = ("/usr/lib/aarch64-linux-gnu", "/usr/lib", "/usr/lib64", "/vendor/lib64",
                "/system/vendor/lib64", "/system/lib64", "/odm/lib64")
PRESENCE_PATHS = ("/dev/adsprpc-smd", "/dsp/cdsp/fastrpc_shell_3", "/dev/ion")


def snpe_paths(basedir: str | None = None) -> dict[str, str]:
  root = os.path.join(basedir or _basedir(), "third_party", "snpe_tlv")
  return {"root": root, "lib": os.path.join(root, "larch64"), "dsp": os.path.join(root, "dsp"),
          "include": os.path.join(root, "include"), "src": os.path.join(root, "src", "tlv_snpe_bench.cc"),
          "dlc": os.path.join(root, "models", "dmonitoring_model_q.dlc")}


SNPE_REQUIRED = (("lib", "libSNPE.so"), ("dsp", "libsnpe_dsp_v65_domains_v2_skel.so"),
                 ("include", os.path.join("SNPE", "SNPE.hpp")), ("src", ""), ("dlc", ""))
MAX_ERROR_CHARS = 2048

MB = 1024 * 1024
# Stop gate: any ordinary stop (red light, traffic) while driving is enough.
SD_SERVICE = "selfdriveState"  # this tree publishes selfdriveState (not controlsState)
STOP_SERVICES = ("carState", SD_SERVICE)
STOP_FRESH_S = 1.0
STOP_HOLD_S = 3.0
GATE_HZ = 2.0
GATE_PERIOD_S = 1.0 / GATE_HZ
RUN_POLL_S = 0.1
RSS_POLL_S = 0.5
CHILD_RSS_LIMIT_BYTES = 300 * MB
LOST_STEP_LIMIT = 3
# Steps whose stored status is final: they are never re-run across stops.
# "skipped" is final too: it only arises when a dependency already ended in a
# final non-pass status, and that decision must not be re-litigated per stop.
FINAL_STATUSES = ("pass", "fail", "timeout", "skipped")
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
  return os.path.join(PROBE_ROOT, f"{PROBE_KEY}.json")


def steps_path(result_path: str) -> str:
  return os.path.splitext(result_path)[0] + ".steps.json"


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
    if params.get_int("TrafficLightDetectMode") < 1:
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
# step a: DSP presence + SNPE bundle + toolchain
# ---------------------------------------------------------------------------

def _path_status(path: str) -> str:
  if not os.path.exists(path):
    return "missing"
  if not os.access(path, os.R_OK):
    return "not readable"
  return "ok"


def _cdsprpc_candidates(listdir=None) -> list[str]:
  listdir = listdir or os.listdir
  found: list[str] = []
  for directory in CDSPRPC_DIRS:
    try:
      names = listdir(directory)
    except OSError:
      continue
    found.extend(os.path.join(directory, name) for name in names if name.startswith("libcdsprpc"))
  return sorted(found)


def step_presence(*, path_status=None, which=None, listdir=None, paths=None) -> dict:
  path_status = path_status or _path_status
  which = which or shutil.which
  p = paths or snpe_paths()

  nodes = {path: path_status(path) for path in PRESENCE_PATHS}
  clangxx = which("clang++") or "missing"
  cdsprpc = _cdsprpc_candidates(listdir)

  snpe_files = {}
  for key, name in SNPE_REQUIRED:
    target = os.path.join(p[key], name) if name else p[key]
    snpe_files[f"{key}/{name}" if name else key] = path_status(target)

  error = None
  if any(status != "ok" for status in nodes.values()):
    error = f"DSP node missing: {next(k for k, v in nodes.items() if v != 'ok')}"
  elif clangxx == "missing":
    error = "clang++ not found on PATH"
  else:
    bad = [key for key, status in snpe_files.items() if status != "ok"]
    if bad:
      error = f"SNPE file missing: {bad[0]}"

  return {
    "status": "pass" if error is None else "fail",
    "paths": nodes,
    "clangxx": clangxx,
    "cdsprpc": cdsprpc,
    "snpe_files": snpe_files,
    "error": error,
  }


# ---------------------------------------------------------------------------
# step b: compile the SNPE runner on the device
# ---------------------------------------------------------------------------

_WORK: dict = {}


def bin_dir() -> str:
  return os.path.join(PROBE_ROOT, "bin", PROBE_KEY)


def bin_path() -> str:
  return os.path.join(bin_dir(), "tlv_snpe_bench")


def dump_path(runtime: str) -> str:
  return os.path.join(bin_dir(), f"{runtime}.f32")


def _preexec():
  """Death signal for the build/runner grandchildren: a pull-away SIGKILLs the
  child, and the compiler or SNPE runner must not outlive it while driving."""
  return _child_preexec if IS_LINUX else None


def _new_work() -> dict:
  """Per-child state: a throwaway dir for --out JSON plus the persistent bin."""
  _WORK.clear()
  _WORK["dir"] = tempfile.mkdtemp(prefix="tlv_snpe_")
  _WORK["bin"] = bin_path()
  return _WORK


def build_argv(clangxx: str, p: dict[str, str], out_bin: str) -> list[str]:
  # SNPE 1.41 exports the pre-cxx11 libstdc++ ABI; every zdl:: API taking a
  # std::string fails to link without the ABI flag.
  return [clangxx, "-std=c++17", "-O2", "-D_GLIBCXX_USE_CXX11_ABI=0", "-I", p["include"], p["src"],
          "-L", p["lib"], "-lSNPE", f"-Wl,-rpath,{p['lib']}", "-o", out_bin]


def step_build(*, run=None, which=None, paths=None, work=None) -> dict:
  run = run or subprocess.run
  which = which or shutil.which
  p = paths or snpe_paths()
  work = work if work is not None else (_WORK or _new_work())
  clangxx = which("clang++") or "clang++"
  out_bin = work.get("bin") or bin_path()
  work["bin"] = out_bin
  try:
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
  except OSError as exc:
    return {"status": "fail", "error": _truncate(f"cannot create {os.path.dirname(out_bin)}: {exc!r}")}
  cmd = build_argv(clangxx, p, out_bin)
  try:
    proc = run(cmd, capture_output=True, timeout=BUILD_TIMEOUT_S, preexec_fn=_preexec())
  except subprocess.TimeoutExpired:
    return {"status": "fail", "cmd": cmd, "error": f"build timed out after {BUILD_TIMEOUT_S:.0f} s"}
  except Exception as exc:
    return {"status": "fail", "cmd": cmd, "error": _truncate(repr(exc))}
  rc = int(getattr(proc, "returncode", 1))
  stderr = getattr(proc, "stderr", b"") or b""
  stderr_text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
  result = {"returncode": rc, "cmd": cmd}
  if rc != 0 or not os.path.exists(out_bin):
    result["status"] = "fail"
    result["stderr"] = _truncate(stderr_text)
    result["error"] = _truncate(stderr_text or f"clang++ returned {rc}")
    return result
  result["status"] = "pass"
  result["bin"] = out_bin
  return result


# ---------------------------------------------------------------------------
# steps c/d: the quantized DM model on the SNPE cpu and dsp runtimes
# ---------------------------------------------------------------------------

def snpe_env(paths) -> dict:
  env = dict(os.environ)
  old = env.get("LD_LIBRARY_PATH", "")
  env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:" + paths["lib"] + (":" + old if old else "")
  env["ADSP_LIBRARY_PATH"] = paths["dsp"] + "/"
  return env


def _bench_argv(work: dict, p: dict[str, str], runtime: str) -> tuple[list[str], str, str]:
  out = os.path.join(work.get("dir") or _new_work()["dir"], f"{runtime}.json")
  dump = dump_path(runtime)
  cmd = [work.get("bin") or bin_path(), "--dlc", p["dlc"], "--runtime", runtime,
         "--runs", str(SNPE_RUNS[runtime]), "--out", out, "--dump", dump]
  return cmd, out, dump


def _run_bench(runtime: str, *, run=None, paths=None, work=None) -> dict:
  run = run or subprocess.run
  p = paths or snpe_paths()
  work = work if work is not None else (_WORK or _new_work())
  cmd, out, dump = _bench_argv(work, p, runtime)
  try:
    proc = run(cmd, capture_output=True, timeout=RUNNER_TIMEOUT_S[runtime], env=snpe_env(p),
               preexec_fn=_preexec())
  except subprocess.TimeoutExpired:
    return {"status": "timeout", "error": f"runner timed out after {RUNNER_TIMEOUT_S[runtime]:.0f} s"}
  except Exception as exc:
    return {"status": "fail", "error": _truncate(repr(exc))}
  rc = int(getattr(proc, "returncode", 1))
  try:
    with open(out, encoding="utf-8") as f:
      bench = json.load(f)
    if not isinstance(bench, dict):
      bench = None
  except (OSError, ValueError):
    bench = None
  status = str(bench.get("status", "fail")) if bench else "fail"
  result = {"returncode": rc, "bench": bench}
  if status != "pass":
    stderr = getattr(proc, "stderr", b"") or b""
    result["stderr"] = _truncate(stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr))
    if bench and bench.get("error"):
      result["error"] = _truncate(bench.get("error"))
    else:
      result["error"] = f"no result json (exit {rc})"
  result["status"] = status
  return result


def step_snpe_cpu(**kw) -> dict:
  return _run_bench("cpu", **kw)


def step_snpe_dsp(**kw) -> dict:
  result = _run_bench("dsp", **kw)
  cpu_dump = dump_path("cpu")
  dsp_dump = dump_path("dsp")
  if os.path.exists(cpu_dump) and os.path.exists(dsp_dump):
    import numpy as np
    result["dump_bytes"] = [os.path.getsize(cpu_dump), os.path.getsize(dsp_dump)]
    if result["dump_bytes"][0] == result["dump_bytes"][1]:
      a = np.fromfile(cpu_dump, dtype=np.float32)
      b = np.fromfile(dsp_dump, dtype=np.float32)
      result["max_abs_diff"] = float(np.max(np.abs(a - b))) if a.size else 0.0
    else:
      result["max_abs_diff"] = None
      result["dump_note"] = "size_mismatch"
  return result


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

STEP_SPECS = (
  ("presence", step_presence, STEP_TIMEOUTS["presence"], ()),
  ("build", step_build, STEP_TIMEOUTS["build"], ("presence",)),
  ("snpe_cpu", step_snpe_cpu, STEP_TIMEOUTS["snpe_cpu"], ("build",)),
  ("snpe_dsp", step_snpe_dsp, STEP_TIMEOUTS["snpe_dsp"], ("build",)),
)


def read_steps(path: str) -> dict:
  """Stored per-step results across stops; missing or corrupt = {}."""
  try:
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    if not isinstance(data, dict):
      return {}
    return {str(name): step for name, step in data.items() if isinstance(step, dict)}
  except (OSError, ValueError):
    return {}


def write_steps(path: str, steps: dict) -> None:
  write_result_atomic(path, steps)


def _is_final(step: dict | None) -> bool:
  return bool(step) and str(step.get("status")) in FINAL_STATUSES


def _resume_valid(name: str, step: dict) -> bool:
  """A stored build only counts while its binary is still on disk."""
  if name == "build":
    return os.path.exists(bin_path())
  return True


def collect_steps(*, timed=None, specs=None, progress_file: str | None = None,
                  steps_file: str | None = None) -> dict:
  """Run the steps that are not final yet; persist after every step.

  A step stored with a final status (pass/fail/timeout) is never re-run: its
  stored dict is reused with "resumed": True, so a red light that ends the
  child early only costs the step that was running."""
  timed = timed or _timed
  specs = STEP_SPECS if specs is None else specs
  stored = read_steps(steps_file) if steps_file else {}
  steps: dict = {}
  for name, fn, timeout_s, needs in specs:
    previous = stored.get(name)
    if _is_final(previous) and _resume_valid(name, previous):
      steps[name] = {**previous, "resumed": True}
      continue
    if any(steps.get(dep, {}).get("status") != "pass" for dep in needs):
      steps[name] = {"name": name, "status": "skipped"}
      if steps_file:
        write_steps(steps_file, steps)
      continue
    if progress_file:
      try:
        write_progress(progress_file, name)
      except Exception:
        pass
    result = timed(name, fn, timeout_s)
    result.setdefault("name", name)
    steps[name] = result
    if steps_file:
      write_steps(steps_file, steps)
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


_STEP_INDEX = {name: index for index, (name, _fn, _timeout, _needs) in enumerate(STEP_SPECS, start=1)}


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
# stop gate loop
# ---------------------------------------------------------------------------

def _make_submaster():
  from openpilot.cereal import messaging
  return messaging.SubMaster(list(STOP_SERVICES), conflate=True)


def _stopped(sm, now: float) -> bool:
  """STOPPED per the probe contract: standstill and slow, messages fresh.

  Any gear and engagement count: a red light with openpilot engaged is the
  normal case."""
  try:
    for service in STOP_SERVICES:
      if not sm.seen[service]:
        return False
      if now - sm.recv_time[service] > STOP_FRESH_S:
        return False
    car = sm["carState"]
    return bool(car.standstill) and float(car.vEgo) < 0.1
  except Exception:
    return False


def wait_for_stop(sm, *, clock=time.monotonic, sleep=time.sleep,
                  stop_hold_s: float = STOP_HOLD_S, notice=None) -> None:
  """Block at GATE_HZ until STOPPED held continuously for stop_hold_s.

  While arming, refreshes the on-screen notice at 2 Hz once STOPPED has held
  ARMING_SHOW_S. If STOPPED drops before the child spawns, the notice is
  marked aborted only when it had been shown; otherwise the key is removed."""
  since: float | None = None
  shown = False
  while True:
    sm.update(0)
    now = clock()
    if _stopped(sm, now):
      if since is None:
        since = now
      held = now - since
      if notice is not None and held >= ARMING_SHOW_S:
        notice(_notice_payload("arming", now, deadline=since + stop_hold_s))
        shown = True
      if held >= stop_hold_s:
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

  "unparked" is any sample that is no longer STOPPED: the car pulled away
  from the red light. While the child runs, the notice sink (when given) is
  refreshed at 2 Hz with the child's current step from its .progress file."""
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
    if not _stopped(sm, now):
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
  """Gate loop: spawn the child only while STOPPED and watch STOPPED + RSS.

  A non-STOPPED sample (the car pulls away) kills the child without a result
  and restores the attempt latch: the steps it already persisted stay in
  steps_file, so the next stop continues where it left off and nothing is a
  probe failure. A child that exits 0 without a result is likewise "incomplete,
  wait for the next stop". Only a timeout, the RSS limit, a nonzero exit or an
  unexpected result file ends the probe with a result. Notices are arming ->
  running step n/4 while stopped, then done/failed once; the aborted notice
  marks a stop that ended before its child finished.
  """
  global _RUNNING_CHILD
  factory = popen_factory or _default_popen
  cmd = child_cmd or _default_child_cmd(result_path)
  while True:
    wait_for_stop(sm, clock=clock, sleep=sleep, notice=notice)
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

    if outcome == "unparked" or (outcome == "exited" and detail == 0
                                 and not os.path.exists(result_path)):
      if notice is not None and outcome == "unparked":
        notice(_notice_payload("aborted", clock()))
      _kill_proc(proc)
      # Stop ended / steps still pending: not a probe failure, keep the progress.
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


def all_final(steps: dict) -> bool:
  """Every declared step has a final status, so the result may be written."""
  return all(_is_final(steps.get(spec[0])) for spec in STEP_SPECS)


def _child_main(result_path: str) -> None:
  """One stop's worth of work: run the pending steps, persist each, and write
  the result only when nothing is left. The progress store survives the exit."""
  store = steps_path(result_path)
  _new_work()
  try:
    steps = collect_steps(progress_file=progress_path(result_path), steps_file=store)
    if not all_final(steps):
      print(f"dsp_probe: steps pending {sorted(k for k, v in steps.items() if not _is_final(v))}",
            flush=True)
      return
    payload = build_result_payload(
      steps,
      commit=current_commit_short(),
      device_type=_device_type(),
      agnos_version=_agnos_version(),
      timestamp=_timestamp_iso(),
    )
    write_result_atomic(result_path, payload)
    _remove_file(store)
    shutil.rmtree(bin_dir(), ignore_errors=True)
  finally:
    shutil.rmtree(_WORK.get("dir") or "", ignore_errors=True)
    _WORK.clear()


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
    print(f"dsp_probe: result already exists: {result_path}", flush=True)
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
