"""Onroad one-time Hexagon DSP probe for the carrot TLV detector.

Runs as the manager process "tlv_dsp_probe" while the car is started
(onroad). The manager starts it when started AND TrafficLightDetectMode >= 1
(W129: no separate vision toggle) and the result file snpe2.json does not
exist; after the result is written the process exits 0 and the manager does not
restart it. The probe runs ONCE in total, not once per commit. There is no offroad window (the car cuts OBD power at ignition
off), so the probe works during ordinary STOPS of a normal drive.

The probe benchmarks SNPE, not tinygrad: presence checks the DSP nodes, the
SNPE bundle under third_party/snpe_tlv, the two detector DLCs with their head
spec and clang++; build compiles src/tlv_snpe_bench.cc on the device; frames
samples up to 30 far + 30 near JPEGs from the newest tlvrec sessions (still
encode the model input 640x320 RGB) and writes them as float32 NHWC raws;
cpu_f32, dsp_q and dsp_f32 run the float DLC on the SNPE CPU runtime and both
DLCs on the Hexagon DSP over that same frame list, decode every dump with the
production decoder and record detection agreement against cpu_f32.

Steps resume across stops: snpe2.steps.json stores each step's result as soon
as it finishes, and a final status (pass/fail/timeout) is never re-run. The
built runner and the dumps persist under bin/snpe2/, so a cpu run from one red
light still diffs against a dsp run from the next. The result JSON is written
only once every step is final; steps.json and bin/snpe2 are then removed.

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
params_memory (arming -> running step 1..6 -> done/failed, aborted when STOPPED
drops) for the onroad HUD capsule; contract in ui/onroad/tlv_dsp_notice.py.
Every notice write is best-effort and never affects the probe outcome.

Lost-step latch: the child writes snpe2.progress (step name + monotonic
start, atomic) before each step. At startup an existing snpe2.attempt means
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
PROBE_KEY = "snpe2"
TOTAL_TIMEOUT_S = 600.0
STEP_TIMEOUTS = {"presence": 15.0, "build": 150.0, "frames": 60.0, "cpu_f32": 150.0,
                 "dsp_q": 120.0, "dsp_f32": 240.0}
BUILD_TIMEOUT_S = 140.0
# Each bench subprocess timeout is its step timeout minus 10 s.
BENCH_TIMEOUT_MARGIN_S = 10.0
# Bench steps: name -> (DLC file, SNPE runtime). Order is the STEP_SPECS order.
DET_MODELS = {"cpu_f32": ("tl_detector_r8s50a40_f32.dlc", "cpu"),
              "dsp_q": ("tl_detector_r8s50a40_q.dlc", "dsp"),
              "dsp_f32": ("tl_detector_r8s50a40_f32.dlc", "dsp")}
HEADS_NAME = "tl_detector_r8s50a40_heads.json"
HEADS_CANON = ("p3_box", "p3_cls", "p4_box", "p4_cls", "p5_box", "p5_cls")
SCORE_THRESHOLD = 0.25
HEADS_TOTAL = 273000
INPUT_BYTES = 320 * 640 * 3 * 4
FRAMES_PER_VIEW = 30
FRAME_STRIDE = 3
MIN_FRAMES = 10
MODEL_H, MODEL_W = 320, 640
IOU_MATCH = 0.5
CONF_SCORE = 0.5
CDSPRPC_DIRS = ("/usr/lib/aarch64-linux-gnu", "/usr/lib", "/usr/lib64", "/vendor/lib64",
                "/system/vendor/lib64", "/system/lib64", "/odm/lib64")
PRESENCE_PATHS = ("/dev/adsprpc-smd", "/dsp/cdsp/fastrpc_shell_3", "/dev/ion")


def snpe_paths(basedir: str | None = None) -> dict[str, str]:
  root = os.path.join(basedir or _basedir(), "third_party", "snpe_tlv")
  return {"root": root, "lib": os.path.join(root, "larch64"), "dsp": os.path.join(root, "dsp"),
          "include": os.path.join(root, "include"), "src": os.path.join(root, "src", "tlv_snpe_bench.cc"),
          "dlc": os.path.join(root, "models", "dmonitoring_model_q.dlc")}


def models_dir() -> str:
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def det_model_path(step: str) -> str:
  return os.path.join(models_dir(), DET_MODELS[step][0])


def heads_json_path() -> str:
  return os.path.join(models_dir(), HEADS_NAME)


def bench_timeout_s(step: str) -> float:
  return STEP_TIMEOUTS[step] - BENCH_TIMEOUT_MARGIN_S


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
  for step in DET_MODELS:
    snpe_files[os.path.basename(det_model_path(step))] = path_status(det_model_path(step))
  snpe_files[HEADS_NAME] = path_status(heads_json_path())

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
# step c: sample tlvrec frames as model inputs
# ---------------------------------------------------------------------------

def snpe_env(paths) -> dict:
  env = dict(os.environ)
  old = env.get("LD_LIBRARY_PATH", "")
  env["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:" + paths["lib"] + (":" + old if old else "")
  env["ADSP_LIBRARY_PATH"] = paths["dsp"] + "/"
  return env


def frame_work_dir() -> str:
  return os.path.join(bin_dir(), "raw")


def dump_dir(step: str) -> str:
  return os.path.join(bin_dir(), f"dump_{step}")


def list_path(step: str) -> str:
  return os.path.join(bin_dir(), f"list_{step}.txt")


def frames_json_path() -> str:
  return os.path.join(bin_dir(), "frames.json")


def _read_index(path: str) -> list[dict]:
  rows: list[dict] = []
  try:
    with open(path, encoding="utf-8") as f:
      for line in f:
        try:
          row = json.loads(line)
        except ValueError:
          continue
        if isinstance(row, dict):
          rows.append(row)
  except OSError:
    return []
  return rows


def _read_active(root: str) -> str | None:
  from openpilot.selfdrive.carrot.tlv import recorder
  try:
    with open(os.path.join(root, recorder.ACTIVE_NAME), encoding="utf-8") as f:
      return f.read().strip() or None
  except OSError:
    return None


def tlvrec_sessions(root: str) -> list[str]:
  """Finished sessions: sub-dirs with index.jsonl, newest name first.

  The session named in root/.active is still being written and dsp_probe
  itself lives under root, so both are excluded."""
  active = _read_active(root)
  try:
    names = sorted(os.listdir(root), reverse=True)
  except OSError:
    return []
  out = []
  for name in names:
    if name == active or name == "dsp_probe":
      continue
    path = os.path.join(root, name)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "index.jsonl")):
      out.append(path)
  return out


def step_frames(*, root: str | None = None, work: str | None = None) -> dict:
  """Up to FRAMES_PER_VIEW far + near JPEGs from the newest tlvrec sessions."""
  import hashlib

  import cv2
  import numpy as np

  from openpilot.selfdrive.carrot.tlv import recorder

  root = root or recorder.DEFAULT_ROOT
  raw_dir = work or frame_work_dir()
  os.makedirs(raw_dir, exist_ok=True)

  picked: list[dict] = []
  counts = {"far": 0, "near": 0}
  bad_shape = 0
  sessions_used: list[str] = []
  for session in tlvrec_sessions(root):
    if counts["far"] >= FRAMES_PER_VIEW and counts["near"] >= FRAMES_PER_VIEW:
      break
    rows = _read_index(os.path.join(session, "index.jsonl"))
    used_here = False
    for index in range(0, len(rows), FRAME_STRIDE):
      row = rows[index]
      view = row.get("view")
      if view not in ("far", "near") or counts[view] >= FRAMES_PER_VIEW:
        continue
      name = row.get("file")
      if not isinstance(name, str) or not name:
        continue
      path = os.path.join(session, name)
      if not os.path.isfile(path):
        continue
      bgr = cv2.imread(path, cv2.IMREAD_COLOR)
      if bgr is None:
        bad_shape += 1
        continue
      if bgr.shape[:2] != (MODEL_H, MODEL_W):
        bad_shape += 1
        continue
      rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
      i = len(picked)
      raw_name = f"{i:04d}.raw"
      with open(os.path.join(raw_dir, raw_name), "wb") as f:
        f.write(rgb.astype(np.float32).tobytes())
      size = os.path.getsize(os.path.join(raw_dir, raw_name))
      if size != INPUT_BYTES:
        return {"status": "fail", "error": f"raw_size:{raw_name}:{size}"}
      with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
      picked.append({"i": i, "session": os.path.basename(session), "file": name,
                     "view": view, "jpeg_sha256": digest})
      counts[view] += 1
      used_here = True
    if used_here:
      sessions_used.append(os.path.basename(session))

  if len(picked) < MIN_FRAMES:
    return {"status": "fail", "error": f"too_few_frames:{len(picked)}",
            "n_far": counts["far"], "n_near": counts["near"], "bad_shape": bad_shape}

  write_result_atomic(frames_json_path(), picked)
  for step in DET_MODELS:
    os.makedirs(dump_dir(step), exist_ok=True)
    lines = [f"{os.path.join(raw_dir, f'{i:04d}.raw')} {os.path.join(dump_dir(step), f'{i:04d}.f32')}"
             for i in range(len(picked))]
    with open(list_path(step), "w", encoding="utf-8", newline="\n") as f:
      f.write("\n".join(lines) + "\n")

  return {"status": "pass", "n_far": counts["far"], "n_near": counts["near"],
          "bad_shape": bad_shape, "sessions_used": sessions_used}


# ---------------------------------------------------------------------------
# steps d/e/f: the two detector DLCs on the SNPE cpu and dsp runtimes
# ---------------------------------------------------------------------------

def _bench_argv(work: dict, step: str, runtime: str | None = None) -> tuple[list[str], str]:
  runtime = runtime or DET_MODELS[step][1]
  out = os.path.join(work.get("dir") or _new_work()["dir"], f"{step}.json")
  cmd = [work.get("bin") or bin_path(), "--dlc", det_model_path(step), "--runtime", runtime,
         "--runs", "1", "--out", out, "--input_list", list_path(step)]
  return cmd, out


def load_heads(path: str | None = None) -> tuple[int, dict]:
  with open(path or heads_json_path(), encoding="utf-8") as f:
    spec = json.load(f)
  return int(spec["total"]), spec["heads"]


def split_heads(flat, heads: dict) -> list:
  """The six head arrays in HEADS_CANON order, reshaped to heads.json dims (NCHW)."""
  import numpy as np

  return [np.asarray(flat[h["offset"]:h["offset"] + h["count"]], dtype=np.float32).reshape(h["dims"])
          for h in (heads[name] for name in HEADS_CANON)]


def _decode_dumps(step: str, frames: list[dict]) -> tuple[list[list], list[float], str | None]:
  """Decode every dump of `step`; (dets per frame, every score, error)."""
  import numpy as np

  from openpilot.selfdrive.carrot.tlv.decode import make_decoder

  total, heads = load_heads()
  decoder = make_decoder(list(HEADS_CANON), SCORE_THRESHOLD)
  dets: list[list] = []
  scores: list[float] = []
  for row in frames:
    path = os.path.join(dump_dir(step), f"{int(row['i']):04d}.f32")
    size = os.path.getsize(path)
    if size != HEADS_TOTAL * 4:
      return dets, scores, f"dump_size:{os.path.basename(path)}:{size}"
    flat = np.fromfile(path, dtype="<f4")
    outs = split_heads(flat, heads)
    got = decoder.decode(outs, str(row.get("view") or "far"))
    dets.append([[int(d.cls), round(float(d.score), 4), round(float(d.x1), 2), round(float(d.y1), 2),
                  round(float(d.x2), 2), round(float(d.y2), 2)] for d in got])
    scores.extend(float(d.score) for d in got)
  return dets, scores, None


def _head_deltas(step: str, frames: list[dict]) -> dict:
  """Per-head max and mean |dsp - cpu| over the frames both sides dumped."""
  import numpy as np

  _, heads = load_heads()
  per_head = {name: {"max": 0.0, "mean": 0.0, "frames": 0} for name in HEADS_CANON}
  for row in frames:
    i = int(row["i"])
    cpu_path = os.path.join(dump_dir("cpu_f32"), f"{i:04d}.f32")
    cand_path = os.path.join(dump_dir(step), f"{i:04d}.f32")
    if not os.path.exists(cpu_path) or not os.path.exists(cand_path):
      continue
    cpu = np.fromfile(cpu_path, dtype="<f4")
    cand = np.fromfile(cand_path, dtype="<f4")
    if cpu.size != HEADS_TOTAL or cand.size != HEADS_TOTAL:
      continue
    for name in HEADS_CANON:
      h = heads[name]
      a = cpu[h["offset"]:h["offset"] + h["count"]]
      b = cand[h["offset"]:h["offset"] + h["count"]]
      diff = np.abs(a - b)
      entry = per_head[name]
      entry["max"] = max(entry["max"], float(diff.max()) if diff.size else 0.0)
      entry["mean"] += float(diff.mean()) if diff.size else 0.0
      entry["frames"] += 1
  for entry in per_head.values():
    entry["mean"] = round(entry["mean"] / entry["frames"], 6) if entry["frames"] else 0.0
    entry["max"] = round(entry["max"], 6)
  return per_head


def step_bench(step: str, *, run=None, paths=None, work=None, command_prefix=None,
               runtime_override: str | None = None) -> dict:
  """One bench run over the frame list, then decode every dump.

  `command_prefix` (e.g. ["qemu-aarch64-static", "-L", "/usr/aarch64-linux-gnu"])
  is prepended to the bench argv for the PC rehearsal; None means run as-is.
  `runtime_override` swaps the step's SNPE runtime (the rehearsal runs the int8
  DLC on the cpu runtime, where no DSP exists); None keeps the step's own. The
  per-head diff is always against cpu_f32's own DLC, i.e. int8-vs-float head
  content, never a tautology."""
  run = run or subprocess.run
  p = paths or snpe_paths()
  work = work if work is not None else (_WORK or _new_work())
  cmd, out = _bench_argv(work, step, runtime_override)
  if command_prefix:
    cmd = list(command_prefix) + cmd
  timeout_s = bench_timeout_s(step)
  try:
    proc = run(cmd, capture_output=True, timeout=timeout_s, env=snpe_env(p), preexec_fn=_preexec())
  except subprocess.TimeoutExpired:
    return {"status": "timeout", "error": f"bench timed out after {timeout_s:.0f} s"}
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

  result: dict = {"returncode": rc, "status": "fail"}
  if rc == 2:
    result["error"] = "runtime_unavailable"
    return result
  if not bench:
    stderr = getattr(proc, "stderr", b"") or b""
    result["stderr"] = _truncate(stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr))
    result["error"] = f"no result json (exit {rc})"
    return result

  for key in ("status", "error", "runtime_available", "load_s", "first_ms", "median_ms",
              "p90_ms", "frames", "frame_ms"):
    if key in bench:
      result[key] = bench[key]
  if str(bench.get("status")) != "pass":
    result["status"] = "fail"
    result["error"] = _truncate(bench.get("error") or "bench_failed")
    return result

  frames = read_frames()
  if len(frames) != int(bench.get("frames") or 0):
    result["status"] = "fail"
    result["error"] = f"frames_mismatch:{len(frames)}!={bench.get('frames')}"
    return result

  dets, scores, error = _decode_dumps(step, frames)
  if error:
    result["status"] = "fail"
    result["error"] = error
    return result
  result["dets"] = dets
  result["score_min"] = min(scores) if scores else None
  result["score_max"] = max(scores) if scores else None
  result["n_dets"] = sum(len(d) for d in dets)

  if step != "cpu_f32":
    # Always meaningful: cpu_f32 is the float DLC, so the diff is int8-vs-float
    # head content even when the rehearsal reruns the int8 DLC on the cpu.
    if os.path.isdir(dump_dir("cpu_f32")):
      result["head_delta"] = _head_deltas(step, frames)
    shutil.rmtree(dump_dir(step), ignore_errors=True)

  result["status"] = "pass"
  return result


def read_frames() -> list[dict]:
  try:
    with open(frames_json_path(), encoding="utf-8") as f:
      data = json.load(f)
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
  except (OSError, ValueError):
    return []


def step_cpu_f32(**kw) -> dict:
  return step_bench("cpu_f32", **kw)


def step_dsp_q(**kw) -> dict:
  return step_bench("dsp_q", **kw)


def step_dsp_f32(**kw) -> dict:
  return step_bench("dsp_f32", **kw)


# ---------------------------------------------------------------------------
# detection agreement (pure)
# ---------------------------------------------------------------------------

def iou(a, b) -> float:
  x1, y1 = max(a[0], b[0]), max(a[1], b[1])
  x2, y2 = min(a[2], b[2]), min(a[3], b[3])
  w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
  inter = w * h
  ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return inter / ua if ua > 0 else 0.0


def match(ref: list, cand: list) -> tuple[list, list, list]:
  """Greedy by ref score descending; (pairs, missed ref idx, used cand).

  Ported verbatim from data/wsnpe6/scripts/det_eval.py: greedy over refs sorted
  by score desc, each takes the best free candidate of the same class with
  IoU >= IOU_MATCH; dets are [cls, score, x1, y1, x2, y2]."""
  order = sorted(range(len(ref)), key=lambda i: -ref[i][1])
  used = [False] * len(cand)
  pairs, missed = [], []
  for ri in order:
    best, best_iou = -1, IOU_MATCH
    for ci, c in enumerate(cand):
      if used[ci] or c[0] != ref[ri][0]:
        continue
      v = iou((ref[ri][2], ref[ri][3], ref[ri][4], ref[ri][5]), (c[2], c[3], c[4], c[5]))
      if v >= best_iou:
        best, best_iou = ci, v
    if best < 0:
      missed.append(ri)
    else:
      used[best] = True
      pairs.append((ri, best, best_iou))
  return pairs, missed, used


def _median(sorted_values: list[float]) -> float:
  """True median (mean of the middle pair for an even count), as det_eval's np.median."""
  n = len(sorted_values)
  mid = n // 2
  return sorted_values[mid] if n % 2 else 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _agreement(ref_frames: list, cand_frames: list) -> dict:
  n_ref = n_cand = matched = conf_ref = conf_missed = 0
  score_deltas: list[float] = []
  ious: list[float] = []
  for ref, cand in zip(ref_frames, cand_frames, strict=False):
    pairs, missed, used = match(ref, cand)
    n_ref += len(ref)
    n_cand += len(cand)
    matched += len(pairs)
    for ri, ci, v in pairs:
      score_deltas.append(abs(ref[ri][1] - cand[ci][1]))
      ious.append(v)
    for ri in missed:
      if ref[ri][1] >= CONF_SCORE:
        conf_ref += 1
        conf_missed += 1
    for ri in range(len(ref)):
      if ref[ri][1] >= CONF_SCORE and ri not in missed:
        conf_ref += 1
  score_deltas.sort()
  out = {
    "ref": n_ref,
    "cand": n_cand,
    "matched": matched,
    "miss_pct": round(100.0 * (n_ref - matched) / n_ref, 2) if n_ref else 0.0,
    "extra_pct": round(100.0 * (n_cand - matched) / n_cand, 2) if n_cand else 0.0,
    "conf_ref": conf_ref,
    "conf_missed": conf_missed,
    "median_score_delta": round(_median(score_deltas), 4) if score_deltas else None,
    "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
  }
  return out


def compare_steps(steps: dict) -> dict:
  """dsp_q / dsp_f32 vs cpu_f32, per view and total (pure)."""
  ref = steps.get("cpu_f32") or {}
  if ref.get("status") != "pass":
    return {"ref": "cpu_f32", "status": "unavailable"}
  frames = read_frames()
  out: dict = {"ref": "cpu_f32", "steps": {}, "views": {}}
  ref_dets = ref.get("dets") or []
  views = list(dict.fromkeys(str(row.get("view") or "far") for row in frames))
  for step in ("dsp_q", "dsp_f32"):
    cand = steps.get(step) or {}
    if cand.get("status") != "pass":
      out["steps"][step] = {"status": str(cand.get("status") or "missing")}
      continue
    cand_dets = cand.get("dets") or []
    if len(cand_dets) != len(ref_dets):
      out["steps"][step] = {"status": "frames_mismatch",
                            "ref_frames": len(ref_dets), "cand_frames": len(cand_dets)}
      continue
    entry = {"status": "pass", "total": _agreement(ref_dets, cand_dets), "views": {}}
    for view in views:
      idx = [i for i, row in enumerate(frames) if str(row.get("view") or "far") == view]
      entry["views"][view] = _agreement([ref_dets[i] for i in idx], [cand_dets[i] for i in idx])
    out["steps"][step] = entry
    out["views"][step] = entry["views"]
  return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

STEP_SPECS = (
  ("presence", step_presence, STEP_TIMEOUTS["presence"], ()),
  ("build", step_build, STEP_TIMEOUTS["build"], ("presence",)),
  ("frames", step_frames, STEP_TIMEOUTS["frames"], ("presence",)),
  ("cpu_f32", step_cpu_f32, STEP_TIMEOUTS["cpu_f32"], ("build", "frames")),
  ("dsp_q", step_dsp_q, STEP_TIMEOUTS["dsp_q"], ("build", "frames")),
  ("dsp_f32", step_dsp_f32, STEP_TIMEOUTS["dsp_f32"], ("build", "frames")),
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
  """A stored step only counts while the files it produced are still on disk."""
  if name == "build":
    return os.path.exists(bin_path())
  if name == "frames":
    return os.path.exists(frames_json_path()) and all(os.path.exists(list_path(s)) for s in DET_MODELS)
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
  payload = {
    "commit": commit,
    "timestamp": timestamp,
    "device_type": device_type,
    "agnos_version": agnos_version,
    "ok": bool(steps) and all(step.get("status") == "pass" for step in steps.values()),
    "error": _truncate(error) if error else None,
    "steps": steps,
  }
  if steps:
    payload["compare"] = compare_steps(steps)
  return payload


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
  return messaging.SubMaster(list(STOP_SERVICES))


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
    frames = read_frames()
    if frames and isinstance(steps.get("frames"), dict):
      steps["frames"]["frames"] = frames
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
