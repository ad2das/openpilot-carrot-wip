"""Tests for the onroad SNPE DSP probe (openpilot/selfdrive/carrot/tlv/dsp_probe.py).

No real DSP exists on the PC: every step is exercised through its injectable
dependencies (path/which/runner fakes), subprocess use is mocked and the stop
gate, step resumption, lost-step latch and UI notice sequence run on a fake
clock and a minimal fake SubMaster, so no real time passes.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import dsp_probe


def _load_json(path):
  with open(path, encoding="utf-8") as f:
    return json.load(f)


class FakeParams:
  def __init__(self, values) -> None:
    self._values = dict(values)

  def get_int(self, key):
    value = self._values.get(key)
    return None if value is None else int(value)

  def get(self, key):
    return self._values.get(key)


@pytest.fixture
def probe_root(tmp_path, monkeypatch):
  root = tmp_path / "dsp_probe"
  monkeypatch.setattr(dsp_probe, "PROBE_ROOT", str(root))
  monkeypatch.setattr(dsp_probe, "current_commit_short", lambda: "abc1234")
  return root


# ---------------------------------------------------------------------------
# fake clock / SubMaster / subprocess infrastructure
# ---------------------------------------------------------------------------

class FakeClock:
  def __init__(self, start: float = 0.0) -> None:
    self.now = float(start)

  def __call__(self) -> float:
    return self.now

  def advance(self, dt: float) -> None:
    self.now += dt


class FakeSM:
  """Minimal SubMaster: carState + selfdriveState with optional staleness."""

  def __init__(self, clock, *, gear="drive", standstill=True, vego=0.0,
               enabled=False, fresh=True) -> None:
    self.clock = clock
    self.gear = gear
    self.standstill = standstill
    self.vego = vego
    self.enabled = enabled
    self.fresh = fresh
    self.updates = 0
    base = clock() if fresh else clock() - 10.0
    self.seen = dict.fromkeys(dsp_probe.STOP_SERVICES, True)
    self.recv_time = dict.fromkeys(dsp_probe.STOP_SERVICES, base)

  def update(self, timeout=0) -> None:
    self.updates += 1
    if self.fresh:
      now = self.clock()
      for service in dsp_probe.STOP_SERVICES:
        self.recv_time[service] = now

  def __getitem__(self, key):
    if key == "carState":
      return SimpleNamespace(gearShifter=self.gear, standstill=self.standstill, vEgo=self.vego)
    if key == dsp_probe.SD_SERVICE:
      return SimpleNamespace(enabled=self.enabled)
    raise KeyError(key)


class NoticeLog:
  """Captures every notice publish; None entries are key removals."""

  def __init__(self) -> None:
    self.items: list = []

  def __call__(self, payload) -> None:
    self.items.append(payload)

  @property
  def states(self):
    return [item.get("state") if isinstance(item, dict) else None for item in self.items]

  def of_state(self, state):
    return [item for item in self.items if isinstance(item, dict) and item.get("state") == state]


class FakeProc:
  def __init__(self, poll_fn=None, pid=4242):
    self._poll_fn = poll_fn or (lambda: None)
    self.pid = pid
    self.killed = False
    self.returncode = None

  def poll(self):
    if self.killed:
      return self.returncode
    code = self._poll_fn()
    if code is not None:
      self.returncode = code
    return code

  def kill(self):
    self.killed = True
    self.returncode = -9

  def wait(self, timeout=None):
    return self.returncode


class PowerCut(BaseException):
  """Abrupt ignition-off: no signal handler runs and nothing is cleaned up."""


def _power_cut(*args, **kwargs):
  raise PowerCut()


def _advancing_sleep(clock):
  def sleep(dt):
    clock.advance(dt)
  return sleep


def _run_paths(tmp_path, name="abc1234"):
  result = str(tmp_path / f"{name}.json")
  return result, dsp_probe.attempt_path(result), dsp_probe.progress_path(result)


def _pid_alive(pid: int) -> bool:
  """False once /proc/<pid> is gone or the process is a zombie."""
  try:
    with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
      fields = f.read().split()
  except OSError:
    return False
  return len(fields) > 2 and fields[2] != "Z"


def _work_bin_file(probe_root):
  """Create the persistent runner so a stored build counts as resumable."""
  probe_root.mkdir(parents=True, exist_ok=True)
  path = os.path.join(str(probe_root), "bin", dsp_probe.PROBE_KEY, "tlv_snpe_bench")
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "wb") as f:
    f.write(b"bin")
  return path


# ---------------------------------------------------------------------------
# enable condition / result path
# ---------------------------------------------------------------------------

def test_should_run_truth_table(probe_root):
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is True  # onroad; STOPPED is gated by run_probe
  assert dsp_probe.should_run(False, params) is False  # manager not started
  assert dsp_probe.should_run(True, FakeParams({"TrafficLightDetectMode": 0})) is False
  assert dsp_probe.should_run(True, FakeParams({})) is False  # missing -> off

  probe_root.mkdir(parents=True, exist_ok=True)
  (probe_root / f"{dsp_probe.PROBE_KEY}.json").write_text("{}", encoding="utf-8")
  assert dsp_probe.should_run(True, params) is False  # one-shot result already there


def test_should_run_false_when_latched(probe_root, monkeypatch):
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", True)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is False


def _install_fake_signal_color_module(monkeypatch, requested):
  module = types.ModuleType("openpilot.selfdrive.modeld.signal_color_shadow")
  module.requested = lambda: requested
  monkeypatch.setitem(sys.modules, "openpilot.selfdrive.modeld.signal_color_shadow", module)


def test_should_run_disabled_when_signal_color_requested(probe_root, monkeypatch):
  _install_fake_signal_color_module(monkeypatch, True)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is False


def test_should_run_unchanged_when_signal_color_not_requested(probe_root, monkeypatch):
  _install_fake_signal_color_module(monkeypatch, False)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is True


def test_result_and_latch_paths_use_probe_key(probe_root):
  result = dsp_probe.probe_result_path()
  assert result == os.path.join(str(probe_root), "snpe1.json")
  assert dsp_probe.attempt_path(result) == os.path.join(str(probe_root), "snpe1.attempt")
  assert dsp_probe.progress_path(result) == os.path.join(str(probe_root), "snpe1.progress")
  assert dsp_probe.steps_path(result) == os.path.join(str(probe_root), "snpe1.steps.json")


def test_probe_paths_do_not_depend_on_commit(probe_root, monkeypatch):
  monkeypatch.setattr(dsp_probe, "current_commit_short", lambda: "abc1234")
  first = dsp_probe.probe_result_path()
  monkeypatch.setattr(dsp_probe, "current_commit_short", lambda: "def5678")
  assert dsp_probe.probe_result_path() == first
  assert dsp_probe.bin_dir() == os.path.join(str(probe_root), "bin", "snpe1")


def test_main_skips_when_result_exists(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  (probe_root / "snpe1.json").write_text("{}", encoding="utf-8")
  called = []
  monkeypatch.setattr(dsp_probe, "run_probe", lambda *a, **k: called.append(1))
  dsp_probe.main()
  assert not called


def test_current_commit_short_prefers_env(monkeypatch):
  monkeypatch.setenv("GIT_COMMIT", "0123456789abcdef")
  assert dsp_probe.current_commit_short() == "0123456"


# ---------------------------------------------------------------------------
# atomic write
# ---------------------------------------------------------------------------

def test_write_result_atomic(tmp_path):
  path = tmp_path / "r.json"
  dsp_probe.write_result_atomic(str(path), {"ok": True, "n": 1})
  assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True, "n": 1}
  assert not (tmp_path / "r.json.tmp").exists()


def test_write_result_atomic_cleans_tmp_on_failure(tmp_path):
  path = tmp_path / "r.json"
  path.write_text('{"old": true}', encoding="utf-8")
  with pytest.raises(TypeError):
    dsp_probe.write_result_atomic(str(path), {"bad": object()})
  assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
  assert not (tmp_path / "r.json.tmp").exists()


# ---------------------------------------------------------------------------
# step a: presence + SNPE bundle + toolchain
# ---------------------------------------------------------------------------

FAKE_PATHS = {"root": "/opt/snpe", "lib": "/opt/snpe/larch64", "dsp": "/opt/snpe/dsp",
              "include": "/opt/snpe/include", "src": "/opt/snpe/src/tlv_snpe_bench.cc",
              "dlc": "/opt/snpe/models/dmonitoring_model_q.dlc"}


def test_step_presence_pass(probe_root):
  result = dsp_probe.step_presence(path_status=lambda p: "ok", which=lambda n: "/usr/bin/clang++",
                                   listdir=lambda d: [], paths=FAKE_PATHS)
  assert result["status"] == "pass"
  assert result["clangxx"] == "/usr/bin/clang++"
  assert all(value == "ok" for value in result["paths"].values())
  assert all(value == "ok" for value in result["snpe_files"].values())
  assert result["error"] is None


def test_step_presence_missing_clangxx(probe_root):
  result = dsp_probe.step_presence(path_status=lambda p: "ok", which=lambda n: None,
                                   listdir=lambda d: [], paths=FAKE_PATHS)
  assert result["status"] == "fail"
  assert result["clangxx"] == "missing"
  assert "clang++" in result["error"]


def test_step_presence_names_the_missing_snpe_file(probe_root):
  def path_status(path):
    return "missing" if path.endswith("dmonitoring_model_q.dlc") else "ok"

  result = dsp_probe.step_presence(path_status=path_status, which=lambda n: "/clang++",
                                   listdir=lambda d: [], paths=FAKE_PATHS)
  assert result["status"] == "fail"
  assert result["snpe_files"]["dlc"] == "missing"
  assert "dlc" in result["error"]


def test_step_presence_cdsprpc_lists_matches_and_skips_unreadable():
  dirs = {"/usr/lib": ["libcdsprpc.so", "libc.so.6"],
          "/vendor/lib64": ["libcdsprpc_v2.so"]}

  def listdir(path):
    if path not in dirs:
      raise OSError("nope")
    return dirs[path]

  result = dsp_probe.step_presence(path_status=lambda p: "ok", which=lambda n: "/clang++",
                                   listdir=listdir, paths=FAKE_PATHS)
  assert result["status"] == "pass"
  assert result["cdsprpc"] == ["/usr/lib/libcdsprpc.so", "/vendor/lib64/libcdsprpc_v2.so"]


# ---------------------------------------------------------------------------
# step b: build the SNPE runner
# ---------------------------------------------------------------------------

def _build_workdir(tmp_path):
  return {"dir": str(tmp_path / "work"), "bin": str(tmp_path / "bin" / "tlv_snpe_bench")}


def test_step_build_argv_matches_wiring(tmp_path):
  work = _build_workdir(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  calls = []

  def fake_run(cmd, capture_output=True, timeout=None, preexec_fn="unset"):
    calls.append((cmd, preexec_fn))
    os.makedirs(os.path.dirname(work["bin"]), exist_ok=True)
    open(work["bin"], "wb").write(b"bin")
    return SimpleNamespace(returncode=0, stderr=b"")

  result = dsp_probe.step_build(run=fake_run, which=lambda n: "/usr/bin/clang++",
                                paths=FAKE_PATHS, work=work)
  assert result["status"] == "pass"
  assert calls[0][0] == ["/usr/bin/clang++", "-std=c++17", "-O2", "-D_GLIBCXX_USE_CXX11_ABI=0",
                      "-I", "/opt/snpe/include", "/opt/snpe/src/tlv_snpe_bench.cc",
                      "-L", "/opt/snpe/larch64", "-lSNPE", "-Wl,-rpath,/opt/snpe/larch64",
                      "-o", work["bin"]]
  assert work["bin"] == result["bin"]


def test_step_build_and_bench_pass_the_death_signal(tmp_path, monkeypatch):
  work = _build_workdir(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  os.makedirs(os.path.dirname(work["bin"]), exist_ok=True)
  open(work["bin"], "wb").write(b"bin")
  seen = []

  def fake_run(cmd, capture_output=True, timeout=None, env=None, preexec_fn="unset"):
    seen.append(preexec_fn)
    return SimpleNamespace(returncode=0, stderr=b"")

  monkeypatch.setattr(dsp_probe, "IS_LINUX", True)
  dsp_probe.step_build(run=fake_run, which=lambda n: "/clang++", paths=FAKE_PATHS, work=work)
  dsp_probe._run_bench("cpu", run=fake_run, paths=FAKE_PATHS, work=work)
  assert seen == [dsp_probe._child_preexec, dsp_probe._child_preexec]
  assert seen[0] is dsp_probe._child_preexec

  monkeypatch.setattr(dsp_probe, "IS_LINUX", False)
  seen.clear()
  dsp_probe.step_build(run=fake_run, which=lambda n: "/clang++", paths=FAKE_PATHS, work=work)
  dsp_probe._run_bench("cpu", run=fake_run, paths=FAKE_PATHS, work=work)
  assert seen == [None, None]


def test_step_build_failure_keeps_stderr(tmp_path):
  work = _build_workdir(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  result = dsp_probe.step_build(
    run=lambda cmd, capture_output=True, timeout=None, preexec_fn=None:
      SimpleNamespace(returncode=1, stderr=b"undefined reference"),
    which=lambda n: "/clang++", paths=FAKE_PATHS, work=work)
  assert result["status"] == "fail"
  assert result["returncode"] == 1
  assert "undefined reference" in result["stderr"]
  assert "bin" not in result


def test_step_build_timeout_message(tmp_path):
  work = _build_workdir(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)

  def fake_run(cmd, capture_output=True, timeout=None, preexec_fn=None):
    raise subprocess.TimeoutExpired(cmd, timeout or 1.0)

  result = dsp_probe.step_build(run=fake_run, which=lambda n: "/clang++", paths=FAKE_PATHS, work=work)
  assert result["status"] == "fail"
  assert result["error"] == "build timed out after 140 s"


# ---------------------------------------------------------------------------
# steps c/d: the SNPE runner on the cpu and dsp runtimes
# ---------------------------------------------------------------------------

def _bench_work(tmp_path):
  return {"dir": str(tmp_path / "work"), "bin": str(tmp_path / "bin" / "tlv_snpe_bench")}


def test_run_bench_cpu_pass_keeps_the_bench(probe_root, tmp_path):
  work = _bench_work(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  os.makedirs(os.path.dirname(work["bin"]), exist_ok=True)
  open(work["bin"], "wb").write(b"bin")
  seen = {}

  def fake_run(cmd, capture_output=True, timeout=None, env=None, preexec_fn="unset"):
    seen["cmd"] = cmd
    seen["env"] = env
    seen["preexec_fn"] = preexec_fn
    out = cmd[cmd.index("--out") + 1]
    with open(out, "w", encoding="utf-8") as f:
      json.dump({"status": "pass", "snpe_version": "1.41", "median_ms": 12.5}, f)
    return SimpleNamespace(returncode=0, stderr=b"")

  result = dsp_probe._run_bench("cpu", run=fake_run, paths=FAKE_PATHS, work=work)
  assert result["status"] == "pass"
  assert result["bench"]["median_ms"] == 12.5
  assert seen["cmd"][:4] == [work["bin"], "--dlc", "/opt/snpe/models/dmonitoring_model_q.dlc", "--runtime"]
  assert seen["cmd"][4] == "cpu"
  assert seen["cmd"][5:7] == ["--runs", "10"]


def test_run_bench_without_result_json(probe_root, tmp_path):
  work = _bench_work(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  result = dsp_probe._run_bench(
    "cpu", run=lambda cmd, capture_output=True, timeout=None, env=None, preexec_fn=None:
      SimpleNamespace(returncode=3, stderr=b""),
    paths=FAKE_PATHS, work=work)
  assert result["status"] == "fail"
  assert result["error"] == "no result json (exit 3)"
  assert result["bench"] is None


def test_snpe_env_prepends_the_snpe_dirs(probe_root, tmp_path):
  work = _bench_work(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  env = dsp_probe.snpe_env(FAKE_PATHS)
  assert env["LD_LIBRARY_PATH"].startswith("/usr/lib/aarch64-linux-gnu:/opt/snpe/larch64")
  assert env["ADSP_LIBRARY_PATH"] == "/opt/snpe/dsp/"


def test_snpe_dsp_diffs_the_dumps(probe_root, tmp_path, monkeypatch):
  work = _bench_work(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  bin_root = os.path.join(str(probe_root), "bin", dsp_probe.PROBE_KEY)
  os.makedirs(bin_root, exist_ok=True)
  np.asarray([0.0, 1.0, 2.0], dtype=np.float32).tofile(os.path.join(bin_root, "cpu.f32"))
  np.asarray([0.0, 1.5, 2.0], dtype=np.float32).tofile(os.path.join(bin_root, "dsp.f32"))

  def fake_run(cmd, capture_output=True, timeout=None, env=None, preexec_fn=None):
    out = cmd[cmd.index("--out") + 1]
    with open(out, "w", encoding="utf-8") as f:
      json.dump({"status": "pass"}, f)
    return SimpleNamespace(returncode=0, stderr=b"")

  result = dsp_probe.step_snpe_dsp(run=fake_run, paths=FAKE_PATHS, work=work)
  assert result["status"] == "pass"
  assert result["max_abs_diff"] == pytest.approx(0.5)
  assert result["dump_bytes"] == [12, 12]


def test_snpe_dsp_reports_size_mismatch(probe_root, tmp_path):
  work = _bench_work(tmp_path)
  os.makedirs(work["dir"], exist_ok=True)
  bin_root = os.path.join(str(probe_root), "bin", dsp_probe.PROBE_KEY)
  os.makedirs(bin_root, exist_ok=True)
  np.asarray([0.0, 1.0], dtype=np.float32).tofile(os.path.join(bin_root, "cpu.f32"))
  np.asarray([0.0, 1.0, 2.0], dtype=np.float32).tofile(os.path.join(bin_root, "dsp.f32"))

  def fake_run(cmd, capture_output=True, timeout=None, env=None, preexec_fn=None):
    out = cmd[cmd.index("--out") + 1]
    with open(out, "w", encoding="utf-8") as f:
      json.dump({"status": "pass"}, f)
    return SimpleNamespace(returncode=0, stderr=b"")

  result = dsp_probe.step_snpe_dsp(run=fake_run, paths=FAKE_PATHS, work=work)
  assert result["max_abs_diff"] is None
  assert result["dump_note"] == "size_mismatch"


# ---------------------------------------------------------------------------
# timeouts add up
# ---------------------------------------------------------------------------

def test_step_timeouts_fit_the_total_budget():
  assert sum(dsp_probe.STEP_TIMEOUTS.values()) <= dsp_probe.TOTAL_TIMEOUT_S
  assert dsp_probe.BUILD_TIMEOUT_S < dsp_probe.STEP_TIMEOUTS["build"]
  for runtime in ("cpu", "dsp"):
    assert dsp_probe.RUNNER_TIMEOUT_S[runtime] < dsp_probe.STEP_TIMEOUTS[f"snpe_{runtime}"]


def test_module_import_does_not_pull_tinygrad():
  probe_import = "import sys, openpilot.selfdrive.carrot.tlv.dsp_probe; print('tinygrad' in sys.modules)"
  out = subprocess.run([sys.executable, "-c", probe_import],
                       capture_output=True, text=True, cwd=dsp_probe._basedir())
  assert out.stdout.strip() == "False"



# ---------------------------------------------------------------------------
# per-step timeout + orchestration
# ---------------------------------------------------------------------------

def test_timed_timeout():
  result = dsp_probe._timed("slow", lambda: time.sleep(5.0), 0.05)
  assert result["status"] == "timeout"
  assert result["name"] == "slow"
  assert "exceeded" in result["error"]


def test_timed_captures_failure():
  def boom():
    raise ValueError("nope")

  result = dsp_probe._timed("boom", boom, 1.0)
  assert result["status"] == "fail"
  assert "ValueError" in result["error"]
  assert result["duration_s"] >= 0.0


def test_collect_steps_skips_dependents_after_failure():
  calls = []

  def timed(name, fn, timeout):
    calls.append(name)
    return {"name": name, "status": "fail" if name == "presence" else "pass"}

  steps = dsp_probe.collect_steps(timed=timed)
  assert [step["status"] for step in steps.values()] == ["fail", "skipped", "skipped", "skipped"]
  assert calls == ["presence"]


def test_collect_steps_cpu_failure_still_runs_dsp():
  calls = []

  def timed(name, fn, timeout):
    calls.append(name)
    return {"name": name, "status": "fail" if name == "snpe_cpu" else "pass"}

  steps = dsp_probe.collect_steps(timed=timed)
  assert calls == ["presence", "build", "snpe_cpu", "snpe_dsp"]
  assert [step["status"] for step in steps.values()] == ["pass", "pass", "fail", "pass"]


def test_collect_steps_runs_all_when_passing():
  order = []

  def timed(name, fn, timeout):
    order.append(name)
    return {"name": name, "status": "pass"}

  steps = dsp_probe.collect_steps(timed=timed)
  assert order == ["presence", "build", "snpe_cpu", "snpe_dsp"]
  assert all(step["status"] == "pass" for step in steps.values())


def test_collect_steps_writes_progress(tmp_path):
  progress = str(tmp_path / "c.progress")
  seen = []
  order = []

  def timed(name, fn, timeout):
    order.append(name)
    seen.append(dsp_probe.read_progress_step(progress))
    return {"name": name, "status": "pass"}

  dsp_probe.collect_steps(timed=timed, progress_file=progress)
  assert order == ["presence", "build", "snpe_cpu", "snpe_dsp"]
  assert seen == ["presence", "build", "snpe_cpu", "snpe_dsp"]


def test_build_result_payload():
  steps = {"presence": {"status": "pass"}, "snpe_cpu": {"status": "fail"},
           "snpe_dsp": {"status": "skipped"}}
  payload = dsp_probe.build_result_payload(steps, commit="abc", device_type="tici",
                                           agnos_version="19.8", timestamp="t0")
  assert payload["ok"] is False
  assert payload["commit"] == "abc"
  assert payload["device_type"] == "tici"
  assert payload["agnos_version"] == "19.8"
  payload = dsp_probe.build_result_payload({"a": {"status": "pass"}}, commit="c", device_type=None,
                                           agnos_version=None, timestamp="t", error="x" * 5000)
  assert payload["ok"] is True
  assert len(payload["error"]) <= 2048


def test_parse_argv():
  assert dsp_probe._parse_argv(["--child", "--result", "x.json"]) == {"child": True, "result": "x.json"}
  assert dsp_probe._parse_argv([]) == {"child": False, "result": None}
  assert dsp_probe._parse_argv(["--result"]) == {"child": False, "result": None}


def test_child_main_writes_result_and_cleans_stores(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  monkeypatch.setenv("GIT_COMMIT", "abc1234deadbeef")
  monkeypatch.setattr(dsp_probe, "collect_steps", lambda **kwargs: {
    "presence": {"name": "presence", "status": "pass"},
    "build": {"name": "build", "status": "pass"},
    "snpe_cpu": {"name": "snpe_cpu", "status": "pass"},
    "snpe_dsp": {"name": "snpe_dsp", "status": "fail", "error": "boom"},
  })
  steps = os.path.join(str(probe_root), "snpe1.steps.json")
  dsp_probe.write_steps(steps, {"snpe_cpu": {"status": "pass"}})
  bin_root = os.path.join(str(probe_root), "bin", dsp_probe.PROBE_KEY)
  os.makedirs(bin_root, exist_ok=True)
  path = os.path.join(str(probe_root), "snpe1.json")

  dsp_probe._child_main(path)

  payload = _load_json(path)
  assert payload["ok"] is False
  assert payload["commit"] == "abc1234"
  assert payload["steps"]["snpe_dsp"]["status"] == "fail"
  assert not os.path.exists(steps)
  assert not os.path.isdir(bin_root)


def test_child_main_without_all_steps_writes_no_result(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  monkeypatch.setattr(dsp_probe, "collect_steps", lambda **kwargs: {
    "presence": {"name": "presence", "status": "pass"},
    "build": {"name": "build", "status": "pass"},
    "snpe_cpu": {"name": "snpe_cpu", "status": "pass"},
  })
  path = os.path.join(str(probe_root), "snpe1.json")
  dsp_probe._child_main(path)
  assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# resumption across stops
# ---------------------------------------------------------------------------

def test_collect_steps_resumes_stored_final_steps(probe_root, tmp_path):
  _work_bin_file(probe_root)
  store = str(tmp_path / "c.steps.json")
  dsp_probe.write_steps(store, {"presence": {"name": "presence", "status": "pass"},
                                "build": {"name": "build", "status": "pass", "bin": "/x"}})
  calls = []

  def timed(name, fn, timeout):
    calls.append(name)
    return {"name": name, "status": "pass"}

  steps = dsp_probe.collect_steps(timed=timed, steps_file=store)
  assert calls == ["snpe_cpu", "snpe_dsp"]
  assert steps["presence"]["resumed"] is True
  assert steps["build"]["resumed"] is True
  assert "resumed" not in steps["snpe_cpu"]


def test_collect_steps_rebuilds_when_binary_is_gone(probe_root, tmp_path):
  probe_root.mkdir(parents=True, exist_ok=True)
  store = str(tmp_path / "c.steps.json")
  dsp_probe.write_steps(store, {"presence": {"name": "presence", "status": "pass"},
                                "build": {"name": "build", "status": "pass"}})
  calls = []

  def timed(name, fn, timeout):
    calls.append(name)
    return {"name": name, "status": "pass"}

  steps = dsp_probe.collect_steps(timed=timed, steps_file=store)
  assert calls == ["build", "snpe_cpu", "snpe_dsp"]
  assert "resumed" not in steps["build"]


def test_collect_steps_keeps_failed_step_final(tmp_path):
  store = str(tmp_path / "c.steps.json")
  dsp_probe.write_steps(store, {"presence": {"name": "presence", "status": "fail", "error": "no clang++"}})
  calls = []

  def timed(name, fn, timeout):
    calls.append(name)
    return {"name": name, "status": "pass"}

  steps = dsp_probe.collect_steps(timed=timed, steps_file=store)
  assert calls == []
  assert steps["presence"]["status"] == "fail"
  assert steps["build"]["status"] == "skipped"


def test_collect_steps_persists_after_each_step(tmp_path):
  store = str(tmp_path / "c.steps.json")
  specs = (("presence", None, 1.0, ()), ("build", None, 1.0, ("presence",)),
           ("snpe_cpu", None, 1.0, ("build",)))

  def timed(name, fn, timeout):
    if name == "snpe_cpu":
      raise SystemExit(0)
    return {"name": name, "status": "pass"}

  with pytest.raises(SystemExit):
    dsp_probe.collect_steps(timed=timed, specs=specs, steps_file=store)
  stored = dsp_probe.read_steps(store)
  assert stored["presence"]["status"] == "pass"
  assert stored["build"]["status"] == "pass"
  assert "snpe_cpu" not in stored



# ---------------------------------------------------------------------------
# stop gate (any standstill, any gear, engaged or not)
# ---------------------------------------------------------------------------

def test_stopped_needs_only_standstill_and_low_speed():
  clock = FakeClock(100.0)
  now = clock()
  assert dsp_probe._stopped(FakeSM(clock), now) is True
  assert dsp_probe._stopped(FakeSM(clock, gear="drive", enabled=True), now) is True
  assert dsp_probe._stopped(FakeSM(clock, gear="reverse", enabled=True), now) is True
  assert dsp_probe._stopped(FakeSM(clock, vego=0.2), now) is False
  assert dsp_probe._stopped(FakeSM(clock, standstill=False), now) is False


def test_stopped_requires_fresh_messages():
  clock = FakeClock(100.0)
  stale = FakeSM(clock, fresh=False)
  assert dsp_probe._stopped(stale, clock()) is False
  fresh = FakeSM(clock)
  assert dsp_probe._stopped(fresh, clock()) is True
  clock.advance(2.0)  # no update(): messages older than STOP_FRESH_S
  assert dsp_probe._stopped(fresh, clock()) is False
  fresh.update(0)
  assert dsp_probe._stopped(fresh, clock()) is True


def test_wait_for_stop_holds_three_seconds():
  clock = FakeClock()
  sm = FakeSM(clock)
  sleeps = []

  def sleep(dt):
    sleeps.append(dt)
    clock.advance(dt)

  dsp_probe.wait_for_stop(sm, clock=clock, sleep=sleep)
  assert clock() == pytest.approx(dsp_probe.STOP_HOLD_S)
  assert len(sleeps) == int(round(dsp_probe.STOP_HOLD_S / dsp_probe.GATE_PERIOD_S))


def test_wait_for_stop_resets_after_interruption():
  clock = FakeClock()
  sm = FakeSM(clock)

  def sleep(dt):
    clock.advance(dt)
    if clock() == pytest.approx(2.0):
      sm.standstill = False
    if clock() == pytest.approx(3.0):
      sm.standstill = True

  dsp_probe.wait_for_stop(sm, clock=clock, sleep=sleep)
  assert clock() == pytest.approx(6.0)


def test_wait_for_stop_arming_notice_and_deadline():
  clock = FakeClock()
  sm = FakeSM(clock)
  logs = NoticeLog()

  def sleep(dt):
    clock.advance(dt)

  dsp_probe.wait_for_stop(sm, clock=clock, sleep=sleep, notice=logs)
  arming = logs.of_state("arming")
  assert arming, "arming notice must be refreshed while stopped"
  assert logs.states[0] == "arming"
  assert all(item["deadline"] == pytest.approx(dsp_probe.STOP_HOLD_S) for item in arming)
  assert arming[0]["ts"] == pytest.approx(dsp_probe.ARMING_SHOW_S)
  assert len(arming) == int(round(dsp_probe.STOP_HOLD_S / dsp_probe.GATE_PERIOD_S))


def test_brief_stop_writes_no_notice():
  clock = FakeClock()
  sm = FakeSM(clock)
  logs = NoticeLog()

  def sleep(dt):
    clock.advance(dt)
    if clock() == pytest.approx(0.5):
      sm.standstill = False
    if clock() == pytest.approx(1.0):
      sm.standstill = True

  dsp_probe.wait_for_stop(sm, clock=clock, sleep=sleep, notice=logs)
  assert clock() == pytest.approx(4.0)
  assert logs.items[0] is None  # drop before ARMING_SHOW_S removes the key, no flash
  assert not logs.of_state("aborted")
  arming = logs.of_state("arming")
  assert arming and arming[0]["ts"] == pytest.approx(1.5)


def test_stop_drop_after_arming_marks_aborted():
  clock = FakeClock()
  sm = FakeSM(clock)
  logs = NoticeLog()

  def sleep(dt):
    clock.advance(dt)
    if clock() == pytest.approx(2.0):
      sm.standstill = False
    if clock() == pytest.approx(3.0):
      sm.standstill = True

  dsp_probe.wait_for_stop(sm, clock=clock, sleep=sleep, notice=logs)
  aborted = logs.of_state("aborted")
  assert len(aborted) == 1
  assert aborted[0]["ts"] == pytest.approx(2.0)
  assert clock() == pytest.approx(6.0)


def test_monitor_child_detects_exit_first():
  clock = FakeClock()
  sm = FakeSM(clock)
  outcome, detail = dsp_probe.monitor_child(FakeProc(poll_fn=lambda: 7), sm, clock=clock,
                                            sleep=_advancing_sleep(clock))
  assert (outcome, detail) == ("exited", 7)


def test_monitor_child_kills_within_one_poll_interval_of_pullaway():
  clock = FakeClock()
  sm = FakeSM(clock)
  flip = {}

  def sleep(dt):
    clock.advance(dt)
    if not flip:
      flip["t"] = clock()
      sm.standstill = False

  outcome, detail = dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=sleep,
                                            rss_reader=lambda p: None)
  assert (outcome, detail) == ("unparked", None)
  assert clock() - flip["t"] <= dsp_probe.RUN_POLL_S


def test_monitor_child_total_timeout():
  clock = FakeClock()
  sm = FakeSM(clock)
  outcome, detail = dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=_advancing_sleep(clock),
                                            total_timeout_s=1.0, rss_reader=lambda p: None)
  assert (outcome, detail) == ("timeout", None)
  # 0.1 s poll steps accumulate in binary floating point, so allow one step.
  assert 1.0 <= clock() <= 1.0 + dsp_probe.RUN_POLL_S + 1e-6


def test_monitor_child_rss_limit():
  clock = FakeClock()
  sm = FakeSM(clock)
  outcome, detail = dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=_advancing_sleep(clock),
                                            rss_reader=lambda p: 999, rss_limit=500)
  assert (outcome, detail) == ("rss", 999)


def test_monitor_child_running_notice_tracks_progress(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  progress = str(tmp_path / "c.progress")
  logs = NoticeLog()
  written = []

  def sleep(dt):
    clock.advance(dt)
    now = clock()
    for threshold, step in ((0.3, "presence"), (1.0, "build"), (1.5, "snpe_cpu")):
      if now >= threshold and step not in written:
        written.append(step)
        dsp_probe.write_progress(progress, step)

  dsp_probe.monitor_child(FakeProc(), sm, clock=clock, sleep=sleep, total_timeout_s=2.0,
                          rss_reader=lambda p: None, progress_file=progress, notice=logs)
  running = logs.of_state("running")
  assert running, "running notice must be refreshed while the child runs"
  assert all(item["steps"] == 4 for item in running)
  assert [item["step"] for item in running] == [1, 1, 2, 3]



# ---------------------------------------------------------------------------
# parent: gate loop, latch restore, result guarantee
# ---------------------------------------------------------------------------

def test_run_probe_clean_finish_deletes_latch_and_posts_done(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  pre_state = {"count": 1, "lost_steps": []}
  dsp_probe.write_attempt_state(attempt, pre_state)
  dsp_probe.write_progress(progress, "build")
  logs = NoticeLog()
  seen_attempts = []

  def factory(cmd, **kwargs):
    ticks = {"n": 0}

    def poll():
      ticks["n"] += 1
      seen_attempts.append(dsp_probe.read_attempt_state(attempt))
      if ticks["n"] >= 3:
        dsp_probe.write_result_atomic(result, {"ok": True, "steps": {}})
        return 0
      return None

    return FakeProc(poll_fn=poll)

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=pre_state,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock),
                               rss_reader=lambda p: None, meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "done"
  assert _load_json(result)["ok"] is True
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)
  assert seen_attempts[0] == {"count": 2, "lost_steps": []}
  assert logs.states[0] == "arming"
  assert "running" in logs.states
  assert logs.states[-1] == "done"
  assert not logs.of_state("failed")


def test_run_probe_pullaway_kills_restores_and_aborts(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  pre_state = {"count": 2, "lost_steps": ["build"]}
  dsp_probe.write_attempt_state(attempt, pre_state)
  dsp_probe.write_progress(progress, "build")
  logs = NoticeLog()
  spawns = []
  capture = {"restored": None}
  phase = {"unparked": False, "reparked": False}

  def factory(cmd, **kwargs):
    spawns.append(dsp_probe.read_attempt_state(attempt))
    index = len(spawns)
    if index == 1:
      flipped = {"done": False}

      def poll():
        if not flipped["done"]:
          flipped["done"] = True
          sm.standstill = False
          phase["unparked"] = True
        return None

      return FakeProc(poll_fn=poll)

    def poll_finish():
      dsp_probe.write_result_atomic(result, {"ok": True, "steps": {}})
      return 0

    return FakeProc(poll_fn=poll_finish)

  def sleep(dt):
    clock.advance(dt)
    if phase["unparked"] and not phase["reparked"]:
      # first gate iteration after the abort: latch restored, no result, progress gone
      capture["restored"] = dsp_probe.read_attempt_state(attempt)
      capture["progress_exists"] = os.path.exists(progress)
      capture["result_exists"] = os.path.exists(result)
      phase["reparked"] = True
      sm.standstill = True

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=pre_state,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=sleep, rss_reader=lambda p: None,
                               meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "done"
  assert capture["restored"] == pre_state
  assert capture["progress_exists"] is False
  assert capture["result_exists"] is False
  aborted = logs.of_state("aborted")
  assert len(aborted) == 1
  assert aborted[0]["ts"] == pytest.approx(3.0)
  assert logs.states[-1] == "done"
  # the first spawn saw the incremented latch, the clean finish removed it
  assert spawns[0] == {"count": 3, "lost_steps": ["build"]}
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)


def test_run_probe_respawns_across_three_stops_until_the_result(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  logs = NoticeLog()
  spawns = []

  def factory(cmd, **kwargs):
    index = len(spawns) + 1
    spawns.append(index)
    if index == 4:
      def poll_finish():
        dsp_probe.write_result_atomic(result, {"ok": True, "steps": {}})
        return 0

      return FakeProc(poll_fn=poll_finish)

    def poll_leave():
      sm.standstill = False
      return None

    return FakeProc(poll_fn=poll_leave)

  def sleep(dt):
    clock.advance(dt)
    if not sm.standstill:
      sm.standstill = True  # next stop starts as soon as the gate is polled again

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=sleep, rss_reader=lambda p: None,
                               meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "done"
  assert spawns == [1, 2, 3, 4]
  assert _load_json(result)["ok"] is True
  assert len(logs.of_state("aborted")) == 3
  assert not os.path.exists(attempt)


def test_run_probe_exit_zero_without_result_waits_for_next_stop(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  spawns = []

  def factory(cmd, **kwargs):
    index = len(spawns) + 1
    spawns.append(index)
    if index == 1:
      return FakeProc(poll_fn=lambda: 0)  # steps pending, clean exit

    def poll_finish():
      dsp_probe.write_result_atomic(result, {"ok": True, "steps": {}})
      return 0

    return FakeProc(poll_fn=poll_finish)

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock), rss_reader=lambda p: None,
                               meta={"commit": "abc1234"})
  assert status["status"] == "done"
  assert spawns == [1, 2]


def test_run_probe_timeout_writes_child_timeout_and_posts_failed(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  logs = NoticeLog()
  procs = []

  def factory(cmd, **kwargs):
    proc = FakeProc()
    procs.append(proc)
    return proc

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock),
                               rss_reader=lambda p: None, total_timeout_s=1.0,
                               meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "timeout"
  assert procs[0].killed is True
  payload = _load_json(result)
  assert payload["error"] == "child_timeout"
  assert payload["commit"] == "abc1234"
  assert payload["ok"] is False
  assert not (tmp_path / "abc1234.json.tmp").exists()
  assert len(logs.of_state("failed")) == 1
  assert "done" not in logs.states
  assert not os.path.exists(attempt)


def test_run_probe_rss_limit_writes_result(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)
  logs = NoticeLog()

  def factory(cmd, **kwargs):
    return FakeProc()

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock),
                               rss_reader=lambda p: dsp_probe.CHILD_RSS_LIMIT_BYTES + 1,
                               meta={"commit": "abc1234"}, notice=logs)
  assert status["status"] == "rss"
  payload = _load_json(result)
  assert payload["error"] == "child_rss_limit"
  assert len(logs.of_state("failed")) == 1


def test_run_probe_child_exit_without_result(tmp_path):
  clock = FakeClock()
  sm = FakeSM(clock)
  result, attempt, progress = _run_paths(tmp_path)

  def factory(cmd, **kwargs):
    return FakeProc(poll_fn=lambda: 3)

  status = dsp_probe.run_probe(result, sm=sm, attempt_file=attempt, pre_state=None,
                               progress_file=progress, popen_factory=factory,
                               clock=clock, sleep=_advancing_sleep(clock), rss_reader=lambda p: None,
                               meta={"commit": "abc1234"})
  assert status["status"] == "exited"
  payload = _load_json(result)
  assert payload["error"] == "child_exit_3"
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)


# ---------------------------------------------------------------------------
# attempt / progress roundtrip + lost-step latch
# ---------------------------------------------------------------------------

def test_attempt_progress_roundtrip(tmp_path):
  attempt = str(tmp_path / "c.attempt")
  progress = str(tmp_path / "c.progress")
  assert dsp_probe.read_attempt_state(attempt) is None
  assert dsp_probe.read_progress_step(progress) is None
  dsp_probe.write_attempt_state(attempt, {"count": 2, "lost_steps": ["build", "snpe_cpu"]})
  assert dsp_probe.read_attempt_state(attempt) == {"count": 2, "lost_steps": ["build", "snpe_cpu"]}
  with open(attempt, "w", encoding="utf-8") as f:
    f.write("{not json")
  assert dsp_probe.read_attempt_state(attempt) is None
  dsp_probe.write_progress(progress, "build")
  assert dsp_probe.read_progress_step(progress) == "build"
  with open(progress, "w", encoding="utf-8") as f:
    f.write("[]")
  assert dsp_probe.read_progress_step(progress) is None


def test_latched_step_requires_same_step_three_times():
  assert dsp_probe._latched_step(["build", "build"]) is None
  assert dsp_probe._latched_step(["build", "build", "build"]) == "build"
  assert dsp_probe._latched_step(["build", "snpe_cpu", "build"]) is None
  assert dsp_probe._latched_step(["presence", "build", "snpe_cpu"]) is None
  assert dsp_probe._latched_step(["snpe_cpu", "snpe_dsp", "snpe_cpu", "snpe_dsp", "snpe_cpu"]) == "snpe_cpu"


def test_main_latches_after_three_same_step_losses(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "snpe1.json")
  dsp_probe.write_attempt_state(dsp_probe.attempt_path(result), {"count": 2, "lost_steps": ["build", "build"]})
  dsp_probe.write_progress(dsp_probe.progress_path(result), "build")
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))

  dsp_probe.main()

  payload = _load_json(result)
  assert payload["error"] == "lost_in_step:build"
  assert payload["ok"] is False
  assert dsp_probe._probe_disabled_by_lost_attempts is True
  assert not os.path.exists(dsp_probe.attempt_path(result))
  assert not os.path.exists(dsp_probe.progress_path(result))
  os.unlink(result)
  params = FakeParams({"TrafficLightDetectMode": 1})
  assert dsp_probe.should_run(True, params) is False  # latched, not the result file


def test_main_does_not_latch_on_mixed_losses(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "snpe1.json")
  dsp_probe.write_attempt_state(dsp_probe.attempt_path(result), {"count": 3, "lost_steps": ["build", "snpe_cpu"]})
  dsp_probe.write_progress(dsp_probe.progress_path(result), "build")
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: None)
  calls = []
  fake_sm = object()
  monkeypatch.setattr(dsp_probe, "_make_submaster", lambda: fake_sm)
  monkeypatch.setattr(dsp_probe, "run_probe", lambda *a, **k: calls.append(k) or {"status": "done"})

  dsp_probe.main()

  assert len(calls) == 1
  assert calls[0]["sm"] is fake_sm
  assert calls[0]["notice"] is not None
  assert dsp_probe._probe_disabled_by_lost_attempts is False
  assert not os.path.exists(result)


def test_main_retries_messaging_then_runs_probe(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  captured = {}
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: captured.__setitem__(sig, fn))
  monkeypatch.setattr(dsp_probe.time, "sleep", lambda _s: None)
  attempts = {"n": 0}
  fake_sm = object()

  def flaky_submaster():
    attempts["n"] += 1
    if attempts["n"] < 3:
      raise RuntimeError("messaging not up")
    return fake_sm

  monkeypatch.setattr(dsp_probe, "_make_submaster", flaky_submaster)
  calls = []
  monkeypatch.setattr(dsp_probe, "run_probe", lambda *a, **k: calls.append(k) or {"status": "done"})

  dsp_probe.main()

  assert attempts["n"] == 3
  assert calls and calls[0]["sm"] is fake_sm
  assert set(captured) == {signal.SIGINT, signal.SIGTERM}


def test_main_consumes_progress_once_without_false_latch(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "snpe1.json")
  attempt = dsp_probe.attempt_path(result)
  progress = dsp_probe.progress_path(result)
  dsp_probe.write_attempt_state(attempt, {"count": 1, "lost_steps": []})
  dsp_probe.write_progress(progress, "build")
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: None)
  monkeypatch.setattr(dsp_probe.time, "sleep", lambda _s: None)
  monkeypatch.setattr(dsp_probe, "_make_submaster", lambda: object())
  monkeypatch.setattr(dsp_probe, "run_probe", _power_cut)

  for _boot in range(2):
    with pytest.raises(PowerCut):
      dsp_probe.main()

  assert dsp_probe.read_attempt_state(attempt) == {"count": 1, "lost_steps": ["build"]}
  assert not os.path.exists(progress)
  assert not os.path.exists(result)
  assert dsp_probe._probe_disabled_by_lost_attempts is False


def test_main_still_latches_three_separate_lost_runs(probe_root, monkeypatch):
  probe_root.mkdir(parents=True, exist_ok=True)
  result = os.path.join(str(probe_root), "snpe1.json")
  attempt = dsp_probe.attempt_path(result)
  progress = dsp_probe.progress_path(result)
  dsp_probe.write_attempt_state(attempt, {"count": 1, "lost_steps": []})
  monkeypatch.setattr(dsp_probe, "_probe_disabled_by_lost_attempts", False)
  monkeypatch.setattr(dsp_probe, "make_notice_sink", lambda *a, **k: (lambda payload: None))
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: None)
  monkeypatch.setattr(dsp_probe.time, "sleep", lambda _s: None)
  monkeypatch.setattr(dsp_probe, "_make_submaster", lambda: object())
  monkeypatch.setattr(dsp_probe, "run_probe", _power_cut)

  for _boot in range(2):
    dsp_probe.write_progress(progress, "build")
    with pytest.raises(PowerCut):
      dsp_probe.main()

  dsp_probe.write_progress(progress, "build")
  dsp_probe.main()

  payload = _load_json(result)
  assert payload["error"] == "lost_in_step:build"
  assert dsp_probe._probe_disabled_by_lost_attempts is True
  assert not os.path.exists(attempt)
  assert not os.path.exists(progress)


# ---------------------------------------------------------------------------
# stop signal handling (orphan protection, DEFECT 1)
# ---------------------------------------------------------------------------

def test_stop_handler_kills_child_restores_and_clears_notice(monkeypatch):
  killed = []
  restored = []
  logs = NoticeLog()

  class P:
    def kill(self):
      killed.append(True)

    def wait(self, timeout=None):
      return 0

  monkeypatch.setattr(dsp_probe, "_RUNNING_CHILD", P())
  handler = dsp_probe._make_stop_handler(lambda: restored.append(1), logs)
  with pytest.raises(SystemExit) as excinfo:
    handler(signal.SIGTERM, None)
  assert excinfo.value.code == 0
  assert killed == [True]
  assert restored == [1]
  assert logs.items == [None]


def test_stop_handler_without_child_is_clean(monkeypatch):
  monkeypatch.setattr(dsp_probe, "_RUNNING_CHILD", None)
  restored = []
  handler = dsp_probe._make_stop_handler(lambda: restored.append(1))
  with pytest.raises(SystemExit):
    handler(signal.SIGINT, None)
  assert restored == [1]


def test_install_signal_handlers_registers_both(monkeypatch):
  captured = {}
  monkeypatch.setattr(dsp_probe, "_SIGNAL", lambda sig, fn: captured.__setitem__(sig, fn))
  dsp_probe._install_signal_handlers(lambda: None)
  assert set(captured) == {signal.SIGINT, signal.SIGTERM}


# ---------------------------------------------------------------------------
# child preexec (PDEATHSIG)
# ---------------------------------------------------------------------------

def test_default_popen_passes_preexec_on_linux(monkeypatch):
  captured = {}

  def fake_popen(cmd, **kwargs):
    captured["cmd"] = list(cmd)
    captured.update(kwargs)
    return object()

  monkeypatch.setattr(dsp_probe, "IS_LINUX", True)
  monkeypatch.setattr(dsp_probe, "_POPEN", fake_popen)
  dsp_probe._default_popen(["x", "y"], stdout=None)
  assert captured["cmd"] == ["x", "y"]
  assert captured["preexec_fn"] is dsp_probe._child_preexec


def test_default_popen_skips_preexec_off_linux(monkeypatch):
  captured = {}

  def fake_popen(cmd, **kwargs):
    captured.update(kwargs)
    return object()

  monkeypatch.setattr(dsp_probe, "IS_LINUX", False)
  monkeypatch.setattr(dsp_probe, "_POPEN", fake_popen)
  dsp_probe._default_popen(["x"])
  assert "preexec_fn" not in captured


def test_default_popen_honors_explicit_preexec(monkeypatch):
  captured = {}

  def fake_popen(cmd, **kwargs):
    captured.update(kwargs)
    return object()

  monkeypatch.setattr(dsp_probe, "IS_LINUX", True)
  monkeypatch.setattr(dsp_probe, "_POPEN", fake_popen)
  dsp_probe._default_popen(["x"], preexec_fn=None)
  assert captured["preexec_fn"] is None


def test_child_preexec_never_raises():
  dsp_probe._child_preexec()  # best-effort on every platform


# ---------------------------------------------------------------------------
# UI notice sink
# ---------------------------------------------------------------------------

def test_make_notice_sink_puts_and_removes():
  calls = []

  class FakeParamsObj:
    def put(self, key, value):
      calls.append(("put", key, value))

    def remove(self, key):
      calls.append(("remove", key))

  sink = dsp_probe.make_notice_sink(params_factory=FakeParamsObj)
  sink({"state": "arming", "ts": 1.0, "deadline": 5.0})
  sink(None)
  assert calls == [("put", "CarrotTlvDspNotice", {"state": "arming", "ts": 1.0, "deadline": 5.0}),
                   ("remove", "CarrotTlvDspNotice")]


def test_make_notice_sink_failure_is_silent():
  def boom():
    raise RuntimeError("no params")

  sink = dsp_probe.make_notice_sink(params_factory=boom)
  sink({"state": "done", "ts": 1.0})
  sink(None)

  class ExplodingParams:
    def put(self, key, value):
      raise RuntimeError("put failed")

    def remove(self, key):
      raise RuntimeError("remove failed")

  sink2 = dsp_probe.make_notice_sink(params_factory=ExplodingParams)
  sink2({"state": "done", "ts": 1.0})
  sink2(None)


def test_publish_outcome_notice_maps_ok(tmp_path):
  logs = NoticeLog()
  result = str(tmp_path / "r.json")
  dsp_probe.write_result_atomic(result, dsp_probe.build_result_payload(
    {"a": {"status": "pass"}}, commit="c", device_type=None, agnos_version=None, timestamp="t"))
  dsp_probe._publish_outcome_notice(result, 1.0, logs)
  dsp_probe.write_result_atomic(result, dsp_probe.build_result_payload(
    {}, commit="c", device_type=None, agnos_version=None, timestamp="t", error="x"))
  dsp_probe._publish_outcome_notice(result, 2.0, logs)
  assert logs.states == ["done", "failed"]
  assert logs.items[0]["ts"] == 1.0
  dsp_probe._publish_outcome_notice(str(tmp_path / "missing.json"), 3.0, logs)
  assert logs.states[-1] == "failed"


def test_step_index_from_progress(tmp_path):
  progress = str(tmp_path / "p")
  assert dsp_probe._step_index_from_progress(None, 2) == 2
  assert dsp_probe._step_index_from_progress(progress, 1) == 1  # missing -> last known
  dsp_probe.write_progress(progress, "snpe_dsp")
  assert dsp_probe._step_index_from_progress(progress, 1) == 4
  dsp_probe.write_progress(progress, "build")
  assert dsp_probe._step_index_from_progress(progress, 1) == 2


# ---------------------------------------------------------------------------
# grandchild death signal (real Linux: the build/runner must not outlive the child)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_grandchild_dies_with_the_sigkilled_child(tmp_path):
  pidfile = tmp_path / "grandchild.pid"
  lines = [
    "import subprocess, sys, threading, time",
    "from openpilot.selfdrive.carrot.tlv import dsp_probe",
    "def go():",
    "  subprocess.run(['/bin/sh', '-c', 'echo $$ > ' + sys.argv[1] + '; exec sleep 30'],",
    "                 preexec_fn=dsp_probe._child_preexec)",
    "threading.Thread(target=go).start()",
    "time.sleep(30)",
  ]
  code = "\n".join(lines) + "\n"
  child = subprocess.Popen([sys.executable, "-c", code, str(pidfile)])
  try:
    deadline = time.monotonic() + 15.0
    while not pidfile.exists() and time.monotonic() < deadline:
      time.sleep(0.05)
    assert pidfile.exists(), "the child never started its grandchild"
    grandchild = int(pidfile.read_text(encoding="utf-8").strip())
    assert _pid_alive(grandchild), "the grandchild must run before the child is killed"
    child.kill()
    child.wait(timeout=5.0)
    deadline = time.monotonic() + 2.0
    while _pid_alive(grandchild) and time.monotonic() < deadline:
      time.sleep(0.05)
    assert _pid_alive(grandchild) is False
  finally:
    if child.poll() is None:
      child.kill()
      child.wait(timeout=5.0)
