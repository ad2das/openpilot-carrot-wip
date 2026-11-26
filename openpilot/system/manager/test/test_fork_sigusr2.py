"""Verify SIGUSR2 fork masking, child unblocking, bounded SIGKILL join, and cold-boot audio retries."""
import ast
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

import openpilot.system.manager.process as proc_mod


SIGUSR2_BIT = 1 << (signal.SIGUSR2 - 1) if hasattr(signal, "SIGUSR2") else 0


def _read_sigblk(path: str = "/proc/self/status") -> int:
  with open(path, encoding="utf-8") as f:
    for line in f:
      if line.startswith("SigBlk:"):
        return int(line.split()[1], 16)
  raise AssertionError(f"SigBlk not found in {path}")


def _write_self_sigblk(out_path: str) -> None:
  Path(out_path).write_text(f"{_read_sigblk():x}\n", encoding="utf-8")


@pytest.mark.skipif(not (hasattr(signal, "SIGUSR2") and hasattr(signal, "pthread_sigmask") and os.path.exists("/proc/self/status")),
                    reason="Linux pthread_sigmask and /proc/self/status required")
def test_fork_blocks_sigusr2_in_parent_and_unblocks_in_python_child(tmp_path):
  out_file = tmp_path / "py_sigblk.txt"
  prev = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR2})
  try:
    assert signal.SIGUSR2 not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    with proc_mod._block_sigusr2_for_fork(check_fork_method=False):
      assert signal.SIGUSR2 in signal.pthread_sigmask(signal.SIG_BLOCK, set())
      p = multiprocessing.get_context("fork").Process(target=_write_self_sigblk, args=(str(out_file),))
      p.start()
      p.join(5.0)
      assert p.exitcode == 0

    # Parent mask must be restored right after fork.
    assert signal.SIGUSR2 not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    child_sigblk = int(out_file.read_text(encoding="utf-8").strip(), 16)
    assert (child_sigblk & SIGUSR2_BIT) == 0, f"forked Python child inherited blocked SIGUSR2: {child_sigblk:#x}"
  finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, prev)


@pytest.mark.skipif(not (hasattr(signal, "SIGUSR2") and hasattr(signal, "pthread_sigmask") and os.path.exists("/proc/self/status")),
                    reason="Linux pthread_sigmask and /proc/self/status required")
def test_native_exec_child_inherits_unblocked_sigusr2(tmp_path):
  out_file = tmp_path / "native_sigblk.txt"
  prev = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR2})
  try:
    cmd = ["/bin/sh", "-c", f"grep '^SigBlk:' /proc/self/status > '{out_file}'"]
    p = proc_mod.NativeProcess("sigblk_probe", ".", cmd, lambda *_: True)
    p.start()
    proc_mod.join_process(p.proc, 5.0)
    assert p.proc.exitcode == 0
    assert signal.SIGUSR2 not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    sigblk_line = out_file.read_text(encoding="utf-8").strip()
    child_sigblk = int(sigblk_line.split()[1], 16)
    assert (child_sigblk & SIGUSR2_BIT) == 0, f"exec'd native child inherited blocked SIGUSR2: {child_sigblk:#x}"
  finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, prev)


@pytest.mark.skipif(not (hasattr(signal, "SIGUSR2") and hasattr(signal, "pthread_sigmask") and os.path.exists("/proc/self/status")),
                    reason="Linux pthread_sigmask and /proc/self/status required")
def test_daemon_preexec_child_inherits_unblocked_sigusr2(tmp_path):
  out_file = tmp_path / "daemon_sigblk.txt"
  prev = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR2})
  try:
    with proc_mod._block_sigusr2_for_fork(check_fork_method=False):
      assert signal.SIGUSR2 in signal.pthread_sigmask(signal.SIG_BLOCK, set())
      code = "; ".join([
        "from pathlib import Path",
        "line = next(l for l in open('/proc/self/status') if l.startswith('SigBlk:'))",
        f"Path({str(out_file)!r}).write_text(line)",
      ])
      p = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=proc_mod._daemon_preexec,
      )
      assert p.wait(timeout=5) == 0

    assert signal.SIGUSR2 not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
    sigblk_line = out_file.read_text(encoding="utf-8").strip()
    child_sigblk = int(sigblk_line.split()[1], 16)
    assert (child_sigblk & SIGUSR2_BIT) == 0, f"daemon child inherited blocked SIGUSR2: {child_sigblk:#x}"
  finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, prev)


@pytest.mark.skipif(not (hasattr(signal, "SIGUSR2") and hasattr(signal, "pthread_sigmask")),
                    reason="POSIX pthread_sigmask required")
def test_parent_sigmask_restored_on_exception():
  prev = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGUSR2})
  try:
    with pytest.raises(RuntimeError, match="boom"):
      with proc_mod._block_sigusr2_for_fork(check_fork_method=False):
        assert signal.SIGUSR2 in signal.pthread_sigmask(signal.SIG_BLOCK, set())
        raise RuntimeError("boom")
    assert signal.SIGUSR2 not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
  finally:
    signal.pthread_sigmask(signal.SIG_SETMASK, prev)


def test_stop_bounds_wait_after_sigkill(monkeypatch):
  class UnkillableProc:
    pid = 999999
    exitcode = None

    def join(self):
      raise AssertionError("unbounded join() must not be called")

  class DummyProc(proc_mod.ManagerProcess):
    name = "stuck_proc"

    def prepare(self):
      pass

    def start(self):
      pass

  signals_sent = []
  joins = []
  errors = []
  monkeypatch.setattr(proc_mod, "join_process", lambda p, timeout: joins.append(timeout))
  monkeypatch.setattr(proc_mod.cloudlog, "error", errors.append)

  dp = DummyProc()
  dp.proc = UnkillableProc()
  dp.signal = lambda sig: signals_sent.append(sig)

  ret = dp.stop(retry=True, block=True)
  assert ret is None
  assert signals_sent == [signal.SIGINT, signal.SIGKILL]
  assert joins == [5, 5]
  assert dp.proc is not None and dp.shutting_down is True
  assert any("did not exit after SIGKILL" in msg for msg in errors)


@pytest.mark.parametrize("rel_path", [
  "openpilot/system/micd.py",
  "openpilot/selfdrive/ui/soundd.py",
])
def test_audio_daemons_have_cold_boot_retry_headroom(rel_path):
  path = Path(__file__).parents[4] / rel_path
  tree = ast.parse(path.read_text(encoding="utf-8"))
  fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_stream")
  retry_dec = next(
    d for d in fn.decorator_list
    if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "retry"
  )
  kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in retry_dec.keywords}
  assert kwargs["attempts"] * kwargs["delay"] >= 60
