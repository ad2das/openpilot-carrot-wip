from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from openpilot.selfdrive.carrot.server.features.tools.actions import validate_shell_argv
from openpilot.selfdrive.carrot.server.features.tools.dispatcher import _clear_origin_remote_refs, _restore_missing_origin_remote_refs


def test_validate_shell_argv_blocks_mutating_git_and_secret_cat_paths():
  # Allowed read-only commands stay allowed.
  assert validate_shell_argv(["git", "status"]) is None
  assert validate_shell_argv(["git", "branch", "-a"]) is None
  assert validate_shell_argv(["git", "branch", "-av"]) is None
  assert validate_shell_argv(["git", "branch", "-avv"]) is None
  assert validate_shell_argv(["git", "branch", "-rv"]) is None
  assert validate_shell_argv(["git", "branch", "--show-current"]) is None
  assert validate_shell_argv(["git", "remote", "-v"]) is None
  assert validate_shell_argv(["git", "remote", "show", "-n", "origin"]) is None
  assert validate_shell_argv(["git", "remote", "get-url", "origin"]) is None
  assert validate_shell_argv(["git", "log", "--oneline", "-5"]) is None
  assert validate_shell_argv(["git", "diff", "--stat", "HEAD~1..HEAD", "--", "openpilot/selfdrive/carrot/carrot_man.py"]) is None
  assert validate_shell_argv(["cat", "/tmp/launch_log"]) is None
  assert validate_shell_argv(["cat", "/proc/meminfo"]) is None
  assert validate_shell_argv(["cat", "README.md"]) is None

  # Dangerous git flags and mutations are rejected.
  assert validate_shell_argv(["git", "diff", "--output=/tmp/x"]) is not None
  assert validate_shell_argv(["git", "diff", "-o", "/tmp/x"]) is not None
  assert validate_shell_argv(["git", "diff", "--ext-diff"]) is not None
  assert validate_shell_argv(["git", "diff", "--no-index", "/etc/shadow", "/dev/null"]) is not None
  assert validate_shell_argv(["git", "diff", "--no-index", "/data/params/d/GithubSshKeys", "/dev/null"]) is not None
  assert validate_shell_argv(["git", "diff", "/etc/hosts", "/dev/null"]) is not None
  assert validate_shell_argv(["git", "diff", "../../etc/passwd", "/dev/null"]) is not None
  assert validate_shell_argv(["git", "log", "--textconv"]) is not None
  assert validate_shell_argv(["git", "status", "-c", "core.pager=sh"]) is not None
  assert validate_shell_argv(["git", "branch", "-D", "main"]) is not None
  assert validate_shell_argv(["git", "branch", "-aD", "main"]) is not None
  assert validate_shell_argv(["git", "branch", "new-branch"]) is not None
  assert validate_shell_argv(["git", "remote", "show", "https://evil.example/x.git"]) is not None
  assert validate_shell_argv(["git", "remote", "show", "/tmp/local-repo"]) is not None
  assert validate_shell_argv(["git", "remote", "get-url", "ssh://evil.example/x.git"]) is not None
  assert validate_shell_argv(["git", "remote", "set-url", "origin", "https://evil.example/x.git"]) is not None
  assert validate_shell_argv(["git", "remote", "remove", "origin"]) is not None

  # Sensitive or traversal cat targets are rejected.
  assert validate_shell_argv(["cat"]) is not None
  assert validate_shell_argv(["cat", "-"]) is not None
  assert validate_shell_argv(["cat", "~/.ssh/id_rsa"]) is not None
  assert validate_shell_argv(["cat", "/data/params/d/GithubSshKeys"]) is not None
  assert validate_shell_argv(["cat", "/etc/shadow"]) is not None
  assert validate_shell_argv(["cat", "../../.ssh/id_rsa"]) is not None
  assert validate_shell_argv(["cat", "/proc/self/root/etc/passwd"]) is not None
  assert validate_shell_argv(["cat", "/proc/self/environ"]) is not None
  assert validate_shell_argv(["cat", "/proc/1/fd/0"]) is not None


def _git(cwd, *args):
  import subprocess
  return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def _git_branch_state(repo):
  return (_git(repo, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads").stdout,
          _git(repo, "branch", "--show-current").stdout)


def test_git_branch_optional_value_flags_cannot_smuggle_mutations(tmp_path):
  # git only accepts --color/--abbrev/--column values in the attached "=" form, so the
  # next argument is a real branch operation (`--color -m --color x` renamed a branch).
  for argv in (["--color", "evil"], ["--abbrev", "evil"], ["--column", "evil"],
               ["--color", "-m", "--color", "renamed"], ["--color", "-D", "--color", "other"],
               ["--column", "-d", "--abbrev", "other"]):
    assert validate_shell_argv(["git", "branch", *argv]) is not None, argv
  for argv in (["--color=always"], ["--abbrev=8", "-v"], ["--column=never", "-a"], ["-a", "--color"],
               ["--no-color", "-r"], ["--contains", "HEAD"], ["--sort", "refname"], ["--format", "%(refname)"]):
    assert validate_shell_argv(["git", "branch", *argv]) is None, argv

  # Whatever the policy accepts must leave real branches untouched.
  repo = tmp_path / "repo"
  _git(tmp_path, "init", "-q", "-b", "main", str(repo))
  _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
  _git(repo, "branch", "other")
  before = _git_branch_state(repo)
  flags = ("--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort", "--format",
           "--color", "--abbrev", "--column")
  for flag in flags:
    for op in ("-D", "-d", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy"):
      for argv in ([flag, op, flag, "other"], [flag, op, flag, "renamed"], [flag, "evil"]):
        if validate_shell_argv(["git", "branch", *argv]) is None:
          _git(repo, "branch", *argv)
          assert _git_branch_state(repo) == before, argv


def test_failed_fetch_restores_origin_refs_cleared_before_fetch(tmp_path):
  remote = tmp_path / "remote.git"
  device = tmp_path / "device"
  _git(tmp_path, "init", "-q", "--bare", str(remote))
  _git(tmp_path, "init", "-q", "-b", "main", str(device))
  _git(device, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
  _git(device, "remote", "add", "origin", str(remote))
  _git(device, "push", "-q", "origin", "main")
  _git(device, "fetch", "-q", "origin")
  _git(device, "branch", "-q", "--set-upstream-to=origin/main")
  _git(device, "pack-refs", "--all")
  _git(device, "update-ref", "refs/remotes/origin/loose", "HEAD")
  before = _git(device, "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/origin").stdout
  assert "refs/remotes/origin/main" in before and "refs/remotes/origin/loose" in before

  snapshot = _clear_origin_remote_refs(str(device))
  _git(device, "pack-refs", "--all")
  _git(device, "remote", "set-url", "origin", str(tmp_path / "offline.git"))
  assert _git(device, "fetch", "origin", "--prune", "--force").returncode != 0
  assert _git(device, "for-each-ref", "refs/remotes/origin").stdout == ""

  assert _restore_missing_origin_remote_refs(str(device), snapshot) == 2
  after = _git(device, "for-each-ref", "--format=%(refname) %(objectname)", "refs/remotes/origin").stdout
  assert after == before
  assert "gone" not in _git(device, "status", "-sb").stdout


def test_origin_ref_restore_skips_corrupt_unsafe_and_newer_refs(tmp_path):
  git_dir = tmp_path / ".git"
  origin_dir = git_dir / "refs" / "remotes" / "origin"
  origin_dir.mkdir(parents=True)
  (origin_dir / "good").write_text("a" * 40 + "\n", encoding="utf-8")
  (origin_dir / "HEAD").write_text("ref: refs/remotes/origin/good\n", encoding="utf-8")
  (origin_dir / "broken").write_text("", encoding="utf-8")
  (git_dir / "packed-refs").write_text(
    "# pack-refs with: peeled fully-peeled sorted\n"
    + ("b" * 40) + " refs/remotes/origin/packed\n"
    + ("c" * 40) + " refs/remotes/origin/../../heads/evil\n",
    encoding="utf-8",
  )

  snapshot = _clear_origin_remote_refs(str(tmp_path))
  assert set(snapshot) == {"refs/remotes/origin/good", "refs/remotes/origin/HEAD", "refs/remotes/origin/packed"}

  origin_dir.mkdir(parents=True)
  (origin_dir / "good").write_text("d" * 40 + "\n", encoding="utf-8")
  assert _restore_missing_origin_remote_refs(str(tmp_path), snapshot) == 2
  assert (origin_dir / "good").read_text(encoding="utf-8") == "d" * 40 + "\n"
  assert (origin_dir / "packed").read_text(encoding="utf-8") == "b" * 40 + "\n"
  assert (origin_dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/remotes/origin/good\n"
  assert not (origin_dir / "broken").exists()
  assert not (git_dir / "refs" / "heads" / "evil").exists()


def test_clear_origin_remote_refs_removes_loose_and_packed_origin_refs(tmp_path):
  git_dir = tmp_path / ".git"
  origin_dir = git_dir / "refs" / "remotes" / "origin"
  origin_dir.mkdir(parents=True)
  (origin_dir / "main").write_text("a" * 40 + "\n", encoding="utf-8")
  packed = git_dir / "packed-refs"
  packed.write_text(
    "# pack-refs with: peeled fully-peeled sorted\n"
    + ("1" * 40) + " refs/heads/main\n"
    + ("2" * 40) + " refs/remotes/origin/main\n"
    + "^" + ("3" * 40) + "\n"
    + ("4" * 40) + " refs/remotes/upstream/main\n",
    encoding="utf-8",
  )

  _clear_origin_remote_refs(str(tmp_path))

  assert not origin_dir.exists()
  remaining = packed.read_text(encoding="utf-8")
  assert "refs/heads/main" in remaining
  assert "refs/remotes/upstream/main" in remaining
  assert "refs/remotes/origin/" not in remaining
  assert ("^" + "3" * 40) not in remaining


def test_tlv_rec_producer_error_does_not_call_write_eof(tmp_path):
  spec = importlib.util.spec_from_file_location(
    "tlv_rec_error_test",
    Path(__file__).resolve().parents[1] / "features" / "tlv_rec.py",
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  module.REC_ROOT = str(tmp_path)
  session_dir = tmp_path / "20260101-120000"
  session_dir.mkdir()
  (session_dir / "0000001.jpg").write_bytes(b"data")

  def failing_chunks(_dir):
    yield b"PARTIAL"
    raise OSError("disk read failure")

  module.iter_zip_chunks = failing_chunks

  eof_called = []
  orig_write_eof = web.StreamResponse.write_eof

  async def tracked_write_eof(self, *args, **kwargs):
    eof_called.append(True)
    return await orig_write_eof(self, *args, **kwargs)

  web.StreamResponse.write_eof = tracked_write_eof
  try:
    async def scenario():
      app = web.Application()
      module.register(app)
      async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/tlv_rec/sessions/20260101-120000.zip")
        try:
          await resp.read()
        except Exception:
          pass

    asyncio.run(scenario())
  finally:
    web.StreamResponse.write_eof = orig_write_eof

  assert eof_called == []


def test_sync_shell_cmd_uses_async_process_group_and_lock_kwargs(monkeypatch):
  import os
  from openpilot.common import async_process
  from openpilot.selfdrive.carrot.server.features.tools import dispatcher

  if os.name != "nt":
    monkeypatch.setattr(async_process, "child_lock_kwargs", lambda: {"pass_fds": (42,)})
    pg_kwargs = async_process.process_group_kwargs()
    assert pg_kwargs["start_new_session"] is True
    assert pg_kwargs["pass_fds"] == (42,)

  captured = {}

  async def fake_capture_exec(argv, *, cwd=None, timeout=None):
    captured["argv"] = argv
    captured["cwd"] = cwd
    captured["timeout"] = timeout
    return 0, "ok"

  monkeypatch.setattr(dispatcher.jobs, "capture_exec", fake_capture_exec)

  resp = asyncio.run(dispatcher._dispatch_sync(None, {"action": "shell_cmd", "cmd": "git status"}))
  assert resp.status == 200
  assert captured == {"argv": ["git", "status"], "cwd": "/data/openpilot", "timeout": 10.0}


def test_build_git_update_summary_async_survives_capture_timeout(monkeypatch):
  from openpilot.selfdrive.carrot.server.features.tools import dispatcher

  async def slow_capture(_cmd, *, cwd=None, timeout=None):
    raise TimeoutError("timed out")

  monkeypatch.setattr(dispatcher.jobs, "capture_exec", slow_capture)

  summary = asyncio.run(dispatcher._build_git_update_summary_async("/tmp", "aaa", "bbb", "Updating aaa..bbb"))
  assert summary["before"] == "aaa"
  assert summary["after"] == "bbb"
  assert summary["commits"] == []


def test_sync_dispatch_cancellation_kills_process_group_and_releases_repo_lock(tmp_path, monkeypatch):
  import os
  import time
  import pytest
  from openpilot.common import async_process, repo_update
  from openpilot.selfdrive.carrot.server.features.tools import dispatcher

  if os.name == "nt":
    pytest.skip("POSIX process-group test")

  lock_path = tmp_path / "repo.lock"
  pid_file = tmp_path / "grandchild.pid"
  monkeypatch.setattr(repo_update, "LOCK_PATH", str(lock_path))

  async def fake_prepare(_repo_dir):
    return False

  monkeypatch.setattr(dispatcher, "prepare_repo", fake_prepare)

  orig_capture = dispatcher.jobs.capture_exec

  async def hijacked_capture(_cmd, *, cwd=None, timeout=None):
    # Spawn a grandchild process in the same session so only killpg kills both.
    script = f"sleep 30 & echo $! > {pid_file}; wait"
    return await orig_capture(["bash", "-c", script], cwd=str(tmp_path), timeout=timeout)

  monkeypatch.setattr(dispatcher.jobs, "capture_exec", hijacked_capture)

  async def scenario():
    task = asyncio.create_task(dispatcher.dispatch_sync(None, {"action": "git_sync"}))
    deadline = time.monotonic() + 3.0
    while not pid_file.exists() and time.monotonic() < deadline:
      await asyncio.sleep(0.02)
    assert pid_file.exists()
    grandchild_pid = int(pid_file.read_text(encoding="utf-8").strip())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

    # Grandchild must be dead AND repo_lock must be immediately acquirable (no leaked flock).
    try:
      os.kill(grandchild_pid, 0)
      alive = True
    except ProcessLookupError:
      alive = False
    assert not alive, f"grandchild pid {grandchild_pid} survived cancellation"

    with repo_update.repo_lock():
      assert async_process.process_group_kwargs().get("start_new_session") is True

  asyncio.run(scenario())


def test_reset_repo_fetch_timeout_restores_origin_refs(monkeypatch):
  # stream_exec kills git and re-raises on timeout; the refs cleared before the
  # fetch must still be put back, and the job must finish as CMD_TIMEOUT.
  from openpilot.selfdrive.carrot.server.features.tools import dispatcher

  monkeypatch.setattr(dispatcher.jobs, "persist_now", lambda: None)
  monkeypatch.setattr(dispatcher.jobs, "persist_changed", lambda *a, **k: None)

  async def fake_capture_exec(_cmd, **_kwargs):
    return 0, ""

  monkeypatch.setattr(dispatcher.jobs, "capture_exec", fake_capture_exec)

  snapshot = {"refs/remotes/origin/carrot-wip": "abc\n"}
  monkeypatch.setattr(dispatcher, "_clear_origin_remote_refs", lambda repo_dir: dict(snapshot))

  restores = []

  def fake_restore(repo_dir, refs):
    restores.append((repo_dir, refs))
    return 1

  monkeypatch.setattr(dispatcher, "_restore_missing_origin_remote_refs", fake_restore)

  async def timing_out_stream_exec(_job, _cmd, **_kwargs):
    raise TimeoutError

  monkeypatch.setattr(dispatcher.jobs, "stream_exec", timing_out_stream_exec)

  job = {
    "id": "w-fix2-reset-fetch",
    "action": "git_reset_repo_fetch",
    "payload": {},
    "status": "running",
    "log": "",
  }
  asyncio.run(dispatcher._run_tool_job(job))

  assert restores == [("/data/openpilot", snapshot)]
  assert job["status"] == "failed"
  assert job["error_code"] == "CMD_TIMEOUT"
  assert "fetch aborted: restored 1 previous origin refs" in job["log"]
