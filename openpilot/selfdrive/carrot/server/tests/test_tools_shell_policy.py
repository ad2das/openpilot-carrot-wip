from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from openpilot.selfdrive.carrot.server.features.tools.actions import validate_shell_argv
from openpilot.selfdrive.carrot.server.features.tools.dispatcher import _clear_origin_remote_refs


def test_validate_shell_argv_blocks_mutating_git_and_secret_cat_paths():
  # Allowed read-only commands stay allowed.
  assert validate_shell_argv(["git", "status"]) is None
  assert validate_shell_argv(["git", "branch", "-a"]) is None
  assert validate_shell_argv(["git", "branch", "--show-current"]) is None
  assert validate_shell_argv(["git", "remote", "-v"]) is None
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
  assert validate_shell_argv(["git", "branch", "new-branch"]) is not None
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
