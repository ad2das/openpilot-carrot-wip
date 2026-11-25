from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.carrot.server.features import system as system_feature
from openpilot.selfdrive.carrot.server.features.tools import dispatcher as tools_dispatcher
from openpilot.selfdrive.carrot.server.services import support_tunnel, terminal_pty, time_sync


def _make_script(tmp_path, name: str, body: str) -> str:
  path = tmp_path / name
  path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
  path.chmod(path.stat().st_mode | stat.S_IXUSR)
  return str(path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX subprocess test")
def test_support_tunnel_eof_does_not_busy_spin_and_exits_promptly(tmp_path, monkeypatch):
  script = _make_script(tmp_path, "exit_fast.sh", "echo 'failed to start' >&2\nexit 1")
  monkeypatch.delenv("CARROT_SUPPORT_FAKE_TUNNEL_URL", raising=False)
  monkeypatch.setattr(support_tunnel, "cloudflared_path", lambda: script)

  async def run():
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="failed to start"):
      await support_tunnel.start_quick_tunnel("http://127.0.0.1:9999", timeout_s=5.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"EOF spun until timeout: {elapsed:.2f}s"

  asyncio.run(run())


@pytest.mark.skipif(os.name == "nt", reason="POSIX subprocess test")
def test_support_tunnel_stderr_url_drains_stdout_without_concurrent_waiter_crash(tmp_path, monkeypatch):
  # Real cloudflared prints URL on stderr while stdout stays open; cancelling the
  # pending stdout reader must complete before _drain_stream attaches to stdout.
  script = _make_script(
    tmp_path,
    "stderr_url.sh",
    "echo ''\n"
    "echo 'https://steady-drain-123.trycloudflare.com' >&2\n"
    "for i in $(seq 1 200); do echo \"stdout-line-$i\"; done\n"
    "sleep 30",
  )
  monkeypatch.delenv("CARROT_SUPPORT_FAKE_TUNNEL_URL", raising=False)
  monkeypatch.setattr(support_tunnel, "cloudflared_path", lambda: script)

  async def run():
    handle = await support_tunnel.start_quick_tunnel("http://127.0.0.1:9999", timeout_s=5.0)
    try:
      assert handle.url == "https://steady-drain-123.trycloudflare.com"
      await asyncio.sleep(0.2)
      for task in handle.drain_tasks:
        if task.done():
          assert task.exception() is None, f"drain task crashed: {task.exception()!r}"
      assert any("stdout-line-200" in line for line in handle.log)
    finally:
      await handle.stop()
      assert handle.process is not None and handle.process.returncode is not None

  asyncio.run(run())


@pytest.mark.skipif(os.name == "nt", reason="POSIX subprocess test")
def test_support_tunnel_closed_stdout_waits_for_stderr_without_spinning(tmp_path, monkeypatch):
  # stdout closes immediately while stderr emits URL after a short delay.
  script = _make_script(
    tmp_path,
    "stdout_eof_first.sh",
    "exec 1>&-\n"
    "sleep 0.15\n"
    "echo 'https://stderr-after-eof.trycloudflare.com' >&2\n"
    "sleep 30",
  )
  monkeypatch.delenv("CARROT_SUPPORT_FAKE_TUNNEL_URL", raising=False)
  monkeypatch.setattr(support_tunnel, "cloudflared_path", lambda: script)

  async def run():
    handle = await support_tunnel.start_quick_tunnel("http://127.0.0.1:9999", timeout_s=3.0)
    try:
      assert handle.url == "https://stderr-after-eof.trycloudflare.com"
    finally:
      await handle.stop()

  asyncio.run(run())


def test_resolve_zoneinfo_path_blocks_traversal_and_directories(tmp_path):
  zone_root = tmp_path / "zoneinfo"
  (zone_root / "Asia").mkdir(parents=True)
  seoul = zone_root / "Asia" / "Seoul"
  seoul.write_bytes(b"TZif2")
  outside = tmp_path / "passwd"
  outside.write_text("secret", encoding="utf-8")
  escape_link = zone_root / "Escape"
  try:
    os.symlink(str(outside), str(escape_link))
  except OSError:
    escape_link = None

  assert time_sync.resolve_zoneinfo_path("Asia/Seoul", str(zone_root)) == os.path.realpath(str(seoul))
  assert time_sync.resolve_zoneinfo_path("", str(zone_root)) is None
  assert time_sync.resolve_zoneinfo_path("Asia", str(zone_root)) is None
  assert time_sync.resolve_zoneinfo_path("../passwd", str(zone_root)) is None
  assert time_sync.resolve_zoneinfo_path("Asia/../../passwd", str(zone_root)) is None
  assert time_sync.resolve_zoneinfo_path("/etc/passwd", str(zone_root)) is None
  assert time_sync.resolve_zoneinfo_path("Asia/Seoul\x00", str(zone_root)) is None
  if escape_link is not None:
    assert time_sync.resolve_zoneinfo_path("Escape", str(zone_root)) is None


def test_time_sync_rejects_traversal_and_uses_argv_with_timeout(tmp_path, monkeypatch):
  calls = []

  def fake_run(cmd, **kwargs):
    calls.append((cmd, kwargs))
    return SimpleNamespace(returncode=0, stdout="", stderr="")

  monkeypatch.setattr(time_sync.subprocess, "run", fake_run)

  bad = time_sync.sync_system_time_from_browser(1_700_000_000_000, "../../../etc/passwd")
  assert bad["ok"] is False
  assert calls == []

  zone_root = tmp_path / "zoneinfo"
  (zone_root / "Asia").mkdir(parents=True)
  seoul = zone_root / "Asia" / "Seoul"
  seoul.write_bytes(b"TZif2")
  monkeypatch.setattr(time_sync, "ZONEINFO_ROOT", str(zone_root))

  res = time_sync.sync_system_time_from_browser(1_700_000_000_000, "Asia/Seoul")
  assert res["ok"] is True
  assert res["applied"] is True
  assert calls
  for cmd, kwargs in calls:
    assert isinstance(cmd, list)
    assert not kwargs.get("shell", False)
    assert kwargs.get("timeout") == time_sync.SUBPROCESS_TIMEOUT_SEC
  assert calls[-1][0][:4] == ["sudo", "env", "TZ=UTC", "date"]


def test_api_time_sync_rejects_bool_and_non_finite_epoch_ms():
  async def run():
    for bad_epoch in (True, False, float("nan"), float("inf"), float("-inf"), "123", None):
      req = SimpleNamespace(
        remote="127.0.0.1",
        json=lambda b=bad_epoch: asyncio.sleep(0, result={"epoch_ms": b, "timezone": "UTC"}),
      )
      resp = await system_feature.api_time_sync(req)
      assert resp.status == 400
      body = json.loads(resp.text)
      assert body["ok"] is False

  asyncio.run(run())


def test_capture_tmux_log_sync_returns_124_on_timeout(tmp_path, monkeypatch):
  monkeypatch.setattr(tools_dispatcher, "TMUX_LOG_PATH", str(tmp_path / "tmux.log"))

  def fake_run(cmd, **kwargs):
    assert kwargs.get("timeout") == 15
    raise subprocess.TimeoutExpired(cmd=cmd, timeout=15)

  monkeypatch.setattr(tools_dispatcher.subprocess, "run", fake_run)
  rc, out = tools_dispatcher.capture_tmux_log_sync()
  assert rc == 124
  assert "timed out" in out


@pytest.mark.skipif(os.name == "nt", reason="POSIX PTY test")
def test_terminal_pty_terminate_reaps_child_and_allows_clean_restart(monkeypatch):
  # Avoid requiring tmux: launch a plain sleeping python process in the PTY.
  monkeypatch.setattr(
    terminal_pty.tmux,
    "start_command",
    lambda: f"{sys.executable} -u -c 'import time; print(\"ready\", flush=True); time.sleep(30)'",
  )

  async def run():
    session = terminal_pty.PersistentPtySession()
    assert await session.ensure() is True
    proc1 = session.proc
    assert proc1 is not None
    await asyncio.sleep(0.15)
    await session.terminate()
    assert proc1.poll() is not None, "child process was not reaped after terminate()"
    assert session.proc is None
    assert session.master_fd == -1

    # Immediate restart must not be clobbered by old reader cleanup.
    assert await session.ensure() is True
    proc2 = session.proc
    assert proc2 is not None and proc2.pid != proc1.pid
    await asyncio.sleep(0.15)
    assert session._alive_locked() is True
    await session.terminate()
    assert proc2.poll() is not None

  asyncio.run(run())


def test_carrot_serv_set_time_invalid_timezone_preserves_localtime_and_uses_timeout(monkeypatch):
  from openpilot.selfdrive.carrot import carrot_serv

  calls = []
  puts = []

  def fake_run(cmd, **kwargs):
    calls.append((cmd, kwargs))
    return SimpleNamespace(returncode=0, stdout="", stderr="")

  monkeypatch.setattr(carrot_serv.subprocess, "run", fake_run)
  serv = carrot_serv.CarrotServ.__new__(carrot_serv.CarrotServ)
  serv.params = SimpleNamespace(put=lambda k, v: puts.append((k, v)))

  # Invalid traversal timezone must NEVER call rm/ln on /data/etc/localtime
  serv.set_time(1_700_000_000, "../../../etc/passwd")
  assert puts == []
  assert len(calls) == 1
  cmd, kwargs = calls[0]
  assert cmd[:2] == ["date", "-s"]
  assert not kwargs.get("shell", False)
  assert kwargs.get("timeout") == 10
  assert kwargs.get("env", {}).get("TZ") == "UTC"


def test_carrot_man_make_tmux_data_uses_timeout(monkeypatch):
  from openpilot.selfdrive.carrot import carrot_man

  seen = []

  def fake_run(cmd, **kwargs):
    seen.append(kwargs.get("timeout"))
    raise subprocess.TimeoutExpired(cmd=cmd, timeout=15)

  monkeypatch.setattr(carrot_man.subprocess, "run", fake_run)
  cm = carrot_man.CarrotMan.__new__(carrot_man.CarrotMan)
  assert cm.make_tmux_data() is False
  assert seen == [15]

