"""Tests for openpilot.common.build_artifact_fetch (best-effort warm fetch).

The helper must never raise and never decide an update outcome, so the update
paths can always continue when the artifact is missing or the network is down.
"""
import asyncio
import subprocess
import sys
from pathlib import Path

from openpilot.common import build_artifact_fetch
from openpilot.common.basedir import BASEDIR
from openpilot.common.build_artifact_fetch import (
  FETCH_NOT_PUBLISHED,
  FETCH_UNAVAILABLE,
  fetch_build_artifact,
  fetch_build_artifact_async,
)


def _write_tool(repo, body: str) -> None:
  tool = repo / "scripts" / "carrot_build_artifact.py"
  tool.parent.mkdir(parents=True, exist_ok=True)
  tool.write_text(f"#!/usr/bin/env python3\n{body}\n", encoding="utf-8")


def test_missing_degenerate_inputs_are_unavailable(tmp_path) -> None:
  assert fetch_build_artifact(tmp_path, "") == FETCH_UNAVAILABLE
  assert fetch_build_artifact(tmp_path, "a" * 40) == FETCH_UNAVAILABLE  # no tool in the checkout


def test_tool_exit_code_is_returned(tmp_path) -> None:
  _write_tool(tmp_path, "import sys; sys.exit(3)")
  assert fetch_build_artifact(tmp_path, "a" * 40, timeout=10) == FETCH_NOT_PUBLISHED

  _write_tool(tmp_path, "import sys; sys.exit(0)")
  assert fetch_build_artifact(tmp_path, "a" * 40, timeout=10) == 0


def test_timeout_is_unavailable(tmp_path, monkeypatch) -> None:
  _write_tool(tmp_path, "print('never runs')")

  def timeout(*args, **kwargs):
    raise subprocess.TimeoutExpired("cmd", 1)

  monkeypatch.setattr(build_artifact_fetch.subprocess, "run", timeout)
  assert fetch_build_artifact(tmp_path, "a" * 40, timeout=0.5) == FETCH_UNAVAILABLE


def test_launch_errors_are_unavailable(tmp_path, monkeypatch) -> None:
  _write_tool(tmp_path, "print('never runs')")

  def broken(*args, **kwargs):
    raise OSError("no python here")

  monkeypatch.setattr(build_artifact_fetch.subprocess, "run", broken)
  assert fetch_build_artifact(tmp_path, "a" * 40) == FETCH_UNAVAILABLE


def test_uses_sys_executable_and_reports_output(tmp_path, monkeypatch, capsys) -> None:
  _write_tool(tmp_path, "print('never runs')")
  seen = {}

  def fake_run(cmd, **kwargs):
    seen["cmd"] = cmd
    return subprocess.CompletedProcess(cmd, 1, stdout="boom", stderr="")

  monkeypatch.setattr(build_artifact_fetch.subprocess, "run", fake_run)
  assert fetch_build_artifact(tmp_path, "b" * 40, timeout=60) == 1
  assert seen["cmd"][0] == sys.executable
  assert "fetch" in seen["cmd"] and "b" * 40 in seen["cmd"]
  assert "boom" in capsys.readouterr().out


def test_async_wrapper_never_raises(tmp_path, monkeypatch) -> None:
  def boom(*args, **kwargs):
    raise RuntimeError("artifact server down")

  monkeypatch.setattr(build_artifact_fetch, "fetch_build_artifact", boom)
  assert asyncio.run(fetch_build_artifact_async(tmp_path, "a" * 40)) is None


def test_async_wrapper_passes_codes_and_timeout(tmp_path, monkeypatch) -> None:
  seen = {}

  def fake(repo, sha, timeout):
    seen.update({"repo": repo, "sha": sha, "timeout": timeout})
    return FETCH_NOT_PUBLISHED

  monkeypatch.setattr(build_artifact_fetch, "fetch_build_artifact", fake)
  assert asyncio.run(fetch_build_artifact_async(tmp_path, "b" * 40, 60.0)) == FETCH_NOT_PUBLISHED
  assert seen == {"repo": tmp_path, "sha": "b" * 40, "timeout": 60.0}


def test_dispatcher_warms_the_artifact_cache_before_pulling() -> None:
  # The web git_pull paths are heavy modules to import (aiohttp, capnp, D-Bus),
  # so their ordering is asserted from the source: both paths must warm the
  # artifact cache for the verified target before touching the checkout.
  text = (Path(BASEDIR) / "openpilot/selfdrive/carrot/server/features/tools/dispatcher.py").read_text(encoding="utf-8")
  assert text.count("fetch_build_artifact_async") == 3  # the import plus both git_pull paths

  job_prepare = text.find("prepare_git_pull, repo_dir")
  job_fetch = text.find("fetch_build_artifact_async", job_prepare)
  job_reset = text.find('message="git reset --hard"')
  assert 0 <= job_prepare < job_fetch < job_reset

  sync_prepare = text.find("prepare_git_pull, REPO_DIR")
  sync_fetch = text.find("fetch_build_artifact_async", sync_prepare)
  sync_merge = text.find('"git", "merge", "--ff-only", target_head', sync_prepare)
  assert 0 <= sync_prepare < sync_fetch < sync_merge
