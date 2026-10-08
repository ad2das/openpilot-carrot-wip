"""Best-effort fetch of the CI build artifact for an update target.

The update paths (automatic update, web git pull, startup recovery) warm the
device artifact cache for the commit they are about to check out, so the next
boot can install it instead of compiling. Fetching is optional by design: the
launcher and the on-device build still work when nothing was fetched, so this
helper never raises and never decides an update outcome. It only runs when the
checkout ships scripts/carrot_build_artifact.py.

Standard library only: it runs from startup recovery before any Params or
cereal import.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

FETCH_TIMEOUT = 120.0
UPDATE_FETCH_TIMEOUT = 60.0

FETCH_OK = 0
FETCH_ERROR = 1
FETCH_REJECTED = 2
FETCH_NOT_PUBLISHED = 3
FETCH_UNAVAILABLE = -1


def fetch_build_artifact(repo_dir: str | Path, sha: str, timeout: float = UPDATE_FETCH_TIMEOUT) -> int:
  """Fetch <sha> into the artifact cache. Never raises.

  Returns the tool exit code, or FETCH_UNAVAILABLE when the tool could not be
  run at all (missing script, missing python, timeout). The caller treats any
  non-zero result as "no artifact cached"; FETCH_NOT_PUBLISHED additionally
  means the target is valid but its artifact does not exist yet.
  """
  if not sha:
    return FETCH_UNAVAILABLE
  tool = Path(repo_dir) / "scripts" / "carrot_build_artifact.py"
  if not tool.is_file():
    print(f"[build_artifact_fetch] {tool} is not present; skipping the artifact fetch", flush=True)
    return FETCH_UNAVAILABLE
  try:
    result = subprocess.run(
      [sys.executable, str(tool), "fetch", sha, "--timeout", str(int(timeout))],
      capture_output=True, text=True, timeout=timeout + 15.0,
    )
  except subprocess.TimeoutExpired:
    print(f"[build_artifact_fetch] fetch of {sha[:12]} timed out; continuing without it", flush=True)
    return FETCH_UNAVAILABLE
  except (OSError, subprocess.SubprocessError) as exc:
    print(f"[build_artifact_fetch] fetch of {sha[:12]} could not run: {exc}", flush=True)
    return FETCH_UNAVAILABLE
  output = (result.stdout + result.stderr).strip()
  if output:
    print(f"[build_artifact_fetch] {output}", flush=True)
  if result.returncode not in (FETCH_OK, FETCH_NOT_PUBLISHED):
    print(f"[build_artifact_fetch] fetch of {sha[:12]} exited {result.returncode}; continuing without it", flush=True)
  return result.returncode


async def fetch_build_artifact_async(repo_dir: str | Path, sha: str, timeout: float = UPDATE_FETCH_TIMEOUT) -> int | None:
  """Await the blocking fetch in a worker thread, swallowing every failure.

  Async callers (the web git_pull paths) use this so a slow or broken artifact
  server can never block or fail the update: any error returns None.
  """
  try:
    return await asyncio.to_thread(fetch_build_artifact, repo_dir, sha, timeout)
  except Exception as exc:
    print(f"[build_artifact_fetch] async fetch of {sha[:12]} skipped: {exc}", flush=True)
    return None
