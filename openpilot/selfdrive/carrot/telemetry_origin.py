"""Upstream identity reported in server-facing payloads.

Local Params, UI and updater keep the real fork values. Payloads that leave
the device report the upstream carrot-wip identity instead, and the retired
compatibility override is reported at its original default, so fleet uploads
match a stock device. The upstream commit is refreshed by the fork sync
workflow in upstream_origin.json; until that file exists the last merged
upstream commit is used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UPSTREAM_REMOTE = "https://github.com/ajouatom/openpilot.git"
UPSTREAM_BRANCH = "carrot-wip"
ORIGIN_FILE = Path(__file__).with_name("upstream_origin.json")

_FALLBACK_ORIGIN = {
  "remote": UPSTREAM_REMOTE,
  "branch": UPSTREAM_BRANCH,
  "commit": "38bc4eeeea136f8f117880a840164b0e1dd1186e",
  "short_commit": "38bc4eee",
  "commit_date": "'1790588996 2026-09-28 18:49:56 +0900'",
  "commit_datetime": "2026-09-28 18:49:56",
}

# Fork-only compatibility keys report the upstream default instead of the
# locally saved override; keys stay present in the payload.
DEFAULT_REPORT_VALUES = {
  "DisableDM": "0",
  "DisableDMActive": "0",
}

_cache: tuple[str, float, dict[str, str]] | None = None


def _read_origin() -> dict[str, str]:
  global _cache

  path = ORIGIN_FILE
  try:
    mtime = path.stat().st_mtime
  except OSError:
    return dict(_FALLBACK_ORIGIN)

  if _cache is not None and _cache[0] == str(path) and _cache[1] == mtime:
    return _cache[2]

  origin = dict(_FALLBACK_ORIGIN)
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    data = None
  if isinstance(data, dict):
    for key in origin:
      value = data.get(key)
      if isinstance(value, str) and value.strip():
        origin[key] = value.strip()

  _cache = (str(path), mtime, origin)
  return origin


def reported_git_remote() -> str:
  return _read_origin()["remote"]


def reported_git_branch() -> str:
  return _read_origin()["branch"]


def reported_git_commit() -> str:
  return _read_origin()["commit"]


def reported_git_short_commit() -> str:
  return _read_origin()["short_commit"]


def reported_git_commit_date() -> str:
  return _read_origin()["commit_date"]


def reported_git_commit_datetime() -> str:
  return _read_origin()["commit_datetime"]


def reported_param_value(name: str, value: Any) -> Any:
  default = DEFAULT_REPORT_VALUES.get(str(name))
  if default is not None:
    return default
  return value
