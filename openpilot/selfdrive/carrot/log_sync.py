import asyncio
import json
import os
import re
import shutil
import time
from typing import Any

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.carrot.server.config import DASHCAM_ROOT
from openpilot.selfdrive.carrot.server.features.dashcam.catalog import (
  segment_creation_key,
  segment_file_summary,
  segment_is_complete,
)
from openpilot.selfdrive.carrot.server.features.dashcam.upload import upload_metadata
from openpilot.selfdrive.carrot.server.services.web_settings import _normalize_log_sync_url, read_web_settings
from openpilot.selfdrive.carrot.web_upload import (
  DEFAULT_WEB_UPLOAD_URL,
  create_web_upload_session,
  upload_device_id,
  upload_folder_to_web,
)


POLL_S = 10.0          # idle / between-segment poll
DISABLED_POLL_S = 15.0
MAX_AGE_S = 7 * 86400  # never sync segments whose directory mtime is older
BACKOFF_MIN_S = 30.0
BACKOFF_MAX_S = 60.0
SESSION_MAX_AGE_S = 3 * 3600
MARKER_NAME = ".carrot_log_sync.json"

TLVREC_ROOT = "/data/media/0/carrot_tlv_rec"
TLVREC_SEGMENT_PREFIX = "tlvrec--"
TLVREC_SESSION_RE = re.compile(r"^\d{8}-\d{6}$")
TLVREC_FILE_RE = re.compile(r"^(?:\d{7}\.jpg|index\.jsonl)$")
TLVREC_ACTIVE_NAME = ".active"
TLVREC_ACTIVE_FRESH_S = 60.0
TLVREC_QUIET_S = 120.0


def target_url(settings: dict[str, Any]) -> str | None:
  try:
    if settings.get("log_sync_enabled") is not True:
      return None
    base_url = _normalize_log_sync_url(settings.get("log_sync_url"))
    if base_url.rstrip("/").casefold() == DEFAULT_WEB_UPLOAD_URL.rstrip("/").casefold():
      return None
    return base_url
  except Exception:
    return None


def wanted_files(seg_dir: str) -> dict[str, int]:
  try:
    return {item["name"]: int(item["size"]) for item in segment_file_summary(seg_dir)}
  except Exception:
    return {}


def is_synced(seg_dir: str, base_url: str, files: dict[str, int]) -> bool:
  try:
    with open(os.path.join(seg_dir, MARKER_NAME), encoding="utf-8") as f:
      marker = json.load(f)
  except Exception:
    return False
  return marker == {"base_url": base_url, "files": files}


def mark_synced(seg_dir: str, base_url: str, files: dict[str, int]) -> None:
  path = os.path.join(seg_dir, MARKER_NAME)
  tmp_path = path + ".tmp"
  with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump({"base_url": base_url, "files": files}, f)
  os.replace(tmp_path, path)


def tlvrec_files(session_dir: str) -> dict[str, int]:
  try:
    with os.scandir(session_dir) as entries:
      scanned = list(entries)
  except OSError:
    return {}
  files: dict[str, int] = {}
  for entry in scanned:
    try:
      if entry.is_file(follow_symlinks=False) and TLVREC_FILE_RE.fullmatch(entry.name):
        files[entry.name] = int(entry.stat(follow_symlinks=False).st_size)
    except OSError:
      continue
  return files


def _tlvrec_active_session(now: float) -> str | None:
  try:
    active_path = os.path.join(TLVREC_ROOT, TLVREC_ACTIVE_NAME)
    with open(active_path, encoding="utf-8") as f:
      name = f.read().strip()
    if now - os.path.getmtime(active_path) < TLVREC_ACTIVE_FRESH_S:
      return name
  except (OSError, ValueError):
    return None
  return None


def _newest_mtime(session_dir: str) -> float:
  try:
    newest = os.path.getmtime(session_dir)
    with os.scandir(session_dir) as entries:
      for entry in entries:
        try:
          newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
        except OSError:
          continue
    return newest
  except OSError:
    return time.time()  # noqa: TID251


def next_tlvrec_session(now: float) -> tuple[str, dict[str, int]] | None:
  """Oldest quiet finished tlv recorder session, or None while all are busy."""
  try:
    names = os.listdir(TLVREC_ROOT)
  except OSError:
    return None
  active = _tlvrec_active_session(now)
  for name in sorted(names):
    if not TLVREC_SESSION_RE.fullmatch(name):
      continue
    session_dir = os.path.join(TLVREC_ROOT, name)
    try:
      if not os.path.isdir(session_dir) or os.path.islink(session_dir):
        continue
    except OSError:
      continue
    if name == active:
      continue
    if now - _newest_mtime(session_dir) < TLVREC_QUIET_S:
      continue
    files = tlvrec_files(session_dir)
    if not files:
      try:
        leftovers = os.listdir(session_dir)
      except OSError:
        continue
      if not leftovers:
        try:
          os.rmdir(session_dir)
        except OSError:
          pass
        cloudlog.event("carrot_log_sync_tlvrec_empty", session=name)
      continue
    return name, files
  return None


def next_segment(base_url: str, now: float) -> tuple[str, dict[str, int]] | None:
  best: tuple[str, dict[str, int]] | None = None
  best_key: tuple[int, int, str, str, int, str] | None = None
  for name in os.listdir(DASHCAM_ROOT):
    try:
      seg_dir = os.path.join(DASHCAM_ROOT, name)
      if not os.path.isdir(seg_dir):
        continue
      if now - os.path.getmtime(seg_dir) > MAX_AGE_S:
        continue
      if not segment_is_complete(name):
        continue
      files = wanted_files(seg_dir)
      if not files:
        continue
      if is_synced(seg_dir, base_url, files):
        continue
      key = segment_creation_key(name)
    except OSError:
      continue
    if best_key is None or key > best_key:
      best_key = key
      best = (name, files)
  return best


class LogSync:
  def __init__(self) -> None:
    self._session: tuple[str, str, dict[str, Any], float] | None = None
    self._backoff = BACKOFF_MIN_S

  async def _token_for(self, base_url: str) -> tuple[str, dict[str, Any]]:
    cached = self._session
    if cached is not None:
      cached_url, token, meta, created = cached
      if cached_url == base_url and time.monotonic() - created < SESSION_MAX_AGE_S:
        return token, meta
    meta = upload_metadata(Params())
    token = await create_web_upload_session(base_url, meta, "dashcam")
    self._session = (base_url, token, meta, time.monotonic())
    return token, meta

  async def run_once(self) -> float:
    base_url = target_url(read_web_settings())
    if base_url is None:
      self._session = None
      return DISABLED_POLL_S
    picked = next_segment(base_url, time.time())  # noqa: TID251
    if picked is None:
      return await self._sync_tlvrec(base_url)
    name, files = picked
    seg_dir = os.path.join(DASHCAM_ROOT, name)
    ok = False
    error: str | None = None
    try:
      token, meta = await self._token_for(base_url)
      ok = await upload_folder_to_web(seg_dir, upload_device_id(meta), name, base_url, token, None, filenames=list(files))
    except Exception as e:
      error = str(e)
    if ok:
      mark_synced(seg_dir, base_url, files)
      self._backoff = BACKOFF_MIN_S
      cloudlog.event("carrot_log_sync_ok", segment=name)
      return 0.5
    self._session = None
    cloudlog.event("carrot_log_sync_fail", segment=name, error=error or "upload failed")
    backoff = self._backoff
    self._backoff = min(BACKOFF_MAX_S, self._backoff * 2)
    return backoff

  async def _sync_tlvrec(self, base_url: str) -> float:
    """Drive segments always go first; tlv sessions are considered only here."""
    picked = next_tlvrec_session(time.time())  # noqa: TID251
    if picked is None:
      return POLL_S
    name, files = picked
    session_dir = os.path.join(TLVREC_ROOT, name)
    remote = TLVREC_SEGMENT_PREFIX + name
    filenames = sorted(filename for filename in files if filename != "index.jsonl")
    if "index.jsonl" in files:
      filenames.append("index.jsonl")
    ok = False
    error: str | None = None
    try:
      token, meta = await self._token_for(base_url)
      ok = await upload_folder_to_web(session_dir, upload_device_id(meta), remote, base_url, token, None, filenames=filenames)
    except Exception as e:
      error = str(e)
    if ok:
      after = tlvrec_files(session_dir)
      if after == files and _tlvrec_active_session(time.time()) != name:  # noqa: TID251
        shutil.rmtree(session_dir, ignore_errors=True)
        cloudlog.event("carrot_log_sync_tlvrec_ok", session=name, files=len(files), bytes=sum(files.values()))
      else:
        cloudlog.event("carrot_log_sync_tlvrec_changed", session=name)
      self._backoff = BACKOFF_MIN_S
      return 0.5
    self._session = None
    cloudlog.event("carrot_log_sync_fail", segment=remote, error=error or "upload failed")
    backoff = self._backoff
    self._backoff = min(BACKOFF_MAX_S, self._backoff * 2)
    return backoff


async def run() -> None:
  sync = LogSync()
  while True:
    try:
      await asyncio.sleep(await sync.run_once())
    except Exception:
      cloudlog.exception("carrot_log_sync")
      await asyncio.sleep(BACKOFF_MIN_S)


def main() -> None:
  # The manager launcher calls main() synchronously; a coroutine here would exit at once.
  try:
    os.nice(15)
  except OSError:
    pass
  asyncio.run(run())


if __name__ == "__main__":
  main()
