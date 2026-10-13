import asyncio
import json
import os
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
BACKOFF_MAX_S = 600.0
SESSION_MAX_AGE_S = 3 * 3600
MARKER_NAME = ".carrot_log_sync.json"


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
      return POLL_S
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


async def main() -> None:
  try:
    os.nice(15)
  except OSError:
    pass
  sync = LogSync()
  while True:
    try:
      await asyncio.sleep(await sync.run_once())
    except Exception:
      cloudlog.exception("carrot_log_sync")
      await asyncio.sleep(BACKOFF_MIN_S)


if __name__ == "__main__":
  asyncio.run(main())
