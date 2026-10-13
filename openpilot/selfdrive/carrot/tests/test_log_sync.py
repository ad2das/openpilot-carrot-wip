import asyncio
import json
import os
import time
from types import SimpleNamespace

from openpilot.selfdrive.carrot import log_sync
from openpilot.selfdrive.carrot.server.features.dashcam import catalog
from openpilot.selfdrive.carrot.server.services.web_settings import sanitize_web_settings


BASE_URL = "https://anjh.duckdns.org"
ROUTE = "0000002a--1a2b3c4d5e"


def write_segment(root, name, files=("rlog.zst", "qcamera.ts"), mtime=None):
  seg_dir = os.path.join(str(root), name)
  os.makedirs(seg_dir, exist_ok=True)
  for filename in files:
    with open(os.path.join(seg_dir, filename), "wb") as f:
      f.write(b"carrot")
  if mtime is not None:
    os.utime(seg_dir, (mtime, mtime))
  return seg_dir


def prepare(monkeypatch, tmp_path, settings, upload_result=True):
  monkeypatch.setattr(log_sync, "DASHCAM_ROOT", str(tmp_path))
  monkeypatch.setattr(catalog, "DASHCAM_ROOT", str(tmp_path))
  monkeypatch.setattr(log_sync, "read_web_settings", lambda: dict(settings))
  monkeypatch.setattr(log_sync, "upload_metadata", lambda params: {"dongleId": "test-dongle"})
  # Other Windows test modules replace openpilot.common.swaglog with a stub
  # that has no event(); keep this module's logging hermetic.
  monkeypatch.setattr(log_sync, "cloudlog", SimpleNamespace(
    event=lambda *args, **kwargs: None,
    exception=lambda *args, **kwargs: None,
  ))
  uploads = []

  async def fake_create_session(base_url, metadata, purpose="dashcam"):
    return "test-token"

  async def fake_upload(local_folder, directory, remote_path, base_url, token, should_cancel, filenames=None, on_progress=None):
    uploads.append({
      "folder": local_folder,
      "directory": directory,
      "remote_path": remote_path,
      "base_url": base_url,
      "token": token,
      "filenames": list(filenames or []),
    })
    return upload_result

  monkeypatch.setattr(log_sync, "create_web_upload_session", fake_create_session)
  monkeypatch.setattr(log_sync, "upload_folder_to_web", fake_upload)
  return uploads


def test_disabled_setting_never_uploads(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0")
  for settings in ({"log_sync_enabled": False, "log_sync_url": BASE_URL}, {}):
    uploads = prepare(monkeypatch, tmp_path, settings)
    sync = log_sync.LogSync()
    assert asyncio.run(sync.run_once()) == log_sync.DISABLED_POLL_S
    assert uploads == []


def test_shared_upload_url_as_log_sync_url_never_uploads(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0")
  uploads = prepare(monkeypatch, tmp_path, {
    "log_sync_enabled": True,
    "log_sync_url": log_sync.DEFAULT_WEB_UPLOAD_URL,
  })
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.DISABLED_POLL_S
  assert uploads == []


def test_locked_segment_is_skipped_until_unlocked(tmp_path, monkeypatch):
  seg_dir = write_segment(tmp_path, f"{ROUTE}--0", files=("rlog.zst", "qcamera.ts", "rlog.lock"))
  uploads = prepare(monkeypatch, tmp_path, {"log_sync_enabled": True, "log_sync_url": BASE_URL})
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []

  os.remove(os.path.join(seg_dir, "rlog.lock"))
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"{ROUTE}--0"]


def test_newest_segment_is_uploaded_first(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0")
  write_segment(tmp_path, f"{ROUTE}--1")
  uploads = prepare(monkeypatch, tmp_path, {"log_sync_enabled": True, "log_sync_url": BASE_URL})
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"{ROUTE}--1", f"{ROUTE}--0"]


def test_success_writes_marker_and_is_not_uploaded_again(tmp_path, monkeypatch):
  seg_dir = write_segment(tmp_path, f"{ROUTE}--0")
  uploads = prepare(monkeypatch, tmp_path, {"log_sync_enabled": True, "log_sync_url": BASE_URL})
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert uploads[0]["directory"] == "test-dongle"
  assert uploads[0]["token"] == "test-token"
  assert sorted(uploads[0]["filenames"]) == ["qcamera.ts", "rlog.zst"]

  marker_path = os.path.join(seg_dir, log_sync.MARKER_NAME)
  with open(marker_path, encoding="utf-8") as f:
    marker = json.load(f)
  assert marker["base_url"] == BASE_URL
  assert marker["files"] == {"qcamera.ts": 6, "rlog.zst": 6}

  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert len(uploads) == 1


def test_marker_with_other_url_or_size_is_uploaded_again(tmp_path, monkeypatch):
  seg_dir = write_segment(tmp_path, f"{ROUTE}--0")
  uploads = prepare(monkeypatch, tmp_path, {"log_sync_enabled": True, "log_sync_url": BASE_URL})
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5

  marker_path = os.path.join(seg_dir, log_sync.MARKER_NAME)
  with open(marker_path, "w", encoding="utf-8") as f:
    json.dump({"base_url": "https://other.example", "files": log_sync.wanted_files(seg_dir)}, f)
  assert asyncio.run(sync.run_once()) == 0.5

  with open(os.path.join(seg_dir, "rlog.zst"), "ab") as f:
    f.write(b"more")
  assert asyncio.run(sync.run_once()) == 0.5
  assert len(uploads) == 3


def test_failures_back_off_to_the_cap(tmp_path, monkeypatch):
  seg_dir = write_segment(tmp_path, f"{ROUTE}--0")
  uploads = prepare(monkeypatch, tmp_path, {"log_sync_enabled": True, "log_sync_url": BASE_URL}, upload_result=False)
  sync = log_sync.LogSync()
  waits = [asyncio.run(sync.run_once()) for _ in range(7)]
  assert waits == [30.0, 60.0, 120.0, 240.0, 480.0, 600.0, 600.0]
  assert not os.path.exists(os.path.join(seg_dir, log_sync.MARKER_NAME))
  assert len(uploads) == 7


def test_segment_older_than_max_age_is_skipped(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0", mtime=time.time() - 8 * 86400)  # noqa: TID251
  write_segment(tmp_path, f"{ROUTE}--1")
  uploads = prepare(monkeypatch, tmp_path, {"log_sync_enabled": True, "log_sync_url": BASE_URL})
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert [u["remote_path"] for u in uploads] == [f"{ROUTE}--1"]


def test_enabled_without_log_sync_url_uploads_to_default(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0")
  uploads = prepare(monkeypatch, tmp_path, {"log_sync_enabled": True})
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["base_url"] for u in uploads] == ["https://anjh.duckdns.org"]


def test_web_upload_url_does_not_change_log_sync_target(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0")
  uploads = prepare(monkeypatch, tmp_path, {
    "log_sync_enabled": True,
    "log_sync_url": BASE_URL,
    "web_upload_url": "https://manual-upload.example",
  })
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["base_url"] for u in uploads] == [BASE_URL]


def test_module_never_calls_shared_notification_paths():
  with open(log_sync.__file__, encoding="utf-8") as f:
    source = f.read()
  assert "send_discord_webhook" not in source
  assert "send_web_upload_complete" not in source
  assert "run_upload_segments" not in source


def test_web_settings_round_trip_log_sync_enabled():
  assert sanitize_web_settings({})["log_sync_enabled"] is False
  assert sanitize_web_settings({"log_sync_enabled": True})["log_sync_enabled"] is True
  assert sanitize_web_settings({})["log_sync_url"] == "https://anjh.duckdns.org"
  assert sanitize_web_settings({"log_sync_url": "ftp://x"})["log_sync_url"] == "https://anjh.duckdns.org"


def test_main_is_synchronous_and_runs_the_loop(monkeypatch):
  # system/manager/process.py launcher() calls mod.main() without awaiting it.
  import inspect
  assert not inspect.iscoroutinefunction(log_sync.main)
  ran = []

  async def fake_run():
    ran.append(True)

  monkeypatch.setattr(log_sync, "run", fake_run)
  monkeypatch.setattr(log_sync.os, "nice", lambda _: 0, raising=False)
  log_sync.main()
  assert ran == [True]
