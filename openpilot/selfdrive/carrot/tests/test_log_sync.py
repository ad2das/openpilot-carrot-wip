import asyncio
import json
import os
import time
from types import SimpleNamespace

import pytest

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
  assert waits == [30.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
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


# ---------------------------------------------------------------------------
# traffic-light recorder sessions
# ---------------------------------------------------------------------------

TLV_SESSION = "20260101-000000"
ENABLED = {"log_sync_enabled": True, "log_sync_url": BASE_URL}


def old_timestamp() -> float:
  return time.time() - 3600.0  # noqa: TID251


def write_tlv_session(root, name=TLV_SESSION, jpgs=("0000001.jpg", "0000002.jpg"),
                      index=True, extra=(), mtime=None):
  session_dir = os.path.join(str(root), name)
  os.makedirs(session_dir, exist_ok=True)
  file_paths = []
  for filename in list(jpgs) + list(extra):
    path = os.path.join(session_dir, filename)
    with open(path, "wb") as f:
      f.write(b"jpg")
    file_paths.append(path)
  if index:
    path = os.path.join(session_dir, "index.jsonl")
    with open(path, "wb") as f:
      f.write(b"{}\n")
    file_paths.append(path)
  if mtime is not None:
    for path in file_paths:
      os.utime(path, (mtime, mtime))
    os.utime(session_dir, (mtime, mtime))
  return session_dir


def tlv_prepare(monkeypatch, tmp_path, settings=ENABLED, upload_result=True):
  tlv_root = tmp_path / "tlvrec"
  tlv_root.mkdir(exist_ok=True)
  monkeypatch.setattr(log_sync, "TLVREC_ROOT", str(tlv_root))
  uploads = prepare(monkeypatch, tmp_path, settings, upload_result=upload_result)
  return tlv_root, uploads


def test_tlvrec_upload_waits_for_drive_segments(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0")
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  write_tlv_session(tlv_root, mtime=old_timestamp())
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"{ROUTE}--0"]

  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"{ROUTE}--0", f"tlvrec--{TLV_SESSION}"]
  assert uploads[1]["folder"] == os.path.join(str(tlv_root), TLV_SESSION)
  assert not os.path.exists(os.path.join(str(tlv_root), TLV_SESSION))


def test_tlvrec_upload_orders_jpgs_then_index_and_deletes(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  session_dir = write_tlv_session(tlv_root, jpgs=("0000002.jpg", "0000001.jpg"),
                                  extra=("x.tmp",), mtime=old_timestamp())
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert len(uploads) == 1
  assert uploads[0]["remote_path"] == f"tlvrec--{TLV_SESSION}"
  assert uploads[0]["directory"] == "test-dongle"
  assert uploads[0]["token"] == "test-token"
  assert uploads[0]["filenames"] == ["0000001.jpg", "0000002.jpg", "index.jsonl"]
  assert not os.path.exists(session_dir)


def test_tlvrec_failure_keeps_session_and_backs_off(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path, upload_result=False)
  session_dir = write_tlv_session(tlv_root, mtime=old_timestamp())
  sync = log_sync.LogSync()
  waits = [asyncio.run(sync.run_once()) for _ in range(3)]
  assert waits == [30.0, 60.0, 60.0]
  assert os.path.isdir(session_dir)
  assert uploads[-1]["remote_path"] == f"tlvrec--{TLV_SESSION}"


def test_tlvrec_active_session_skips_fresh_and_uploads_stale(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  session_dir = write_tlv_session(tlv_root, mtime=old_timestamp())
  active_path = os.path.join(str(tlv_root), log_sync.TLVREC_ACTIVE_NAME)
  with open(active_path, "w", encoding="utf-8") as f:
    f.write(TLV_SESSION)
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []
  assert os.path.isdir(session_dir)

  stale = time.time() - (log_sync.TLVREC_ACTIVE_FRESH_S + 30.0)  # noqa: TID251
  os.utime(active_path, (stale, stale))
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"tlvrec--{TLV_SESSION}"]
  assert not os.path.exists(session_dir)


def test_tlvrec_recent_file_blocks_upload(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  session_dir = write_tlv_session(tlv_root, mtime=old_timestamp())
  recent = time.time() - (log_sync.TLVREC_QUIET_S / 2.0)  # noqa: TID251
  os.utime(os.path.join(session_dir, "0000002.jpg"), (recent, recent))
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []

  old = time.time() - (log_sync.TLVREC_QUIET_S + 30.0)  # noqa: TID251
  os.utime(os.path.join(session_dir, "0000002.jpg"), (old, old))
  assert asyncio.run(sync.run_once()) == 0.5
  assert len(uploads) == 1


def append_tlv_jpg(folder: str, name: str = "0000003.jpg") -> None:
  with open(os.path.join(folder, name), "wb") as f:
    f.write(b"new")


def test_tlvrec_changed_during_upload_keeps_session(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  session_dir = write_tlv_session(tlv_root, mtime=old_timestamp())
  sync = log_sync.LogSync()
  old = old_timestamp()

  async def growing_upload(local_folder, directory, remote_path, base_url, token, should_cancel,
                           filenames=None, on_progress=None):
    uploads.append({"remote_path": remote_path, "filenames": list(filenames or [])})
    append_tlv_jpg(local_folder)
    return True

  monkeypatch.setattr(log_sync, "upload_folder_to_web", growing_upload)
  assert asyncio.run(sync.run_once()) == 0.5
  assert os.path.isdir(session_dir)
  assert os.path.exists(os.path.join(session_dir, "0000003.jpg"))

  os.utime(os.path.join(session_dir, "0000003.jpg"), (old, old))
  os.utime(session_dir, (old, old))
  assert asyncio.run(sync.run_once()) == 0.5
  assert not os.path.exists(session_dir)


def test_tlvrec_oldest_session_uploaded_first(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  session_new = write_tlv_session(tlv_root, name="20260101-010000", mtime=old_timestamp())
  session_old = write_tlv_session(tlv_root, name="20260101-000000", mtime=old_timestamp())
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == ["tlvrec--20260101-000000", "tlvrec--20260101-010000"]
  assert not os.path.exists(session_old)
  assert not os.path.exists(session_new)


def test_tlvrec_never_touched_when_disabled_or_shared_url(tmp_path, monkeypatch):
  session_dir = None
  for settings in ({"log_sync_enabled": False, "log_sync_url": BASE_URL},
                   {"log_sync_enabled": True, "log_sync_url": log_sync.DEFAULT_WEB_UPLOAD_URL}):
    tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path, settings)
    if session_dir is None:
      session_dir = write_tlv_session(tlv_root, mtime=old_timestamp())
    sync = log_sync.LogSync()
    assert asyncio.run(sync.run_once()) == log_sync.DISABLED_POLL_S
    assert uploads == []
    assert os.path.isdir(session_dir)


def test_tlvrec_unknown_entries_are_ignored(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  old = old_timestamp()
  junk_dir = os.path.join(str(tlv_root), "not-a-session")
  os.makedirs(junk_dir)
  junk_file = os.path.join(junk_dir, "0000001.jpg")
  with open(junk_file, "wb") as f:
    f.write(b"x")
  short_dir = os.path.join(str(tlv_root), "2024010-000000")
  os.makedirs(short_dir)
  short_file = os.path.join(short_dir, "index.jsonl")
  with open(short_file, "wb") as f:
    f.write(b"x")
  like_file = os.path.join(str(tlv_root), "20240101-000000")
  with open(like_file, "wb") as f:
    f.write(b"x")
  for path in (junk_file, junk_dir, short_file, short_dir, like_file):
    os.utime(path, (old, old))

  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []
  assert os.path.isdir(junk_dir) and os.path.isdir(short_dir) and os.path.exists(like_file)


def test_tlvrec_symlinked_session_is_skipped(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  old = old_timestamp()
  target = tmp_path / "real_session"
  target.mkdir()
  target_file = target / "0000001.jpg"
  with open(target_file, "wb") as f:
    f.write(b"x")
  os.utime(target_file, (old, old))
  os.utime(target, (old, old))
  link = os.path.join(str(tlv_root), "20240102-000000")
  try:
    os.symlink(str(target), link, target_is_directory=True)
  except (OSError, NotImplementedError):
    pytest.skip("directory symlinks are not permitted here")

  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []
  assert os.path.islink(link)
  assert os.path.isdir(str(target)) and os.path.exists(str(target_file))


def test_tlvrec_missing_root_is_ignored(tmp_path, monkeypatch):
  monkeypatch.setattr(log_sync, "TLVREC_ROOT", str(tmp_path / "does-not-exist"))
  prepare(monkeypatch, tmp_path, ENABLED)
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S


def test_tlvrec_empty_dir_removed_unknown_only_dir_kept(tmp_path, monkeypatch):
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  old = old_timestamp()
  empty_dir = os.path.join(str(tlv_root), "20260101-000000")
  os.makedirs(empty_dir)
  os.utime(empty_dir, (old, old))
  unknown_dir = os.path.join(str(tlv_root), "20260101-010000")
  os.makedirs(unknown_dir)
  unknown_file = os.path.join(unknown_dir, "note.txt")
  with open(unknown_file, "wb") as f:
    f.write(b"x")
  os.utime(unknown_file, (old, old))
  os.utime(unknown_dir, (old, old))

  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []
  assert not os.path.exists(empty_dir)
  assert os.path.isdir(unknown_dir) and os.path.exists(unknown_file)


def test_tlvrec_constants_match_recorder():
  from openpilot.selfdrive.carrot.tlv import recorder
  assert log_sync.TLVREC_ROOT == recorder.DEFAULT_ROOT
  assert log_sync.TLVREC_SESSION_RE.pattern == recorder.SESSION_NAME_RE.pattern
  assert log_sync.TLVREC_ACTIVE_NAME == recorder.ACTIVE_NAME


def test_tlvrec_during_drive_rotation_uploads_old_keeps_new(tmp_path, monkeypatch):
  """60 s sessions: after the recorder rotates and the old session is quiet for
  TLVREC_QUIET_S, it is uploaded and deleted while the new (active) one is left
  alone; between drive segments next_segment is None so tlvrec gets a turn."""
  assert log_sync.TLVREC_QUIET_S == 15.0
  assert log_sync.TLVREC_ACTIVE_FRESH_S == 20.0
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  now = time.time()  # noqa: TID251
  old_session = write_tlv_session(tlv_root, name="20260101-000000",
                                  mtime=now - (log_sync.TLVREC_QUIET_S * 2))
  new_session = write_tlv_session(tlv_root, name="20260101-000100", mtime=now)
  # the recorder has rotated and its heartbeat names only the new session
  active_path = os.path.join(str(tlv_root), log_sync.TLVREC_ACTIVE_NAME)
  with open(active_path, "w", encoding="utf-8") as f:
    f.write("20260101-000100")
  os.utime(active_path, (now, now))

  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == ["tlvrec--20260101-000000"]
  assert not os.path.exists(old_session)
  assert os.path.isdir(new_session)


def test_tlvrec_rotated_session_when_drive_segments_absent(tmp_path, monkeypatch):
  """While driving there is no complete drive segment, so next_segment returns
  None between them and the tlv session is uploaded on that same pass."""
  tlv_root, uploads = tlv_prepare(monkeypatch, tmp_path)
  session_dir = write_tlv_session(tlv_root, mtime=old_timestamp())
  assert log_sync.next_segment(BASE_URL, time.time()) is None  # noqa: TID251
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"tlvrec--{TLV_SESSION}"]
  assert not os.path.exists(session_dir)


# ---------------------------------------------------------------------------
# DSP probe results
# ---------------------------------------------------------------------------

DSP_COMMIT = "8a34d9f"


def write_probe_result(root, name=f"{DSP_COMMIT}.json", body=b"{\"ok\": true}\n", mtime=None):
  path = os.path.join(str(root), name)
  with open(path, "wb") as f:
    f.write(body)
  if mtime is not None:
    os.utime(path, (mtime, mtime))
  return path


def dsp_prepare(monkeypatch, tmp_path, settings=ENABLED, upload_result=True):
  probe_root = tmp_path / "dsp_probe"
  probe_root.mkdir(exist_ok=True)
  tlv_root = tmp_path / "tlvrec"
  tlv_root.mkdir(exist_ok=True)
  uploads = prepare(monkeypatch, tmp_path, settings, upload_result=upload_result)
  monkeypatch.setattr(log_sync, "DSPPROBE_ROOT", str(probe_root))
  monkeypatch.setattr(log_sync, "TLVREC_ROOT", str(tlv_root))
  return probe_root, tlv_root, uploads


def test_dspprobe_result_is_uploaded_once_and_kept(tmp_path, monkeypatch):
  probe_root, _tlv_root, uploads = dsp_prepare(monkeypatch, tmp_path)
  path = write_probe_result(probe_root, mtime=old_timestamp())
  before_bytes = open(path, "rb").read()
  before_mtime_ns = os.stat(path).st_mtime_ns

  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5
  assert len(uploads) == 1
  assert uploads[0]["folder"] == str(probe_root)
  assert uploads[0]["remote_path"] == f"dspprobe--{DSP_COMMIT}"
  assert uploads[0]["filenames"] == [f"{DSP_COMMIT}.json"]
  assert uploads[0]["directory"] == "test-dongle"
  assert os.path.exists(path + log_sync.DSPPROBE_MARKER_SUFFIX)

  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert len(uploads) == 1
  assert open(path, "rb").read() == before_bytes
  assert os.stat(path).st_mtime_ns == before_mtime_ns


def test_dspprobe_runs_after_drive_segments_and_tlvrec(tmp_path, monkeypatch):
  write_segment(tmp_path, f"{ROUTE}--0")
  probe_root, tlv_root, uploads = dsp_prepare(monkeypatch, tmp_path)
  write_tlv_session(tlv_root, mtime=old_timestamp())
  write_probe_result(probe_root, mtime=old_timestamp())

  sync = log_sync.LogSync()
  for _ in range(3):
    assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"{ROUTE}--0", f"tlvrec--{TLV_SESSION}",
                                                 f"dspprobe--{DSP_COMMIT}"]


def test_dspprobe_failure_keeps_result_and_backs_off(tmp_path, monkeypatch):
  probe_root, _tlv_root, uploads = dsp_prepare(monkeypatch, tmp_path, upload_result=False)
  path = write_probe_result(probe_root, mtime=old_timestamp())

  sync = log_sync.LogSync()
  waits = [asyncio.run(sync.run_once()) for _ in range(2)]
  assert waits == [30.0, 60.0]
  assert not os.path.exists(path + log_sync.DSPPROBE_MARKER_SUFFIX)
  assert os.path.exists(path)
  assert [u["remote_path"] for u in uploads] == [f"dspprobe--{DSP_COMMIT}"] * 2


def test_dspprobe_ignores_non_result_entries(tmp_path, monkeypatch):
  probe_root, _tlv_root, uploads = dsp_prepare(monkeypatch, tmp_path)
  old = old_timestamp()
  for name in (f"{DSP_COMMIT}.attempt", f"{DSP_COMMIT}.progress",
               f"{DSP_COMMIT}.json.tmp", f"{DSP_COMMIT}.json{log_sync.DSPPROBE_MARKER_SUFFIX}"):
    write_probe_result(probe_root, name=name, mtime=old)
  os.makedirs(os.path.join(str(probe_root), "x.json"))
  os.utime(os.path.join(str(probe_root), "x.json"), (old, old))
  target = write_probe_result(tmp_path, name="outside.json", mtime=old)
  try:
    os.symlink(target, os.path.join(str(probe_root), "s.json"))
  except (OSError, NotImplementedError):
    pass

  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []
  assert os.path.exists(target)
  assert os.path.isdir(os.path.join(str(probe_root), "x.json"))


def test_dspprobe_waits_for_quiet_result(tmp_path, monkeypatch):
  probe_root, _tlv_root, uploads = dsp_prepare(monkeypatch, tmp_path)
  path = write_probe_result(probe_root, mtime=time.time())  # noqa: TID251

  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S
  assert uploads == []

  old = time.time() - (log_sync.TLVREC_QUIET_S + 30.0)  # noqa: TID251
  os.utime(path, (old, old))
  assert asyncio.run(sync.run_once()) == 0.5
  assert [u["remote_path"] for u in uploads] == [f"dspprobe--{DSP_COMMIT}"]


def test_dspprobe_marker_with_other_url_or_size_is_uploaded_again(tmp_path, monkeypatch):
  probe_root, _tlv_root, uploads = dsp_prepare(monkeypatch, tmp_path)
  old = old_timestamp()
  path = write_probe_result(probe_root, mtime=old)
  marker_path = path + log_sync.DSPPROBE_MARKER_SUFFIX
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == 0.5

  with open(marker_path, "w", encoding="utf-8") as f:
    json.dump({"base_url": "https://other.example", "size": os.path.getsize(path),
               "mtime_ns": os.stat(path).st_mtime_ns}, f)
  assert asyncio.run(sync.run_once()) == 0.5

  with open(path, "ab") as f:
    f.write(b"more")
  os.utime(path, (old, old))
  assert asyncio.run(sync.run_once()) == 0.5
  assert len(uploads) == 3


def test_dspprobe_never_touched_when_disabled_or_shared_url(tmp_path, monkeypatch):
  path = None
  for settings in ({"log_sync_enabled": False, "log_sync_url": BASE_URL},
                   {"log_sync_enabled": True, "log_sync_url": log_sync.DEFAULT_WEB_UPLOAD_URL}):
    probe_root, _tlv_root, uploads = dsp_prepare(monkeypatch, tmp_path, settings)
    if path is None:
      path = write_probe_result(probe_root, mtime=old_timestamp())
    sync = log_sync.LogSync()
    assert asyncio.run(sync.run_once()) == log_sync.DISABLED_POLL_S
    assert uploads == []
    assert os.path.exists(path)
    assert not os.path.exists(path + log_sync.DSPPROBE_MARKER_SUFFIX)


def test_dspprobe_missing_root_is_ignored(tmp_path, monkeypatch):
  monkeypatch.setattr(log_sync, "DSPPROBE_ROOT", str(tmp_path / "does-not-exist"))
  monkeypatch.setattr(log_sync, "TLVREC_ROOT", str(tmp_path / "tlv-does-not-exist"))
  prepare(monkeypatch, tmp_path, ENABLED)
  sync = log_sync.LogSync()
  assert asyncio.run(sync.run_once()) == log_sync.POLL_S


def test_dspprobe_root_matches_probe_and_web_page():
  from openpilot.selfdrive.carrot.tlv import dsp_probe
  from openpilot.selfdrive.carrot.server.features import tlv_rec
  assert log_sync.DSPPROBE_ROOT == dsp_probe.PROBE_ROOT == tlv_rec.DSP_PROBE_ROOT
