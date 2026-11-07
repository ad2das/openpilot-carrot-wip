import asyncio
import json
import os
import shutil
import subprocess
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from openpilot.selfdrive.carrot.server.features.dashcam import ffmpeg, routes


def make_app():
  app = web.Application()
  routes.register(app)
  return app


def make_client(monkeypatch, tmp_path, segment="route--0", files=None):
  (tmp_path / segment).mkdir(exist_ok=True)
  for name, payload in (files or {}).items():
    (tmp_path / segment / name).write_bytes(payload)
  (tmp_path / "dl").mkdir(exist_ok=True)
  monkeypatch.setattr(routes, "segment_dir", lambda s: str(tmp_path / s))

  counter = {"n": 0}

  def fake_tmp(ext):
    counter["n"] += 1
    return str(tmp_path / "dl" / f"{counter['n']}{ext}")

  monkeypatch.setattr(routes, "download_tmp_path", fake_tmp)
  return counter


async def with_client(monkeypatch, tmp_path, files, body):
  make_client(monkeypatch, tmp_path, files=files)
  async with TestClient(TestServer(make_app())) as client:
    return await body(client)


def test_files_endpoint_lists_known_artifacts(monkeypatch, tmp_path):
  files = {"qcamera.ts": b"0123456789", "fcamera.hevc": b"h" * 200, "rlog.zst": b"r" * 20}

  async def body(client):
    response = await client.get("/api/dashcam/files/route--0")
    assert response.status == 200
    return json.loads(await response.text())

  payload = asyncio.run(with_client(monkeypatch, tmp_path, files, body))
  assert payload["files"] == [
    {"kind": "qcamera", "name": "qcamera.ts", "size": 10},
    {"kind": "fcamera", "name": "fcamera.hevc", "size": 200},
    {"kind": "rlog", "name": "rlog.zst", "size": 20},
  ]


def test_camera_download_serves_remuxed_mp4_and_cleans_tmp(monkeypatch, tmp_path):
  files = {"fcamera.hevc": b"RAWHEVC"}

  async def body(client):
    def stub(source, out):
      with open(out, "wb") as f:
        f.write(b"MP4DATA")
      return True

    monkeypatch.setattr(routes, "remux_camera_mp4", stub)
    response = await client.get("/api/dashcam/download/route--0/fcamera")
    assert response.status == 200
    assert response.headers["Content-Type"].startswith("video/mp4")
    assert "route--0--fcamera.mp4" in response.headers["Content-Disposition"]
    return await response.read()

  assert asyncio.run(with_client(monkeypatch, tmp_path, files, body)) == b"MP4DATA"
  assert os.listdir(tmp_path / "dl") == []


def test_camera_download_falls_back_to_raw_hevc(monkeypatch, tmp_path):
  files = {"fcamera.hevc": b"RAWHEVC"}

  async def body(client):
    monkeypatch.setattr(routes, "remux_camera_mp4", lambda source, out: False)
    response = await client.get("/api/dashcam/download/route--0/fcamera")
    assert response.status == 200
    assert "route--0--fcamera.hevc" in response.headers["Content-Disposition"]
    return await response.read()

  assert asyncio.run(with_client(monkeypatch, tmp_path, files, body)) == b"RAWHEVC"
  assert os.listdir(tmp_path / "dl") == []


def test_download_keeps_qcamera_and_rlog_behaviour(monkeypatch, tmp_path):
  files = {"qcamera.ts": b"TS", "rlog.zst": b"RLOG"}

  async def body(client):
    rlog = await client.get("/api/dashcam/download/route--0/rlog")
    assert rlog.status == 200
    assert "route--0--rlog.zst" in rlog.headers["Content-Disposition"]
    assert await rlog.read() == b"RLOG"
    camera = await client.get("/api/dashcam/download/route--0/qcamera")
    assert camera.status == 200
    assert "route--0--qcamera.ts" in camera.headers["Content-Disposition"]
    return await camera.read()

  assert asyncio.run(with_client(monkeypatch, tmp_path, files, body)) == b"TS"


def test_download_rejects_unknown_and_missing_kinds(monkeypatch, tmp_path):
  files = {"fcamera.hevc": b"RAWHEVC"}

  async def body(client):
    unknown = await client.get("/api/dashcam/download/route--0/foo")
    assert unknown.status == 404
    absent = await client.get("/api/dashcam/download/route--0/ecamera")
    assert absent.status == 404

  asyncio.run(with_client(monkeypatch, tmp_path, files, body))


def test_upload_routes_are_unregistered_but_test_route_stays():
  app = make_app()
  paths = [r.resource.canonical for r in app.router.routes() if getattr(r, "resource", None)]
  assert "/api/dashcam/files/{segment}" in paths
  assert "/api/dashcam/download/{segment}/{kind}" in paths
  assert "/api/dashcam/upload/test" in paths
  for removed in (
    "/api/dashcam/upload",
    "/api/dashcam/upload/start",
    "/api/dashcam/upload/summary",
    "/api/dashcam/upload/job",
    "/api/dashcam/upload/cancel",
  ):
    assert removed not in paths


def test_download_tmp_path_prunes_stale_files(monkeypatch, tmp_path):
  monkeypatch.setattr(ffmpeg, "DASHCAM_CACHE_DIR", str(tmp_path))
  directory = tmp_path / "download"
  directory.mkdir()
  stale = directory / "stale.mp4"
  fresh = directory / "fresh.mp4"
  stale.write_bytes(b"x")
  fresh.write_bytes(b"y")
  now = time.time()
  os.utime(stale, (now - 7200, now - 7200))
  os.utime(fresh, (now - 60, now - 60))

  path = ffmpeg.download_tmp_path(".mp4")

  assert not stale.exists()
  assert fresh.exists()
  assert path.endswith(".mp4")
  assert os.path.dirname(path) == str(directory)


def _ffmpeg_candidates() -> list[str]:
  """Every distinct ffmpeg on PATH plus the usual system locations; a venv shim may lack encoders."""
  seen = []
  for entry in (os.environ.get("PATH") or "").split(os.pathsep):
    candidate = os.path.join(entry, "ffmpeg")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK) and candidate not in seen:
      seen.append(candidate)
  for candidate in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
    if os.path.isfile(candidate) and candidate not in seen:
      seen.append(candidate)
  return seen


def _ffmpeg_with_libx265() -> str | None:
  for candidate in _ffmpeg_candidates():
    try:
      result = subprocess.run([candidate, "-hide_banner", "-encoders"], capture_output=True, text=True)
    except OSError:
      continue
    if "libx265" in (result.stdout or ""):
      return candidate
  return None


def test_real_ffmpeg_remux_produces_hvc1_mp4(tmp_path):
  encoder = _ffmpeg_with_libx265()
  if not encoder:
    pytest.skip("no ffmpeg with libx265 on this host")
  if not shutil.which("ffprobe"):
    pytest.fail("ffprobe is missing; the remux result cannot be verified")
  source = str(tmp_path / "src.hevc")
  out = str(tmp_path / "out.mp4")
  made = subprocess.run(
    [encoder, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=20",
     "-frames:v", "41", "-c:v", "libx265", "-f", "hevc", source],
    capture_output=True, text=True,
  )
  assert made.returncode == 0, made.stderr

  assert ffmpeg.remux_camera_mp4(source, out) is True

  probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=codec_name,codec_tag_string,avg_frame_rate",
     "-show_entries", "format=duration", "-of", "json", out],
    capture_output=True, text=True,
  )
  assert probe.returncode == 0, probe.stderr
  info = json.loads(probe.stdout)
  stream = info["streams"][0]
  for key in ("codec_name", "codec_tag_string", "avg_frame_rate"):
    assert stream.get(key), f"ffprobe reported no {key}"
  assert info["format"].get("duration"), "ffprobe reported no duration"
  assert stream["codec_name"] == "hevc"
  assert stream["codec_tag_string"] == "hvc1"
  assert stream["avg_frame_rate"] == "20/1"
  assert abs(float(info["format"]["duration"]) - 2.0) <= 0.1


def test_remux_returns_false_without_ffmpeg(monkeypatch, tmp_path):
  monkeypatch.setattr(ffmpeg.shutil, "which", lambda name: None)
  assert ffmpeg.remux_camera_mp4(str(tmp_path / "in.hevc"), str(tmp_path / "out.mp4")) is False
