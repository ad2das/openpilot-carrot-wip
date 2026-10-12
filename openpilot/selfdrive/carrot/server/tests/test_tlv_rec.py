"""Tests for the tlv_rec Carrot Web feature (server/features/tlv_rec.py).

Loaded via importlib like test_stream.py so the whole features package (device
routes) is never imported. Covers the session list, the streamed ZIP round-trip,
delete, traversal rejection and the Korean page.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def _load_module(root) -> object:
  name = "openpilot.selfdrive.carrot.server.features.tlv_rec_under_test"
  spec = importlib.util.spec_from_file_location(
    name, Path(__file__).resolve().parents[1] / "features" / "tlv_rec.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  module.REC_ROOT = str(root)
  return module


def _make_session(root, name: str = "20260101-120000"):
  session_dir = root / name
  session_dir.mkdir(parents=True)
  jpeg = b"\xff\xd8jpeg-bytes" * 500
  (session_dir / "0000001.jpg").write_bytes(jpeg)
  (session_dir / "index.jsonl").write_text('{"file":"0000001.jpg"}\n', encoding="utf-8")
  return session_dir, jpeg


async def _start_client(module) -> TestClient:
  app = web.Application()
  module.register(app)
  client = TestClient(TestServer(app))
  await client.start_server()
  return client


def test_sessions_listing_and_zip_roundtrip(tmp_path):
  module = _load_module(tmp_path)
  session_dir, jpeg = _make_session(tmp_path)

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.get("/api/tlv_rec/sessions")
      assert resp.status == 200
      data = await resp.json()
      assert isinstance(data, list) and len(data) == 1
      entry = data[0]
      assert entry["name"] == "20260101-120000"
      assert entry["files"] == 2
      assert entry["bytes"] == sum(os.path.getsize(p) for p in session_dir.iterdir())
      assert entry["first"] and entry["last"]

      resp = await client.get("/api/tlv_rec/sessions/20260101-120000.zip")
      assert resp.status == 200
      assert resp.headers["Content-Type"] == "application/zip"
      assert "no-store" in resp.headers.get("Cache-Control", "")
      assert "Content-Length" not in resp.headers  # streamed, never buffered
      body = await resp.read()
      archive = zipfile.ZipFile(io.BytesIO(body))
      assert sorted(archive.namelist()) == ["0000001.jpg", "index.jsonl"]
      assert archive.read("0000001.jpg") == jpeg
    finally:
      await client.close()

  asyncio.run(scenario())


def test_zip_is_streamed_in_multiple_chunks(tmp_path):
  module = _load_module(tmp_path)
  session_dir, _jpeg = _make_session(tmp_path)
  chunks = list(module.iter_zip_chunks(str(session_dir), chunk_size=1024))
  assert len(chunks) > 5
  archive = zipfile.ZipFile(io.BytesIO(b"".join(chunks)))
  assert sorted(archive.namelist()) == ["0000001.jpg", "index.jsonl"]


def test_delete_removes_session(tmp_path):
  module = _load_module(tmp_path)
  session_dir, _jpeg = _make_session(tmp_path)

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.delete("/api/tlv_rec/sessions/20260101-120000")
      assert resp.status == 200
      assert (await resp.json())["ok"] is True
      assert not session_dir.exists()
      resp = await client.delete("/api/tlv_rec/sessions/20260101-120000")
      assert resp.status == 404
      resp = await client.get("/api/tlv_rec/sessions")
      assert await resp.json() == []
    finally:
      await client.close()

  asyncio.run(scenario())


def test_download_missing_session_404(tmp_path):
  module = _load_module(tmp_path)

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.get("/api/tlv_rec/sessions/20260101-120000.zip")
      assert resp.status == 404
    finally:
      await client.close()

  asyncio.run(scenario())


def test_traversal_rejected(tmp_path):
  module = _load_module(tmp_path)
  assert module.resolve_session("20260101-120000") == os.path.join(str(tmp_path), "20260101-120000")
  for bad in ("..", "../etc", "..%2F..", "20260101-120000/..", "../../secret", "2026-01-01", "x.zip", ""):
    assert module.resolve_session(bad) is None, bad

  # handler level: the invalid name is rejected before touching the filesystem
  request = SimpleNamespace(match_info={"name": "../secret"})
  assert asyncio.run(module.api_tlv_rec_delete(request)).status == 400
  assert asyncio.run(module.api_tlv_rec_download(request)).status == 400

  outside = tmp_path.parent / "outside.txt"
  outside.write_text("secret", encoding="utf-8")

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.get("/api/tlv_rec/sessions/..%2f..%2foutside.txt.zip")
      assert resp.status in (400, 404)
      resp = await client.delete("/api/tlv_rec/sessions/..%2f..%2foutside.txt")
      assert resp.status in (400, 404)
      assert outside.exists()
    finally:
      await client.close()

  asyncio.run(scenario())


def test_page_is_korean_and_self_contained(tmp_path):
  module = _load_module(tmp_path)
  _make_session(tmp_path)

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.get("/tlv_rec")
      assert resp.status == 200
      assert resp.content_type == "text/html"
      html = await resp.text()
      assert "신호등 인식 기록" in html
      assert "20260101-120000" in html
      assert "/api/tlv_rec/sessions/20260101-120000.zip" in html
      assert "http://" not in html and "https://" not in html
    finally:
      await client.close()

  asyncio.run(scenario())


def test_dsp_probe_api_and_page_without_result(tmp_path, monkeypatch):
  module = _load_module(tmp_path)
  monkeypatch.setattr(module, "DSP_PROBE_ROOT", str(tmp_path / "dsp_probe"))

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.get("/api/tlv_rec/dsp_probe")
      assert resp.status == 200
      assert await resp.json() == {"ok": True, "probe": None}
      page = await (await client.get("/tlv_rec")).text()
      assert "DSP 시험 결과" in page
      assert "아직 결과가 없습니다" in page
      assert "DSP 시험 다시 하기" in page
      assert "주차(P) 상태로 시동을 켠 채 약 3분 기다리면 자동으로 시험합니다. 기어를 바꾸면 즉시 중단됩니다." in page
    finally:
      await client.close()

  asyncio.run(scenario())


def test_dsp_probe_api_and_page_with_result(tmp_path, monkeypatch):
  module = _load_module(tmp_path)
  probe_root = tmp_path / "dsp_probe"
  probe_root.mkdir()
  payload = {
    "commit": "abc1234",
    "timestamp": "2026-10-13T08:00:00+09:00",
    "device_type": "tici",
    "agnos_version": "19.8",
    "ok": False,
    "error": None,
    "steps": {
      "presence": {"name": "presence", "status": "pass", "clang": {"status": "pass"}},
      "smoke": {"name": "smoke", "status": "pass", "first_ms": 12.0, "steady_median_ms": 3.0},
      "detector": {"name": "detector", "status": "fail", "error": "DSP op unsupported"},
    },
  }
  (probe_root / "abc1234.json").write_text(json.dumps(payload), encoding="utf-8")
  monkeypatch.setattr(module, "DSP_PROBE_ROOT", str(probe_root))

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.get("/api/tlv_rec/dsp_probe")
      data = await resp.json()
      assert data["ok"] is True
      assert data["probe"]["commit"] == "abc1234"
      assert data["probe"]["file"] == "abc1234.json"
      assert data["probe"]["steps"]["detector"]["status"] == "fail"
      page = await (await client.get("/tlv_rec")).text()
      assert "DSP 시험 결과" in page
      assert "통과" in page
      assert "실패" in page
      assert "DSP op unsupported" in page
      assert "3.0" in page
      assert "DSP 시험 다시 하기" in page
      assert "주차(P) 상태로 시동을 켠 채 약 3분 기다리면 자동으로 시험합니다. 기어를 바꾸면 즉시 중단됩니다." in page
    finally:
      await client.close()

  asyncio.run(scenario())


def test_dsp_probe_retry_clears_this_commit_only(tmp_path, monkeypatch):
  module = _load_module(tmp_path)
  probe_root = tmp_path / "dsp_probe"
  probe_root.mkdir()
  for suffix in (".json", ".attempt", ".progress"):
    (probe_root / f"abc1234{suffix}").write_text("{}", encoding="utf-8")
  (probe_root / "other.json").write_text("{}", encoding="utf-8")
  monkeypatch.setattr(module, "DSP_PROBE_ROOT", str(probe_root))
  monkeypatch.setattr(module, "_probe_commit", lambda: "abc1234")

  async def scenario():
    client = await _start_client(module)
    try:
      resp = await client.post("/api/tlv_rec/dsp_probe/retry")
      assert resp.status == 200
      data = await resp.json()
      assert data["ok"] is True
      assert sorted(data["removed"]) == ["abc1234.attempt", "abc1234.json", "abc1234.progress"]
      assert not (probe_root / "abc1234.json").exists()
      assert not (probe_root / "abc1234.attempt").exists()
      assert not (probe_root / "abc1234.progress").exists()
      assert (probe_root / "other.json").exists()
      # idempotent: a second retry finds nothing left to remove
      resp = await client.post("/api/tlv_rec/dsp_probe/retry")
      assert (await resp.json())["removed"] == []
    finally:
      await client.close()

  asyncio.run(scenario())
