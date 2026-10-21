"""Tests for the tlv_rec Carrot Web feature (server/features/tlv_rec.py).

Loaded via importlib like test_stream.py so the whole features package (device
routes) is never imported. Covers the session list, the streamed ZIP round-trip,
delete, traversal rejection and the Korean page.
"""
from __future__ import annotations

import asyncio
import importlib.util
import io
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
