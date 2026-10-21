"""Carrot Web endpoints for the shadow-mode TLV recorder.

Serves the per-session recordings written by
openpilot/selfdrive/carrot/tlv/recorder.py:

  GET    /api/tlv_rec/sessions              -> [{name, files, bytes, first, last}]
  GET    /api/tlv_rec/sessions/{name}.zip   -> streamed ZIP_STORED archive
  DELETE /api/tlv_rec/sessions/{name}       -> remove one session dir
  GET    /tlv_rec                           -> minimal Korean page (no assets)

The zip is never built in RAM or on disk: a producer thread writes the archive
into a ZIP sink that hands out fixed-size chunks, and the response streams them
to the client one by one. {name} is validated against ^\\d{8}-\\d{6}$ so a path
traversal cannot reach the filesystem.
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import shutil
import threading
import zipfile
from datetime import datetime

from aiohttp import web

REC_ROOT = "/data/media/0/carrot_tlv_rec"
DSP_PROBE_ROOT = "/data/media/0/carrot_tlv_rec/dsp_probe"
SESSION_NAME_RE = re.compile(r"^\d{8}-\d{6}$")
ZIP_CHUNK_BYTES = 256 * 1024
STREAM_QUEUE_MAX = 8


def resolve_session(name: str | None) -> str | None:
  """Absolute session dir for a validated name, else None (no traversal)."""
  if not name or not SESSION_NAME_RE.fullmatch(name):
    return None
  return os.path.join(REC_ROOT, name)


def _iso(mtime: float | None) -> str | None:
  if mtime is None:
    return None
  try:
    return datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds")
  except (OSError, OverflowError, ValueError):
    return None


def list_sessions(root: str | None = None) -> list[dict]:
  """Session summaries for the web page and API; read-only and best-effort."""
  root = root if root is not None else REC_ROOT
  try:
    names = sorted(os.listdir(root))
  except OSError:
    return []
  sessions = []
  for name in names:
    if not SESSION_NAME_RE.fullmatch(name):
      continue
    session_dir = os.path.join(root, name)
    if not os.path.isdir(session_dir):
      continue
    files = 0
    nbytes = 0
    first = None
    last = None
    for base, _dirs, fnames in os.walk(session_dir):
      for fname in fnames:
        try:
          st = os.stat(os.path.join(base, fname))
        except OSError:
          continue
        files += 1
        nbytes += int(st.st_size)
        mtime = float(st.st_mtime)
        first = mtime if first is None or mtime < first else first
        last = mtime if last is None or mtime > last else last
    sessions.append({"name": name, "files": files, "bytes": nbytes, "first": _iso(first), "last": _iso(last)})
  return sessions


def latest_dsp_probe(root: str | None = None) -> dict | None:
  """Newest dsp_probe result (openpilot/selfdrive/carrot/tlv/dsp_probe.py), best-effort."""
  root = root if root is not None else DSP_PROBE_ROOT
  try:
    names = [name for name in os.listdir(root) if name.endswith(".json")]
  except OSError:
    return None
  newest_path = None
  newest_mtime = -1.0
  for name in names:
    path = os.path.join(root, name)
    try:
      mtime = os.path.getmtime(path)
    except OSError:
      continue
    if mtime > newest_mtime:
      newest_path, newest_mtime = path, mtime
  if newest_path is None:
    return None
  try:
    with open(newest_path, encoding="utf-8") as f:
      payload = json.load(f)
  except (OSError, ValueError):
    return None
  if not isinstance(payload, dict):
    return None
  payload["file"] = os.path.basename(newest_path)
  return payload


def _probe_key() -> str:
  from openpilot.selfdrive.carrot.tlv import dsp_probe
  return dsp_probe.PROBE_KEY


def clear_dsp_probe_files(root: str | None = None) -> list[str]:
  """Delete the single probe run's files so it can run again (WSNPE2)."""
  root = root if root is not None else DSP_PROBE_ROOT
  key = _probe_key()
  removed = []
  for suffix in (".json", ".attempt", ".progress", ".steps.json"):
    path = os.path.join(root, f"{key}{suffix}")
    try:
      os.unlink(path)
      removed.append(os.path.basename(path))
    except OSError:
      pass
  bin_root = os.path.join(root, "bin", key)
  if os.path.isdir(bin_root):
    shutil.rmtree(bin_root, ignore_errors=True)
    removed.append(f"bin/{key}")
  return removed


class _ChunkSink:
  """Unseekable zipfile sink; drain() hands the buffered bytes out as chunks."""

  def __init__(self) -> None:
    self._buf = bytearray()

  def write(self, data) -> int:
    self._buf += data
    return len(data)

  def flush(self) -> None:
    pass

  def drain(self, chunk_size: int, final: bool = False):
    while len(self._buf) >= chunk_size:
      chunk = bytes(self._buf[:chunk_size])
      del self._buf[:chunk_size]
      yield chunk
    if final and self._buf:
      chunk = bytes(self._buf)
      self._buf.clear()
      yield chunk


def _iter_session_files(session_dir: str):
  for base, _dirs, fnames in os.walk(session_dir):
    for fname in sorted(fnames):
      path = os.path.join(base, fname)
      arcname = os.path.relpath(path, session_dir).replace(os.sep, "/")
      yield path, arcname


def iter_zip_chunks(session_dir: str, chunk_size: int = ZIP_CHUNK_BYTES):
  """One ZIP_STORED archive of the session dir, yielded as fixed-size chunks."""
  sink = _ChunkSink()
  with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED) as zf:
    for path, arcname in _iter_session_files(session_dir):
      zf.write(path, arcname)
      yield from sink.drain(chunk_size)
  yield from sink.drain(chunk_size, final=True)


async def api_tlv_rec_sessions(_request: web.Request) -> web.Response:
  sessions = await asyncio.to_thread(list_sessions)
  return web.json_response(sessions)


async def api_tlv_rec_dsp_probe(_request: web.Request) -> web.Response:
  probe = await asyncio.to_thread(latest_dsp_probe)
  return web.json_response({"ok": True, "probe": probe})


async def api_tlv_rec_dsp_probe_retry(_request: web.Request) -> web.Response:
  """User-triggered re-run: drop this commit's result/attempt/progress files."""
  removed = await asyncio.to_thread(clear_dsp_probe_files)
  return web.json_response({"ok": True, "removed": removed})


async def api_tlv_rec_download(request: web.Request) -> web.StreamResponse:
  name = request.match_info.get("name", "")
  session_dir = resolve_session(name)
  if session_dir is None:
    return web.json_response({"ok": False, "error": "invalid session name"}, status=400)
  if not os.path.isdir(session_dir):
    return web.json_response({"ok": False, "error": "no such session"}, status=404)

  response = web.StreamResponse(status=200, headers={
    "Content-Type": "application/zip",
    "Content-Disposition": f'attachment; filename="{name}.zip"',
    "Cache-Control": "no-store",
  })
  await response.prepare(request)
  loop = asyncio.get_running_loop()
  queue_: asyncio.Queue = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
  abort = threading.Event()

  def _produce() -> None:
    try:
      for chunk in iter_zip_chunks(session_dir):
        if abort.is_set():
          return
        asyncio.run_coroutine_threadsafe(queue_.put(chunk), loop).result()
    except Exception:
      pass
    finally:
      try:
        asyncio.run_coroutine_threadsafe(queue_.put(None), loop).result()
      except Exception:
        pass

  producer = loop.run_in_executor(None, _produce)
  try:
    while True:
      chunk = await queue_.get()
      if chunk is None:
        break
      await response.write(chunk)
    await response.write_eof()
  except ConnectionResetError:
    abort.set()
  finally:
    abort.set()
    # Unblock a producer stuck on a full queue so its thread can exit.
    while True:
      try:
        queue_.get_nowait()
      except asyncio.QueueEmpty:
        break
    if not producer.done():
      producer.cancel()
  return response


async def api_tlv_rec_delete(request: web.Request) -> web.Response:
  name = request.match_info.get("name", "")
  session_dir = resolve_session(name)
  if session_dir is None:
    return web.json_response({"ok": False, "error": "invalid session name"}, status=400)
  if not os.path.isdir(session_dir):
    return web.json_response({"ok": False, "error": "no such session"}, status=404)
  await asyncio.to_thread(shutil.rmtree, session_dir, True)
  return web.json_response({"ok": True})


_PAGE_CSS = """
body{font-family:sans-serif;background:#111;color:#eee;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 12px}
p.hint{color:#999;font-size:13px;margin:0 0 16px}
table{border-collapse:collapse;width:100%;max-width:860px}
th,td{border-bottom:1px solid #333;padding:6px 8px;font-size:14px;text-align:left}
th{color:#aaa;font-weight:normal}
a.btn,button.btn{display:inline-block;background:#2a6df4;color:#fff;border:0;border-radius:4px;
padding:5px 10px;font-size:13px;text-decoration:none;cursor:pointer;margin-right:6px}
button.del{background:#a33}
td.ok{color:#6f6}
td.bad{color:#f77}
tr.empty td{color:#888;text-align:center;padding:24px}
"""

_PAGE_JS = """
function retryDspProbe(){
  if(!confirm('DSP \uc2dc\ud5d8\uc744 \ub2e4\uc2dc \uc2dc\uc791\ud560\uae4c\uc694? \uae30\ub85d\uc744 \uc9c0\uc6c1\ub2c8\ub2e4.')) return;
  fetch('/api/tlv_rec/dsp_probe/retry', {method:'POST'})
    .then(function(){location.reload();});
}
document.querySelectorAll('button.del').forEach(function(btn){
  btn.addEventListener('click', function(){
    if(!confirm('\uc774 \uc138\uc158\uc744 \uc0ad\uc81c\ud560\uae4c\uc694?')) return;
    fetch('/api/tlv_rec/sessions/' + btn.dataset.name, {method:'DELETE'})
      .then(function(){location.reload();});
  });
});
"""


_PROBE_STEP_LABELS = {
  "presence": "SNPE \ud30c\uc77c/\ud234\uccb4\uc778",
  "build": "\ub7ec\ub108 \ube4c\ub4dc",
  "snpe_cpu": "SNPE CPU",
  "snpe_dsp": "SNPE DSP",
}
_PROBE_STATUS_LABELS = {"pass": "\ud1b5\uacfc", "fail": "\uc2e4\ud328", "timeout": "\uc2dc\uac04 \ucd08\uacfc", "skipped": "\uac74\ub108\ub700"}
_PROBE_STEP_ORDER = ("presence", "build", "snpe_cpu", "snpe_dsp")


def _probe_detail(step: dict) -> str:
  name = step.get("name")
  if step.get("status") != "pass":
    return html.escape(str(step.get("error") or ""))[:160]
  if name == "presence":
    clang = "ok" if step.get("clangxx") not in (None, "missing") else "missing"
    return f"clang++ {clang} / cdsprpc {len(step.get('cdsprpc') or [])}"
  if name == "build":
    return f"{step.get('duration_s', '?')} s"
  if name in ("snpe_cpu", "snpe_dsp"):
    bench = step.get("bench") or {}
    detail = (f"v{bench.get('snpe_version', '?')} / \uccab {bench.get('first_ms', '?')} ms / "
              f"\uc911\uc559\uac12 {bench.get('median_ms', '?')} ms / p90 {bench.get('p90_ms', '?')} ms")
    if name == "snpe_dsp":
      detail += f" / CPU \ub300\ube44 \ucd5c\ub300\ucc28\uc774 {step.get('max_abs_diff', '?')}"
    return detail
  return ""


_PROBE_HINT = ("\uc8fc\ud589 \uc911 \uc815\ucc28(\uc2e0\ud638 \ub300\uae30)\ud560 \ub54c\ub9c8\ub2e4 "
               "\uc790\ub3d9\uc73c\ub85c \uc870\uae08\uc529 \uc2dc\ud5d8\ud569\ub2c8\ub2e4. "
               "\ub530\ub85c \ud560 \uc77c\uc740 \uc5c6\uc2b5\ub2c8\ub2e4.")
_PROBE_RETRY_BUTTON = ('<button class="btn" onclick="retryDspProbe()">'
                       'DSP \uc2dc\ud5d8 \ub2e4\uc2dc \ud558\uae30</button>')


def _probe_section(probe: dict | None) -> str:
  section = ("<h2>DSP \uc2dc\ud5d8 \uacb0\uacfc</h2>"
             f'<p class="hint">{_PROBE_HINT}</p>')
  if not probe:
    return (section + '<p class="hint">\uc544\uc9c1 \uacb0\uacfc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.</p>'
            + f"<p>{_PROBE_RETRY_BUTTON}</p>")
  steps = probe.get("steps") or {}
  rows = []
  for name in _PROBE_STEP_ORDER:
    step = steps.get(name) or {"status": "skipped"}
    status = _PROBE_STATUS_LABELS.get(str(step.get("status")), str(step.get("status", "?")))
    css = "ok" if step.get("status") == "pass" else "bad"
    detail = _probe_detail(step)
    rows.append(f'<tr><td>{_PROBE_STEP_LABELS[name]}</td><td class="{css}">{status}</td><td>{detail}</td></tr>')
  overall = "\ud1b5\uacfc" if probe.get("ok") else "\uc2e4\ud328"
  meta = f"commit {html.escape(str(probe.get('commit', '?')))} / {html.escape(str(probe.get('timestamp', '?')))} / \uc885\ud569 {overall}"
  return (section + f'<p class="hint">{meta}</p>'
          + "<table><thead><tr><th>\ub2e8\uacc4</th><th>\uacb0\uacfc</th><th>\uc2dc\uac04/\uc624\ub958</th></tr></thead><tbody>"
          + "".join(rows) + "</tbody></table>"
          + f"<p>{_PROBE_RETRY_BUTTON}</p>")


def _page_html(sessions: list[dict], probe: dict | None = None) -> str:
  rows = []
  for session in sessions:
    size_mb = session["bytes"] / (1024 * 1024)
    rows.append(
      "<tr>"
      f'<td>{session["name"]}</td>'
      f'<td>{size_mb:.1f} MB</td>'
      f'<td>{session["files"]}</td>'
      f'<td>{session["first"] or "-"}</td>'
      f'<td>{session["last"] or "-"}</td>'
      f'<td><a class="btn" href="/api/tlv_rec/sessions/{session["name"]}.zip">\ub2e4\uc6b4\ub85c\ub4dc</a>'
      f'<button class="btn del" data-name="{session["name"]}">\uc0ad\uc81c</button></td>'
      "</tr>"
    )
  if not rows:
    rows.append('<tr class="empty"><td colspan="6">\uae30\ub85d\uc774 \uc5c6\uc2b5\ub2c8\ub2e4</td></tr>')
  return (
    "<!doctype html>\n"
    '<html lang="ko"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    "<title>\uc2e0\ud638\ub4f1 \uc778\uc2dd \uae30\ub85d</title>"
    f"<style>{_PAGE_CSS}</style></head><body>"
    "<h1>\uc2e0\ud638\ub4f1 \uc778\uc2dd \uae30\ub85d</h1>"
    '<p class="hint">\uc2e0\ud638\uac10\uc9c0 \uae30\ub2a5\uc774 \ucf1c\uc838 \uc788\uc744 \ub54c \uc2e0\ud638\ub4f1 \ube44\uc804 \uc778\uc2dd\uc774 \uc800\uc7a5\ud55c \ub370\uc774\ud130\uc785\ub2c8\ub2e4. '
    "\uc138\uc158\uc744 \ub2e4\uc6b4\ub85c\ub4dc\ud574 PC\uc5d0\uc11c \ud559\uc2b5 \ub370\uc774\ud130\uc14b\uc73c\ub85c \ubcc0\ud658\ud558\uc138\uc694.</p>"
    "<table><thead><tr><th>\uc138\uc158</th><th>\ud06c\uae30</th><th>\ud30c\uc77c</th>"
    "<th>\ucc98\uc74c</th><th>\ub9c8\uc9c0\ub9c9</th><th></th></tr></thead><tbody>"
    + "".join(rows) +
    f"</tbody></table>{_probe_section(probe)}<script>{_PAGE_JS}</script></body></html>"
  )


async def tlv_rec_page(_request: web.Request) -> web.Response:
  sessions = await asyncio.to_thread(list_sessions)
  probe = await asyncio.to_thread(latest_dsp_probe)
  return web.Response(text=_page_html(sessions, probe), content_type="text/html", charset="utf-8")


def register(app: web.Application) -> None:
  app.router.add_get("/api/tlv_rec/sessions", api_tlv_rec_sessions)
  app.router.add_get("/api/tlv_rec/sessions/{name}.zip", api_tlv_rec_download)
  app.router.add_delete("/api/tlv_rec/sessions/{name}", api_tlv_rec_delete)
  app.router.add_get("/api/tlv_rec/dsp_probe", api_tlv_rec_dsp_probe)
  app.router.add_post("/api/tlv_rec/dsp_probe/retry", api_tlv_rec_dsp_probe_retry)
  app.router.add_get("/tlv_rec", tlv_rec_page)
