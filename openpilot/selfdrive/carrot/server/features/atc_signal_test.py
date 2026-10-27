import asyncio
import json
import time

from aiohttp import web

from ..services.params import HAS_PARAMS, Params


# card runs the check only while engaged, stopped in D with no driver signal
# input; Panda still gates every frame. The nonce is CLOCK_MONOTONIC, shared
# with card on the device, so an old request cannot fire a later drive.
def _start() -> dict:
  params = Params()
  if not params.get_bool("AutoTurnSignal"):
    return {"ok": False, "error": "disabled"}
  request = time.monotonic_ns()
  params.put("AtcSignalTestRequest", request)
  return {"ok": True, "request": request}


def _result() -> dict:
  raw = Params().get("AtcSignalTestResult")
  try:
    result = json.loads(raw) if raw else None
  except (TypeError, ValueError):
    result = None
  return {"ok": True, "result": result if isinstance(result, dict) else None}


async def api_atc_signal_test_start(_request: web.Request) -> web.Response:
  if not HAS_PARAMS:
    return web.json_response({"ok": False, "error": "params unavailable"}, status=500)
  payload = await asyncio.to_thread(_start)
  return web.json_response(payload, status=200 if payload["ok"] else 409)


async def api_atc_signal_test_result(_request: web.Request) -> web.Response:
  if not HAS_PARAMS:
    return web.json_response({"ok": False, "error": "params unavailable"}, status=500)
  return web.json_response(await asyncio.to_thread(_result))


def register(app: web.Application) -> None:
  app.router.add_post("/api/atc_signal_test", api_atc_signal_test_start)
  app.router.add_get("/api/atc_signal_test", api_atc_signal_test_result)
