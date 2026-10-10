#!/usr/bin/env python3
"""carrotTlv bench: 60 s baseline (tlvd off) then 60 s shadow (tlvd on).

Paste into the Carrot web terminal on the C3X:

  python3 openpilot/tools/carrot_tlv_bench.py --backend null
  python3 openpilot/tools/carrot_tlv_bench.py --backend onnx --model-path /data/models/tlv/model.onnx

Collects modelV2 executionTime/frameDropPerc, UI draw times (uiDebug, if
published), camerad roadCameraState timings, deviceState temps/thermalStatus and
tlvd inferMs/procMs/droppedFrames, then prints ONE JSON summary with pass/fail
per "no difference beyond noise" gate. Restores the original CarrotTlvMode (and
backend/model path) on exit, including Ctrl+C.

The gate evaluation (`summarize`) is a pure function, unit-tested on a PC; the
openpilot messaging imports happen lazily so this file stays importable anywhere.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time

import numpy as np

FRAME_DROP_DELTA = 1.0      # allowed frameDropPerc mean/p95 increase
EXEC_P95_RATIO = 1.10       # allowed model execution p95 increase
UI_P95_RATIO = 1.10
UI_P95_ABS_SLACK_MS = 1.0
CAM_PT_RATIO = 1.20
CAM_SOF_SLACK_MS = 5.0
TEMP_SLACK_C = 3.0
TLV_INFER_P95_BUDGET_MS = 100.0   # 10 Hz service budget
TLV_PROC_P95_BUDGET_MS = 150.0


def _p95(values: list[float]) -> float:
  return float(np.percentile(values, 95.0)) if values else 0.0


def _mean(values: list[float]) -> float:
  return float(np.mean(values)) if values else 0.0


def _max(values: list[float]) -> float:
  return float(np.max(values)) if values else 0.0


def summarize(baseline: dict, tlv: dict) -> dict:
  """Pure gate evaluation over two collection reports."""
  base_fd, tlv_fd = baseline.get("frame_drop", []), tlv.get("frame_drop", [])
  base_exec, tlv_exec = baseline.get("model_exec_ms", []), tlv.get("model_exec_ms", [])

  fd_pass = (
    _mean(tlv_fd) <= _mean(base_fd) + FRAME_DROP_DELTA
    and _p95(tlv_fd) <= _p95(base_fd) + FRAME_DROP_DELTA
  ) if base_fd and tlv_fd else False

  exec_pass = (_p95(tlv_exec) <= _p95(base_exec) * EXEC_P95_RATIO) if base_exec and tlv_exec else False

  ui_base, ui_tlv = baseline.get("ui_draw_ms", []), tlv.get("ui_draw_ms", [])
  if ui_base and ui_tlv:
    ui_pass = _p95(ui_tlv) <= _p95(ui_base) * UI_P95_RATIO + UI_P95_ABS_SLACK_MS
    ui_available = True
  else:
    ui_pass, ui_available = True, False

  cam_base_pt, cam_tlv_pt = baseline.get("cam_processing_ms", []), tlv.get("cam_processing_ms", [])
  cam_base_sof, cam_tlv_sof = baseline.get("cam_sof_gap_ms", []), tlv.get("cam_sof_gap_ms", [])
  if cam_base_pt and cam_tlv_pt and cam_base_sof and cam_tlv_sof:
    cam_pass = (
      _p95(cam_tlv_pt) <= _p95(cam_base_pt) * CAM_PT_RATIO
      and _max(cam_tlv_sof) <= _max(cam_base_sof) * CAM_PT_RATIO + CAM_SOF_SLACK_MS
    )
    cam_available = True
  else:
    cam_pass, cam_available = True, False

  base_thermal = baseline.get("thermal_status", [])
  tlv_thermal = tlv.get("thermal_status", [])
  base_temp = baseline.get("max_temp_c", [])
  tlv_temp = tlv.get("max_temp_c", [])
  if base_thermal and tlv_thermal and base_temp and tlv_temp:
    thermal_pass = _max(tlv_thermal) <= _max(base_thermal) and _max(tlv_temp) <= _max(base_temp) + TEMP_SLACK_C
    thermal_available = True
  else:
    thermal_pass, thermal_available = False, False

  infer = tlv.get("tlv_infer_ms", [])
  proc = tlv.get("tlv_proc_ms", [])
  if infer and proc:
    tlv_budget_pass = _p95(infer) <= TLV_INFER_P95_BUDGET_MS and _p95(proc) <= TLV_PROC_P95_BUDGET_MS
  else:
    tlv_budget_pass = False

  gates = {
    "model_frame_drop_unchanged": {"pass": bool(fd_pass), "available": True},
    "model_exec_p95_unchanged": {"pass": bool(exec_pass), "available": True},
    "ui_frame_time_unchanged": {"pass": bool(ui_pass), "available": ui_available},
    "camerad_timing_unchanged": {"pass": bool(cam_pass), "available": cam_available},
    "thermal_no_regression": {"pass": bool(thermal_pass), "available": thermal_available},
    "tlvd_within_budget": {"pass": bool(tlv_budget_pass), "available": bool(infer and proc)},
  }
  verdict = all(gate["pass"] for gate in gates.values())
  return {
    "verdict": "pass" if verdict else "fail",
    "gates": gates,
    "baseline": {
      "frame_drop_mean": _mean(base_fd), "frame_drop_p95": _p95(base_fd),
      "model_exec_p95_ms": _p95(base_exec), "ui_draw_p95_ms": _p95(ui_base),
      "cam_processing_p95_ms": _p95(cam_base_pt), "cam_sof_gap_max_ms": _max(cam_base_sof),
      "thermal_status_max": int(_max(base_thermal)) if base_thermal else None,
      "max_temp_c_max": _max(base_temp) if base_temp else None,
    },
    "tlv": {
      "frame_drop_mean": _mean(tlv_fd), "frame_drop_p95": _p95(tlv_fd),
      "model_exec_p95_ms": _p95(tlv_exec), "ui_draw_p95_ms": _p95(ui_tlv),
      "cam_processing_p95_ms": _p95(cam_tlv_pt), "cam_sof_gap_max_ms": _max(cam_tlv_sof),
      "thermal_status_max": int(_max(tlv_thermal)) if tlv_thermal else None,
      "max_temp_c_max": _max(tlv_temp) if tlv_temp else None,
      "infer_p95_ms": _p95(infer), "proc_p95_ms": _p95(proc),
      "infer_msgs": len(infer), "dropped_frames_final": tlv.get("tlv_dropped_final"),
      "status_seen": tlv.get("tlv_status_seen", []),
    },
    "baseline_clean": not baseline.get("carrot_tlv_seen", 0),
  }


def _collect(sm, seconds: float, rate_hz: float = 20.0) -> dict:
  """Collect one phase; openpilot services must already be wired by the caller."""
  from openpilot.common.realtime import Ratekeeper
  report: dict = {
    "frame_drop": [], "model_exec_ms": [], "ui_draw_ms": [], "cam_processing_ms": [],
    "cam_sof_gap_ms": [], "thermal_status": [], "max_temp_c": [],
    "tlv_infer_ms": [], "tlv_proc_ms": [], "carrot_tlv_seen": 0,
    "tlv_dropped_final": None, "tlv_status_seen": [],
  }
  rk = Ratekeeper(rate_hz, print_delay_threshold=None)
  last_sof = None
  end = time.monotonic() + seconds
  while time.monotonic() < end:
    sm.update(0)
    if sm.updated.get("modelV2") and sm.valid["modelV2"]:
      model = sm["modelV2"]
      report["frame_drop"].append(float(model.frameDropPerc))
      if len(model.modelExecutionTime) > 0:
        report["model_exec_ms"].append(1000.0 * max(float(v) for v in model.modelExecutionTime))
    if sm.seen.get("uiDebug") and sm.updated.get("uiDebug") and sm.valid["uiDebug"]:
      report["ui_draw_ms"].append(float(sm["uiDebug"].drawTimeMillis))
    if sm.seen.get("roadCameraState") and sm.updated.get("roadCameraState") and sm.valid["roadCameraState"]:
      cam = sm["roadCameraState"]
      report["cam_processing_ms"].append(float(cam.processingTime))
      if last_sof is not None and cam.timestampSof > last_sof:
        report["cam_sof_gap_ms"].append((cam.timestampSof - last_sof) / 1e6)
      last_sof = cam.timestampSof
    if sm.seen.get("deviceState") and sm.updated.get("deviceState") and sm.valid["deviceState"]:
      state = sm["deviceState"]
      report["thermal_status"].append(float(state.thermalStatus.raw))
      if state.maxTempC > 0.0:
        report["max_temp_c"].append(float(state.maxTempC))
    if sm.seen.get("carrotTlv") and sm.updated.get("carrotTlv") and sm.valid["carrotTlv"]:
      report["carrot_tlv_seen"] += 1
      tlv = sm["carrotTlv"]
      report["tlv_infer_ms"].append(float(tlv.inferMs))
      report["tlv_proc_ms"].append(float(tlv.procMs))
      report["tlv_dropped_final"] = int(tlv.droppedFrames)
      status = str(tlv.status)
      if status not in report["tlv_status_seen"]:
        report["tlv_status_seen"].append(status)
    rk.keep_time()
  return report


def _param_bytes(params, key: str):
  try:
    value = params.get(key)
  except Exception:
    return None
  return value


def _put_raw(params, key: str, value) -> None:
  try:
    params.put(key, value)
  except Exception as exc:
    print(f"carrot_tlv_bench: failed to write {key}: {exc!r}", file=sys.stderr)


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description="carrotTlv A/B bench (baseline vs shadow)")
  parser.add_argument("--baseline-s", type=float, default=60.0)
  parser.add_argument("--tlv-s", type=float, default=60.0)
  parser.add_argument("--backend", default="null", help="null/onnx (tlvd CarrotTlvBackend)")
  parser.add_argument("--model-path", default=None, help="CarrotTlvModelPath override")
  parser.add_argument("--startup-wait-s", type=float, default=20.0)
  args = parser.parse_args(argv)

  from openpilot.common.params import Params
  import openpilot.cereal.messaging as messaging

  params = Params()
  original_mode = _param_bytes(params, "CarrotTlvMode")
  original_backend = _param_bytes(params, "CarrotTlvBackend")
  original_model_path = _param_bytes(params, "CarrotTlvModelPath")

  def restore():
    for key, value in (("CarrotTlvMode", original_mode), ("CarrotTlvBackend", original_backend),
                       ("CarrotTlvModelPath", original_model_path)):
      if value is not None:
        _put_raw(params, key, value)
    print("carrot_tlv_bench: restored original CarrotTlv params", file=sys.stderr)

  def on_signal(signum, frame):
    raise KeyboardInterrupt

  signal.signal(signal.SIGINT, on_signal)
  if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, on_signal)

  sm = messaging.SubMaster(["modelV2", "deviceState", "roadCameraState", "uiDebug", "carrotTlv"],
                           ignore_avg_freq=["uiDebug", "carrotTlv"])
  try:
    print("carrot_tlv_bench: baseline phase (tlvd off) ...", file=sys.stderr)
    _put_raw(params, "CarrotTlvMode", b"0")
    time.sleep(3.0)
    baseline = _collect(sm, args.baseline_s)

    print("carrot_tlv_bench: shadow phase (tlvd on) ...", file=sys.stderr)
    _put_raw(params, "CarrotTlvBackend", args.backend.encode())
    if args.model_path is not None:
      _put_raw(params, "CarrotTlvModelPath", args.model_path.encode())
    _put_raw(params, "CarrotTlvMode", b"1")
    deadline = time.monotonic() + args.startup_wait_s
    while time.monotonic() < deadline and not sm.seen.get("carrotTlv"):
      sm.update(100)
    tlv = _collect(sm, args.tlv_s)
  except KeyboardInterrupt:
    print("carrot_tlv_bench: interrupted", file=sys.stderr)
    restore()
    return 130

  restore()
  summary = summarize(baseline, tlv)
  summary["parameters"] = {
    "baseline_s": args.baseline_s, "tlv_s": args.tlv_s, "backend": args.backend,
    "model_path": args.model_path, "original_mode": None if original_mode is None else original_mode.decode("utf-8", "replace"),
  }
  print(json.dumps(summary, indent=2))
  return 0 if summary["verdict"] == "pass" else 1


if __name__ == "__main__":
  sys.exit(main())
