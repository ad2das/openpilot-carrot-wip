#!/usr/bin/env python3
"""On-device traffic-light detector process (carrotTlv service).

Pipeline per processed frame:
  VisionIPC road NV12 -> per-view crop (FAR native 640x320 / NEAR 1280x640 down X2)
  -> NV12->RGB on the crop only -> CPU-only backend -> placeholder/W71 decoder
  -> full-frame detections -> tracker (IoU+centre, M-of-N) -> relevant-light
  selection -> Korean signal state machine -> carrotTlv @10 Hz.

Zero-stutter policy: cores 1-3, SCHED_IDLE, nice 19, never config_realtime_process,
newest frame only (no queueing; busy frames counted as droppedFrames), and a
self-throttle watchdog that suspends inference when modeld/deviceState suffer.

Messaging imports are lazy so this module stays importable (and lintable) on a PC.
"""
from __future__ import annotations

import time

from openpilot.selfdrive.carrot.tlv import backends, crops, tracker
from openpilot.selfdrive.carrot.tlv.watchdog import STATUS_BASELINE, STATUS_RUNNING, STATUS_SUSPENDED, SuspendWatchdog

CONNECT_RETRY_S = 0.5
FRAME_STALE_S = 1.2
PUBLISH_HZ = 10.0
MODEL_FRESH_S = 0.5
LEFT_INTENT_ATCTYPES = ("turn left", "atc left")


def _param_int(params, key: str, default: int = 0) -> int:
  try:
    value = params.get_int(key)
    return default if value is None else int(value)
  except Exception:
    return default


def _param_text(params, key: str, default: str = "") -> str:
  try:
    value = params.get(key)
  except Exception:
    return default
  if value is None:
    return default
  if isinstance(value, bytes):
    value = value.decode("utf-8", "replace")
  value = str(value).strip()
  return value if value else default


def apply_cpu_policy() -> list[str]:
  """Cores 1-3 + SCHED_IDLE + nice19 on POSIX; silently skipped on Windows."""
  applied: list[str] = []
  try:
    import os
    os.sched_setaffinity(0, {1, 2, 3})
    applied.append("affinity=1,2,3")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    import os
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    applied.append("sched=SCHED_IDLE")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    import os
    os.nice(19)
    applied.append("nice=19")
  except (AttributeError, OSError, ValueError):
    pass
  return applied


class TlvRunner:
  def __init__(self, params) -> None:
    self.params = params
    self.watchdog = SuspendWatchdog()
    self.tracker = tracker.TrafficLightTracker()
    self.machine = tracker.SignalStateMachine()

    self.backend = None
    self.decoder = None
    self.backend_error = ""
    if _param_int(params, "CarrotTlvMode") >= 1:
      name = _param_text(params, "CarrotTlvBackend", "onnx")
      model_path = _param_text(params, "CarrotTlvModelPath", "")
      try:
        self.backend, self.decoder = backends.make_backend(name, model_path)
      except NotImplementedError as exc:
        self.backend_error = str(exc)
      except backends.BackendError as exc:
        self.backend_error = str(exc)

    self.sm = None
    self.pub = None
    self.client = None
    self.last_connect_try = 0.0
    self.last_frame_mono = 0.0

    self.frame_idx = 0
    self.inferring_frames = 0
    self.dropped = 0
    self.infer_ms = 0.0
    self.proc_ms = 0.0
    self.last_view = "far"
    self.last_dets: list[tracker.Detection] = []
    self.last_matches: list[tuple[tracker.Detection, int]] = []
    self.last_state = tracker.SignalState.none
    self.last_conf = 0.0
    self.last_track_id = -1
    self._horizon_y: float | None = None
    self._calib_rpy = None

  # -- vision ---------------------------------------------------------------

  def _connect(self) -> bool:
    if self.client is not None and self.client.is_connected():
      return True
    now = time.monotonic()
    if now - self.last_connect_try < CONNECT_RETRY_S:
      return False
    self.last_connect_try = now
    from msgq.visionipc import VisionIpcClient, VisionStreamType
    client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    if not client.connect(False) or not client.num_buffers:
      return False
    self.client = client
    print(f"tlvd: connected road stream {client.width}x{client.height}", flush=True)
    return True

  def _disconnect(self) -> None:
    print("tlvd: vision stream lost; reconnecting", flush=True)
    self.client = None
    self.last_frame_mono = 0.0

  def _drain_newest_frame(self):
    """Newest unread frame; intermediate frames are skips while busy."""
    newest = None
    skipped = 0
    while True:
      frame = self.client.recv(timeout_ms=0)
      if frame is None:
        break
      if newest is not None:
        skipped += 1
      newest = frame
    if newest is not None and self.watchdog.inferring:
      self.dropped += skipped
    return newest

  def _update_horizon(self) -> float:
    calib = self.sm["liveCalibration"] if self.sm is not None and self.sm.seen["liveCalibration"] else None
    if calib is not None:
      rpy = list(calib.rpyCalib) if len(calib.rpyCalib) == 3 else None
      self._calib_rpy = rpy
    width = self.client.width if self.client is not None else crops.REFERENCE_WIDTH
    height = self.client.height if self.client is not None else crops.REFERENCE_HEIGHT
    focal = crops.focal_px_for_width(width)
    if self._horizon_y is None or calib is not None:
      self._horizon_y = crops.horizon_row(self._calib_rpy, focal, width, height)
    return float(self._horizon_y)

  # -- inference ------------------------------------------------------------

  def _process_frame(self, frame) -> None:
    width, height = frame.width, frame.height
    focal = crops.focal_px_for_width(width)
    horizon = self._update_horizon()
    view = "far" if self.inferring_frames % 2 == 0 else "near"
    geom = (crops.far_geometry if view == "far" else crops.near_geometry)(horizon, width, height, focal)

    started = time.perf_counter()
    model_input = crops.model_input_from_frame(frame.data, width, height, frame.stride, frame.uv_offset, geom)
    infer_started = time.perf_counter()
    raw = self.backend.infer(model_input)
    self.infer_ms = (time.perf_counter() - infer_started) * 1000.0
    decoded = self.decoder.decode(raw, view)
    dets = []
    for det in decoded:
      x1, y1, x2, y2 = crops.map_model_box_to_full((det.x1, det.y1, det.x2, det.y2), geom)
      dets.append(tracker.Detection(cls=det.cls, score=det.score, x1=x1, y1=y1, x2=x2, y2=y2))

    now = time.monotonic()
    self.last_matches = self.tracker.update(self.frame_idx, dets, view, focal)
    confirmed = self.tracker.confirmed(self.frame_idx)
    relevant = tracker.select_relevant(
      confirmed, self.frame_idx, image_width=width,
      path_x=self._path_x(), path_y=self._path_y(), left_intent=self._left_intent(),
    )
    lamp_state = tracker.lamp_state_for(relevant[1]) if relevant is not None else None
    self.last_state, self.last_conf = self.machine.update(now, lamp_state, relevant[2] if relevant is not None else 0.0)
    self.last_track_id = relevant[0].track_id if relevant is not None else -1

    self.last_dets = dets
    self.last_view = view
    self.frame_idx += 1
    self.inferring_frames += 1
    self.proc_ms = (time.perf_counter() - started) * 1000.0

  def _path_x(self):
    if self.sm is not None and self.sm.seen["modelV2"] and self._model_fresh():
      return tuple(self.sm["modelV2"].position.x)
    return ()

  def _path_y(self):
    if self.sm is not None and self.sm.seen["modelV2"] and self._model_fresh():
      return tuple(self.sm["modelV2"].position.y)
    return ()

  def _model_fresh(self) -> bool:
    return bool(self.sm is not None and time.monotonic() - self.sm.recv_time["modelV2"] <= MODEL_FRESH_S)

  def _left_intent(self) -> bool:
    if self.sm is None:
      return False
    if self.sm.seen["carState"] and bool(self.sm["carState"].leftBlinker):
      return True
    if self.sm.seen["carrotMan"] and str(self.sm["carrotMan"].atcType) in LEFT_INTENT_ATCTYPES:
      return True
    return False

  # -- watchdog -------------------------------------------------------------

  def _feed_watchdog(self, now: float) -> None:
    frame_drop = None
    exec_time = None
    thermal = None
    if self.sm.updated["modelV2"]:
      model = self.sm["modelV2"]
      frame_drop = float(model.frameDropPerc)
      if len(model.modelExecutionTime) > 0:
        exec_time = max(float(v) for v in model.modelExecutionTime)
    if self.sm.updated["deviceState"]:
      thermal = int(self.sm["deviceState"].thermalStatus)
    self.watchdog.update(now, frame_drop=frame_drop, exec_time=exec_time, thermal=thermal)

  # -- publish --------------------------------------------------------------

  def _publish(self, status: str, reason: str) -> None:
    from openpilot.cereal import messaging
    msg = messaging.new_message("carrotTlv")
    out = msg.carrotTlv
    out.status = status
    out.suspendReason = reason
    out.view = self.last_view
    out.inferMs = float(self.infer_ms)
    out.procMs = float(self.proc_ms)
    out.droppedFrames = int(self.dropped)
    ordered = self._ordered_dets()
    out.init("dets", len(ordered))
    for i, det in enumerate(ordered):
      d = out.dets[i]
      d.cls = int(det.cls)
      d.score = float(det.score)
      d.x1 = float(det.x1)
      d.y1 = float(det.y1)
      d.x2 = float(det.x2)
      d.y2 = float(det.y2)
    out.state = self.last_state.name
    out.stateConf = float(self.last_conf)
    out.relevantTrackId = int(self.last_track_id)
    self.pub.send("carrotTlv", msg)

  def _ordered_dets(self) -> list[tracker.Detection]:
    if self.last_track_id < 0 or not self.last_matches:
      return self.last_dets
    relevant = None
    rest: list[tracker.Detection] = []
    for det, track_id in self.last_matches:
      if track_id == self.last_track_id and relevant is None:
        relevant = det
      else:
        rest.append(det)
    if relevant is None:
      return self.last_dets
    return [relevant, *rest]

  def _status(self) -> tuple[str, str]:
    if self.backend_error:
      return "error", self.backend_error
    if self.watchdog.status == STATUS_BASELINE:
      return "suspended", STATUS_BASELINE
    if self.watchdog.status == STATUS_SUSPENDED:
      return "suspended", self.watchdog.reason or STATUS_SUSPENDED
    return STATUS_RUNNING, ""

  # -- main loop ------------------------------------------------------------

  def run(self) -> None:
    from openpilot.cereal import messaging

    self.sm = messaging.SubMaster(["modelV2", "deviceState", "carState", "carrotMan", "liveCalibration"])
    self.pub = messaging.PubMaster(["carrotTlv"])
    from openpilot.common.realtime import Ratekeeper
    rk = Ratekeeper(PUBLISH_HZ, print_delay_threshold=None)

    while True:
      now = time.monotonic()
      self.sm.update(0)
      self._feed_watchdog(now)

      if self.client is None or not self.client.is_connected():
        self._connect()
      if self.client is not None:
        frame = self._drain_newest_frame()
        if frame is not None:
          self.last_frame_mono = now
          if self.backend is not None and self.watchdog.inferring:
            try:
              self._process_frame(frame)
            except Exception as exc:
              print(f"tlvd: frame processing error: {exc!r}", flush=True)
        elif self.last_frame_mono > 0.0 and now - self.last_frame_mono > FRAME_STALE_S:
          self._disconnect()

      status, reason = self._status()
      self._publish(status, reason)
      rk.keep_time()


def main() -> None:
  from openpilot.common.params import Params
  params = Params()
  if _param_int(params, "CarrotTlvMode") < 1:
    print("tlvd: disabled (CarrotTlvMode off)", flush=True)
    return
  print(f"tlvd: start cpu-policy={apply_cpu_policy()}", flush=True)
  TlvRunner(params).run()


if __name__ == "__main__":
  main()
