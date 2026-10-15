#!/usr/bin/env python3
"""On-device traffic-light detector process (carrotTlv service).

Pipeline per processed frame:
  VisionIPC road NV12 -> per-view crop (FAR native 640x320 / NEAR 1280x640 down X2)
  -> NV12->RGB on the crop only -> CPU-only backend -> raw-head decoder (train/decode.py port)
  -> full-frame detections -> tracker (IoU+centre, M-of-N) -> relevant-light
  selection -> Korean signal state machine -> carrotTlv @10 Hz.

Zero-stutter policy (RESOURCE_BUDGET.md, enforced here, not by the scheduler):
  cores 1-3, SCHED_IDLE, nice 19, never config_realtime_process; newest frame
  only (no queueing; busy frames counted as droppedFrames); thread caps set by
  tlvd_launch before this module's numpy import (main() re-applies them for
  direct runs); strict CPU credit budget at <= 40% of one
  core of TOTAL process CPU (W99: no 1 Hz floor; frames that cannot afford
  their estimated cost are skipped and counted in skippedBudget, never in
  droppedFrames); /proc/stat cores 1-3 idle guard; RSS soft/hard caps with a
  clean non-restart-looping exit; MemAvailable floor; oom_score_adj=1000;
  copy-then-release for VisionIPC frames; no per-frame prints; <= 16 dets
  published.

Messaging imports are lazy so this module stays importable (and lintable) on a PC.
"""
from __future__ import annotations

import gc
import math
import os
import time
from collections import deque

# Thread caps (one thread for BLAS; the tinygrad backend additionally caps its
# own CPU pool at 2) and DEV=CPU must be in the environment BEFORE this module
# is imported: numpy/OpenBLAS load through the tlv imports below, and with DEV
# unset tinygrad's Device.DEFAULT probes ALL_DEVICES in order (CUDA/QCOM ahead
# of CPU, tinygrad_repo/tinygrad/device.py:14,45-51) and opens the first that
# exists - on TICI the QCOM GPU next to modeld. tlvd_launch sets both in the
# spawn child before importing this module; the module level must stay free of
# os.environ writes so the manager's preimport (process.py:181-187) leaks
# nothing. main() re-applies setdefaults for direct runs.
from openpilot.selfdrive.carrot.tlv import backends, crops, pipeline, recorder, resources, tracker
from openpilot.selfdrive.carrot.tlv.watchdog import STATUS_BASELINE, STATUS_RUNNING, STATUS_SUSPENDED, SuspendWatchdog

CONNECT_RETRY_S = 0.5
FRAME_STALE_S = 1.2
PUBLISH_HZ = 10.0
MODEL_FRESH_S = 0.5
LEFT_INTENT_ATCTYPES = ("turn left", "atc left")
PROC_POLL_S = 1.0
STATS_INTERVAL_S = 10.0
ERROR_LOG_INTERVAL_S = 10.0
RECORDER_RETRY_S = 60.0
MAX_PUBLISHED_DETS = 16
CAPNP_VIEW = {"far": "farView", "near": "nearView"}
# W99 cost-aware scheduling (credit). Classify-only frames keep a 5 Hz cadence
# target; the detector runs at 5 Hz without tracks and 2.5 Hz with them, but the
# budget decides what actually runs - there is no frame-rate floor.
CLASSIFY_INTERVAL_S = 0.2
# W121 L2: faster lamp votes while all tracks are in the far tier (>120 m)
CLASSIFY_INTERVAL_FAR_S = 0.1
DETECT_INTERVAL_S = 0.2
DETECT_INTERVAL_TRACKS_S = 0.4
DETECT_MAX_GAP_S = 3.0
# Flat-road tail-light gate needs a calibrated camera (crops.horizon_row uses
# rpyCalib); str(calStatus) because pycapnp enums stringify as the enum name.
GATE_CAL_STATUS = "calibrated"
# Crash-loop guard: more than this many starts inside the window sets the
# CarrotTlvDisabled latch so a struggling device cannot keep relaunching tlvd.
CRASH_LOOP_MAX_STARTS = 5
CRASH_LOOP_WINDOW_S = 300.0

PROC_STAT = "/proc/stat"
PROC_MEMINFO = "/proc/meminfo"
PROC_SELF_STATM = "/proc/self/statm"


def choose_frame_kind(*, now: float, last_detect: float, last_classify: float, has_tracks: bool,
                      classifier_enabled: bool, budget: resources.CpuCreditBudget,
                      far_tracks_only: bool = False) -> str | None:
  """What this frame should run: "detect", "classify", or None (skip).

  Detector first, then the cheap classifier while tracks exist; when the
  detector is overdue (DETECT_MAX_GAP_S) a classify frame that would starve it
  of its estimate is refused. Pure apart from budget.can_run()/estimate().
  """
  detect_interval = DETECT_INTERVAL_TRACKS_S if has_tracks else DETECT_INTERVAL_S
  if now - last_detect >= detect_interval and budget.can_run("detect", now):
    return "detect"
  classify_interval = CLASSIFY_INTERVAL_FAR_S if far_tracks_only else CLASSIFY_INTERVAL_S
  if not (classifier_enabled and has_tracks and now - last_classify >= classify_interval):
    return None
  if not budget.can_run("classify", now):
    return None
  if (now - last_detect >= DETECT_MAX_GAP_S
      and budget.credit - budget.estimate("classify") < budget.estimate("detect")):
    return None  # save the credit for the overdue detector
  return "classify"


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


def _param_bool(params, key: str, default: bool = False) -> bool:
  try:
    value = params.get_bool(key)
    return default if value is None else bool(value)
  except Exception:
    return default

def _host_is_tici() -> bool:
  try:
    from openpilot.system.hardware import TICI
    return bool(TICI)
  except Exception:
    return False


def _as_float(value):
  return None if value is None else float(value)


def _enum_ordinal(value) -> int:
  """Ordinal for a capnp enum field (int() on _DynamicEnum raises TypeError).

  Real cereal enums (e.g. deviceState.thermalStatus) expose `.raw` as the
  schema index (green=0, yellow=1, ...); plain ints from test fakes pass
  through int()."""
  raw = getattr(value, "raw", None)
  return int(raw) if raw is not None else int(value)


def resolve_backend_name(value: str, is_tici: bool) -> str:
  """Resolve CarrotTlvBackend. The param default 'onnx' means the onnxruntime
  CPU backend on PC, but tinygrad on TICI where onnxruntime is not a device
  dependency. 'null'/'dsp' and any explicit other name are untouched."""
  name = (value or "").strip()
  if name in ("", "onnx"):
    return "tinygrad" if is_tici else "onnx"
  return name


def apply_cpu_policy() -> list[str]:
  """Cores 1-3 + SCHED_IDLE + nice19 on POSIX; silently skipped on Windows."""
  applied: list[str] = []
  try:
    os.sched_setaffinity(0, {1, 2, 3})
    applied.append("affinity=1,2,3")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    applied.append("sched=SCHED_IDLE")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    os.nice(19)
    applied.append("nice=19")
  except (AttributeError, OSError, ValueError):
    pass
  return applied


class TlvRunner:
  def __init__(self, params, *, recorder_root: str | None = None) -> None:
    self.params = params
    self.watchdog = SuspendWatchdog()

    self.backend = None
    self.decoder = None
    self.backend_error = ""
    self.classifier = backends.NullClassifier()
    self.classifier_error = ""
    # W129: vision is always on while tlvd runs (gated by TrafficLightDetectMode
    # in the manager); the mode toggle no longer exists.
    name = resolve_backend_name(_param_text(params, "CarrotTlvBackend", ""), _host_is_tici())
    model_path = _param_text(params, "CarrotTlvModelPath", "") or backends.BUNDLED_DETECTOR
    try:
      self.backend, self.decoder = backends.make_backend(name, model_path)
    except NotImplementedError as exc:
      self.backend_error = str(exc)
    except backends.BackendError as exc:
      self.backend_error = str(exc)
    if self.backend is not None:
      # Lamp classifier: a bad/missing lamp model must never kill tlvd; fall
      # back to the detector-only behavior (CONTRACT.md detect-then-track).
      lamp_path = _param_text(params, "CarrotTlvLampModelPath", "") or backends.BUNDLED_LAMP
      try:
        self.classifier = backends.make_classifier(name, lamp_path)
        self.classifier.warmup()
      except Exception as exc:
        self.classifier_error = str(exc)
        self.classifier = backends.NullClassifier()
      # Detector warm-up BEFORE the run loop and before the budget exists:
      # the first real detect frame otherwise pays the one-time compile
      # (route 00000443: 12.2 s, credit -9.0, then rss_soft forever).
      try:
        self.backend.warmup()
      except Exception as exc:
        self.backend_error = f"detector warmup failed: {exc!r}"
    self.pipeline = pipeline.FramePipeline(self.backend, self.decoder, classifier=self.classifier)

    # Training recorder: wanted whenever tlvd runs (W129: no mode gate).
    # The recorder is constructed lazily on the first poll (TICI or an injected
    # test root); the root is only injected by tests.
    self._recorder_root = recorder_root
    self._recorder_last_try = -RECORDER_RETRY_S
    self._recorder_failures = 0
    self.recorder = None
    self._sync_recorder_mode()

    # W99: strict CPU credit budget (replaces the pacing governor) and the
    # ego-motion odometry used for track-box prediction between detect frames.
    # W127: built only AFTER both model warm-ups; the budget's accounting
    # baseline is taken at creation, so warm-up CPU is never charged to frames.
    self.budget = resources.CpuCreditBudget()
    self.odo = tracker.EgoOdometry()
    self.last_detect = -float("inf")
    self.last_classify = -float("inf")
    self.skipped_budget = 0
    self._frame_times: dict[str, deque[float]] = {"detect": deque(maxlen=256), "classify": deque(maxlen=256)}
    self.cpu_guard = resources.CpuContentionGuard()
    self.mem_guard = resources.MemoryGuard()
    self.page_size = resources.default_page_size()
    self._cpu_reason: str | None = None
    self._mem = resources.MemoryDecision()
    self._hard_exit_reason: str | None = None
    self._rss_bytes = 0
    self._last_proc_poll = 0.0
    self._last_stats = 0.0
    self._last_error_log = 0.0

    self.sm = None
    self.pub = None
    self.client = None
    self.last_connect_try = 0.0
    self.last_frame_mono = 0.0

    self.dropped = 0
    self.processed = 0
    self.last_view = "far"
    self._horizon_y: float | None = None
    self._calib_rpy = None
    self._gate_enabled = False

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
    if newest is not None and self._can_infer():
      self.dropped += skipped
    return newest

  def _update_horizon(self) -> float:
    calib = self.sm["liveCalibration"] if self.sm is not None and self.sm.seen["liveCalibration"] else None
    if calib is not None:
      rpy = list(calib.rpyCalib) if len(calib.rpyCalib) == 3 else None
      self._calib_rpy = rpy
      # Flat-road tail-light gate is only valid on a calibrated camera.
      self._gate_enabled = bool(
        rpy is not None and all(math.isfinite(v) for v in rpy)
        and str(calib.calStatus) == GATE_CAL_STATUS
      )
    width = self.client.width if self.client is not None else crops.REFERENCE_WIDTH
    height = self.client.height if self.client is not None else crops.REFERENCE_HEIGHT
    focal = crops.focal_px_for_width(width)
    if self._horizon_y is None or calib is not None:
      self._horizon_y = crops.horizon_row(self._calib_rpy, focal, width, height)
    return float(self._horizon_y)

  # -- inference ------------------------------------------------------------

  def _can_infer(self) -> bool:
    return (self.backend is not None
            and not self.backend_error
            and self.watchdog.inferring
            and self._cpu_reason is None
            and self._mem.suspend_reason is None)

  def _process_frame(self, frame, *, kind: str, v_ego_kph: float) -> None:
    detect = kind == "detect"
    width, height = frame.width, frame.height
    focal = crops.focal_px_for_width(width)
    horizon = self._update_horizon()
    # Ego-motion context for predicted track boxes (anchors/crops/association).
    foe_x, _foe_y = crops.vanishing_point(self._calib_rpy, focal, width, height)
    self.pipeline.ego = tracker.EgoFrame(
      odo=self.odo, focal_px=focal, foe_x=foe_x, foe_y=horizon, image_w=width, image_h=height,
    )
    if detect:
      # Only detect frames advance the speed-aware ViewScheduler.
      view = self.pipeline.next_view(v_ego_kph)
    else:
      far_geom = pipeline.FramePipeline.geometry("far", horizon, width, height, focal)
      view = self.pipeline.classify_view(far_geom)
    raised = view == "near" and v_ego_kph < pipeline.STANDSTILL_V_KPH
    geom = pipeline.FramePipeline.geometry(view, horizon, width, height, focal, raised=raised)

    data = frame.data
    # VisionIPC: copy the ROI into the preallocated workspace NOW and drop the
    # frame before inference. The VisionBuf only borrows a client-owned buffer
    # (msgq/visionipc/visionipc_client.cc:108) that camerad can reuse as soon as
    # we recv again (msgq/visionipc/visionipc_pyx.pyx:153), so it must never be
    # held across the model call.
    self.pipeline.prepare(data, width, height, frame.stride, frame.uv_offset, geom)
    del frame, data

    out = self.pipeline.infer(
      view, geom, image_width=width, path_x=self._path_x(), path_y=self._path_y(),
      left_intent=self._left_intent(), now=time.monotonic(), detect=detect,
      horizon_y=horizon, gate_enabled=self._gate_enabled,
    )
    self.last_view = out.view
    self.processed += 1
    if self.recorder is not None:
      try:
        self._offer_recorder(out, geom, view, raised, horizon)
      except Exception as exc:
        self._log_error(f"tlvd: recorder offer error: {exc!r}", time.monotonic())

  def _offer_recorder(self, out, geom: crops.ViewGeometry, view: str, raised: bool, horizon: float) -> None:
    """Hand this frame's crop + metadata to the shadow-mode recorder (hot path)."""
    car = self.sm["carState"] if self.sm is not None and self.sm.seen["carState"] else None
    if self.sm is not None and self.sm.seen["roadCameraState"]:
      frame_id = int(self.sm["roadCameraState"].frameId)
    else:
      frame_id = int(self.pipeline.frame_idx)

    self.recorder.offer(
      self.pipeline.preprocess.model_input,
      now=time.monotonic(),
      frame_id=frame_id,
      view=out.view,
      raised=bool(raised),
      geom=geom,
      horizon_y=float(horizon),
      calib_rpy=self._calib_rpy,
      v_ego=_as_float(car.vEgo) if car is not None else None,
      a_ego=_as_float(car.aEgo) if car is not None else None,
      yaw_rate=_as_float(getattr(car, "yawRate", None)),
      steering_angle_deg=_as_float(getattr(car, "steeringAngleDeg", None)),
      left_blinker=bool(car.leftBlinker) if car is not None else None,
      right_blinker=bool(car.rightBlinker) if car is not None else None,
      dets=out.dets,
      gated=out.gated,
      tracks=self.pipeline.tracker.tracks,
      relevant_track_id=int(out.track_id),
      state=int(out.state),
      state_conf=float(out.state_conf),
      red_likely=bool(out.red_likely),
    )

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

  def _v_ego_kph(self) -> float:
    if self.sm is not None and self.sm.seen["carState"]:
      return float(self.sm["carState"].vEgo) * 3.6
    return 0.0

  # -- system guards --------------------------------------------------------

  def _poll_system(self, now: float) -> None:
    """1 Hz: /proc/stat cores guard, RSS soft/hard, MemAvailable floor."""
    if now - self._last_proc_poll < PROC_POLL_S:
      return
    self._last_proc_poll = now
    self._sync_recorder_mode()
    self._cpu_reason = self.cpu_guard.update(now, resources.read_text(PROC_STAT))
    rss_bytes = resources.read_rss_bytes(PROC_SELF_STATM, self.page_size)
    mem_available = resources.read_mem_available_bytes(PROC_MEMINFO)
    if rss_bytes is None:
      return
    self._rss_bytes = rss_bytes
    self._mem = self.mem_guard.update(now, rss_bytes, mem_available)
    if self._mem.run_gc:
      gc.collect()
    if self._mem.drop_caches:
      resources.drop_caches()
    if self._mem.exit_reason is not None:
      self._hard_exit_reason = self._mem.exit_reason

  def _handle_hard_exit(self) -> None:
    """Hard RSS limit: publish status=error and set the CarrotTlvDisabled latch
    so the manager's enable_tlvd() goes False and tlvd is not restarted
    (restart_if_crash alone would start the process again regardless of exit
    code). The latch clears on the next manager start."""
    reason = self._hard_exit_reason or "rss_hard"
    self._publish("error", reason)
    try:
      self.params.put_bool("CarrotTlvDisabled", True)
    except Exception as exc:
      print(f"tlvd: failed to set CarrotTlvDisabled on {reason}: {exc!r}", flush=True)
    print(f"tlvd: hard memory limit ({reason}); exiting without restart", flush=True)

  def close_recorder(self) -> None:
    """Drain the recorder queue (bounded) when tlvd shuts down."""
    if self.recorder is None:
      return
    try:
      self.recorder.close(timeout_s=1.0)
    except Exception:
      pass

  def _sync_recorder_mode(self) -> None:
    """Recorder is wanted whenever tlvd runs (W129: no mode gate).

    The manager does not restart tlvd on setting changes, so this stays on the
    1 Hz poll; a tlvd with no recorder yet constructs it lazily (TICI or an
    injected test root) with the RECORDER_RETRY_S backoff."""
    if self.recorder is not None:
      self.recorder.active = True
      return
    if not (_host_is_tici() or self._recorder_root is not None):
      return
    now = time.monotonic()
    if now - self._recorder_last_try < RECORDER_RETRY_S:
      return  # backoff so a failing construction cannot retry every second
    self._recorder_last_try = now
    try:
      self.recorder = recorder.TlvRecorder(root=self._recorder_root, active=True)
      print(f"tlvd: recorder root={self.recorder.root}", flush=True)
    except Exception as exc:
      self._recorder_failures += 1
      if self._recorder_failures == 1:
        print(f"tlvd: recorder disabled: {exc!r}", flush=True)

  def _update_odometry(self, now: float) -> None:
    """Integrate fresh carState vEgo/yawRate for ego-motion track prediction."""
    if self.sm is not None and self.sm.updated["carState"]:
      car = self.sm["carState"]
      self.odo.update(now, _as_float(car.vEgo), _as_float(getattr(car, "yawRate", None)))

  # -- watchdog -------------------------------------------------------------

  def _feed_watchdog(self, now: float) -> None:
    # Every signal is intrinsic to its message (a value the publisher wrote),
    # never an inter-arrival time measured by tlvd: tlvd blocks ~0.7 s in each
    # detect, so its own sampling gaps looked like controls/camera stress and
    # suspended tlvd right after every resume (field route 447).
    frame_drop = None
    exec_time = None
    thermal = None
    controls_lag_ms = None
    if self.sm.updated["modelV2"]:
      model = self.sm["modelV2"]
      frame_drop = float(model.frameDropPerc)
      exec_time = _as_float(model.modelExecutionTime)
    if self.sm.updated["deviceState"]:
      thermal = _enum_ordinal(self.sm["deviceState"].thermalStatus)
    if self.sm.seen["carState"] and self.sm.updated["carState"]:
      controls_lag_ms = _as_float(getattr(self.sm["carState"], "cumLagMs", None))
    self.watchdog.update(
      now, frame_drop=frame_drop, exec_time=exec_time, thermal=thermal,
      controls_lag_ms=controls_lag_ms,
    )

  # -- publish --------------------------------------------------------------

  def _publish(self, status: str, reason: str) -> None:
    from openpilot.cereal import messaging
    msg = messaging.new_message("carrotTlv")
    out = msg.carrotTlv
    out.status = status
    out.suspendReason = reason
    out.view = CAPNP_VIEW[self.last_view]
    out.inferMs = float(self.pipeline.infer_ms)
    out.procMs = float(self.pipeline.proc_ms)
    out.droppedFrames = int(self.dropped)
    now = time.monotonic()
    out.cpuCredit = float(self.budget.credit)
    out.cpuDuty = float(self.budget.duty or 0.0)
    out.detHz = float(self._frame_rate("detect", now))
    out.clsHz = float(self._frame_rate("classify", now))
    out.detCostMs = float(self.budget.estimate_ms("detect"))
    out.clsCostMs = float(self.budget.estimate_ms("classify"))
    out.skippedBudget = int(self.skipped_budget)
    out.tailGated = int(self.pipeline.tail_gated_total)
    ordered = self._ordered_dets()[:MAX_PUBLISHED_DETS]
    out.init("dets", len(ordered))
    for i, det in enumerate(ordered):
      d = out.dets[i]
      d.cls = int(det.cls)
      d.score = float(det.score)
      d.x1 = float(det.x1)
      d.y1 = float(det.y1)
      d.x2 = float(det.x2)
      d.y2 = float(det.y2)
    out.state = "redLikely" if self.pipeline.last_red_likely else self.pipeline.last_state.name
    out.stateConf = float(self.pipeline.last_conf)
    out.relevantTrackId = int(self.pipeline.last_track_id)
    # Same-track green (W74D): only the light that was confirmed red/redLeft
    # within the last 120 s may release a TLV green/left start downstream.
    out.goAfterRed = bool(
      self.pipeline.last_state in (tracker.SignalState.green, tracker.SignalState.greenLeft, tracker.SignalState.left)
      and self.pipeline.tracker.go_after_red(self.pipeline.last_track_id, time.monotonic())
    )
    self.pub.send("carrotTlv", msg)

  def _ordered_dets(self) -> list[tracker.Detection]:
    last_track_id = self.pipeline.last_track_id
    if last_track_id < 0 or not self.pipeline.tracker.last_matches:
      return self.pipeline.last_dets
    relevant = None
    rest: list[tracker.Detection] = []
    for det, track_id in self.pipeline.tracker.last_matches:
      if track_id == last_track_id and relevant is None:
        relevant = det
      else:
        rest.append(det)
    if relevant is None:
      return self.pipeline.last_dets
    return [relevant, *rest]

  def _status(self) -> tuple[str, str]:
    if self.backend_error:
      return "error", self.backend_error
    if self._cpu_reason is not None:
      return "suspended", self._cpu_reason
    if self._mem.suspend_reason is not None:
      return "suspended", self._mem.suspend_reason
    if self.watchdog.status == STATUS_BASELINE:
      return "suspended", STATUS_BASELINE
    if self.watchdog.status == STATUS_SUSPENDED:
      return "suspended", self.watchdog.reason or STATUS_SUSPENDED
    return STATUS_RUNNING, ""

  # -- logging --------------------------------------------------------------

  def _log_error(self, text: str, now: float) -> None:
    if now - self._last_error_log >= ERROR_LOG_INTERVAL_S:
      self._last_error_log = now
      print(text, flush=True)

  def _frame_rate(self, kind: str, now: float) -> float:
    """Frames per second of `kind` over the last STATS_INTERVAL_S seconds."""
    times = self._frame_times[kind]
    while times and now - times[0] > STATS_INTERVAL_S:
      times.popleft()
    if len(times) < 2:
      return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0.0 else 0.0

  def _maybe_log_stats(self, now: float) -> None:
    if now - self._last_stats < STATS_INTERVAL_S:
      return
    self._last_stats = now
    duty = self.budget.duty
    duty_text = "n/a" if duty is None else f"{duty * 100.0:.0f}%"
    status, _ = self._status()
    rss_mb = self._rss_bytes / resources.MB
    decoder_nc = getattr(self.decoder, "num_classes", None)
    det_nc_text = "-" if decoder_nc is None else str(decoder_nc)
    parts = (
      f"frames={self.processed}",
      f"dropped={self.dropped}",
      f"skipped_budget={self.skipped_budget}",
      f"credit={self.budget.credit:.2f}s",
      f"duty={duty_text}",
      f"est_det={self.budget.estimate_ms('detect'):.0f}ms",
      f"est_cls={self.budget.estimate_ms('classify'):.0f}ms",
      f"det_nc={det_nc_text}",
      f"det_hz={self._frame_rate('detect', now):.2f}",
      f"cls_hz={self._frame_rate('classify', now):.2f}",
      f"tail_gated={self.pipeline.tail_gated_total}",
      f"rss={rss_mb:.0f}MB",
      f"infer={self.pipeline.infer_ms:.0f}ms",
      f"proc={self.pipeline.proc_ms:.0f}ms",
      status,
    )
    if self.recorder is not None:
      rec = self.recorder.stats()
      parts = parts + (
        f"rec_saved={rec['saved']}",
        f"rec_full={rec['dropped_full']}",
        f"rec_rate={rec['dropped_rate']}",
        f"rec_mb={rec['bytes'] / resources.MB:.1f}",
        f"rec_pause={rec['paused'] or 'no'}",
      )
    print("tlvd: stats " + " ".join(parts), flush=True)

  # -- main loop ------------------------------------------------------------

  def run(self) -> None:
    from openpilot.cereal import messaging

    self.sm = messaging.SubMaster(["modelV2", "deviceState", "controlsState", "roadCameraState", "carState", "carrotMan", "liveCalibration"])
    self.pub = messaging.PubMaster(["carrotTlv"])
    if self.classifier_error:
      print(f"tlvd: lamp classifier disabled: {self.classifier_error}", flush=True)
    elif self.classifier.enabled:
      print("tlvd: lamp classifier ready", flush=True)
    from openpilot.common.realtime import Ratekeeper
    rk = Ratekeeper(PUBLISH_HZ, print_delay_threshold=None)

    while True:
      now = time.monotonic()
      self.sm.update(0)
      self._update_odometry(now)
      self._feed_watchdog(now)
      self._poll_system(now)
      # Keep credit/duty live even across suspensions (all process CPU charged).
      self.budget.tick(now)
      if self._hard_exit_reason is not None:
        self._handle_hard_exit()
        return  # exit code 0; the CarrotTlvDisabled latch keeps the manager from restarting tlvd

      if self.client is None or not self.client.is_connected():
        self._connect()
      if self.client is not None:
        frame = self._drain_newest_frame()
        if frame is not None:
          self.last_frame_mono = now
          if self._can_infer():
            kind = choose_frame_kind(
              now=now, last_detect=self.last_detect, last_classify=self.last_classify,
              has_tracks=self.pipeline.has_tracks, classifier_enabled=self.classifier.enabled,
              budget=self.budget, far_tracks_only=self.pipeline.far_tracks_only,
            )
            if kind is None:
              self.skipped_budget += 1  # budget/schedule skip, NOT a dropped frame
            else:
              self.budget.begin(kind, now)
              try:
                self._process_frame(frame, kind=kind, v_ego_kph=self._v_ego_kph())
              except Exception as exc:
                self._log_error(f"tlvd: frame processing error: {exc!r}", now)
              finally:
                self._frame_times[kind].append(now)
                if kind == "detect":
                  # A detect frame also classifies the tracked lights.
                  self.last_detect = now
                  self.last_classify = now
                else:
                  self.last_classify = now
                self.budget.end(now)
        elif self.last_frame_mono > 0.0 and now - self.last_frame_mono > FRAME_STALE_S:
          self._disconnect()

      status, reason = self._status()
      self._publish(status, reason)
      self._maybe_log_stats(now)
      rk.keep_time()


def crash_loop_guard(params, now: float) -> bool:
  """Record this start; True (and CarrotTlvDisabled set) when tlvd has started
  more than CRASH_LOOP_MAX_STARTS times within CRASH_LOOP_WINDOW_S."""
  starts: list[float] = []
  for item in _param_text(params, "CarrotTlvRecentStarts", "").split(","):
    try:
      t = float(item)
    except ValueError:
      continue
    if 0.0 <= now - t <= CRASH_LOOP_WINDOW_S:
      starts.append(t)
  starts.append(now)
  try:
    params.put("CarrotTlvRecentStarts", ",".join(f"{t:.1f}" for t in starts))
  except Exception:
    pass
  if len(starts) <= CRASH_LOOP_MAX_STARTS:
    return False
  try:
    params.put_bool("CarrotTlvDisabled", True)
  except Exception as exc:
    print(f"tlvd: failed to set CarrotTlvDisabled on crash loop: {exc!r}", flush=True)
  return True


def main() -> None:
  # Direct-run fallback: tlvd_launch sets these before importing this module in
  # the spawn child; setdefault keeps an explicit parent value for direct runs.
  for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")
  os.environ.setdefault("DEV", "CPU")
  from openpilot.common.params import Params
  params = Params()
  # W129: no CarrotTlvMode gate; the manager starts tlvd whenever
  # TrafficLightDetectMode >= 1 and the CarrotTlvDisabled latch is clear.
  if _param_int(params, "TrafficLightDetectMode") < 1:
    print("tlvd: disabled (TrafficLightDetectMode off)", flush=True)
    return
  if _param_bool(params, "CarrotTlvDisabled"):
    print("tlvd: disabled (CarrotTlvDisabled latch)", flush=True)
    return
  if crash_loop_guard(params, time.monotonic()):
    print(f"tlvd: >{CRASH_LOOP_MAX_STARTS} starts in {CRASH_LOOP_WINDOW_S:.0f} s; CarrotTlvDisabled set", flush=True)
    return
  print(f"tlvd: start cpu-policy={apply_cpu_policy()}", flush=True)
  if not resources.write_oom_score_adj(1000):
    print("tlvd: oom_score_adj not writable (non-Linux or missing /proc)", flush=True)
  runner = TlvRunner(params)
  try:
    runner.run()
  finally:
    close_recorder = getattr(runner, "close_recorder", None)
    if close_recorder is not None:
      close_recorder()


if __name__ == "__main__":
  main()
