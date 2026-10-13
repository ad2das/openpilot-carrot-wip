"""Shadow-mode C3X-native training-data recorder for tlvd (CarrotTlvMode == 1).

Records exactly the 640x320 RGB model input the detector just saw, one JPEG per
saved unit plus one metadata line in <session>/index.jsonl. The hot path only
copies the model input into a preallocated 4-slot ring and hands a metadata dict
to a single daemon writer thread, which does the JPEG encode, the file write and
the storage-cap checks. Nothing here may slow the tlvd frame loop: every public
call is defensive, and offer() returns False instead of raising.

Sampling (decided on the hot path with cheap comparisons only):
  light      >=1 det OR >=1 track OR red_likely OR state != none:
             at most 1 per 0.5 s while v_ego >= 1 m/s, 1 per 3 s below;
  background none of the above: at most 1 per 5 s and only while v_ego >= 3;
  transition machine state changed vs the previous processed frame: saved
             immediately (bypasses the intervals), at most 4 per 10 s.

Storage: root <= 1.0 GB (oldest session dirs deleted first, never the current
one; a current session over 1.0 GB stops recording), and a pause below 5 GB
free on the filesystem that clears above 6 GB. Caps are enforced in the writer
thread with a running byte counter and a full rescan at most once per 60 s.

Sessions: the session being written is named in root/.active (heartbeat every
5 s) and rotates every 10 minutes (or when its directory disappears). Finished
sessions are uploaded by carrot_log_sync as folder tlvrec--<session> and are
deleted from the device after a complete upload (see log_sync.py).

No GPS or location data of any kind is recorded.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np

try:
  import cv2
  cv2.setNumThreads(1)
except Exception:  # pragma: no cover - cv2 is required in production
  cv2 = None

MB = 1024 * 1024
MODEL_H = 320
MODEL_W = 640
JPEG_QUALITY = 90
RING_SLOTS = 4
QUEUE_MAXSIZE = 4

DEFAULT_ROOT = "/data/media/0/carrot_tlv_rec"
DEFAULT_MAX_ROOT_BYTES = 1024 * MB
DEFAULT_DISK_MIN_FREE_BYTES = 5 * 1024 * MB
DEFAULT_DISK_REARM_FREE_BYTES = 6 * 1024 * MB

LIGHT_INTERVAL_S = 0.5
STANDSTILL_INTERVAL_S = 3.0
STANDSTILL_V_MS = 1.0
BACKGROUND_INTERVAL_S = 5.0
BACKGROUND_MIN_V_MS = 3.0
TRANSITION_WINDOW_S = 10.0
TRANSITION_MAX = 4
RESCAN_INTERVAL_S = 60.0
ERROR_LOG_INTERVAL_S = 60.0

SESSION_ROTATE_S = 600.0
ACTIVE_NAME = ".active"
ACTIVE_HEARTBEAT_S = 5.0
SESSION_NAME_BUMP_MAX = 60

SESSION_NAME_RE = re.compile(r"^\d{8}-\d{6}$")


def _encode_jpeg(rgb_hwc: np.ndarray) -> bytes | None:
  """HWC RGB uint8 -> JPEG quality 90 (BGR for cv2)."""
  if cv2 is None:
    return None
  bgr = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
  ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
  if not ok:
    return None
  return buf.tobytes()


def _geom_dict(geom) -> dict:
  return {
    "x0": int(geom.x0),
    "y0": int(geom.y0),
    "w": int(geom.w),
    "h": int(geom.h),
    "scale": int(geom.scale),
    "focal_px": float(geom.focal_px),
    "model_w": int(geom.model_w),
    "model_h": int(geom.model_h),
  }


class TlvRecorder:
  """One recorder instance per tlvd process; active only in CarrotTlvMode == 1."""

  def __init__(self, root: str | None = None, *, active: bool = True, clock=time.monotonic,
               statvfs_fn=None, encode_fn=None,
               max_root_bytes: int = DEFAULT_MAX_ROOT_BYTES,
               disk_min_free_bytes: int = DEFAULT_DISK_MIN_FREE_BYTES,
               disk_rearm_free_bytes: int = DEFAULT_DISK_REARM_FREE_BYTES,
               rescan_interval_s: float = RESCAN_INTERVAL_S,
               session_name_fn=None) -> None:
    self.root = str(root) if root else DEFAULT_ROOT
    self.active = bool(active)
    self._clock = clock
    self._statvfs_fn = statvfs_fn
    self._encode = encode_fn if encode_fn is not None else _encode_jpeg
    self._max_root_bytes = int(max_root_bytes)
    self._disk_min_free_bytes = int(disk_min_free_bytes)
    self._disk_rearm_free_bytes = int(disk_rearm_free_bytes)
    self._rescan_interval_s = float(rescan_interval_s)
    self._session_name_fn = session_name_fn if session_name_fn is not None else (
      lambda: time.strftime("%Y%m%d-%H%M%S", time.localtime()))

    self._slots = [np.empty((MODEL_H, MODEL_W, 3), dtype=np.uint8) for _ in range(RING_SLOTS)]
    self._busy = [False] * RING_SLOTS
    self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    self._stop = threading.Event()

    self._prev_state = 0
    self._last_light = -float("inf")
    self._last_background = -float("inf")
    self._transition_times: deque[float] = deque(maxlen=TRANSITION_MAX)
    self._next_seq = 0

    self.saved = 0
    self.dropped_full = 0
    self.dropped_rate = 0
    self._paused: str | None = None
    self._bytes = 0
    self._sizes: dict[str, int] = {}
    self._session_name: str | None = None
    self._session_started = -float("inf")
    self._last_heartbeat = -float("inf")
    self._last_scan = -float("inf")
    self._last_error_log = -float("inf")

    self._thread = threading.Thread(target=self._writer_loop, name="tlv-recorder", daemon=True) if self.active else None
    if self._thread is not None:
      self._thread.start()

  # -- hot path ---------------------------------------------------------------

  def offer(self, model_input, *, now: float, frame_id: int, view: str, raised: bool, geom,
            horizon_y: float, calib_rpy, v_ego, a_ego, yaw_rate, steering_angle_deg,
            left_blinker, right_blinker, dets, tracks, relevant_track_id: int,
            state: int, state_conf: float, red_likely: bool,
            gated=(), wall_time_iso: str | None = None) -> bool:
    """Queue one view crop if the sampling policy allows it. Never raises."""
    slot: int | None = None
    try:
      prev_state = self._prev_state
      self._prev_state = int(state)
      if not self.active or self._paused is not None:
        return False

      light = bool(dets) or bool(tracks) or bool(red_likely) or int(state) != 0
      speed = float(v_ego) if v_ego is not None else 0.0
      reason = None
      if int(state) != prev_state and self._transition_due(now):
        reason = "transition"
      elif light and now - self._last_light >= (LIGHT_INTERVAL_S if speed >= STANDSTILL_V_MS else STANDSTILL_INTERVAL_S):
        reason = "light"
      elif (not light) and speed >= BACKGROUND_MIN_V_MS and now - self._last_background >= BACKGROUND_INTERVAL_S:
        reason = "background"
      if reason is None:
        if light or speed >= BACKGROUND_MIN_V_MS:
          self.dropped_rate += 1
        return False

      slot = self._take_slot()
      if slot is None:
        self.dropped_full += 1
        return False

      plane = model_input[0]
      dest = self._slots[slot]
      np.copyto(dest[:, :, 0], plane[0])
      np.copyto(dest[:, :, 1], plane[1])
      np.copyto(dest[:, :, 2], plane[2])
      seq = self._next_seq
      self._next_seq = seq + 1
      if wall_time_iso is None:
        wall_time_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
      meta = {
        "file": f"{seq:07d}.jpg",
        "t_mono": float(now),
        "wall_time_iso": wall_time_iso,
        "frame_id": int(frame_id),
        "view": str(view),
        "raised": bool(raised),
        "geom": _geom_dict(geom),
        "horizon_y": float(horizon_y),
        "calib_rpy": list(calib_rpy) if calib_rpy is not None else None,
        "v_ego": None if v_ego is None else float(v_ego),
        "a_ego": None if a_ego is None else float(a_ego),
        "yaw_rate": None if yaw_rate is None else float(yaw_rate),
        "steering_angle_deg": None if steering_angle_deg is None else float(steering_angle_deg),
        "left_blinker": None if left_blinker is None else bool(left_blinker),
        "right_blinker": None if right_blinker is None else bool(right_blinker),
        "dets": [
          {"cls": int(d.cls), "score": float(d.score), "x1": float(d.x1), "y1": float(d.y1),
           "x2": float(d.x2), "y2": float(d.y2)}
          for d in dets
        ],
        # Flat-road tail-light gate: detections dropped this frame, same format
        # as "dets" (the kept list) so uploaded photos can be audited.
        "gated": [
          {"cls": int(d.cls), "score": float(d.score), "x1": float(d.x1), "y1": float(d.y1),
           "x2": float(d.x2), "y2": float(d.y2)}
          for d in gated
        ],
        "tracks": [self._track_meta(t) for t in tracks],
        "relevant_track_id": int(relevant_track_id),
        "state": int(state),
        "state_conf": float(state_conf),
        "red_likely": bool(red_likely),
        "reason": reason,
      }
      try:
        self._queue.put_nowait((slot, meta))
      except queue.Full:
        self._busy[slot] = False
        self.dropped_full += 1
        return False
      if reason == "transition":
        self._transition_times.append(float(now))
      if reason == "background":
        self._last_background = float(now)
      # Any saved unit resets the light cadence, so a transition burst cannot be
      # followed by an immediate "light" duplicate save.
      self._last_light = float(now)
      return True
    except Exception:
      if slot is not None:
        self._busy[slot] = False
      return False

  @staticmethod
  def _track_meta(track) -> dict:
    meta = {
      "id": int(track.track_id),
      "cls": int(track.cls),
      "score": float(track.score),
      "box": [float(v) for v in track.box],
      "last_frame": int(track.last_frame),
    }
    if track.votes:
      _, vote_cls, vote_prob = track.votes[-1]
      meta["vote_cls"] = int(vote_cls)
      meta["vote_prob"] = float(vote_prob)
    return meta

  def _transition_due(self, now: float) -> bool:
    times = self._transition_times
    if len(times) >= TRANSITION_MAX and now - times[0] < TRANSITION_WINDOW_S:
      return False
    return True

  def _take_slot(self) -> int | None:
    busy = self._busy
    for i in range(RING_SLOTS):
      if not busy[i]:
        busy[i] = True
        return i
    return None

  # -- writer thread ----------------------------------------------------------

  def _writer_loop(self) -> None:
    while True:
      try:
        item = self._queue.get(timeout=0.5)
      except queue.Empty:
        self._poll_storage()
        if self._stop.is_set():
          return
        continue
      if item is None:
        return
      self._handle(item)
      if self._stop.is_set() and self._queue.empty():
        return

  def _handle(self, item) -> None:
    slot, meta = item
    try:
      if self._paused is None:
        self._ensure_session()
        jpeg = self._encode(self._slots[slot])
        if not jpeg:
          raise RuntimeError("jpeg encode failed")
        path = os.path.join(self.root, self._session_name, meta["file"])
        with open(path, "wb") as f:
          f.write(jpeg)
        line = json.dumps(meta, separators=(",", ":")) + "\n"
        with open(os.path.join(self.root, self._session_name, "index.jsonl"), "a", encoding="utf-8") as f:
          f.write(line)
        written = len(jpeg) + len(line)
        self._sizes[self._session_name] = self._sizes.get(self._session_name, 0) + written
        self._bytes += written
        self.saved += 1
    except Exception as exc:
      self._log_error(exc)
    finally:
      # Caps are enforced before the slot is released, so a drained queue
      # always has its storage state observed (tests rely on this order).
      self._poll_storage()
      self._busy[slot] = False

  def _ensure_session(self) -> None:
    name = self._session_name
    if name is not None:
      rotated = self._clock() - self._session_started >= SESSION_ROTATE_S
      vanished = not os.path.isdir(os.path.join(self.root, name))
      if not rotated and not vanished:
        return
      self._session_name = None
    chosen = self._session_name_fn()
    bumps = 0
    while os.path.exists(os.path.join(self.root, chosen)):
      if bumps >= SESSION_NAME_BUMP_MAX:
        raise RuntimeError("no free session name")
      chosen = (datetime.strptime(chosen, "%Y%m%d-%H%M%S")
                + timedelta(seconds=1)).strftime("%Y%m%d-%H%M%S")
      bumps += 1
    os.makedirs(os.path.join(self.root, chosen), exist_ok=False)
    self._session_name = chosen
    self._sizes.setdefault(chosen, 0)
    self._session_started = self._clock()
    self._write_heartbeat(force=True)

  def _write_heartbeat(self, force: bool = False) -> None:
    name = self._session_name
    if name is None:
      return
    now = self._clock()
    if not force and now - self._last_heartbeat < ACTIVE_HEARTBEAT_S:
      return
    try:
      tmp_path = os.path.join(self.root, ACTIVE_NAME + ".tmp")
      with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(name)
      os.replace(tmp_path, os.path.join(self.root, ACTIVE_NAME))
      self._last_heartbeat = now
    except OSError as exc:
      self._log_error(exc)

  def _log_error(self, exc: Exception) -> None:
    now = self._clock()
    if now - self._last_error_log >= ERROR_LOG_INTERVAL_S:
      self._last_error_log = now
      print(f"tlvrec: writer error: {exc!r}", flush=True)

  # -- storage caps -----------------------------------------------------------

  def _poll_storage(self) -> None:
    now = self._clock()
    free = self._free_disk_bytes()
    if free is not None:
      if self._paused == "disk_low":
        if free >= self._disk_rearm_free_bytes:
          self._paused = None
      elif free < self._disk_min_free_bytes:
        self._paused = "disk_low"
    if now - self._last_scan >= self._rescan_interval_s:
      self._last_scan = now
      self._rescan()
    self._enforce_caps()
    if self.active:
      self._write_heartbeat()

  def _free_disk_bytes(self) -> int | None:
    fn = self._statvfs_fn if self._statvfs_fn is not None else getattr(os, "statvfs", None)
    if fn is None:
      return None
    path = self.root
    while path and not os.path.exists(path):
      parent = os.path.dirname(path)
      if parent == path:
        path = "."
        break
      path = parent
    try:
      st = fn(path or ".")
      return int(st.f_bavail) * int(st.f_frsize)
    except Exception:
      return None

  def _rescan(self) -> None:
    sizes: dict[str, int] = {}
    try:
      names = os.listdir(self.root)
    except OSError:
      return
    for name in names:
      if not SESSION_NAME_RE.match(name):
        continue
      dir_path = os.path.join(self.root, name)
      if not os.path.isdir(dir_path):
        continue
      total = 0
      for base, _dirs, files in os.walk(dir_path):
        for fname in files:
          try:
            total += os.path.getsize(os.path.join(base, fname))
          except OSError:
            continue
      sizes[name] = total
    self._sizes = sizes
    self._bytes = sum(sizes.values())

  def _enforce_caps(self) -> None:
    if self._bytes <= self._max_root_bytes:
      return
    current = self._session_name
    for name in sorted(self._sizes):
      if self._bytes <= self._max_root_bytes:
        break
      if name == current:
        continue
      freed = self._sizes.pop(name, 0)
      shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)
      self._bytes = max(0, self._bytes - freed)
    if current is not None and self._sizes.get(current, 0) > self._max_root_bytes:
      self._paused = "size_cap"

  # -- lifecycle / introspection ----------------------------------------------

  def stats(self) -> dict:
    return {
      "saved": self.saved,
      "dropped_full": self.dropped_full,
      "dropped_rate": self.dropped_rate,
      "bytes": self._bytes,
      "paused": self._paused,
      "session": self._session_name,
    }

  def wait_idle(self, timeout_s: float = 5.0) -> bool:
    """Test helper: wait until the queue drained and no slot is in flight."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
      if self._queue.empty() and not any(self._busy):
        return True
      time.sleep(0.005)
    return False

  def close(self, timeout_s: float = 1.0) -> None:
    """Drain the queue with a bounded join; safe to call more than once."""
    self._stop.set()
    try:
      self._queue.put_nowait(None)
    except queue.Full:
      pass
    thread = self._thread
    if thread is not None and thread.is_alive():
      thread.join(timeout_s)
    name = self._session_name
    if name is None:
      return
    try:
      active_path = os.path.join(self.root, ACTIVE_NAME)
      with open(active_path, encoding="utf-8") as f:
        content = f.read()
      if content == name:
        os.remove(active_path)
    except (OSError, ValueError):
      pass
