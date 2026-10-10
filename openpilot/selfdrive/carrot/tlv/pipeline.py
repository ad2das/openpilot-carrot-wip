"""Allocation-free per-frame pipeline for the carrot TLV detector.

crop -> CPU-only backend -> decoder -> full-frame mapping -> tracker -> signal
state machine. Holds only preallocated buffers (crops.FramePreprocessor) and
long-lived tracker/machine state, so it is usable directly by tlvd, the PC
allocation test and the PC soak runner.

The caller must copy the ROI out of the VisionIPC buffer with prepare() and may
then release the frame (the budget forbids holding it across inference).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from openpilot.selfdrive.carrot.tlv import crops, tracker


@dataclass
class FrameOutput:
  view: str
  dets: list[tracker.Detection]
  matches: list[tuple[tracker.Detection, int]]
  state: tracker.SignalState
  state_conf: float
  track_id: int
  infer_ms: float
  proc_ms: float


class FramePipeline:
  def __init__(self, backend, decoder) -> None:
    self.backend = backend
    self.decoder = decoder
    self.preprocess = crops.FramePreprocessor()
    self.tracker = tracker.TrafficLightTracker()
    self.machine = tracker.SignalStateMachine()
    self.frame_idx = 0
    self.infer_ms = 0.0
    self.proc_ms = 0.0
    self.last_dets: list[tracker.Detection] = []
    self.last_state = tracker.SignalState.none
    self.last_conf = 0.0
    self.last_track_id = -1

  @staticmethod
  def geometry(view: str, horizon_y: float, width: int, height: int, focal_px: float) -> crops.ViewGeometry:
    build = crops.far_geometry if view == "far" else crops.near_geometry
    return build(horizon_y, width, height, focal_px)

  def next_view(self) -> str:
    """Views alternate per processed frame: FAR, NEAR, FAR, ..."""
    return "far" if self.frame_idx % 2 == 0 else "near"

  def prepare(self, data, width: int, height: int, stride: int, uv_offset: int, geom: crops.ViewGeometry) -> np.ndarray:
    """Copy the ROI into the preallocated workspace.

    The returned array references only the workspace, never `data`, so the
    owner of `data` (e.g. a VisionIPC frame) can be released immediately.
    """
    return self.preprocess.model_input_from_frame(data, width, height, stride, uv_offset, geom)

  def infer(self, view: str, geom: crops.ViewGeometry, *, image_width: int, path_x=(), path_y=(),
            left_intent: bool = False, now: float | None = None) -> FrameOutput:
    started = time.perf_counter()
    infer_started = time.perf_counter()
    raw = self.backend.infer(self.preprocess.model_input)
    infer_ms = (time.perf_counter() - infer_started) * 1000.0

    decoded = self.decoder.decode(raw, view)
    dets: list[tracker.Detection] = []
    for det in decoded:
      x1, y1, x2, y2 = crops.map_model_box_to_full((det.x1, det.y1, det.x2, det.y2), geom)
      dets.append(tracker.Detection(cls=det.cls, score=det.score, x1=x1, y1=y1, x2=x2, y2=y2))

    if now is None:
      now = time.monotonic()
    self.tracker.update(self.frame_idx, dets, view, geom.focal_px)
    confirmed = self.tracker.confirmed(self.frame_idx)
    relevant = tracker.select_relevant(
      confirmed, self.frame_idx, image_width=image_width,
      path_x=path_x, path_y=path_y, left_intent=left_intent,
    )
    lamp_state = tracker.lamp_state_for(relevant[1]) if relevant is not None else None
    state, conf = self.machine.update(now, lamp_state, relevant[2] if relevant is not None else 0.0)

    self.last_dets = dets
    self.last_track_id = relevant[0].track_id if relevant is not None else -1
    self.last_state, self.last_conf = state, conf
    self.frame_idx += 1
    self.infer_ms = infer_ms
    self.proc_ms = (time.perf_counter() - started) * 1000.0
    return FrameOutput(
      view=view, dets=dets, matches=self.tracker.last_matches, state=state, state_conf=conf,
      track_id=self.last_track_id, infer_ms=infer_ms, proc_ms=self.proc_ms,
    )

  def process(self, data, width: int, height: int, stride: int, uv_offset: int, geom: crops.ViewGeometry, **infer_kwargs) -> FrameOutput:
    """prepare() + infer() convenience (tests/soak); tlvd splits them."""
    view = infer_kwargs.pop("view", None)
    if view is None:
      view = self.next_view()
    self.prepare(data, width, height, stride, uv_offset, geom)
    return self.infer(view, geom, **infer_kwargs)
