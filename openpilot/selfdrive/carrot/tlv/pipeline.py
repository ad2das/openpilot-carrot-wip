"""Allocation-free per-frame pipeline for the carrot TLV detector.

crop -> CPU-only backend -> decoder -> full-frame mapping -> tracker -> signal
state machine. Holds only preallocated buffers (crops.FramePreprocessor) and
long-lived tracker/machine state, so it is usable directly by tlvd, the PC
allocation test and the PC soak runner.

The caller must copy the ROI out of the VisionIPC buffer with prepare() and may
then release the frame (the budget forbids holding it across inference).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from openpilot.selfdrive.carrot.tlv import crops, tracker

# CONTRACT.md "High-speed approach" view scheduling thresholds (km/h).
FAR_RATE_V_KPH = 50.0
NEAR_RATE_V_KPH = 20.0
STANDSTILL_V_KPH = 5.0
MAX_CLASSIFY_TRACKS = crops.MAX_LAMP_CROPS

# Flat-road tail-light gate (user decision: every road is assumed FLAT). On flat
# ground anything whose whole box lies below the calibrated horizon is closer to
# the camera than the road plane at that image row, so it cannot be a traffic
# light (>= 4.5 m up); car tail lights always are below it. The 0.25 deg margin
# keeps boxes that merely touch the horizon.
TAIL_GATE_MARGIN_DEG = 0.25


class ViewScheduler:
  """Speed-aware FAR/NEAR pattern (CONTRACT.md High-speed approach).

  v >= 50: FAR on 3 of 4 inferences; v < 20: NEAR on 3 of 4; standstill
  (v < 5): NEAR every inference (the crop itself is raised, see geometry());
  otherwise the historical 1:1 alternation.
  """

  def __init__(self) -> None:
    self._frame = 0

  def next_view(self, v_ego_kph: float) -> str:
    idx = self._frame % 4
    self._frame += 1
    if v_ego_kph >= FAR_RATE_V_KPH:
      return "near" if idx == 3 else "far"
    if v_ego_kph < NEAR_RATE_V_KPH:
      if v_ego_kph < STANDSTILL_V_KPH:
        return "near"
      return "far" if idx == 3 else "near"
    return "far" if idx % 2 == 0 else "near"


def _softmax_rows_into(out: np.ndarray, logits) -> None:
  """Row softmax of `logits` (n, 8) into the preallocated `out`."""
  out[...] = logits
  out -= out.max(axis=1, keepdims=True)
  np.exp(out, out=out)
  out /= out.sum(axis=1, keepdims=True)


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
  red_likely: bool = False
  # Flat-road tail-light gate: detections dropped this frame (below the
  # horizon + margin while calibrated), full-frame boxes like dets.
  gated: list[tracker.Detection] = field(default_factory=list)


class FramePipeline:
  def __init__(self, backend, decoder, classifier=None) -> None:
    self.backend = backend
    self.decoder = decoder
    self.classifier = classifier
    self.preprocess = crops.FramePreprocessor()
    self.lamp_crops = crops.LampCropBuffer()
    self.tracker = tracker.TrafficLightTracker()
    self.machine = tracker.SignalStateMachine()
    self.scheduler = ViewScheduler()
    self.frame_idx = 0
    self.infer_ms = 0.0
    self.proc_ms = 0.0
    self.last_dets: list[tracker.Detection] = []
    self.last_gated: list[tracker.Detection] = []
    self.tail_gated_total = 0
    self.last_state = tracker.SignalState.none
    self.last_conf = 0.0
    self.last_track_id = -1
    self.last_red_likely = False
    # Per-frame ego snapshot (tracker.EgoFrame) set by tlvd; None keeps the
    # legacy no-prediction behaviour (PC tools/tests).
    self.ego: tracker.EgoFrame | None = None
    self._classifier_probs = np.empty((MAX_CLASSIFY_TRACKS, 8), dtype=np.float32)

  @staticmethod
  def geometry(view: str, horizon_y: float, width: int, height: int, focal_px: float,
               raised: bool = False) -> crops.ViewGeometry:
    if view == "far":
      return crops.far_geometry(horizon_y, width, height, focal_px)
    fraction = crops.STANDSTILL_HORIZON_CROP_FRACTION if raised else crops.HORIZON_CROP_FRACTION
    return crops.near_geometry(horizon_y, width, height, focal_px, horizon_fraction=fraction)

  def next_view(self, v_ego_kph: float | None = None) -> str:
    """Speed-aware scheduler when vEgo is given; legacy alternation otherwise."""
    if v_ego_kph is None:
      return "far" if self.frame_idx % 2 == 0 else "near"
    return self.scheduler.next_view(v_ego_kph)

  def prepare(self, data, width: int, height: int, stride: int, uv_offset: int, geom: crops.ViewGeometry) -> np.ndarray:
    """Copy the ROI into the preallocated workspace.

    The returned array references only the workspace, never `data`, so the
    owner of `data` (e.g. a VisionIPC frame) can be released immediately.
    """
    return self.preprocess.model_input_from_frame(data, width, height, stride, uv_offset, geom)

  @property
  def has_tracks(self) -> bool:
    return bool(self.tracker.tracks)

  def _confirmed(self, now: float) -> list[tuple[tracker.Track, int, float]]:
    """Per-track confirmation: classifier votes OR the detector M-of-N belt."""
    if self.classifier is None:
      out = self.tracker.confirmed(self.frame_idx)
    else:
      out: list[tuple[tracker.Track, int, float]] = []
      for track in self.tracker.tracks:
        result = self.tracker.confirm_track(track, self.frame_idx, now, use_classifier=True)
        if result is not None:
          out.append((track, result[0], result[1]))
    # Same-track green (W74D): remember red-family confirmations per track.
    self.tracker.note_red_confirmations(now, out)
    return out

  def _classify_order(self) -> list[tracker.Track]:
    """Relevant track first, then the most recently detected, max 4."""
    relevant: list[tracker.Track] = []
    rest: list[tracker.Track] = []
    for track in self.tracker.tracks:
      (relevant if track.track_id == self.last_track_id else rest).append(track)
    rest.sort(key=lambda t: t.last_frame, reverse=True)
    return (relevant + rest)[:MAX_CLASSIFY_TRACKS]

  def _predicted_box(self, track: tracker.Track) -> tuple[float, float, float, float]:
    """Ego-motion predicted full-frame box, or the stored box without ego state."""
    if self.ego is None:
      return track.box
    return tracker.predict_box(
      track, self.ego.odo, focal_px=self.ego.focal_px, foe_x=self.ego.foe_x, foe_y=self.ego.foe_y,
      image_w=self.ego.image_w, image_h=self.ego.image_h,
    )

  def classify_view(self, far_geom: crops.ViewGeometry) -> str:
    """View for a classify-only frame; NEVER advances the detect scheduler.

    "far" when the first (relevant-first) track's predicted box is fully inside
    the FAR geometry, else "near" (crops are built from this frame's geometry).
    """
    ordered = self._classify_order()
    if not ordered:
      return "near"
    x1, y1, x2, y2 = self._predicted_box(ordered[0])
    inside = (x1 >= far_geom.x0 and y1 >= far_geom.y0
              and x2 <= far_geom.x0 + far_geom.w and y2 <= far_geom.y0 + far_geom.h)
    return "far" if inside else "near"

  def _classify_tracks(self, now: float, geom: crops.ViewGeometry) -> None:
    """Re-classify up to 4 tracked lights from THIS frame, any view."""
    classifier = self.classifier
    if classifier is None or not self.tracker.tracks:
      return
    ordered = self._classify_order()
    boxes = [self._predicted_box(t) for t in ordered]
    batch, count = self.lamp_crops.build(self.preprocess.model_input, geom, boxes)
    if count == 0:
      return
    logits = classifier.classify(batch)
    if logits is None:
      return
    _softmax_rows_into(self._classifier_probs, logits)
    votes: list[tuple[int, int, float]] = []
    for i in range(min(count, len(ordered))):
      row = self._classifier_probs[i]
      cls = int(np.argmax(row))
      votes.append((ordered[i].track_id, cls, float(row[cls])))
    self.tracker.record_votes(now, votes)

  def infer(self, view: str, geom: crops.ViewGeometry, *, image_width: int, path_x=(), path_y=(),
            left_intent: bool = False, now: float | None = None, detect: bool = True,
            horizon_y: float | None = None, gate_enabled: bool = False) -> FrameOutput:
    started = time.perf_counter()
    infer_started = time.perf_counter()
    raw = self.backend.infer(self.preprocess.model_input) if detect else []
    infer_ms = (time.perf_counter() - infer_started) * 1000.0

    dets: list[tracker.Detection] = []
    if detect:
      decoded = self.decoder.decode(raw, view)
      for det in decoded:
        x1, y1, x2, y2 = crops.map_model_box_to_full((det.x1, det.y1, det.x2, det.y2), geom)
        dets.append(tracker.Detection(cls=det.cls, score=det.score, x1=x1, y1=y1, x2=x2, y2=y2))

      # Flat-road tail-light gate (calibrated only): a box whose TOP is below
      # the horizon + margin is lower than the camera on flat ground, so it
      # cannot be a >= 4.5 m traffic light; car tail lights are always below.
      if gate_enabled and horizon_y is not None:
        limit = float(horizon_y) + crops.focal_px_for_width(image_width) * math.tan(math.radians(TAIL_GATE_MARGIN_DEG))
        gated: list[tracker.Detection] = []
        kept: list[tracker.Detection] = []
        for det in dets:
          (gated if det.y1 > limit else kept).append(det)
        self.last_gated = gated
        self.tail_gated_total += len(gated)
        dets = kept
      else:
        self.last_gated = []

    if now is None:
      now = time.monotonic()
    if detect:
      predict = None
      anchor = None
      frame_box = None
      if self.ego is not None:
        ego = self.ego
        predict = lambda t: tracker.predict_box(  # noqa: E731 - hot path closure
          t, ego.odo, focal_px=ego.focal_px, foe_x=ego.foe_x, foe_y=ego.foe_y,
          image_w=ego.image_w, image_h=ego.image_h,
        )
        anchor = (ego.odo.s, ego.odo.psi, now)
        frame_box = (0.0, 0.0, float(ego.image_w), float(ego.image_h))
      crop_box = (float(geom.x0), float(geom.y0), float(geom.x0 + geom.w), float(geom.y0 + geom.h))
      self.tracker.update(self.frame_idx, dets, view, geom.focal_px, now=now, predict=predict,
                          anchor=anchor, crop_box=crop_box, frame_box=frame_box)
    self._classify_tracks(now, geom)

    confirmed_relevant = tracker.select_relevant(
      self._confirmed(now), self.frame_idx, image_width=image_width,
      path_x=path_x, path_y=path_y, left_intent=left_intent,
    )
    relevant = confirmed_relevant
    lamp_state = tracker.lamp_state_for(confirmed_relevant[1]) if confirmed_relevant is not None else None
    machine_conf = confirmed_relevant[2] if confirmed_relevant is not None else 0.0
    red_likely = False
    if self.classifier is not None:
      if confirmed_relevant is not None:
        red_likely = (int(confirmed_relevant[1]) not in tracker.RED_FAMILY
                      and confirmed_relevant[0].classifier_red_likely(now))
      else:
        # Candidate-only frame: relevant selection for crop ordering and the
        # graded redLikely state; the state machine stays on confirmed input.
        candidates = [(t, int(t.cls), float(t.score)) for t in self.tracker.tracks
                      if int(t.cls) != int(tracker.LampClass.OFF)]
        candidate = tracker.select_relevant(
          candidates, self.frame_idx, image_width=image_width,
          path_x=path_x, path_y=path_y, left_intent=left_intent,
        )
        if candidate is not None:
          relevant = candidate
          red_likely = candidate[0].classifier_red_likely(now)
    state, conf = self.machine.update(now, lamp_state, machine_conf)

    if detect:
      self.last_dets = dets
    else:
      self.last_gated = []
    self.last_track_id = relevant[0].track_id if relevant is not None else -1
    self.last_state, self.last_conf = state, conf
    self.last_red_likely = red_likely
    self.frame_idx += 1
    self.infer_ms = infer_ms
    self.proc_ms = (time.perf_counter() - started) * 1000.0
    return FrameOutput(
      view=view, dets=dets, matches=self.tracker.last_matches, state=state, state_conf=conf,
      track_id=self.last_track_id, infer_ms=infer_ms, proc_ms=self.proc_ms, red_likely=red_likely,
      gated=self.last_gated,
    )

  def process(self, data, width: int, height: int, stride: int, uv_offset: int, geom: crops.ViewGeometry, **infer_kwargs) -> FrameOutput:
    """prepare() + infer() convenience (tests/soak); tlvd splits them."""
    view = infer_kwargs.pop("view", None)
    if view is None:
      view = self.next_view()
    self.prepare(data, width, height, stride, uv_offset, geom)
    return self.infer(view, geom, **infer_kwargs)
