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

from openpilot.selfdrive.carrot.tlv import crops, refine, tracker

# CONTRACT.md "High-speed approach" view scheduling thresholds (km/h).
FAR_RATE_V_KPH = 50.0
NEAR_RATE_V_KPH = 20.0
STANDSTILL_V_KPH = 5.0
MAX_CLASSIFY_TRACKS = crops.MAX_LAMP_CROPS
# W153: at standstill the camera does not move, so the ego-motion predicted box
# equals the stored box and the W134 Addendum C "verified on this frame" rule
# cannot protect against drift that does not exist. After a red->green change
# the lit spot moves inside the housing, so the NCC refine fails and the track
# gets no vote until the next detector match (~1 s). This lets the relevant
# track vote on its LAST-VERIFIED box instead.
#
# The vote is recorded ONLY when it is green-family at high confidence. Any
# other class is discarded: a stale-box red/OFF vote would count as "a
# red-family vote newer than run_start" and break the W150b sticky release,
# which would DELAY starts versus today. Discarding it also makes a passing
# vehicle (which can only produce a non-green class) harmless.
STANDSTILL_BOX_KEEP_S = 3.0
STANDSTILL_BOX_MIN_S = 0.5
STANDSTILL_BOX_VOTE_PROB = 0.80
# W134 Addendum C: the real-frame device CPU estimate for 4 refines (~12.9 ms)
# exceeds the 12 ms budget, so refine only the 2 relevance-first tracks.
REFINE_TRACKS_2 = True

# W139c red continuity: while the machine state stays red/redLeft, a strong red
# vote on the current relevant track can keep the red alive for up to this long
# after the last CONFIRMED red/published 1/5 (the fast classifier votes on the
# new relevant track faster than the state machine re-confirms it, e.g. after a
# relevance switch). It never creates red from none and never blocks a
# green/yellow/left transition (the machine still receives those).
RED_SUSTAIN_MAX_S = 10.0

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


# A lamp vote may only be GREEN/GREEN_LEFT when the green family holds at least this
# probability mass; otherwise the vote falls back to the best non-green class.
# W110: r9 on cleaned val -> red->green 0/894 at 0.5 (green recall 95.1 %).
GREEN_VOTE_MIN_PROB = 0.5
_GREEN_IDX = (int(tracker.LampClass.GREEN), int(tracker.LampClass.GREEN_LEFT))
_NON_GREEN_IDX = tuple(i for i in range(8) if i not in _GREEN_IDX)


def vote_class(row) -> int:
  """Lamp vote class for one softmax row (8,), with the green-mass gate."""
  cls = int(np.argmax(row))
  if cls in _GREEN_IDX and float(row[_GREEN_IDX[0]] + row[_GREEN_IDX[1]]) < GREEN_VOTE_MIN_PROB:
    best = _NON_GREEN_IDX[0]
    for i in _NON_GREEN_IDX[1:]:
      if row[i] > row[best]:
        best = i
    cls = best
  return cls


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
  # W150: this frame released a held red (state machine went red -> none).
  released: bool = False
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
    # W136: the relevant track's ego-predicted full-frame box for THIS frame
    # (None when there is no relevant track). Unlike dets[0] it never falls back
    # to an unrelated detection. Ego is already updated for this frame before
    # infer() runs (tlvd._update_odometry then FramePipeline.ego is set).
    self.last_relevant_box: tuple[float, float, float, float] | None = None
    self.last_red_likely = False
    # W175: monotonic time of the last publish of a red-family value (redLikely
    # or state red/redLeft); gates the onset-strict red rules off mid-red.
    self._last_red_pub_s: float | None = None
    # W150: the last infer() released a held red (see Track.released()).
    self.last_released = False
    # W150b: sticky release state -- the track it applies to and the first
    # vote of the run that released it.
    self.release_sticky = False
    self.release_track_id = -1
    self.release_run_start = 0.0
    # W139c: monotonic time of the last confirmed red-family relevant track, and
    # cumulative frames a confirmed red was sustained by a red vote on the new
    # relevant track (telemetry), and the relevant track's detector support.
    self._last_red_confirm_s: float | None = None
    self.red_sustain_total = 0
    self.last_relevant_det_support = 0
    # Per-frame ego snapshot (tracker.EgoFrame) set by tlvd; None keeps the
    # legacy no-prediction behaviour (PC tools/tests).
    self.ego: tracker.EgoFrame | None = None
    self._classifier_probs = np.empty((MAX_CLASSIFY_TRACKS, 8), dtype=np.float32)
    # W134 Addendum A: full-resolution luma copy + NCC template re-localisation.
    # `luma` is a uint8[height, width] copy of the Y plane of the CURRENT frame,
    # owned here and never published/recorded; reallocated only on size change.
    self.luma: np.ndarray | None = None
    self.relocalizer = refine.TemplateRelocalizer()
    self.refine_ms = 0.0
    self.refine_locks = 0
    # Track IDs whose position on THIS frame is verified (detector match or
    # accepted refine). Only these get a lamp crop and a vote (W134 Addendum C).
    self._verified_ids: set[int] = set()
    # W153: monotonic time the ego has been continuously at standstill (None
    # when moving or when the speed is unknown), the resulting duration, and
    # the tracks voting on a stale box this frame (non-green votes discarded).
    self._standstill_since: float | None = None
    self._standstill_s = 0.0
    self._stale_box_ids: set[int] = set()
    self.standstill_box_votes = 0

  def capture_luma(self, y, width: int) -> None:
    """Copy the full-resolution luma of this frame (y is [height, stride])."""
    height = y.shape[0]
    if self.luma is None or self.luma.shape != (height, width):
      self.luma = np.empty((height, width), dtype=np.uint8)
    np.copyto(self.luma, y[:, :width])

  def _apply_refined(self, track, box, now: float) -> None:
    """Adopt a refined box and re-anchor the odometry to this frame."""
    track.box = box
    if self.ego is not None:
      track.anchor_box = box
      track.anchor_s = self.ego.odo.s
      track.anchor_psi = self.ego.odo.psi
      track.anchor_t = float(now)
    track.last_verified_s = float(now)

  def _relocalize(self, now: float, detect: bool) -> None:
    """NCC template re-localisation (W134 Addendum A 2A).

    Detect frames: refresh the template of every matched/created track from its
    detector box, then refine the tracks the detector did NOT match. Classify-only
    frames: refine up to MAX_REFINE_TRACKS relevance-first tracks. Refinement
    never refreshes a template and never resets the expiry counters, so a track
    with no detector match for > 2.0 s still expires by the existing rules.
    """
    started = time.perf_counter()
    rel = self.relocalizer
    # W134 Addendum C: a detector-matched box IS verified on this frame, even
    # without cv2/luma, so a cv2-less device still votes on detect frames. Only
    # the classify-only frames depend on an accepted refine.
    matched_ids = {tid for _det, tid in self.tracker.last_matches} if detect else set()
    self._verified_ids = set(matched_ids)
    self.refine_locks = 0
    if rel.enabled and self.luma is not None and self.tracker.tracks:
      rel.prune([t.track_id for t in self.tracker.tracks])
      if detect:
        for track in self.tracker.tracks:
          if track.track_id in matched_ids:
            rel.refresh(self.luma, track)
      locks = 0
      limit = rel.MAX_REFINE_TRACKS_2 if REFINE_TRACKS_2 else rel.MAX_REFINE_TRACKS
      for track in self._classify_order():
        if locks >= limit:
          break
        if track.track_id in matched_ids:
          continue
        new_box = rel.refine(self.luma, track, self._predicted_box(track))
        if new_box is not None:
          self._apply_refined(track, new_box, now)
          self._verified_ids.add(track.track_id)
          locks += 1
      self.refine_locks = locks
    self.refine_ms = (time.perf_counter() - started) * 1000.0
    # W153: a classify-only frame at standstill lets the relevant track vote on
    # its last verified box when the refine missed it (the lit spot moved inside
    # the housing on a red->green change). The camera does not move, so the
    # predicted box IS the stored box; nothing is refreshed -- not the box, the
    # template, the anchor or the expiry counters.
    self._stale_box_ids = set()
    if not detect and self._standstill_s >= STANDSTILL_BOX_MIN_S and self.last_track_id >= 0:
      track = self.tracker.track_by_id(self.last_track_id)
      if (track is not None
          and track.track_id not in self._verified_ids
          and track.last_verified_s is not None
          and now - track.last_verified_s <= STANDSTILL_BOX_KEEP_S):
        self._verified_ids.add(track.track_id)
        self._stale_box_ids.add(track.track_id)

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

  @property
  def single_class_detector(self) -> bool:
    return getattr(self.decoder, "num_classes", None) == 1

  @property
  def far_tracks_only(self) -> bool:
    """True when every live track is beyond the near gate (W121 L2)."""
    if not self.tracker.tracks:
      return False
    odo = self.ego.odo if self.ego is not None else None
    for track in self.tracker.tracks:
      d = tracker.track_distance_m(track, odo)
      if not (math.isfinite(d) and d > tracker.RELEVANT_MAX_DISTANCE_M):
        return False
    return True

  def _confirmed(self, now: float, onset_strict: bool = False) -> list[tuple[tracker.Track, int, float]]:
    """Per-track confirmation: classifier votes OR the detector M-of-N belt.

    onset_strict (W175) is passed to the classifier vote rules only.
    """
    if self.single_class_detector:
      # Single-class detector (class 0 for every lamp): the detector M-of-N belt
      # would publish RED for green lights (W115 A6), so only lamp-classifier
      # votes may confirm. No classifier/NullClassifier -> nothing confirms.
      out = []
      for track in self.tracker.tracks:
        result = track.classifier_confirmed(now, onset_strict=onset_strict)
        if result is not None:
          out.append((track, result[0], result[1]))
      self.tracker.note_red_confirmations(now, out)
      return out
    if self.classifier is None:
      out = self.tracker.confirmed(self.frame_idx)
    else:
      out: list[tuple[tracker.Track, int, float]] = []
      for track in self.tracker.tracks:
        result = self.tracker.confirm_track(track, self.frame_idx, now, use_classifier=True,
                                            onset_strict=onset_strict)
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
    """Re-classify up to 4 tracked lights from THIS frame, any view.

    W134 Addendum C (safety): a track is classified ONLY if its box is verified
    on this frame (detector match OR accepted refine). A rejected/skipped refine
    gets no crop and no vote, so a lamp colour is never read from a box whose
    position was not confirmed this frame.
    """
    classifier = self.classifier
    if classifier is None or not self.tracker.tracks:
      return
    ordered = [t for t in self._classify_order() if t.track_id in self._verified_ids]
    if not ordered:
      return
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
      cls = vote_class(row)
      prob = float(row[cls])
      if ordered[i].track_id in self._stale_box_ids:
        # W153: a standstill stale-box vote is recorded ONLY when it is a
        # confident green. Red/yellow/OFF/low-p votes are dropped: a stale red
        # would count as "a red-family vote newer than run_start" and break the
        # W150b sticky release (delaying the start), and a passing vehicle can
        # only produce a non-green class.
        if cls not in tracker.GREEN_RELEASE_CLASSES or prob < STANDSTILL_BOX_VOTE_PROB:
          continue
        self.standstill_box_votes += 1
      votes.append((ordered[i].track_id, cls, prob))
    self.tracker.record_votes(now, votes)

  def infer(self, view: str, geom: crops.ViewGeometry, *, image_width: int, path_x=(), path_y=(),
            left_intent: bool = False, now: float | None = None, detect: bool = True,
            horizon_y: float | None = None, gate_enabled: bool = False,
            v_ego_kph: float | None = None) -> FrameOutput:
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
    # W153: how long the ego has been continuously at standstill (unknown speed
    # is never standstill). Only the classify-only stale-box vote uses this.
    if v_ego_kph is not None and float(v_ego_kph) < STANDSTILL_V_KPH:
      if self._standstill_since is None:
        self._standstill_since = float(now)
      self._standstill_s = max(0.0, float(now) - self._standstill_since)
    else:
      self._standstill_since = None
      self._standstill_s = 0.0
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
      # dets are full-frame; the tracker needs the FULL-frame focal, not the
      # crop's model-space focal (geom.focal_px = focal / geom.scale).
      self.tracker.update(self.frame_idx, dets, view, geom.focal_px * geom.scale, now=now, predict=predict,
                          anchor=anchor, crop_box=crop_box, frame_box=frame_box)
    # W134 Addendum A: re-locate tracks between detects (and refine un-matched
    # tracks on a detect) BEFORE the lamp crops so _classify_tracks uses them.
    self._relocalize(now, detect)
    self._classify_tracks(now, geom)

    live = [(t, int(t.cls), float(t.score)) for t in self.tracker.tracks
            if int(t.cls) != int(tracker.LampClass.OFF)]
    # W175: tlvd republishes the last infer's state every loop, so a red-family
    # result counts as published right up to this infer.
    if self.last_red_likely or self.machine.state in (tracker.SignalState.red, tracker.SignalState.redLeft):
      self._last_red_pub_s = now
    # W175: onset-strict red only for a NEW red (machine not red, no red-family
    # publish in the last tracker.ONSET_GATE_S).
    onset_strict = tracker.onset_strict_active(self.machine.state, self._last_red_pub_s, now)
    confirmed_relevant = tracker.select_relevant_tiered(
      self._confirmed(now, onset_strict), self.frame_idx, near_candidates=live,
      image_width=image_width,
      path_x=path_x, path_y=path_y, left_intent=left_intent,
      odo=self.ego.odo if self.ego is not None else None,
    )
    relevant = confirmed_relevant
    lamp_state = tracker.lamp_state_for(confirmed_relevant[1]) if confirmed_relevant is not None else None
    machine_conf = confirmed_relevant[2] if confirmed_relevant is not None else 0.0
    red_likely = False
    if self.classifier is not None:
      if confirmed_relevant is not None:
        red_likely = (int(confirmed_relevant[1]) not in tracker.RED_FAMILY
                      and confirmed_relevant[0].classifier_red_likely(now, onset_strict=onset_strict))
      else:
        # Candidate-only frame: relevant selection for crop ordering and the
        # graded redLikely state; the state machine stays on confirmed input.
        candidate = tracker.select_relevant_tiered(
          live, self.frame_idx, near_candidates=live, image_width=image_width,
          path_x=path_x, path_y=path_y, left_intent=left_intent,
          odo=self.ego.odo if self.ego is not None else None,
        )
        if candidate is not None:
          relevant = candidate
          red_likely = candidate[0].classifier_red_likely(now, onset_strict=onset_strict)

    # W139c red continuity: remember when the relevant track last confirmed red.
    if confirmed_relevant is not None and int(confirmed_relevant[1]) in tracker.RED_FAMILY:
      self._last_red_confirm_s = now
    # While the machine is red and the red was confirmed recently, a strong red
    # vote on the current relevant candidate refreshes the lamp so the 2 s off
    # timer does not expire during a relevance switch. This only refreshes an
    # already-red machine; it never creates red from none.
    if (lamp_state is None and self.machine.state in (tracker.SignalState.red, tracker.SignalState.redLeft)
        and self._last_red_confirm_s is not None and now - self._last_red_confirm_s <= RED_SUSTAIN_MAX_S
        and relevant is not None and relevant[0].classifier_red_likely(now, onset_strict=onset_strict)):
      lamp_state = self.machine.state
      machine_conf = self.last_conf  # refresh only
      self.red_sustain_total += 1
    self.last_relevant_det_support = relevant[0].det_support(now) if relevant is not None else 0
    # W150: report a release of a held red when the newest GREEN_RELEASE_VOTES
    # relevant-track votes are green at prob >= GREEN_RELEASE_PROB
    # (tracker.GREEN_RELEASE_*). This does NOT touch the state machine: the
    # published state stays red, so without a model cue the car keeps today's
    # off_timeout + 0.8 s hold exactly. carrotServ decides, per publish and only
    # while the model start cue is active, whether that red is lifted at once.
    #
    # W150b: the release is STICKY. A sub-threshold green vote or votes ageing
    # out of the deque breaks the GREEN_RELEASE_VOTES-vote run, which used to
    # drop the flag while the machine was still red -- carrotServ then unlatched
    # and the red came back (a start-then-brake jerk). Once a run has been
    # released it holds until a red-family vote arrives on the same track, the
    # track changes, the machine leaves red, or a green is confirmed.
    green_confirmed = lamp_state in (tracker.SignalState.green, tracker.SignalState.greenLeft,
                                     tracker.SignalState.left)
    machine_red = self.machine.state in (tracker.SignalState.red, tracker.SignalState.redLeft)
    run_start = relevant[0].release_run_start() if relevant is not None else None
    if run_start is not None and machine_red and not green_confirmed:
      if (not getattr(self, "release_sticky", False)
          or self.release_track_id != relevant[0].track_id):
        self.release_track_id = relevant[0].track_id
        self.release_run_start = run_start
      self.release_sticky = True
    sticky_ok = bool(
      getattr(self, "release_sticky", False)
      and relevant is not None
      and relevant[0].track_id == self.release_track_id
      and machine_red
      and not green_confirmed
      and not relevant[0].red_vote_since(self.release_run_start))
    if not sticky_ok:
      self.release_sticky = False
      self.release_track_id = -1
      self.release_run_start = 0.0
    self.last_released = sticky_ok
    state, conf = self.machine.update(now, lamp_state, machine_conf)

    if detect:
      self.last_dets = dets
    else:
      self.last_gated = []
    self.last_track_id = relevant[0].track_id if relevant is not None else -1
    # W136: the relevant track's ego-predicted full-frame box for this frame.
    self.last_relevant_box = self._predicted_box(relevant[0]) if relevant is not None else None
    self.last_state, self.last_conf = state, conf
    self.last_red_likely = red_likely
    self.frame_idx += 1
    self.infer_ms = infer_ms
    self.proc_ms = (time.perf_counter() - started) * 1000.0
    return FrameOutput(
      view=view, dets=dets, matches=self.tracker.last_matches, state=state, state_conf=conf,
      track_id=self.last_track_id, infer_ms=infer_ms, proc_ms=self.proc_ms, red_likely=red_likely,
      released=self.last_released,
      gated=self.last_gated,
    )

  def process(self, data, width: int, height: int, stride: int, uv_offset: int, geom: crops.ViewGeometry, **infer_kwargs) -> FrameOutput:
    """prepare() + infer() convenience (tests/soak); tlvd splits them."""
    view = infer_kwargs.pop("view", None)
    if view is None:
      view = self.next_view()
    self.prepare(data, width, height, stride, uv_offset, geom)
    return self.infer(view, geom, **infer_kwargs)
