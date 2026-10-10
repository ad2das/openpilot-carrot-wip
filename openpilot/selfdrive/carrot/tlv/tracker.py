"""Pure traffic-light tracking and signal-state logic for the carrot TLV detector.

No openpilot messaging imports, so everything here is unit-testable on a PC.
Coordinates are full-frame road-camera pixels (1928x1208 on a C3X); distances are
metres. Class ids follow CONTRACT.md:

  0 RED, 1 YELLOW, 2 GREEN, 3 RED_LEFT, 4 GREEN_LEFT, 5 RED_YELLOW, 6 LEFT, 7 OFF
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

HOUSING_HEIGHT_M = 0.35


class LampClass(IntEnum):
  RED = 0
  YELLOW = 1
  GREEN = 2
  RED_LEFT = 3
  GREEN_LEFT = 4
  RED_YELLOW = 5
  LEFT = 6
  OFF = 7


class SignalState(IntEnum):
  none = 0
  red = 1
  yellow = 2
  green = 3
  left = 4
  redLeft = 5
  greenLeft = 6
  # Graded state (CONTRACT.md High-speed approach): one strong red detection on
  # the relevant track, not yet confirmed. Advisory only; never a lamp class.
  redLikely = 7


RED_FAMILY = (LampClass.RED, LampClass.RED_LEFT, LampClass.RED_YELLOW)
GREEN_FAMILY = (LampClass.GREEN, LampClass.GREEN_LEFT)
ARROW_CLASSES = (LampClass.RED_LEFT, LampClass.GREEN_LEFT, LampClass.LEFT)
SLOW_CLASSES = (LampClass.YELLOW, LampClass.GREEN, LampClass.GREEN_LEFT, LampClass.LEFT)

# Lamp-classifier confirmation counts VOTES BY TIME, not frames (CONTRACT.md).
# Red is confirmed fast but must not override a conflicting green vote in the
# same window; green/yellow/left keep the stricter 6-in-1.2 s rule.
CLASSIFIER_RED_WINDOW_S = 0.6
CLASSIFIER_RED_VOTES = 3
CLASSIFIER_SLOW_WINDOW_S = 1.2
CLASSIFIER_SLOW_VOTES = 6
# Graded redLikely: one strong red vote (probability >= 0.6) is enough.
RED_LIKELY_MIN_PROB = 0.6
# Classifier votes keep a track alive this long after the last detection, so a
# slow detector cadence cannot drop a light the fast classifier still sees.
VOTE_TRACK_KEEP_S = 0.75

# (window frames, required hits). Asymmetric on purpose: red is confirmed fast
# (safety stop), green is confirmed slowly (never start on a flicker).
CONFIRM_WINDOWS: dict[LampClass, tuple[int, int]] = {
  LampClass.RED: (4, 3),
  LampClass.RED_LEFT: (4, 3),
  LampClass.RED_YELLOW: (4, 3),
  LampClass.YELLOW: (6, 4),
  LampClass.GREEN: (8, 6),
  LampClass.GREEN_LEFT: (8, 6),
  LampClass.LEFT: (8, 6),
  LampClass.OFF: (6, 4),
}

LAMP_TO_STATE = {
  LampClass.RED: SignalState.red,
  LampClass.RED_YELLOW: SignalState.red,
  LampClass.RED_LEFT: SignalState.redLeft,
  LampClass.YELLOW: SignalState.yellow,
  LampClass.GREEN: SignalState.green,
  LampClass.GREEN_LEFT: SignalState.greenLeft,
  LampClass.LEFT: SignalState.left,
}


def lamp_state_for(cls: int) -> SignalState | None:
  """Signal state for a confirmed lamp class; None means "no evidence"."""
  try:
    return LAMP_TO_STATE.get(LampClass(int(cls)))
  except ValueError:
    return None


@dataclass
class Detection:
  cls: int
  score: float
  x1: float
  y1: float
  x2: float
  y2: float

  @property
  def cx(self) -> float:
    return 0.5 * (self.x1 + self.x2)

  @property
  def cy(self) -> float:
    return 0.5 * (self.y1 + self.y2)

  @property
  def width(self) -> float:
    return max(0.0, self.x2 - self.x1)

  @property
  def height(self) -> float:
    return max(0.0, self.y2 - self.y1)

  @property
  def box(self) -> tuple[float, float, float, float]:
    return self.x1, self.y1, self.x2, self.y2


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
  ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
  ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
  iw, ih = ix2 - ix1, iy2 - iy1
  if iw <= 0.0 or ih <= 0.0:
    return 0.0
  inter = iw * ih
  area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
  area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
  union = area_a + area_b - inter
  return inter / union if union > 0.0 else 0.0


def _box_diag(box: tuple[float, float, float, float]) -> float:
  return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _center_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
  return float(np.hypot(0.5 * (a[0] + a[2]) - 0.5 * (b[0] + b[2]), 0.5 * (a[1] + a[3]) - 0.5 * (b[1] + b[3])))


def match_score(a: tuple[float, float, float, float], b: tuple[float, float, float, float],
                 center_factor: float = 0.35) -> float:
  """Association score in [0, 1]: IoU first, centre proximity as fallback."""
  overlap = iou(a, b)
  if overlap > 0.0:
    return overlap
  limit = center_factor * max(_box_diag(a), _box_diag(b), 1.0)
  dist = _center_distance(a, b)
  if dist < limit:
    return 0.1 * (1.0 - dist / limit)
  return 0.0


@dataclass
class Track:
  track_id: int
  first_frame: int
  last_frame: int
  cls: int
  score: float
  box: tuple[float, float, float, float]
  view: str
  focal_px: float
  hits: int = 1
  misses: int = 0
  history: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=64))
  # Lamp-classifier votes: (monotonic time, class, probability), bounded deque.
  votes: deque[tuple[float, int, float]] = field(default_factory=lambda: deque(maxlen=32))

  @property
  def cx(self) -> float:
    return 0.5 * (self.box[0] + self.box[2])

  @property
  def cy(self) -> float:
    return 0.5 * (self.box[1] + self.box[3])

  @property
  def height(self) -> float:
    return max(0.0, self.box[3] - self.box[1])

  def evidence(self, frame_idx: int, cls: int, window: int) -> int:
    return sum(1 for f, c in self.history if c == cls and f > frame_idx - window)

  def confirmed(self, frame_idx: int) -> tuple[int, float] | None:
    """(cls, confidence) once the class meets its M-of-N window, else None."""
    best: tuple[int, float] | None = None
    for cls, (window, needed) in CONFIRM_WINDOWS.items():
      hits = self.evidence(frame_idx, cls, window)
      if hits >= needed:
        conf = hits / window
        if best is None or conf > best[1]:
          best = (int(cls), conf)
    return best

  def classifier_confirmed(self, now: float) -> tuple[int, float] | None:
    """(cls, confidence) from classifier votes counted by TIME, else None.

    Red family: >= 3 votes within the last 0.6 s and no green-family vote in
    that window. Yellow/green/left: the stricter >= 6 votes in 1.2 s. Red is
    evaluated first so a stale green cannot delay a safety stop.
    """
    red_votes = [c for t, c, _ in self.votes if t > now - CLASSIFIER_RED_WINDOW_S and c in RED_FAMILY]
    green_votes = [c for t, c, _ in self.votes if t > now - CLASSIFIER_RED_WINDOW_S and c in GREEN_FAMILY]
    if len(red_votes) >= CLASSIFIER_RED_VOTES and not green_votes:
      return int(_most_common(red_votes)), min(1.0, len(red_votes) / CLASSIFIER_RED_VOTES)
    best: tuple[int, float] | None = None
    for cls in SLOW_CLASSES:
      hits = sum(1 for t, c, _ in self.votes if t > now - CLASSIFIER_SLOW_WINDOW_S and c == cls)
      if hits >= CLASSIFIER_SLOW_VOTES:
        conf = min(1.0, hits / CLASSIFIER_SLOW_VOTES)
        if best is None or conf > best[1]:
          best = (int(cls), conf)
    return best

  def classifier_red_likely(self, now: float) -> bool:
    """True when the LATEST vote is a strong red vote inside the red window."""
    if not self.votes:
      return False
    t, cls, prob = self.votes[-1]
    if t <= now - CLASSIFIER_RED_WINDOW_S:
      return False
    return cls in RED_FAMILY and prob >= RED_LIKELY_MIN_PROB


def _most_common(values: list[int]) -> int:
  counts: dict[int, int] = {}
  for value in values:
    counts[value] = counts.get(value, 0) + 1
  return max(counts, key=lambda value: (counts[value], value))


class TrafficLightTracker:
  """IoU + centre tracker across views, in full-frame coordinates."""

  def __init__(self, center_factor: float = 0.35, max_misses: int = 15) -> None:
    self.center_factor = center_factor
    self.max_misses = max_misses
    self.tracks: list[Track] = []
    self.last_matches: list[tuple[Detection, int]] = []
    self._next_id = 1

  def update(self, frame_idx: int, detections: list[Detection], view: str, focal_px: float,
             now: float | None = None) -> list[tuple[Detection, int]]:
    pairs: list[tuple[float, int, int]] = []
    for di, det in enumerate(detections):
      for ti, track in enumerate(self.tracks):
        score = match_score(track.box, det.box, self.center_factor)
        if score > 0.0:
          pairs.append((score, di, ti))
    pairs.sort(key=lambda p: p[0], reverse=True)

    det_to_track: dict[int, int] = {}
    used_tracks: set[int] = set()
    for _, di, ti in pairs:
      if di in det_to_track or ti in used_tracks:
        continue
      det_to_track[di] = ti
      used_tracks.add(ti)

    matched_track_ids: set[int] = set()
    self.last_matches = []
    for di, det in enumerate(detections):
      if di in det_to_track:
        track = self.tracks[det_to_track[di]]
        track.cls = int(det.cls)
        track.score = float(det.score)
        track.box = det.box
        track.view = view
        track.focal_px = focal_px
        track.last_frame = frame_idx
        track.hits += 1
        track.misses = 0
        track.history.append((frame_idx, int(det.cls)))
        matched_track_ids.add(track.track_id)
        self.last_matches.append((det, track.track_id))
      else:
        track = Track(
          track_id=self._next_id, first_frame=frame_idx, last_frame=frame_idx, cls=int(det.cls),
          score=float(det.score), box=det.box, view=view, focal_px=focal_px,
        )
        track.history.append((frame_idx, int(det.cls)))
        self._next_id += 1
        self.tracks.append(track)
        matched_track_ids.add(track.track_id)
        self.last_matches.append((det, track.track_id))

    for track in self.tracks:
      if track.track_id not in matched_track_ids:
        track.misses += 1
    self.tracks = [t for t in self.tracks if self._alive(t, frame_idx, now)]
    return self.last_matches

  def _alive(self, track: Track, frame_idx: int, now: float | None) -> bool:
    if frame_idx - track.last_frame <= self.max_misses:
      return True
    return now is not None and bool(track.votes) and now - track.votes[-1][0] <= VOTE_TRACK_KEEP_S

  def confirmed(self, frame_idx: int) -> list[tuple[Track, int, float]]:
    out: list[tuple[Track, int, float]] = []
    for track in self.tracks:
      result = track.confirmed(frame_idx)
      if result is not None:
        cls, conf = result
        out.append((track, cls, conf))
    return out

  def record_votes(self, now: float, votes: list[tuple[int, int, float]]) -> None:
    """Append lamp-classifier votes: (track_id, class, probability)."""
    by_id = {track.track_id: track for track in self.tracks}
    for track_id, cls, prob in votes:
      track = by_id.get(int(track_id))
      if track is not None:
        track.votes.append((float(now), int(cls), float(prob)))

  def confirm_track(self, track: Track, frame_idx: int, now: float,
                    use_classifier: bool) -> tuple[int, float] | None:
    """Best confirmation for one track: classifier votes OR the detector belt.

    Red-family results win over non-red ones (safety), otherwise the higher
    confidence wins. With no classifier this is exactly track.confirmed().
    """
    vote = track.classifier_confirmed(now) if use_classifier else None
    det = track.confirmed(frame_idx)
    candidates = [c for c in (vote, det) if c is not None]
    if not candidates:
      return None
    red = [c for c in candidates if int(c[0]) in RED_FAMILY]
    pool = red if red else candidates
    return max(pool, key=lambda c: float(c[1]))

  def track_by_id(self, track_id: int) -> Track | None:
    for track in self.tracks:
      if track.track_id == track_id:
        return track
    return None


def estimate_distance_m(box_height_px: float, focal_px: float, housing_m: float = HOUSING_HEIGHT_M) -> float:
  """Distance from box height and the pinhole model; inf for unusable boxes."""
  if box_height_px <= 0.5 or focal_px <= 0.0:
    return float("inf")
  return focal_px * housing_m / box_height_px


def lateral_offset_m(center_x_px: float, image_width: int, focal_px: float, distance_m: float) -> float:
  """Lateral (left-positive? camera axis offset) position in metres at distance_m."""
  return (center_x_px - 0.5 * image_width) * distance_m / focal_px


def path_lateral_m(distance_m: float, path_x, path_y) -> float:
  """Model path lateral offset at distance_m; 0.0 when the path is unusable."""
  xs = np.asarray(path_x, dtype=np.float64)
  ys = np.asarray(path_y, dtype=np.float64)
  if xs.size < 2 or ys.size != xs.size or xs[-1] <= xs[0] or not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
    return 0.0
  return float(np.interp(distance_m, xs, ys))


def select_relevant(confirmed: list[tuple[Track, int, float]], frame_idx: int, *,
                    image_width: int, path_x, path_y, left_intent: bool,
                    max_lateral_m: float = 4.0, max_distance_m: float = 120.0,
                    min_distance_m: float = 2.0) -> tuple[Track, int, float] | None:
  """Track nearest to the projected model path at its estimated distance.

  With left intent, arrow-bearing tracks are preferred over the plain ones when
  both are plausible (the turn arrow is the light that matters for the path).
  """
  candidates: list[tuple[float, float, Track, int, float]] = []
  arrows: list[tuple[float, float, Track, int, float]] = []
  for track, cls, conf in confirmed:
    if cls == LampClass.OFF:
      continue
    distance = estimate_distance_m(track.height, track.focal_px)
    if not (min_distance_m <= distance <= max_distance_m):
      continue
    lateral = lateral_offset_m(track.cx, image_width, track.focal_px, distance)
    error = abs(lateral - path_lateral_m(distance, path_x, path_y))
    if error > max_lateral_m:
      continue
    key = (error, distance, track, cls, conf)
    candidates.append(key)
    if cls in ARROW_CLASSES:
      arrows.append(key)
  pool = arrows if (left_intent and arrows) else candidates
  if not pool:
    return None
  best = min(pool, key=lambda entry: (entry[0], entry[1]))
  return best[2], best[3], best[4]


class SignalStateMachine:
  """Korean signal order: green->yellow->red, red->green/left/redLeft, timeout to off."""

  def __init__(self, off_timeout_s: float = 2.0) -> None:
    self.off_timeout_s = float(off_timeout_s)
    self.state = SignalState.none
    self.conf = 0.0
    self.last_lamp_time: float | None = None

  @staticmethod
  def _allowed(cur: SignalState, new: SignalState) -> bool:
    if cur == SignalState.none:
      return True
    if cur in (SignalState.red, SignalState.redLeft):
      return new in (SignalState.red, SignalState.redLeft, SignalState.green, SignalState.greenLeft, SignalState.left)
    if cur == SignalState.yellow:
      return new in (SignalState.red, SignalState.redLeft)
    if cur in (SignalState.green, SignalState.greenLeft):
      return new in (SignalState.yellow, SignalState.green, SignalState.greenLeft)
    if cur == SignalState.left:
      return new in (SignalState.yellow, SignalState.red, SignalState.redLeft)
    return False

  def update(self, now_s: float, lamp_state: SignalState | None, conf: float = 0.0) -> tuple[SignalState, float]:
    if lamp_state is None or lamp_state == SignalState.none:
      if self.state != SignalState.none and self.last_lamp_time is not None:
        if now_s - self.last_lamp_time >= self.off_timeout_s:
          self.state = SignalState.none
          self.conf = 0.0
          self.last_lamp_time = None
      return self.state, self.conf

    if lamp_state == self.state or self._allowed(self.state, lamp_state):
      self.state = lamp_state
      self.conf = float(conf)
    # Rejected transitions (order violated) hold the current state; the lamp is
    # still present, so the off-timeout clock is refreshed.
    self.last_lamp_time = now_s
    return self.state, self.conf
