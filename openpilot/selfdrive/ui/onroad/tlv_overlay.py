"""Camera traffic-light (tlvd) overlay: the light the vision pipeline is following.

Draws corner brackets around carrotTlv.relevantBox (the relevant track's ego-predicted box, full-frame
road-camera pixels) and a small chip above it with the state and the box-height distance. Only the relevant light
is shown; nothing is drawn while tlvd is not running, has no relevant track, or its message is stale. carrotTlv
arrives at 10 Hz, so the screen box eases toward each update. Toggle: Params CarrotTlvShowBox.
"""
import math
import time

import numpy as np
import pyray as rl

from openpilot.selfdrive.carrot.tlv import crops, tracker
from openpilot.selfdrive.ui.onroad import hud_style as hs

# carrotTlv.State ordinals (custom.capnp): none, red, yellow, green, left, redLeft, greenLeft, redLikely.
_RED, _YELLOW, _GREEN, _LEFT, _RED_LEFT, _GREEN_LEFT, _RED_LIKELY = 1, 2, 3, 4, 5, 6, 7
_STATUS_RUNNING = 1

STALE_S = 0.6          # carrotTlv older than this hides the overlay
FADE_TAU_S = 0.12      # bracket fade in/out time constant
EASE_TAU_S = 0.08      # screen box easing between 10 Hz updates
_FOCAL = crops.focal_px_for_width(crops.REFERENCE_WIDTH)

_STYLE = {
  # state: (colour, label, bracket alpha)
  _RED: (hs.WARN_RED, "빨간불", 255),
  _RED_LEFT: (hs.WARN_RED, "빨간불", 255),
  _YELLOW: (hs.PROMPT_AMBER, "노란불", 255),
  _GREEN: (hs.LIVE_GREEN, "초록불", 255),
  _GREEN_LEFT: (hs.LIVE_GREEN, "좌회전", 255),
  _LEFT: (hs.LIVE_GREEN, "좌회전", 255),
  _RED_LIKELY: (hs.WARN_RED, "빨간불?", 150),
}
_TRACKING = (hs.WHITE, "", 200)

CHIP_H = 52.0
CHIP_TEXT = 36.0
STROKE = 5.0
OUTLINE = 9.0


class TlvOverlay:
  def __init__(self):
    self._type = hs.Type()
    self._pixel_to_screen = np.eye(3)
    self._alpha = 0.0
    self._last = None          # (box, state, track id) of the last visible frame, kept for the fade-out
    self._scr = None           # eased screen box
    self._last_t = time.monotonic()

  def set_pixel_transform(self, matrix: np.ndarray) -> None:
    """3x3 homography: road-camera (fcam) full-frame pixels -> screen pixels for the stream on screen."""
    self._pixel_to_screen = np.asarray(matrix, dtype=np.float64)

  def _current(self, sm):
    if not sm.seen['carrotTlv'] or time.monotonic() - sm.recv_time['carrotTlv'] > STALE_S:
      return None
    msg = sm['carrotTlv']
    if int(msg.status.raw) != _STATUS_RUNNING or int(msg.relevantTrackId) < 0:
      return None
    b = msg.relevantBox
    box = (float(b.x1), float(b.y1), float(b.x2), float(b.y2))
    if not (box[2] > box[0] and box[3] > box[1]):
      return None
    return box, int(msg.state.raw), int(msg.relevantTrackId)

  def _project(self, box):
    x1, y1, x2, y2 = box
    pts = self._pixel_to_screen @ np.array([[x1, x2], [y1, y2], [1.0, 1.0]])
    if abs(pts[2, 0]) < 1e-6 or abs(pts[2, 1]) < 1e-6:
      return None
    sx1, sy1 = pts[0, 0] / pts[2, 0], pts[1, 0] / pts[2, 0]
    sx2, sy2 = pts[0, 1] / pts[2, 1], pts[1, 1] / pts[2, 1]
    return min(sx1, sx2), min(sy1, sy2), max(sx1, sx2), max(sy1, sy2)

  def render(self, sm, rect: rl.Rectangle, claimed: list) -> None:
    now = time.monotonic()
    dt = min(0.2, max(0.0, now - self._last_t))
    self._last_t = now
    cur = self._current(sm)
    if cur is not None:
      new_track = self._last is None or self._last[2] != cur[2]
      self._last = cur
      if new_track:
        self._scr = None
    target = 1.0 if cur is not None else 0.0
    self._alpha += (target - self._alpha) * (1.0 - math.exp(-dt / FADE_TAU_S)) if dt > 0 else 0.0
    if cur is not None and self._alpha < 0.05:
      self._alpha = 0.05
    if self._last is None or self._alpha < 0.02:
      return
    box, state, _track = self._last
    target_scr = self._project(box)
    if target_scr is None:
      return
    if self._scr is None:
      self._scr = target_scr
    else:
      k = 1.0 - math.exp(-dt / EASE_TAU_S) if dt > 0 else 0.0
      self._scr = tuple(a + (b - a) * k for a, b in zip(self._scr, target_scr, strict=True))
    self._draw(self._scr, box, state, rect, claimed, self._alpha)

  def _draw(self, scr, cam_box, state, rect, claimed, alpha):
    colour, label, bracket_a = _STYLE.get(state, _TRACKING)
    x1, y1, x2, y2 = scr
    w, h = x2 - x1, y2 - y1
    # Pad the housing box so the brackets frame the light instead of cutting into it.
    pad = max(8.0, 0.45 * h)
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    w, h = max(72.0, x2 - x1), max(44.0, y2 - y1)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    x1, x2, y1, y2 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    if x2 < rect.x or x1 > rect.x + rect.width or y2 < rect.y or y1 > rect.y + rect.height:
      return
    arm = max(12.0, min(24.0, 0.32 * min(w, h)))
    a = alpha
    shade = hs.rgba(0, 0, 0, int(120 * a))
    ink = hs.with_alpha(colour, int(bracket_a * a))
    corners = (
      ((x1, y1 + arm), (x1, y1), (x1 + arm, y1)),
      ((x2 - arm, y1), (x2, y1), (x2, y1 + arm)),
      ((x2, y2 - arm), (x2, y2), (x2 - arm, y2)),
      ((x1 + arm, y2), (x1, y2), (x1, y2 - arm)),
    )
    for width, col in ((OUTLINE, shade), (STROKE, ink)):
      for p0, p1, p2 in corners:
        hs.polyline([p0, p1, p2], width, col)

    # Chip above the brackets: state word (confirmed states only) + box-height distance.
    dist = tracker.estimate_distance_m(cam_box[3] - cam_box[1], _FOCAL)
    dist_text = f"{int(round(dist))}m" if math.isfinite(dist) and dist < 400 else ""
    if not dist_text and not label:
      return
    gap = 10.0
    lw = self._type.width(label, CHIP_TEXT, hs.SEMI) if label else 0.0
    dw = self._type.width(dist_text, CHIP_TEXT, hs.SEMI) if dist_text else 0.0
    inner = lw + dw + (12.0 if label and dist_text else 0.0)
    dot_r = 8.0
    chip_w = 20.0 + 2 * dot_r + 10.0 + inner + 20.0
    spot = hs.place_label(cx, y1 - gap - CHIP_H / 2, chip_w, CHIP_H, claimed, gap=8.0)
    if spot is None:
      return
    tx, mid = spot
    left = tx - chip_w / 2
    hs.pill(left, mid - CHIP_H / 2, chip_w, CHIP_H, hs.rgba(13, 16, 21, int(214 * a)), hs.rgba(255, 255, 255, int(30 * a)))
    hs.dot(left + 20.0 + dot_r, mid, dot_r, hs.with_alpha(colour, int(255 * a)))
    x = left + 20.0 + 2 * dot_r + 10.0
    if label:
      self._type.draw_mid(label, x, mid, CHIP_TEXT, hs.with_alpha(hs.TEXT, int(255 * a)), hs.SEMI)
      x += lw + 12.0
    if dist_text:
      self._type.draw_mid(dist_text, x, mid, CHIP_TEXT, hs.rgba(255, 255, 255, int(200 * a)), hs.SEMI)
    claimed.append((left, mid - CHIP_H / 2, chip_w, CHIP_H))
