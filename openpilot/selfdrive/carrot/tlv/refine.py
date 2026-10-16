"""Template re-localisation for the carrot TLV tracker (W134 Addendum A).

Ego prediction alone cannot hold a lamp between ~1 Hz detects (residual sigma
~34 px). This module re-locates each track on EVERY processed frame by
normalised cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED) of a stored
full-resolution luma template around the ego-predicted box.

Design (W134 Addendum A 2A):
  * A template is captured from the RAW full-resolution Y plane on a DETECTOR
    match only (never from the RGB model input, never from a refinement) so it
    cannot drift. It is the detector box padded 25 % per side (50 % larger),
    downsampled to <= 48 px tall, stored as float32 zero-mean unit-norm.
  * Refine runs on every processed frame for up to MAX_REFINE_TRACKS tracks in
    the caller's relevance-first order: search = ego-predicted box expanded by
    +-max(3*h, 40 px), resampled to the template scale, matchTemplate; accept
    only on peak >= PEAK_MIN and a clear margin over the best peak outside a
    +-half-template neighbourhood.
  * On accept the track box is re-centred at the match (predicted size kept) and
    the odometry anchor is refreshed by the caller.

cv2 is optional (opencv-python-headless bootstrap wheel): if it is missing the
relocaliser is inert and behaviour is exactly the pre-W134 path plus the yaw
fix. There is no numpy fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
  import cv2
except ImportError:  # pragma: no cover - exercised via the cv2-missing test
  cv2 = None

CV2_AVAILABLE = cv2 is not None


@dataclass
class _Template:
  data: np.ndarray        # float32, zero-mean, unit-norm
  th: int                 # template height (px)
  tw: int                 # template width (px)
  src_h: int              # padded patch height at capture (before downsample)
  src_w: int
  anchor_box_h: float     # detector box height at capture
  anchor_box_w: float


class TemplateRelocalizer:
  """Per-frame NCC re-localisation of tracked lamps on the full-res luma plane."""

  TEMPLATE_MAX_H = 48
  PAD_FRACTION = 0.25          # per side -> patch is 1.5x the detector box
  SEARCH_MARGIN_MIN_PX = 40.0
  SEARCH_MARGIN_H_FACTOR = 3.0
  MAX_WINDOW_PX = 160          # hard clamp on the resampled search window
  MAX_REFINE_TRACKS = 4
  # W134 Addendum C: the real-frame device estimate for 4 tracks is > 12 ms, so
  # only the 2 relevance-first tracks are refined when this is set.
  MAX_REFINE_TRACKS_2 = 2
  PEAK_MIN = 0.60
  PEAK_MARGIN_MIN = 0.10

  def __init__(self) -> None:
    self._templates: dict[int, _Template] = {}
    self.enabled = CV2_AVAILABLE

  def has_template(self, track_id: int) -> bool:
    return int(track_id) in self._templates

  def drop(self, track_id: int) -> None:
    self._templates.pop(int(track_id), None)

  def prune(self, live_ids) -> None:
    live = {int(i) for i in live_ids}
    for tid in [t for t in self._templates if t not in live]:
      del self._templates[tid]

  def refresh(self, luma: np.ndarray, track) -> bool:
    """(Re)capture the template for `track` from its current detector box."""
    if not self.enabled or luma is None:
      return False
    x1, y1, x2, y2 = (float(v) for v in track.box)
    w, h = x2 - x1, y2 - y1
    if w < 3.0 or h < 3.0:
      return False
    pad_x = self.PAD_FRACTION * w
    pad_y = self.PAD_FRACTION * h
    px1 = int(round(x1 - pad_x)); py1 = int(round(y1 - pad_y))
    px2 = int(round(x2 + pad_x)); py2 = int(round(y2 + pad_y))
    px1 = max(0, px1); py1 = max(0, py1)
    px2 = min(luma.shape[1], px2); py2 = min(luma.shape[0], py2)
    src_h, src_w = py2 - py1, px2 - px1
    if src_h < 2 or src_w < 2:
      return False
    patch = luma[py1:py2, px1:px2].astype(np.float32)
    f = min(1.0, self.TEMPLATE_MAX_H / float(src_h))
    if f < 1.0:
      th = max(1, int(round(src_h * f)))
      tw = max(1, int(round(src_w * f)))
      patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
    else:
      th, tw = src_h, src_w
    patch = patch - float(patch.mean())
    norm = float(np.linalg.norm(patch))
    if norm <= 1e-6:
      return False
    patch /= norm
    self._templates[int(track.track_id)] = _Template(
      data=patch, th=th, tw=tw, src_h=src_h, src_w=src_w,
      anchor_box_h=float(h), anchor_box_w=float(w))
    return True

  def refine(self, luma: np.ndarray, track, pred_box: tuple[float, float, float, float]):
    """Return a re-centred box for `track`, or None if the match is rejected."""
    if not self.enabled or luma is None:
      return None
    tpl = self._templates.get(int(track.track_id))
    if tpl is None:
      return None
    px1, py1, px2, py2 = (float(v) for v in pred_box)
    pw, ph = px2 - px1, py2 - py1
    if pw < 2.0 or ph < 2.0:
      return None

    margin = max(self.SEARCH_MARGIN_H_FACTOR * ph, self.SEARCH_MARGIN_MIN_PX)
    sx1 = int(round(px1 - margin)); sy1 = int(round(py1 - margin))
    sx2 = int(round(px2 + margin)); sy2 = int(round(py2 + margin))
    sx1 = max(0, sx1); sy1 = max(0, sy1)
    sx2 = min(luma.shape[1], sx2); sy2 = min(luma.shape[0], sy2)
    sw, sh = sx2 - sx1, sy2 - sy1
    if sw < 4 or sh < 4:
      return None

    # Resample the search window to the template scale: f = th/src_h is the
    # capture downsample factor and k = ph/anchor_box_h the predicted growth, so
    # the window scale is f/k.
    f = tpl.th / float(tpl.src_h)
    k = ph / tpl.anchor_box_h if tpl.anchor_box_h > 0 else 1.0
    r = f / k if k > 0 else f
    if r <= 0.0:
      return None
    rw = max(1, int(round(sw * r))); rh = max(1, int(round(sh * r)))
    m = max(rw, rh)
    if m > self.MAX_WINDOW_PX:
      scale = self.MAX_WINDOW_PX / float(m)
      rw = max(1, int(round(rw * scale))); rh = max(1, int(round(rh * scale)))
      r *= scale
    if rh < tpl.th or rw < tpl.tw:
      return None

    win = luma[sy1:sy2, sx1:sx2].astype(np.float32)
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    win = cv2.resize(win, (rw, rh), interpolation=interp)
    res = cv2.matchTemplate(win, tpl.data, cv2.TM_CCOEFF_NORMED)
    _min_v, peak, _min_l, loc = cv2.minMaxLoc(res)
    if peak < self.PEAK_MIN:
      return None

    # Margin: the peak must beat the best response outside a +-half-template
    # neighbourhood (a sharp single-light peak; adjacent heads do not confuse it).
    nx = max(1, int(round(0.5 * tpl.tw))); ny = max(1, int(round(0.5 * tpl.th)))
    ox0 = max(0, loc[0] - nx); oy0 = max(0, loc[1] - ny)
    ox1 = min(res.shape[1], loc[0] + nx + 1); oy1 = min(res.shape[0], loc[1] + ny + 1)
    outside = res.copy()
    outside[oy0:oy1, ox0:ox1] = -1.0
    best_out = float(outside.max()) if outside.size else -1.0
    if peak - best_out < self.PEAK_MARGIN_MIN:
      return None

    full_tx = sx1 + loc[0] / r
    full_ty = sy1 + loc[1] / r
    cx = full_tx + 0.5 * (tpl.tw / r)
    cy = full_ty + 0.5 * (tpl.th / r)
    return (cx - 0.5 * pw, cy - 0.5 * ph, cx + 0.5 * pw, cy + 0.5 * ph)
