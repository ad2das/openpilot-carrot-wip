"""Decoder interface for TLV model raw outputs.

Real models (the bundled r3b detector, the r8 family) export 6 raw YOLO-style
heads: `pN_box` / `pN_cls` per level, with Distribution Focal Loss box
regression (reg_max 16) and either 8 classes (CONTRACT classes) or 1 class
(single-class detectors, whose class 0 maps to RED on the device).
`RawHeadDecoder` decodes those heads; it is a port of the `decode_raw` function
from carrot-tl-vision `train/decode.py`, the decoder the training/eval tooling
and every W70-W115 sim actually used.

`PlaceholderDecoder` stays for the 3-output stub/test contract:

  raw_outputs = (boxes, scores, classes) or a dict with those keys
    boxes:   float array [N, 4], xyxy in model-input pixels (640x320)
    scores:  float array [N]
    classes: int/float array [N], CONTRACT class indices

`decode(raw_outputs, view) -> list[DecodedDetection]`; the process maps boxes
to full-frame pixels afterwards (crops.map_model_box_to_full).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from openpilot.selfdrive.carrot.tlv.crops import MODEL_H, MODEL_W

REG_MAX = 16

RAW_HEAD_RE = re.compile(r"^p(\d+)_(box|cls)$")


# Ported VERBATIM from carrot-tl-vision train/decode.py (decode_raw and its
# helpers), reindented only; the math, dtypes and class offset must not change.
def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def _dfl(box):
  b, c, h, w = box.shape
  b = box.reshape(b, 4, REG_MAX, h * w)
  b = b - b.max(axis=2, keepdims=True)
  e = np.exp(b)
  p = e / e.sum(axis=2, keepdims=True)
  idx = np.arange(REG_MAX, dtype=np.float32).reshape(1, 1, REG_MAX, 1)
  return (p * idx).sum(axis=2)


def _make_anchors(shapes, strides):
  pts = []
  st = []
  for (h, w), s in zip(shapes, strides):
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    pts.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
    st.append(np.full(h * w, s, dtype=np.float32))
  return np.concatenate(pts, 0), np.concatenate(st, 0)


def _nms(boxes, scores, classes, iou_thres, max_det):
  offset = classes.astype(np.float32) * 7680.0
  x = boxes + offset[:, None]
  order = np.argsort(-scores, kind="stable")
  suppressed = np.zeros(len(order), dtype=bool)
  areas = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
  keep = []
  for ii, i in enumerate(order):
    if suppressed[ii]:
      continue
    keep.append(i)
    if len(keep) >= max_det:
      break
    rest = order[ii + 1:]
    xx1 = np.maximum(x[i, 0], x[rest, 0])
    yy1 = np.maximum(x[i, 1], x[rest, 1])
    xx2 = np.minimum(x[i, 2], x[rest, 2])
    yy2 = np.minimum(x[i, 3], x[rest, 3])
    ww = np.clip(xx2 - xx1, 0, None)
    hh = np.clip(yy2 - yy1, 0, None)
    inter = ww * hh
    iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
    suppressed[ii + 1:] |= iou > iou_thres
  return np.array(keep, dtype=int)


def is_raw_head_layout(output_names) -> bool:
  """True when the output names are a complete pN_box/pN_cls raw-head set."""
  names = tuple(output_names or ())
  if not names:
    return False
  levels_box = set()
  levels_cls = set()
  for name in names:
    match = RAW_HEAD_RE.match(str(name))
    if match is None:
      return False
    (levels_box if match.group(2) == "box" else levels_cls).add(match.group(1))
  if levels_box != levels_cls:
    return False
  return len(set(names)) == len(names)


@dataclass
class DecodedDetection:
  cls: int
  score: float
  x1: float
  y1: float
  x2: float
  y2: float


class Decoder(Protocol):
  def decode(self, raw_outputs, view: str) -> list[DecodedDetection]:
    ...


def _as_arrays(raw_outputs):
  if isinstance(raw_outputs, dict):
    return raw_outputs.get("boxes"), raw_outputs.get("scores"), raw_outputs.get("classes")
  if isinstance(raw_outputs, (list, tuple)) and len(raw_outputs) == 3:
    return raw_outputs[0], raw_outputs[1], raw_outputs[2]
  return None, None, None


class RawHeadDecoder:
  """decode_raw port for real detector heads (pN_box/pN_cls, DFL reg_max 16).

  `decode` never raises: malformed outputs return an empty list.
  """

  def __init__(self, output_names, score_threshold: float = 0.25,
               iou_threshold: float = 0.45, max_det: int = 50) -> None:
    names = tuple(output_names or ())
    if not is_raw_head_layout(names):
      raise ValueError(f"not a raw-head output layout: {names!r}")
    self.output_names = names
    self.score_threshold = float(score_threshold)
    self.iou_threshold = float(iou_threshold)
    self.max_det = int(max_det)
    self._anchor_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

  def _anchors_for(self, shapes, strides):
    key = tuple(shapes)
    cached = self._anchor_cache.get(key)
    if cached is None:
      cached = _make_anchors(shapes, strides)
      self._anchor_cache[key] = cached
    return cached

  def decode(self, raw_outputs, view: str) -> list[DecodedDetection]:
    if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != len(self.output_names):
      return []
    raw = {name: np.asarray(o, dtype=np.float32) for name, o in zip(self.output_names, raw_outputs)}
    try:
      # Body of decode_raw (train/decode.py): levels sorted by the int N in
      # pN_box, stride 2**N; 3-level exports decode exactly as before.
      levels = []
      shapes = []
      level_strides = []
      for lvl, _k in sorted((int(k[1:-4]), k) for k in raw.keys()
                            if k.startswith("p") and k.endswith("_box")):
        box = raw[f"p{lvl}_box"]
        cls = raw[f"p{lvl}_cls"]
        shapes.append(box.shape[-2:])
        level_strides.append(2 ** lvl)
        levels.append((box, cls))
      anchors, strides = self._anchors_for(shapes, level_strides)
      all_boxes, all_scores, all_classes = [], [], []
      off = 0
      for (box, cls) in levels:
        dist = _dfl(box)[0].reshape(4, -1)
        h, w = box.shape[-2:]
        n = h * w
        a = anchors[off:off + n]
        st = strides[off:off + n]
        off += n
        xyxy = np.concatenate([a.T - dist[:2], a.T + dist[2:]], axis=0) * st[None, :]
        sc = _sigmoid(cls[0].reshape(cls.shape[1], -1))
        all_boxes.append(xyxy.T)
        all_scores.append(sc.max(axis=0))
        all_classes.append(sc.argmax(axis=0))
      boxes = np.concatenate(all_boxes, 0)
      scores = np.concatenate(all_scores, 0)
      classes = np.concatenate(all_classes, 0)
      m = scores > self.score_threshold  # strictly greater, like the original
      boxes, scores, classes = boxes[m], scores[m], classes[m]
      if len(boxes) == 0:
        return []
      keep = _nms(boxes, scores, classes, self.iou_threshold, self.max_det)
      boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
      boxes[:, 0::2] = boxes[:, 0::2].clip(0, MODEL_W)
      boxes[:, 1::2] = boxes[:, 1::2].clip(0, MODEL_H)
    except (ValueError, IndexError):
      return []
    dets: list[DecodedDetection] = []
    for box, score, cls in zip(boxes, scores, classes, strict=True):
      # Device-only addition after the exact port: drop degenerate boxes (the
      # original 6-tuple contract could carry them; the device tracker cannot).
      if not np.isfinite(box).all() or not np.isfinite(score):
        continue
      if box[2] <= box[0] or box[3] <= box[1]:
        continue
      dets.append(DecodedDetection(cls=int(cls), score=float(score),
                                   x1=float(box[0]), y1=float(box[1]),
                                   x2=float(box[2]), y2=float(box[3])))
    return dets


def make_decoder(output_names, score_threshold: float = 0.25) -> Decoder:
  """Pick the decoder for a model's output names (raw heads vs stub)."""
  if is_raw_head_layout(output_names):
    return RawHeadDecoder(output_names, score_threshold=score_threshold)
  return PlaceholderDecoder(score_threshold)


class PlaceholderDecoder:
  """Stub-model decoder; tolerant to empty/odd outputs."""

  def __init__(self, score_threshold: float = 0.25) -> None:
    self.score_threshold = float(score_threshold)

  def decode(self, raw_outputs, view: str) -> list[DecodedDetection]:
    boxes, scores, classes = _as_arrays(raw_outputs)
    if boxes is None or scores is None or classes is None:
      return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != scores.size or boxes.shape[0] != classes.size:
      return []
    dets: list[DecodedDetection] = []
    for box, score, cls in zip(boxes, scores, classes, strict=True):
      if not np.isfinite(box).all() or not np.isfinite(score):
        continue
      if float(score) < self.score_threshold:
        continue
      x1 = float(np.clip(box[0], 0.0, MODEL_W))
      y1 = float(np.clip(box[1], 0.0, MODEL_H))
      x2 = float(np.clip(box[2], 0.0, MODEL_W))
      y2 = float(np.clip(box[3], 0.0, MODEL_H))
      if x2 <= x1 or y2 <= y1:
        continue
      dets.append(DecodedDetection(cls=int(cls), score=float(score), x1=x1, y1=y1, x2=x2, y2=y2))
    return dets
