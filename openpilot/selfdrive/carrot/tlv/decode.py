"""Decoder interface for TLV model raw outputs.

The real decoder will be ported from W71's train/decode.py once the model is
trained. Until then `PlaceholderDecoder` defines the stub-model contract:

  raw_outputs = (boxes, scores, classes) or a dict with those keys
    boxes:   float array [N, 4], xyxy in model-input pixels (640x320)
    scores:  float array [N]
    classes: int/float array [N], CONTRACT class indices

`decode(raw_outputs, view) -> list[DecodedDetection]`; the process maps boxes
to full-frame pixels afterwards (crops.map_model_box_to_full).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from openpilot.selfdrive.carrot.tlv.crops import MODEL_H, MODEL_W


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
