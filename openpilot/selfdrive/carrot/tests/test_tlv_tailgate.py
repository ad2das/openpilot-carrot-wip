"""W99 tests: flat-road tail-light gate (calibrated frames only) in pipeline.infer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import crops, decode, pipeline

W, H = crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT
FOCAL = crops.FOCAL_REFERENCE_PX
HORIZON = 600.0


class _StubBackend:
  """Returns fixed raw outputs for every infer call (no model involved)."""

  def __init__(self, boxes, scores, classes) -> None:
    self.raw = {"boxes": np.asarray(boxes, dtype=np.float64),
                "scores": np.asarray(scores, dtype=np.float64),
                "classes": np.asarray(classes)}

  def infer(self, model_input):
    return self.raw


def make_pipe(boxes, cls=0, score=0.9):
  n = len(boxes)
  return pipeline.FramePipeline(_StubBackend(boxes, [score] * n, [cls] * n), decode.PlaceholderDecoder())


def far_geom():
  return pipeline.FramePipeline.geometry("far", HORIZON, W, H, FOCAL)


def gate_limit() -> float:
  return HORIZON + FOCAL * math.tan(math.radians(pipeline.TAIL_GATE_MARGIN_DEG))


def run(pipe, *, gate_enabled=True, horizon_y=HORIZON, now=0.0):
  return pipe.infer("far", far_geom(), image_width=W, now=now, detect=True,
                    horizon_y=horizon_y, gate_enabled=gate_enabled)


def det_box(model_x1, model_y1, model_x2, model_y2):
  """Model-space xyxy -> a numpy row for the stub backend."""
  return (model_x1, model_y1, model_x2, model_y2)


# ---------------------------------------------------------------------------
# gate behaviour
# ---------------------------------------------------------------------------

def test_below_horizon_box_is_gated():
  # Full-frame y1 = 360 + 290 = 650, comfortably below the 611.5 px limit.
  pipe = make_pipe([det_box(300.0, 290.0, 340.0, 319.0)])
  out = run(pipe)
  assert out.dets == []
  assert len(out.gated) == 1
  assert out.gated[0].y1 == pytest.approx(650.0)
  assert pipe.tail_gated_total == 1
  assert pipe.last_gated == out.gated
  assert pipe.last_dets == []


def test_box_straddling_the_horizon_line_is_kept():
  # y1 = 360 + 245 = 605 <= 611.5 (top above the limit) -> kept.
  pipe = make_pipe([det_box(300.0, 245.0, 340.0, 300.0)])
  out = run(pipe)
  assert len(out.dets) == 1
  assert out.gated == []
  assert pipe.tail_gated_total == 0


def test_gate_margin_is_respected():
  limit = gate_limit()
  above = limit - 0.5 - 360.0  # full y1 just above the limit -> kept
  below = limit + 0.5 - 360.0  # full y1 just below the limit -> gated
  pipe = make_pipe([det_box(300.0, above, 340.0, above + 5.0),
                    det_box(400.0, below, 440.0, below + 5.0)])
  out = run(pipe)
  assert len(out.dets) == 1
  assert out.dets[0].x1 == pytest.approx(944.0)  # model 300 + geom x0 644
  assert len(out.gated) == 1
  assert out.gated[0].y1 == pytest.approx(limit + 0.5, abs=0.01)


def test_uncalibrated_frames_are_not_gated():
  pipe = make_pipe([det_box(300.0, 290.0, 340.0, 319.0)])
  out = run(pipe, gate_enabled=False)
  assert len(out.dets) == 1
  assert out.gated == []
  assert pipe.tail_gated_total == 0


def test_gate_never_runs_on_classify_only_frames():
  pipe = make_pipe([det_box(300.0, 290.0, 340.0, 319.0)])
  out = pipe.infer("far", far_geom(), image_width=W, now=0.0, detect=False,
                   horizon_y=HORIZON, gate_enabled=True)
  assert out.dets == [] and out.gated == []
  assert pipe.tail_gated_total == 0


def test_gated_counter_accumulates_across_frames():
  pipe = make_pipe([det_box(300.0, 290.0, 340.0, 319.0)])
  run(pipe, now=0.0)
  run(pipe, now=1.0)
  assert pipe.tail_gated_total == 2


def test_mixed_boxes_keep_tracker_input_clean():
  pipe = make_pipe([det_box(300.0, 100.0, 340.0, 140.0),
                    det_box(500.0, 290.0, 540.0, 319.0)])
  out = run(pipe)
  assert len(out.dets) == 1 and len(out.gated) == 1
  # The tracker only ever saw the kept detection.
  assert len(pipe.tracker.tracks) == 1
  assert pipe.tracker.tracks[0].box[1] == pytest.approx(460.0)
