"""W101: full-frame focal mapping in pipeline.infer and ego-odometry pass-through.

The NEAR crop is a x2 downsample, so crops._geometry stores the MODEL-space
focal (focal/2) in ViewGeometry.focal_px while pipeline.infer maps detections
back to FULL-frame pixels. The tracker must receive geom.focal_px * geom.scale,
which equals crops.focal_px_for_width(image_width) for both views, and
select_relevant must see the pipeline's odometry so the distance gate decays
with ego travel instead of freezing at the last matched box.
"""
from __future__ import annotations

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import crops, decode, pipeline, tracker

W, H = crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT
FOCAL = crops.FOCAL_REFERENCE_PX
HORIZON = 600.0
HOUSING = tracker.HOUSING_HEIGHT_M

# Full-frame light centre used by every test (inside both crops, above the
# horizon, on the image centre line).
CENTER_X = 0.5 * W
CENTER_Y = 550.0


class _StubBackend:
  """Returns fixed raw outputs for every infer call (no model involved)."""

  def __init__(self, boxes, scores, classes) -> None:
    self.raw = {"boxes": np.asarray(boxes, dtype=np.float64),
                "scores": np.asarray(scores, dtype=np.float64),
                "classes": np.asarray(classes)}

  def infer(self, model_input):
    return self.raw


class _FakeClassifier:
  """Uniform logits: any crop yields class 0 with a low probability."""

  def classify(self, batch):
    return np.zeros((batch.shape[0], 8), dtype=np.float32)


def geometry(view, width=W, height=H):
  return pipeline.FramePipeline.geometry(view, HORIZON, width, height, crops.focal_px_for_width(width))


def model_box_at(distance_m, geom):
  """Model-space xyxy of a light at distance_m centred on (CENTER_X, CENTER_Y)."""
  full_h = (geom.focal_px * geom.scale) * HOUSING / distance_m
  xm = (CENTER_X - geom.x0) / geom.scale
  ym = (CENTER_Y - geom.y0) / geom.scale
  half = 0.5 * full_h / geom.scale
  return (xm - half, ym - half, xm + half, ym + half)


def make_pipe(geom, distance_m):
  return pipeline.FramePipeline(
    _StubBackend([model_box_at(distance_m, geom)], [0.9], [int(tracker.LampClass.RED)]),
    decode.PlaceholderDecoder(),
  )


def single_track(pipe):
  assert len(pipe.tracker.tracks) == 1
  return pipe.tracker.tracks[0]


def track_estimate(pipe):
  track = single_track(pipe)
  return track, tracker.estimate_distance_m(track.height, track.focal_px)


def empty_raw():
  return {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "classes": np.zeros((0,))}


@pytest.mark.parametrize("width", [crops.REFERENCE_WIDTH, 1280, 964])
def test_full_frame_focal_equals_focal_for_width(width):
  focal = crops.focal_px_for_width(width)
  for view in ("far", "near"):
    geom = pipeline.FramePipeline.geometry(view, HORIZON, width, H, focal)
    assert geom.focal_px * geom.scale == pytest.approx(focal)


@pytest.mark.parametrize("view", ["far", "near"])
def test_track_distance_matches_true_distance(view):
  distance_m = 45.0
  geom = geometry(view)
  pipe = make_pipe(geom, distance_m)
  pipe.infer(view, geom, image_width=W, now=0.0)
  track, estimated = track_estimate(pipe)
  # The tracker stores the FULL-frame focal, not the crop's model-space focal.
  assert track.focal_px == pytest.approx(FOCAL)
  assert estimated == pytest.approx(distance_m, rel=0.02)


def test_far_to_near_switch_keeps_distance_continuous():
  distance_m = 45.0
  far, near = geometry("far"), geometry("near")
  backend = _StubBackend([model_box_at(distance_m, far)], [0.9], [int(tracker.LampClass.RED)])
  pipe = pipeline.FramePipeline(backend, decode.PlaceholderDecoder())
  pipe.infer("far", far, image_width=W, now=0.0)
  _, far_est = track_estimate(pipe)
  backend.raw["boxes"] = np.asarray([model_box_at(distance_m, near)], dtype=np.float64)
  pipe.infer("near", near, image_width=W, now=0.05)
  track, near_est = track_estimate(pipe)
  assert track.view == "near"  # the same track survived the view switch
  assert far_est == pytest.approx(distance_m, rel=0.02)
  assert near_est == pytest.approx(distance_m, rel=0.02)
  assert near_est == pytest.approx(far_est, rel=0.02)


def test_pipeline_passes_ego_odometry_to_confirmed_selection():
  distance_m = 210.0
  geom = geometry("far")
  backend = _StubBackend([model_box_at(distance_m, geom)], [0.9], [int(tracker.LampClass.RED)])
  pipe = pipeline.FramePipeline(backend, decode.PlaceholderDecoder())
  odo = tracker.EgoOdometry()
  pipe.ego = tracker.EgoFrame(odo=odo, focal_px=FOCAL, foe_x=CENTER_X, foe_y=HORIZON,
                              image_w=W, image_h=H)
  for i in range(3):
    pipe.infer("far", geom, image_width=W, now=0.6 * i)
  # Stale 210 m anchor: outside both gate tiers, the state machine never sees red.
  assert pipe.last_state == tracker.SignalState.none
  odo.s = 40.0  # 210 - 40 = 170 m -> inside the far tier (120, 200] m
  backend.raw = empty_raw()
  out = pipe.infer("far", geom, image_width=W, now=1.8)
  assert out.state == tracker.SignalState.red
  assert out.track_id == single_track(pipe).track_id


def test_pipeline_passes_ego_odometry_to_candidate_selection():
  distance_m = 210.0
  geom = geometry("far")

  def run(with_ego):
    backend = _StubBackend([model_box_at(distance_m, geom)], [0.9], [int(tracker.LampClass.RED)])
    pipe = pipeline.FramePipeline(backend, decode.PlaceholderDecoder(), classifier=_FakeClassifier())
    odo = tracker.EgoOdometry()
    if with_ego:
      pipe.ego = tracker.EgoFrame(odo=odo, focal_px=FOCAL, foe_x=CENTER_X, foe_y=HORIZON,
                                  image_w=W, image_h=H)
    pipe.infer("far", geom, image_width=W, now=0.0)  # one hit: not confirmed
    backend.raw = empty_raw()
    odo.s = 40.0
    out = pipe.infer("far", geom, image_width=W, now=0.6)
    return pipe, out

  pipe_with, out_with = run(True)
  pipe_without, out_without = run(False)
  # Candidate-only frames (no confirmation yet) still select with odometry:
  # the stale 150 m anchor is only relevant after the ego travel decays it.
  assert out_without.track_id == -1
  assert out_with.track_id == single_track(pipe_with).track_id
