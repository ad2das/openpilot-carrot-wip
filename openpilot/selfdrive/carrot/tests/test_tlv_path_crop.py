"""WCV4: the TLV crop follows the model path on curves.

On a curve the lamp that decides the signal sits off to the side of the centred
crop, so each view's crop is aimed at where modelV2 says the road is at that
view's look-ahead distance (FAR 80 m, NEAR 50 m). The relevant track's predicted
box always stays inside the crop, and a straight/stale/empty path, standstill or
the raised standstill view leaves the crop exactly where it was.
"""
from __future__ import annotations

from dataclasses import astuple

import numpy as np

from openpilot.selfdrive.carrot.tlv import crops, decode, pipeline, tracker

W, H = crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT
FOCAL = crops.focal_px_for_width(W)
HORIZON_Y = 640.0

# A path that bends 10 m left over 80 m: the far look-ahead sits 331 px left of
# the image centre at the reference focal (2648 * 10 / 80).
LEFT_X = (0.0, 50.0, 100.0)
LEFT_Y = (0.0, -6.25, -12.5)
RIGHT_Y = (0.0, 6.25, 12.5)

# Real modelV2 path from the WCV1 S2 left curve (+112.5 deg over 6 s), route
# 000006cc--e118115683, one frame at 13:18:35.008 (frame 3760, vEgo ~11 km/h),
# extracted with carrot-tl-vision/eval/tools/w150_common.py (C.ROUTES set as in
# carrot-tl-vision/.scratch/k5tl.py:1-5), position.x / position.y in metres.
REAL_PATH_X = (0.0, 0.0532, 0.2128, 0.4804, 0.8556, 1.3433, 1.9341, 2.6401, 3.452,
               4.3669, 5.3812, 6.4398, 7.6084, 8.8194, 10.0498, 11.2976, 12.5753,
               13.8289, 15.1436, 16.4442, 17.8606, 19.1412, 20.5853, 22.1593,
               23.8502, 25.5927, 27.4683, 29.5904, 31.6082, 33.9664, 36.3624,
               38.9795, 41.5966)
REAL_PATH_Y = (-0.0, -0.0043, -0.0174, -0.0421, -0.0835, -0.1492, -0.2483,
               -0.3922, -0.5937, -0.8654, -1.2386, -1.7039, -2.3186, -3.0572,
               -3.9629, -5.0404, -6.3167, -7.7999, -9.4914, -11.3792, -13.3652,
               -15.7203, -18.1198, -20.7262, -23.6036, -26.5716, -29.8283,
               -33.2925, -36.9532, -40.9235, -45.1665, -49.5849, -54.0793)


class _StubBackend:
  """Returns fixed raw outputs for every infer call (no model involved)."""

  def __init__(self, boxes, scores, classes) -> None:
    self.raw = {"boxes": np.asarray(boxes, dtype=np.float64),
                "scores": np.asarray(scores, dtype=np.float64),
                "classes": np.asarray(classes)}

  def infer(self, model_input):
    return self.raw


def empty_pipe():
  return pipeline.FramePipeline(
    _StubBackend(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))),
    decode.PlaceholderDecoder(),
  )


def geometry(view, raised=False):
  return pipeline.FramePipeline.geometry(view, HORIZON_Y, W, H, FOCAL, raised=raised)


def path_geometry(pipe, view, v_ego_kph, path_x=LEFT_X, path_y=LEFT_Y, raised=False):
  return pipe.path_geometry(view, HORIZON_Y, W, H, FOCAL, raised=raised,
                            path_x=path_x, path_y=path_y, v_ego_kph=v_ego_kph)


def model_box(full_box, geom):
  """Model-space xyxy of a full-frame box (inverse of map_model_box_to_full)."""
  x1, y1, x2, y2 = full_box
  s = geom.scale
  return ((x1 - geom.x0) / s, (y1 - geom.y0) / s, (x2 - geom.x0) / s, (y2 - geom.y0) / s)


def pipe_with_box(full_box):
  """Pipeline holding one track whose stored (ego-free) box is `full_box`.

  Built through the stub backend exactly like test_tlv_nearview.py, then the
  stored box is replaced: the placeholder decoder clips model-space boxes into
  the crop, so a box the centred crop does NOT contain (the case under test)
  cannot come out of the detector path by construction.
  """
  geom = geometry("far")
  cx, cy = geom.x0 + 0.5 * geom.w, geom.y0 + 0.5 * geom.h
  pipe = pipeline.FramePipeline(
    _StubBackend([model_box((cx - 20.0, cy - 10.0, cx + 20.0, cy + 10.0), geom)],
                 [0.9], [int(tracker.LampClass.RED)]),
    decode.PlaceholderDecoder(),
  )
  pipe.infer("far", geom, image_width=W, now=1.0, v_ego_kph=40.0)
  assert len(pipe.tracker.tracks) == 1, "stub detection must create exactly one track"
  track = pipe.tracker.tracks[0]
  track.box = full_box
  track.anchor_box = full_box
  pipe.last_track_id = track.track_id
  return pipe, track


def test_straight_path_is_centred():
  pipe = empty_pipe()
  far = path_geometry(pipe, "far", 40.0, path_y=(0.0, 0.0, 0.0))
  near = path_geometry(pipe, "near", 40.0, path_y=(0.0, 0.0, 0.0))
  assert pipeline.path_shift_px("far", LEFT_X, (0.0, 0.0, 0.0), FOCAL) == 0.0
  assert astuple(far) == astuple(geometry("far"))
  assert astuple(near) == astuple(geometry("near"))
  assert far.x0 == 644 and near.x0 == 324


def test_left_curve_far():
  pipe = empty_pipe()
  shift = pipeline.path_shift_px("far", LEFT_X, LEFT_Y, FOCAL)
  far = path_geometry(pipe, "far", 40.0)
  print(f"left far shift={shift:.1f}px x0={far.x0}")
  assert shift == -331.0
  assert far.x0 == 312


def test_right_curve_far():
  pipe = empty_pipe()
  shift = pipeline.path_shift_px("far", LEFT_X, RIGHT_Y, FOCAL)
  far = path_geometry(pipe, "far", 40.0, path_y=RIGHT_Y)
  print(f"right far shift={shift:.1f}px x0={far.x0}")
  assert shift == 331.0
  assert far.x0 == 974


def test_left_curve_near_clamps_to_frame():
  pipe = empty_pipe()
  shift = pipeline.path_shift_px("near", LEFT_X, LEFT_Y, FOCAL)
  near = path_geometry(pipe, "near", 40.0)
  print(f"left near shift={shift:.1f}px x0={near.x0}")
  assert shift == -331.0
  assert near.x0 == 0


def test_short_or_empty_path_is_centred():
  pipe = empty_pipe()
  assert pipeline.path_shift_px("far", (0.0, 10.0), (0.0, -1.25), FOCAL) == 0.0
  assert pipeline.path_shift_px("far", (), (), FOCAL) == 0.0
  short = path_geometry(pipe, "far", 40.0, path_x=(0.0, 10.0), path_y=(0.0, -1.25))
  assert astuple(short) == astuple(geometry("far"))
  assert short.x0 == 644


def test_standstill_and_raised_are_centred():
  pipe = empty_pipe()
  still = path_geometry(pipe, "far", 4.0)
  assert astuple(still) == astuple(geometry("far"))
  raised = path_geometry(pipe, "near", 40.0, raised=True)
  assert astuple(raised) == astuple(geometry("near", raised=True))
  assert radius_check(raised)


def radius_check(geom):
  """The raised near crop is higher in the frame than the plain one."""
  return geom.y0 < geometry("near").y0


def test_keep_box_limits_shift():
  far = geometry("far")
  kept = pipeline.shift_geometry(far, -331.0, W, keep_box=(900.0, 400.0, 940.0, 420.0))
  print(f"keep-box x0={kept.x0} (was {far.x0})")
  assert kept.x0 == 316


def test_keep_box_never_moves_crop_without_path():
  far = geometry("far")
  kept = pipeline.shift_geometry(far, 0.0, W, keep_box=(1700.0, 400.0, 1740.0, 420.0))
  assert kept.x0 == 644


def test_classify_view_uses_shifted_far():
  box = (400.0, 450.0, 440.0, 470.0)
  pipe, track = pipe_with_box(box)
  assert tuple(track.box) == box
  far = geometry("far")
  shifted = path_geometry(pipe, "far", 40.0)
  print(f"classify far y0={shifted.y0} centred x0={far.x0} shifted x0={shifted.x0}")
  assert shifted.x0 == 312
  assert pipe.classify_view(shifted) == "far"
  assert pipe.classify_view(far) == "near"


def test_real_left_curve_moves_crop_left():
  shift = pipeline.path_shift_px("far", REAL_PATH_X, REAL_PATH_Y, FOCAL)
  far = empty_pipe().path_geometry("far", HORIZON_Y, W, H, FOCAL, raised=False,
                                    path_x=REAL_PATH_X, path_y=REAL_PATH_Y, v_ego_kph=40.0)
  end_x, end_y = REAL_PATH_X[-1], REAL_PATH_Y[-1]
  print(f"real left curve: path end x={end_x:.1f}m y={end_y:.1f}m shift={shift:.1f}px far x0={far.x0}")
  assert shift < -100.0
  assert far.x0 < 644
