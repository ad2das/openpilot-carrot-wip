"""W99 tests: ego-motion prediction (EgoOdometry/predict_box/vanishing_point),
predicted association, view-miss/frame expiry and classify_view."""
from __future__ import annotations

import math

import pytest

from openpilot.selfdrive.carrot.tlv import crops, pipeline, tracker

W, H = crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT
FOCAL = crops.FOCAL_REFERENCE_PX
V80 = 80.0 / 3.6


def _box_at(distance_m: float, *, focal: float = FOCAL, cx: float = 964.0, cy: float = 600.0):
  half_h = focal * tracker.HOUSING_HEIGHT_M / distance_m / 2.0
  return (cx - half_h, cy - half_h, cx + half_h, cy + half_h)


def _det(box, cls: int = 0, score: float = 0.9) -> tracker.Detection:
  return tracker.Detection(cls=cls, score=score, x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def _anchored_track(box, *, anchor_s: float = 0.0, anchor_psi: float = 0.0) -> tracker.Track:
  track = tracker.Track(track_id=1, first_frame=0, last_frame=0, cls=0, score=0.9,
                        box=box, view="far", focal_px=FOCAL)
  track.anchor_box = box
  track.anchor_s = anchor_s
  track.anchor_psi = anchor_psi
  return track


def _predict(track, ego):
  return tracker.predict_box(track, ego.odo, focal_px=ego.focal_px, foe_x=ego.foe_x, foe_y=ego.foe_y,
                             image_w=ego.image_w, image_h=ego.image_h)


def _advance(odo, v, *, yaw=0.0, seconds=1.0, t0=0.0):
  """Integrate exactly `seconds` of motion at the 0.2 s dt-clamp cadence."""
  steps = max(1, int(round(seconds / 0.2)))
  for i in range(steps + 1):
    odo.update(t0 + 0.2 * i, v, yaw)


# ---------------------------------------------------------------------------
# EgoOdometry
# ---------------------------------------------------------------------------

def test_ego_odometry_integrates_and_clamps():
  odo = tracker.EgoOdometry()
  odo.update(0.0, 10.0, 0.2)
  assert odo.s == 0.0 and odo.psi == 0.0  # first sample seeds only
  odo.update(1.0, 10.0, 0.2)  # dt clamped to 0.2 s
  assert odo.s == pytest.approx(2.0)
  assert odo.psi == pytest.approx(0.04)


def test_ego_odometry_ignores_none_and_nan_keeps_last():
  odo = tracker.EgoOdometry()
  odo.update(0.0, 10.0, 0.2)
  odo.update(0.1, None, None)
  assert odo.s == pytest.approx(1.0)
  assert odo.psi == pytest.approx(0.02)
  odo.update(0.2, float("nan"), float("nan"))
  assert odo.s == pytest.approx(2.0)
  assert odo.psi == pytest.approx(0.04)


def test_ego_odometry_dt_clamp_and_reversal():
  odo = tracker.EgoOdometry()
  odo.update(0.0, 10.0, 1.0)
  odo.update(5.0, 10.0, 1.0)  # 5 s gap integrates at most 0.2 s
  assert odo.s == pytest.approx(2.0)
  assert odo.psi == pytest.approx(0.2)
  odo.update(4.0, 10.0, 1.0)  # negative dt ignored
  assert odo.s == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# predict_box / vanishing_point
# ---------------------------------------------------------------------------

def test_predict_box_matches_pinhole_approach_at_80_kph():
  anchor_box = _box_at(120.0)
  track = _anchored_track(anchor_box)
  odo = tracker.EgoOdometry()
  _advance(odo, V80)  # one full second -> ds = 22.222 m
  assert odo.s == pytest.approx(V80)
  box = tracker.predict_box(track, odo, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)
  true = _box_at(120.0 - odo.s)
  for got, want in zip(box, true, strict=True):
    assert got == pytest.approx(want, abs=0.05)


def test_predict_box_k_clamps_and_close_range():
  anchor_box = _box_at(120.0)
  track = _anchored_track(anchor_box)
  odo = tracker.EgoOdometry()
  _advance(odo, 100.0)  # ds = 100 m -> raw k = 6, clamped to 2
  box = tracker.predict_box(track, odo, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)
  assert (box[3] - box[1]) == pytest.approx(2.0 * (anchor_box[3] - anchor_box[1]), rel=1e-6)
  # d0 - ds < 5 m -> no scaling at all.
  close = _anchored_track(_box_at(6.0))
  odo2 = tracker.EgoOdometry()
  _advance(odo2, 2.0)  # ds = 2 m, d0 - ds = 4 < 5
  box = tracker.predict_box(close, odo2, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)
  assert box == pytest.approx(close.anchor_box)


def test_predict_box_reversal_keeps_anchor_geometry():
  anchor_box = _box_at(60.0)
  track = _anchored_track(anchor_box, anchor_s=5.0)
  odo = tracker.EgoOdometry()
  odo.update(0.0, -1.0, 0.0)
  odo.update(1.0, -1.0, 0.0)  # s goes 0 -> -1: ds < 0
  box = tracker.predict_box(track, odo, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)
  assert box == pytest.approx(anchor_box)


def test_predict_box_yaw_sign_moves_static_object_right():
  anchor_box = _box_at(60.0)
  track = _anchored_track(anchor_box)
  odo = tracker.EgoOdometry()
  _advance(odo, 0.0, yaw=0.1)  # psi = 0.1, no translation
  assert odo.psi == pytest.approx(0.1)
  box = tracker.predict_box(track, odo, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)
  shift = 0.5 * (box[0] + box[2]) - 964.0
  # openpilot convention: positive yawRate = LEFT turn; a static object moves
  # to +x (right) in the image, hence x' += focal_px * tan(dpsi).
  assert shift == pytest.approx(FOCAL * math.tan(0.1), rel=1e-6)
  assert shift > 0.0


def test_predict_box_without_anchor_is_the_track_box():
  box = _box_at(30.0)
  track = tracker.Track(track_id=1, first_frame=0, last_frame=0, cls=0, score=0.9,
                        box=box, view="far", focal_px=FOCAL)
  odo = tracker.EgoOdometry()
  odo.update(0.0, V80, 0.0)
  odo.update(1.0, V80, 0.0)
  got = tracker.predict_box(track, odo, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)
  assert got == pytest.approx(box)


def test_vanishing_point_with_calib_yaw():
  yaw = 0.01
  x, y = crops.vanishing_point([0.0, 0.0, yaw], FOCAL, W, H)
  assert x == pytest.approx(0.5 * W - FOCAL * math.tan(yaw), abs=1e-6)
  assert y == pytest.approx(crops.horizon_row([0.0, 0.0, yaw], FOCAL, W, H))
  assert crops.vanishing_point(None, FOCAL, W, H) == (0.5 * W, 0.5 * H)
  assert crops.vanishing_point([0.0, 0.0, float("nan")], FOCAL, W, H)[0] == pytest.approx(0.5 * W)


# ---------------------------------------------------------------------------
# predicted association
# ---------------------------------------------------------------------------

def _ego(odo):
  return tracker.EgoFrame(odo=odo, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)


def test_association_with_prediction_at_80_kph_straight():
  light = tracker.TrafficLightTracker()
  odo = tracker.EgoOdometry()
  odo.update(0.0, V80, 0.0)  # seed the odometry at t=0
  ego = _ego(odo)
  box = _box_at(120.0)
  light.update(0, [_det(box)], "far", FOCAL, now=0.0,
               predict=lambda t: _predict(t, ego), anchor=(odo.s, odo.psi, 0.0))
  assert len(light.tracks) == 1
  track = light.tracks[0]
  _advance(odo, V80)  # one second -> ds = 22.222 m
  d = 120.0 - odo.s
  box2 = _box_at(d)
  pred = _predict(track, _ego(odo))
  assert (pred[3] - pred[1]) == pytest.approx(box2[3] - box2[1], abs=0.1)
  light.update(1, [_det(box2)], "far", FOCAL, now=1.0,
               predict=lambda t: _predict(t, ego), anchor=(odo.s, odo.psi, 1.0))
  assert len(light.tracks) == 1
  assert light.tracks[0].track_id == track.track_id
  assert light.tracks[0].hits == 2
  assert light.tracks[0].anchor_box == pytest.approx(_det(box2).box)
  assert light.tracks[0].anchor_s == pytest.approx(odo.s)


def test_association_survives_yaw_only_with_prediction():
  odo = tracker.EgoOdometry()
  odo.update(0.0, 0.0, 0.0)
  ego = _ego(odo)
  box = _box_at(60.0)
  stale = _det(box)
  _advance(odo, 0.0, yaw=0.05)  # one second of left turn -> detection shifts right
  assert odo.psi == pytest.approx(0.05)
  shift = FOCAL * math.tan(0.05)
  moved = _det((box[0] + shift, box[1], box[2] + shift, box[3]))
  # Without prediction the stale box cannot associate...
  assert tracker.match_score(stale.box, moved.box) == 0.0
  # ...with prediction it matches the moved detection.
  light = tracker.TrafficLightTracker()
  light.update(0, [stale], "far", FOCAL, now=0.0)
  light.tracks[0].anchor_s = 0.0
  light.tracks[0].anchor_psi = 0.0
  light.update(1, [moved], "far", FOCAL, now=1.0,
               predict=lambda t: _predict(t, ego), anchor=(odo.s, odo.psi, 1.0))
  assert len(light.tracks) == 1
  assert light.tracks[0].hits == 2


# ---------------------------------------------------------------------------
# expiry rules
# ---------------------------------------------------------------------------

def test_predicted_box_leaving_the_frame_drops_track():
  light = tracker.TrafficLightTracker()
  box = (1900.0, 600.0, 1920.0, 620.0)
  light.update(0, [_det(box)], "far", FOCAL, now=0.0)
  assert len(light.tracks) == 1
  light.update(1, [], "far", FOCAL, now=0.1,
               predict=lambda t: (2100.0, 600.0, 2120.0, 620.0),
               frame_box=(0.0, 0.0, float(W), float(H)))
  assert light.tracks == []


def test_view_miss_expiry_counts_only_visible_misses_and_resets_on_match():
  light = tracker.TrafficLightTracker()
  box = _box_at(30.0)
  det = _det(box)
  crop = (600.0, 300.0, 1300.0, 700.0)  # predicted centre is inside
  light.update(0, [det], "far", FOCAL, now=0.0, crop_box=crop)
  # misses outside the crop never accumulate
  for frame in range(1, 6):
    light.update(frame, [], "far", FOCAL, now=0.1 * frame, crop_box=(0.0, 0.0, 100.0, 100.0))
  assert len(light.tracks) == 1
  assert light.tracks[0].view_misses == 0
  # two visible misses, then a match resets the counter
  light.update(6, [], "far", FOCAL, now=0.7, crop_box=crop)
  light.update(7, [], "far", FOCAL, now=0.8, crop_box=crop)
  assert light.tracks[0].view_misses == 2
  light.update(8, [det], "far", FOCAL, now=0.9, crop_box=crop)
  assert light.tracks[0].view_misses == 0
  # three consecutive visible misses drop the track
  for frame in (9, 10, 11):
    light.update(frame, [], "far", FOCAL, now=0.1 * frame, crop_box=crop)
  assert light.tracks == []


# ---------------------------------------------------------------------------
# classify_view
# ---------------------------------------------------------------------------

def test_classify_view_uses_first_track_and_far_geometry():
  pipe = pipeline.FramePipeline(None, None)
  far_geom = pipeline.FramePipeline.geometry("far", 600.0, W, H, FOCAL)
  assert pipe.classify_view(far_geom) == "near"  # no tracks
  track = _anchored_track(_box_at(60.0))
  pipe.tracker.tracks.append(track)
  pipe.last_track_id = track.track_id
  assert pipe.classify_view(far_geom) == "far"
  big = _box_at(4.0)
  track.box = big
  track.anchor_box = big
  assert pipe.classify_view(far_geom) == "near"


def test_classify_view_uses_predicted_box_when_ego_present():
  pipe = pipeline.FramePipeline(None, None)
  far_geom = pipeline.FramePipeline.geometry("far", 600.0, W, H, FOCAL)
  box = _box_at(8.0)  # 115.8 px high: fits the 320 px far crop
  track = _anchored_track(box)
  pipe.tracker.tracks.append(track)
  pipe.last_track_id = track.track_id
  assert pipe.classify_view(far_geom) == "far"
  odo = tracker.EgoOdometry()
  _advance(odo, 3.0)  # one second -> s = 3 m, k = 8/(8-3) = 1.6
  assert odo.s == pytest.approx(3.0)
  pipe.ego = _ego(odo)
  predicted = _predict(track, pipe.ego)
  assert predicted[3] > far_geom.y0 + far_geom.h  # predicted box outgrew the crop
  assert pipe.classify_view(far_geom) == "near"
