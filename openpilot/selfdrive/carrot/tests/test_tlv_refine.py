"""W134 Addendum A tests: template re-localisation (refine.py) + the livePose
yaw fix in tlvd._update_odometry, plus the classify-crop refinement path."""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import crops, pipeline, refine, tracker

W, H = crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT
FOCAL = crops.FOCAL_REFERENCE_PX

cv2 = pytest.importorskip("cv2")


def _luma_with_lamp(cx, cy, hw=8, hh=6, base=60, value=210, seed=0):
  rng = np.random.default_rng(seed)
  img = np.clip(base + rng.integers(-6, 7, (H, W)), 0, 255).astype(np.uint8)
  img[cy - hh:cy + hh, cx - hw:cx + hw] = value
  return img


def _track(box, track_id=1):
  t = tracker.Track(track_id=track_id, first_frame=0, last_frame=0, cls=0, score=0.9,
                    box=box, view="far", focal_px=FOCAL)
  t.anchor_box = box
  return t


# -- refine: capture + accept ------------------------------------------------

def test_refine_locks_a_pure_translation():
  rel = refine.TemplateRelocalizer()
  luma0 = _luma_with_lamp(900, 600)
  t = _track((892.0, 594.0, 908.0, 606.0))
  assert rel.refresh(luma0, t)
  # Lamp moved +7 px right in the next frame; prediction is the old box.
  luma1 = _luma_with_lamp(907, 600)
  got = rel.refine(luma1, t, t.box)
  assert got is not None
  cx = 0.5 * (got[0] + got[2])
  assert cx == pytest.approx(907.0, abs=3.0)


def test_refine_scaling_lamp_still_locks():
  rel = refine.TemplateRelocalizer()
  luma0 = _luma_with_lamp(900, 600, hw=8, hh=6)
  t = _track((892.0, 594.0, 908.0, 606.0))
  rel.refresh(luma0, t)
  # Lamp grows to hw=12, hh=9 and moves +4 px right; predicted box scaled 1.5x.
  luma1 = _luma_with_lamp(904, 600, hw=12, hh=9)
  pred = (892.0 - 6, 594.0 - 3, 908.0 + 6, 606.0 + 3)
  got = rel.refine(luma1, t, pred)
  assert got is not None
  cx = 0.5 * (got[0] + got[2])
  assert cx == pytest.approx(904.0, abs=3.0)


def test_refine_no_drift_over_a_sequence():
  """Only detector refreshes change the template; a light moving 2 px/frame is
  tracked without accumulating error (the template never updates from refine)."""
  rel = refine.TemplateRelocalizer()
  cx = 900
  luma = _luma_with_lamp(cx, 600)
  t = _track((cx - 8.0, 594.0, cx + 8.0, 606.0))
  rel.refresh(luma, t)
  box = t.box
  for step in range(1, 6):
    cx = 900 + 2 * step
    luma = _luma_with_lamp(cx, 600)
    got = rel.refine(luma, t, box)
    assert got is not None
    box = got
  assert 0.5 * (box[0] + box[2]) == pytest.approx(910.0, abs=3.0)


def test_refine_false_lock_rejected_when_template_absent():
  rel = refine.TemplateRelocalizer()
  luma0 = _luma_with_lamp(900, 600)
  t = _track((892.0, 594.0, 908.0, 606.0))
  rel.refresh(luma0, t)
  flat = np.full((H, W), 60, np.uint8)  # no lamp anywhere
  assert rel.refine(flat, t, t.box) is None


def test_refine_rejects_ambiguous_two_lamp_window():
  """Two identical lamps in the search window give two equal NCC peaks, so the
  margin gate rejects (safety for adjacent signal heads)."""
  rel = refine.TemplateRelocalizer()
  luma0 = _luma_with_lamp(900, 600)
  t = _track((892.0, 594.0, 908.0, 606.0))
  rel.refresh(luma0, t)
  luma1 = _luma_with_lamp(900, 600)          # lamp 1 (the template source)
  luma1[594:606, 916:932] = 210              # lamp 2, identical, +16 px (in window)
  got = rel.refine(luma1, t, t.box)
  assert got is None


def test_refine_window_clamped_to_max():
  rel = refine.TemplateRelocalizer()
  luma = _luma_with_lamp(900, 600)
  t = _track((892.0, 594.0, 908.0, 606.0))
  rel.refresh(luma, t)
  # A huge predicted box would make a huge window; the clamp keeps it <= 160.
  big_pred = (-4000.0, -4000.0, 5000.0, 5000.0)
  got = rel.refine(luma, t, big_pred)
  # Either it locks near the lamp or rejects, but it must not raise/allocate.
  assert got is None or (got[2] - got[0]) >= 0


# -- refine: cv2-missing path ------------------------------------------------

def test_relocalizer_disabled_without_cv2(monkeypatch):
  monkeypatch.setattr(refine, "cv2", None)
  monkeypatch.setattr(refine, "CV2_AVAILABLE", False)
  rel = refine.TemplateRelocalizer()
  rel.enabled = False
  luma = _luma_with_lamp(900, 600)
  t = _track((892.0, 594.0, 908.0, 606.0))
  assert rel.refresh(luma, t) is False
  assert rel.refine(luma, t, t.box) is None


# -- pipeline: luma capture + classify uses refined box ----------------------

class _FakeEgo:
  pass


def _ego(odo):
  return tracker.EgoFrame(odo=odo, focal_px=FOCAL, foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)


def test_pipeline_capture_luma_reallocates_on_size_change():
  pipe = pipeline.FramePipeline(None, None)
  y1 = np.zeros((1208, 1928), np.uint8)
  y1[:, 100:110] = 200
  pipe.capture_luma(y1, 1928)
  assert pipe.luma.shape == (1208, 1928)
  assert pipe.luma[0, 105] == 200
  assert int(pipe.luma[:, 1928:].size) == 0 if pipe.luma.shape[1] == 1928 else True
  # New size -> reallocated
  y2 = np.full((760, 1344), 55, np.uint8)
  pipe.capture_luma(y2, 1344)
  assert pipe.luma.shape == (760, 1344)


def test_pipeline_relocalize_uses_luma_and_reanchors():
  pipe = pipeline.FramePipeline(None, None)
  luma = _luma_with_lamp(900, 600)
  pipe.capture_luma(luma, W)
  t = _track((892.0, 594.0, 908.0, 606.0))
  pipe.tracker.tracks.append(t)
  pipe.last_track_id = t.track_id
  pipe.relocalizer.refresh(pipe.luma, t)
  odo = tracker.EgoOdometry()
  for i in range(6):
    odo.update(0.2 * i, 10.0, 0.0)
  pipe.ego = _ego(odo)
  # Move the lamp; classify-only relocalise should adopt it.
  luma2 = _luma_with_lamp(906, 600)
  pipe.capture_luma(luma2, W)
  pipe._relocalize(now=2.0, detect=False)
  assert pipe.refine_locks == 1
  assert 0.5 * (t.box[0] + t.box[2]) == pytest.approx(906.0, abs=3.0)
  assert t.anchor_t == pytest.approx(2.0)


def test_pipeline_detect_refreshes_only_matched_templates():
  pipe = pipeline.FramePipeline(None, None)
  luma = _luma_with_lamp(900, 600)
  pipe.capture_luma(luma, W)
  matched = _track((892.0, 594.0, 908.0, 606.0), track_id=1)
  unmatched = _track((1200.0, 500.0, 1216.0, 512.0), track_id=2)
  pipe.tracker.tracks.extend([matched, unmatched])
  pipe.tracker.last_matches = [(None, 1)]  # only track 1 matched
  pipe.relocalizer.refresh(pipe.luma, matched)
  pipe.relocalizer.refresh(pipe.luma, unmatched)
  old_tpl = pipe.relocalizer._templates[2]
  pipe._relocalize(now=1.0, detect=True)
  # Track 2's template is untouched: a detect frame refreshes only matched
  # tracks, and track 2 (unmatched) is refined but never re-templated.
  assert pipe.relocalizer._templates.get(2) is old_tpl


# -- Addendum C: votes only from verified boxes ------------------------------

class _FakeClassifier:
  enabled = True

  def __init__(self):
    self.batches = []

  def classify(self, batch):
    self.batches.append(batch)
    return np.zeros((batch.shape[0], 8), dtype=np.float32)


def test_classify_only_votes_for_verified_tracks():
  """A track with a rejected/skipped refine gets NO crop and NO vote this frame
  (W134 Addendum C, safety)."""
  pipe = pipeline.FramePipeline(None, None, classifier=_FakeClassifier())
  luma = _luma_with_lamp(900, 600)
  pipe.capture_luma(luma, W)
  verified = _track((892.0, 594.0, 908.0, 606.0), track_id=1)
  unverified = _track((1200.0, 500.0, 1216.0, 512.0), track_id=2)
  pipe.tracker.tracks.extend([verified, unverified])
  pipe.last_track_id = 1
  pipe.relocalizer.refresh(pipe.luma, verified)
  # Only track 1 has a template; track 2's refine is skipped (no template).
  odo = tracker.EgoOdometry()
  for i in range(6):
    odo.update(0.2 * i, 10.0, 0.0)
  pipe.ego = _ego(odo)
  pipe._relocalize(now=2.0, detect=False)
  assert pipe._verified_ids == {1}
  geom = pipeline.FramePipeline.geometry("far", 600.0, W, H, FOCAL)
  recorded = []
  orig = pipe.tracker.record_votes

  def spy(now, votes):
    recorded.extend(votes)
    return orig(now, votes)

  pipe.tracker.record_votes = spy
  pipe._classify_tracks(now=2.0, geom=geom)
  voted_ids = {v[0] for v in recorded}
  assert 1 in voted_ids and 2 not in voted_ids


def test_classify_only_no_votes_when_nothing_verified():
  pipe = pipeline.FramePipeline(None, None, classifier=_FakeClassifier())
  luma = _luma_with_lamp(900, 600)
  pipe.capture_luma(luma, W)
  # A track with no template and no detector match -> refine skipped.
  t = _track((892.0, 594.0, 908.0, 606.0), track_id=1)
  pipe.tracker.tracks.append(t)
  pipe.last_track_id = 1
  odo = tracker.EgoOdometry()
  for i in range(6):
    odo.update(0.2 * i, 10.0, 0.0)
  pipe.ego = _ego(odo)
  pipe._relocalize(now=2.0, detect=False)
  assert pipe._verified_ids == set()
  geom = pipeline.FramePipeline.geometry("far", 600.0, W, H, FOCAL)
  pipe._classify_tracks(now=2.0, geom=geom)
  assert len(t.votes) == 0


# -- yaw fix: tlvd._update_odometry ------------------------------------------

class _FakeAv:
  def __init__(self, z, valid=True):
    self.z = z
    self.valid = valid


class _FakeLivePose:
  def __init__(self, z, valid=True):
    self.angularVelocityDevice = _FakeAv(z, valid)


def _runner_with_sm(monkeypatch, *, z=None, valid=True, seen=True, vego=10.0, yaw_car=None, age=0.0):
  import openpilot.selfdrive.carrot.tlv.tlvd as tlvd
  runner = object.__new__(tlvd.TlvRunner)
  runner.odo = tracker.EgoOdometry()

  class _SM:
    def __init__(self):
      self.seen = {"livePose": seen, "carState": True}
      self.valid = {"livePose": valid, "carState": True}
      self.updated = {"carState": True}
      self.recv_time = {"livePose": 0.0}
      self._lp = _FakeLivePose(z, valid)
      self._cs = types.SimpleNamespace(vEgo=vego, yawRate=yaw_car)

    def __getitem__(self, key):
      return self._lp if key == "livePose" else self._cs

  runner.sm = _SM()
  monkeypatch.setattr(tlvd.time, "monotonic", lambda: age)
  return runner


def test_yaw_pipeline_livepose_left_is_plus(monkeypatch):
  # Addendum (i): livePose z = -0.1 rad/s for 1 s -> psi ~ +0.1 (left).
  runner = _runner_with_sm(monkeypatch, z=-0.1, age=0.0)
  for i in range(6):  # t = 0.0 .. 1.0, dt = 0.2 each
    runner.sm.recv_time["livePose"] = 0.0  # age 0, always fresh
    runner._update_odometry(i * 0.2)
  assert runner.odo.psi == pytest.approx(0.1, abs=0.001)


def test_yaw_ignores_carstate_when_no_livepose(monkeypatch):
  # Addendum (ii): carState.yawRate = 30 (deg/s style), no livePose -> psi stays 0.
  runner = _runner_with_sm(monkeypatch, z=None, seen=False, yaw_car=30.0)
  for i in range(1, 6):
    runner._update_odometry(i * 0.2)
  assert runner.odo.psi == pytest.approx(0.0)


def test_yaw_stale_or_invalid_livepose_not_used(monkeypatch):
  # Addendum (iii): a stale or invalid livePose must not update the yaw.
  runner = _runner_with_sm(monkeypatch, z=-0.5, valid=False, vego=10.0)
  runner._update_odometry(0.0)  # seed
  runner.odo._last_now = 0.0
  for i in range(1, 6):
    runner._update_odometry(i * 0.2)
  assert runner.odo.psi == pytest.approx(0.0)

  runner2 = _runner_with_sm(monkeypatch, z=-0.5, valid=True, age=5.0, vego=10.0)
  runner2.sm.recv_time["livePose"] = 0.0  # age 5 s > 0.5 s
  for i in range(1, 6):
    runner2._update_odometry(i * 0.2)
  assert runner2.odo.psi == pytest.approx(0.0)


def test_yaw_bias_straight_road_keeps_track_id(monkeypatch):
  # Addendum (iv): static light + carState.yawRate bias 0.77 must NOT churn IDs.
  # Straight road: livePose z ~ 0 (no yaw), carState yaw = 0.77 (old trap value).
  runner = _runner_with_sm(monkeypatch, z=0.0, vego=10.0, yaw_car=0.77)
  light = tracker.TrafficLightTracker()
  odo = runner.odo
  box = (900.0, 590.0, 918.0, 610.0)
  det = tracker.Detection(cls=0, score=0.9, x1=box[0], y1=box[1], x2=box[2], y2=box[3])
  ids = []
  for k in range(5):
    t = 1.0 * k
    runner.sm.recv_time["livePose"] = t - 0.1
    runner._update_odometry(t)
    ego = _ego(odo)
    predict = lambda tk, e=ego: tracker.predict_box(
      tk, e.odo, focal_px=e.focal_px, foe_x=e.foe_x, foe_y=e.foe_y, image_w=e.image_w, image_h=e.image_h)
    light.update(k, [det], "far", FOCAL, now=t, predict=predict,
                 anchor=(odo.s, odo.psi, t), crop_box=(0.0, 0.0, W, H), frame_box=(0.0, 0.0, W, H))
    ids.append(light.last_matches[0][1] if light.last_matches else -1)
  assert len(set(ids)) == 1 and ids[0] == 1
