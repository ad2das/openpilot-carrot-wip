"""W153: standstill stale-box vote for the relevant traffic light.

After a red->green change the lit spot moves inside the housing, so the NCC
refine fails and the track gets no vote until the next detector match (~1 s).
At standstill the camera does not move, so the ego-predicted box IS the stored
box; this lets the relevant track vote on its last verified box.

The vote is recorded ONLY when it is green-family at >= 0.80. A stale-box red or
OFF vote would count as "a red-family vote newer than run_start" and break the
W150b sticky release, which would delay starts versus today.
"""
from __future__ import annotations

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import crops, decode, pipeline, tracker
from openpilot.selfdrive.carrot.tlv.backends import StubClassifier

W, H = crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT
FOCAL = crops.FOCAL_REFERENCE_PX
RED = int(tracker.LampClass.RED)
GREEN = int(tracker.LampClass.GREEN)
GREEN_LEFT = int(tracker.LampClass.GREEN_LEFT)
OFF = int(tracker.LampClass.OFF)
BOX = (900.0, 100.0, 940.0, 120.0)


class _NoDetBackend:
  """Classifier-only frames: the detector is not run (detect=False)."""

  def infer(self, model_input):
    return (np.empty((0, 4), np.float32), np.empty(0, np.float32),
            np.empty(0, dtype=np.int64))

  def close(self):
    pass


def _geom():
  return pipeline.FramePipeline.geometry("far", 600.0, W, H, FOCAL)


def _track(track_id=1, box=BOX):
  tr = tracker.Track(track_id=track_id, first_frame=0, last_frame=0, cls=RED, score=0.9,
                     box=box, view="far", focal_px=FOCAL)
  tr.anchor_box = box
  return tr


def _pipe(cls, prob, *, track_id=1, relevant_id=1, verified_at=0.0, extra_tracks=()):
  pipe = pipeline.FramePipeline(_NoDetBackend(), decode.PlaceholderDecoder(),
                                classifier=StubClassifier(cls=cls, prob=prob))
  tr = _track(track_id)
  tr.last_verified_s = verified_at
  pipe.tracker.tracks = [tr, *extra_tracks]
  pipe.last_track_id = relevant_id
  return pipe, tr


def _run(pipe, times, *, v_ego_kph):
  """Run classify-only frames at the given monotonic times."""
  out = None
  for now in times:
    out = pipe.infer("far", _geom(), image_width=W, now=now, detect=False,
                     v_ego_kph=v_ego_kph)
  return out


def _votes(tr):
  return list(tr.votes)


# -- the filter: only a confident green is recorded ---------------------------

@pytest.mark.parametrize("cls,prob", [(RED, 0.95), (OFF, 0.95), (GREEN, 0.75)])
def test_stale_box_non_confident_or_non_green_is_not_recorded(cls, prob):
  pipe, tr = _pipe(cls, prob, verified_at=0.0)
  # 0.0 -> standstill starts; 0.6 -> 0.6 s of standstill, the rule is armed.
  out = _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  assert out is not None
  assert _votes(tr) == []
  assert pipe.standstill_box_votes == 0


def test_stale_box_confident_green_left_is_recorded():
  pipe, tr = _pipe(GREEN_LEFT, 0.94, verified_at=0.0)
  _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  assert _votes(tr) == [(0.6, GREEN_LEFT, pytest.approx(0.94, abs=1e-3))]
  assert pipe.standstill_box_votes == 1


def test_stale_box_green_is_recorded():
  pipe, tr = _pipe(GREEN, 0.90, verified_at=0.0)
  _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  assert [v[1] for v in _votes(tr)] == [GREEN]


# -- the guards ---------------------------------------------------------------

def test_moving_gets_no_stale_box_vote():
  pipe, tr = _pipe(GREEN_LEFT, 0.99, verified_at=0.0)
  _run(pipe, (0.0, 0.6, 1.2), v_ego_kph=30.0)
  assert _votes(tr) == []


def test_unknown_speed_gets_no_stale_box_vote():
  pipe, tr = _pipe(GREEN_LEFT, 0.99, verified_at=0.0)
  _run(pipe, (0.0, 0.6), v_ego_kph=None)
  assert _votes(tr) == []


def test_non_relevant_track_gets_no_stale_box_vote():
  other = _track(2, box=(1200.0, 100.0, 1240.0, 120.0))
  other.last_verified_s = 0.0
  pipe, tr = _pipe(GREEN_LEFT, 0.99, track_id=1, relevant_id=1,
                   verified_at=0.0, extra_tracks=(other,))
  # Make only track 2 stale: track 1 is verified on this frame, so the rule
  # must not add anything for track 2 even though it is old and alive.
  pipe.last_track_id = 2
  _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  assert _votes(other) == []


def test_verification_older_than_three_seconds_gets_no_vote():
  pipe, tr = _pipe(GREEN_LEFT, 0.99, verified_at=0.0)
  _run(pipe, (4.0, 4.6), v_ego_kph=0.0)
  assert _votes(tr) == []


def test_standstill_shorter_than_half_a_second_gets_no_vote():
  pipe, tr = _pipe(GREEN_LEFT, 0.99, verified_at=0.0)
  _run(pipe, (0.0, 0.4), v_ego_kph=0.0)
  assert _votes(tr) == []


def test_moving_frame_resets_the_standstill_timer():
  pipe, tr = _pipe(GREEN_LEFT, 0.99, verified_at=0.0)
  _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  assert len(_votes(tr)) == 1
  # Speed appears -> the timer restarts; 0.4 s later the rule is not armed.
  _run(pipe, (0.7, 0.9), v_ego_kph=30.0)
  _run(pipe, (1.0,), v_ego_kph=0.0)
  _run(pipe, (1.3,), v_ego_kph=0.0)
  assert len(_votes(tr)) == 1


def test_box_template_anchor_and_expiry_are_untouched():
  pipe, tr = _pipe(GREEN_LEFT, 0.94, verified_at=0.0)
  before_box = tuple(tr.box)
  before_anchor = (tr.anchor_box, tr.anchor_s, tr.anchor_psi, tr.anchor_t)
  before_misses = tr.misses
  before_view_misses = tr.view_misses
  before_verified = tr.last_verified_s
  before_templates = dict(pipe.relocalizer._templates)
  _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  assert tuple(tr.box) == before_box
  assert (tr.anchor_box, tr.anchor_s, tr.anchor_psi, tr.anchor_t) == before_anchor
  assert (tr.misses, tr.view_misses) == (before_misses, before_view_misses)
  assert tr.last_verified_s == before_verified
  assert dict(pipe.relocalizer._templates) == before_templates


def test_a_verified_frame_still_votes_non_green():
  """The filter applies to stale boxes only: a box verified by an accepted
  refine this frame votes RED exactly like before."""
  luma = np.full((H, W), 60, np.uint8)
  x1, y1, x2, y2 = [int(v) for v in BOX]
  luma[y1:y2, x1:x2] = 210
  pipe, tr = _pipe(RED, 0.95, verified_at=0.0)
  pipe.capture_luma(luma, W)
  pipe.relocalizer.refresh(pipe.luma, tr)
  pipe.ego = tracker.EgoFrame(odo=tracker.EgoOdometry(), focal_px=FOCAL,
                              foe_x=964.0, foe_y=600.0, image_w=W, image_h=H)
  pipe.capture_luma(luma, W)
  pipe._relocalize(now=0.6, detect=False)
  assert pipe._verified_ids == {1}          # verified by the refine...
  assert pipe._stale_box_ids == set()       # ...so it is not a stale box
  pipe._classify_tracks(now=0.6, geom=_geom())
  assert [v[1] for v in _votes(tr)] == [RED]
  assert pipe.standstill_box_votes == 0


# -- the sticky release must never be broken by a stale-box vote --------------

def test_sticky_release_survives_a_stale_box_non_green_result():
  pipe, tr = _pipe(RED, 0.95, verified_at=0.0)
  pipe.machine.state = tracker.SignalState.red
  pipe.machine.conf = 0.75
  pipe.machine.last_lamp_time = 0.6
  pipe._last_red_confirm_s = 0.0
  tr.red_confirmed_s = 0.0
  # Three confident green votes: the release run that already exists today.
  tr.votes = [(0.0, GREEN, 0.95), (0.2, GREEN, 0.95), (0.4, GREEN, 0.95)]
  out = _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  # The stale box reads RED at 0.95, and that vote must NOT be recorded.
  assert [v[1] for v in _votes(tr)] == [GREEN, GREEN, GREEN]
  assert out.released
  assert pipe.machine.state == tracker.SignalState.red


def test_stale_box_green_extends_the_release_run():
  pipe, tr = _pipe(GREEN, 0.95, verified_at=0.0)
  pipe.machine.state = tracker.SignalState.red
  pipe.machine.conf = 0.75
  pipe.machine.last_lamp_time = 0.6
  pipe._last_red_confirm_s = 0.0
  tr.red_confirmed_s = 0.0
  tr.votes = [(0.0, GREEN, 0.95), (0.2, GREEN, 0.95)]
  out = _run(pipe, (0.0, 0.6), v_ego_kph=0.0)
  assert [v[1] for v in _votes(tr)] == [GREEN, GREEN, GREEN]
  assert out.released
