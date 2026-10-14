"""Single-class detector confirmation + tiered relevance gate (W117).

The bundled r8s50a detector is single-class (class 0 for every lamp): its
M-of-N detector belt would publish RED for green lights (W115 A6), so
FramePipeline confirms single-class tracks through lamp-classifier votes only.
The relevance gate is tiered: the nearest-path light within 120 m first, and
only when none qualifies, the (120, 200] m far tier (W115 A5).

Tests reuse the W116 analytic head builder (test_tlv_decode), the W101 track
builder (test_tlv_tracker) and the scripted pipeline helpers
(test_tlv_classifier).
"""
import hashlib

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import backends, decode, pipeline, tracker
from openpilot.selfdrive.carrot.tlv.decode import RawHeadDecoder
from openpilot.selfdrive.carrot.tlv.tracker import (RELEVANT_FAR_MAX_DISTANCE_M,
                                                    RELEVANT_MAX_DISTANCE_M, select_relevant,
                                                    select_relevant_tiered)
from test_tlv_classifier import W, ScriptedBackend, _far_geom, _red_frame
from test_tlv_decode import R3B_NAMES, _base, _pack
from test_tlv_tracker import _track_at, box_at

BUNDLED_DETECTOR_SHA256 = "5f108f62567557e0ac35902a444812230b3da4bd9f0729163a8ff4c3bb909f94"
RED = int(tracker.LampClass.RED)
GREEN = int(tracker.LampClass.GREEN)


class _StubDecoder(decode.PlaceholderDecoder):
  """PlaceholderDecoder with a pinned num_classes (1 = single-class model)."""

  def __init__(self, nc: int) -> None:
    super().__init__()
    self.num_classes = nc


def _make_pipe(frames, classifier, nc: int):
  backend = ScriptedBackend(frames)
  return pipeline.FramePipeline(backend, _StubDecoder(nc), classifier=classifier), backend


def test_single_class_belt_never_confirms():
  # 20 detect frames of one class-0 track: the RED belt alone is satisfied.
  pipe, backend = _make_pipe([_red_frame()] * 20, backends.NullClassifier(), nc=1)
  far = _far_geom()
  for i in range(20):
    out = pipe.infer("far", far, image_width=W, now=0.1 * i, detect=True)
    assert pipe._confirmed(0.1 * i) == []
    assert out.state == tracker.SignalState.none
  track = pipe.tracker.tracks[0]
  assert track.cls == 0
  assert backend.calls == 20
  # The same history WOULD satisfy the multi-class belt; single-class never uses it.
  assert track.confirmed(pipe.frame_idx - 1) is not None

  # Control: an 8-class decoder on identical input confirms RED through the belt.
  pipe, _backend = _make_pipe([_red_frame()] * 20, backends.NullClassifier(), nc=8)
  for i in range(20):
    pipe.infer("far", far, image_width=W, now=0.1 * i, detect=True)
  conf = pipe._confirmed(2.0)
  assert len(conf) == 1
  assert conf[0][1] == RED


def test_single_class_votes_confirm_red_and_green():
  far = _far_geom()

  # 3 RED votes within 0.6 s confirm RED.
  pipe, _backend = _make_pipe([_red_frame()], backends.NullClassifier(), nc=1)
  pipe.infer("far", far, image_width=W, now=0.0, detect=True)
  track = pipe.tracker.tracks[0]
  assert track.cls == 0
  for when in (0.1, 0.2, 0.3):
    pipe.tracker.record_votes(when, [(track.track_id, RED, 0.9)])
  conf = pipe._confirmed(0.4)
  assert len(conf) == 1 and conf[0][0] is track and conf[0][1] == RED

  # A fresh track: 6 GREEN votes within 1.2 s confirm GREEN.
  pipe, _backend = _make_pipe([_red_frame()], backends.NullClassifier(), nc=1)
  pipe.infer("far", far, image_width=W, now=0.0, detect=True)
  track = pipe.tracker.tracks[0]
  assert track.cls == 0
  for k in range(6):
    pipe.tracker.record_votes(0.1 + 0.1 * k, [(track.track_id, GREEN, 0.9)])
  conf = pipe._confirmed(0.7)
  assert len(conf) == 1 and conf[0][0] is track and conf[0][1] == GREEN


def test_tiered_gate():
  kwargs = dict(image_width=1928, path_x=(0.0, 200.0), path_y=(0.0, 0.0), left_intent=False)
  near = _track_at(100.0)
  far_track = _track_at(150.0)
  # Both on the path: the 100 m one wins; the 150 m one is only a fallback.
  chosen = select_relevant_tiered([(near, RED, 0.75), (far_track, RED, 0.75)], 0, **kwargs)
  assert chosen is not None and chosen[0] is near
  # Only the 150 m one: the far tier (120, 200] m picks it up.
  chosen = select_relevant_tiered([(far_track, RED, 0.75)], 0, **kwargs)
  assert chosen is not None and chosen[0] is far_track
  # 210 m is beyond the far tier; 1 m is below the near minimum.
  assert select_relevant_tiered([(_track_at(210.0), RED, 0.75)], 0, **kwargs) is None
  assert select_relevant_tiered([(_track_at(1.0), RED, 0.75)], 0, **kwargs) is None


def test_select_relevant_unchanged():
  # The original gate keeps its default: a 150 m track stays out of 120 m.
  assert RELEVANT_FAR_MAX_DISTANCE_M == 200.0
  kwargs = dict(image_width=1928, path_x=(0.0, 200.0), path_y=(0.0, 0.0), left_intent=False)
  entry = [(_track_at(150.0), RED, 0.75)]
  assert select_relevant(entry, 0, **kwargs) is None
  assert select_relevant(entry, 0, **kwargs, max_distance_m=RELEVANT_MAX_DISTANCE_M) is None


def test_decoder_num_classes():
  dec = RawHeadDecoder(R3B_NAMES)
  assert dec.num_classes is None
  dec.decode(None, "far")  # malformed input leaves it unchanged
  assert dec.num_classes is None
  dec.decode(_pack(_base(1), R3B_NAMES), "far")
  assert dec.num_classes == 1
  dec8 = RawHeadDecoder(R3B_NAMES)
  dec8.decode(_pack(_base(8), R3B_NAMES), "far")
  assert dec8.num_classes == 8


def test_bundled_detector_is_single_class():
  pytest.importorskip("onnxruntime")
  backend, dec = backends.make_backend("onnx", backends.BUNDLED_DETECTOR)
  try:
    spec = backend._session.get_inputs()[0]
    shape = tuple(int(d) for d in spec.shape)
    dtype = np.uint8 if "uint8" in spec.type else np.float32
    raw = backend.infer(np.zeros(shape, dtype))
    assert dec.decode(raw, "far") is not None
    assert dec.num_classes == 1
    digest = hashlib.sha256()
    with open(backends.BUNDLED_DETECTOR, "rb") as f:
      for chunk in iter(lambda: f.read(1 << 20), b""):
        digest.update(chunk)
    assert digest.hexdigest() == BUNDLED_DETECTOR_SHA256
  finally:
    backend.close()


# ---------------------------------------------------------------------------
# W117B: the far tier must not jump over an unconfirmed near light
# ---------------------------------------------------------------------------

def _track_at_cx(distance_m, cx, *, focal=2648.0):
  """Track at distance_m laterally at cx (box_at centres on image centre by default)."""
  track = tracker.Track(track_id=9, first_frame=0, last_frame=0, cls=RED, score=0.9,
                        box=box_at(distance_m, focal=focal, cx=cx), view="far", focal_px=focal)
  track.anchor_box = track.box
  return track


def test_far_tier_blocked_by_unconfirmed_near_track():
  kwargs = dict(image_width=1928, path_x=(0.0, 200.0), path_y=(0.0, 0.0), left_intent=False)
  near = _track_at(80.0)                 # unconfirmed, on the path
  far_track = _track_at(170.0)           # confirmed RED at the next intersection
  assert select_relevant_tiered([(far_track, RED, 0.75)], 0,
                                near_candidates=[(near, RED, 0.9), (far_track, RED, 0.75)],
                                **kwargs) is None


def test_far_tier_used_without_near_track():
  kwargs = dict(image_width=1928, path_x=(0.0, 200.0), path_y=(0.0, 0.0), left_intent=False)
  far_track = _track_at(170.0)
  chosen = select_relevant_tiered([(far_track, RED, 0.75)], 0,
                                  near_candidates=[(far_track, RED, 0.75)], **kwargs)
  assert chosen is not None and chosen[0] is far_track
  chosen = select_relevant_tiered([(far_track, RED, 0.75)], 0, near_candidates=[], **kwargs)
  assert chosen is not None and chosen[0] is far_track


def test_near_off_path_does_not_block():
  kwargs = dict(image_width=1928, path_x=(0.0, 200.0), path_y=(0.0, 0.0), left_intent=False)
  near_off = _track_at_cx(80.0, 1200.0)  # ~7.1 m lateral > max_lateral_m (4 m)
  far_track = _track_at(170.0)
  chosen = select_relevant_tiered([(far_track, RED, 0.75)], 0,
                                  near_candidates=[(near_off, RED, 0.9), (far_track, RED, 0.75)],
                                  **kwargs)
  assert chosen is not None and chosen[0] is far_track


def test_near_confirmed_wins():
  kwargs = dict(image_width=1928, path_x=(0.0, 200.0), path_y=(0.0, 0.0), left_intent=False)
  near = _track_at(80.0)
  far_track = _track_at(170.0)
  chosen = select_relevant_tiered([(near, RED, 0.75), (far_track, RED, 0.75)], 0,
                                  near_candidates=[(near, RED, 0.9), (far_track, RED, 0.75)],
                                  **kwargs)
  assert chosen is not None and chosen[0] is near
