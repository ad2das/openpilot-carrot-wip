"""W175 onset-strict red: a NEW red needs RED/RED_LEFT votes at p >= 0.6, and
RED_YELLOW is neither red nor a green veto at onset. The pipeline applies that
rule only while the state machine is not red and no red-family value was
published in the last ONSET_GATE_S; a red already in progress keeps today's
rules. Default arguments reproduce the pre-W175 behaviour exactly.
"""
import numpy as np

from openpilot.selfdrive.carrot.tlv import decode, pipeline, tracker

W, H = 1928, 1208
RED = int(tracker.LampClass.RED)
RED_LEFT = int(tracker.LampClass.RED_LEFT)
RED_YELLOW = int(tracker.LampClass.RED_YELLOW)
GREEN = int(tracker.LampClass.GREEN)
# A box near the image centre (path lateral ~0) close enough to be relevant.
BOX = (940.0, 50.0, 990.0, 90.0)


def _far_geom():
  return pipeline.FramePipeline.geometry("far", 600.0, W, H, 2648.0)


class _ScriptedBackend:
  def __init__(self, frames):
    self.frames = list(frames)

  def infer(self, model_input):
    return self.frames.pop(0) if self.frames else (
      np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, dtype=np.int64))

  def close(self):
    pass


def _candidate_track(box=BOX, votes=((0.4, RED, 0.8),)):
  t = tracker.Track(track_id=7, first_frame=0, last_frame=0, cls=RED,
                    score=0.9, box=box, view="far", focal_px=2648.0)
  for v in votes:
    t.votes.append(v)
  return t


def _track(votes):
  """A bare track with exactly these (t, cls, p) votes and nothing else."""
  t = tracker.Track(track_id=1, first_frame=0, last_frame=0, cls=0, score=1.0,
                    box=(0.0, 0.0, 1.0, 1.0), view="far", focal_px=1.0)
  for v in votes:
    t.votes.append(v)
  return t


def _votes(now, cls, prob, offsets):
  return tuple((now + off, cls, prob) for off in offsets)


# --------------------------------------------------------------- Track rules
def test_red_yellow_alone_confirms_by_default_but_not_at_onset():
  now = 10.0
  t = _track(_votes(now, RED_YELLOW, 0.9, (-0.3, -0.2, -0.1)))
  assert t.classifier_confirmed(now) == (RED_YELLOW, 1.0)
  assert t.classifier_red_likely(now) is True
  assert t.classifier_confirmed(now, onset_strict=True) is None
  assert t.classifier_red_likely(now, onset_strict=True) is False


def test_weak_red_votes_do_not_confirm_at_onset():
  now = 10.0
  t = _track(_votes(now, RED, 0.5, (-0.3, -0.2, -0.1)))
  assert t.classifier_confirmed(now) == (RED, 1.0)
  assert t.classifier_red_likely(now) is False
  assert t.classifier_confirmed(now, onset_strict=True) is None


def test_strong_red_votes_confirm_at_onset():
  now = 10.0
  t = _track(_votes(now, RED, 0.7, (-0.3, -0.2, -0.1)))
  assert t.classifier_confirmed(now) == (RED, 1.0)
  assert t.classifier_red_likely(now) is True
  assert t.classifier_confirmed(now, onset_strict=True) == (RED, 1.0)
  assert t.classifier_red_likely(now, onset_strict=True) is True


def test_two_of_three_strong_red_votes_do_not_confirm_at_onset():
  now = 10.0
  t = _track(((now - 0.3, RED, 0.7), (now - 0.2, RED, 0.5), (now - 0.1, RED, 0.7)))
  assert t.classifier_confirmed(now) == (RED, 1.0)
  assert t.classifier_confirmed(now, onset_strict=True) is None


def test_red_left_at_the_prob_floor_is_onset_red():
  now = 10.0
  t = _track(_votes(now, RED_LEFT, 0.6, (-0.3, -0.2, -0.1)))
  assert t.classifier_confirmed(now) == (RED_LEFT, 1.0)
  assert t.classifier_confirmed(now, onset_strict=True) == (RED_LEFT, 1.0)


def test_red_yellow_does_not_veto_green_at_onset():
  now = 10.0
  green = _votes(now, GREEN, 0.9, (-1.0, -0.7, -0.4, -0.1))
  votes = green + ((now - 0.05, RED_YELLOW, 0.7),)
  t = _track(votes)
  assert t.classifier_confirmed(now) is None
  assert t.classifier_confirmed(now, onset_strict=True) == (GREEN, 1.0)


def test_green_confirm_is_unchanged_at_onset():
  now = 10.0
  t = _track(_votes(now, GREEN, 0.9, (-1.0, -0.7, -0.4, -0.1)))
  assert t.classifier_confirmed(now) == (GREEN, 1.0)
  assert t.classifier_confirmed(now, onset_strict=True) == (GREEN, 1.0)


def test_red_likely_latest_vote_rules_at_onset():
  now = 10.0
  cases = (
    (RED_YELLOW, 0.84, True, False),
    (RED, 0.84, True, True),
    (RED_LEFT, 0.59, False, False),
    (RED_LEFT, 0.60, True, True),
  )
  for cls, prob, expected_default, expected_strict in cases:
    t = _track(((now - 0.1, cls, prob),))
    assert t.classifier_red_likely(now) is expected_default, (cls, prob)
    assert t.classifier_red_likely(now, onset_strict=True) is expected_strict, (cls, prob)


# --------------------------------------------------------------- gate helper
def test_onset_strict_active():
  now = 10.0
  S = tracker.SignalState
  assert tracker.onset_strict_active(S.none, None, now) is True
  assert tracker.onset_strict_active(S.red, None, now) is False
  assert tracker.onset_strict_active(S.redLeft, None, now) is False
  # Exactly ONSET_GATE_S since the publish: the window has closed.
  assert tracker.onset_strict_active(S.none, 8.0, now) is False
  assert tracker.onset_strict_active(S.none, 7.999, now) is True
  assert tracker.onset_strict_active(S.green, 9.5, now) is False


# --------------------------------------------------------------- pipeline
def test_red_yellow_onset_candidate_does_not_publish_red():
  """PA: fresh pipeline, RYEL-only votes: no red state and no redLikely."""
  from openpilot.selfdrive.carrot.tlv import backends
  now = 10.0
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(),
                                classifier=backends.StubClassifier(cls=RED_YELLOW, prob=0.9))
  pipe.tracker.tracks = [_candidate_track(votes=_votes(now, RED_YELLOW, 0.9, (-0.3, -0.2, -0.1)))]
  out = pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out.state != tracker.SignalState.red
  assert out.red_likely is False
  assert pipe._last_red_pub_s is None


def test_red_yellow_onset_publishes_red_once_the_gate_is_off():
  """PB: a red-family publish 1.0 s ago (inside ONSET_GATE_S) keeps identity."""
  from openpilot.selfdrive.carrot.tlv import backends
  now = 10.0
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(),
                                classifier=backends.StubClassifier(cls=RED_YELLOW, prob=0.9))
  pipe._last_red_pub_s = now - 1.0
  pipe.tracker.tracks = [_candidate_track(votes=_votes(now, RED_YELLOW, 0.9, (-0.3, -0.2, -0.1)))]
  out = pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out.state == tracker.SignalState.red


def test_red_yellow_keeps_a_red_already_in_progress():
  """PC: machine already red; RYEL votes neither cut the red nor restart it."""
  from openpilot.selfdrive.carrot.tlv import backends
  now = 10.0
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(),
                                classifier=backends.StubClassifier(cls=RED_YELLOW, prob=0.9))
  pipe.machine.state = tracker.SignalState.red
  pipe.machine.conf = 0.75
  pipe.machine.last_lamp_time = now
  pipe._last_red_confirm_s = now
  pipe.last_conf = 0.75
  pipe.tracker.tracks = [_candidate_track(votes=_votes(now, RED_YELLOW, 0.9, (-0.3, -0.2, -0.1)))]
  out = pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out.state == tracker.SignalState.red
  # A red already in progress is not onset-strict.
  assert pipe._last_red_pub_s == now


def test_last_red_pub_s_counts_the_republished_red():
  """PD: a red-family result republished every loop counts as published."""
  from openpilot.selfdrive.carrot.tlv import backends
  now = 10.0
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(),
                                classifier=backends.StubClassifier(cls=RED, prob=0.9))
  pipe.tracker.tracks = [_candidate_track(votes=_votes(now, RED, 0.7, (-0.3, -0.2, -0.1)))]
  out = pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out.state == tracker.SignalState.red
  # Stamped BEFORE this infer from what was already published, so the frame that
  # first publishes red leaves it None.
  assert pipe._last_red_pub_s is None

  out2 = pipe.infer("far", _far_geom(), image_width=W, now=now + 0.5, detect=False)
  assert out2.state == tracker.SignalState.red
  assert pipe._last_red_pub_s == now + 0.5

  # The red is gone (state none, no redLikely, no tracks): the stamp holds.
  pipe.machine.state = tracker.SignalState.none
  pipe.last_red_likely = False
  pipe.tracker.tracks = []
  pipe.infer("far", _far_geom(), image_width=W, now=now + 3.0, detect=False)
  assert pipe._last_red_pub_s == now + 0.5
  assert tracker.onset_strict_active(pipe.machine.state, pipe._last_red_pub_s, now + 3.0) is True
