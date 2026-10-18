"""W139c red-continuity tests: an already-confirmed red survives a relevance
switch while the new relevant track is still voting red, bounded at 10 s, never
created from none, never blocking a green/yellow/left transition.

These drive FramePipeline.infer directly. A red family vote on the newly relevant
track is a "strong red vote" when the latest vote is red with p >= 0.6 within the
0.6 s window (Track.classifier_red_likely), the same check the continuity uses.
"""
import numpy as np

from openpilot.selfdrive.carrot.tlv import crops, decode, pipeline, tracker

W, H = 1928, 1208
RED = int(tracker.LampClass.RED)
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


def _frame(box=BOX, cls=RED, score=0.9):
  return (np.asarray([box], np.float32), np.asarray([score], np.float32), np.asarray([cls]))


def _empty():
  return (np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, dtype=np.int64))


def _candidate_track(box=BOX, votes=((0.4, RED, 0.8),)):
  t = tracker.Track(track_id=7, first_frame=0, last_frame=0, cls=RED,
                    score=0.9, box=box, view="far", focal_px=2648.0)
  for v in votes:
    t.votes.append(v)
  return t


def _pipe_with_confirmed_red(now):
  """Pipeline whose machine is already confirmed red, no relevant track yet.

  A classifier is installed so the candidate-only branch runs (the new relevant
  track has a strong red vote but no confirmed color, i.e. lamp_state is None) —
  the exact relevance-switch situation the continuity targets.
  """
  from openpilot.selfdrive.carrot.tlv import backends
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(),
                                classifier=backends.StubClassifier(cls=RED, prob=0.9))
  pipe.machine.state = tracker.SignalState.red
  pipe.machine.conf = 0.75
  pipe.machine.last_lamp_time = now
  pipe._last_red_confirm_s = now
  return pipe


def test_relevance_switch_sustains_red():
  now = 10.0
  pipe = _pipe_with_confirmed_red(now)
  pipe.last_conf = 0.75
  pipe.tracker.tracks = [_candidate_track(votes=((now - 0.1, RED, 0.8),))]
  out = pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out.state == tracker.SignalState.red
  assert pipe.red_sustain_total == 1


def test_sustain_bounded_by_last_confirm():
  now = 10.0
  # 9.9 s since the last confirm and a fresh red vote: still sustained.
  pipe = _pipe_with_confirmed_red(now - 9.9)
  pipe.last_conf = 0.75
  pipe.tracker.tracks = [_candidate_track(votes=((now - 0.1, RED, 0.8),))]
  out = pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out.state == tracker.SignalState.red
  assert pipe.red_sustain_total == 1

  # 10.1 s since the last confirm: continuity does not refresh; the machine's own
  # 2 s off-timeout then expires the red.
  pipe2 = _pipe_with_confirmed_red(now - 10.1)
  pipe2.last_conf = 0.75
  pipe2.tracker.tracks = [_candidate_track(votes=((now - 0.1, RED, 0.8),))]
  out2 = pipe2.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out2.state == tracker.SignalState.none
  assert pipe2.red_sustain_total == 0


def test_no_sustain_for_weak_or_stale_or_nonred_votes():
  now = 10.0
  cases = [
    ((now - 0.1, RED, 0.5),),          # p < 0.6
    ((now - 0.1, GREEN, 0.9),),        # green vote
    ((now - 0.9, RED, 0.9),),          # older than 0.6 s
  ]
  for votes in cases:
    pipe = _pipe_with_confirmed_red(now)
    pipe.last_conf = 0.75
    pipe.tracker.tracks = [_candidate_track(votes=votes)]
    # detect=False with no luma adds no votes, so the latest vote stays as given.
    pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
    assert pipe.red_sustain_total == 0, votes


def test_continuity_never_creates_red_from_none():
  now = 10.0
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(), classifier=None)
  # machine none, but a red-voting candidate is relevant and a stale confirm time
  pipe.machine.state = tracker.SignalState.none
  pipe._last_red_confirm_s = now - 1.0
  pipe.tracker.tracks = [_candidate_track(votes=((now - 0.1, RED, 0.8),))]
  out = pipe.infer("far", _far_geom(), image_width=W, now=now, detect=False)
  assert out.state == tracker.SignalState.none
  assert pipe.red_sustain_total == 0


def test_green_confirm_still_transitions_from_red():
  # Three red detect frames confirm red, then a green confirm (4 in 1.6 s via
  # classifier votes) must move the machine to green even while continuity could
  # hold red.
  from openpilot.selfdrive.carrot.tlv import backends
  stub = backends.StubClassifier(cls=RED, prob=0.9)
  pipe = pipeline.FramePipeline(_ScriptedBackend([_frame()]), decode.PlaceholderDecoder(), classifier=stub)
  pipe.machine.state = tracker.SignalState.red
  pipe.machine.conf = 0.75
  pipe.machine.last_lamp_time = 0.0
  pipe._last_red_confirm_s = 0.0
  # Feed a green-family tracked light directly.
  t = tracker.Track(track_id=1, first_frame=0, last_frame=0, cls=GREEN,
                    score=0.9, box=BOX, view="far", focal_px=2648.0)
  for i in range(4):
    t.votes.append((0.0 + i * 0.357, GREEN, 0.9))
  pipe.tracker.tracks = [t]
  out = pipe.infer("far", _far_geom(), image_width=W, now=1.1, detect=True)
  assert out.state in (tracker.SignalState.green, tracker.SignalState.greenLeft)


def test_green_confirm_at_field_cadence_within_1_3s():
  from openpilot.selfdrive.carrot.tlv import backends
  # W143: votes arrive at 2.8 Hz (0.357 s spacing); the machine must reach green
  # within 1.3 s of the FIRST green vote.
  stub = backends.StubClassifier(cls=GREEN, prob=0.9)
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(), classifier=stub)
  pipe.machine.state = tracker.SignalState.red
  pipe.machine.conf = 0.75
  pipe.machine.last_lamp_time = 0.0
  pipe._last_red_confirm_s = 0.0
  t = tracker.Track(track_id=1, first_frame=0, last_frame=0, cls=GREEN,
                    score=0.9, box=BOX, view="far", focal_px=2648.0)
  t.red_confirmed_s = 0.0
  pipe.tracker.tracks = [t]
  state = None
  for i in range(4):
    vote_t = i * 0.357
    t.votes.append((vote_t, GREEN, 0.9))
    out = pipe.infer("far", _far_geom(), image_width=W, now=vote_t + 0.05, detect=False)
    state = out.state
  assert state in (tracker.SignalState.green, tracker.SignalState.greenLeft)
  # The last vote at 3*0.357 = 1.071 s is within 1.3 s of the first (0.0).
  assert 1.071 <= 1.3
  assert pipe.tracker.go_after_red(1, 1.2) is True


def test_green_on_a_different_track_does_not_give_go_after_red():
  from openpilot.selfdrive.carrot.tlv import backends
  stub = backends.StubClassifier(cls=GREEN, prob=0.9)
  pipe = pipeline.FramePipeline(_ScriptedBackend([]), decode.PlaceholderDecoder(), classifier=stub)
  pipe.machine.state = tracker.SignalState.red
  pipe.machine.conf = 0.75
  pipe.machine.last_lamp_time = 0.0
  pipe._last_red_confirm_s = 0.0
  red_track = _candidate_track(box=(300.0, 50.0, 340.0, 90.0), votes=((0.0, RED, 0.9),))
  green_track = tracker.Track(track_id=99, first_frame=0, last_frame=0, cls=GREEN,
                              score=0.9, box=BOX, view="far", focal_px=2648.0)
  for i in range(4):
    green_track.votes.append((i * 0.357, GREEN, 0.9))
  pipe.tracker.tracks = [red_track, green_track]
  pipe.infer("far", _far_geom(), image_width=W, now=1.2, detect=False)
  assert pipe.tracker.go_after_red(99, 1.2) is False
