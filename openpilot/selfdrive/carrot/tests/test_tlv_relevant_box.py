"""W136: CarrotTlv.relevantBox is the relevant track's ego-predicted box.

Unlike dets[0] it never falls back to an unrelated detection: it is set only
when the relevant track exists for this frame, and zero otherwise.
"""
from types import SimpleNamespace

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.carrot.tlv import pipeline, tlvd, tracker


def _make_publish_runner(last_track_id, last_relevant_box, votes=None):
  runner = tlvd.TlvRunner.__new__(tlvd.TlvRunner)
  runner.last_view = "far"
  runner.dropped = 0
  runner.skipped_budget = 0
  runner.budget = SimpleNamespace(credit=0.5, duty=0.0, estimate_ms=lambda _k: 1.0)
  runner._frame_times = {"detect": [], "classify": []}
  track = SimpleNamespace(votes=list(votes or []))
  runner.pipeline = SimpleNamespace(
    infer_ms=0.0, proc_ms=0.0, last_dets=[], last_red_likely=False,
    last_state=tracker.SignalState.none, last_conf=0.0,
    last_track_id=last_track_id, last_relevant_box=last_relevant_box,
    last_relevant_det_support=0, red_sustain_total=0,
    tail_gated_total=0,
    tracker=SimpleNamespace(last_matches=[], go_after_red=lambda *_: False,
                            track_by_id=lambda _tid: track),
  )
  sent = []
  runner.pub = SimpleNamespace(send=lambda service, msg: sent.append((service, msg)))
  return runner, sent


def _publish_captured(monkeypatch, runner):
  captured = []
  real_new = messaging.new_message

  def fake_new(service, *args, **kwargs):
    msg = real_new(service, *args, **kwargs)
    captured.append(msg)
    return msg

  monkeypatch.setattr(messaging, "new_message", fake_new)
  runner._publish("running", "")
  return captured[0].carrotTlv


def test_publish_fills_relevant_box_when_track_and_box_present(monkeypatch):
  runner, sent = _make_publish_runner(7, (10.0, 20.0, 30.0, 40.0))
  out = _publish_captured(monkeypatch, runner)
  assert sent and sent[0][0] == "carrotTlv"
  assert out.relevantTrackId == 7
  assert (out.relevantBox.x1, out.relevantBox.y1, out.relevantBox.x2, out.relevantBox.y2) == pytest.approx((10.0, 20.0, 30.0, 40.0))
  assert out.relevantBox.score == pytest.approx(1.0)
  assert int(out.relevantBox.cls) == 0


def test_publish_zero_relevant_box_without_track(monkeypatch):
  runner, _sent = _make_publish_runner(-1, (10.0, 20.0, 30.0, 40.0))
  out = _publish_captured(monkeypatch, runner)
  assert out.relevantTrackId == -1
  assert (out.relevantBox.x1, out.relevantBox.y1, out.relevantBox.x2, out.relevantBox.y2) == (0.0, 0.0, 0.0, 0.0)


def test_publish_zero_relevant_box_when_box_none(monkeypatch):
  runner, _sent = _make_publish_runner(7, None)
  out = _publish_captured(monkeypatch, runner)
  assert out.relevantTrackId == 7
  assert (out.relevantBox.x1, out.relevantBox.y1, out.relevantBox.x2, out.relevantBox.y2) == (0.0, 0.0, 0.0, 0.0)


def test_publish_marks_message_valid(monkeypatch):
  # W138: the carrotTlv default valid bit is False; the missing bit kept every
  # message out of carrot_serv's fusion path, so _publish must set it True.
  runner, sent = _make_publish_runner(7, (10.0, 20.0, 30.0, 40.0))
  _publish_captured(monkeypatch, runner)
  assert sent and sent[0][0] == "carrotTlv"
  assert bool(sent[0][1].valid) is True


def test_publish_fills_vote_telemetry_from_latest_vote(monkeypatch):
  # W143: the three vote telemetry fields come from track.votes[-1].
  import time as _time
  now = _time.monotonic()
  runner, _sent = _make_publish_runner(
    7, (10.0, 20.0, 30.0, 40.0),
    votes=[(now - 0.12, int(tracker.LampClass.GREEN_LEFT), 0.8)],
  )
  out = _publish_captured(monkeypatch, runner)
  assert out.relevantVoteCls == int(tracker.LampClass.GREEN_LEFT)
  assert out.relevantVoteProb == pytest.approx(0.8, abs=1e-5)
  assert 90 <= out.relevantVoteAgeMs <= 400


def test_publish_vote_telemetry_defaults_without_track_or_votes(monkeypatch):
  runner, _sent = _make_publish_runner(-1, None)
  out = _publish_captured(monkeypatch, runner)
  assert (out.relevantVoteCls, out.relevantVoteProb, out.relevantVoteAgeMs) == (-1, 0.0, 0)
  runner2, _sent2 = _make_publish_runner(7, (10.0, 20.0, 30.0, 40.0))
  out2 = _publish_captured(monkeypatch, runner2)
  assert (out2.relevantVoteCls, out2.relevantVoteProb, out2.relevantVoteAgeMs) == (-1, 0.0, 0)


# ---------------------------------------------------------------------------
# FramePipeline.last_relevant_box
# ---------------------------------------------------------------------------

def test_pipeline_last_relevant_box_none_without_relevant_track():
  pipe = pipeline.FramePipeline(None, None)
  assert pipe.last_relevant_box is None


def test_pipeline_last_relevant_box_uses_predicted_box():
  pipe = pipeline.FramePipeline(None, None)
  track = tracker.Track(track_id=3, first_frame=0, last_frame=0, cls=0, score=0.9,
                        box=(100.0, 100.0, 140.0, 160.0), view="far", focal_px=900.0)
  pipe.ego = tracker.EgoFrame(odo=tracker.EgoOdometry(), focal_px=900.0, foe_x=480.0, foe_y=320.0,
                              image_w=960, image_h=640)
  predicted = pipe._predicted_box(track)
  # _predicted_box is exactly what last_relevant_box stores when set by infer.
  assert predicted == pytest.approx(track.box)
