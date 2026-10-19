"""W150: release a held TLV red on 3 confident green votes, gated by the model cue.

Two layers are tested separately:

  tracker/pipeline - the release itself: the newest 3 relevant-track votes must
    be green-family/left at prob >= 0.80 with no newer red vote, and it turns
    the state machine to none (never to green).
  carrotServ - the released red is cleared immediately ONLY while the model
    start cue is active (plan trafficState 2, an xState edge out of e2eStopped,
    or trafficSignGreen). Without the cue today's hold must be unchanged.
"""
from types import SimpleNamespace

from test_tlv_fusion import FakeSM, make_serv as base_make_serv, tlv_message

import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
from openpilot.selfdrive.carrot.carrot_serv import CarrotServ
from openpilot.selfdrive.carrot.tlv.tracker import (
  GREEN_RELEASE_PROB,
  GREEN_RELEASE_VOTES,
  LampClass,
  SignalState,
  Track,
)

GREEN = int(LampClass.GREEN)
GREEN_LEFT = int(LampClass.GREEN_LEFT)
LEFT = int(LampClass.LEFT)
RED = int(LampClass.RED)


def track_with_votes(votes):
  tr = Track(track_id=1, first_frame=0, last_frame=0, cls=GREEN, score=0.9,
             box=(0.0, 0.0, 1.0, 1.0), view='far', focal_px=1.0)
  tr.votes = [(t, c, p) for t, c, p in votes]
  return tr


def test_release_requires_three_votes():
  now = 100.0
  tr = track_with_votes([(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9)])
  assert not tr.released()
  tr.votes.append((now, GREEN, 0.9))
  assert tr.released()


def test_release_accepts_green_family_and_left():
  now = 100.0
  tr = track_with_votes([(now - 0.4, GREEN, 0.85), (now - 0.2, LEFT, 0.9), (now, GREEN_LEFT, 0.95)])
  assert tr.released()


def test_release_rejects_a_low_probability_vote():
  now = 100.0
  tr = track_with_votes([(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, GREEN_RELEASE_PROB - 0.01),
                         (now, GREEN, 0.9)])
  assert not tr.released()


def test_release_rejects_yellow_in_the_run():
  now = 100.0
  tr = track_with_votes([(now - 0.4, GREEN, 0.9), (now - 0.2, int(LampClass.YELLOW), 0.9),
                         (now, GREEN, 0.9)])
  assert not tr.released()


def test_a_newer_red_vote_blocks_the_release():
  now = 100.0
  tr = track_with_votes([(now - 0.6, GREEN, 0.9), (now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                         (now, RED, 0.9)])
  # Newest 3 are green,green,red -> no release.
  assert not tr.released()
  # Two greens after the red: the newest 3 are green,green,red -> still blocked.
  tr.votes.append((now + 0.2, GREEN, 0.9))
  assert not tr.released()
  # Third green after the red: the red is older than the run -> release.
  tr.votes.append((now + 0.4, GREEN, 0.9))
  tr.votes.append((now + 0.6, GREEN, 0.9))
  assert tr.released()


def test_release_count_constant_is_three():
  assert GREEN_RELEASE_VOTES == 3


# ---------------------------------------------------------------------------
# Pipeline: the release moves red -> none, never red -> green.
# ---------------------------------------------------------------------------

class _FakeDecoder:
  num_classes = 8


def _release_pipe(now, votes, machine_state=SignalState.red):
  """Real FramePipeline driving infer(): the release must NOT move the machine."""
  import numpy as np
  from openpilot.selfdrive.carrot.tlv import decode, pipeline as pl
  W, H = 1928, 1208

  class _Backend:
    def infer(self, model_input):
      return (np.empty((0, 4), np.float32), np.empty(0, np.float32),
              np.empty(0, dtype=np.int64))

    def close(self):
      pass

  from openpilot.selfdrive.carrot.tlv import backends
  # A classifier selects the candidate-only branch, so the track is relevant
  # without detections; 3 green votes do not yet confirm a green either.
  pipe = pl.FramePipeline(_Backend(), decode.PlaceholderDecoder(),
                          classifier=backends.StubClassifier(cls=RED, prob=0.9))
  pipe.machine.state = machine_state
  pipe.machine.conf = 0.75
  # A red confirmed just now: the machine must hold it through this frame.
  pipe.machine.last_lamp_time = now
  pipe._last_red_confirm_s = now
  tr = Track(track_id=1, first_frame=0, last_frame=0, cls=GREEN, score=0.9,
             box=(940.0, 50.0, 990.0, 90.0), view="far", focal_px=2648.0)
  tr.red_confirmed_s = 0.0
  for v in votes:
    tr.votes.append(v)
  pipe.tracker.tracks = [tr]
  out = pipe.infer("far", pl.FramePipeline.geometry("far", 600.0, W, H, 2648.0),
                   image_width=W, now=now, detect=False)
  return pipe, out


def test_release_flag_set_but_machine_stays_red():
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                                  (now, GREEN, 0.9)])
  assert out.released
  # Published state must stay red: without a cue today's off_timeout still holds.
  assert out.state in (SignalState.red, SignalState.redLeft)
  assert pipe.machine.state in (SignalState.red, SignalState.redLeft)


def _step(pipe, now, detect=False):
  """Run one more infer() on the same pipeline at `now`."""
  from openpilot.selfdrive.carrot.tlv import pipeline as pl
  return pipe.infer("far", pl.FramePipeline.geometry("far", 600.0, 1928, 1208, 2648.0),
                    image_width=1928, now=now, detect=detect)


def test_sticky_survives_a_sub_threshold_green_vote():
  """W150b: p<0.80 breaks the 3-vote run; the release must stay sticky.

  The low-probability vote lands after the W143 green window, so it breaks the
  release run without also confirming a green (4 greens in 1.6 s would).
  """
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                                  (now, GREEN, 0.9)])
  assert out.released
  tr = pipe.tracker.tracks[0]
  later = now + 2.0                      # outside GREEN_CONFIRM_WINDOW_S
  tr.votes.append((later, GREEN, 0.70))
  pipe.machine.last_lamp_time = later     # keep the machine red
  pipe._last_red_confirm_s = later
  out2 = _step(pipe, later)
  assert not tr.released()               # the raw run really is broken
  assert out2.released                   # ... but the release is sticky
  assert out2.state in (SignalState.red, SignalState.redLeft)


def test_sticky_survives_votes_ageing_out():
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                                  (now, GREEN, 0.9)])
  assert out.released
  tr = pipe.tracker.tracks[0]
  tr.votes.clear()                       # fewer than 3 votes left
  pipe.machine.last_lamp_time = now
  out2 = _step(pipe, now + 0.1)
  assert out2.released
  assert out2.state in (SignalState.red, SignalState.redLeft)


def test_one_red_vote_drops_the_sticky_release():
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                                  (now, GREEN, 0.9)])
  assert out.released
  pipe.tracker.tracks[0].votes.append((now + 0.2, RED, 0.9))
  pipe.machine.last_lamp_time = now
  out2 = _step(pipe, now + 0.2)
  assert not out2.released


def test_a_track_change_drops_the_sticky_release():
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                                  (now, GREEN, 0.9)])
  assert out.released
  other = Track(track_id=2, first_frame=0, last_frame=0, cls=GREEN, score=0.9,
                box=(940.0, 50.0, 990.0, 90.0), view="far", focal_px=2648.0)
  other.red_confirmed_s = 0.0
  pipe.tracker.tracks = [other]
  pipe.machine.last_lamp_time = now
  out2 = _step(pipe, now + 0.1)
  assert not out2.released


def test_the_machine_leaving_red_drops_the_sticky_release():
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                                  (now, GREEN, 0.9)])
  assert out.released
  pipe.machine.state = SignalState.none
  out2 = _step(pipe, now + 0.1)
  assert not out2.released


def _geom():
  from openpilot.selfdrive.carrot.tlv import pipeline as pl
  return pl.FramePipeline.geometry("far", 600.0, 1928, 1208, 2648.0)


def test_no_cue_keeps_the_red_until_the_usual_off_timeout():
  """Spec: without a cue the red must end exactly when it does today.

  The release flag goes True as soon as the 3 green votes land, but nothing is
  cleared: the machine keeps red and expires off_timeout_s after the last lamp,
  which is the pre-W150 behaviour.
  """
  now = 100.0
  votes = [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9), (now, GREEN, 0.9)]
  pipe, out = _release_pipe(now, votes)
  assert out.released
  assert out.state in (SignalState.red, SignalState.redLeft)
  # Same pipeline, same votes, but the red was last lit at `now`: off_timeout_s
  # is 2.0 s, so one tick before that keeps red and one after expires it.
  pipe.machine.last_lamp_time = now
  pipe._last_red_confirm_s = now
  pipe.tracker.tracks[0].votes.clear()
  for v in votes:
    pipe.tracker.tracks[0].votes.append(v)
  out_before = pipe.infer("far", _geom(), image_width=1928, now=now + 1.9, detect=False)
  assert out_before.state in (SignalState.red, SignalState.redLeft)
  out_after = pipe.infer("far", _geom(), image_width=1928, now=now + 2.1, detect=False)
  assert out_after.state == SignalState.none


def test_no_release_flag_without_three_votes():
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.2, GREEN, 0.9), (now, GREEN, 0.9)])
  assert not out.released
  assert out.state in (SignalState.red, SignalState.redLeft)


def test_no_release_flag_when_machine_is_not_red():
  now = 100.0
  pipe, out = _release_pipe(now, [(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9),
                                  (now, GREEN, 0.9)], machine_state=SignalState.none)
  # The run is there, but the machine was not red, so this is not a release.
  assert not out.released



# ---------------------------------------------------------------------------
# carrotServ cue gate.
# ---------------------------------------------------------------------------

def make_serv(detect_mode: int = 2) -> CarrotServ:
  serv = base_make_serv(detect_mode)
  serv.tlv_navi_road_category = None
  serv.tlv_navi_road_category_t = -1e9
  serv.tlv_highway_gate = False
  serv.tlv_highway_clear_since = None
  serv.tlv_last_stopped = None
  serv.tlv_xstate_left_at = -1e9
  serv.tlv_green_event_at = -1e9
  serv.tlv_release_clears = 0
  serv.tlv_release_latched = False
  serv.tlv_release_track_id = -1
  return serv


def released_message(*args, **kw):
  msg = tlv_message(*args, **kw)
  msg.relevantReleased = True
  return msg


def sm_with_cue(ts: int | None = None, xstate: int | None = None, green_event: bool = False):
  services = {"carrotTlv": released_message("red", conf=0.9)}
  if ts is not None:
    plan = SimpleNamespace(trafficState=ts, xState=SimpleNamespace(raw=xstate if xstate is not None else 5))
    services["longitudinalPlan"] = plan
  if green_event:
    services["onroadEvents"] = SimpleNamespace(events=[SimpleNamespace(name="trafficSignGreen")])
  return FakeSM(**services)


def test_no_cue_keeps_todays_hold(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  for _ in range(2):
    serv._update_tlv_fusion(sm_with_cue(ts=1, green_event=False))
    clock[0] += 0.05
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  assert serv.tlv_release_clears == 0
  assert serv.tlv_hold_until > clock[0]


def test_cue_clears_the_red_immediately(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  for _ in range(2):
    serv._update_tlv_fusion(sm_with_cue(ts=1))
    clock[0] += 0.05
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  serv._update_tlv_fusion(sm_with_cue(ts=2))
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
  assert serv.tlv_hold_until == 0.0
  assert serv.tlv_det_support == 0
  assert serv.tlv_release_clears == 1


def test_traffic_sign_green_is_a_cue(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  for _ in range(2):
    serv._update_tlv_fusion(sm_with_cue(ts=1))
    clock[0] += 0.05
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  serv._update_tlv_fusion(sm_with_cue(ts=1, green_event=True))
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
  assert serv.tlv_release_clears == 1


def test_xstate_edge_out_of_stopped_is_a_cue(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  # stopped, no cue yet (red needs two fusion calls to arm)
  for _ in range(2):
    serv._update_tlv_fusion(sm_with_cue(ts=1, xstate=5))
    clock[0] += 0.05
  assert serv.traffic_state == 1
  clock[0] += 0.05
  # edge out of e2eStopped -> cue
  serv._update_tlv_fusion(sm_with_cue(ts=1, xstate=2))
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)


def test_xstate_level_does_not_keep_the_gate_open(monkeypatch):
  """xState stays non-stopped after the car moves; that must not be a cue."""
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  serv._update_tlv_fusion(sm_with_cue(ts=1, xstate=5))
  clock[0] += 0.05
  serv._update_tlv_fusion(sm_with_cue(ts=1, xstate=2))   # edge: clears
  clock[0] += 1.0
  # Re-arm a red (release flag gone), then a long non-stopped xState LEVEL must
  # NOT clear it: only the edge out of e2eStopped is a cue.
  level = FakeSM(carrotTlv=tlv_message("red", conf=0.9),
                 longitudinalPlan=SimpleNamespace(trafficState=1,
                                                  xState=SimpleNamespace(raw=2)))
  serv._update_tlv_fusion(level)
  clock[0] += 1.0
  serv._update_tlv_fusion(level)
  assert serv.traffic_state == 1


def test_red_without_release_flag_is_not_cleared_by_a_cue(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  plain = tlv_message("red", conf=0.9)
  sm = FakeSM(carrotTlv=plain,
              longitudinalPlan=SimpleNamespace(trafficState=2, xState=SimpleNamespace(raw=5)))
  serv._update_tlv_fusion(sm)
  clock[0] += 0.05
  serv._update_tlv_fusion(sm)
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)


def test_red_reconfirms_after_a_release(monkeypatch):
  """After the release the normal red confirm must be able to come back."""
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  serv._update_tlv_fusion(sm_with_cue(ts=1))
  clock[0] += 0.05
  serv._update_tlv_fusion(sm_with_cue(ts=2))
  assert serv.traffic_state == 0
  clock[0] += 0.05
  # cue gone, red again: fusion accepts the red and re-arms the hold
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.9)))
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  assert serv.tlv_hold_until > clock[0]


def test_highway_gate_still_wins_over_a_release(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  serv.tlv_navi_road_category = 0
  serv.tlv_navi_road_category_t = clock[0]
  serv._update_tlv_fusion(sm_with_cue(ts=2))
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
  assert serv.tlv_release_clears == 0


def test_w143_green_confirm_is_unchanged(monkeypatch):
  """The existing 4-vote green confirm must still work through the same path."""
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  for _ in range(2):
    serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.9)))
  assert serv.traffic_state == 1
  msg = tlv_message("green", conf=0.9, go_after_red=True)
  msg.relevantReleased = False
  serv._update_tlv_fusion(FakeSM(carrotTlv=msg))
  assert serv.traffic_state == 2


def test_release_flag_defaults_false_on_old_messages():
  msg = tlv_message("red")
  assert not bool(msg.relevantReleased)



# ---------------------------------------------------------------------------
# W150 latch: the red stays lifted across publishes, and only the specified
# events end it.
# ---------------------------------------------------------------------------

def _released(state="red", rel_id=1, conf=0.9):
  return released_message(state, conf=conf, rel_id=rel_id)


def _publish(serv, clock, msg, ts=2, **kw):
  services = {"carrotTlv": msg}
  services["longitudinalPlan"] = SimpleNamespace(
    trafficState=ts, xState=SimpleNamespace(raw=kw.get("xstate", 5)))
  clock[0] += kw.get("dt", 0.05)
  serv._update_tlv_fusion(FakeSM(**services))


def test_latch_keeps_the_red_lifted_after_the_cue_window_expires(monkeypatch):
  """After the 0.3 s cue, a still-releasing track must not re-arm the red."""
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  for _ in range(2):
    _publish(serv, clock, tlv_message("red", conf=0.9), ts=1)
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  _publish(serv, clock, _released("red"), ts=2)
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
  assert serv.tlv_release_latched
  # The cue is long gone, yet tlvd still publishes red (it keeps the machine
  # red), so the latch is what keeps the cleared red from coming back.
  for _ in range(10):
    _publish(serv, clock, _released("red"), ts=0, dt=0.1)
    assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
  assert serv.tlv_release_clears == 1


def test_latch_suppresses_graded_red_likely_too(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  _publish(serv, clock, tlv_message("red", conf=0.9), ts=1)
  _publish(serv, clock, _released("red"), ts=2)
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
  _publish(serv, clock, _released("redLikely", conf=0.9), ts=0)
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)


def test_a_new_red_vote_unlatches_and_red_returns(monkeypatch):
  """release flag False = a newer red vote broke the run."""
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  _publish(serv, clock, tlv_message("red", conf=0.9), ts=1)
  _publish(serv, clock, _released("red"), ts=2)
  assert serv.tlv_release_latched
  _publish(serv, clock, tlv_message("red", conf=0.9))
  assert not serv.tlv_release_latched
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  assert serv.tlv_hold_until > clock[0]


def test_a_track_id_change_unlatches(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  _publish(serv, clock, tlv_message("red", conf=0.9), ts=1)
  _publish(serv, clock, _released("red", rel_id=1), ts=2)
  assert serv.tlv_release_latched
  # A different light is relevant now: the old latch must not suppress it.
  _publish(serv, clock, _released("red", rel_id=2), ts=0)
  assert not serv.tlv_release_latched
  # With no cue the new track's red is accepted as today.
  for _ in range(2):
    _publish(serv, clock, tlv_message("red", conf=0.9, rel_id=2), ts=0)
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)


def test_green_state_unlatches_and_takes_the_w143_path(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  _publish(serv, clock, tlv_message("red", conf=0.9), ts=1)
  _publish(serv, clock, _released("red"), ts=2)
  assert serv.tlv_release_latched
  _publish(serv, clock, tlv_message("green", conf=0.9, go_after_red=True), ts=2)
  assert not serv.tlv_release_latched
  assert (serv.traffic_state, serv.traffic_state_src) == (2, 1)


def test_no_cue_never_latches(monkeypatch):
  """The exact spec case: without a cue the red is accepted as today."""
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  for _ in range(5):
    _publish(serv, clock, _released("red", conf=0.9), ts=1)
  assert not serv.tlv_release_latched
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  assert serv.tlv_release_clears == 0
  assert serv.tlv_hold_until > clock[0]
