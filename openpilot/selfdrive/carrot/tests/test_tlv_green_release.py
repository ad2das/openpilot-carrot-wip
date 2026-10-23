"""W150/WREL4: release a held TLV red on confident green votes, gated by the cue.

Two layers are tested separately:

  tracker/pipeline - the release itself: the newest GREEN_RELEASE_VOTES
    relevant-track votes must be green-family/left at prob >= GREEN_RELEASE_PROB
    with no newer red vote, and it turns the state machine to none (never to
    green). WREL4 moved the constants from 3 votes at 0.80 to 4 votes at 0.75;
    release-only tests space their votes 0.15 s apart (span 0.45 s), because at
    0.2 s apart a 4-vote run would also pass the W143 green confirm and move the
    machine off red.
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
RED_LEFT = int(LampClass.RED_LEFT)


def track_with_votes(votes):
  tr = Track(track_id=1, first_frame=0, last_frame=0, cls=GREEN, score=0.9,
             box=(0.0, 0.0, 1.0, 1.0), view='far', focal_px=1.0)
  tr.votes = [(t, c, p) for t, c, p in votes]
  return tr


def green_run(now, prob=0.9):
  """GREEN_RELEASE_VOTES green votes ending at `now`, 0.15 s apart.

  The 0.45 s span stays below GREEN_CONFIRM_MIN_SPAN_S, so a plain release run
  never doubles as a W143 green confirm.
  """
  return [(now - 0.15 * k, GREEN, prob) for k in reversed(range(GREEN_RELEASE_VOTES))]


def test_release_requires_four_votes():
  now = 100.0
  tr = track_with_votes(green_run(now)[:-1])
  assert not tr.released()
  tr.votes.append((now, GREEN, 0.9))
  assert tr.released()


def test_release_accepts_green_family_and_left():
  now = 100.0
  tr = track_with_votes([(now - 0.45, GREEN, 0.85), (now - 0.30, LEFT, 0.9),
                         (now - 0.15, GREEN_LEFT, 0.95), (now, GREEN, 0.9)])
  assert tr.released()


def test_release_rejects_a_low_probability_vote():
  now = 100.0
  tr = track_with_votes([(now - 0.45, GREEN, 0.9), (now - 0.30, GREEN, GREEN_RELEASE_PROB - 0.01),
                         (now - 0.15, GREEN, 0.9), (now, GREEN, 0.9)])
  assert not tr.released()


def test_release_accepts_075():
  now = 100.0
  tr = track_with_votes(green_run(now, prob=0.75))
  assert tr.released()
  # 0.749 in any single slot is below GREEN_RELEASE_PROB -> no release.
  for k in range(4):
    votes = green_run(now, prob=0.75)
    votes[k] = (votes[k][0], votes[k][1], 0.749)
    assert not track_with_votes(votes).released()


def test_release_rejects_yellow_in_the_run():
  now = 100.0
  tr = track_with_votes([(now - 0.45, GREEN, 0.9), (now - 0.30, int(LampClass.YELLOW), 0.9),
                         (now - 0.15, GREEN, 0.9), (now, GREEN, 0.9)])
  assert not tr.released()


def test_a_newer_red_vote_blocks_the_release():
  now = 100.0
  tr = track_with_votes(green_run(now))
  assert tr.released()
  tr.votes.append((now + 0.15, RED, 0.9))
  assert not tr.released()
  # One to three greens after the red keep the red inside the newest 4 -> blocked.
  for k in (2, 3, 4):
    tr.votes.append((now + 0.15 * k, GREEN, 0.9))
  assert not tr.released()
  # The fourth green after the red leaves the red older than the run -> release.
  tr.votes.append((now + 0.75, GREEN, 0.9))
  assert tr.released()


def test_release_count_constant_is_four():
  assert GREEN_RELEASE_VOTES == 4
  assert GREEN_RELEASE_PROB == 0.75


def test_route477_dusk_votes_release_at_03643(monkeypatch):
  """The field miss: route 477 dusk greens voted 0.756-0.776 and never released.

  Votes are the relevant-track (tid 17) sequence logged 2026-10-23 17:12, with
  t in seconds after 17:12:00. N=4/P=0.75 releases on the 17:12:03.643 vote; the
  landed N=3/P=0.80 never released at all.
  """
  import openpilot.selfdrive.carrot.tlv.tracker as tracker

  votes = [
    (2.862, RED, 0.972),
    (2.949, RED_LEFT, 0.901),
    (3.160, GREEN, 0.776),
    (3.343, GREEN, 0.768),
    (3.446, GREEN, 0.766),
    (3.643, GREEN, 0.765),
    (3.842, GREEN, 0.756),
    (4.044, GREEN, 0.795),
    (4.254, GREEN, 0.710),
  ]
  released_at = []
  tr = track_with_votes([])
  for vote in votes:
    tr.votes.append(vote)
    if tr.released():
      released_at.append(vote[0])
  assert released_at == [3.643, 3.842, 4.044]
  # Not yet after the 03.446 vote: the fourth green had not landed.
  tr.votes = [v for v in votes if v[0] <= 3.446]
  assert not tr.released()
  # The landed constants reproduce the field miss exactly.
  monkeypatch.setattr(tracker, "GREEN_RELEASE_VOTES", 3)
  monkeypatch.setattr(tracker, "GREEN_RELEASE_PROB", 0.80)
  tr.votes = []
  for vote in votes:
    tr.votes.append(vote)
    assert not tr.released()


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
  # without detections; green_run()'s 0.45 s span does not confirm a green either.
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
  pipe, out = _release_pipe(now, green_run(now))
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
  """W150b: p < GREEN_RELEASE_PROB breaks the release run; it must stay sticky.

  The low-probability vote lands after the W143 green window, so it breaks the
  release run without also confirming a green (4 greens in 1.6 s would).
  """
  now = 100.0
  pipe, out = _release_pipe(now, green_run(now))
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
  pipe, out = _release_pipe(now, green_run(now))
  assert out.released
  tr = pipe.tracker.tracks[0]
  tr.votes.clear()                       # fewer than GREEN_RELEASE_VOTES left
  pipe.machine.last_lamp_time = now
  out2 = _step(pipe, now + 0.1)
  assert out2.released
  assert out2.state in (SignalState.red, SignalState.redLeft)


def test_one_red_vote_drops_the_sticky_release():
  now = 100.0
  pipe, out = _release_pipe(now, green_run(now))
  assert out.released
  pipe.tracker.tracks[0].votes.append((now + 0.2, RED, 0.9))
  pipe.machine.last_lamp_time = now
  out2 = _step(pipe, now + 0.2)
  assert not out2.released


def test_a_track_change_drops_the_sticky_release():
  now = 100.0
  pipe, out = _release_pipe(now, green_run(now))
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
  pipe, out = _release_pipe(now, green_run(now))
  assert out.released
  pipe.machine.state = SignalState.none
  out2 = _step(pipe, now + 0.1)
  assert not out2.released


def _geom():
  from openpilot.selfdrive.carrot.tlv import pipeline as pl
  return pl.FramePipeline.geometry("far", 600.0, 1928, 1208, 2648.0)


def test_no_cue_keeps_the_red_until_the_usual_off_timeout():
  """Spec: without a cue the red must end exactly when it does today.

  The release flag goes True as soon as the GREEN_RELEASE_VOTES green votes
  land, but nothing is cleared: the machine keeps red and expires off_timeout_s
  after the last lamp, which is the pre-W150 behaviour.
  """
  now = 100.0
  votes = green_run(now)
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


def test_no_release_flag_without_four_votes():
  now = 100.0
  pipe, out = _release_pipe(now, green_run(now)[:-1])
  assert not out.released
  assert out.state in (SignalState.red, SignalState.redLeft)


def test_no_release_flag_when_machine_is_not_red():
  now = 100.0
  pipe, out = _release_pipe(now, green_run(now), machine_state=SignalState.none)
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
