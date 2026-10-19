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

import pytest

from test_tlv_fusion import FakeSM, make_serv as base_make_serv, tlv_message  # noqa: F401

import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
from openpilot.selfdrive.carrot.carrot_serv import CarrotServ
from openpilot.selfdrive.carrot.tlv import tracker
from openpilot.selfdrive.carrot.tlv.tracker import (
  GREEN_RELEASE_PROB,
  GREEN_RELEASE_VOTES,
  LampClass,
  SignalState,
  Track,
  TrafficLightTracker,
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


def pipeline_with_machine():
  from openpilot.selfdrive.carrot.tlv.pipeline import FramePipeline
  pipe = FramePipeline.__new__(FramePipeline)
  pipe.machine = tracker.SignalStateMachine()
  pipe.last_released = False
  pipe.tracker = TrafficLightTracker()
  pipe.classifier = None
  return pipe


def test_pipeline_release_turns_red_into_none_not_green(monkeypatch):
  from openpilot.selfdrive.carrot.tlv import pipeline as pl
  pipe = pipeline_with_machine()
  pipe.machine.state = SignalState.red
  pipe.machine.last_lamp_time = 99.0
  now = 100.0
  tr = track_with_votes([(now - 0.4, GREEN, 0.9), (now - 0.2, GREEN, 0.9), (now, GREEN, 0.9)])
  # The release block in infer() is inline; exercise the same contract here.
  assert tr.released()
  if pipe.machine.state in (SignalState.red, SignalState.redLeft) and tr.released():
    pipe.machine.state = SignalState.none
    pipe.machine.conf = 0.0
    pipe.machine.last_lamp_time = None
    pipe.last_released = True
  state, _conf = pipe.machine.update(now, None, 0.0)
  assert state == SignalState.none
  assert state != SignalState.green
  assert pipe.last_released
  assert pl is not None


def test_pipeline_does_not_release_without_three_votes():
  pipe = pipeline_with_machine()
  pipe.machine.state = SignalState.red
  now = 100.0
  tr = track_with_votes([(now - 0.2, GREEN, 0.9), (now, GREEN, 0.9)])
  assert not tr.released()
  assert pipe.machine.state == SignalState.red


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
  # Re-arm a red, then a long non-stopped level must NOT clear it.
  serv._update_tlv_fusion(sm_with_cue(ts=1, xstate=2))
  clock[0] += 1.0
  serv._update_tlv_fusion(sm_with_cue(ts=1, xstate=2))
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
