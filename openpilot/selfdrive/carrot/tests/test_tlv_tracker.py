"""Tests for the pure carrot TLV tracker / state machine / relevant selection."""
import math

from openpilot.selfdrive.carrot.tlv.tracker import (
  Detection,
  LampClass,
  SignalState,
  SignalStateMachine,
  TrafficLightTracker,
  estimate_distance_m,
  iou,
  lamp_state_for,
  lateral_offset_m,
  match_score,
  path_lateral_m,
  select_relevant,
)

BOX = (100.0, 50.0, 140.0, 90.0)
BOX_SHIFTED = (104.0, 52.0, 144.0, 92.0)


def det(cls=LampClass.RED, score=0.8, box=BOX):
  return Detection(cls=int(cls), score=score, x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def test_iou_and_match_score():
  assert iou(BOX, BOX) == 1.0
  assert iou(BOX, (200.0, 200.0, 240.0, 240.0)) == 0.0
  assert match_score(BOX, BOX) == 1.0
  assert match_score(BOX, BOX_SHIFTED) > 0.0
  assert match_score(BOX, (1000.0, 1000.0, 1040.0, 1040.0)) == 0.0


def test_track_persists_across_views():
  t = TrafficLightTracker()
  matches = t.update(0, [det()], "far", 2648.0)
  assert len(matches) == 1
  first_id = matches[0][1]
  t.update(1, [det(box=BOX_SHIFTED)], "near", 2648.0)
  assert len(t.tracks) == 1
  assert t.tracks[0].track_id == first_id
  assert t.tracks[0].view == "near"
  assert t.tracks[0].hits == 2


def test_red_confirmation_is_3_of_4():
  t = TrafficLightTracker()
  for frame in range(3):
    t.update(frame, [det(LampClass.RED)], "far", 2648.0)
  assert t.tracks[0].confirmed(2) == (int(LampClass.RED), 0.75)
  # A miss inside the 4-frame window still confirms: frames 0,1,3.
  t2 = TrafficLightTracker()
  for frame, dets in ((0, [det(LampClass.RED)]), (1, [det(LampClass.RED)]), (2, []), (3, [det(LampClass.RED)])):
    t2.update(frame, dets, "far", 2648.0)
  assert t2.confirmed(3)[0][1] == int(LampClass.RED)
  # Only 2 of the last 4 frames is not enough.
  t3 = TrafficLightTracker()
  for frame, dets in ((0, [det(LampClass.RED)]), (1, [det(LampClass.RED)]), (2, []), (3, [])):
    t3.update(frame, dets, "far", 2648.0)
  assert t3.confirmed(3) == []


def test_green_confirmation_is_6_of_8():
  t = TrafficLightTracker()
  for frame in range(5):
    t.update(frame, [det(LampClass.GREEN)], "far", 2648.0)
  assert t.confirmed(4) == []
  t.update(5, [det(LampClass.GREEN)], "far", 2648.0)
  confirmed = t.confirmed(5)
  assert len(confirmed) == 1
  assert confirmed[0][1] == int(LampClass.GREEN)
  assert confirmed[0][2] == 0.75  # 6/8


def test_track_dropped_after_misses():
  t = TrafficLightTracker(max_misses=3)
  t.update(0, [det()], "far", 2648.0)
  for frame in range(1, 5):
    t.update(frame, [], "far", 2648.0)
  assert t.tracks == []


def test_lamp_state_mapping():
  assert lamp_state_for(LampClass.RED) == SignalState.red
  assert lamp_state_for(LampClass.RED_YELLOW) == SignalState.red
  assert lamp_state_for(LampClass.RED_LEFT) == SignalState.redLeft
  assert lamp_state_for(LampClass.GREEN) == SignalState.green
  assert lamp_state_for(LampClass.GREEN_LEFT) == SignalState.greenLeft
  assert lamp_state_for(LampClass.LEFT) == SignalState.left
  assert lamp_state_for(LampClass.OFF) is None


def test_state_machine_korean_order():
  m = SignalStateMachine(off_timeout_s=2.0)
  assert m.update(0.0, SignalState.red, 0.75) == (SignalState.red, 0.75)
  # Red cannot jump to yellow; the order is green -> yellow -> red.
  assert m.update(0.1, SignalState.yellow, 0.7) == (SignalState.red, 0.75)
  # Red -> green is the normal start trigger.
  assert m.update(0.2, SignalState.green, 0.75) == (SignalState.green, 0.75)
  # Green -> red directly is rejected (yellow must come first).
  assert m.update(0.3, SignalState.red, 0.75) == (SignalState.green, 0.75)
  assert m.update(0.4, SignalState.yellow, 0.7) == (SignalState.yellow, 0.7)
  assert m.update(0.5, SignalState.red, 0.75) == (SignalState.red, 0.75)
  # Red -> left / redLeft are allowed by spec.
  assert m.update(0.6, SignalState.left, 0.75) == (SignalState.left, 0.75)
  assert m.update(0.7, SignalState.red, 0.75) == (SignalState.red, 0.75)
  assert m.update(0.8, SignalState.redLeft, 0.75) == (SignalState.redLeft, 0.75)
  # The red family rejects yellow; only green -> yellow -> red is legal.
  assert m.update(0.9, SignalState.yellow, 0.7) == (SignalState.redLeft, 0.75)


def test_state_machine_timeout_to_none():
  m = SignalStateMachine(off_timeout_s=2.0)
  m.update(0.0, SignalState.red, 0.75)
  assert m.state == SignalState.red
  assert m.update(1.9, None)[0] == SignalState.red
  assert m.update(2.1, None) == (SignalState.none, 0.0)


def test_estimate_distance_and_lateral():
  assert math.isclose(estimate_distance_m(35.0, 2648.0), 2648.0 * 0.35 / 35.0)
  assert estimate_distance_m(0.0, 2648.0) == float("inf")
  assert math.isclose(lateral_offset_m(1928 / 2, 1928, 2648.0, 30.0), 0.0)
  assert lateral_offset_m(1928 / 2 + 100, 1928, 2648.0, 26.48) > 0.0
  assert path_lateral_m(20.0, (), ()) == 0.0
  assert math.isclose(path_lateral_m(20.0, [0.0, 10.0, 30.0], [0.0, -1.0, 1.0]), 0.0)


def make_confirmed_track(box, cls=LampClass.GREEN, focal=2648.0):
  t = TrafficLightTracker()
  for frame in range(8):
    t.update(frame, [det(cls, box=box)], "far", focal)
  confirmed = t.confirmed(7)
  assert len(confirmed) == 1
  return confirmed


def test_select_relevant_prefers_path_center():
  # Track A: right in front (centre x = image centre). Track B: off to the side.
  center_box = (944.0, 400.0, 984.0, 440.0)
  side_box = (1500.0, 400.0, 1540.0, 440.0)
  confirmed = make_confirmed_track(center_box) + make_confirmed_track(side_box)
  chosen = select_relevant(
    confirmed, 8, image_width=1928, path_x=(0.0, 10.0, 30.0, 60.0), path_y=(0.0, 0.0, 0.0, 0.0), left_intent=False,
  )
  assert chosen is not None
  assert abs(chosen[0].cx - 964.0) < 1.0
  # A track far off the path is ignored.
  assert select_relevant(
    make_confirmed_track(side_box), 8, image_width=1928,
    path_x=(0.0, 30.0), path_y=(0.0, 0.0), left_intent=False,
  ) is None


def test_select_relevant_left_intent_prefers_arrow():
  plain_box = (944.0, 400.0, 984.0, 440.0)        # on the path
  arrow_box = (1100.0, 400.0, 1140.0, 440.0)      # ~1.2 m right of the path
  plain = make_confirmed_track(plain_box, cls=LampClass.GREEN)
  arrow = make_confirmed_track(arrow_box, cls=LampClass.GREEN_LEFT)
  without_intent = select_relevant(
    plain + arrow, 8, image_width=1928, path_x=(0.0, 30.0), path_y=(0.0, 0.0), left_intent=False,
  )
  assert without_intent is not None and abs(without_intent[0].cx - 964.0) < 1.0
  with_intent = select_relevant(
    plain + arrow, 8, image_width=1928, path_x=(0.0, 30.0), path_y=(0.0, 0.0), left_intent=True,
  )
  assert with_intent is not None
  assert with_intent[1] == int(LampClass.GREEN_LEFT)


def _track_with_votes(votes, cls=LampClass.RED, box=BOX):
  t = TrafficLightTracker()
  t.update(0, [det(cls, box=box)], "far", 2648.0, now=0.0)
  track = t.tracks[0]
  for when, vote_cls, prob in votes:
    t.record_votes(when, [(track.track_id, int(vote_cls), prob)])
  return t, track


def test_classifier_votes_confirm_red_by_time():
  t, track = _track_with_votes([(0.0, LampClass.RED, 0.9), (0.2, LampClass.RED, 0.9)])
  assert track.classifier_confirmed(0.2) is None  # only 2 votes inside 0.6 s
  t.record_votes(0.4, [(track.track_id, int(LampClass.RED), 0.9)])
  assert track.classifier_confirmed(0.4) == (int(LampClass.RED), 1.0)
  # Votes age out of the red window: no fresh votes, no confirmation.
  assert track.classifier_confirmed(1.2) is None


def test_classifier_green_vote_blocks_red_confirmation():
  t = TrafficLightTracker()
  t.update(0, [det(LampClass.RED)], "far", 2648.0, now=0.0)
  track = t.tracks[0]
  red, green = int(LampClass.RED), int(LampClass.GREEN)
  t.record_votes(0.0, [(track.track_id, red, 0.9)])
  t.record_votes(0.2, [(track.track_id, red, 0.9)])
  t.record_votes(0.4, [(track.track_id, green, 0.8)])
  assert track.classifier_confirmed(0.4) is None
  # Once the green ages out of the red window, three fresh reds confirm; two
  # fresh reds are not enough.
  t.record_votes(1.0, [(track.track_id, red, 0.9)])
  t.record_votes(1.2, [(track.track_id, red, 0.9)])
  assert track.classifier_confirmed(1.2) is None
  t.record_votes(1.4, [(track.track_id, red, 0.9)])
  assert track.classifier_confirmed(1.4) == (red, 1.0)


def test_classifier_green_needs_six_votes_in_1_2s():
  t = TrafficLightTracker()
  t.update(0, [det(LampClass.GREEN)], "far", 2648.0, now=0.0)
  track = t.tracks[0]
  green = int(LampClass.GREEN)
  for i in range(5):
    t.record_votes(i * 0.2, [(track.track_id, green, 0.8)])
  assert track.classifier_confirmed(0.8) is None
  t.record_votes(1.0, [(track.track_id, green, 0.8)])
  assert track.classifier_confirmed(1.0) == (green, 1.0)


def test_classifier_red_likely_uses_latest_strong_vote():
  t, track = _track_with_votes([(0.0, LampClass.RED, 0.9)])
  assert track.classifier_red_likely(0.2) is True
  assert track.classifier_red_likely(0.7) is False  # vote older than the window
  t.record_votes(1.0, [(track.track_id, int(LampClass.RED), 0.5)])  # weak vote
  assert track.classifier_red_likely(1.0) is False
  t.record_votes(1.2, [(track.track_id, int(LampClass.GREEN), 0.9)])  # latest is green
  assert track.classifier_red_likely(1.2) is False


def test_votes_keep_a_track_alive_beyond_misses():
  plain = TrafficLightTracker(max_misses=2)
  plain.update(0, [det()], "far", 2648.0, now=0.0)
  for frame in range(1, 6):
    plain.update(frame, [], "far", 2648.0, now=0.05 * frame)
  assert plain.tracks == []  # no votes: the plain miss budget applies

  voted = TrafficLightTracker(max_misses=2)
  voted.update(0, [det()], "far", 2648.0, now=0.0)
  track_id = voted.tracks[0].track_id
  for frame in range(1, 6):
    voted.update(frame, [], "far", 2648.0, now=0.05 * frame)
    voted.record_votes(0.05 * frame, [(track_id, int(LampClass.RED), 0.9)])
  assert len(voted.tracks) == 1 and voted.tracks[0].track_id == track_id
  # Vote expiry eventually drops a stale track that the detector lost.
  for frame in range(6, 30):
    voted.update(frame, [], "far", 2648.0, now=0.05 * frame)
  assert voted.tracks == []


def test_confirm_track_prefers_red_and_keeps_the_detector_belt():
  t = TrafficLightTracker()
  for frame in range(6):
    t.update(frame, [det(LampClass.GREEN)], "far", 2648.0, now=0.1 * frame)
  track = t.tracks[0]
  # No classifier votes: the detector M-of-N belt behaves like today.
  result = t.confirm_track(track, 5, 0.5, use_classifier=True)
  assert result is not None and result[0] == int(LampClass.GREEN)
  # Three red votes override a green detector history (safety first).
  for i in range(3):
    t.record_votes(1.0 + 0.2 * i, [(track.track_id, int(LampClass.RED), 0.9)])
  assert t.confirm_track(track, 5, 1.4, use_classifier=True) == (int(LampClass.RED), 1.0)
  # The classifier-less path is byte-identical to the detector belt.
  assert t.confirm_track(track, 5, 1.4, use_classifier=False) == (int(LampClass.GREEN), 0.75)


# ---------------------------------------------------------------------------
# W74D (B2): same-track goAfterRed window
# ---------------------------------------------------------------------------

OTHER_BOX = (600.0, 300.0, 640.0, 340.0)


def test_go_after_red_same_track_red_confirmed_then_green():
  t = TrafficLightTracker()
  t.update(0, [det(LampClass.RED)], "far", 2648.0, now=0.0)
  track = t.tracks[0]
  t.note_red_confirmations(0.0, [(track, int(LampClass.RED), 1.0)])
  assert t.go_after_red(track.track_id, 0.5) is True


def test_go_after_red_other_tracks_green_is_false():
  t = TrafficLightTracker()
  t.update(0, [det(LampClass.RED)], "far", 2648.0, now=0.0)
  t.update(1, [det(LampClass.GREEN, box=OTHER_BOX)], "far", 2648.0, now=0.05)
  assert len(t.tracks) == 2
  red_track = next(track for track in t.tracks if track.cx < 500.0)
  green_track = next(track for track in t.tracks if track.cx >= 500.0)
  t.note_red_confirmations(0.05, [(red_track, int(LampClass.RED), 1.0)])
  assert t.go_after_red(red_track.track_id, 1.0) is True
  assert t.go_after_red(green_track.track_id, 1.0) is False


def test_go_after_red_without_prior_red_is_false():
  t = TrafficLightTracker()
  t.update(0, [det(LampClass.GREEN)], "far", 2648.0, now=0.0)
  assert t.go_after_red(t.tracks[0].track_id, 0.0) is False
  assert t.go_after_red(123456, 0.0) is False  # unknown track id


def test_go_after_red_window_is_120s():
  t = TrafficLightTracker()
  t.update(0, [det(LampClass.RED)], "far", 2648.0, now=0.0)
  track = t.tracks[0]
  # redLeft is red-family: it also refreshes the window (Korean left phase).
  t.note_red_confirmations(10.0, [(track, int(LampClass.RED_LEFT), 1.0)])
  assert t.go_after_red(track.track_id, 130.0) is True    # exactly 120 s
  assert t.go_after_red(track.track_id, 130.001) is False  # beyond the window
