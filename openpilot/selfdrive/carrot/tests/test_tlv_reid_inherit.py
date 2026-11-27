"""GAR1: a re-identified light inherits ONLY red_confirmed_s from a proven same light.

11-27 13:23 Palisade route 00000505--859e0b16fd seg 4: the red head was
re-identified 27 -> 28 -> 31 -> 33 while red, so the same-track goAfterRed rule
saw a track that was never red-confirmed and the car braked to the line on a
green. These cases pin the inheritance gate and its revocations.
"""
import pytest

from openpilot.selfdrive.carrot.tlv.tracker import (
  INHERIT_GRAVE_MAX,
  Detection,
  EgoOdometry,
  LampClass,
  TrafficLightTracker,
  inherit_cost,
  match_score,
  predict_box,
)

F = 2648.0
FRAME = (0.0, 0.0, 1928.0, 1208.0)
# 27 x 9 px head at ~103 m in the 13:23 rlog.
HEAD = (911.5, 319.5, 938.5, 328.5)
# A near light whose whole box is closer than the 5 m predict_box snap-back limit.
NEAR = (600.0, 400.0, 700.0, 600.0)


def det(cls, box):
  return Detection(cls=int(cls), score=0.9, x1=box[0], y1=box[1], x2=box[2], y2=box[3])


def _shift(box, dx=0.0, dy=0.0):
  return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _cxcywh(cx, cy, w, h):
  return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def _setup():
  """A real EgoOdometry held at v = 0, so predict() is the identity by default."""
  odo = EgoOdometry()
  odo.update(0.0, 0.0, 0.0)
  def pred(tr):
    return predict_box(tr, odo, focal_px=F, foe_x=964.0, foe_y=425.0, image_w=1928, image_h=1208)
  return odo, pred


def _step(t, frame, dets, now, odo, pred, frame_box=FRAME):
  return t.update(frame, dets, "far", F, now=now, predict=pred, anchor=(odo.s, odo.psi, now),
                  crop_box=FRAME, frame_box=frame_box)


def _created(t, before):
  return next(tr for tr in t.tracks if tr.track_id not in before)


def _reid_setup():
  """One red-confirmed head at t 0.0."""
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  parent = t.tracks[0]
  t.note_red_confirmations(0.0, [(parent, int(LampClass.RED), 1.0)])
  return t, odo, pred, parent


def _reid_child(t, odo, pred, frame=1, now=0.2, dy=22.5):
  before = {tr.track_id for tr in t.tracks}
  _step(t, frame, [det(LampClass.RED, _shift(HEAD, dy=dy))], now, odo, pred)
  return _created(t, before)


def test_unmatched_alive_parent_reid():
  t, odo, pred, parent = _reid_setup()
  # The 2.5 h det is NOT the tracked box by the association rules.
  assert match_score(HEAD, _shift(HEAD, dy=22.5)) == 0.0
  child = _reid_child(t, odo, pred)
  assert child.red_confirmed_s == 0.0       # the parent's time, not this frame's
  assert child.red_hops == 1
  assert child.inherited_from == (parent.track_id,)
  assert len(child.votes) == 0              # credit only, never votes
  assert child.hits == 1
  # GAR1 R7: the credit counts only after the child's own red vote.
  t.record_votes(0.2, [(child.track_id, int(LampClass.RED), 0.9)])
  for _ in range(4):
    t.record_votes(1.0, [(child.track_id, int(LampClass.GREEN), 0.9)])
  assert t.go_after_red(child.track_id, 1.5) is True


def test_graveyard_boundary():
  def run(gap):
    odo, pred = _setup()
    t = TrafficLightTracker()
    _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
    t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
    for i, frame in enumerate((1, 2, 3)):
      _step(t, frame, [], 0.1 * (i + 1), odo, pred)
    assert t.tracks == []
    assert len(t.graveyard) == 1
    t_drop = t.graveyard[0][0]
    assert t_drop == pytest.approx(0.3)
    _step(t, 4, [det(LampClass.RED, HEAD)], t_drop + gap, odo, pred)
    assert len(t.tracks) == 1
    return t.tracks[0]

  assert run(0.99).red_confirmed_s == 0.0
  assert run(1.01).red_confirmed_s is None


def test_frame_exit_and_snapback_never_donate():
  # (a) A frame exit is not a miss expiry: nothing is graveyarded, nothing donates.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  _step(t, 1, [], 0.1, odo, pred, frame_box=(0.0, 400.0, 1928.0, 1208.0))
  assert t.tracks == []
  assert t.graveyard == []
  _step(t, 2, [det(LampClass.RED, HEAD)], 0.4, odo, pred)
  assert t.tracks[0].red_confirmed_s is None

  # (b) With ego travel, a parent inside the 5 m snap-back zone cannot predict its box.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  d0 = F * 0.35 / 9.0
  odo.s = d0 - 4.0
  _step(t, 1, [], 0.1, odo, pred)
  assert len(t.tracks) == 1
  _step(t, 2, [det(LampClass.RED, _shift(HEAD, dy=22.5))], 0.2, odo, pred)
  child = t.tracks[-1]
  assert child.red_confirmed_s is None
  assert child.red_hops == 0

  # (c) ds == 0 with the whole box inside 5 m is not a snap-back: it still donates.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, NEAR)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  _step(t, 1, [det(LampClass.RED, _shift(NEAR, dy=400.0))], 0.2, odo, pred)
  assert t.tracks[-1].red_confirmed_s == 0.0


def test_matched_parent_never_donates():
  t, odo, pred, parent = _reid_setup()
  side = _shift(HEAD, dx=27.0)
  assert match_score(HEAD, side) == 0.0
  _step(t, 1, [det(LampClass.RED, HEAD), det(LampClass.RED, side)], 0.2, odo, pred)
  assert parent.hits == 2
  child = t.tracks[-1]
  assert child.track_id != parent.track_id
  assert child.red_confirmed_s is None


def test_revocation_on_parent_rematch():
  t, odo, pred, parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  assert child.inherited_from == (parent.track_id,)
  _step(t, 2, [det(LampClass.RED, HEAD)], 0.4, odo, pred)
  assert child.red_confirmed_s is None
  assert child.red_hops == 0
  assert child.inherited_from == ()
  assert t.go_after_red(child.track_id, 0.5) is False

  # Transitive: a re-match of the ancestor revokes every descendant.
  t, odo, pred, parent = _reid_setup()
  c1 = _reid_child(t, odo, pred)
  t.record_votes(0.2, [(c1.track_id, int(LampClass.RED), 0.9)])  # R7: confirm before it donates
  before = {tr.track_id for tr in t.tracks}
  _step(t, 2, [det(LampClass.RED, _shift(HEAD, dy=45.0))], 0.4, odo, pred)
  c2 = _created(t, before)
  assert c2.red_hops == 2
  assert c2.inherited_from == (parent.track_id, c1.track_id)
  _step(t, 3, [det(LampClass.RED, HEAD)], 0.6, odo, pred)
  assert c1.red_confirmed_s is None and c2.red_confirmed_s is None
  assert c1.inherited_from == () and c2.inherited_from == ()

  # A self-confirmed red is the child's own and survives the parent re-match.
  t, odo, pred, parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  t.note_red_confirmations(0.35, [(child, int(LampClass.RED), 1.0)])
  _step(t, 2, [det(LampClass.RED, HEAD)], 0.4, odo, pred)
  assert child.red_confirmed_s == 0.35
  assert child.red_hops == 0
  assert child.inherited_from == ()


def test_1323_pairs_inherit_cost():
  p27 = _cxcywh(925.3, 355.2, 22.1, 7.3)
  assert inherit_cost(p27, _cxcywh(927.9, 341.9, 21.8, 7.4)) is not None                    # 27 -> 28
  assert inherit_cost(_cxcywh(925.6, 324.1, 27.1, 9.0),
                      _cxcywh(923.2, 300.9, 26.9, 9.3)) is not None                          # 31 -> 33
  assert inherit_cost(p27, _cxcywh(1002.7, 349.3, 19.3, 8.2)) is None                        # 27 -> 29: next head
  assert inherit_cost(_cxcywh(1045.3, 307.5, 29.2, 10.0),
                      _cxcywh(1033.2, 253.4, 43.4, 13.8)) is None                            # 30 -> 34: dy 54.1 > 30
  assert inherit_cost(_cxcywh(1068.1, 302.6, 30.4, 10.5),
                      _cxcywh(1032.7, 320.2, 18.3, 7.9)) is None                             # 30 -> 36: w 0.60
  assert inherit_cost(_cxcywh(1066.4, 311.6, 28.1, 8.5),
                      _cxcywh(1032.7, 320.2, 18.3, 7.9)) is None                             # 32 -> 36: w 0.65
  # The next intersection's light: too narrow (0.67) and far below the head.
  assert inherit_cost((900.0, 300.0, 930.0, 310.0), _cxcywh(900.0, 331.5, 20.1, 6.7)) is None


def test_adjacent_head_and_same_head_arrow():
  # 1.6 w to the side is a different head, not a re-ID.
  t, odo, pred, parent = _reid_setup()
  _step(t, 1, [det(LampClass.RED, _shift(HEAD, dx=1.6 * 27.0))], 0.2, odo, pred)
  assert len(t.tracks) == 2
  assert t.tracks[-1].red_confirmed_s is None

  # The same-head re-ID keeps the credit for a green arrow as well.
  for cls in (LampClass.GREEN_LEFT, LampClass.LEFT):
    t, odo, pred, parent = _reid_setup()
    child = _reid_child(t, odo, pred)
    t.record_votes(0.2, [(child.track_id, int(LampClass.RED), 0.9)])  # R7: own red first
    for _ in range(4):
      t.record_votes(1.0, [(child.track_id, int(cls), 0.9)])
    assert t.go_after_red(child.track_id, 1.0) is True


def test_one_to_one_and_no_same_frame_chain():
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  for i, frame in enumerate((1, 2, 3)):
    _step(t, frame, [], 0.1 * (i + 1), odo, pred)
  assert t.tracks == []
  t_drop = t.graveyard[0][0]
  # Two dets inside the gate on one frame: only the lower-cost one inherits.
  _step(t, 4, [det(LampClass.RED, HEAD), det(LampClass.RED, _shift(HEAD, dy=15.0))],
        t_drop + 0.5, odo, pred)
  first, second = t.tracks
  assert first.red_confirmed_s == 0.0
  assert second.red_confirmed_s is None

  # No red track in the pool: nothing to inherit on one frame.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD), det(LampClass.RED, _shift(HEAD, dx=60.0))], 0.0, odo, pred)
  assert len(t.tracks) == 2
  assert all(tr.red_confirmed_s is None for tr in t.tracks)


def test_hops_and_stale_and_green():
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  chain = []
  for i in (1, 2, 3):
    link = _reid_child(t, odo, pred, frame=i, now=0.1 * i, dy=22.5 * i)
    # GAR1 R7: a pending credit cannot donate, so each hop confirms its own red.
    t.record_votes(0.1 * i, [(link.track_id, int(LampClass.RED), 0.9)])
    chain.append(link)
  assert [tr.red_hops for tr in chain] == [1, 2, 3]
  assert all(tr.red_confirmed_s == 0.0 for tr in chain)
  fourth = _reid_child(t, odo, pred, frame=4, now=0.4, dy=90.0)
  assert fourth.red_confirmed_s is None

  # A red older than INHERIT_RED_MAX_AGE_S cannot donate.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  _step(t, 1, [det(LampClass.RED, _shift(HEAD, dy=22.5))], 5.01, odo, pred)
  assert t.tracks[-1].red_confirmed_s is None

  # A parent confirmed green after its red is not a light still showing red.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  parent = t.tracks[0]
  t.note_red_confirmations(0.0, [(parent, int(LampClass.RED), 1.0)])
  t.note_red_confirmations(0.1, [(parent, int(LampClass.GREEN), 1.0)])
  assert parent.green_confirmed_s == 0.1
  _step(t, 1, [det(LampClass.GREEN, _shift(HEAD, dy=22.5))], 0.2, odo, pred)
  assert t.tracks[-1].red_confirmed_s is None

  # Three green votes without a green confirmation do not block the donation.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  parent = t.tracks[0]
  t.note_red_confirmations(0.0, [(parent, int(LampClass.RED), 1.0)])
  t.record_votes(0.1, [(parent.track_id, int(LampClass.GREEN), 0.9)] * 3)
  _step(t, 1, [det(LampClass.GREEN, _shift(HEAD, dy=22.5))], 0.2, odo, pred)
  assert t.tracks[-1].red_confirmed_s == 0.0

  # A self-confirmed red resets the chain back to hop 0.
  t, odo, pred, parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  assert child.red_hops == 1 and child.inherited_from == (parent.track_id,)
  t.note_red_confirmations(0.9, [(child, int(LampClass.RED), 1.0)])
  assert child.red_confirmed_s == 0.9
  assert child.red_hops == 0
  assert child.inherited_from == ()


def test_donated_parent_never_donates_twice():
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  _step(t, 1, [], 0.1, odo, pred)
  _step(t, 2, [], 0.2, odo, pred)
  # Frame 3: P donates to C1 and drops on the same frame.
  _step(t, 3, [det(LampClass.RED, _shift(HEAD, dy=22.5))], 0.3, odo, pred)
  assert len(t.tracks) == 1
  c1 = t.tracks[0]
  assert c1.red_confirmed_s == 0.0
  assert c1.red_donated is False
  assert t.graveyard == []                  # a spent parent is not kept for another donation
  # Neon GAR3 5e negative form: C1 stays matched on frame 4, so the HEAD det is a new
  # track whose only possible donor (C1) carries an unconfirmed credit; P donated
  # already. Nothing to inherit: no second donation from P, no donation from C1.
  _step(t, 4, [det(LampClass.RED, _shift(HEAD, dy=22.5)), det(LampClass.RED, HEAD)], 0.4, odo, pred)
  child = next(tr for tr in t.tracks if tr.track_id != c1.track_id)
  assert child.red_confirmed_s is None
  assert child.inherited_from == ()
  assert child.red_hops == 0


def test_alive_donor_never_donates_twice():
  # A donor that is still alive (fewer misses before the re-ID frame) never donates again.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  parent = t.tracks[0]
  t.note_red_confirmations(0.0, [(parent, int(LampClass.RED), 1.0)])
  _step(t, 1, [], 0.1, odo, pred)
  # Frame 2: P is alive and unmatched; C1 takes the credit and P stops donating.
  _step(t, 2, [det(LampClass.RED, _shift(HEAD, dy=22.5))], 0.2, odo, pred)
  c1 = t.tracks[-1]
  assert c1.red_confirmed_s == 0.0
  assert parent.red_donated is True
  assert parent.view_misses == 2            # alive (limit 3), never a frame-exit drop
  t.record_votes(0.2, [(c1.track_id, int(LampClass.RED), 0.9)])
  # The det 22.5 px below P's box is not a detector match for P, and passes inherit_cost
  # against P's predicted box: only the spent-donor rule can block it.
  assert match_score(HEAD, _shift(HEAD, dy=-22.5)) == 0.0
  assert inherit_cost(HEAD, _shift(HEAD, dy=-22.5)) is not None
  _step(t, 3, [det(LampClass.RED, _shift(HEAD, dy=22.5)), det(LampClass.RED, _shift(HEAD, dy=-22.5))],
        0.3, odo, pred)
  child = t.tracks[-1]
  assert child.track_id != c1.track_id
  assert child.red_confirmed_s is None
  assert child.inherited_from == ()


def test_lifecycle():
  t = TrafficLightTracker()
  assert t.graveyard == []
  odo, pred = _setup()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  t.note_red_confirmations(0.0, [(t.tracks[0], int(LampClass.RED), 1.0)])
  # now=None and a call without predict/anchor never inherit and never raise.
  t.update(1, [det(LampClass.RED, _shift(HEAD, dy=22.5))], "far", F, now=None)
  assert t.tracks[-1].red_confirmed_s is None
  t.update(2, [det(LampClass.RED, _shift(HEAD, dy=45.0))], "far", F, now=0.1)
  assert t.tracks[-1].red_confirmed_s is None

  # 40 red tracks dropping at once never grow the graveyard past the cap.
  odo, pred = _setup()
  t = TrafficLightTracker()
  boxes = [(100.0 + 37.0 * i, 300.0, 127.0 + 37.0 * i, 309.0) for i in range(40)]
  _step(t, 0, [det(LampClass.RED, b) for b in boxes], 0.0, odo, pred)
  assert len(t.tracks) == 40
  t.note_red_confirmations(0.0, [(tr, int(LampClass.RED), 1.0) for tr in t.tracks])
  for i, frame in enumerate((1, 2, 3)):
    _step(t, frame, [], 0.1 * (i + 1), odo, pred)
  assert t.tracks == []
  assert len(t.graveyard) == INHERIT_GRAVE_MAX


def test_r7_red_first_vote_keeps_credit():
  t, odo, pred, _parent = _reid_setup()
  child = _reid_child(t, odo, pred)                      # born at 0.2
  assert child.inherit_pending_since == 0.2
  t.record_votes(0.2, [(child.track_id, int(LampClass.RED), 0.89)])
  assert child.inherit_pending_since is None
  assert child.red_confirmed_s == 0.0
  assert t.go_after_red(child.track_id, 0.4) is True


@pytest.mark.parametrize("cls", [LampClass.GREEN, LampClass.GREEN_LEFT, LampClass.LEFT, LampClass.YELLOW])
def test_r7_go_or_yellow_first_vote_drops_credit(cls):
  t, odo, pred, _parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  t.record_votes(0.2, [(child.track_id, int(cls), 0.9)])
  assert child.red_confirmed_s is None
  assert child.red_hops == 0
  assert child.inherited_from == ()
  assert t.go_after_red(child.track_id, 0.4) is False
  # Four green votes over 0.8 s and a green confirmation do not revive the credit.
  for i in range(4):
    t.record_votes(0.3 + 0.2 * i, [(child.track_id, int(LampClass.GREEN), 0.9)])
  t.note_red_confirmations(1.1, [(child, int(LampClass.GREEN), 0.9)])
  assert t.go_after_red(child.track_id, 1.2) is False


def test_r7_undecided_votes_are_bounded():
  # Two OFF votes keep the credit pending; a red 0.8 then confirms it.
  t, odo, pred, _parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  t.record_votes(0.3, [(child.track_id, int(LampClass.OFF), 0.9)])
  t.record_votes(0.4, [(child.track_id, int(LampClass.OFF), 0.9)])
  assert child.inherit_pending_since == 0.2
  assert t.go_after_red(child.track_id, 0.45) is False
  t.record_votes(0.5, [(child.track_id, int(LampClass.RED), 0.8)])
  assert child.inherit_pending_since is None
  assert t.go_after_red(child.track_id, 0.6) is True

  # Three OFF votes drop it.
  t, odo, pred, _parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  for i in range(3):
    t.record_votes(0.25 + 0.05 * i, [(child.track_id, int(LampClass.OFF), 0.9)])
  assert child.red_confirmed_s is None
  assert child.inherit_pending_since is None
  assert t.go_after_red(child.track_id, 0.45) is False

  # A red below 0.6 is undecided: RED 0.5 then RED 0.7 keeps the credit.
  t, odo, pred, _parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  t.record_votes(0.3, [(child.track_id, int(LampClass.RED), 0.5)])
  assert child.inherit_pending_since == 0.2
  t.record_votes(0.4, [(child.track_id, int(LampClass.RED), 0.7)])
  assert child.inherit_pending_since is None
  assert t.go_after_red(child.track_id, 0.5) is True


def test_r7_pending_times_out():
  t, odo, pred, _parent = _reid_setup()
  child = _reid_child(t, odo, pred)                      # born at 0.2
  t.record_votes(0.8, [(child.track_id, int(LampClass.RED), 0.9)])
  assert child.red_confirmed_s is None
  assert child.inherit_pending_since is None
  assert t.go_after_red(child.track_id, 0.9) is False


def test_r7_pending_child_never_donates():
  # A still-pending C1 is not a donor: the det 45 px below takes nothing.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  parent = t.tracks[0]
  t.note_red_confirmations(0.0, [(parent, int(LampClass.RED), 1.0)])
  _step(t, 1, [], 0.1, odo, pred)
  _step(t, 2, [det(LampClass.RED, _shift(HEAD, dy=22.5))], 0.2, odo, pred)
  c1 = t.tracks[-1]
  assert c1.red_confirmed_s == 0.0
  assert c1.inherit_pending_since == 0.2
  _step(t, 3, [det(LampClass.RED, _shift(HEAD, dy=45.0))], 0.3, odo, pred)
  child = t.tracks[-1]
  assert child.red_confirmed_s is None
  assert child.inherited_from == ()

  # A pending child that drops by misses is not kept for a second donation.
  odo, pred = _setup()
  t = TrafficLightTracker()
  _step(t, 0, [det(LampClass.RED, HEAD)], 0.0, odo, pred)
  parent = t.tracks[0]
  t.note_red_confirmations(0.0, [(parent, int(LampClass.RED), 1.0)])
  _step(t, 1, [], 0.1, odo, pred)
  _step(t, 2, [det(LampClass.RED, _shift(HEAD, dy=22.5))], 0.2, odo, pred)
  c1 = t.tracks[-1]
  assert c1.inherit_pending_since == 0.2
  for i, frame in enumerate((3, 4, 5)):
    _step(t, frame, [], 0.3 + 0.1 * i, odo, pred)
  assert all(tr is not c1 for tr in t.tracks)
  assert t.graveyard == []


def test_r7_own_red_confirmation_clears_pending():
  t, odo, pred, _parent = _reid_setup()
  child = _reid_child(t, odo, pred)
  assert child.inherit_pending_since == 0.2
  t.note_red_confirmations(0.5, [(child, int(LampClass.RED), 0.9)])
  assert child.inherit_pending_since is None
  assert child.red_hops == 0
  assert child.inherited_from == ()
  assert child.red_confirmed_s == 0.5


def test_left_confirmation_does_not_block_donation():
  # A bare LEFT is red for straight traffic: it never stamps the parent green.
  t, odo, pred, parent = _reid_setup()
  t.note_red_confirmations(0.1, [(parent, int(LampClass.LEFT), 0.9)])
  assert parent.green_confirmed_s is None
  child = _reid_child(t, odo, pred)
  assert child.red_confirmed_s == 0.0
  t.record_votes(0.2, [(child.track_id, int(LampClass.RED), 0.9)])
  assert t.go_after_red(child.track_id, 0.4) is True

  # GREEN still stamps the parent and the child gets no credit.
  t, odo, pred, parent = _reid_setup()
  t.note_red_confirmations(0.1, [(parent, int(LampClass.GREEN), 0.9)])
  assert parent.green_confirmed_s == 0.1
  _reid_child(t, odo, pred)
  assert t.tracks[-1].red_confirmed_s is None
