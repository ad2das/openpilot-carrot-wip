"""WTOP1: a red that left the frame top at the stop line, and the model green veto.

Route 479, 2026-XX-XX 18:57:07 (KST): the car stopped at a red whose relevant box
had already walked up to the frame top (y1 = 0..1 px). The track was lost at
07.178, the src-1 state expired at the normal W143 hold (carrotMan 1/1 -> 0/0 at
07.866 while v = 2.5 km/h), and at 08.33 the model's own green started the car
into a red it could no longer see.

Two independent defences, tested separately:

  carrotServ - `_tlv_top_exit_hold`: after a src-1 red whose box touched the
    frame top was lost, the src-1 state keeps being held while the car is slow,
    until it drives TLV_TOP_EXIT_MAX_M past the stop line (or 120 s, or the gas
    pedal, or a visible relevant track returns).
  carrot_functions - `_model_green_start_allowed`: the model's trafficState
    green no longer starts the car while a published TLV red is in the state
    (state_in == 1, src == 1). W150 release (0/0) and W143 green (2) both clear
    it, so TLV-originated starts are unaffected.
"""
from types import SimpleNamespace

import numpy as np

from test_tlv_fusion import (FakeSM, carrot_man_message, gate_sm, make_serv as base_make_serv,
                             planner_for_gate, tlv_message)
from test_vehicle_speed_camera_control import _car_state

import openpilot.selfdrive.carrot.carrot_functions as carrot_functions
import openpilot.selfdrive.carrot.carrot_man_input as carrot_man_input
import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.carrot.carrot_functions import (
  TLV_GAS_COOLDOWN_FRAMES, CarrotPlanner, TrafficState,
)
from openpilot.selfdrive.carrot.carrot_serv import CarrotServ

# ---------------------------------------------------------------------------
# The 479 replay: the logged relevant-track rows, then the track vanishing.
# ---------------------------------------------------------------------------

RED_ROWS = [
  (5.983, (967.0, -5.0, 1070.0, 28.0)),
  (6.091, (969.0, 0.0, 1069.0, 23.0)),
  (6.292, (967.0, 0.0, 1073.0, 16.0)),
  (6.495, (966.0, 0.0, 1061.0, 8.0)),
  (6.684, (970.0, 1.0, 1065.0, 10.0)),
  (6.883, (970.0, 0.0, 1066.0, 9.0)),
  (7.022, (970.0, 0.0, 1066.0, 8.0)),
]
LOST_AT = 7.178
REPLAY_END = 12.0
# car vEgo in km/h, read off the 479 log around the stop.
SPEED_KPH = [(6.0, 7.9), (6.5, 5.9), (7.0, 4.2), (7.5, 3.0), (8.0, 2.0), (8.5, 1.3),
             (9.0, 2.4), (12.0, 0.2)]
SPEED_TS = [p[0] for p in SPEED_KPH]
SPEED_VS = [p[1] for p in SPEED_KPH]


def speed_mps(t: float) -> float:
  return float(np.interp(t, SPEED_TS, SPEED_VS)) / 3.6


def car_state(t: float, gas: bool = False) -> SimpleNamespace:
  return SimpleNamespace(vEgo=speed_mps(t), gasPressed=gas)


def make_serv() -> CarrotServ:
  serv = base_make_serv(2)
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
  serv.tlv_last_red_top = None
  serv.tlv_top_exit = None
  return serv


def red_msg(box):
  return tlv_message("red", conf=0.9, rel_box=box, box=box)


def lost_msg():
  return tlv_message("red", conf=0.9, rel_id=-1, rel_box=(0.0, 0.0, 0.0, 0.0))


def drive_red_phase(serv: CarrotServ, clock: list, rows=RED_ROWS) -> None:
  """Publish the logged relevant-track red rows at their own timestamps."""
  t0, box0 = rows[0]
  clock[0] = t0
  serv._update_cmd()
  serv._update_tlv_fusion(FakeSM(carrotTlv=red_msg(box0), carState=car_state(t0)))
  for t, box in rows[1:]:
    while clock[0] < t - 0.049:
      clock[0] = round(min(clock[0] + 0.05, t), 3)
      serv._update_cmd()
    clock[0] = t
    serv._update_cmd()
    serv._update_tlv_fusion(FakeSM(carrotTlv=red_msg(box), carState=car_state(t)))


def drive_loss(serv: CarrotServ, clock: list, start: float = LOST_AT, end: float = REPLAY_END,
               cs_for_t=None, step: float = 0.05):
  """Publish relevantTrackId = -1 at 20 Hz; return [(t, state, src, holding)]."""
  cs_for_t = cs_for_t or (lambda t: car_state(t))
  out = []
  t = start
  while t <= end + 1e-9:
    clock[0] = t
    serv._update_cmd()
    serv._update_tlv_fusion(FakeSM(carrotTlv=lost_msg(), carState=cs_for_t(t)))
    out.append((t, serv.traffic_state, serv.traffic_state_src, serv.tlv_top_exit is not None))
    t = round(t + step, 3)
  return out


def at(states, t: float):
  return min(states, key=lambda s: abs(s[0] - t))


def freeze(monkeypatch):
  clock = [0.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  return clock


# ---------------------------------------------------------------------------
# T1: the 479 replay. Validate-first: with the hold disabled the logged drop must
# reproduce (0/0 by 07.9 s) before the enabled case is allowed to count.
# ---------------------------------------------------------------------------

def test_t1_replay_reproduces_the_logged_drop_with_the_hold_disabled(monkeypatch):
  monkeypatch.setattr(carrot_serv, "TLV_TOP_EXIT_Y1_PX", -1.0)
  clock = freeze(monkeypatch)
  serv = make_serv()
  drive_red_phase(serv, clock)
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  states = drive_loss(serv, clock)
  assert at(states, 7.2)[1:3] == (1, 1)  # still inside the normal W143 hold
  assert at(states, 7.9)[1:3] == (0, 0)  # the 479 drop, logged at 07.866


def test_t1_replay_keeps_the_red_through_the_loss_and_the_model_green(monkeypatch):
  clock = freeze(monkeypatch)
  serv = make_serv()
  drive_red_phase(serv, clock)
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  states = drive_loss(serv, clock)
  assert at(states, 8.33)[1:3] == (1, 1)  # 479: model green started the car here
  assert at(states, 9.2)[1:3] == (1, 1)
  assert at(states, REPLAY_END)[1:3] == (1, 1)


# ---------------------------------------------------------------------------
# T2-T7: the hold's entry gate, its three ways to end, and its hand-off.
# ---------------------------------------------------------------------------

def test_t2_a_mid_frame_loss_does_not_hold(monkeypatch):
  clock = freeze(monkeypatch)
  serv = make_serv()
  rows = RED_ROWS[:-1] + [(7.022, (970.0, 40.0, 1066.0, 68.0))]
  drive_red_phase(serv, clock, rows)
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  states = drive_loss(serv, clock)
  assert at(states, 7.9)[1:3] == (0, 0)
  assert not any(s[3] for s in states)


def test_t3_a_fast_car_does_not_hold(monkeypatch):
  clock = freeze(monkeypatch)
  serv = make_serv()
  drive_red_phase(serv, clock)
  states = drive_loss(serv, clock, cs_for_t=lambda t: SimpleNamespace(vEgo=4.0, gasPressed=False))
  assert at(states, 7.9)[1:3] == (0, 0)
  assert not any(s[3] for s in states)


def test_t4_gas_pedal_ends_the_hold(monkeypatch):
  clock = freeze(monkeypatch)
  serv = make_serv()
  drive_red_phase(serv, clock)
  drive_loss(serv, clock, start=LOST_AT, end=7.5)
  assert serv.tlv_top_exit is not None
  states = drive_loss(serv, clock, start=7.55, end=8.6,
                      cs_for_t=lambda t: SimpleNamespace(vEgo=1.0, gasPressed=True))
  assert at(states, 8.6)[1:3] == (0, 0)
  assert not any(s[3] for s in states)


def test_t5_driving_15_m_past_the_line_ends_the_hold(monkeypatch):
  clock = freeze(monkeypatch)
  serv = make_serv()
  drive_red_phase(serv, clock)
  states = drive_loss(serv, clock, end=18.0,
                      cs_for_t=lambda t: SimpleNamespace(vEgo=2.0, gasPressed=False))
  assert at(states, 9.0)[1:3] == (1, 1)    # 3.6 m in, still held
  assert at(states, 13.0)[1:3] == (1, 1)   # 11.6 m
  assert at(states, 17.5)[1:3] == (0, 0)   # 20.6 m > 15 m


def test_t6_the_hold_caps_at_120_s(monkeypatch):
  clock = freeze(monkeypatch)
  serv = make_serv()
  drive_red_phase(serv, clock)
  # The cap is measured from the hold's first held tick (7.178), so it fires at 127.2.
  states = drive_loss(serv, clock, end=131.0,
                      cs_for_t=lambda t: SimpleNamespace(vEgo=0.0, gasPressed=False))
  assert at(states, 126.0)[1:3] == (1, 1)
  assert at(states, 130.5)[1:3] == (0, 0)


def test_t7_a_new_relevant_track_ends_the_hold_and_green_still_fires(monkeypatch):
  clock = freeze(monkeypatch)
  serv = make_serv()
  drive_red_phase(serv, clock)
  drive_loss(serv, clock, start=LOST_AT, end=7.45)
  assert serv.tlv_top_exit is not None
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  clock[0] = 7.5
  box = RED_ROWS[-1][1]
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.9, go_after_red=True,
                                                       rel_id=2, rel_box=box, box=box)))
  assert serv.tlv_top_exit is None
  assert (serv.traffic_state, serv.traffic_state_src) == (2, 1)


# ---------------------------------------------------------------------------
# T8: the planner veto. Published TLV red stops the model green start; the W150
# release state, the W143 green, gas and the gas cooldown all let it through.
# ---------------------------------------------------------------------------

def freeze_man(monkeypatch):
  # FakeSM stamps every service at 100.0; get_carrot_man drops stale messages.
  monkeypatch.setattr(carrot_man_input.time, "monotonic", lambda: 100.0)


def planner_with(monkeypatch, published: tuple[int, int], speed: int = 60, **gate_kw):
  freeze_man(monkeypatch)
  planner = planner_for_gate(False)
  planner.trafficState = TrafficState.green
  planner._update_carrot_man(
    gate_sm(carrot_man_message(state=published[0], src=published[1], speed=speed), **gate_kw), 20.0, 60.0)
  return planner


def test_t8_published_tlv_red_vetoes_the_model_green_start(monkeypatch):
  assert planner_with(monkeypatch, (1, 1))._model_green_start_allowed() is False


def test_t8_w150_release_and_w143_green_are_not_vetoed(monkeypatch):
  assert planner_with(monkeypatch, (0, 0))._model_green_start_allowed() is True
  assert planner_with(monkeypatch, (2, 1))._model_green_start_allowed() is True


def test_t8_navi_state_and_gas_are_not_vetoed(monkeypatch):
  assert planner_with(monkeypatch, (1, 0))._model_green_start_allowed() is True
  assert planner_with(monkeypatch, (1, 1), gas=True)._model_green_start_allowed() is True


def test_t8_gas_cooldown_is_not_vetoed(monkeypatch):
  freeze_man(monkeypatch)
  planner = planner_for_gate(False)
  planner.trafficState = TrafficState.green
  planner.tlv_gas_cooldown_frames = 200
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1)), 20.0, 60.0)
  assert planner._model_green_start_allowed() is True


def test_t8_a_missing_carrot_man_leaves_the_model_start_alone(monkeypatch):
  # desiredSpeed 0 makes get_carrot_man drop the message: the else branch runs.
  planner = planner_with(monkeypatch, (1, 1), speed=0)
  assert planner.tlv_red_veto is False
  assert planner._model_green_start_allowed() is True


# ---------------------------------------------------------------------------
# T9: the published field the C3X prompt reads (carrotMan.tlvTopExit @37). The UI
# can only show what carrot_serv publishes, so the replay is driven through the
# real update_navi() message-building path, not _update_tlv_fusion() alone.
# ---------------------------------------------------------------------------

class TopExitParams:
  """Just enough typed Params for a real CarrotServ(); only the detector mode matters here."""

  def __init__(self, *_args):
    pass

  def get(self, key):
    return str({"TrafficLightDetectMode": 2}.get(key, 0))

  def get_int(self, key):
    return int(self.get(key))

  def get_float(self, key):
    return float(self.get(key))

  def get_bool(self, key):
    return bool(self.get_int(key))


class PublishSM(FakeSM):
  """FakeSM plus the one service update_navi() indexes but a publish test does not drive."""

  def __init__(self, **services):
    super().__init__(**services)
    for name in ("navInstruction",):
      self.seen[name] = True
      self.updated[name] = False
      self.alive[name] = False
      self.valid[name] = False
      self.recv_time[name] = 100.0


def publish_serv(monkeypatch):
  """A real CarrotServ() with the navigation/display inputs stubbed out as test_carrot_navi_serv does."""
  monkeypatch.setattr(carrot_serv, "Params", TopExitParams)
  serv = CarrotServ()
  monkeypatch.setattr(serv, "_update_carrot_navi", lambda _sm: False)
  monkeypatch.setattr(serv, "_update_gps", lambda *_args: 0.)
  monkeypatch.setattr(serv, "update_nav_instruction", lambda _sm: None)
  monkeypatch.setattr(serv, "update_auto_turn", lambda *_args: (250., "none", 250., 0.))
  monkeypatch.setattr(serv, "_vehicle_navigation_display", lambda _cs: (False, 0, False))
  monkeypatch.setattr(serv, "_rear_camera_speed", lambda *_args: (False, 0.0))
  monkeypatch.setattr(serv, "_vehicle_speed_camera_enabled", lambda _cs: False)
  monkeypatch.setattr(serv, "_vehicle_speed_bump_enabled", lambda _cs: False)
  monkeypatch.setattr(serv, "_vehicle_section_zone_enabled", lambda _cs: False)
  monkeypatch.setattr(serv, "_vehicle_school_zone_speed", lambda _cs: 250)
  return serv, {}


def publish(serv: CarrotServ, clock: list, sent: dict, t: float, msg) -> bool:
  """One update_navi() tick (which runs _update_cmd and the fusion itself); returns the published tlvTopExit."""
  clock[0] = t
  sm = PublishSM(carrotTlv=msg, carState=_car_state(v_ego=speed_mps(t)),
                 selfdriveState=SimpleNamespace(distanceTraveled=0.))
  pm = SimpleNamespace(send=lambda name, message: sent.update({name: message}))
  serv.update_navi("", sm, pm, 0., [], [], 0., "gpsLocationExternal")
  return sent["carrotMan"].carrotMan.tlvTopExit


def test_publish_top_exit_true_only_while_red_held(monkeypatch):
  clock = freeze(monkeypatch)
  serv, sent = publish_serv(monkeypatch)
  drive_red_phase(serv, clock)

  # A tracked red is not a top-exit hold: nothing for the prompt to show.
  assert publish(serv, clock, sent, RED_ROWS[-1][0], red_msg(RED_ROWS[-1][1])) is False

  published = []
  t = LOST_AT
  while t <= REPLAY_END + 1e-9:
    published.append((t, publish(serv, clock, sent, t, lost_msg())))
    t = round(t + 0.05, 3)

  assert at(published, 8.33)[1] is True   # 479: the model green started the car here
  assert at(published, 9.2)[1] is True
  assert at(published, REPLAY_END)[1] is True


def test_publish_top_exit_false_when_red_expired_but_state_stale(monkeypatch):
  clock = freeze(monkeypatch)
  serv, sent = publish_serv(monkeypatch)
  # Plant the state the publish reads and skip the two tick steps that would rewrite it
  # (_update_cmd expires a lapsed red; the fusion ends a stale hold): only the publish
  # expression is under test here.
  monkeypatch.setattr(serv, "_update_tlv_fusion", lambda _sm: None)
  monkeypatch.setattr(serv, "_update_cmd", lambda: None)
  serv.tlv_top_exit = {"since": 0, "dist": 0.0, "t": 0.0}
  serv.traffic_state = 0
  serv.traffic_state_src = 0
  assert publish(serv, clock, sent, 7.5, lost_msg()) is False

  # The prompt also needs the published red to still be a TLV (src 1) red, not a navi red.
  serv.traffic_state = 1
  assert publish(serv, clock, sent, 7.55, lost_msg()) is False

  serv.traffic_state_src = 1
  assert publish(serv, clock, sent, 7.6, lost_msg()) is True




# ---------------------------------------------------------------------------
# V1-V8 (W-VETO1): the veto has to survive the 1-3 frame (0,0) / redLikely (4)
# gaps inside one real TLV red (6e1 169.77), and still end at once on a TLV
# go, on gas, on the gas cooldown and on a lead-follow latch.
# ---------------------------------------------------------------------------

def veto_planner(monkeypatch, start: float = 100.0) -> tuple[CarrotPlanner, list]:
  """planner_for_gate on a mutable clock; carrotMan freshness uses the same clock."""
  clock = [start]
  monkeypatch.setattr(carrot_functions.time, "monotonic", lambda: clock[0])
  planner = planner_for_gate(False)
  planner.trafficState = TrafficState.green
  planner.tlv_red_veto = False
  planner.tlv_red_seen_t = -1e9
  planner.lead_follow_active = False
  return planner, clock


def man_tick(planner, clock: list, t: float, state: int, src: int = 1,
             v_ego_kph: float = 20.0, speed: int = 60, **gate_kw) -> bool:
  clock[0] = t
  sm = gate_sm(carrot_man_message(state=state, src=src, speed=speed), **gate_kw)
  # FakeSM stamps every service at construction time; keep carrotMan fresh on the
  # mutable clock so get_carrot_man's 1 s timeout does not drop the message.
  sm.recv_time['carrotMan'] = t
  planner._update_carrot_man(sm, v_ego_kph, 60.0)
  return planner._model_green_start_allowed()


def test_v1_the_veto_survives_the_6e1_flicker(monkeypatch):
  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1) is False
  # 6e1 169.77: the fusion published (0,0) for 250 ms with the light still red.
  assert man_tick(planner, clock, 100.25, 0) is False


def test_v2_the_veto_survives_a_redLikely_gap(monkeypatch):
  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1) is False
  assert man_tick(planner, clock, 100.25, 4) is False


def test_v3_the_grace_expires(monkeypatch):
  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1) is False
  # Moving (20 km/h), so the B1 standstill hold cannot carry the red either.
  assert man_tick(planner, clock, 100.5, 0) is False
  assert man_tick(planner, clock, 101.5, 0) is False
  assert man_tick(planner, clock, 102.9, 0) is False
  assert man_tick(planner, clock, 103.1, 0) is True


def test_v4_the_b1_standstill_hold_keeps_the_veto_past_the_grace(monkeypatch):
  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1, v_ego_kph=0.0) is False
  t = 100.05
  while t <= 110.0 + 1e-9:
    assert man_tick(planner, clock, round(t, 3), 0, v_ego_kph=0.0) is False
    t = round(t + DT_MDL, 3)


def test_v5_a_tlv_go_ends_the_veto_at_once(monkeypatch):
  for go_state in (2, 3):
    planner, clock = veto_planner(monkeypatch)
    assert man_tick(planner, clock, 100.0, 1) is False
    assert man_tick(planner, clock, 100.25, go_state) is True
    # The (0,0) after a real departure must not re-arm the veto.
    assert man_tick(planner, clock, 100.5, 0) is True


def test_v6_gas_cooldown_and_lead_follow_all_let_the_start_through(monkeypatch):
  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1) is False
  assert man_tick(planner, clock, 100.25, 0, gas=True) is True

  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1) is False
  planner.tlv_gas_cooldown_frames = TLV_GAS_COOLDOWN_FRAMES
  assert man_tick(planner, clock, 100.25, 0) is True

  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1) is False
  planner.lead_follow_active = True
  assert man_tick(planner, clock, 100.25, 0) is True


def test_v7_a_navi_red_never_vetoes(monkeypatch):
  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1, src=0) is True
  assert planner.tlv_red_seen_t == -1e9


def test_v8_a_missing_carrot_man_clears_the_stamp(monkeypatch):
  planner, clock = veto_planner(monkeypatch)
  assert man_tick(planner, clock, 100.0, 1) is False
  assert planner.tlv_red_seen_t == 100.0
  # desiredSpeed 0 makes get_carrot_man drop the message: the else branch runs.
  assert man_tick(planner, clock, 100.25, 1, speed=0) is True
  assert planner.tlv_red_seen_t == -1e9
  assert planner.tlv_red_veto is False
