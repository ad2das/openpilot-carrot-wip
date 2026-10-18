"""W148: the TLV must not feed carrotMan on expressways.

carrotNavi's laneCurrent.roadCategory is 0 (Expressway) / 1 (CityExpressway),
2 (national and below) and -1 (other), or absent. On 0/1 the whole TLV effect
turns off together; leaving an expressway takes a 5 s exit hysteresis and a
category older than 2 s (the navi repeats at 2 Hz) counts as non-expressway.

Reuses the fakes/shim installer of test_tlv_fusion.py.
"""
import pytest

from test_tlv_fusion import FakeSM, make_serv as base_make_serv, tlv_message

import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
from openpilot.selfdrive.carrot.carrot_serv import (
  CarrotServ,
  TLV_HIGHWAY_CATEGORIES,
  TLV_HIGHWAY_EXIT_S,
  TLV_HIGHWAY_FRESH_S,
)


def make_serv(detect_mode: int = 2) -> CarrotServ:
  serv = base_make_serv(detect_mode)
  serv.tlv_navi_road_category = None
  serv.tlv_navi_road_category_t = -1e9
  serv.tlv_highway_gate = False
  serv.tlv_highway_clear_since = None
  return serv


def record_category(serv: CarrotServ, category, now: float) -> None:
  serv.tlv_navi_road_category = category
  serv.tlv_navi_road_category_t = now


def capture_events(monkeypatch):
  events = []
  monkeypatch.setattr(carrot_serv, "cloudlog",
                      type("_Cloudlog", (), {"event": staticmethod(
                        lambda name, **kw: events.append((name, kw)))}))
  return events


def idle_sm() -> FakeSM:
  """A carrotTlv service that is present but not updated this cycle."""
  sm = FakeSM(carrotTlv=tlv_message("none"))
  sm.updated["carrotTlv"] = False
  return sm


@pytest.mark.parametrize("category", (0, 1))
def test_expressway_leaves_traffic_state_untouched(monkeypatch, category):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  events = capture_events(monkeypatch)
  serv = make_serv(2)
  record_category(serv, category, clock[0])
  for msg in (tlv_message("red", conf=0.9), tlv_message("red", conf=0.9),
              tlv_message("redLikely", conf=0.9), tlv_message("green", conf=0.9)):
    serv._update_tlv_fusion(FakeSM(carrotTlv=msg))
    assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
    assert serv.tlv_det_support == 0
    clock[0] += 0.05
  assert serv.traffic_light_count == -1
  assert list(serv.traffic_light_q) == []
  # Gate on transition logged once, with the category.
  on = [e for e in events if e[0] == "tlv_highway_gate" and e[1]["on"]]
  assert len(on) == 1
  assert on[0][1]["road_category"] == category


@pytest.mark.parametrize("category", (2, -1, None))
def test_non_expressway_matches_todays_fusion(monkeypatch, category):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  capture_events(monkeypatch)
  serv = make_serv(2)
  record_category(serv, category, clock[0])
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.75, go_after_red=True)))
  assert (serv.traffic_state, serv.traffic_state_src) == (2, 1)

  graded = make_serv(2)
  record_category(graded, category, clock[0])
  graded._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("redLikely", conf=0.8)))
  assert (graded.traffic_state, graded.traffic_state_src) == (4, 1)
  assert graded.tlv_highway_gate is False


def test_src1_red_held_then_expressway_clears_the_hold(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  capture_events(monkeypatch)
  serv = make_serv(2)
  record_category(serv, 2, clock[0])
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  assert serv.tlv_hold_until > clock[0]
  assert serv.traffic_light_count == 5

  clock[0] += 0.2
  record_category(serv, 0, clock[0])
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.9)))
  assert (serv.traffic_state, serv.traffic_state_src) == (0, 0)
  assert serv.tlv_hold_until == 0.0
  assert serv.traffic_light_count == -1
  assert serv.tlv_det_support == 0


def test_navi_sourced_state_is_not_cleared_by_the_gate(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  capture_events(monkeypatch)
  serv = make_serv(2)
  serv._handle_detect_command("Red Light,0.5,0.5,0.9")
  serv._handle_detect_command("Red Light,0.5,0.5,0.9")
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 0)
  record_category(serv, 0, clock[0])
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.9)))
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 0)


def test_exit_hysteresis_and_reentry_resets_the_timer(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  capture_events(monkeypatch)
  serv = make_serv(2)

  record_category(serv, 0, clock[0])
  serv._update_tlv_fusion(idle_sm())
  assert serv.tlv_highway_gate is True

  record_category(serv, 2, clock[0])
  serv._update_tlv_fusion(idle_sm())   # the exit timer starts here
  assert serv.tlv_highway_clear_since == 100.0
  clock[0] = 104.9
  serv._update_tlv_fusion(idle_sm())
  assert serv.tlv_highway_gate is True
  clock[0] = 105.0
  serv._update_tlv_fusion(idle_sm())
  assert serv.tlv_highway_gate is False
  assert serv.tlv_highway_clear_since is None

  # Re-entry to the expressway during the exit window keeps the gate on and
  # restarts the 5 s timer when it is left again.
  reentry = make_serv(2)
  clock[0] = 200.0
  record_category(reentry, 0, clock[0])
  reentry._update_tlv_fusion(idle_sm())
  assert reentry.tlv_highway_gate is True
  record_category(reentry, 2, clock[0])
  reentry._update_tlv_fusion(idle_sm())
  clock[0] = 202.0
  reentry._update_tlv_fusion(idle_sm())
  assert reentry.tlv_highway_gate is True
  record_category(reentry, 0, clock[0])
  reentry._update_tlv_fusion(idle_sm())
  assert reentry.tlv_highway_gate is True
  assert reentry.tlv_highway_clear_since is None
  record_category(reentry, 2, clock[0])
  reentry._update_tlv_fusion(idle_sm())
  clock[0] = 206.9
  reentry._update_tlv_fusion(idle_sm())
  assert reentry.tlv_highway_gate is True
  clock[0] = 207.0
  reentry._update_tlv_fusion(idle_sm())
  assert reentry.tlv_highway_gate is False


def test_stale_category_is_treated_as_non_expressway(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  capture_events(monkeypatch)
  serv = make_serv(2)
  record_category(serv, 0, clock[0])
  serv._update_tlv_fusion(idle_sm())
  assert serv.tlv_highway_gate is True

  # The 2 Hz navi stops; 2 s after the last category it is no longer fresh.
  record_category(serv, 0, 100.0)
  clock[0] = 100.0 + TLV_HIGHWAY_FRESH_S + 0.05
  serv._update_tlv_fusion(idle_sm())
  assert serv.tlv_highway_gate is True  # exit window started, still gated
  assert serv.tlv_highway_clear_since is not None
  gate_off = clock[0] + TLV_HIGHWAY_EXIT_S
  clock[0] = gate_off - 0.01
  serv._update_tlv_fusion(idle_sm())
  assert serv.tlv_highway_gate is True
  clock[0] = gate_off
  serv._update_tlv_fusion(idle_sm())
  assert serv.tlv_highway_gate is False


def test_legacy_json_roadcate_without_carrot_navi_does_not_gate(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  events = capture_events(monkeypatch)
  serv = make_serv(2)
  serv.roadcate = 0  # the legacy JSON default
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  assert serv.tlv_highway_gate is False
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  assert [e for e in events if e[0] == "tlv_highway_gate"] == []


def test_gate_logs_exactly_one_event_per_transition(monkeypatch):
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  events = capture_events(monkeypatch)
  serv = make_serv(2)
  assert TLV_HIGHWAY_CATEGORIES == (0, 1)

  record_category(serv, 0, clock[0])
  for _ in range(5):
    serv._update_tlv_fusion(idle_sm())
    clock[0] += 0.05
  gate = [e for e in events if e[0] == "tlv_highway_gate"]
  assert len(gate) == 1 and gate[0][1]["on"] is True

  # Still non-expressway after the exit window: exactly one off event.
  record_category(serv, 2, clock[0])
  end = clock[0] + TLV_HIGHWAY_EXIT_S
  while clock[0] < end - 1e-6:
    serv._update_tlv_fusion(idle_sm())
    clock[0] += 0.05
  clock[0] = end
  serv._update_tlv_fusion(idle_sm())
  gate = [e for e in events if e[0] == "tlv_highway_gate"]
  assert len(gate) == 2
  assert gate[0][1] == {"on": True, "road_category": 0}
  assert gate[1][1] == {"on": False, "road_category": 2}


def test_updated_carrot_navi_records_category_always(monkeypatch):
  """The gate reads only the carrotNavi path; a stale/None lane is recorded too."""
  from test_carrot_navi_serv import _SubMaster, _message, _serv

  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = _serv()
  serv.tlv_navi_road_category = 2
  serv.tlv_navi_road_category_t = 50.0

  data = _message()
  data["laneCurrent"] = {"meta": {"present": True, "sequence": 1}, "roadCategory": 0}
  assert serv._update_carrot_navi(_SubMaster(data))
  assert serv.tlv_navi_road_category == 0
  assert serv.tlv_navi_road_category_t == 100.0

  # Lane absent: recorded as None (and self.roadcate is left alone).
  clock[0] = 101.0
  data = _message()
  data["laneCurrent"] = {"meta": {"present": False, "sequence": 2}}
  serv.roadcate = 8
  assert serv._update_carrot_navi(_SubMaster(data))
  assert serv.tlv_navi_road_category is None
  assert serv.tlv_navi_road_category_t == 101.0
  assert serv.roadcate == 8
