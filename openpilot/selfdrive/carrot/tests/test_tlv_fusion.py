"""Fusion tests: carrot_serv mode gating (0/1 byte-identical) and the carrotTlv
green gate in carrot_functions.

On a PC the fork's Cython modules (params_pyx, msgq) and fcntl are not built, so
this module installs minimal import shims when those imports are missing. On a
device the shims are no-ops and the tests run against the real modules.
"""
import collections
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


def _install_pc_shims() -> None:
  """Stub Unix-only/Cython modules when running on a PC; no-ops on a device."""

  def stub(name: str, attrs: dict) -> None:
    try:
      __import__(name)
      return
    except ImportError:
      pass
    module = types.ModuleType(name)
    for key, value in attrs.items():
      setattr(module, key, value)
    sys.modules[name] = module

  stub("fcntl", {"LOCK_EX": 2, "LOCK_NB": 4, "LOCK_SH": 1, "LOCK_UN": 8,
                 "flock": lambda *a, **k: None, "ioctl": lambda *a, **k: 0, "fcntl": lambda *a, **k: 0})
  stub("termios", {"tcgetattr": lambda *a, **k: [], "tcsetattr": lambda *a, **k: None,
                   "tcflush": lambda *a, **k: None, "tcsendbreak": lambda *a, **k: None,
                   "TCSANOW": 0, "TCSADRAIN": 1, "TCSAFLUSH": 2, "TCIFLUSH": 0, "TCOFLUSH": 1,
                   "TCIOFLUSH": 2, "B9600": 0, "B115200": 0, "B38400": 0, "ISIG": 0, "ICANON": 0})
  stub("serial", {"Serial": type("Serial", (), {}), "EIGHTBITS": 8, "PARITY_NONE": "N",
                  "PARITY_EVEN": "E", "STOPBITS_ONE": 1})
  stub("setproctitle", {"getproctitle": lambda: "", "setproctitle": lambda *a, **k: None})

  class _ZmqSock:
    def setsockopt(self, *a, **k):
      pass

    def connect(self, *a, **k):
      pass

    def send(self, *a, **k):
      pass

    def close(self, *a, **k):
      pass

  class _ZmqContext:
    def socket(self, *a, **k):
      return _ZmqSock()

    def term(self):
      pass

  class _ZmqAgain(Exception):
    pass

  stub("zmq", {"Context": _ZmqContext, "PUSH": 1, "LINGER": 2, "NOBLOCK": 4,
               "error": types.SimpleNamespace(Again=_ZmqAgain)})

  try:
    import msgq
  except ImportError:
    msgq = types.ModuleType("msgq")

    class _IpcError(Exception):
      pass

    for name in ("Context", "Poller", "SubSocket", "PubSocket", "SocketEventHandle"):
      setattr(msgq, name, type(name, (), {}))
    msgq.IpcError = _IpcError
    msgq.MultiplePublishersError = _IpcError
    for name in ("fake_event_handle", "drain_sock_raw", "toggle_fake_events", "set_fake_prefix",
                 "get_fake_prefix", "delete_fake_prefix", "wait_for_one_event"):
      setattr(msgq, name, lambda *args, **kwargs: None)
    sys.modules["msgq"] = msgq

  try:
    import openpilot.common.params_pyx  # noqa: F401
  except ImportError:
    params_pyx = types.ModuleType("openpilot.common.params_pyx")

    class _Params:
      def __init__(self, *a, **k):
        self._store = {}

      def get(self, key, return_default=False, default=None, **k):
        return self._store.get(key, default)

      def get_bool(self, key, **k):
        return bool(self._store.get(key, False))

      def get_int(self, key, **k):
        try:
          return int(self._store.get(key, 0))
        except (TypeError, ValueError):
          return 0

      def get_float(self, key, **k):
        try:
          return float(self._store.get(key, 0.0))
        except (TypeError, ValueError):
          return 0.0

      def put(self, key, value, **k):
        self._store[key] = value

      def put_bool(self, key, value, **k):
        self._store[key] = bool(value)

      def put_nonblocking(self, key, value, **k):
        self._store[key] = value

      def remove(self, key, **k):
        self._store.pop(key, None)

      def check_key(self, key):
        return True

    params_pyx.Params = _Params
    params_pyx.ParamKeyFlag = type("ParamKeyFlag", (), {"PERSISTENT": 1, "CLEAR_ON_MANAGER_START": 2,
                                                        "CLEAR_ON_OFFROAD_TRANSITION": 4, "CLEAR_ON_IGNITION_ON": 8})
    params_pyx.ParamKeyType = type("ParamKeyType", (), {"BOOL": 1, "INT": 2, "FLOAT": 3, "STRING": 4, "JSON": 5, "BYTES": 6})
    params_pyx.UnknownKeyName = type("UnknownKeyName", (Exception,), {})
    sys.modules["openpilot.common.params_pyx"] = params_pyx


_install_pc_shims()

from openpilot.cereal import log
from openpilot.common.filter_simple import MyMovingAverage
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.carrot.carrot_functions import (
  CarrotPlanner,
  TLV_A4_APPROACH_WINDOW_FRAMES,
  TLV_GAS_COOLDOWN_FRAMES,
  TLV_RED_HOLD_FRAMES,
  XState,
)
from openpilot.selfdrive.carrot.carrot_serv import CarrotServ
from openpilot.selfdrive.carrot.driving_mode import DrivingMode

BOX = (900.0, 400.0, 940.0, 440.0)


class FakeSM(dict):
  def __init__(self, **services):
    super().__init__(**services)
    self.seen = dict.fromkeys(services, True)
    self.valid = dict.fromkeys(services, True)
    self.alive = dict.fromkeys(services, True)
    self.updated = dict.fromkeys(services, True)
    self.recv_time = dict.fromkeys(services, 100.0)


def tlv_message(state: str = "green", conf: float = 0.75, status: str = "running", box=BOX,
                go_after_red: bool = False):
  msg = log.Event.new_message()
  msg.init("carrotTlv")
  t = msg.carrotTlv
  t.status = status
  t.state = state
  t.stateConf = conf
  t.relevantTrackId = 1
  t.goAfterRed = go_after_red
  t.init("dets", 1)
  d = t.dets[0]
  d.cls = 2
  d.score = conf
  d.x1, d.y1, d.x2, d.y2 = box
  return t


def make_serv(mode: int) -> CarrotServ:
  serv = CarrotServ.__new__(CarrotServ)
  serv.traffic_light_q = collections.deque(maxlen=20)
  serv.traffic_light_count = -1
  serv.traffic_state = 0
  serv.traffic_state_src = 0
  serv.tlv_mode = mode
  serv.tlv_mode_poll = 0.0
  serv.params = SimpleNamespace(get_int=lambda key, _mode=mode: _mode if key == "CarrotTlvMode" else 0)
  return serv


def snapshot(serv: CarrotServ):
  return serv.traffic_state, serv.traffic_state_src, serv.traffic_light_count, list(serv.traffic_light_q)


def _replay_detect(serv: CarrotServ) -> None:
  for color, cnf in (("Red Light", 0.9), ("Red Light", 0.9), ("Green Light", 0.8)):
    serv.traffic_light(0.5, 0.5, color, cnf)


def test_mode0_and_mode1_ignore_carrotTlv_and_match_no_tlv_outputs():
  def replay(mode, with_tlv):
    serv = make_serv(mode)
    if with_tlv:
      serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
      serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("redLikely")))
      serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green")))
    _replay_detect(serv)
    return snapshot(serv)

  baseline = replay(0, False)
  assert replay(0, True) == baseline
  assert replay(1, True) == baseline
  assert baseline[0] == 2  # the DETECT green trigger still works


def test_mode2_red_likely_is_graded_not_a_confirmed_color():
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("redLikely", conf=0.8)))
  assert (serv.traffic_state, serv.traffic_state_src) == (4, 1)
  assert serv.traffic_light_count == 5
  assert list(serv.traffic_light_q) == []  # graded state is not a color latch
  # A confirmed red message overrides the grade through the existing path.
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  # Weak graded confidence is ignored entirely.
  weak = make_serv(2)
  weak._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("redLikely", conf=0.2)))
  assert snapshot(weak) == (0, 0, -1, [])


def test_mode2_feed_red_then_green_uses_existing_trigger_logic():
  serv = make_serv(2)
  # traffic_light() latches a color only once the same spot repeats.
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  assert serv.traffic_state == 1
  assert serv.traffic_state_src == 1
  assert serv.traffic_light_count == 5
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.75, go_after_red=True)))
  assert serv.traffic_state == 2
  assert serv.traffic_state_src == 1


def test_mode2_green_without_go_after_red_keeps_previous_and_fresh_stays_off():
  # Same-track green (W74D B2): a green from a light NOT confirmed red within
  # 120 s must not feed the confirmed-colour path.
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", conf=0.75)))
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  # goAfterRed False keeps the previous red: a stored 0 would kill the edge.
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.75)))
  assert (serv.traffic_state, serv.traffic_state_src) == (1, 1)
  # goAfterRed True still fires the existing red->green trigger.
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.75, go_after_red=True)))
  assert (serv.traffic_state, serv.traffic_state_src) == (2, 1)
  # A fresh serv with no prior red stays off for a gated green.
  fresh = make_serv(2)
  fresh._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.75)))
  assert snapshot(fresh) == (0, 0, -1, [])


def test_mode2_left_arrow_uses_left_turn_path():
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("greenLeft", go_after_red=True),
                                 carState=SimpleNamespace(leftBlinker=True)))
  assert serv.traffic_state == 3
  assert serv.traffic_state_src == 1


def test_tlv_state_color_maps_every_state_and_intent():
  from openpilot.selfdrive.carrot.carrot_serv import tlv_state_color
  cases = {
    (0, False): None, (0, True): None,
    (1, False): "Red Light", (1, True): "Red Light",
    (2, False): "Yellow Light", (2, True): "Yellow Light",
    (3, False): "Green Light", (3, True): "Green Light",
    (4, False): "Red Light", (4, True): "Left turn",
    (5, False): "Red Light", (5, True): "Left turn",
    (6, False): "Green Light", (6, True): "Left turn",
    (7, False): None, (7, True): None,  # graded redLikely is not a color
  }
  for (state, intent), color in cases.items():
    assert tlv_state_color(state, intent) == color, (state, intent)


def test_mode2_left_phase_intent_from_carstate_and_atctype():
  # redLeft + left blinker -> Left turn -> left path (state 3)
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("redLeft", go_after_red=True),
                                 carState=SimpleNamespace(leftBlinker=True)))
  assert serv.traffic_state == 3
  # greenLeft + carrotMan atcType intent (same definition tlvd uses)
  serv2 = make_serv(2)
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("greenLeft", go_after_red=True),
                                  carrotMan=SimpleNamespace(atcType="atc left")))
  assert serv2.traffic_state == 3
  # "turn left" atcType also counts
  serv3 = make_serv(2)
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("left", go_after_red=True),
                                  carrotMan=SimpleNamespace(atcType="turn left")))
  assert serv3.traffic_state == 3


def test_mode2_left_phases_without_intent_are_straight_phases():
  # redLeft without intent -> Red Light: stays red
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("redLeft")))
  assert serv.traffic_state == 1
  # greenLeft without intent -> Green Light: the red->green edge fires
  serv2 = make_serv(2)
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("greenLeft", go_after_red=True)))
  assert serv2.traffic_state == 2
  # bare left arrow without intent -> Red Light (straight is not allowed)
  serv3 = make_serv(2)
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("left")))
  assert serv3.traffic_state == 1


def test_tlv_mode_is_cached_and_refreshed_at_most_once_per_second(monkeypatch):
  import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
  calls = []

  def fake_get_int(key):
    calls.append(key)
    return 0

  serv = CarrotServ.__new__(CarrotServ)
  serv.traffic_light_q = collections.deque(maxlen=20)
  serv.traffic_light_count = -1
  serv.traffic_state = 0
  serv.traffic_state_src = 0
  serv.tlv_mode = 2
  serv.tlv_mode_poll = 0.0
  serv.params = SimpleNamespace(get_int=fake_get_int)
  clock = [200.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  sm = FakeSM(carrotTlv=tlv_message("green"))
  serv._update_tlv_fusion(sm)
  serv._update_tlv_fusion(sm)
  assert calls == ["CarrotTlvMode"]
  clock[0] += 2.0
  serv._update_tlv_fusion(sm)
  assert calls == ["CarrotTlvMode", "CarrotTlvMode"]


def test_mode2_ignores_bad_or_inactive_tlv_messages():
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", status="suspended")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.2)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("none")))
  empty = log.Event.new_message()
  empty.init("carrotTlv")
  empty.carrotTlv.status = "running"
  empty.carrotTlv.state = "green"
  empty.carrotTlv.stateConf = 0.75
  empty.carrotTlv.init("dets", 0)
  serv._update_tlv_fusion(FakeSM(carrotTlv=empty.carrotTlv))
  assert snapshot(serv) == (0, 0, -1, [])


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
  import openpilot.selfdrive.carrot.carrot_man_input as carrot_man_input
  monkeypatch.setattr(carrot_man_input.time, "monotonic", lambda: 100.0)


def carrot_man_message(state: int = 2, src: int = 1, speed: int = 60,
                       atc_type: str = "", x_dist: int = 0):
  msg = log.Event.new_message()
  msg.init("carrotMan")
  msg.carrotMan.desiredSpeed = speed
  msg.carrotMan.trafficState = state
  msg.carrotMan.trafficStateSrc = src
  msg.carrotMan.atcType = atc_type
  msg.carrotMan.activeCarrot = 0
  msg.carrotMan.xDistToTurn = x_dist
  return msg.carrotMan


def planner_for_gate(stop_sign: bool) -> CarrotPlanner:
  planner = CarrotPlanner.__new__(CarrotPlanner)
  planner.trafficState_carrot = 1
  planner.trafficState_carrot_src = 1
  planner.soft_hold_active = 0
  planner.xState = XState.e2eStopped
  planner.carrot_stay_stop = False
  planner.last_model_stop_sign = stop_sign
  planner.traffic_starting_count = 0
  planner.tlv_standstill_hold_frames = 0
  planner.tlv_hold_run = False
  planner.tlv_gas_cooldown_frames = 0
  planner.coast_cap_active = False
  planner.left_blinker_pre_stop = False
  planner.left_blinker_off_frames = 0
  planner.a4_frame = 0
  planner.a4_approach = collections.deque(maxlen=TLV_A4_APPROACH_WINDOW_FRAMES + 5)
  planner.a4_stop_frame = None
  planner.a4_approach_median = None
  planner.a4_standstill_dists = collections.deque(maxlen=200)
  planner.a4_standstill_ok_frames = 0
  planner.a4_lane_ok = False
  return planner


def gate_sm(carrot_man, left=False, right=False, gas=False, enabled=True):
  return FakeSM(carrotMan=carrot_man,
                carState=SimpleNamespace(leftBlinker=left, rightBlinker=right, gasPressed=gas),
                selfdriveState=SimpleNamespace(enabled=enabled))


def test_tlv_green_holds_start_while_model_signals_stop():
  planner = planner_for_gate(stop_sign=True)
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 1  # held so the edge can fire later


def test_tlv_green_starts_once_both_cues_agree():
  planner = planner_for_gate(stop_sign=True)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  planner.last_model_stop_sign = False
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eCruise
  assert planner.traffic_starting_count == 10.0 / DT_MDL
  assert planner.trafficState_carrot == 2


def test_navi_green_trigger_is_unchanged_even_with_model_stop():
  # src == 0 keeps the pre-existing red->green trigger byte-identical.
  planner = planner_for_gate(stop_sign=True)
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=0)), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eCruise
  assert planner.trafficState_carrot == 2


def test_tlv_left_holds_start_while_model_signals_stop():
  planner = planner_for_gate(stop_sign=True)
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 1  # held so the edge can fire later


def test_tlv_left_starts_once_both_cues_agree():
  planner = planner_for_gate(stop_sign=True)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 20.0, 60.0)
  planner.last_model_stop_sign = False
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eCruise
  assert planner.traffic_starting_count == 10.0 / DT_MDL
  assert planner.trafficState_carrot == 3


def test_navi_left_trigger_is_unchanged_even_with_model_stop():
  # src == 0 keeps the pre-existing red->left trigger byte-identical.
  planner = planner_for_gate(stop_sign=True)
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=0), left=True), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eCruise
  assert planner.trafficState_carrot == 3


def test_tlv_green_decay_leaves_normal_behavior():
  planner = planner_for_gate(stop_sign=True)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  assert planner.trafficState_carrot == 1
  # tlv trigger decays to none before the model cue clears.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 20.0, 60.0)
  assert planner.trafficState_carrot == 0
  assert planner.xState == XState.e2eStopped


def test_red_likely_grade_holds_start_edge_for_green():
  # Confirmed red (1) -> graded redLikely (4) -> green (2): the grade must
  # hold 1 so the red->green edge still fires.
  planner = planner_for_gate(stop_sign=False)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=4, src=1)), 20.0, 60.0)
  assert planner.trafficState_carrot == 1  # grade held, not stored
  assert planner.carrot_stay_stop is False
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  assert planner.xState == XState.e2eCruise
  assert planner.traffic_starting_count == 10.0 / DT_MDL
  assert planner.trafficState_carrot == 2


def test_red_likely_to_green_without_confirmed_red_does_not_start():
  # 0 -> 4 -> 2: the grade is not a confirmed red, so no start edge.
  planner = planner_for_gate(stop_sign=False)
  planner.trafficState_carrot = 0
  planner._update_carrot_man(gate_sm(carrot_man_message(state=4, src=1)), 20.0, 60.0)
  assert planner.trafficState_carrot == 0
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 2


def test_red_likely_grade_holds_start_edge_for_left():
  # 1 -> 4 -> 3 with left intent: same hold for the red->left edge; 4 is not
  # in [1, 2], so the left-blinker stay-stop branch is untouched.
  planner = planner_for_gate(stop_sign=False)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=4, src=1), left=True), 20.0, 60.0)
  assert planner.trafficState_carrot == 1
  assert planner.carrot_stay_stop is False
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 20.0, 60.0)
  assert planner.xState == XState.e2eCruise
  assert planner.traffic_starting_count == 10.0 / DT_MDL
  assert planner.trafficState_carrot == 3


def test_red_likely_grade_never_overwrites_confirmed_state():
  # 0 -> 4 alone leaves the confirmed state untouched (the grade is not red).
  planner = planner_for_gate(stop_sign=False)
  planner.trafficState_carrot = 0
  planner._update_carrot_man(gate_sm(carrot_man_message(state=4, src=1)), 20.0, 60.0)
  assert planner.trafficState_carrot == 0
  assert planner.xState == XState.e2eStopped
  # 4 is not in (2, 3): the grade alone cannot create tlv_start_disagrees
  # (src 1 + model stop sign) either.
  stopped = planner_for_gate(stop_sign=True)
  stopped._update_carrot_man(gate_sm(carrot_man_message(state=4, src=1)), 20.0, 60.0)
  assert stopped.trafficState_carrot == 1  # held, no store, no start
  assert stopped.xState == XState.e2eStopped


# ---------------------------------------------------------------------------
# Graded redLikely -> coast (mode 2 only)
# ---------------------------------------------------------------------------

def accel_planner(detect_mode: int = 2) -> CarrotPlanner:
  planner = CarrotPlanner.__new__(CarrotPlanner)
  planner.trafficLightDetectMode = detect_mode
  planner.myDrivingMode = DrivingMode.Normal
  planner.myHighModeFactor = 1.2
  planner.mySafeFactor = 1.0
  for index in range(7):
    setattr(planner, f"cruiseMaxVals{index}", 1.6)
  planner.tlv_red_active = False
  planner.last_model_start_sign = False
  planner.coast_cap_active = False
  planner.tlv_gas_cooldown_frames = 0
  planner.tlv_standstill_hold_frames = 0
  planner.trafficState_carrot = 0
  planner.trafficState_carrot_src = 0
  planner.carrot_stay_stop = False
  return planner


def test_red_active_reader_requires_fresh_tlv_source():
  planner = CarrotPlanner.__new__(CarrotPlanner)
  assert planner._read_tlv_red_active(FakeSM(carrotMan=carrot_man_message(state=4, src=1))) is True
  # W74D fix: a confirmed red (state 1, src 1) must also read as active.
  assert planner._read_tlv_red_active(FakeSM(carrotMan=carrot_man_message(state=1, src=1))) is True
  assert planner._read_tlv_red_active(FakeSM(carrotMan=carrot_man_message(state=4, src=0))) is False
  assert planner._read_tlv_red_active(FakeSM(carrotMan=carrot_man_message(state=1, src=0))) is False
  assert planner._read_tlv_red_active(FakeSM(carrotMan=carrot_man_message(state=2, src=1))) is False
  assert planner._read_tlv_red_active(FakeSM(carrotMan=carrot_man_message(state=3, src=1))) is False
  stale = FakeSM(carrotMan=carrot_man_message(state=4, src=1))
  stale.recv_time['carrotMan'] = 50.0  # frozen clock at 100.0 -> 50 s old
  assert planner._read_tlv_red_active(stale) is False
  assert planner._read_tlv_red_active(FakeSM()) is False


def test_coast_cap_only_mode2_without_model_start():
  planner = accel_planner(2)
  planner.tlv_red_active = True
  planner.last_model_start_sign = True
  planner._update_coast_cap()
  assert planner.coast_cap_active is False  # model is signalling a go
  planner.last_model_start_sign = False
  planner._update_coast_cap()
  assert planner.coast_cap_active is True
  for mode in (0, 1):
    legacy = accel_planner(mode)
    legacy.tlv_red_active = True
    legacy.last_model_start_sign = False
    legacy._update_coast_cap()
    assert legacy.coast_cap_active is False
  idle = accel_planner(2)
  idle._update_coast_cap()
  assert idle.coast_cap_active is False  # no red active


def test_coast_cap_survives_promotion_from_graded_to_confirmed_red():
  # W74D fix scenario: 80 km/h, red confirmed about 110 m out before
  # supercombo arms its stopSign. 4 -> 1 must not open a cap-off gap frame.
  planner = accel_planner(2)
  for state in (4, 1):
    planner.tlv_red_active = planner._read_tlv_red_active(
      FakeSM(carrotMan=carrot_man_message(state=state, src=1)))
    planner._update_coast_cap()
    assert planner.tlv_red_active is True
    assert planner.coast_cap_active is True


def test_confirmed_red_cap_drops_when_model_signals_go():
  planner = accel_planner(2)
  planner.tlv_red_active = planner._read_tlv_red_active(
    FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
  assert planner.tlv_red_active is True
  planner.last_model_start_sign = True
  planner._update_coast_cap()
  assert planner.coast_cap_active is False


def test_confirmed_red_cap_only_mode2():
  for mode in (0, 1):
    planner = accel_planner(mode)
    planner.tlv_red_active = planner._read_tlv_red_active(
      FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
    assert planner.tlv_red_active is True
    planner._update_coast_cap()
    assert planner.coast_cap_active is False


def test_non_tlv_confirmed_red_keeps_cap_off():
  planner = accel_planner(2)
  planner.tlv_red_active = planner._read_tlv_red_active(
    FakeSM(carrotMan=carrot_man_message(state=1, src=0)))
  assert planner.tlv_red_active is False
  planner._update_coast_cap()
  assert planner.coast_cap_active is False


def test_confirmed_red_cap_gas_cooldown_and_rearm():
  planner = accel_planner(2)
  planner.tlv_red_active = planner._read_tlv_red_active(
    FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
  planner._update_coast_cap()
  assert planner.coast_cap_active is True
  planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=True))
  planner._update_coast_cap()
  assert planner.coast_cap_active is False  # suppressed during the cooldown
  for _ in range(TLV_GAS_COOLDOWN_FRAMES):
    planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=False))
    planner._update_coast_cap()
  assert planner.tlv_gas_cooldown_frames == 0
  assert planner.coast_cap_active is True  # re-armed while still confirmed red


def test_coast_cap_zeroes_accel_and_never_adds_braking():
  planner = accel_planner()
  base = planner.get_carrot_accel(10.0)
  assert base > 0.0
  planner.coast_cap_active = True
  assert planner.get_carrot_accel(10.0) == 0.0
  # Only the upper bound moves: no braking request is created here.
  assert planner.get_carrot_accel(0.0) == 0.0
  legacy = accel_planner(1)
  assert legacy.coast_cap_active is False
  assert legacy.get_carrot_accel(10.0) == pytest.approx(base)


def test_check_model_stopping_records_start_sign():
  def fresh_planner() -> CarrotPlanner:
    planner = CarrotPlanner.__new__(CarrotPlanner)
    planner.xStopFilter = MyMovingAverage(3)
    planner.xStopFilter2 = MyMovingAverage(15)
    planner.vFilter = MyMovingAverage(10)
    planner.stopSignCount = 0
    planner.startSignCount = 0
    planner.trafficState = None
    planner.last_model_stop_sign = True
    planner.last_model_start_sign = False
    return planner

  go = fresh_planner()
  v_go = np.zeros(75, dtype=np.float64)
  v_go[-1] = 10.0  # model velocity implies a start
  go.check_model_stopping(0.0, v_go, 0.1, 0.0, 15.0, np.zeros_like(v_go), 100.0)
  assert go.last_model_start_sign is True

  stop = fresh_planner()
  v_stop = np.zeros(75, dtype=np.float64)
  stop.check_model_stopping(0.0, v_stop, 0.1, 0.0, 15.0, np.zeros_like(v_stop), 100.0)
  assert stop.last_model_start_sign is False
  assert stop.last_model_stop_sign is True


# ---------------------------------------------------------------------------
# W74D (A): TLV red->left start policy
# ---------------------------------------------------------------------------

def test_tlv_left_route_near_allows_start_without_blinker():
  planner = planner_for_gate(stop_sign=False)
  sm = gate_sm(carrot_man_message(state=3, src=1, atc_type="turn left", x_dist=50))
  assert planner._update_carrot_man(sm, 0.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eCruise
  assert planner.traffic_starting_count == 10.0 / DT_MDL
  assert planner.trafficState_carrot == 3


def test_tlv_left_route_far_away_does_not_start():
  planner = planner_for_gate(stop_sign=False)
  sm = gate_sm(carrot_man_message(state=3, src=1, atc_type="turn left", x_dist=200))
  assert planner._update_carrot_man(sm, 0.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eStopped


def test_tlv_left_blinker_before_stop_allows_start():
  planner = planner_for_gate(stop_sign=False)
  # Blinker already on while moving primes the pre-stop condition.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1), left=True), 20.0, 60.0)
  assert planner.left_blinker_pre_stop is True
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eCruise
  assert planner.trafficState_carrot == 3


def test_tlv_left_blinker_after_stop_without_lane_signal_does_not_start():
  planner = planner_for_gate(stop_sign=False)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1)), 20.0, 60.0)
  assert planner.left_blinker_pre_stop is False
  sm = gate_sm(carrot_man_message(state=3, src=1), left=True)  # blinker on only after the stop
  planner._update_carrot_man(sm, 0.0, 60.0)
  assert planner.xState == XState.e2eStopped


def test_tlv_left_blinker_off_gap_over_half_second_does_not_start():
  planner = planner_for_gate(stop_sign=False)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1), left=True), 20.0, 60.0)
  assert planner.left_blinker_pre_stop is True
  for _ in range(11):  # 11 * 0.05 s = 0.55 s > 0.5 s off-gap while stopped
    planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1)), 0.0, 60.0)
  assert planner.left_blinker_pre_stop is False
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped


def test_tlv_left_right_blinker_vetoes_route_left():
  planner = planner_for_gate(stop_sign=False)
  sm = gate_sm(carrot_man_message(state=3, src=1, atc_type="turn left", x_dist=50), right=True)
  planner._update_carrot_man(sm, 0.0, 60.0)
  assert planner.xState == XState.e2eStopped


# ---------------------------------------------------------------------------
# W74D (B1): standstill red hold survives unknown
# ---------------------------------------------------------------------------

def test_tlv_red_hold_survives_unknown_then_green_starts():
  planner = planner_for_gate(stop_sign=False)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1)), 0.0, 60.0)
  assert (planner.trafficState_carrot, planner.trafficState_carrot_src) == (1, 1)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)
  assert planner.trafficState_carrot == 1  # held through the unknown
  assert planner.tlv_standstill_hold_frames == TLV_RED_HOLD_FRAMES
  for _ in range(100):
    planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)
  assert planner.trafficState_carrot == 1
  assert planner.tlv_standstill_hold_frames == TLV_RED_HOLD_FRAMES - 100
  # A (same-track) green still fires the red->green start edge.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 0.0, 60.0)
  assert planner.xState == XState.e2eCruise
  assert planner.traffic_starting_count == 10.0 / DT_MDL
  assert planner.trafficState_carrot == 2


def test_tlv_red_hold_expires_after_120s_and_no_late_start():
  planner = planner_for_gate(stop_sign=False)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1)), 0.0, 60.0)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)  # armed
  for _ in range(TLV_RED_HOLD_FRAMES - 1):
    planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)
  assert planner.trafficState_carrot == 1  # still held at 2400 state-0 frames
  assert planner.tlv_standstill_hold_frames == 1
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)
  assert planner.tlv_standstill_hold_frames == 0
  assert planner.trafficState_carrot == 0  # 120 s elapsed: hold ends, no re-arm
  # A green now cannot create a start edge: the red has been dropped.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 2


def test_tlv_red_hold_gas_clears_and_does_not_rearm():
  planner = planner_for_gate(stop_sign=False)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1)), 0.0, 60.0)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)
  assert planner.tlv_standstill_hold_frames == TLV_RED_HOLD_FRAMES
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0), gas=True), 0.0, 60.0)
  assert planner.tlv_standstill_hold_frames == 0
  assert planner.trafficState_carrot == 0
  # Still unknown and stopped: this run does not re-arm after the gas.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)
  assert planner.tlv_standstill_hold_frames == 0
  assert planner.trafficState_carrot == 0


# ---------------------------------------------------------------------------
# W74D (B3): gas override cooldown
# ---------------------------------------------------------------------------

def test_gas_cooldown_suppresses_and_rearms_the_red_active_cap():
  planner = accel_planner(2)
  planner.coast_cap_active = True
  planner.tlv_red_active = True
  planner.last_model_start_sign = False
  planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=True))
  assert planner.tlv_gas_cooldown_frames == TLV_GAS_COOLDOWN_FRAMES
  assert planner.coast_cap_active is False
  planner._update_coast_cap()
  assert planner.coast_cap_active is False  # suppressed during the cooldown
  for _ in range(TLV_GAS_COOLDOWN_FRAMES - 1):
    planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=False))
    planner._update_coast_cap()
  assert planner.tlv_gas_cooldown_frames == 1
  assert planner.coast_cap_active is False
  planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=False))
  planner._update_coast_cap()
  assert planner.tlv_gas_cooldown_frames == 0
  assert planner.coast_cap_active is True  # rearmed 10 s after release


def test_gas_during_tlv_red_suppresses_edges_for_10s():
  planner = planner_for_gate(stop_sign=False)
  # TLV red with left intent arms the src-1 stay-stop (a TLV control effect).
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1), left=True), 0.0, 60.0)
  assert planner.carrot_stay_stop is True
  planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=True))
  assert planner.tlv_gas_cooldown_frames == TLV_GAS_COOLDOWN_FRAMES
  planner._update_carrot_man(gate_sm(carrot_man_message(state=1, src=1), left=True), 0.0, 60.0)
  assert planner.carrot_stay_stop is False  # suppressed during the cooldown
  # A green edge inside the cooldown is consumed (stored) but must not start.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 2
  for _ in range(TLV_GAS_COOLDOWN_FRAMES):
    planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=False))
  assert planner.tlv_gas_cooldown_frames == 0
  # The consumed edge cannot start later.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 2


# ---------------------------------------------------------------------------
# W74D: non-TLV (src 0) regression table
# ---------------------------------------------------------------------------

def test_non_tlv_src0_transitions_unchanged():
  cases = [
    # (state, left, right, exp_state, exp_started, exp_stay_stop)
    (1, True, False, 1, False, True),
    (2, False, False, 2, True, False),
    (3, True, True, 3, True, False),   # src 0 never used the right-blinker veto
    (2, True, False, 2, False, True),  # left intent: stay-stop, no green start
    (0, False, False, 0, False, False),
  ]
  for state, left, right, exp_state, exp_started, exp_stay_stop in cases:
    planner = planner_for_gate(stop_sign=False)
    planner.trafficState_carrot_src = 0  # the confirmed red came from navi (src 0)
    sm = gate_sm(carrot_man_message(state=state, src=0), left=left, right=right)
    planner._update_carrot_man(sm, 0.0, 60.0)
    assert planner.trafficState_carrot == exp_state, (state, left, right)
    assert (planner.xState == XState.e2eCruise) is exp_started, (state, left, right)
    assert planner.carrot_stay_stop is exp_stay_stop, (state, left, right)


# ---------------------------------------------------------------------------
# W74D (A4): left blinker turned on after stopping, leftmost two lanes
# ---------------------------------------------------------------------------

def _a4_model(edge_y: float, std: float = 0.3):
  edge = SimpleNamespace(x=np.array([0.0, 5.0, 10.0]), y=np.array([0.0, edge_y, 0.0]))
  return SimpleNamespace(roadEdges=[edge], roadEdgeStds=np.array([std, 0.5]))


def _run_a4_stop(planner: CarrotPlanner, approach_y: float, stop_y: float, std: float = 0.3,
                 approach_frames: int = 65, stop_frames: int = 25) -> None:
  for _ in range(approach_frames):
    planner._update_a4_lane_tracking(_a4_model(approach_y, std), 20.0)
  for _ in range(stop_frames):
    planner._update_a4_lane_tracking(_a4_model(stop_y, std), 0.0)


def test_a4_blinker_after_stop_starts_in_leftmost_lane():
  planner = planner_for_gate(stop_sign=False)
  _run_a4_stop(planner, 1.7, 1.7)
  assert planner.a4_lane_ok is True
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eCruise
  assert planner.traffic_starting_count == 10.0 / DT_MDL
  assert planner.trafficState_carrot == 3


def test_a4_blinker_after_stop_starts_in_the_second_lane():
  planner = planner_for_gate(stop_sign=False)
  _run_a4_stop(planner, 5.1, 5.1)
  assert planner.a4_lane_ok is True
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eCruise


def test_a4_third_lane_does_not_start():
  planner = planner_for_gate(stop_sign=False)
  _run_a4_stop(planner, 8.5, 8.5)
  assert planner.a4_lane_ok is False
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped


def test_a4_high_edge_std_does_not_start():
  planner = planner_for_gate(stop_sign=False)
  _run_a4_stop(planner, 1.7, 1.7, std=0.9)
  assert planner.a4_lane_ok is False
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped


def test_a4_approach_standstill_disagreement_does_not_start():
  planner = planner_for_gate(stop_sign=False)
  _run_a4_stop(planner, 1.7, 5.3)
  assert planner.a4_lane_ok is False
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped


def test_a4_missing_model_does_not_start():
  planner = planner_for_gate(stop_sign=False)
  for _ in range(65):
    planner._update_a4_lane_tracking(None, 20.0)
  for _ in range(25):
    planner._update_a4_lane_tracking(SimpleNamespace(), 0.0)
  assert planner.a4_lane_ok is False
  planner._update_carrot_man(gate_sm(carrot_man_message(state=3, src=1), left=True), 0.0, 60.0)
  assert planner.xState == XState.e2eStopped


def test_a4_right_blinker_still_blocks():
  planner = planner_for_gate(stop_sign=False)
  _run_a4_stop(planner, 1.7, 1.7)
  assert planner.a4_lane_ok is True
  sm = gate_sm(carrot_man_message(state=3, src=1), left=True, right=True)
  planner._update_carrot_man(sm, 0.0, 60.0)
  assert planner.xState == XState.e2eStopped
