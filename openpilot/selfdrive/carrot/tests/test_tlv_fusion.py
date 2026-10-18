"""Fusion tests: carrot_serv mode gating (0/1 byte-identical) and the carrotTlv
green gate in carrot_functions.

On a PC the fork's Cython modules (params_pyx, msgq) and fcntl are not built, so
this module installs minimal import shims when those imports are missing. On a
device the shims are no-ops and the tests run against the real modules.
"""
import collections
import sys
import time
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
from openpilot.selfdrive.carrot import carrot_functions
from openpilot.selfdrive.carrot.carrot_functions import (
  CarrotPlanner,
  TLV_A4_APPROACH_WINDOW_FRAMES,
  TLV_COAST_MAX_MOVING_FRAMES,
  TLV_COAST_MAX_MOVING_S,
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
                go_after_red: bool = False, rel_id: int = 1, rel_box=BOX):
  msg = log.Event.new_message()
  msg.init("carrotTlv")
  t = msg.carrotTlv
  t.status = status
  t.state = state
  t.stateConf = conf
  t.relevantTrackId = rel_id
  t.goAfterRed = go_after_red
  # W138: the fusion follows relevantBox, not dets[0]; dets is kept so tests can
  # prove the two are independent.
  t.relevantBox.x1, t.relevantBox.y1, t.relevantBox.x2, t.relevantBox.y2 = rel_box
  t.init("dets", 1)
  d = t.dets[0]
  d.cls = 2
  d.score = conf
  d.x1, d.y1, d.x2, d.y2 = box
  return t


def make_serv(detect_mode: int) -> CarrotServ:
  serv = CarrotServ.__new__(CarrotServ)
  serv.traffic_light_q = collections.deque(maxlen=20)
  serv.traffic_light_count = -1
  serv.traffic_state = 0
  serv.traffic_state_src = 0
  serv.tlv_detect_mode = detect_mode
  serv.tlv_detect_mode_poll = 0.0
  serv.tlv_det_support = 0
  serv.tlv_hold_until = 0.0
  serv.carrotCmdIndex = 0
  serv.carrotCmdIndex_last = 0
  serv.carrotCmd = ""
  serv.carrotArg = ""
  serv.params = SimpleNamespace(get_int=lambda key, _mode=detect_mode: _mode if key == "TrafficLightDetectMode" else 0)
  return serv


def snapshot(serv: CarrotServ):
  return serv.traffic_state, serv.traffic_state_src, serv.traffic_light_count, list(serv.traffic_light_q)


def _replay_detect(serv: CarrotServ) -> None:
  for color, cnf in (("Red Light", 0.9), ("Red Light", 0.9), ("Green Light", 0.8)):
    serv.traffic_light(0.5, 0.5, color, cnf)


def test_mode0_ignores_carrotTlv_and_mode1_feeds_like_mode2():
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
  # W129: TrafficLightDetectMode 1 and 2 both run the vision feed; the
  # stop-only vs stop-go distinction is enforced downstream (carrot_functions).
  assert replay(1, True) == replay(2, True)
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


def test_tlv_hold_keeps_src1_state_across_a_05s_carrotTlv_gap(monkeypatch):
  # W143: a src-1 TLV state is held by wall time (0.8 s), not loop count, so the
  # ~0.4 s tlvd inference gaps cannot drop trafficState while TLV red is active.
  import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  assert serv.traffic_state == 1 and serv.traffic_state_src == 1
  # 20 Hz loop stepping 0.05 s with no new carrotTlv: still held at 0.6 s.
  for _ in range(12):
    clock[0] += 0.05
    serv._update_cmd()
  assert serv.traffic_state == 1 and serv.traffic_state_src == 1
  # At 0.85 s the hold has expired: state drops to none.
  clock[0] += 0.25
  serv._update_cmd()
  assert serv.traffic_state == 0 and serv.traffic_state_src == 0


def test_navi_detect_path_still_expires_after_five_loops(monkeypatch):
  import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  serv._handle_detect_command("Red Light,0.5,0.5,0.9")
  assert serv.traffic_state_src == 0 and serv.traffic_light_count == 5
  for _ in range(5):
    serv._update_cmd()
  assert serv.traffic_light_count == 0  # 5 loops of validity, no wall-time hold
  serv._update_cmd()
  assert serv.traffic_light_count == -1
  assert serv.traffic_state == 0


def test_hold_keeps_tlv_det_support_while_held(monkeypatch):
  import openpilot.selfdrive.carrot.carrot_serv as carrot_serv
  clock = [100.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  serv = make_serv(2)
  msg = tlv_message("red")
  msg.relevantDetSupport = 3
  serv._update_tlv_fusion(FakeSM(carrotTlv=msg))
  serv._update_tlv_fusion(FakeSM(carrotTlv=msg))
  assert serv.tlv_det_support == 3
  for _ in range(8):
    clock[0] += 0.05
    serv._update_cmd()
  assert serv.traffic_state == 1 and serv.tlv_det_support == 3


def test_tlv_detect_mode_is_cached_and_refreshed_at_most_once_per_second(monkeypatch):
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
  serv.tlv_detect_mode = 2
  serv.tlv_detect_mode_poll = 0.0
  serv.params = SimpleNamespace(get_int=fake_get_int)
  clock = [200.0]
  monkeypatch.setattr(carrot_serv.time, "monotonic", lambda: clock[0])
  sm = FakeSM(carrotTlv=tlv_message("green"))
  serv._update_tlv_fusion(sm)
  serv._update_tlv_fusion(sm)
  assert calls == ["TrafficLightDetectMode"]
  clock[0] += 2.0
  serv._update_tlv_fusion(sm)
  assert calls == ["TrafficLightDetectMode", "TrafficLightDetectMode"]


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
  empty.carrotTlv.relevantTrackId = -1
  empty.carrotTlv.init("dets", 0)
  serv._update_tlv_fusion(FakeSM(carrotTlv=empty.carrotTlv))
  assert snapshot(serv) == (0, 0, -1, [])


# ---------------------------------------------------------------------------
# W138: fusion follows relevantBox, not dets[0]; the message must be valid.
# ---------------------------------------------------------------------------

def test_fusion_uses_relevant_box_not_dets0():
  serv = make_serv(2)
  # red confirmed with the relevant box far from dets[0]; the recorded position
  # must be the relevantBox centre, not the unrelated detection.
  rel = (900.0, 400.0, 940.0, 440.0)      # centre (920, 420)
  det0 = (100.0, 600.0, 140.0, 640.0)     # centre (120, 640)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", box=det0, rel_box=rel)))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", box=det0, rel_box=rel)))
  assert serv.traffic_state == 1
  assert (serv.traffic_light_q[-1][0], serv.traffic_light_q[-1][1]) == (920.0 / 1928.0, 420.0 / 1208.0)


def test_fusion_returns_when_relevant_track_missing():
  serv = make_serv(2)
  base = snapshot(serv)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", rel_id=-1)))
  assert snapshot(serv) == base


def test_fusion_returns_when_relevant_box_degenerate():
  serv = make_serv(2)
  base = snapshot(serv)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", rel_box=(0.0, 0.0, 0.0, 0.0))))
  assert snapshot(serv) == base


def test_fusion_requires_valid_message():
  serv = make_serv(2)
  base = snapshot(serv)
  msg = tlv_message("red")
  sm = FakeSM(carrotTlv=msg)
  sm.valid["carrotTlv"] = False
  serv._update_tlv_fusion(sm)
  assert snapshot(serv) == base


def test_fusion_accepts_running_message_without_dets():
  # W138: the len(dets) == 0 gate is dropped; a running message with a valid
  # relevantBox feeds the fusion even when dets is empty (relevant track had no
  # detector match this frame).
  serv = make_serv(2)
  msg = log.Event.new_message()
  msg.init("carrotTlv")
  t = msg.carrotTlv
  t.status = "running"
  t.state = "red"
  t.stateConf = 0.75
  t.relevantTrackId = 1
  t.relevantBox.x1, t.relevantBox.y1, t.relevantBox.x2, t.relevantBox.y2 = BOX
  t.init("dets", 0)
  serv._update_tlv_fusion(FakeSM(carrotTlv=t))
  serv._update_tlv_fusion(FakeSM(carrotTlv=t))
  assert serv.traffic_state == 1


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
  import openpilot.selfdrive.carrot.carrot_man_input as carrot_man_input
  monkeypatch.setattr(carrot_man_input.time, "monotonic", lambda: 100.0)


def carrot_man_message(state: int = 2, src: int = 1, speed: int = 60,
                       atc_type: str = "", x_dist: int = 0, det_support: int = 0):
  msg = log.Event.new_message()
  msg.init("carrotMan")
  msg.carrotMan.desiredSpeed = speed
  msg.carrotMan.trafficState = state
  msg.carrotMan.trafficStateSrc = src
  msg.carrotMan.tlvDetSupport = det_support
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
  planner.tlv_coast_moving_frames = 0
  planner.tlv_coast_stop_sign_seen = False
  planner.tlv_coast_released = False
  planner.tlv_veto_active = False
  planner.tlv_veto_started = 0.0
  planner.tlv_red_confirmed_at = None
  planner.tlv_log_veto_t = -1e9
  planner.tlv_log_clear_t = -1e9
  planner.tlv_log_coast_t = -1e9
  # W143: model-clear timer. A model that is already clear has been clear for
  # long enough unless a test explicitly simulates the transition.
  planner.model_stop_clear_since = None if stop_sign else time.monotonic() - 10.0
  return planner


def gate_sm(carrot_man, left=False, right=False, gas=False, enabled=True, model=None, radar=None):
  services = {
    'carrotMan': carrot_man,
    'carState': SimpleNamespace(leftBlinker=left, rightBlinker=right, gasPressed=gas),
    'selfdriveState': SimpleNamespace(enabled=enabled),
  }
  if model is not None:
    services['modelV2'] = model
  if radar is not None:
    services['radarState'] = radar
  return FakeSM(**services)


def test_tlv_green_holds_start_while_model_signals_stop():
  planner = planner_for_gate(stop_sign=True)
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 1  # held so the edge can fire later


def test_tlv_green_starts_once_both_cues_agree():
  planner = planner_for_gate(stop_sign=True)
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  planner.last_model_stop_sign = False
  planner.model_stop_clear_since = time.monotonic() - 2.0  # clear for good
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
  planner.model_stop_clear_since = time.monotonic() - 2.0  # clear for good
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
  planner.last_model_stop_sign = False
  planner.coast_cap_active = False
  planner.tlv_gas_cooldown_frames = 0
  planner.tlv_standstill_hold_frames = 0
  planner.trafficState_carrot = 0
  planner.trafficState_carrot_src = 0
  planner.carrot_stay_stop = False
  planner.tlv_coast_moving_frames = 0
  planner.tlv_coast_stop_sign_seen = False
  planner.tlv_coast_released = False
  planner.tlv_veto_active = False
  planner.tlv_veto_started = 0.0
  planner.tlv_red_confirmed_at = None
  planner.tlv_log_veto_t = -1e9
  planner.tlv_log_clear_t = -1e9
  planner.tlv_log_coast_t = -1e9
  planner.model_stop_sign_t = None
  planner.tlv_fast_gate_open = False
  planner._fast_gate_ep_start = 0.0
  planner._fast_gate_started_slow = False
  planner._fast_gate_applied = False
  planner._fast_gate_opened_by = None
  planner._raw_red_prev = False
  planner._fast_gate_last_red_t = None
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


def test_coast_cap_stays_on_across_a_05s_tlv_gap_while_moving(monkeypatch):
  # W143: while the held src-1 state keeps carrotMan.trafficState at 1, the
  # moving counter must keep accumulating across a carrotTlv gap instead of
  # resetting every second.
  import openpilot.selfdrive.carrot.carrot_functions as carrot_functions
  monkeypatch.setattr(carrot_functions.time, "monotonic", lambda: 100.0)
  planner = accel_planner(2)
  for _ in range(10):
    planner.tlv_red_active = planner._read_tlv_red_active(FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
    planner._update_coast_cap(22.2)
  assert planner.coast_cap_active is True
  assert planner.tlv_coast_moving_frames == 10


def test_coast_cap_ignores_model_start_sign_and_requires_mode2():
  # W74D follow-up: startSign is True while cruising toward a red the model has
  # not noticed (model_v ~ 17-22 m/s), so it must not release the cap.
  planner = accel_planner(2)
  planner.tlv_red_active = True
  planner.last_model_start_sign = True
  planner._update_coast_cap(22.2)
  assert planner.coast_cap_active is True
  planner.last_model_start_sign = False
  planner._update_coast_cap(22.2)
  assert planner.coast_cap_active is True
  for mode in (0, 1):
    legacy = accel_planner(mode)
    legacy.tlv_red_active = True
    legacy._update_coast_cap(22.2)
    assert legacy.coast_cap_active is False
  idle = accel_planner(2)
  idle._update_coast_cap(22.2)
  assert idle.coast_cap_active is False  # no red active


def test_highway_red_with_model_start_sign_keeps_cap_on():
  # Regression for the Claude-found defect: 80 km/h, model_v ~ 22 m/s makes
  # startSign True; the cap must still engage toward a TLV-confirmed red.
  planner = accel_planner(2)
  planner.tlv_red_active = planner._read_tlv_red_active(
    FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
  planner.last_model_start_sign = True
  planner._update_coast_cap(80.0 / 3.6)
  assert planner.coast_cap_active is True


def test_coast_cap_survives_promotion_from_graded_to_confirmed_red():
  # W74D fix scenario: 80 km/h, red confirmed about 110 m out before
  # supercombo arms its stopSign. 4 -> 1 must not open a cap-off gap frame.
  planner = accel_planner(2)
  for state in (4, 1):
    planner.tlv_red_active = planner._read_tlv_red_active(
      FakeSM(carrotMan=carrot_man_message(state=state, src=1)))
    planner._update_coast_cap(22.2)
    assert planner.tlv_red_active is True
    assert planner.coast_cap_active is True


def test_confirmed_red_cap_stays_on_at_standstill_with_model_start_sign():
  planner = accel_planner(2)
  planner.tlv_red_active = planner._read_tlv_red_active(
    FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
  assert planner.tlv_red_active is True
  planner.last_model_start_sign = True
  planner._update_coast_cap(0.0)
  assert planner.coast_cap_active is True  # no creeping into the red


def test_confirmed_red_cap_only_mode2():
  for mode in (0, 1):
    planner = accel_planner(mode)
    planner.tlv_red_active = planner._read_tlv_red_active(
      FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
    assert planner.tlv_red_active is True
    planner._update_coast_cap(22.2)
    assert planner.coast_cap_active is False


def test_non_tlv_confirmed_red_keeps_cap_off():
  planner = accel_planner(2)
  planner.tlv_red_active = planner._read_tlv_red_active(
    FakeSM(carrotMan=carrot_man_message(state=1, src=0)))
  assert planner.tlv_red_active is False
  planner._update_coast_cap(22.2)
  assert planner.coast_cap_active is False


def test_confirmed_red_cap_gas_cooldown_and_rearm():
  planner = accel_planner(2)
  planner.tlv_red_active = planner._read_tlv_red_active(
    FakeSM(carrotMan=carrot_man_message(state=1, src=1)))
  planner._update_coast_cap(0.0)
  assert planner.coast_cap_active is True
  planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=True))
  planner._update_coast_cap(0.0)
  assert planner.coast_cap_active is False  # suppressed during the cooldown
  for _ in range(TLV_GAS_COOLDOWN_FRAMES):
    planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=False))
    planner._update_coast_cap(0.0)
  assert planner.tlv_gas_cooldown_frames == 0
  assert planner.coast_cap_active is True  # re-armed while still confirmed red


def test_coast_timeout_releases_after_12s_moving_without_stop_sign(monkeypatch):
  events = []
  monkeypatch.setattr(carrot_functions, 'cloudlog',
                      SimpleNamespace(event=lambda name, **kw: events.append((name, kw))))
  planner = accel_planner(2)
  planner.tlv_red_active = True
  planner.tlv_red_confirmed_at = 90.0  # frozen clock at 100.0 -> 10 s old
  for _ in range(TLV_COAST_MAX_MOVING_FRAMES):
    planner._update_coast_cap(22.2)
    assert planner.coast_cap_active is True
  planner._update_coast_cap(22.2)
  assert planner.tlv_coast_released is True
  assert planner.coast_cap_active is False
  timeouts = [e for e in events if e[0] == 'tlv_coast_timeout']
  assert len(timeouts) == 1
  assert timeouts[0][1]['v_ego'] == 22.2
  assert timeouts[0][1]['moving_s'] >= TLV_COAST_MAX_MOVING_S
  assert timeouts[0][1]['red_age_s'] == pytest.approx(10.0)
  # Latched: stays released while the red persists, no repeated events.
  for _ in range(50):
    planner._update_coast_cap(22.2)
    assert planner.coast_cap_active is False
  assert len([e for e in events if e[0] == 'tlv_coast_timeout']) == 1
  # Leaving (1, 4) and coming back re-arms a fresh window.
  planner.tlv_red_active = False
  planner._update_coast_cap(22.2)
  assert planner.tlv_coast_released is False
  planner.tlv_red_active = True
  planner._update_coast_cap(22.2)
  assert planner.coast_cap_active is True


def test_coast_timeout_never_releases_when_stop_sign_seen(monkeypatch):
  events = []
  monkeypatch.setattr(carrot_functions, 'cloudlog',
                      SimpleNamespace(event=lambda name, **kw: events.append((name, kw))))
  planner = accel_planner(2)
  planner.tlv_red_active = True
  for i in range(TLV_COAST_MAX_MOVING_FRAMES + 80):
    planner.last_model_stop_sign = (i == 5)  # stopSign arms within the window
    planner._update_coast_cap(22.2)
    assert planner.coast_cap_active is True
  assert planner.tlv_coast_released is False
  assert [e for e in events if e[0] == 'tlv_coast_timeout'] == []


def test_coast_cap_standstill_holds_indefinitely(monkeypatch):
  events = []
  monkeypatch.setattr(carrot_functions, 'cloudlog',
                      SimpleNamespace(event=lambda name, **kw: events.append((name, kw))))
  planner = accel_planner(2)
  planner.tlv_red_active = True
  planner.last_model_start_sign = True
  for _ in range(TLV_COAST_MAX_MOVING_FRAMES + 60):
    planner._update_coast_cap(0.0)
    assert planner.coast_cap_active is True
  assert planner.tlv_coast_released is False
  assert [e for e in events if e[0] == 'tlv_coast_timeout'] == []


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
    planner.model_stop_clear_since = None
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
  planner._update_coast_cap(0.0)
  assert planner.coast_cap_active is False  # suppressed during the cooldown
  for _ in range(TLV_GAS_COOLDOWN_FRAMES - 1):
    planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=False))
    planner._update_coast_cap(0.0)
  assert planner.tlv_gas_cooldown_frames == 1
  assert planner.coast_cap_active is False
  planner._update_tlv_gas_cooldown(SimpleNamespace(gasPressed=False))
  planner._update_coast_cap(0.0)
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


# ---------------------------------------------------------------------------
# W74D follow-up (Claude review): start-veto measurement events
# ---------------------------------------------------------------------------

def _record_events(monkeypatch):
  events = []
  monkeypatch.setattr(carrot_functions, 'cloudlog',
                      SimpleNamespace(event=lambda name, **kw: events.append((name, kw))))
  return events


def test_tlv_start_veto_logs_one_event_per_blocked_edge(monkeypatch):
  events = _record_events(monkeypatch)
  planner = planner_for_gate(stop_sign=True)
  planner.tlv_red_confirmed_at = 97.0  # frozen clock at 100.0 -> 3 s old
  model = SimpleNamespace(position=SimpleNamespace(x=np.array([0.0, 5.0, 42.0])),
                          velocity=SimpleNamespace(x=np.array([12.0, 15.0, 22.0])))
  radar = SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=30.5))
  sm = gate_sm(carrot_man_message(state=2, src=1), model=model, radar=radar)
  for _ in range(3):
    planner._update_carrot_man(sm, 0.0, 60.0)
  vetoes = [e for e in events if e[0] == 'tlv_start_veto']
  assert len(vetoes) == 1  # one per blocked edge, not per frame
  fields = vetoes[0][1]
  assert fields['state_in'] == 2
  assert fields['v_ego'] == 0.0
  assert fields['model_x'] == 42.0
  assert fields['model_v'] == 22.0
  assert fields['lead_d_rel'] == 30.5
  assert fields['left_intent'] is False
  assert fields['red_age_s'] == pytest.approx(3.0)
  assert planner.xState == XState.e2eStopped  # the start is still vetoed
  # The model stop cue clears: the held start fires once and is logged.
  planner.last_model_stop_sign = False
  planner.model_stop_clear_since = time.monotonic() - 2.0
  planner._update_carrot_man(sm, 0.0, 60.0)
  cleared = [e for e in events if e[0] == 'tlv_start_veto_cleared']
  assert len(cleared) == 1
  assert cleared[0][1]['delay_s'] >= 0.0
  assert planner.xState == XState.e2eCruise


def test_tlv_start_veto_logs_left_edge_once(monkeypatch):
  events = _record_events(monkeypatch)
  planner = planner_for_gate(stop_sign=True)
  sm = gate_sm(carrot_man_message(state=3, src=1, atc_type="turn left", x_dist=50))
  for _ in range(2):
    planner._update_carrot_man(sm, 0.0, 60.0)
  vetoes = [e for e in events if e[0] == 'tlv_start_veto']
  assert len(vetoes) == 1
  assert vetoes[0][1]['state_in'] == 3
  assert vetoes[0][1]['left_intent'] is True
  planner.last_model_stop_sign = False
  planner.model_stop_clear_since = time.monotonic() - 2.0
  planner._update_carrot_man(sm, 0.0, 60.0)
  assert planner.xState == XState.e2eCruise
  assert len([e for e in events if e[0] == 'tlv_start_veto_cleared']) == 1


def test_tlv_start_veto_rate_limited_within_2s(monkeypatch):
  events = _record_events(monkeypatch)
  planner = planner_for_gate(stop_sign=True)
  sm = gate_sm(carrot_man_message(state=2, src=1))
  planner._update_carrot_man(sm, 0.0, 60.0)
  assert [e[0] for e in events] == ['tlv_start_veto']
  # The edge decays, then a new blocked edge appears inside the 2 s window.
  planner._update_carrot_man(gate_sm(carrot_man_message(state=0, src=0)), 0.0, 60.0)
  assert planner.tlv_veto_active is False
  planner._update_carrot_man(sm, 0.0, 60.0)
  assert planner.tlv_veto_active is True
  assert [e[0] for e in events] == ['tlv_start_veto']  # second event suppressed


# ---------------------------------------------------------------------------
# W143: sustained model-clear requirement for TLV starts
# ---------------------------------------------------------------------------

def test_tlv_green_edge_with_model_clear_09s_still_vetoed():
  planner = planner_for_gate(stop_sign=False)
  planner.model_stop_clear_since = time.monotonic() - 0.9
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  assert planner.xState == XState.e2eStopped
  assert planner.trafficState_carrot == 1  # the blocked edge keeps holding red


def test_tlv_green_edge_with_model_clear_105s_triggers():
  planner = planner_for_gate(stop_sign=False)
  planner.model_stop_clear_since = time.monotonic() - 1.05
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  assert planner.xState == XState.e2eCruise
  assert planner.trafficState_carrot == 2


def test_model_stop_flicker_resets_the_clear_timer(monkeypatch):
  planner = planner_for_gate(stop_sign=False)
  planner.vFilter = MyMovingAverage(10)
  planner.stopSignCount = 0
  planner.startSignCount = 0
  planner.model_stop_clear_since = time.monotonic() - 0.6
  clock = [time.monotonic()]
  monkeypatch.setattr(carrot_functions.time, "monotonic", lambda: clock[0])
  # A model stop frame resets the timer to None.
  planner.check_model_stopping(0.0, np.array([0.0] * 70 + [0.0]), 0.0, 0.0, 10.0,
                               np.zeros(70), 100.0)
  assert planner.model_stop_clear_since is None
  # Two clear frames 0.3 s later restart it; the edge is still vetoed at 0.6 s.
  clock[0] += 0.3
  planner.check_model_stopping(0.0, np.array([0.0] * 70 + [0.0]), 0.0, 0.0, 60.0,
                               np.zeros(70), 100.0)
  assert planner.model_stop_clear_since is not None
  clock[0] += 0.6
  planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=1)), 20.0, 60.0)
  assert planner.xState == XState.e2eStopped


def test_navi_start_is_unaffected_by_the_model_clear_guard():
  planner = planner_for_gate(stop_sign=True)
  planner.model_stop_clear_since = None
  assert planner._update_carrot_man(gate_sm(carrot_man_message(state=2, src=0)), 20.0, 60.0) == (60.0, False)
  assert planner.xState == XState.e2eCruise
  assert planner.trafficState_carrot == 2


def test_model_clear_guard_logs_one_veto_per_blocked_edge(monkeypatch):
  events = _record_events(monkeypatch)
  planner = planner_for_gate(stop_sign=False)
  planner.model_stop_clear_since = time.monotonic() - 0.2
  sm = gate_sm(carrot_man_message(state=2, src=1))
  for _ in range(3):
    planner._update_carrot_man(sm, 0.0, 60.0)
  assert len([e for e in events if e[0] == 'tlv_start_veto']) == 1
  assert planner.xState == XState.e2eStopped


# ---------------------------------------------------------------------------
# W139c: tlvDetSupport pass-through
# ---------------------------------------------------------------------------

def test_fusion_publishes_det_support_through():
  serv = make_serv(2)
  msg = tlv_message("red", rel_id=1)
  msg.relevantDetSupport = 3
  serv._update_tlv_fusion(FakeSM(carrotTlv=msg))
  assert serv.tlv_det_support == 3
  # graded redLikely (state 7) also carries it
  serv = make_serv(2)
  graded = tlv_message("redLikely", conf=0.75, rel_id=1)
  graded.relevantDetSupport = 2
  serv._update_tlv_fusion(FakeSM(carrotTlv=graded))
  assert serv.tlv_det_support == 2


def test_det_support_is_zero_on_every_rejecting_path():
  # no relevant track
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", rel_id=-1)))
  assert serv.tlv_det_support == 0
  # degenerate relevant box
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red", rel_box=(0.0, 0.0, 0.0, 0.0))))
  assert serv.tlv_det_support == 0
  # low conf red
  serv = make_serv(2)
  msg = tlv_message("red", conf=0.1)
  msg.relevantDetSupport = 5
  serv._update_tlv_fusion(FakeSM(carrotTlv=msg))
  assert serv.tlv_det_support == 0
  # not running
  serv = make_serv(2)
  msg = tlv_message("red", status="suspended")
  msg.relevantDetSupport = 5
  serv._update_tlv_fusion(FakeSM(carrotTlv=msg))
  assert serv.tlv_det_support == 0


# ---------------------------------------------------------------------------
# W139c: fast-road phantom gate (carrot_functions)
# ---------------------------------------------------------------------------

def _gate_planner(v_ego_mps, det_support, model_stop_age_s=None):
  planner = accel_planner(2)
  planner.model_stop_sign_t = (100.0 - model_stop_age_s) if model_stop_age_s is not None else None
  sm = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=det_support))
  raw = planner._read_tlv_red_active(sm)
  planner.tlv_red_active = raw and planner._apply_fast_gate(raw, sm, v_ego_mps) if raw else (
    planner._apply_fast_gate(False, sm, v_ego_mps) and False)
  planner._update_coast_cap(v_ego_mps)
  return planner


def test_fast_gate_below_threshold_is_ungated():
  # 20 m/s < 22.2: byte-identical cap behaviour (cap on with raw red).
  planner = _gate_planner(20.0, det_support=0)
  assert planner.coast_cap_active is True


def test_fast_gate_high_speed_needs_support_or_model():
  # 25 m/s, support 1, no model stop -> no cap.
  planner = _gate_planner(25.0, det_support=1)
  assert planner.coast_cap_active is False
  # support 2 -> cap.
  planner = _gate_planner(25.0, det_support=2)
  assert planner.coast_cap_active is True


def test_fast_gate_model_stop_window():
  # model stop 2.9 s ago -> cap; 3.1 s ago -> no cap (support 0).
  planner = _gate_planner(25.0, det_support=0, model_stop_age_s=2.9)
  assert planner.coast_cap_active is True
  planner = _gate_planner(25.0, det_support=0, model_stop_age_s=3.1)
  assert planner.coast_cap_active is False


def test_fast_gate_latches_after_support_drops():
  planner = accel_planner(2)
  sm = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=2))
  planner.tlv_red_active = planner._read_tlv_red_active(sm)
  planner.tlv_red_active = planner._apply_fast_gate(True, sm, 25.0)
  assert planner.tlv_fast_gate_open is True
  # support falls back to 0; the latch keeps the gate open.
  sm2 = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=0))
  assert planner._apply_fast_gate(True, sm2, 25.0) is True


def test_fast_gate_resets_when_red_ends(monkeypatch):
  import openpilot.selfdrive.carrot.carrot_functions as cf
  clock = [100.0]
  monkeypatch.setattr(cf.time, "monotonic", lambda: clock[0])
  planner = accel_planner(2)
  sm = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=2))
  planner._apply_fast_gate(True, sm, 25.0)
  assert planner.tlv_fast_gate_open is True
  clock[0] += 0.6
  planner._apply_fast_gate(False, sm, 25.0)   # red absent >= 0.5 s
  assert planner.tlv_fast_gate_open is False
  # A new red episode at high speed with no support is gated again.
  sm0 = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=0))
  assert planner._apply_fast_gate(True, sm0, 25.0) is False


def test_fast_gate_episode_started_slow_is_never_gated():
  planner = accel_planner(2)
  # Episode begins at 20 m/s (< 22.2): started_slow latches.
  sm = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=0))
  assert planner._apply_fast_gate(True, sm, 20.0) is True
  # Speed rises to 25 m/s with no support: still ungated.
  assert planner._apply_fast_gate(True, sm, 25.0) is True

def test_fast_gate_survives_brief_red_flicker(monkeypatch):
  # W139c review: a ~0.1-0.2 s raw-red flicker inside one real red must NOT drop
  # the latched cap or re-arm the gate (CHECKPOINT 1b 18:47:59 / 19:13:13).
  import openpilot.selfdrive.carrot.carrot_functions as cf
  clock = [100.0]
  monkeypatch.setattr(cf.time, "monotonic", lambda: clock[0])
  planner = accel_planner(2)
  sm = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=2))

  def run(raw, v=25.0):
    # exact update() wiring
    if raw:
      return planner._apply_fast_gate(True, sm, v)
    planner._apply_fast_gate(False, sm, v)
    return False

  assert run(True) is True            # gate opens (det support)
  assert planner.tlv_fast_gate_open is True
  clock[0] += 0.1
  assert run(False) is False          # 0.1 s flicker: gate stays latched/open
  assert planner.tlv_fast_gate_open is True
  clock[0] += 0.1
  assert run(True) is True            # red returns; same episode
  assert planner.tlv_fast_gate_open is True
  # A support-less new sample does not close the latch.
  sm0 = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=0))
  assert planner._apply_fast_gate(True, sm0, 25.0) is True


def test_fast_gate_resets_after_long_red_absence(monkeypatch):
  import openpilot.selfdrive.carrot.carrot_functions as cf
  clock = [100.0]
  monkeypatch.setattr(cf.time, "monotonic", lambda: clock[0])
  planner = accel_planner(2)
  sm = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=2))
  assert planner._apply_fast_gate(True, sm, 25.0) is True
  clock[0] += 0.6
  assert planner._apply_fast_gate(False, sm, 25.0) is False
  assert planner.tlv_fast_gate_open is False   # reset after >= 0.5 s
  # New episode with no support is gated again.
  sm0 = FakeSM(carrotMan=carrot_man_message(state=1, src=1, det_support=0))
  assert planner._apply_fast_gate(True, sm0, 25.0) is False