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
from openpilot.selfdrive.carrot.carrot_functions import CarrotPlanner, XState
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


def tlv_message(state: str = "green", conf: float = 0.75, status: str = "running", box=BOX):
  msg = log.Event.new_message()
  msg.init("carrotTlv")
  t = msg.carrotTlv
  t.status = status
  t.state = state
  t.stateConf = conf
  t.relevantTrackId = 1
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
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("green", conf=0.75)))
  assert serv.traffic_state == 2
  assert serv.traffic_state_src == 1


def test_mode2_left_arrow_uses_left_turn_path():
  serv = make_serv(2)
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("greenLeft"),
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
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("redLeft"),
                                 carState=SimpleNamespace(leftBlinker=True)))
  assert serv.traffic_state == 3
  # greenLeft + carrotMan atcType intent (same definition tlvd uses)
  serv2 = make_serv(2)
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("greenLeft"),
                                  carrotMan=SimpleNamespace(atcType="atc left")))
  assert serv2.traffic_state == 3
  # "turn left" atcType also counts
  serv3 = make_serv(2)
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("red")))
  serv3._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("left"),
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
  serv2._update_tlv_fusion(FakeSM(carrotTlv=tlv_message("greenLeft")))
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


def carrot_man_message(state: int = 2, src: int = 1, speed: int = 60):
  msg = log.Event.new_message()
  msg.init("carrotMan")
  msg.carrotMan.desiredSpeed = speed
  msg.carrotMan.trafficState = state
  msg.carrotMan.trafficStateSrc = src
  msg.carrotMan.atcType = ""
  msg.carrotMan.activeCarrot = 0
  msg.carrotMan.xDistToTurn = 0
  return msg.carrotMan


def planner_for_gate(stop_sign: bool) -> CarrotPlanner:
  planner = CarrotPlanner.__new__(CarrotPlanner)
  planner.trafficState_carrot = 1
  planner.soft_hold_active = 0
  planner.xState = XState.e2eStopped
  planner.carrot_stay_stop = False
  planner.last_model_stop_sign = stop_sign
  planner.traffic_starting_count = 0
  return planner


def gate_sm(carrot_man, left=False):
  return FakeSM(carrotMan=carrot_man, carState=SimpleNamespace(leftBlinker=left))


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
  planner.tlv_red_likely = False
  planner.last_model_start_sign = False
  planner.coast_cap_active = False
  return planner


def test_red_likely_reader_requires_fresh_tlv_source():
  planner = CarrotPlanner.__new__(CarrotPlanner)
  assert planner._read_tlv_red_likely(FakeSM(carrotMan=carrot_man_message(state=4, src=1))) is True
  assert planner._read_tlv_red_likely(FakeSM(carrotMan=carrot_man_message(state=4, src=0))) is False
  assert planner._read_tlv_red_likely(FakeSM(carrotMan=carrot_man_message(state=1, src=1))) is False
  stale = FakeSM(carrotMan=carrot_man_message(state=4, src=1))
  stale.recv_time['carrotMan'] = 50.0  # frozen clock at 100.0 -> 50 s old
  assert planner._read_tlv_red_likely(stale) is False
  assert planner._read_tlv_red_likely(FakeSM()) is False


def test_coast_cap_only_mode2_without_model_start():
  planner = accel_planner(2)
  planner.tlv_red_likely = True
  planner.last_model_start_sign = True
  planner._update_coast_cap()
  assert planner.coast_cap_active is False  # model is signalling a go
  planner.last_model_start_sign = False
  planner._update_coast_cap()
  assert planner.coast_cap_active is True
  for mode in (0, 1):
    legacy = accel_planner(mode)
    legacy.tlv_red_likely = True
    legacy.last_model_start_sign = False
    legacy._update_coast_cap()
    assert legacy.coast_cap_active is False
  idle = accel_planner(2)
  idle._update_coast_cap()
  assert idle.coast_cap_active is False  # no redLikely


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
