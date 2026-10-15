"""Gating tests for tlvd.

Covers process_config.enable_tlvd truth table (started x TrafficLightDetectMode x
CarrotTlvDisabled), tlvd.main() early returns (TrafficLightDetectMode=0 and the
CarrotTlvDisabled latch, crash loop) and crash_loop_guard start bookkeeping.

On a PC the fork's Cython modules (params_pyx, msgq) and Unix-only modules are
not built, so this module installs minimal import shims when those imports are
missing. On a device the shims are no-ops and the tests run against the real
modules.
"""
import sys
import time
import types

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

      def put_int(self, key, value, **k):
        self._store[key] = int(value)

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

  stub("openpilot.cereal.car", {"CarParams": type("CarParams", (), {})})
  stub("openpilot.system.sentry", {"set_tag": lambda *a, **k: None, "capture_exception": lambda *a, **k: None})


_install_pc_shims()

from openpilot.selfdrive.carrot.tlv import tlvd
from openpilot.system.manager import process_config


_KEY_TYPES = {
  "TrafficLightDetectMode": int,
  "CarrotTlvDisabled": bool,
  "CarrotTlvRecentStarts": str,
  "CarrotTlvBackend": str,
  "CarrotTlvModelPath": str,
}


class FakeParams:
  """Mirrors params_pyx exact-type casting: put() rejects a mismatched type."""

  def __init__(self, values=None):
    self.values = dict(values or {})
    self.writes = []

  def get(self, key, block=False, return_default=False):
    return self.values.get(key)

  def get_int(self, key, block=False):
    value = self.values.get(key, 0)
    return 0 if value is None else int(value)

  def get_bool(self, key, block=False):
    return bool(self.values.get(key, False))

  def put(self, key, value, block=True):
    expected = _KEY_TYPES.get(key)
    if expected is None:
      raise KeyError(f"unknown param: {key}")
    if type(value) is not expected:
      raise TypeError(f"Type mismatch while writing param {key}: {type(value)!r} != {expected.__name__}")
    self.writes.append((key, value))
    self.values[key] = value

  def put_int(self, key, value, block=True):
    if type(value) is not int:
      raise TypeError(f"put_int requires int for {key}: {type(value)!r}")
    self.put(key, value, block)

  def put_bool(self, key, value, block=True):
    if type(value) is not bool:
      raise TypeError(f"put_bool requires bool for {key}: {type(value)!r}")
    self.put(key, value, block)


def _install_fake_params_module(monkeypatch, params):
  module = types.ModuleType("openpilot.common.params")
  module.Params = lambda: params
  monkeypatch.setitem(sys.modules, "openpilot.common.params", module)


def _install_recording_runner(monkeypatch):
  started = []

  def make(params):
    started.append(params)
    return types.SimpleNamespace(run=lambda: None)

  monkeypatch.setattr(tlvd, "TlvRunner", make)
  return started


# -- enable_tlvd -------------------------------------------------------------

def test_enable_tlvd_truth_table():
  for started in (False, True):
    for tld in (0, 1, 2):
      for disabled in (False, True):
        params = FakeParams({"TrafficLightDetectMode": tld, "CarrotTlvDisabled": disabled})
        result = process_config.enable_tlvd(started, params, None)
        assert result == (started and tld >= 1 and not disabled), (started, tld, disabled)


def test_enable_tlvd_absent_params_is_off():
  assert process_config.enable_tlvd(True, FakeParams(), None) is False


def test_enable_tlvd_param_read_failure_is_off():
  class Broken(FakeParams):
    def get_int(self, key, block=False):
      raise RuntimeError("param backend unavailable")

  assert process_config.enable_tlvd(True, Broken(), None) is False


def _install_fake_signal_color_module(monkeypatch, requested):
  module = types.ModuleType("openpilot.selfdrive.modeld.signal_color_shadow")
  module.requested = lambda: requested
  monkeypatch.setitem(sys.modules, "openpilot.selfdrive.modeld.signal_color_shadow", module)


def test_enable_tlvd_disabled_when_signal_color_requested(monkeypatch):
  _install_fake_signal_color_module(monkeypatch, True)
  params = FakeParams({"TrafficLightDetectMode": 2})
  assert process_config.enable_tlvd(True, params, None) is False


def test_enable_tlvd_unchanged_when_signal_color_not_requested(monkeypatch):
  _install_fake_signal_color_module(monkeypatch, False)
  for started in (False, True):
    for tld in (0, 1, 2):
      for disabled in (False, True):
        params = FakeParams({"TrafficLightDetectMode": tld, "CarrotTlvDisabled": disabled})
        result = process_config.enable_tlvd(started, params, None)
        assert result == (started and tld >= 1 and not disabled), (started, tld, disabled)


# -- tlv_dsp_probe process registration ---------------------------------------

def test_tlv_dsp_probe_process_registered():
  proc = process_config.managed_processes["tlv_dsp_probe"]
  assert proc.module == "openpilot.selfdrive.carrot.tlv.dsp_probe"
  assert proc.should_run is process_config.enable_tlv_dsp_probe


def test_tlv_dsp_probe_never_runs():
  # DSP path parked 2026-10-14 (no lld on AGNOS); the probe gating is off for
  # every started x TrafficLightDetectMode combination.
  for started in (False, True):
    for tld in (0, 1, 2):
      params = FakeParams({"TrafficLightDetectMode": tld})
      assert process_config.enable_tlv_dsp_probe(started, params, None) is False, (started, tld)
  assert process_config.enable_tlv_dsp_probe(True, FakeParams(), None) is False


# -- crash_loop_guard --------------------------------------------------------

def test_crash_loop_guard_five_starts_ok_sixth_sets_latch():
  params = FakeParams({"CarrotTlvRecentStarts": ""})
  base = 1000.0
  for i in range(5):
    assert tlvd.crash_loop_guard(params, base + i) is False
  assert all(key != "CarrotTlvDisabled" for key, _ in params.writes)
  assert tlvd.crash_loop_guard(params, base + 5) is True
  assert params.values["CarrotTlvDisabled"] is True
  assert "1005.0" in params.values["CarrotTlvRecentStarts"]


def test_crash_loop_guard_drops_entries_outside_window():
  params = FakeParams({"CarrotTlvRecentStarts": "10,20,30"})
  assert tlvd.crash_loop_guard(params, 1000.0) is False
  assert params.values["CarrotTlvRecentStarts"] == "1000.0"
  assert "CarrotTlvDisabled" not in params.values


def test_crash_loop_guard_window_boundary_is_inclusive():
  params = FakeParams({"CarrotTlvRecentStarts": "700.0,699.9"})
  assert tlvd.crash_loop_guard(params, 1000.0) is False
  assert params.values["CarrotTlvRecentStarts"] == "700.0,1000.0"


def test_crash_loop_guard_ignores_garbage_values():
  params = FakeParams({"CarrotTlvRecentStarts": "abc,,12x, 34 ,inf,-inf,nan"})
  assert tlvd.crash_loop_guard(params, 100.0) is False
  assert params.values["CarrotTlvRecentStarts"] == "34.0,100.0"


def test_crash_loop_guard_accepts_bytes_stored_value():
  params = FakeParams({"CarrotTlvRecentStarts": b"98.0,99.0"})
  assert tlvd.crash_loop_guard(params, 100.0) is False
  assert params.values["CarrotTlvRecentStarts"] == "98.0,99.0,100.0"


def test_crash_loop_guard_survives_recent_starts_write_failure():
  class NoPut(FakeParams):
    def put(self, key, value, block=True):
      raise RuntimeError("disk full")

  assert tlvd.crash_loop_guard(NoPut(), 100.0) is False


def test_crash_loop_guard_never_writes_carrot_tlv_mode():
  params = FakeParams({"CarrotTlvRecentStarts": ""})
  base = 1000.0
  for i in range(6):
    tlvd.crash_loop_guard(params, base + i)
  assert all(key != "CarrotTlvMode" for key, _ in params.writes)


def test_strict_fake_rejects_the_old_bytes_idiom():
  params = FakeParams({"CarrotTlvDisabled": False, "CarrotTlvRecentStarts": ""})
  with pytest.raises(TypeError):
    params.put("CarrotTlvDisabled", b"1")
  with pytest.raises(TypeError):
    params.put("CarrotTlvDisabled", "1")
  with pytest.raises(TypeError):
    params.put("CarrotTlvRecentStarts", b"1.0")
  params.put("CarrotTlvDisabled", True)
  params.put_bool("CarrotTlvDisabled", True)
  assert params.values["CarrotTlvDisabled"] is True


# -- tlvd.main() -------------------------------------------------------------

def test_main_returns_early_when_traffic_light_detect_off(monkeypatch):
  params = FakeParams({"TrafficLightDetectMode": 0})
  _install_fake_params_module(monkeypatch, params)
  started = _install_recording_runner(monkeypatch)
  tlvd.main()
  assert started == []
  assert params.writes == []


def test_main_returns_early_when_disabled_latch_set(monkeypatch):
  params = FakeParams({"TrafficLightDetectMode": 2, "CarrotTlvDisabled": True})
  _install_fake_params_module(monkeypatch, params)
  started = _install_recording_runner(monkeypatch)
  tlvd.main()
  assert started == []
  assert params.writes == []


def test_main_starts_runner_when_gates_on(monkeypatch):
  params = FakeParams({"TrafficLightDetectMode": 2})
  _install_fake_params_module(monkeypatch, params)
  started = _install_recording_runner(monkeypatch)
  tlvd.main()
  assert started == [params]
  assert "CarrotTlvRecentStarts" in params.values


def test_main_crash_loop_sets_latch_and_skips_runner(monkeypatch):
  now = time.monotonic()
  seed = ",".join(f"{now - i:.1f}" for i in range(1, 6))
  params = FakeParams({"TrafficLightDetectMode": 2, "CarrotTlvRecentStarts": seed})
  _install_fake_params_module(monkeypatch, params)
  started = _install_recording_runner(monkeypatch)
  tlvd.main()
  assert started == []
  assert params.values["CarrotTlvDisabled"] is True


def test_handle_hard_exit_sets_latch(monkeypatch):
  params = FakeParams({})
  runner = tlvd.TlvRunner.__new__(tlvd.TlvRunner)
  runner.params = params
  runner._hard_exit_reason = "rss_hard"
  published = []
  monkeypatch.setattr(runner, "_publish", lambda status, reason: published.append((status, reason)))
  runner._handle_hard_exit()
  assert params.values["CarrotTlvDisabled"] is True
  assert published == [("error", "rss_hard")]
  assert all(key != "CarrotTlvMode" for key, _ in params.writes)


def test_main_treats_raising_traffic_light_detect_read_as_off(monkeypatch):
  class RaisingParams(FakeParams):
    def get_int(self, key, block=False):
      if key == "TrafficLightDetectMode":
        raise RuntimeError("param read failed")
      return super().get_int(key, block)

  params = RaisingParams({})
  _install_fake_params_module(monkeypatch, params)
  started = _install_recording_runner(monkeypatch)
  tlvd.main()
  assert started == []
  assert params.writes == []
