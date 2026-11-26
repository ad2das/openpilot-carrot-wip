import threading

from pytest_mock import MockerFixture

from openpilot.system.ui.lib.networkmanager import NMDeviceState, NMDeviceStateReason
from openpilot.system.ui.lib.wifi_manager import ConnectStatus, WifiManager, WifiState


def _run_scanner_once(mocker: MockerFixture, *, active: bool, status: ConnectStatus, elapsed: float,
                      device_state: int = NMDeviceState.DISCONNECTED):
  wm = WifiManager.__new__(WifiManager)
  wm._exit = False
  wm._active = active
  wm._device_state = device_state
  wm._wifi_state = WifiState(status=status)
  wm._last_network_scan = 100.0
  wm._request_scan = mocker.MagicMock()

  mocker.patch("openpilot.system.ui.lib.wifi_manager.time.monotonic", side_effect=[100.0 + elapsed, 100.0 + elapsed])

  def stop_after_iteration(_):
    wm._exit = True

  mocker.patch("openpilot.system.ui.lib.wifi_manager.time.sleep", side_effect=stop_after_iteration)
  wm._network_scanner()
  return wm


def test_background_scans_after_five_seconds_when_disconnected(mocker):
  wm = _run_scanner_once(mocker, active=False, status=ConnectStatus.DISCONNECTED, elapsed=5.1)

  wm._request_scan.assert_called_once_with()


def test_background_does_not_scan_while_connected(mocker):
  wm = _run_scanner_once(mocker, active=False, status=ConnectStatus.CONNECTED, elapsed=60.0)

  wm._request_scan.assert_not_called()


def test_scanner_skips_request_scan_while_device_unavailable(mocker):
  for state in (NMDeviceState.UNAVAILABLE, NMDeviceState.UNMANAGED):
    wm = _run_scanner_once(mocker, active=True, status=ConnectStatus.DISCONNECTED, elapsed=60.0, device_state=state)
    wm._request_scan.assert_not_called()


def test_unavailable_to_disconnected_wakes_scanner_and_refreshes_active_ui(mocker):
  wm = WifiManager.__new__(WifiManager)
  wm._exit = True
  wm._active = True
  wm._wifi_device = "/org/freedesktop/NetworkManager/Devices/1"
  wm._device_state = NMDeviceState.UNAVAILABLE
  wm._wifi_state = WifiState()
  wm._user_epoch = 0
  wm._connections = {}
  wm._last_network_scan = 123.0
  wm._scan_event = threading.Event()
  wm._update_networks = mocker.MagicMock()

  assert wm.is_device_unavailable is True
  wm._handle_state_change(NMDeviceState.DISCONNECTED, NMDeviceState.UNAVAILABLE, NMDeviceStateReason.NONE)

  assert wm.is_device_unavailable is False
  assert wm._last_network_scan == 0.0
  assert wm._scan_event.is_set() is True
  wm._update_networks.assert_called_once_with(block=False)


def test_set_active_wakes_scanner_when_networks_empty(mocker):
  wm = WifiManager.__new__(WifiManager)
  wm._exit = True
  wm._active = False
  wm._networks = []
  wm._last_network_scan = 123.0
  wm._scan_event = threading.Event()
  wm._init_wifi_state = mocker.MagicMock()
  wm._update_networks = mocker.MagicMock()

  wm.set_active(True)

  assert wm._last_network_scan == 0.0
  assert wm._scan_event.is_set() is True
  wm._init_wifi_state.assert_called_once_with(block=False)
  wm._update_networks.assert_called_once_with(block=False)


def test_initialize_starts_monitor_before_scanner_and_refreshes_active_ui(mocker):
  order = []
  wm = WifiManager.__new__(WifiManager)
  wm._exit = False
  wm._active = True
  wm._connections = {"weedle": "/conn/1"}
  wm._tethering_ssid = "weedle"
  wm._monitor_ready = threading.Event()
  wm._wait_for_wifi_device = lambda: order.append("wait_device")

  def start_state():
    order.append("start_state")
    wm._monitor_ready.set()

  wm._state_thread = mocker.MagicMock(start=mocker.MagicMock(side_effect=start_state))
  wm._scan_thread = mocker.MagicMock(start=mocker.MagicMock(side_effect=lambda: order.append("start_scan")))
  wm._init_connections = lambda: order.append("init_connections")
  wm._init_wifi_state = lambda: order.append("init_wifi_state")
  wm._update_networks = mocker.MagicMock(side_effect=lambda **kw: order.append(("update_networks", kw)))
  wm._get_tethering_password = lambda: "pw"

  started = []

  class _SyncThread:
    def __init__(self, target, daemon=True):
      self._target = target

    def start(self):
      started.append(True)
      self._target()

  mocker.patch("openpilot.system.ui.lib.wifi_manager.threading.Thread", _SyncThread)
  wm._initialize()

  assert started == [True]
  assert order.index("start_state") < order.index("init_wifi_state") < order.index("start_scan")
  assert ("update_networks", {"block": False}) in order
  wm._exit = True


def _make_init_state_wm(mocker: MockerFixture):
  wm = WifiManager.__new__(WifiManager)
  wm._exit = True
  wm._active = False
  wm._wifi_device = "/org/freedesktop/NetworkManager/Devices/1"
  wm._device_state = NMDeviceState.UNKNOWN
  wm._device_state_seq = 0
  wm._wifi_state = WifiState()
  wm._user_epoch = 0
  wm._connections = {"Home": "/conn/home"}
  wm._callback_queue = []
  wm._activated = []
  wm._need_auth = []
  wm._scan_event = threading.Event()
  wm._conn_monitor = mocker.MagicMock()
  wm._router_main = mocker.MagicMock()
  wm._update_networks = mocker.MagicMock()
  wm._update_active_connection_info = mocker.MagicMock()
  wm._get_active_wifi_connection = mocker.MagicMock(return_value=(None, None))
  return wm


def test_init_wifi_state_applies_device_state_when_uncontested(mocker):
  wm = _make_init_state_wm(mocker)
  reply = mocker.MagicMock()
  reply.body = [("u", NMDeviceState.UNAVAILABLE)]
  wm._router_main.send_and_get_reply.return_value = reply

  wm._init_wifi_state(block=True)

  assert wm._device_state == NMDeviceState.UNAVAILABLE
  assert wm.is_device_unavailable is True


def test_init_wifi_state_does_not_clobber_newer_state_change_during_get_state(mocker):
  wm = _make_init_state_wm(mocker)
  stale_reply = mocker.MagicMock()
  stale_reply.body = [("u", NMDeviceState.UNAVAILABLE)]

  def race_state_change(_msg):
    wm._handle_state_change(NMDeviceState.DISCONNECTED, NMDeviceState.UNAVAILABLE, NMDeviceStateReason.NONE)
    return stale_reply

  wm._router_main.send_and_get_reply.side_effect = race_state_change

  wm._init_wifi_state(block=True)

  assert wm._device_state == NMDeviceState.DISCONNECTED
  assert wm.is_device_unavailable is False
  assert wm._scan_event.is_set() is True


def test_init_wifi_state_does_not_clobber_activated_during_active_conn_lookup(mocker):
  wm = _make_init_state_wm(mocker)
  reply = mocker.MagicMock()
  reply.body = [("u", NMDeviceState.DISCONNECTED)]
  wm._router_main.send_and_get_reply.return_value = reply

  def race_activated(router=None):
    if router is wm._conn_monitor:
      return ("/conn/home", {})
    wm._handle_state_change(NMDeviceState.ACTIVATED, NMDeviceState.SECONDARIES, NMDeviceStateReason.NONE)
    return (None, None)

  wm._get_active_wifi_connection.side_effect = race_activated

  wm._init_wifi_state(block=True)

  assert wm._device_state == NMDeviceState.ACTIVATED
  assert wm._wifi_state == WifiState(ssid="Home", status=ConnectStatus.CONNECTED)
