import sys
from types import ModuleType

from openpilot.selfdrive.carrot.server.services import device_info


class _Hardware:
  def get_device_type(self):
    return "tici"

  def get_imei(self, slot):
    assert slot == 0
    return "351234567890123"

  def get_serial(self):
    return "hardware-serial"

  def get_modem_version(self):
    return "RM500Q"

  def get_network_info(self):
    return {"technology": "LTE", "operator": "Carrier", "state": "CONNECTED"}

  def get_sim_info(self):
    return {"sim_state": ["READY"]}


class _SerialUnavailableHardware(_Hardware):
  def get_serial(self):
    raise RuntimeError("serial service is starting")


def test_tools_device_info_keeps_hardware_identity_outside_params(monkeypatch):
  monkeypatch.setattr(device_info, "get_param_values", lambda _names, _defaults: {
    "DongleId": "dongle-id",
    "HardwareSerial": "",
    "DevicePosition": "center",
    "GitBranch": "carrot-wip",
    "GitCommit": "2df34f5d",
    "GitCommitDate": "2026-09-10",
    "GitPullTime": "2026-09-10T10:11:12Z",
  })
  hardware_module = ModuleType("openpilot.system.hardware")
  hardware_module.HARDWARE = _Hardware()
  monkeypatch.setitem(sys.modules, "openpilot.system.hardware", hardware_module)
  monkeypatch.setattr(device_info, "_boot_time", lambda: "2026-09-10T10:11:12Z")

  info = device_info.get_tools_device_info()

  assert info["identity"] == {
    "device_type": "tici",
    "imei": "351234567890123",
    "imei_available": True,
    "dongle_id": "dongle-id",
    "hardware_serial": "hardware-serial",
    "position": "center",
  }
  assert info["connectivity"]["carrier"] == "Carrier"
  assert info["connectivity"]["sim_state"] == "READY"
  assert info["runtime"] == {"boot_time": "2026-09-10T10:11:12Z"}
  assert "iccid" not in str(info).lower()


def test_tools_device_info_keeps_imei_when_another_hardware_field_is_unavailable(monkeypatch):
  monkeypatch.setattr(device_info, "get_param_values", lambda _names, _defaults: {"HardwareSerial": "param-serial"})
  hardware_module = ModuleType("openpilot.system.hardware")
  hardware_module.HARDWARE = _SerialUnavailableHardware()
  monkeypatch.setitem(sys.modules, "openpilot.system.hardware", hardware_module)

  info = device_info.get_tools_device_info()

  assert info["identity"]["imei"] == "351234567890123"
  assert info["identity"]["hardware_serial"] == "param-serial"


def test_get_wifi_networks_uses_cached_nmcli_without_rescan(monkeypatch):
  calls = []

  class _Proc:
    def __init__(self, rc, stdout):
      self.returncode = rc
      self.stdout = stdout

  def fake_run(cmd, **_kwargs):
    calls.append(cmd)
    assert cmd[-2:] == ["--rescan", "no"]
    return _Proc(0, "yes:HomeWifi:WPA2:80\nno:Cafe:WPA2:40\n")

  monkeypatch.setattr(device_info.subprocess, "run", fake_run)

  nets = device_info.get_wifi_networks()

  assert len(calls) == 1
  assert [n["ssid"] for n in nets] == ["HomeWifi", "Cafe"]
  assert nets[0]["connected"] is True


def test_get_wifi_networks_falls_back_to_auto_rescan_when_cache_empty(monkeypatch):
  calls = []

  class _Proc:
    def __init__(self, rc, stdout):
      self.returncode = rc
      self.stdout = stdout

  def fake_run(cmd, **_kwargs):
    calls.append(cmd[-1])
    if cmd[-1] == "no":
      return _Proc(0, "")
    return _Proc(0, "no:GarageAP:WPA2:65\n")

  monkeypatch.setattr(device_info.subprocess, "run", fake_run)

  nets = device_info.get_wifi_networks()

  assert calls == ["no", "auto"]
  assert [n["ssid"] for n in nets] == ["GarageAP"]


def test_refresh_device_network_keeps_previous_wifi_on_transient_nmcli_failure(monkeypatch):
  with device_info._network_cache_lock:
    prev_cache = dict(device_info._network_cache)
    device_info._network_cache["data"] = {
      "wifi": [{"ssid": "HomeWifi", "connected": True, "security": "WPA2", "signal": 80, "secure": True}],
      "ip_address": "192.168.0.10",
    }
  try:
    monkeypatch.setattr(device_info, "_fetch_wifi_networks", lambda: None)
    monkeypatch.setattr(device_info, "get_wifi_ip_address", lambda: "192.168.0.10")
    monkeypatch.setattr(device_info, "_network_param_values", lambda: {
      "tethering_enabled": False, "roaming_enabled": False, "gsm_metered": False, "apn": "",
    })

    refreshed = device_info.refresh_device_network()

    assert [n["ssid"] for n in refreshed["wifi"]] == ["HomeWifi"]
  finally:
    with device_info._network_cache_lock:
      device_info._network_cache.clear()
      device_info._network_cache.update(prev_cache)
