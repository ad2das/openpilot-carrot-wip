import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car import interfaces
from opendbc.car.hyundai import hyundaicanfd, interface
from opendbc.car.hyundai.atc_blinker import classic_atc_enabled
from opendbc.car.hyundai.values import CAR, HyundaiFlags, HyundaiSafetyFlags


@pytest.fixture
def setting_params(monkeypatch):
  values = {"AutoTurnSignal": False}

  class MemoryParams:
    def get_int(self, key):
      return int(values.get(key, 0))

    def get_bool(self, key):
      return bool(values.get(key, False))

    def get(self, key):
      return values.get(key, "")

    def put_int(self, key, value):
      values[key] = value

  for module in (interface, interfaces, hyundaicanfd):
    monkeypatch.setattr(module, "Params", MemoryParams)
  return values


@pytest.mark.parametrize("candidate", list(CAR), ids=lambda candidate: candidate.name)
def test_setting_only_enables_classic_palisade_for_every_hyundai_platform(setting_params, candidate):
  """Compare real CP results without changing an unrelated upstream torque key.

  origin/carrot-wip 05c1678033c1c778f0386fba00c1e105bb26c427 spells the torque
  key KIA_K5_DL3_HEV_24, but its CAR enum is KIA_K5_DL3_24_HEV. Only that exact
  KeyError is allowed, and only if both setting values produce the same error.
  Once upstream fixes it, the normal successful-CP comparison runs unchanged.
  The independent fingerprint guard below still runs for every candidate.
  """
  def build(enabled):
    setting_params["AutoTurnSignal"] = enabled
    try:
      return interface.CarInterface.get_params(candidate, gen_empty_fingerprint(), [], False, False, False), None
    except KeyError as error:
      assert candidate == CAR.KIA_K5_DL3_24_HEV
      assert type(error) is KeyError and error.args == (candidate,)
      return None, (type(error), error.args)

  baseline, baseline_error = build(False)
  cp, enabled_error = build(True)
  assert baseline_error == enabled_error
  expected = candidate == CAR.HYUNDAI_PALISADE
  if baseline_error is None:
    assert not baseline.flags & HyundaiFlags.CLASSIC_ATC
    assert all(not config.safetyParam & HyundaiSafetyFlags.CLASSIC_ATC for config in baseline.safetyConfigs)
    assert not classic_atc_enabled(baseline)
    assert bool(cp.flags & HyundaiFlags.CLASSIC_ATC) == expected
    assert any(config.safetyParam & HyundaiSafetyFlags.CLASSIC_ATC for config in cp.safetyConfigs) == expected
    assert classic_atc_enabled(cp) == expected
    assert not cp.flags & HyundaiFlags.ENABLE_BLINKERS
    if expected:
      assert cp.safetyConfigs[-1].safetyModel == structs.CarParams.SafetyModel.hyundai

    # The opt-in cannot change any other parameter, including another safety bit.
    enabled_values = cp.to_dict()
    enabled_values["flags"] &= ~HyundaiFlags.CLASSIC_ATC.value
    for config in enabled_values["safetyConfigs"]:
      config["safetyParam"] &= ~HyundaiSafetyFlags.CLASSIC_ATC.value
    assert enabled_values == baseline.to_dict()

  # Exercise the fingerprint guard independently of get_params withholding flags.
  forced = structs.CarParams(carFingerprint=candidate, flags=HyundaiFlags.CLASSIC_ATC.value,
                             safetyConfigs=[dict(safetyModel="hyundai", safetyParam=HyundaiSafetyFlags.CLASSIC_ATC.value)])
  assert classic_atc_enabled(forced) == expected


@pytest.mark.parametrize("flags_type", (HyundaiFlags, HyundaiSafetyFlags))
def test_classic_atc_bit_does_not_overlap_any_other_flag(flags_type):
  bit = flags_type.CLASSIC_ATC.value
  assert bit > 0 and bit & (bit - 1) == 0
  for name, flag in flags_type.__members__.items():
    if name != "CLASSIC_ATC":
      assert not bit & flag.value, name


@pytest.mark.parametrize(("flags", "safety_configs"), (
  (0, [("hyundai", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, [("hyundai", 0)]),
  (HyundaiFlags.CLASSIC_ATC, [("hyundai", HyundaiSafetyFlags.CANFD_CLUSTER_DIRECT_TX)]),
  (HyundaiFlags.CLASSIC_ATC | HyundaiFlags.CANFD, [("hyundai", 1024)]),
  (HyundaiFlags.CLASSIC_ATC | HyundaiFlags.LEGACY, [("hyundai", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, [("hyundaiCanfd", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, [("hyundaiLegacy", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, [("noOutput", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, [("hyundai", 0), ("noOutput", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, [("noOutput", 0), ("hyundai", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, [("hyundai", 1024), ("hyundai", 1024)]),
  (HyundaiFlags.CLASSIC_ATC, []),
))
def test_runtime_gate_rejects_inconsistent_or_unsupported_configuration(setting_params, flags, safety_configs):
  setting_params["AutoTurnSignal"] = True
  cp = interface.CarInterface.get_params(CAR.HYUNDAI_PALISADE, gen_empty_fingerprint(), [], False, False, False)
  assert classic_atc_enabled(cp)
  cp.flags = int(flags)
  cp.safetyConfigs = [dict(safetyModel=model, safetyParam=int(bits)) for model, bits in safety_configs]
  assert not classic_atc_enabled(cp)


@pytest.mark.parametrize("extra_flags", (HyundaiFlags.LEGACY, HyundaiFlags.CANFD))
def test_palisade_name_cannot_bypass_safety_model_gate(setting_params, extra_flags):
  setting_params["AutoTurnSignal"] = True
  cp = interfaces.CarInterfaceBase.get_std_params(CAR.HYUNDAI_PALISADE)
  cp.flags |= extra_flags.value
  cp = interface.CarInterface._get_params(cp, CAR.HYUNDAI_PALISADE, gen_empty_fingerprint(), [], False, False, False)
  assert not cp.flags & HyundaiFlags.CLASSIC_ATC
  assert not cp.safetyConfigs[-1].safetyParam & HyundaiSafetyFlags.CLASSIC_ATC


def test_setting_change_only_affects_next_car_params(setting_params):
  def make():
    return interface.CarInterface.get_params(CAR.HYUNDAI_PALISADE, gen_empty_fingerprint(), [], False, False, False)
  before = make()
  setting_params["AutoTurnSignal"] = True
  after = make()
  assert not before.flags & HyundaiFlags.CLASSIC_ATC
  assert after.flags & HyundaiFlags.CLASSIC_ATC
  setting_params["AutoTurnSignal"] = False
  assert after.flags & HyundaiFlags.CLASSIC_ATC
  assert not make().flags & HyundaiFlags.CLASSIC_ATC


def test_atc_setting_does_not_disable_any_ecu(setting_params, monkeypatch):
  setting_params["AutoTurnSignal"] = True
  cp = interface.CarInterface.get_params(CAR.HYUNDAI_PALISADE, gen_empty_fingerprint(), [], False, False, False)
  calls = []
  monkeypatch.setattr(interface, "disable_ecu", lambda *args, **kwargs: calls.append((args, kwargs)))
  interface.CarInterface.init(cp, None, None)
  assert calls == []


def test_experimental_setting_is_registered_off_by_default_in_all_locales():
  root = Path(__file__).resolve().parents[5]
  catalog = json.loads((root / "openpilot/selfdrive/carrot_settings.json").read_text(encoding="utf-8"))
  entries = [entry for entry in catalog["params"] if entry["name"] == "AutoTurnSignal"]
  assert len(entries) == 1
  setting = entries[0]
  assert (setting["default"], setting["control"], setting["risk"]) == (0, "toggle", "high")
  assert all(setting[key] for key in ("title", "descr", "etitle", "edescr", "ctitle", "cdescr"))
  assert '{"AutoTurnSignal", {PERSISTENT, BOOL, "0"}}' in (root / "openpilot/common/params_keys.h").read_text(encoding="utf-8")


def make_receive_interface(enabled=True):
  ci = interface.CarInterface.__new__(interface.CarInterface)
  ci.CP = SimpleNamespace(flags=HyundaiFlags.CLASSIC_ATC if enabled else 0)
  ci.v_ego_cluster_seen = False
  observations = []

  def state_update(parsers):
    observations.append(list(ci.CS.atc_diagnostic_replies))
    return structs.CarState()

  ci.CS = SimpleNamespace(out=structs.CarState(), update=state_update, update_button_enable=lambda events: False,
                          cluster_speed_hyst_gap=0, cluster_min_speed=0, atc_diagnostic_replies=[("old",)])
  parser_packets = []
  ci.can_parsers = {"pt": SimpleNamespace(update=parser_packets.append, can_valid=True, bus_timeout=False)}
  return ci, observations, parser_packets


def test_received_reply_snapshot_precedes_actual_base_update_and_copies_bytes():
  ci, observations, parser_packets = make_receive_interface()
  payload = bytearray.fromhex("0250030000000000")
  packets = [(100, [(0x778, payload, 0), (0x770, bytes(8), 0), (0x778, bytes(8), 128),
                     (0x778, bytes(8), 1), (0x778, bytes(7), 0)])]
  result = ci.update(packets)
  expected = [(0x778, bytes(payload), 0, 100)]
  payload[0] = 0xFF
  assert observations == [expected]
  assert ci.CS.atc_diagnostic_replies == expected
  assert parser_packets[0] is packets
  assert result.canValid
  ci.update([])
  assert observations[-1] == []
  assert ci.CS.atc_diagnostic_replies == []


def test_reply_capture_is_bounded_and_preserves_packet_timestamps():
  ci, observations, _ = make_receive_interface()
  packets = [(100 + index, [(0x778, bytes([index]) * 8, 0)]) for index in range(40)]
  ci.update(packets)
  assert len(observations[0]) == 32
  assert [reply[3] for reply in observations[0]] == list(range(100, 132))


@pytest.mark.parametrize("stamp", (0, -1, True, 1.5))
def test_reply_capture_rejects_invalid_receive_timestamps(stamp):
  ci, observations, _ = make_receive_interface()
  ci.update([(stamp, [(0x778, bytes(8), 0)])])
  assert observations == [[]]


def test_disabled_reply_capture_stays_empty_and_preserves_base_update():
  ci, observations, parser_packets = make_receive_interface(enabled=False)
  packets = [(100, [(0x778, bytes(8), 0)])]
  ci.update(packets)
  assert observations == [[]]
  assert parser_packets == [packets]
