import json
import time

from openpilot.selfdrive.carrot.server.features import atc_signal_test


class FakeParams:
  store: dict = {}

  def get_bool(self, key):
    return bool(self.store.get(key))

  def get(self, key):
    return self.store.get(key)

  def put(self, key, value):
    self.store[key] = value


def test_start_requires_setting_and_writes_monotonic_nonce(monkeypatch):
  monkeypatch.setattr(atc_signal_test, "Params", FakeParams)
  FakeParams.store = {}
  assert atc_signal_test._start() == {"ok": False, "error": "disabled"}
  assert "AtcSignalTestRequest" not in FakeParams.store

  FakeParams.store = {"AutoTurnSignal": True}
  before = time.monotonic_ns()
  started = atc_signal_test._start()
  assert started["ok"] and before <= started["request"] <= time.monotonic_ns()
  assert FakeParams.store["AtcSignalTestRequest"] == started["request"]


def test_result_parses_card_json_and_ignores_garbage(monkeypatch):
  monkeypatch.setattr(atc_signal_test, "Params", FakeParams)
  FakeParams.store = {}
  assert atc_signal_test._result() == {"ok": True, "result": None}
  FakeParams.store = {"AtcSignalTestResult": "{not json"}
  assert atc_signal_test._result()["result"] is None
  FakeParams.store = {"AtcSignalTestResult": json.dumps({"request": 5, "verdict": "steady"})}
  assert atc_signal_test._result()["result"] == {"request": 5, "verdict": "steady"}
