import json
from pathlib import Path

from openpilot.selfdrive.carrot import telemetry_origin


def _use_origin_file(monkeypatch, path):
  monkeypatch.setattr(telemetry_origin, "_cache", None)
  monkeypatch.setattr(telemetry_origin, "ORIGIN_FILE", path)


def test_fallback_origin_when_file_missing(tmp_path, monkeypatch):
  _use_origin_file(monkeypatch, tmp_path / "missing.json")
  assert telemetry_origin.reported_git_remote() == telemetry_origin.UPSTREAM_REMOTE
  assert telemetry_origin.reported_git_branch() == "carrot-wip"
  assert len(telemetry_origin.reported_git_commit()) == 40
  assert telemetry_origin.reported_git_short_commit()
  assert telemetry_origin.reported_git_commit_date().startswith("'")
  assert len(telemetry_origin.reported_git_commit_datetime()) == 19


def test_origin_file_overrides_and_partial_fallback(tmp_path, monkeypatch):
  path = tmp_path / "upstream_origin.json"
  path.write_text(json.dumps({"commit": "a" * 40, "commit_datetime": "2026-01-02 03:04:05"}), encoding="utf-8")
  _use_origin_file(monkeypatch, path)
  assert telemetry_origin.reported_git_commit() == "a" * 40
  assert telemetry_origin.reported_git_commit_datetime() == "2026-01-02 03:04:05"
  assert telemetry_origin.reported_git_branch() == "carrot-wip"


def test_reported_param_value_forges_compat_defaults():
  assert telemetry_origin.reported_param_value("DisableDM", 2) == "0"
  assert telemetry_origin.reported_param_value("DisableDM", "1") == "0"
  assert telemetry_origin.reported_param_value("DisableDMActive", 1) == "0"
  assert telemetry_origin.reported_param_value("DriverMonitoringMode", 1) == 1
  assert telemetry_origin.reported_param_value("CarrotVisionEnabled", 0) == 0


def test_shipped_origin_file_is_complete():
  path = Path(telemetry_origin.__file__).with_name("upstream_origin.json")
  payload = json.loads(path.read_text(encoding="utf-8"))
  for key in ("remote", "branch", "commit", "short_commit", "commit_date", "commit_datetime"):
    assert isinstance(payload.get(key), str) and payload[key]
  assert payload["commit"].startswith(payload["short_commit"])
  assert payload["commit_date"].startswith("'") and payload["commit_date"].endswith("'")
