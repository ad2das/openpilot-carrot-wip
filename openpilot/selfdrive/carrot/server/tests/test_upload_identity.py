from pathlib import Path

import pytest

from openpilot.selfdrive.carrot import telemetry_origin

ROOT = Path(__file__).resolve().parents[5]


class _FakeParams:
  def __init__(self, values):
    self._values = values

  def get(self, key, *args, **kwargs):
    return self._values.get(key)


def _use_fallback_origin(monkeypatch, tmp_path):
  monkeypatch.setattr(telemetry_origin, "_cache", None)
  monkeypatch.setattr(telemetry_origin, "ORIGIN_FILE", tmp_path / "missing.json")


def test_snapshot_reports_upstream_identity_and_default_override(tmp_path, monkeypatch):
  pytest.importorskip("aiohttp")
  from openpilot.selfdrive.carrot.server.services import popular_values

  _use_fallback_origin(monkeypatch, tmp_path)

  by_name = {
    "DisableDM": {"min": 0, "max": 2, "default": 0, "unit": None},
    "DriverMonitoringMode": {"min": 0, "max": 2, "default": 0, "unit": None},
  }
  data = {"apilot": 1, "params": [{"name": name, **by_name[name]} for name in by_name]}
  groups = {"driver": [{"name": name} for name in by_name]}
  params = _FakeParams({
    "CarSelected3": "IONIQ5_PE",
    "DongleId": "test-dongle",
    "DisableDM": 2,
    "DriverMonitoringMode": 1,
  })

  monkeypatch.setattr(popular_values, "HAS_PARAMS", True)
  monkeypatch.setattr(popular_values, "Params", lambda: params)
  monkeypatch.setattr(popular_values, "get_settings_cached", lambda: (data, groups, by_name, []))
  monkeypatch.setattr(
    popular_values, "get_param_values",
    lambda names, defaults: {"DisableDM": 2, "DriverMonitoringMode": 1},
  )

  payload = popular_values.build_snapshot_payload()
  assert payload is not None
  assert payload["repo_remote"] == telemetry_origin.reported_git_remote()
  assert payload["repo_id"] == "ajouatom/openpilot"
  assert payload["app_commit"] == telemetry_origin.reported_git_commit()
  assert payload["values"]["DisableDM"] == 0
  assert "DisableDM" in payload["param_catalog"]
  assert payload["values"]["DriverMonitoringMode"] == 1


def test_support_and_vision_metadata_report_upstream():
  pytest.importorskip("aiohttp")
  from openpilot.selfdrive.carrot.server.services import support_discord, vision_diag

  support_meta = support_discord.support_metadata()
  assert support_meta["branch"] == telemetry_origin.reported_git_branch()
  assert support_meta["commit"] == telemetry_origin.reported_git_short_commit()
  assert support_meta["commitDate"] == telemetry_origin.reported_git_commit_datetime()

  vision_meta = vision_diag._diagnostic_metadata(None)
  assert vision_meta["branch"] == telemetry_origin.reported_git_branch()
  assert vision_meta["commit"] == telemetry_origin.reported_git_short_commit()
  assert vision_meta["commitDate"] == telemetry_origin.reported_git_commit_datetime()


def _source(rel_path):
  return (ROOT / rel_path).read_text(encoding="utf-8")


def test_carrot_man_payloads_use_reported_identity():
  source = _source("openpilot/selfdrive/carrot/carrot_man.py")
  assert "reported_git_remote()" in source
  assert "reported_git_branch()" in source
  assert "reported_git_commit()" in source
  assert "reported_git_commit_date()" in source
  assert "reported_param_value(key, v)" in source
  assert '_pstr("GitRemote")' not in source
  assert '_pstr("GitCommit")' not in source
  assert 'self._param_text("GitRemote")' not in source


def test_diagnostic_payloads_use_reported_identity():
  for rel_path in (
    "openpilot/selfdrive/carrot/server/services/vision_diag.py",
    "openpilot/selfdrive/carrot/server/services/support_discord.py",
    "openpilot/selfdrive/carrot/server/features/dashcam/upload.py",
  ):
    source = _source(rel_path)
    assert "reported_git_short_commit()" in source
    assert "reported_git_commit_datetime()" in source
    assert '"rev-parse", "--short", "HEAD"' not in source
