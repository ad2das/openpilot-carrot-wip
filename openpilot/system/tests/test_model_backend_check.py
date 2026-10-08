import json
import subprocess
import sys
from pathlib import Path

from openpilot.common.basedir import BASEDIR

HELPER = Path(BASEDIR) / "scripts" / "carrot_model_backend_check.py"

TG_QCOM = {
  "openpilot.selfdrive.modeld.modeld": {
    "default": {"WARP_DEV": "QCOM", "QUEUE_DEV": "QCOM"},
    "usbgpu": {"WARP_DEV": "QCOM", "QUEUE_DEV": "AMD"},
  },
  "openpilot.selfdrive.modeld.dmonitoringmodeld": {"default": {"DEV": "QCOM"}},
}
TG_CPU = {
  "openpilot.selfdrive.modeld.modeld": {
    "default": {"WARP_DEV": "CPU", "QUEUE_DEV": "CPU"},
    "usbgpu": {"WARP_DEV": "CPU", "QUEUE_DEV": "AMD"},
  },
  "openpilot.selfdrive.modeld.dmonitoringmodeld": {"default": {"DEV": "CPU"}},
}


def _run(tmp_path: Path, content: str | None, *, on_device: bool) -> subprocess.CompletedProcess:
  tg_path = tmp_path / "tg_input_devices.json"
  if content is not None:
    tg_path.write_text(content, encoding="utf-8")
  marker = tmp_path / "TICI"
  if on_device:
    marker.write_text("", encoding="utf-8")
  return subprocess.run([sys.executable, str(HELPER), str(tg_path), "--tici-marker", str(marker)],
                        capture_output=True, text=True, check=False)


def test_off_device_cpu_backend_passes(tmp_path: Path) -> None:
  result = _run(tmp_path, json.dumps(TG_CPU), on_device=False)
  assert result.returncode == 0, result.stdout + result.stderr


def test_off_device_missing_file_passes(tmp_path: Path) -> None:
  result = _run(tmp_path, None, on_device=False)
  assert result.returncode == 0, result.stdout + result.stderr


def test_device_qcom_backend_passes(tmp_path: Path) -> None:
  result = _run(tmp_path, json.dumps(TG_QCOM), on_device=True)
  assert result.returncode == 0, result.stdout + result.stderr
  assert "QCOM" in result.stdout


def test_device_cpu_backend_fails(tmp_path: Path) -> None:
  result = _run(tmp_path, json.dumps(TG_CPU), on_device=True)
  assert result.returncode == 1
  assert "WARP_DEV" in result.stderr


def test_device_missing_file_fails(tmp_path: Path) -> None:
  result = _run(tmp_path, None, on_device=True)
  assert result.returncode == 1


def test_device_invalid_json_fails(tmp_path: Path) -> None:
  result = _run(tmp_path, "{not json", on_device=True)
  assert result.returncode == 1


def test_device_missing_entries_fail(tmp_path: Path) -> None:
  result = _run(tmp_path, json.dumps({"openpilot.selfdrive.modeld.modeld": {"default": {"WARP_DEV": "QCOM"}}}), on_device=True)
  assert result.returncode == 1
