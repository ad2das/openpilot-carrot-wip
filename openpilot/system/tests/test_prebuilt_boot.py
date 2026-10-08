import hashlib
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from openpilot.common.basedir import BASEDIR


def _bash() -> str:
  if os.name == "nt":
    candidate = "C:/Program Files/Git/bin/bash.exe"
    if Path(candidate).exists():
      return candidate
  return shutil.which("bash") or ""


pytestmark = pytest.mark.skipif(_bash() == "", reason="bash is unavailable")


def _launcher_function(name: str) -> str:
  source = (Path(BASEDIR) / "launch_chffrplus.sh").read_text(encoding="utf-8")
  start = source.index(f"function {name} {{")
  end = source.index("\n}", start) + 2
  return source[start:end]


def _write_model_tree(root: Path, *, prebuilt: bool, stamp: str | None, artifacts: bool) -> Path:
  models = root / "openpilot" / "selfdrive" / "modeld" / "models"
  models.mkdir(parents=True, exist_ok=True)
  if prebuilt:
    (root / "prebuilt").write_text("", encoding="utf-8")
  if stamp is not None:
    (models / ".build_stamp").write_text(stamp, encoding="utf-8")
  if artifacts:
    (models / "tg_input_devices.json").write_text("{}", encoding="utf-8")
    (models / "driving_tinygrad.pkl").write_text("pkl", encoding="utf-8")
  return models


def _run_model_stamp(root: Path, big_model_sha: str = "", big_ready: bool = False) -> str:
  script = (f"DIR={shlex.quote(root.as_posix())}\n"
            f"BIG_MODEL_SHA={shlex.quote(big_model_sha)}\n"
            "FORCE_REBUILD=0\n"
            "git() { echo stamp; }\n"
            f"big_model_artifact_ready() {{ return {0 if big_ready else 1}; }}\n"
            + _launcher_function("invalidate_modeld_build_if_needed") +
            "\ninvalidate_modeld_build_if_needed\necho \"$FORCE_REBUILD\"\n")
  result = subprocess.run([_bash(), "-c", script], capture_output=True, text=True, timeout=60, check=True)
  return result.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("prebuilt, stamp, artifacts, expected", [
  (False, "stamp:", True, "0"),
  (False, None, True, "1"),
  (True, None, True, "0"),
  (True, "outdated:", True, "0"),
  (False, None, False, "1"),
  (True, None, False, "1"),
  (True, "stamp:", False, "1"),
])
def test_model_stamp_trusts_only_prebuilt_artifacts(tmp_path: Path, prebuilt: bool, stamp: str | None,
                                                    artifacts: bool, expected: str) -> None:
  models = _write_model_tree(tmp_path, prebuilt=prebuilt, stamp=stamp, artifacts=artifacts)

  assert _run_model_stamp(tmp_path) == expected

  # A prebuilt tree with intact artifacts refreshes the self-referential stamp;
  # a source checkout never rewrites it as part of invalidation.
  expected_stamp = "stamp:" if (prebuilt and artifacts) else stamp
  if expected_stamp is None:
    assert not (models / ".build_stamp").exists()
  else:
    assert (models / ".build_stamp").read_text(encoding="utf-8") == expected_stamp


@pytest.mark.parametrize("big_stamp, ready, expected", [
  ("model-sha", True, "0"),
  ("model-sha", False, "1"),
  (None, True, "1"),
])
def test_model_stamp_keeps_egpu_invalidation(tmp_path: Path, big_stamp: str | None, ready: bool, expected: str) -> None:
  models = _write_model_tree(tmp_path, prebuilt=True, stamp=None, artifacts=True)
  if big_stamp is not None:
    (models / ".big_model_build_stamp").write_text(big_stamp, encoding="utf-8")

  assert _run_model_stamp(tmp_path, big_model_sha="model-sha", big_ready=ready) == expected


ENSURE_HARNESS = r'''#!/usr/bin/env bash
set -u

SCRIPT="$1"
SIM="$2"
mkdir -p "$SIM"

python3() {
  printf 'python3 %s\n' "$*" >> "$SIM/python.log"
  return "${PYTHON_RC:-0}"
}

scons() {
  printf 'scons %s\n' "$*" >> "$SIM/scons.log"
  return "${SCONS_RC:-0}"
}

source "$SCRIPT"
'''


HEADER_CONTENT = b'{"Foo", {PERSISTENT, BOOL}};\n'


def _run_ensure_params_build(tmp_path: Path, *, prebuilt: bool, module: bool, stamp: str | None,
                             python_rc: str = "0", scons_rc: str = "0"):
  root = tmp_path / "tree"
  (root / "scripts").mkdir(parents=True)
  shutil.copy2(Path(BASEDIR) / "scripts" / "ensure_params_build.sh", root / "scripts" / "ensure_params_build.sh")
  header = root / "openpilot" / "common" / "params_keys.h"
  header.parent.mkdir(parents=True, exist_ok=True)
  header.write_bytes(HEADER_CONTENT)
  if module:
    (root / "openpilot" / "common" / "params_pyx.so").write_text("so", encoding="utf-8")
  if prebuilt:
    (root / "prebuilt").write_text("", encoding="utf-8")

  cache = tmp_path / "scons_cache"
  cache.mkdir()
  if stamp is not None:
    (cache / "carrot_params_keys.sha256").write_text(stamp, encoding="utf-8")

  harness = tmp_path / "ensure_harness.sh"
  harness.write_text(ENSURE_HARNESS, encoding="utf-8", newline="\n")
  sim = tmp_path / "sim"
  env = dict(os.environ)
  env.update({"PYTHON_RC": python_rc, "SCONS_RC": scons_rc, "SCONS_CACHE_DIR": cache.as_posix()})
  result = subprocess.run([_bash(), harness.as_posix(), (root / "scripts" / "ensure_params_build.sh").as_posix(),
                           sim.as_posix()], env=env, capture_output=True, text=True, timeout=60, check=False)
  return result, root, cache, sim


def test_prebuilt_params_module_is_trusted_and_stamped(tmp_path: Path) -> None:
  result, root, cache, sim = _run_ensure_params_build(tmp_path, prebuilt=True, module=True, stamp=None)

  assert result.returncode == 0, result.stdout + result.stderr
  assert not (sim / "scons.log").exists()
  assert (root / "openpilot" / "common" / "params_pyx.so").exists()
  header_hash = hashlib.sha256((root / "openpilot" / "common" / "params_keys.h").read_bytes()).hexdigest()
  assert (cache / "carrot_params_keys.sha256").read_text(encoding="utf-8").strip() == header_hash
  assert "params_check.py" in (sim / "python.log").read_text(encoding="utf-8")


def test_prebuilt_params_check_failure_falls_back_to_rebuild(tmp_path: Path) -> None:
  result, root, cache, sim = _run_ensure_params_build(tmp_path, prebuilt=True, module=True, stamp=None, python_rc="1")

  assert result.returncode != 0
  assert "-u -j4 openpilot/common/params_pyx.so" in (sim / "scons.log").read_text(encoding="utf-8")
  assert not (root / "openpilot" / "common" / "params_pyx.so").exists()
  assert not (cache / "carrot_params_keys.sha256").exists()


def test_source_checkout_still_rebuilds_without_prebuilt_marker(tmp_path: Path) -> None:
  result, root, cache, sim = _run_ensure_params_build(tmp_path, prebuilt=False, module=True, stamp=None)

  assert result.returncode == 0, result.stdout + result.stderr
  assert "-u -j4 openpilot/common/params_pyx.so" in (sim / "scons.log").read_text(encoding="utf-8")
  assert not (root / "openpilot" / "common" / "params_pyx.so").exists()
  assert (cache / "carrot_params_keys.sha256").exists()


def test_source_checkout_keeps_stamp_fast_path(tmp_path: Path) -> None:
  header_hash = hashlib.sha256(HEADER_CONTENT).hexdigest()
  result, _, _, sim = _run_ensure_params_build(tmp_path, prebuilt=False, module=True, stamp=header_hash + "\n")

  assert result.returncode == 0, result.stdout + result.stderr
  assert not (sim / "scons.log").exists()
  assert not (sim / "python.log").exists()


def test_prebuilt_missing_module_still_rebuilds(tmp_path: Path) -> None:
  result, _, cache, sim = _run_ensure_params_build(tmp_path, prebuilt=True, module=False, stamp=None)

  assert result.returncode == 0, result.stdout + result.stderr
  assert "-u -j4 openpilot/common/params_pyx.so" in (sim / "scons.log").read_text(encoding="utf-8")
  assert (cache / "carrot_params_keys.sha256").exists()
