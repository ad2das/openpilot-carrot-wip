import hashlib
import json
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

QCOM_TG = json.dumps({
  "openpilot.selfdrive.modeld.modeld": {
    "default": {"WARP_DEV": "QCOM", "QUEUE_DEV": "QCOM"},
    "usbgpu": {"WARP_DEV": "QCOM", "QUEUE_DEV": "AMD"},
  },
  "openpilot.selfdrive.modeld.dmonitoringmodeld": {"default": {"DEV": "QCOM"}},
})
CPU_TG = json.dumps({
  "openpilot.selfdrive.modeld.modeld": {
    "default": {"WARP_DEV": "CPU", "QUEUE_DEV": "CPU"},
    "usbgpu": {"WARP_DEV": "CPU", "QUEUE_DEV": "AMD"},
  },
  "openpilot.selfdrive.modeld.dmonitoringmodeld": {"default": {"DEV": "CPU"}},
})


def _launcher_function(name: str) -> str:
  source = (Path(BASEDIR) / "launch_chffrplus.sh").read_text(encoding="utf-8")
  start = source.index(f"function {name} {{")
  end = source.index("\n}", start) + 2
  return source[start:end]


def _write_model_tree(root: Path, *, prebuilt: bool, stamp: str | None = None,
                      artifacts: bool = True, tg: str | None = None) -> Path:
  models = root / "openpilot" / "selfdrive" / "modeld" / "models"
  models.mkdir(parents=True, exist_ok=True)
  scripts = root / "scripts"
  scripts.mkdir(parents=True, exist_ok=True)
  shutil.copy2(Path(BASEDIR) / "scripts" / "carrot_model_backend_check.py", scripts / "carrot_model_backend_check.py")
  if prebuilt:
    (root / "prebuilt").write_text("", encoding="utf-8")
  if stamp is not None:
    (models / ".build_stamp").write_text(stamp, encoding="utf-8")
  if artifacts:
    if tg is not None:
      (models / "tg_input_devices.json").write_text(tg, encoding="utf-8")
    (models / "driving_tinygrad.pkl").write_text("pkl", encoding="utf-8")
    (models / "dmonitoring_model_tinygrad.pkl").write_text("pkl", encoding="utf-8")
    (models / "dmonitoring_model_metadata.pkl").write_text("pkl", encoding="utf-8")
  return models


def _run_model_stamp(root: Path, *, big_model_sha: str = "", big_ready: bool = False,
                     tici_marker: Path | None = None, artifact_ok: bool = False) -> str:
  script = (f"DIR={shlex.quote(root.as_posix())}\n"
            f"BIG_MODEL_SHA={shlex.quote(big_model_sha)}\n"
            "FORCE_REBUILD=0\n"
            "MODEL_REBUILD=0\n"
            f"ARTIFACT_OK={1 if artifact_ok else 0}\n"
            "git() { echo stamp; }\n"
            f"big_model_artifact_ready() {{ return {0 if big_ready else 1}; }}\n"
            + _launcher_function("carrot_model_backend_guard") + "\n"
            + _launcher_function("invalidate_modeld_build_if_needed") +
            "\ninvalidate_modeld_build_if_needed\necho \"$FORCE_REBUILD:$MODEL_REBUILD\"\n")
  env = dict(os.environ)
  env["CARROT_TICI_MARKER"] = (tici_marker or (root / "no-tici-marker")).as_posix()
  result = subprocess.run([_bash(), "-c", script], capture_output=True, text=True, timeout=60, check=False, env=env)
  assert result.returncode == 0, result.stdout + result.stderr
  return result.stdout.strip().splitlines()[-1]


# (prebuilt, artifact_ok, stamp, tg, artifacts, on_device, expected FORCE_REBUILD:MODEL_REBUILD)
MODEL_STAMP_CASES = [
  # Source checkout (ARTIFACT_OK=0, no legacy prebuilt marker): unchanged.
  (False, False, "stamp:", QCOM_TG, True, False, "0:0"),
  (False, False, None, QCOM_TG, True, False, "1:0"),
  (False, False, "stamp:", None, False, False, "1:0"),
  (False, False, "outdated:", None, True, False, "1:0"),
  # Legacy on-device prebuilt: trust the shipped artifacts, but never a CPU backend.
  (True, False, None, QCOM_TG, True, False, "0:0"),
  (True, False, "outdated:", QCOM_TG, True, False, "0:0"),
  (True, False, "stamp:", QCOM_TG, True, True, "0:0"),
  (True, False, None, None, True, False, "1:0"),
  (True, False, "stamp:", CPU_TG, True, True, "1:0"),
  (True, False, None, CPU_TG, True, True, "1:0"),
  # Installed CI build artifact (ARTIFACT_OK=1): a missing or stale model tree
  # schedules only the device model build, never a full SCons rebuild.
  (False, True, None, None, False, False, "0:1"),
  (False, True, None, QCOM_TG, True, False, "0:1"),
  (False, True, "stamp:", QCOM_TG, True, False, "0:0"),
  (False, True, "stamp:", QCOM_TG, True, True, "0:0"),
  (False, True, "outdated:", QCOM_TG, True, True, "0:1"),
  (False, True, "stamp:", CPU_TG, True, True, "0:1"),
  (False, True, None, QCOM_TG, True, True, "0:1"),
]


@pytest.mark.parametrize("prebuilt, artifact_ok, stamp, tg, artifacts, on_device, expected", MODEL_STAMP_CASES)
def test_model_stamp_decision(tmp_path: Path, prebuilt: bool, artifact_ok: bool, stamp: str | None, tg: str | None,
                              artifacts: bool, on_device: bool, expected: str) -> None:
  marker = tmp_path / "TICI"
  if on_device:
    marker.write_text("", encoding="utf-8")
  _write_model_tree(tmp_path, prebuilt=prebuilt, stamp=stamp, artifacts=artifacts, tg=tg)

  assert _run_model_stamp(tmp_path, tici_marker=marker, artifact_ok=artifact_ok) == expected


def test_prebuilt_stamp_refresh_behaviour(tmp_path: Path) -> None:
  # A legacy prebuilt tree with intact artifacts refreshes the self-referential
  # stamp; a source checkout never rewrites it as part of invalidation.
  models = _write_model_tree(tmp_path, prebuilt=True, stamp="outdated:", artifacts=True, tg=QCOM_TG)
  assert _run_model_stamp(tmp_path, tici_marker=tmp_path / "no-tici-marker") == "0:0"
  assert (models / ".build_stamp").read_text(encoding="utf-8") == "stamp:"

  source_root = tmp_path / "source"
  source_models = _write_model_tree(source_root, prebuilt=False, stamp="outdated:", artifacts=True, tg=QCOM_TG)
  assert _run_model_stamp(source_root, tici_marker=tmp_path / "no-tici-marker") == "1:0"
  assert (source_models / ".build_stamp").read_text(encoding="utf-8") == "outdated:"


@pytest.mark.parametrize("big_stamp, ready, artifact_ok, expected", [
  ("model-sha", True, False, "0:0"),
  ("model-sha", False, False, "1:0"),
  (None, True, False, "1:0"),
  # With an installed artifact a stale eGPU model stays a model-only rebuild.
  ("model-sha", True, True, "0:0"),
  ("model-sha", False, True, "0:1"),
])
def test_model_stamp_keeps_egpu_invalidation(tmp_path: Path, big_stamp: str | None, ready: bool,
                                             artifact_ok: bool, expected: str) -> None:
  models = _write_model_tree(tmp_path, prebuilt=not artifact_ok, stamp="stamp:", artifacts=True, tg=QCOM_TG)
  if big_stamp is not None:
    (models / ".big_model_build_stamp").write_text(big_stamp, encoding="utf-8")

  assert _run_model_stamp(tmp_path, big_model_sha="model-sha", big_ready=ready,
                          tici_marker=tmp_path / "no-tici-marker", artifact_ok=artifact_ok) == expected


def _build_harness_functions() -> str:
  return "\n".join(_launcher_function(name) for name in (
    "carrot_boot_stamp_path", "carrot_drop_fastpath_stamp", "run_full_build", "run_model_build", "carrot_build_if_needed",
  ))


def _build_harness(tmp_path: Path, *, startup_rc: int, phases: str) -> subprocess.CompletedProcess:
  script = (f"DIR={shlex.quote(tmp_path.as_posix())}\n"
            f"export CARROT_BOOT_STAMP_PATH={shlex.quote((tmp_path / 'fastpath.stamp').as_posix())}\n"
            "MODEL_BUILD_STAMP_VALUE=stamp:value\n"
            "BIG_MODEL_SHA=\n"
            "big_model_artifact_ready() { return 1; }\n"
            f"run_startup_command() {{ printf '%s\\n' \"$*\" >> \"$LOG\"; return {startup_rc}; }}\n"
            "show_startup_failure() { echo \"SHOW: $*\"; exit 42; }\n"
            + _build_harness_functions() + "\n" + phases)
  return subprocess.run([_bash(), "-c", script], capture_output=True, text=True, timeout=60, check=False)


def test_model_build_runs_targets_and_writes_stamp(tmp_path: Path) -> None:
  models = _write_model_tree(tmp_path, prebuilt=False, stamp=None, artifacts=False)
  log = tmp_path / "calls.log"
  result = _build_harness(tmp_path, startup_rc=0, phases=f"LOG={shlex.quote(log.as_posix())}\n"
                          "FORCE_REBUILD=0\nMODEL_REBUILD=1\nARTIFACT_OK=1\nfast_boot=0\n"
                          "carrot_build_if_needed\n")

  assert result.returncode == 0, result.stdout + result.stderr
  assert log.read_text(encoding="utf-8").splitlines() == ["./build.py --targets openpilot/selfdrive/modeld/models"]
  assert (models / ".build_stamp").read_text(encoding="utf-8") == "stamp:value"


def test_model_build_failure_drops_stamp_and_shows_failure(tmp_path: Path) -> None:
  models = _write_model_tree(tmp_path, prebuilt=False, stamp=None, artifacts=False)
  stamp_file = tmp_path / "fastpath.stamp"
  stamp_file.write_text("fingerprint", encoding="utf-8")
  log = tmp_path / "calls.log"
  result = _build_harness(tmp_path, startup_rc=1, phases=f"LOG={shlex.quote(log.as_posix())}\n"
                          "FORCE_REBUILD=0\nMODEL_REBUILD=1\nARTIFACT_OK=1\nfast_boot=0\n"
                          "carrot_build_if_needed\n")

  assert result.returncode == 42, result.stdout + result.stderr
  assert "SHOW: model build failed" in result.stdout
  assert not stamp_file.exists()
  assert not (models / ".build_stamp").exists()


def test_build_dispatch_matrix(tmp_path: Path) -> None:
  _write_model_tree(tmp_path, prebuilt=False, stamp=None, artifacts=False)
  log_force = tmp_path / "force.log"
  log_model = tmp_path / "model.log"
  log_artifact_idle = tmp_path / "artifact_idle.log"
  log_source_full = tmp_path / "source_full.log"
  log_source_model = tmp_path / "source_model.log"
  log_fast = tmp_path / "fast.log"
  result = _build_harness(tmp_path, startup_rc=0, phases=(
    f"LOG={shlex.quote(log_force.as_posix())}\nFORCE_REBUILD=1\nMODEL_REBUILD=0\nARTIFACT_OK=1\nfast_boot=0\ncarrot_build_if_needed\n"
    f"LOG={shlex.quote(log_model.as_posix())}\nFORCE_REBUILD=0\nMODEL_REBUILD=1\nARTIFACT_OK=1\nfast_boot=0\ncarrot_build_if_needed\n"
    f"LOG={shlex.quote(log_artifact_idle.as_posix())}\nFORCE_REBUILD=0\nMODEL_REBUILD=0\nARTIFACT_OK=1\nfast_boot=0\ncarrot_build_if_needed\n"
    f"LOG={shlex.quote(log_source_full.as_posix())}\nFORCE_REBUILD=0\nMODEL_REBUILD=0\nARTIFACT_OK=0\nfast_boot=0\ncarrot_build_if_needed\n"
    f"LOG={shlex.quote(log_source_model.as_posix())}\nFORCE_REBUILD=0\nMODEL_REBUILD=1\nARTIFACT_OK=0\nfast_boot=0\ncarrot_build_if_needed\n"
    f"LOG={shlex.quote(log_fast.as_posix())}\nFORCE_REBUILD=1\nMODEL_REBUILD=1\nARTIFACT_OK=1\nfast_boot=1\ncarrot_build_if_needed\n"
  ))

  assert result.returncode == 0, result.stdout + result.stderr
  assert log_force.read_text(encoding="utf-8").splitlines() == ["./build.py"]
  assert log_model.read_text(encoding="utf-8").splitlines() == ["./build.py --targets openpilot/selfdrive/modeld/models"]
  assert not log_artifact_idle.exists()
  assert log_source_full.read_text(encoding="utf-8").splitlines() == ["./build.py"]
  assert log_source_model.read_text(encoding="utf-8").splitlines() == ["./build.py"]
  assert not log_fast.exists()
  assert (tmp_path / "openpilot/selfdrive/modeld/models/.build_stamp").read_text(encoding="utf-8") == "stamp:value"


def test_legacy_prebuilt_tree_still_skips_the_build(tmp_path: Path) -> None:
  _write_model_tree(tmp_path, prebuilt=True, stamp="stamp:", artifacts=True, tg=QCOM_TG)
  log = tmp_path / "calls.log"
  result = _build_harness(tmp_path, startup_rc=0, phases=f"LOG={shlex.quote(log.as_posix())}\n"
                          "FORCE_REBUILD=0\nMODEL_REBUILD=0\nARTIFACT_OK=0\nfast_boot=0\n"
                          "carrot_build_if_needed\n")

  assert result.returncode == 0, result.stdout + result.stderr
  assert not log.exists()


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
                             python_rc: str = "0", scons_rc: str = "0", artifact: bool = False):
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
  if artifact:
    env["CARROT_BUILD_ARTIFACT"] = "1"
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


def test_artifact_params_module_is_trusted_and_stamped(tmp_path: Path) -> None:
  # An installed CI build artifact (CARROT_BUILD_ARTIFACT=1) gets the same trust
  # as a legacy on-device prebuilt tree: record the stamp, no SCons.
  result, root, cache, sim = _run_ensure_params_build(tmp_path, prebuilt=False, module=True, stamp=None, artifact=True)

  assert result.returncode == 0, result.stdout + result.stderr
  assert not (sim / "scons.log").exists()
  assert (root / "openpilot" / "common" / "params_pyx.so").exists()
  header_hash = hashlib.sha256((root / "openpilot" / "common" / "params_keys.h").read_bytes()).hexdigest()
  assert (cache / "carrot_params_keys.sha256").read_text(encoding="utf-8").strip() == header_hash
  assert "params_check.py" in (sim / "python.log").read_text(encoding="utf-8")


def test_artifact_params_check_failure_falls_back_to_rebuild(tmp_path: Path) -> None:
  result, root, cache, sim = _run_ensure_params_build(tmp_path, prebuilt=False, module=True, stamp=None,
                                                      python_rc="1", artifact=True)

  assert result.returncode != 0
  assert "-u -j4 openpilot/common/params_pyx.so" in (sim / "scons.log").read_text(encoding="utf-8")
  assert not (root / "openpilot" / "common" / "params_pyx.so").exists()
  assert not (cache / "carrot_params_keys.sha256").exists()


def test_params_module_without_any_marker_still_rebuilds(tmp_path: Path) -> None:
  # Without the prebuilt marker or the artifact env the source behavior stands.
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
