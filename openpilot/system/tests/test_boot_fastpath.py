import os
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


HARNESS = r'''#!/usr/bin/env bash

REAL_PY="${W53_REAL_PY:?}"
W53_ROOT="${W53_ROOT:?}"
export W53_REAL_PY W53_ROOT

source "$W53_ROOT/launch_chffrplus.sh"

DIR="$W53_ROOT"
export DIR
CARROT_DATA_DIR="$DIR/data"
export CARROT_DATA_DIR
PYDEPS="$DIR/pydeps"
export PYDEPS
export CARROT_BOOT_TIMING_LOG="$CARROT_DATA_DIR/carrot_boot_timing.log"
export CARROT_BOOT_STAMP_PATH="$CARROT_DATA_DIR/carrot_boot_fastpath.stamp"
export CARROT_FORCE_FULL_BOOT_FILE="$CARROT_DATA_DIR/carrot_force_full_boot"
export CARROT_VERSION_FILE="$DIR/VERSION"
LOG="$W53_ROOT/boot.log"
export LOG

# stubbed system helpers
python3() { "$REAL_PY" "$@"; }
flock() { return 0; }
ln() { :; }
cleanup_stale_git_lfs_hooks() { :; }
start_carrot_recovery() { :; }
start_carrot_web() { :; }
start_big_model_update() { :; }
tmux() { :; }
sleep() { exit "${W53_SLEEP_RC:-0}"; }

run_startup_command() {
  if carrot_fast_boot_skip_command "$@"; then
    return 0
  fi
  echo "RUN $*" >> "$LOG"
  case "$*" in
    *bootstrap_runtime_dependencies*) echo "DEPS" >> "$LOG"; boot_timing deps; return "${W53_DEPS_RC:-0}" ;;
    *build_time.py*) return "${W53_CLOCK_RC:-0}" ;;
    *ensure_params_build.sh*) boot_timing params_build; return "${W53_ENSURE_RC:-0}" ;;
    *./build.py*) boot_timing scons; return "${W53_BUILD_RC:-0}" ;;
    *params_check.py*) boot_timing params_check; return "${W53_PARAMS_RC:-0}" ;;
    *start_manager*) return "${W53_MANAGER_RC:-0}" ;;
    *) "$@"; return $? ;;
  esac
}

show_startup_failure() {
  echo "SHOW_FAILURE $*" >> "$LOG"
  exit "${W53_FAILURE_RC:-41}"
}

launch
'''


@pytest.fixture
def boot_tree(tmp_path: Path) -> Path:
  root = tmp_path / "repo"
  (root / "scripts").mkdir(parents=True)
  shutil.copy2(Path(BASEDIR) / "launch_chffrplus.sh", root / "launch_chffrplus.sh")
  shutil.copy2(Path(BASEDIR) / "launch_env.sh", root / "launch_env.sh")
  shutil.copy2(Path(BASEDIR) / "scripts" / "carrot_boot_fingerprint.py", root / "scripts" / "carrot_boot_fingerprint.py")
  shutil.copy2(Path(BASEDIR) / "scripts" / "ensure_params_build.sh", root / "scripts" / "ensure_params_build.sh")

  def write(rel: str, text: str = "x", mode: int = 0o644) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path

  write("openpilot/common/params_keys.h", '{"Foo", {PERSISTENT, BOOL}};\n')
  write("pydeps/dep1.txt")
  write("VERSION", "19.8-carrot-bt1\n")
  write("openpilot/common/params_pyx.so", "so", 0o755)
  write("openpilot/system/loggerd/loggerd", "#!/bin/sh\n", 0o755)
  write("openpilot/system/loggerd/encoderd", "#!/bin/sh\n", 0o755)
  write("openpilot/system/camerad/camerad", "#!/bin/sh\n", 0o755)
  write("openpilot/selfdrive/modeld/models/driving_tinygrad.pkl", "pkl")
  write("openpilot/selfdrive/modeld/models/tg_input_devices.json", '{"openpilot.selfdrive.modeld.modeld": {}}')
  write("tracked.txt", "original\n")
  (root / "openpilot" / "system" / "manager").mkdir(parents=True, exist_ok=True)
  (root / "openpilot" / "selfdrive" / "pandad").mkdir(parents=True, exist_ok=True)
  (root / "data" / "params" / "d").mkdir(parents=True, exist_ok=True)

  subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True, capture_output=True)
  subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                  "add", "-A"], check=True, capture_output=True)
  subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                  "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"], check=True, capture_output=True)

  harness = tmp_path / "harness.sh"
  harness.write_text(HARNESS, encoding="utf-8", newline="\n")
  return root


def run_boot(root: Path, **env_extra) -> str:
  log = root / "boot.log"
  if log.exists():
    log.unlink()
  env = dict(os.environ)
  env.update({
    "W53_ROOT": root.as_posix(),
    "W53_REAL_PY": Path(__import__("sys").executable).as_posix(),
  })
  env.update({key: str(value) for key, value in env_extra.items()})
  result = subprocess.run([_bash(), str(root.parent / "harness.sh")], cwd=str(root), env=env,
                          capture_output=True, text=True, timeout=300, check=False)
  assert result.stdout is not None
  assert log.exists(), f"launcher did not run: {result.stdout}\n{result.stderr}"
  return log.read_text(encoding="utf-8")


def stamp_path(root: Path) -> Path:
  return root / "data" / "carrot_boot_fastpath.stamp"


def timing(root: Path) -> str:
  path = root / "data" / "carrot_boot_timing.log"
  return path.read_text(encoding="utf-8") if path.exists() else ""


def assert_full(root: Path, log: str, reason: str) -> None:
  assert f"fastpath=0 reason={reason}" in timing(root), timing(root)
  assert "DEPS" in log
  assert "RUN bash" in log and "ensure_params_build.sh" in log
  assert "RUN python3" in log and "params_check.py" in log


def assert_fast(root: Path, log: str) -> None:
  assert "fastpath=1" in timing(root), timing(root)
  assert "DEPS" not in log
  assert "ensure_params_build.sh" not in log
  assert "params_check.py" not in log
  assert "./build.py" not in log


def test_first_boot_full_then_second_boot_fast_without_build(boot_tree: Path) -> None:
  log = run_boot(boot_tree)
  assert_full(boot_tree, log, "stamp-missing")
  assert stamp_path(boot_tree).is_file()
  assert "boot_start uptime=" in timing(boot_tree)
  assert "manager_exec" in timing(boot_tree)

  log = run_boot(boot_tree)
  assert_fast(boot_tree, log)
  assert "RUN start_manager" in log


def commit_all(root: Path, message: str) -> None:
  subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                  "add", "-A"], check=True, capture_output=True)
  subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                  "-c", "commit.gpgsign=false", "commit", "-q", "-m", message], check=True, capture_output=True)


@pytest.mark.parametrize("mutation, reason", [
  ("head", "fingerprint-mismatch"),
  ("tracked", "fingerprint-mismatch"),
  ("version", "fingerprint-mismatch"),
  ("pydeps", "fingerprint-mismatch"),
  ("loggerd", "fingerprint-mismatch"),
])
def test_checkout_changes_force_full_boot(boot_tree: Path, mutation: str, reason: str) -> None:
  run_boot(boot_tree)
  assert stamp_path(boot_tree).is_file()

  if mutation == "head":
    commit_all(boot_tree, "new head")
  elif mutation == "tracked":
    (boot_tree / "tracked.txt").write_text("edited\n", encoding="utf-8")
  elif mutation == "version":
    (boot_tree / "VERSION").write_text("19.9-next\n", encoding="utf-8")
  elif mutation == "pydeps":
    (boot_tree / "pydeps" / "dep2.txt").write_text("new", encoding="utf-8")
  elif mutation == "loggerd":
    (boot_tree / "openpilot" / "system" / "loggerd" / "loggerd").unlink()

  log = run_boot(boot_tree)
  assert f"fastpath=0 reason={reason}" in timing(boot_tree), timing(boot_tree)
  assert "DEPS" in log


def test_force_file_and_env_variable_force_full_boot(boot_tree: Path) -> None:
  run_boot(boot_tree)
  force_file = boot_tree / "data" / "carrot_force_full_boot"
  force_file.write_text("", encoding="utf-8")
  run_boot(boot_tree)
  assert "fastpath=0 reason=force-file" in timing(boot_tree), timing(boot_tree)
  assert stamp_path(boot_tree).is_file(), "force file must not delete the stamp"
  assert not force_file.exists(), "a successful full boot consumes the force file"

  run_boot(boot_tree, CARROT_FULL_BOOT="1")
  assert "fastpath=0 reason=CARROT_FULL_BOOT=1" in timing(boot_tree), timing(boot_tree)


def test_fast_path_manager_failure_deletes_stamp(boot_tree: Path) -> None:
  run_boot(boot_tree)
  assert stamp_path(boot_tree).is_file()

  log = run_boot(boot_tree, W53_MANAGER_RC="1")
  assert "fastpath=1" in timing(boot_tree), timing(boot_tree)
  assert "SHOW_FAILURE Manager failed to start" in log
  assert not stamp_path(boot_tree).exists()


def test_full_path_failure_leaves_no_stamp(boot_tree: Path) -> None:
  log = run_boot(boot_tree, W53_ENSURE_RC="1")
  assert "SHOW_FAILURE Params registry build failed" in log
  assert not stamp_path(boot_tree).exists()


def test_full_path_removes_force_file_after_success(boot_tree: Path) -> None:
  (boot_tree / "data" / "carrot_force_full_boot").write_text("", encoding="utf-8")
  run_boot(boot_tree)
  assert not (boot_tree / "data" / "carrot_force_full_boot").exists()
  assert stamp_path(boot_tree).is_file()
