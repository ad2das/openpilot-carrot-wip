import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

from openpilot.common.basedir import BASEDIR

BUILD_PY = Path(BASEDIR) / "openpilot" / "system" / "manager" / "build.py"


def _load_build(tmp_path: Path, commands: list[list[str]]):
  # Execute only the build() function with the SCons boundary faked, so the
  # hardware imports of the real module do not apply.
  tree = ast.parse(BUILD_PY.read_text(encoding="utf-8"))
  body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"]

  class FakeStdout:
    def readline(self):
      return b""

    def read(self):
      return b""

  class FakePopen:
    def __init__(self, args, **kwargs):
      commands.append(list(args))
      self.returncode = 0
      self.stdout = FakeStdout()

    def poll(self):
      return 0

  cache = tmp_path / "scons_cache"
  cache.mkdir(exist_ok=True)
  namespace = {
    "os": os,
    "subprocess": SimpleNamespace(Popen=FakePopen, PIPE=-1, STDOUT=-2, DEVNULL=-3),
    "BASEDIR": str(tmp_path),
    "AGNOS": False,
    "HARDWARE": SimpleNamespace(set_power_save=lambda enabled: None),
    "CACHE_DIR": cache,
    "MAX_CACHE_SIZE": 4e9,
    "Spinner": object,
    "TOTAL_SCONS_NODES": 2705,
    "MAX_BUILD_PROGRESS": 100,
    "cloudlog": SimpleNamespace(error=lambda *a, **k: None),
    "add_file_handler": lambda *a, **k: None,
    "TextWindow": object,
  }
  exec(compile(ast.Module(body=body, type_ignores=[]), "<build>", "exec"), namespace)
  return namespace["build"]


def _spinner() -> SimpleNamespace:
  return SimpleNamespace(update=lambda *a: None, update_progress=lambda *a: None)


def test_build_passes_targets_after_existing_args(tmp_path: Path) -> None:
  commands: list[list[str]] = []
  build = _load_build(tmp_path, commands)

  build(_spinner(), targets=["openpilot/selfdrive/modeld/models"])

  nproc = os.cpu_count() or 2
  assert commands == [["scons", f"-j{nproc}", "--cache-populate", "openpilot/selfdrive/modeld/models"]]


def test_build_default_targets_unchanged(tmp_path: Path) -> None:
  commands: list[list[str]] = []
  build = _load_build(tmp_path, commands)

  build(_spinner(), dirty=False, minimal=True)

  nproc = os.cpu_count() or 2
  assert commands == [["scons", f"-j{nproc}", "--cache-populate", "--minimal"]]


def test_build_targets_follow_minimal_flag(tmp_path: Path) -> None:
  commands: list[list[str]] = []
  build = _load_build(tmp_path, commands)

  build(_spinner(), minimal=True, targets=["target-a", "target-b"])

  nproc = os.cpu_count() or 2
  assert commands == [["scons", f"-j{nproc}", "--cache-populate", "--minimal", "target-a", "target-b"]]


def test_usbgpu_model_build_still_runs_after_the_main_build() -> None:
  source = BUILD_PY.read_text(encoding="utf-8")
  assert source.index("build(spinner, build_metadata.openpilot.is_dirty") < source.index("build_usbgpu_model(spinner)")
