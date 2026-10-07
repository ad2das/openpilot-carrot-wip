#!/usr/bin/env python3
"""Cheap warm-boot fingerprint for launch_chffrplus.sh.

Prints one line: "<sha256> ready=0|1".

The launcher stores this line in /data/carrot_boot_fastpath.stamp after a full
successful boot and skips dependency bootstrap, SCons and the Params checks on
the next boot when it matches exactly. Keep this script cheap: it is the only
python process the fast-path decision may start.
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

SEP = b"\x1e"


def _git(root: Path, *args: str) -> str:
  try:
    proc = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, timeout=30)
    return proc.stdout.strip()
  except Exception as exc:
    return f"<error:{type(exc).__name__}>"


def _file_state(path) -> str:
  try:
    stat = os.stat(path)
    return f"{stat.st_size}:{stat.st_mtime_ns}"
  except OSError:
    return "missing"


def _sha256_file(path) -> str:
  try:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
      for block in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()
  except OSError:
    return "missing"


def _pydeps_state(pydeps: Path) -> str:
  try:
    entries = sorted((entry.name, entry.stat().st_mtime_ns) for entry in os.scandir(pydeps))
  except OSError:
    return "missing"
  return ";".join(f"{name}:{mtime}" for name, mtime in entries)


def _model_probe(root: Path, pydeps: Path, parts: list):
  for path in (str(pydeps), str(root)):
    if path not in sys.path:
      sys.path.insert(0, path)
  try:
    from openpilot.selfdrive.modeld.helpers import active_usbgpu_compiled_path, usbgpu_present
    present = usbgpu_present()
    parts.append(("egpu_present", "1" if present else "0"))
    if present:
      from openpilot.selfdrive.modeld.big_model import active_manifest
      manifest = active_manifest()
      parts.append(("big_model_sha", manifest.sha256 if manifest is not None else ""))
      parts.append(("big_model_ready", "1" if active_usbgpu_compiled_path() is not None else "0"))
    else:
      parts.append(("big_model_sha", ""))
      parts.append(("big_model_ready", "0"))
  except Exception as exc:
    # Import failures stay in the digest so a repaired tree forces one full
    # boot; they must not block the fast path on their own.
    parts.append(("model_probe", f"error:{type(exc).__name__}"))


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", default=os.getcwd())
  parser.add_argument("--pydeps", default=None)
  parser.add_argument("--version-file", default=os.environ.get("CARROT_VERSION_FILE", "/VERSION"))
  args = parser.parse_args()

  root = Path(args.root).resolve()
  pydeps = Path(args.pydeps).resolve() if args.pydeps else root / "pydeps"
  models = root / "openpilot" / "selfdrive" / "modeld" / "models"

  parts: list[tuple[str, str]] = []
  parts.append(("head", _git(root, "rev-parse", "HEAD")))
  parts.append(("status", _git(root, "status", "--porcelain=v1", "--untracked-files=no")))
  parts.append(("submodules", _git(root, "submodule", "status")))

  try:
    version = Path(args.version_file).read_text(encoding="utf-8", errors="replace").strip()
  except OSError:
    version = "missing"
  parts.append(("version", version))
  parts.append(("agnos_version", os.environ.get("AGNOS_VERSION", "")))
  parts.append(("python", f"{sys.executable}:{sys.version.split()[0]}"))
  parts.append(("pydeps", _pydeps_state(pydeps)))
  parts.append(("params_keys", _sha256_file(root / "openpilot" / "common" / "params_keys.h")))
  _model_probe(root, pydeps, parts)

  params_pyx = root / "openpilot" / "common" / "params_pyx.so"
  loggerd = root / "openpilot" / "system" / "loggerd" / "loggerd"
  encoderd = root / "openpilot" / "system" / "loggerd" / "encoderd"
  camerad = root / "openpilot" / "system" / "camerad" / "camerad"
  driving_pkl = models / "driving_tinygrad.pkl"
  driving_manifest = Path(str(driving_pkl) + ".chunkmanifest")
  tg_input_devices = models / "tg_input_devices.json"

  parts.append(("params_pyx", _file_state(params_pyx)))
  parts.append(("loggerd", _file_state(loggerd)))
  parts.append(("encoderd", _file_state(encoderd)))
  parts.append(("camerad", _file_state(camerad)))
  parts.append(("driving_pkl", f"{_file_state(driving_pkl)}|{_file_state(driving_manifest)}"))
  parts.append(("tg_input_devices", _file_state(tg_input_devices)))
  parts.append(("prebuilt", _file_state(root / "prebuilt")))

  ready = all((
    params_pyx.is_file(),
    loggerd.is_file(),
    encoderd.is_file(),
    camerad.is_file(),
    driving_pkl.is_file() or driving_manifest.is_file(),
    tg_input_devices.is_file(),
  ))

  digest = hashlib.sha256()
  for label, value in parts:
    digest.update(label.encode("utf-8", "surrogateescape"))
    digest.update(b"=")
    digest.update(value.encode("utf-8", "surrogateescape"))
    digest.update(SEP)

  print(f"{digest.hexdigest()} ready={int(ready)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
