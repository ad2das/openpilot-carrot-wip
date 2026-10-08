#!/usr/bin/env python3
"""Device-side fetch/install/check for CI build artifacts.

CI (`.github/workflows/carrot-wip-build-artifacts.yaml`) builds every
carrot-wip commit for the comma 3/3X and publishes two assets per commit to the
rolling `carrot-wip-builds` GitHub Release:

  <sha>.json    manifest: source commit, build environment, file hashes
  <sha>.tar.xz  the untracked build outputs plus the allowlisted regenerated
                tracked files (never model artifacts, pydeps or intermediates)

This tool runs on AGNOS and uses only the standard library, so the device can
fetch the artifact for its update target before rebooting and install it at
boot. After installation the checkout is equivalent to a successful on-device
build of the same commit, except for the model artifacts (which the device
compiles itself) and build intermediates (.o/.os/.a, .sconsign.dblite, the
SCons cache).

Exit codes:
  fetch    0 ready, 3 not published (404), 1 error
  install  0 installed, 2 rejected (reason printed), 1 error
  check    0 installed and current, 1 anything else
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://github.com/ad2das/openpilot-carrot-wip/releases/download/carrot-wip-builds"
DEFAULT_DATA_ROOT = "/data"
DEFAULT_TIMEOUT = 120.0
PROBE_TIMEOUT = 5.0
KEEP_CACHE_ENTRIES = 3  # the installed commit plus at most two others
USER_AGENT = "carrot-build-artifact/1"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MODES = {"0755", "0644"}

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REJECTED = 2
EXIT_NOT_PUBLISHED = 3


def log(message: str) -> None:
  print(f"[carrot_build_artifact] {message}", flush=True)


def base_url() -> str:
  return (os.environ.get("CARROT_BUILD_ARTIFACT_URL") or DEFAULT_BASE_URL).rstrip("/")


def data_root(explicit: str | None) -> str:
  return explicit or os.environ.get("CARROT_BUILD_ARTIFACT_DATA_ROOT") or DEFAULT_DATA_ROOT


def cache_dir(root: str, sha: str) -> str:
  return os.path.join(root, "carrot_build_cache", sha)


def state_dir(root: str) -> str:
  return os.path.join(root, "carrot_build_artifacts")


def installed_manifest_path(root: str) -> str:
  return os.path.join(state_dir(root), "installed.json")


def host_arch() -> str:
  machine = platform.machine().lower()
  if machine in ("aarch64", "arm64"):
    return "larch64"
  if machine in ("x86_64", "amd64"):
    return "x86_64"
  return machine


def git(repo: str, *args: str) -> tuple[int, str]:
  try:
    result = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=30)
  except (OSError, subprocess.SubprocessError) as exc:
    return 1, str(exc)
  return result.returncode, result.stdout


def read_json(path: str):
  with open(path, "rb") as f:
    return json.load(f)


def write_json_atomic(path: str, data) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.tmp.{os.getpid()}"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, sort_keys=True)
    f.write("\n")
  os.replace(tmp, path)


def sha256_file(path: str) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def http_get(url: str, timeout: float) -> bytes:
  request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
  with urllib.request.urlopen(request, timeout=timeout) as response:
    return response.read()


def fetch(sha: str, timeout: float, root: str) -> int:
  if not SHA_RE.match(sha):
    log(f"not a commit sha: {sha!r}")
    return EXIT_ERROR

  destination = cache_dir(root, sha)
  os.makedirs(destination, exist_ok=True)
  manifest_url = f"{base_url()}/{sha}.json"
  tarball_url = f"{base_url()}/{sha}.tar.xz"

  # A short probe first: a missing network or a 404 must skip the whole fetch
  # instead of holding the boot for the full download timeout.
  probe_timeout = min(PROBE_TIMEOUT, timeout)
  try:
    manifest_bytes = http_get(manifest_url, probe_timeout)
  except urllib.error.HTTPError as exc:
    if exc.code == 404:
      log(f"artifact for {sha[:12]} is not published (404)")
      return EXIT_NOT_PUBLISHED
    log(f"manifest request failed: HTTP {exc.code}")
    return EXIT_ERROR
  except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
    log(f"manifest request failed: {exc}")
    return EXIT_ERROR

  try:
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    tarball_size = int(manifest["tarball_size"])
    tarball_sha256 = str(manifest["tarball_sha256"])
    assert manifest["source_commit"] == sha
  except Exception as exc:
    log(f"invalid manifest for {sha[:12]}: {exc}")
    return EXIT_ERROR

  try:
    free = shutil.disk_usage(destination if os.path.isdir(destination) else root).free
  except OSError as exc:
    log(f"cannot determine free space: {exc}")
    return EXIT_ERROR
  needed = 3 * tarball_size
  if free < needed:
    log(f"not enough free space for {sha[:12]}: need {needed} bytes, have {free}")
    return EXIT_ERROR

  manifest_tmp = os.path.join(destination, f"{sha}.json.tmp")
  with open(manifest_tmp, "wb") as f:
    f.write(manifest_bytes)
  os.replace(manifest_tmp, os.path.join(destination, f"{sha}.json"))

  tarball_path = os.path.join(destination, f"{sha}.tar.xz")
  tmp_path = os.path.join(destination, f"{sha}.tar.xz.tmp")
  request = urllib.request.Request(tarball_url, headers={"User-Agent": USER_AGENT})
  written = 0
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp_path, "wb") as f:
      while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
          break
        written += len(chunk)
        if written > tarball_size:
          raise OSError(f"download exceeds the manifest size ({tarball_size} bytes)")
        f.write(chunk)
  except urllib.error.HTTPError as exc:
    _remove_quietly(tmp_path)
    log(f"tarball request failed: HTTP {exc.code}")
    return EXIT_ERROR
  except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
    _remove_quietly(tmp_path)
    log(f"tarball download failed: {exc}")
    return EXIT_ERROR

  if written != tarball_size:
    _remove_quietly(tmp_path)
    log(f"tarball size mismatch: got {written}, expected {tarball_size}")
    return EXIT_ERROR
  actual = sha256_file(tmp_path)
  if actual != tarball_sha256:
    _remove_quietly(tmp_path)
    log(f"tarball sha256 mismatch: got {actual}, expected {tarball_sha256}")
    return EXIT_ERROR

  os.replace(tmp_path, tarball_path)
  prune_cache(root, sha)
  log(f"artifact {sha[:12]} ready: {written} bytes")
  return EXIT_OK


def prune_cache(root: str, current_sha: str) -> None:
  cache_root = os.path.join(root, "carrot_build_cache")
  try:
    entries = [e for e in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, e))]
  except OSError:
    return

  installed_sha = ""
  try:
    installed_sha = str(read_json(installed_manifest_path(root)).get("sha") or "")
  except Exception:
    pass

  keep = {current_sha}
  if installed_sha and installed_sha in entries:
    keep.add(installed_sha)

  candidates = []
  for entry in entries:
    if entry in keep:
      continue
    path = os.path.join(cache_root, entry)
    try:
      mtime = os.path.getmtime(path)
    except OSError:
      mtime = 0.0
    candidates.append((mtime, entry))
  candidates.sort(reverse=True)

  for _, entry in candidates[: max(0, KEEP_CACHE_ENTRIES - len(keep))]:
    keep.add(entry)

  for entry in entries:
    if entry not in keep:
      shutil.rmtree(os.path.join(cache_root, entry), ignore_errors=True)


def _remove_quietly(path: str) -> None:
  try:
    os.remove(path)
  except OSError:
    pass


def load_manifest(sha: str, root: str) -> tuple[dict | None, str]:
  path = os.path.join(cache_dir(root, sha), f"{sha}.json")
  if not os.path.isfile(path):
    return None, f"manifest for {sha[:12]} is not cached"
  try:
    manifest = read_json(path)
  except Exception as exc:
    return None, f"manifest for {sha[:12]} does not parse: {exc}"
  if manifest.get("format") != 1:
    return None, f"unsupported artifact format {manifest.get('format')!r}"
  if manifest.get("source_commit") != sha:
    return None, f"manifest source_commit does not match {sha[:12]}"
  return manifest, ""


def safe_member_path(name: str) -> str | None:
  """Return the normalized relative path, or None when it is unsafe."""
  if not name or "\\" in name or name.startswith("/") or name.startswith("../"):
    return None
  if re.match(r"^[A-Za-z]:", name):
    return None
  parts = name.split("/")
  if any(part in ("", ".", "..") for part in parts):
    return None
  return "/".join(parts)


def tracked_files(repo: str) -> set[str]:
  rc, out = git(repo, "ls-files", "-z")
  if rc != 0:
    return set()
  return {p for p in out.split("\0") if p}


def tracked_modifications(repo: str) -> list[str]:
  """Tracked paths that differ from HEAD (staged or unstaged, renames included)."""
  rc, out = git(repo, "status", "--porcelain", "--untracked-files=no", "-z")
  if rc != 0:
    return []
  tokens = out.split("\0")
  paths: list[str] = []
  i = 0
  while i < len(tokens):
    entry = tokens[i]
    if len(entry) < 4:
      i += 1
      continue
    status, path = entry[:2], entry[3:]
    paths.append(path)
    if status[0] in ("R", "C") and i + 1 < len(tokens):
      paths.append(tokens[i + 1])
      i += 1
    i += 1
  return paths


def allowlist_paths(repo: str) -> set[str]:
  path = os.path.join(repo, "scripts", "build_artifact_tracked_outputs.txt")
  try:
    with open(path, encoding="utf-8") as f:
      lines = f.readlines()
  except OSError:
    return set()
  entries = set()
  for line in lines:
    entry = line.split("#", 1)[0].strip()
    if entry:
      entries.add(entry)
  return entries


def check(repo: str, root: str) -> int:
  try:
    installed = read_json(installed_manifest_path(root))
  except Exception:
    return 1
  sha = str(installed.get("sha") or "")
  if not SHA_RE.match(sha):
    return 1
  rc, head = git(repo, "rev-parse", "HEAD")
  if rc != 0 or head.strip() != sha:
    return 1
  for rel, info in (installed.get("files") or {}).items():
    full = os.path.join(repo, rel)
    try:
      if not os.path.isfile(full) or os.path.getsize(full) != int(info.get("size", -1)):
        return 1
    except OSError:
      return 1
  return 0


def install(sha: str, repo: str, root: str, tici_marker: str, ion_path: str) -> int:
  manifest, reason = load_manifest(sha, root)
  if manifest is None:
    return _reject(reason)

  tarball = os.path.join(cache_dir(root, sha), f"{sha}.tar.xz")
  if not os.path.isfile(tarball):
    return _reject(f"tarball for {sha[:12]} is not cached")

  rc, head = git(repo, "rev-parse", "HEAD")
  if rc != 0:
    return _reject(f"cannot read HEAD of {repo}")
  if head.strip() != sha:
    return _reject(f"checkout is at {head.strip()[:12]}, not {sha[:12]}")

  if os.path.exists(tici_marker) != bool(manifest.get("tici")):
    return _reject(f"{tici_marker} presence does not match manifest tici={manifest.get('tici')}")
  if os.path.exists(ion_path) != bool(manifest.get("ion")):
    return _reject(f"{ion_path} presence does not match manifest ion={manifest.get('ion')}")

  version = os.environ.get("AGNOS_VERSION")
  if version != manifest.get("agnos_version"):
    return _reject(f"AGNOS_VERSION {version!r} does not match manifest {manifest.get('agnos_version')!r}")

  if host_arch() != manifest.get("arch"):
    return _reject(f"host arch {host_arch()!r} does not match manifest {manifest.get('arch')!r}")

  files = manifest.get("files") or {}
  if not isinstance(files, dict) or not files:
    return _reject("manifest has no files")

  tracked = tracked_files(repo)
  if not tracked and git(repo, "rev-parse", "--git-dir")[0] != 0:
    return _reject(f"cannot enumerate tracked files of {repo}")

  regenerated = manifest.get("tracked_regenerated") or []
  if not isinstance(regenerated, list):
    return _reject("manifest tracked_regenerated is not a list")
  allowlist = allowlist_paths(repo)
  regenerated_set = set(regenerated)
  for rel in sorted(regenerated_set):
    if rel not in allowlist:
      return _reject(f"tracked_regenerated path is not allowlisted by this checkout: {rel}")

  for rel in tracked_modifications(repo):
    if rel not in regenerated_set:
      return _reject(f"tracked modification is not part of the artifact: {rel}")

  for rel, info in sorted(files.items()):
    if not isinstance(info, dict):
      return _reject(f"manifest entry for {rel} is not an object")
    mode = str(info.get("mode", ""))
    if mode not in MODES:
      return _reject(f"unsupported mode {mode!r} for {rel}")
    if rel in tracked and rel not in regenerated_set:
      return _reject(f"artifact would overwrite tracked file {rel}")
    if safe_member_path(rel) != rel:
      return _reject(f"unsafe artifact path {rel!r}")

  staging_root = tempfile.mkdtemp(prefix="carrot_artifact_", dir=state_dir(root) if os.path.isdir(state_dir(root)) else root)
  try:
    try:
      extracted = _extract_verified(tarball, files, staging_root)
    except _Rejected as exc:
      return _reject(str(exc))
    except Exception as exc:
      log(f"install error: {exc}")
      return EXIT_ERROR
    try:
      _apply(repo, staging_root, extracted, files, old_files=_installed_files(root), tracked=tracked)
    except _Rejected as exc:
      return _reject(str(exc))
    except Exception as exc:
      log(f"install error while applying files: {exc}")
      return EXIT_ERROR
    write_json_atomic(installed_manifest_path(root),
                      {"sha": sha, "files": {rel: {"sha256": files[rel]["sha256"], "size": files[rel]["size"]} for rel in sorted(files)}})
  finally:
    shutil.rmtree(staging_root, ignore_errors=True)
  log(f"installed artifact {sha[:12]} ({len(files)} files)")
  return EXIT_OK


class _Rejected(Exception):
  pass


def _reject(reason: str) -> int:
  print(f"[carrot_build_artifact] rejected: {reason}", file=sys.stderr, flush=True)
  return EXIT_REJECTED


def _installed_files(root: str) -> set[str]:
  try:
    return set((read_json(installed_manifest_path(root)).get("files") or {}).keys())
  except Exception:
    return set()


def _extract_verified(tarball: str, files: dict, staging_root: str) -> list[str]:
  """Extract every member with strict path safety, then verify hashes."""
  try:
    tar = tarfile.open(tarball, mode="r:xz")
  except (tarfile.TarError, OSError, EOFError) as exc:
    raise _Rejected(f"tarball is not readable: {exc}") from exc

  extracted: list[str] = []
  try:
    for member in tar:
      if not member.isfile():
        raise _Rejected(f"tarball member {member.name!r} is not a regular file")
      rel = safe_member_path(member.name)
      if rel is None or rel != member.name:
        raise _Rejected(f"unsafe tarball path {member.name!r}")
      if rel not in files:
        raise _Rejected(f"tarball member {rel} is not in the manifest")
      source = tar.extractfile(member)
      if source is None:
        raise _Rejected(f"cannot read tarball member {rel}")
      destination = os.path.join(staging_root, rel)
      os.makedirs(os.path.dirname(destination), exist_ok=True)
      with source, open(destination, "wb") as f:
        shutil.copyfileobj(source, f)
      extracted.append(rel)
    missing = sorted(set(files) - set(extracted))
    if missing:
      raise _Rejected(f"tarball is missing manifest files: {', '.join(missing[:5])}")
  finally:
    tar.close()

  for rel in extracted:
    info = files[rel]
    full = os.path.join(staging_root, rel)
    if os.path.getsize(full) != int(info["size"]):
      raise _Rejected(f"size mismatch for {rel}")
    if sha256_file(full) != str(info["sha256"]):
      raise _Rejected(f"sha256 mismatch for {rel}")
  return extracted


def _apply(repo: str, staging_root: str, extracted: list[str], files: dict, old_files: set[str], tracked: set[str]) -> None:
  for rel in sorted(extracted):
    destination = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(destination) or repo, exist_ok=True)
    tmp = f"{destination}.tmp.{os.getpid()}"
    shutil.copyfile(os.path.join(staging_root, rel), tmp)
    os.replace(tmp, destination)
    os.chmod(destination, int(str(files[rel]["mode"]), 8))

  for rel in sorted(old_files - set(files)):
    # A file that is tracked by the current commit belongs to the source tree
    # now; only previously installed build outputs may be removed.
    if rel in tracked:
      continue
    destination = os.path.join(repo, rel)
    if os.path.isfile(destination):
      os.remove(destination)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--root", default=None, help="data root (default: /data or CARROT_BUILD_ARTIFACT_DATA_ROOT)")
  sub = parser.add_subparsers(dest="command", required=True)

  fetch_parser = sub.add_parser("fetch", help="download the artifact for a commit into the cache")
  fetch_parser.add_argument("sha")
  fetch_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)

  install_parser = sub.add_parser("install", help="install a cached artifact into a checkout")
  install_parser.add_argument("sha")
  install_parser.add_argument("--repo", required=True)
  install_parser.add_argument("--tici-marker", default="/TICI")
  install_parser.add_argument("--ion-path", default="/dev/ion")

  check_parser = sub.add_parser("check", help="verify the installed artifact for a checkout")
  check_parser.add_argument("--repo", required=True)

  args = parser.parse_args(argv)
  root = data_root(args.root)

  if args.command == "fetch":
    return fetch(args.sha, max(1.0, args.timeout), root)
  if args.command == "install":
    return install(args.sha, os.path.abspath(args.repo), root, args.tici_marker, args.ion_path)
  if args.command == "check":
    return check(os.path.abspath(args.repo), root)
  return EXIT_ERROR


if __name__ == "__main__":
  sys.exit(main())
