"""Tests for scripts/carrot_build_artifact.py (fetch/install/check).

The tool is exercised as a subprocess (the way the launcher and the update
paths run it) against throwaway git repositories, fake artifact tarballs and a
local HTTP server standing in for the GitHub release.
"""
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from openpilot.common.basedir import BASEDIR

TOOL = Path(BASEDIR) / "scripts" / "carrot_build_artifact.py"
AGNOS_VERSION = "test-agnos-1"


def host_arch() -> str:
  machine = platform.machine().lower()
  if machine in ("aarch64", "arm64"):
    return "larch64"
  if machine in ("x86_64", "amd64"):
    return "x86_64"
  return machine


def run_tool(root: Path, *args: str, env: dict | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
  full_env = dict(os.environ)
  if env:
    full_env.update(env)
  return subprocess.run([sys.executable, str(TOOL), "--root", str(root), *args],
                        capture_output=True, text=True, timeout=timeout, env=full_env, check=False)


def git(repo: Path, *args: str) -> str:
  result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
  if result.returncode:
    raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
  return result.stdout.strip()


def make_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
  repo = tmp_path / "repo"
  repo.mkdir()
  git(repo, "init", "-q", "-b", "main")
  git(repo, "config", "user.name", "artifact test")
  git(repo, "config", "user.email", "artifact-test@example.com")
  git(repo, "config", "commit.gpgsign", "false")
  for rel, content in files.items():
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
  git(repo, "add", "-A")
  git(repo, "commit", "-q", "-m", "init")
  return repo, git(repo, "rev-parse", "HEAD")


def make_artifact(directory: Path, sha: str, files: dict[str, bytes], *, tici: bool = False, ion: bool = False,
                  arch: str | None = None, agnos: str = AGNOS_VERSION, tracked_regenerated: list[str] | None = None,
                  tarball_sha256: str | None = None, tarball_size: int | None = None,
                  manifest_files: dict | None = None, modes: dict[str, str] | None = None,
                  extra_member: tuple[str, bytes] | None = None, symlink_member: str | None = None) -> tuple[Path, Path]:
  directory.mkdir(parents=True, exist_ok=True)
  tarball = directory / f"{sha}.tar.xz"
  with tarfile.open(tarball, "w:xz", preset=1) as tar:
    for rel, data in sorted(files.items()):
      info = tarfile.TarInfo(rel)
      info.size = len(data)
      info.mode = int((modes or {}).get(rel, "0644"), 8)
      tar.addfile(info, io.BytesIO(data))
    if extra_member is not None:
      info = tarfile.TarInfo(extra_member[0])
      info.size = len(extra_member[1])
      tar.addfile(info, io.BytesIO(extra_member[1]))
    if symlink_member is not None:
      info = tarfile.TarInfo(symlink_member)
      info.type = tarfile.SYMTYPE
      info.linkname = "target"
      tar.addfile(info)

  real_sha256 = hashlib.sha256(tarball.read_bytes()).hexdigest()
  manifest = {
    "format": 1,
    "source_commit": sha,
    "agnos_version": agnos,
    "arch": arch or host_arch(),
    "ion": ion,
    "tici": tici,
    "tarball": f"{sha}.tar.xz",
    "tarball_sha256": tarball_sha256 or real_sha256,
    "tarball_size": tarball.stat().st_size if tarball_size is None else tarball_size,
    "files": manifest_files if manifest_files is not None else {
      rel: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
            "mode": (modes or {}).get(rel, "0644")}
      for rel, data in files.items()
    },
    "tracked_regenerated": tracked_regenerated or [],
    "gates": {},
  }
  manifest_path = directory / f"{sha}.json"
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
  return manifest_path, tarball


def stage_cache(root: Path, sha: str, manifest: Path, tarball: Path) -> Path:
  cache = root / "carrot_build_cache" / sha
  cache.mkdir(parents=True, exist_ok=True)
  shutil.copy2(manifest, cache / f"{sha}.json")
  shutil.copy2(tarball, cache / f"{sha}.tar.xz")
  return cache


def run_install(root: Path, repo: Path, sha: str, *, tici_marker: Path | None = None, ion_path: Path | None = None,
                agnos: str = AGNOS_VERSION, env: dict | None = None) -> subprocess.CompletedProcess:
  extra = {"AGNOS_VERSION": agnos}
  if env:
    extra.update(env)
  return run_tool(root, "install", sha, "--repo", str(repo),
                  "--tici-marker", str(tici_marker or (root / "TICI")),
                  "--ion-path", str(ion_path or (root / "ion")), env=extra)


ALLOWLIST = "scripts/build_artifact_tracked_outputs.txt"


def test_install_basic_and_check(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {
    ALLOWLIST: "generated.txt\n",
    "generated.txt": "old\n",
    "keep.txt": "keep\n",
  })
  root = tmp_path / "data"
  files = {
    "generated.txt": b"new\n",
    "build/bin/app": b"binary",
    "lib/util.so": b"shared",
  }
  manifest, tarball = make_artifact(tmp_path / "art", sha, files, tracked_regenerated=["generated.txt"],
                                    modes={"build/bin/app": "0755"})
  stage_cache(root, sha, manifest, tarball)

  result = run_install(root, repo, sha)
  assert result.returncode == 0, result.stdout + result.stderr
  assert (repo / "generated.txt").read_text(encoding="utf-8") == "new\n"
  assert (repo / "build/bin/app").read_bytes() == b"binary"
  assert (repo / "lib/util.so").read_bytes() == b"shared"
  installed = json.loads((root / "carrot_build_artifacts/installed.json").read_text(encoding="utf-8"))
  assert installed["sha"] == sha
  assert set(installed["files"]) == set(files)
  if os.name != "nt":
    assert os.stat(repo / "build/bin/app").st_mode & 0o111
    assert not os.stat(repo / "lib/util.so").st_mode & 0o111

  assert run_tool(root, "check", "--repo", str(repo)).returncode == 0

  (repo / "build/bin/app").unlink()
  assert run_tool(root, "check", "--repo", str(repo)).returncode == 1


def test_check_rejects_other_commit(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n"})
  root = tmp_path / "data"
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"out/a": b"a"})
  stage_cache(root, sha, manifest, tarball)
  assert run_install(root, repo, sha).returncode == 0
  (repo / "extra.txt").write_text("x", encoding="utf-8")
  git(repo, "add", "-A")
  git(repo, "commit", "-q", "-m", "next")
  assert run_tool(root, "check", "--repo", str(repo)).returncode == 1


def test_install_rejects_path_traversal(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n"})
  root = tmp_path / "data"
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"out/ok.txt": b"ok"},
                                    extra_member=("../evil.txt", b"x"))
  stage_cache(root, sha, manifest, tarball)

  result = run_install(root, repo, sha)
  assert result.returncode == 2, result.stdout + result.stderr
  assert "unsafe tarball path" in result.stderr
  assert not (tmp_path / "evil.txt").exists()
  assert not (root / "carrot_build_artifacts/installed.json").exists()


def test_install_rejects_symlink_member(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n"})
  root = tmp_path / "data"
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"out/ok.txt": b"ok"}, symlink_member="link")
  stage_cache(root, sha, manifest, tarball)

  result = run_install(root, repo, sha)
  assert result.returncode == 2, result.stdout + result.stderr
  assert "not a regular file" in result.stderr


def test_install_rejects_tracked_file_overwrite(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n", "keep.txt": "keep\n"})
  root = tmp_path / "data"
  # keep.txt is tracked and not in tracked_regenerated, so the artifact must
  # never be allowed to overwrite it.
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"keep.txt": b"changed"})
  stage_cache(root, sha, manifest, tarball)

  result = run_install(root, repo, sha)
  assert result.returncode == 2, result.stdout + result.stderr
  assert "overwrite tracked file" in result.stderr
  assert (repo / "keep.txt").read_text(encoding="utf-8") == "keep\n"


def test_install_rejects_untracked_tracked_modification(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n", "src.py": "x\n"})
  root = tmp_path / "data"
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"out/a": b"a"})
  stage_cache(root, sha, manifest, tarball)
  (repo / "src.py").write_text("modified\n", encoding="utf-8")

  result = run_install(root, repo, sha)
  assert result.returncode == 2, result.stdout + result.stderr
  assert "tracked modification" in result.stderr


def test_install_config_mismatches(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n"})
  root = tmp_path / "data"
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"out/a": b"a"}, tici=True, ion=True)
  stage_cache(root, sha, manifest, tarball)

  # /TICI missing while the manifest says tici=true
  assert run_install(root, repo, sha).returncode == 2
  (root / "TICI").write_text("", encoding="utf-8")
  # /dev/ion missing while the manifest says ion=true
  assert run_install(root, repo, sha).returncode == 2
  (root / "ion").write_text("", encoding="utf-8")
  # AGNOS_VERSION mismatch
  assert run_install(root, repo, sha, agnos="other-version").returncode == 2
  # arch mismatch
  manifest2, tarball2 = make_artifact(tmp_path / "art2", sha, {"out/a": b"a"}, tici=True, ion=True,
                                      arch=f"not-{host_arch()}")
  stage_cache(root, sha, manifest2, tarball2)
  assert run_install(root, repo, sha).returncode == 2
  # HEAD mismatch
  fake_sha = "f" * 40
  manifest3, tarball3 = make_artifact(tmp_path / "art3", fake_sha, {"out/a": b"a"})
  stage_cache(root, fake_sha, manifest3, tarball3)
  assert run_install(root, repo, fake_sha).returncode == 2


def test_install_rejects_tarball_manifest_mismatch(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n"})
  root = tmp_path / "data"
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"out/a": b"a"}, extra_member=("out/b", b"b"))
  stage_cache(root, sha, manifest, tarball)

  result = run_install(root, repo, sha)
  assert result.returncode == 2, result.stdout + result.stderr
  assert "not in the manifest" in result.stderr


def test_install_removes_stale_untracked_files_only(tmp_path: Path) -> None:
  repo, sha_a = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n"})
  root = tmp_path / "data"
  manifest_a, tarball_a = make_artifact(tmp_path / "art-a", sha_a, {"out/a": b"a", "out/b": b"b"})
  stage_cache(root, sha_a, manifest_a, tarball_a)
  assert run_install(root, repo, sha_a).returncode == 0

  # Track out/b so the stale removal must leave it alone.
  git(repo, "add", "--", "out/b")
  git(repo, "commit", "-q", "-m", "track b")
  sha_b = git(repo, "rev-parse", "HEAD")
  manifest_b, tarball_b = make_artifact(tmp_path / "art-b", sha_b, {"out/a": b"a2", "out/c": b"c"})
  stage_cache(root, sha_b, manifest_b, tarball_b)
  assert run_install(root, repo, sha_b).returncode == 0
  assert (repo / "out/a").read_bytes() == b"a2"
  assert (repo / "out/b").read_bytes() == b"b"
  assert (repo / "out/c").read_bytes() == b"c"

  # A later commit and artifact drop out/c; out/b stays because it is tracked.
  (repo / "dummy.txt").write_text("x", encoding="utf-8")
  git(repo, "add", "--", "dummy.txt")
  git(repo, "commit", "-q", "-m", "next")
  sha_c = git(repo, "rev-parse", "HEAD")
  manifest_c, tarball_c = make_artifact(tmp_path / "art-c", sha_c, {"out/a": b"a3"})
  stage_cache(root, sha_c, manifest_c, tarball_c)
  assert run_install(root, repo, sha_c).returncode == 0
  assert (repo / "out/a").read_bytes() == b"a3"
  assert not (repo / "out/c").exists()
  assert (repo / "out/b").read_bytes() == b"b"


def test_install_partial_failure_leaves_no_installed_state(tmp_path: Path) -> None:
  repo, sha = make_repo(tmp_path, {ALLOWLIST: "generated.txt\n"})
  root = tmp_path / "data"
  # A plain file occupies the "blocked" path, so applying blocked/nested.txt
  # fails midway; installed.json must stay absent so the boot builds instead.
  (repo / "blocked").write_text("not a directory", encoding="utf-8")
  manifest, tarball = make_artifact(tmp_path / "art", sha, {"blocked/nested.txt": b"x"})
  stage_cache(root, sha, manifest, tarball)

  result = run_install(root, repo, sha)
  assert result.returncode == 1, result.stdout + result.stderr
  assert not (root / "carrot_build_artifacts/installed.json").exists()
  assert run_tool(root, "check", "--repo", str(repo)).returncode == 1


class _QuietHandler(SimpleHTTPRequestHandler):
  def log_message(self, *args):  # noqa: ANN002
    pass


@pytest.fixture
def served(tmp_path: Path):
  directory = tmp_path / "serve"
  directory.mkdir()
  handler = partial(_QuietHandler, directory=str(directory))
  server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  yield directory, f"http://127.0.0.1:{server.server_address[1]}"
  server.shutdown()
  server.server_close()


def test_fetch_downloads_and_verifies(tmp_path: Path, served) -> None:
  serve_dir, base = served
  sha = "a" * 40
  root = tmp_path / "data"
  manifest, tarball = make_artifact(serve_dir, sha, {"out/a": b"hello"})

  result = run_tool(root, "fetch", sha, "--timeout", "30", env={"CARROT_BUILD_ARTIFACT_URL": base})
  assert result.returncode == 0, result.stdout + result.stderr
  cache = root / "carrot_build_cache" / sha
  assert (cache / f"{sha}.json").exists()
  assert (cache / f"{sha}.tar.xz").read_bytes() == tarball.read_bytes()


def test_fetch_not_published_is_exit_3(tmp_path: Path, served) -> None:
  _, base = served
  root = tmp_path / "data"
  result = run_tool(root, "fetch", "b" * 40, "--timeout", "10", env={"CARROT_BUILD_ARTIFACT_URL": base})
  assert result.returncode == 3, result.stdout + result.stderr


def test_fetch_hash_mismatch_is_error(tmp_path: Path, served) -> None:
  serve_dir, base = served
  sha = "c" * 40
  root = tmp_path / "data"
  make_artifact(serve_dir, sha, {"out/a": b"hello"}, tarball_sha256="0" * 64)

  result = run_tool(root, "fetch", sha, "--timeout", "30", env={"CARROT_BUILD_ARTIFACT_URL": base})
  assert result.returncode == 1, result.stdout + result.stderr
  cache = root / "carrot_build_cache" / sha
  assert not (cache / f"{sha}.tar.xz").exists()
  assert not (cache / f"{sha}.tar.xz.tmp").exists()


def test_fetch_free_space_check_precedes_download(tmp_path: Path, served) -> None:
  serve_dir, base = served
  sha = "d" * 40
  root = tmp_path / "data"
  make_artifact(serve_dir, sha, {"out/a": b"hello"}, tarball_size=10**18)

  result = run_tool(root, "fetch", sha, "--timeout", "30", env={"CARROT_BUILD_ARTIFACT_URL": base})
  assert result.returncode == 1, result.stdout + result.stderr
  assert "free space" in result.stdout + result.stderr
  assert not (root / "carrot_build_cache" / sha / f"{sha}.tar.xz").exists()


def test_fetch_unreachable_network_is_error(tmp_path: Path) -> None:
  root = tmp_path / "data"
  result = run_tool(root, "fetch", "e" * 40, "--timeout", "5",
                    env={"CARROT_BUILD_ARTIFACT_URL": "http://127.0.0.1:1"},
                    timeout=30)
  assert result.returncode == 1, result.stdout + result.stderr


def test_fetch_prunes_cache_to_three_entries(tmp_path: Path, served) -> None:
  serve_dir, base = served
  root = tmp_path / "data"
  installed_sha = "1" * 40
  new_sha = "2" * 40
  make_artifact(serve_dir, new_sha, {"out/a": b"new"})

  cache_root = root / "carrot_build_cache"
  cache_root.mkdir(parents=True)
  for entry in (installed_sha, "3" * 40, "4" * 40, "5" * 40):
    (cache_root / entry).mkdir()
  state = root / "carrot_build_artifacts"
  state.mkdir(parents=True)
  (state / "installed.json").write_text(json.dumps({"sha": installed_sha, "files": {}}), encoding="utf-8")

  result = run_tool(root, "fetch", new_sha, "--timeout", "30", env={"CARROT_BUILD_ARTIFACT_URL": base})
  assert result.returncode == 0, result.stdout + result.stderr
  remaining = {p.name for p in cache_root.iterdir() if p.is_dir()}
  assert remaining == {installed_sha, new_sha, "5" * 40}, remaining
