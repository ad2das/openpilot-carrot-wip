#!/usr/bin/env python3
"""Pack and gate the carrot-wip device build artifact.

Used by .github/workflows/carrot-wip-build-artifacts.yaml on the runner host,
inside the checked-out, freshly built tree:

  pack     select build outputs, run the publish gates, write <sha>.tar.xz and
           <sha>.json (exit non-zero = no publish)
  repro    compare two manifests for identical inputs (report only)
  attest   record an externally produced gate result (device-install simulation)
  verify   re-check a published tarball against its manifest

Artifact content: every git-untracked file left by the build, minus models,
pydeps, python caches, build intermediates and test caches, plus the tracked
files the build regenerates (scripts/build_artifact_tracked_outputs.txt). Any
other tracked file the build modified fails the run.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import lzma
import os
import re
import subprocess
import sys
import tarfile

MODELS_DIR = "openpilot/selfdrive/modeld/models/"
ION_TARGETS = (
  "openpilot/system/camerad/camerad",
  "msgq_repo/msgq/visionipc/visionipc_pyx.so",
)
ION_STRING = b"/dev/ion"
SHM_STRING = b"/dev/shm/msgq_visionbuf"

EXCLUDED_COMPONENTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".hypothesis", "pydeps"}
EXCLUDED_SUFFIXES = (".pyc", ".o", ".os", ".a")
EXCLUDED_BASENAMES = {".sconsign.dblite"}

COMPILER_RE = re.compile(
  r"^\s*(?:\S*/)?(?:[a-z0-9_+.-]+-)?(?:gcc|g\+\+|cc|c\+\+|clang|clang\+\+)(?:-\d+(?:\.\d+)*)?\s"
)
SOURCE_RE = re.compile(r"[\w+./-]+\.(?:c|cc|cpp|cxx|C)(?![\w])")

AARCH64_EMACHINE = 0xB7
ELF_MAGIC = b"\x7fELF"


def git_bytes(repo: str, *args: str) -> bytes:
  result = subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)
  return result.stdout


def z_split(data: bytes) -> list[str]:
  return [p.decode("utf-8", "surrogateescape") for p in data.split(b"\0") if p]


def sha256_file(path: str) -> str:
  digest = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def read_json(path: str):
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def write_json_atomic(path: str, data) -> None:
  tmp = f"{path}.tmp.{os.getpid()}"
  with open(tmp, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
  os.replace(tmp, path)


def parse_allowlist(path: str) -> list[str]:
  entries: list[str] = []
  with open(path, encoding="utf-8") as f:
    for lineno, raw in enumerate(f, 1):
      line = raw.split("#", 1)[0].strip()
      if not line:
        continue
      entry = line.split()[0]
      if entry != line:
        raise SystemExit(f"{path}:{lineno}: tracked-output entry must be a single exact path: {raw.rstrip()}")
      entries.append(entry)
  if not entries:
    raise SystemExit(f"{path}: allowlist is empty")
  return entries


def normalized_mode(path: str) -> str:
  mode = os.stat(path).st_mode
  return "0755" if mode & 0o111 else "0644"


def is_excluded(rel: str) -> bool:
  parts = rel.split("/")
  if any(part in EXCLUDED_COMPONENTS for part in parts):
    return True
  base = parts[-1]
  if base.endswith(EXCLUDED_SUFFIXES) or base in EXCLUDED_BASENAMES or base.startswith("moc_"):
    return True
  return False


def select_files(repo: str, allowlist: list[str]) -> tuple[list[str], list[str], list[str]]:
  """Return (selected untracked paths, allowlisted tracked paths, untracked model artifacts)."""
  untracked = z_split(git_bytes(repo, "ls-files", "--others", "-z"))
  selected = []
  model_artifacts = []
  for rel in untracked:
    if rel.startswith(MODELS_DIR):
      model_artifacts.append(rel)
      continue
    if is_excluded(rel):
      continue
    selected.append(rel)

  tracked_outputs = []
  for rel in allowlist:
    full = os.path.join(repo, rel)
    if not os.path.isfile(full):
      raise SystemExit(f"tracked regenerated output is missing from the tree: {rel}")
    tracked_outputs.append(rel)

  return sorted(set(selected)), tracked_outputs, sorted(model_artifacts)


def check_tracked_modifications(repo: str, allowlist: list[str]) -> tuple[bool, list[str]]:
  modified = z_split(git_bytes(repo, "status", "--porcelain", "--untracked-files=no", "-z"))
  paths: list[str] = []
  i = 0
  while i < len(modified):
    entry = modified[i]
    if len(entry) >= 4:
      paths.append(entry[3:])
      if entry[0:1] in ("R", "C") and i + 1 < len(modified):
        paths.append(modified[i + 1])
        i += 1
    i += 1
  allowed = set(allowlist)
  unexpected = sorted(path for path in paths if path not in allowed)
  return not unexpected, unexpected


def check_ion(repo: str, selected: set[str]) -> tuple[bool, dict]:
  evidence = {}
  ok = True
  for rel in ION_TARGETS:
    if rel not in selected:
      evidence[rel] = "missing from the artifact"
      ok = False
      continue
    with open(os.path.join(repo, rel), "rb") as f:
      data = f.read()
    ion = data.count(ION_STRING)
    shm = data.count(SHM_STRING)
    evidence[rel] = {"bytes": len(data), "ion": ion, "shm": shm}
    if ion < 1 or shm != 0:
      ok = False
  return ok, evidence


def check_tici_compiles(log_path: str, minimum: int = 50) -> tuple[bool, dict]:
  if not log_path or not os.path.isfile(log_path):
    return False, {"error": f"scons log not found: {log_path}"}
  compile_commands = 0
  offenders: list[str] = []
  with open(log_path, encoding="utf-8", errors="replace") as f:
    for lineno, line in enumerate(f, 1):
      line = line.rstrip("\n")
      if not COMPILER_RE.match(line) or " -c " not in f"{line} ":
        continue
      sources = SOURCE_RE.findall(line)
      if not sources:
        continue
      if not [s for s in sources if not s.startswith("panda/") and "/panda/" not in s]:
        continue
      compile_commands += 1
      if "-D__TICI__" not in line:
        offenders.append(f"{log_path}:{lineno}: {line[:200]}")
  ok = not offenders and compile_commands >= minimum
  return ok, {"compile_commands": compile_commands, "offenders": offenders[:20],
              "minimum": minimum, "error": "" if ok else "missing -D__TICI__ or too few compiles"}


def elf_machine(path: str) -> str | None:
  with open(path, "rb") as f:
    header = f.read(20)
  if len(header) < 20 or header[:4] != ELF_MAGIC:
    return None
  if header[4] != 2:  # 64-bit
    return "not-64-bit"
  machine = int.from_bytes(header[18:20], "little")
  return "AArch64" if machine == AARCH64_EMACHINE else f"machine={machine:#x}"


def check_elf_arch(repo: str, selected: list[str]) -> tuple[bool, dict]:
  checked = 0
  firmware = 0
  offenders: list[str] = []
  for rel in selected:
    machine = elf_machine(os.path.join(repo, rel))
    if machine is None:
      continue
    if rel.startswith("panda/board/"):
      firmware += 1
      continue
    checked += 1
    if machine != "AArch64":
      offenders.append(f"{rel}: {machine}")
  ok = not offenders and checked > 0
  return ok, {"elf_files": checked, "panda_firmware": firmware, "offenders": offenders[:20]}


def pack_tarball(repo: str, paths: list[str], tarball_path: str) -> None:
  def reset(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mtime = 0
    tarinfo.mode = int(normalized_mode(os.path.join(repo, tarinfo.name)), 8)
    return tarinfo

  with tarfile.open(tarball_path, "w:xz", preset=6, format=tarfile.PAX_FORMAT) as tar:
    for rel in sorted(paths):
      tar.add(os.path.join(repo, rel), arcname=rel, recursive=False, filter=reset)


def tarball_files(tarball_path: str) -> dict[str, dict]:
  """Read the tarball back and return path -> {sha256, size, mode}."""
  out = {}
  with tarfile.open(tarball_path, mode="r:xz") as tar:
    for member in tar:
      if not member.isfile():
        raise SystemExit(f"packed tarball contains a non-regular entry: {member.name}")
      source = tar.extractfile(member)
      assert source is not None
      digest = hashlib.sha256()
      size = 0
      for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
      out[member.name] = {"sha256": digest.hexdigest(), "size": size, "mode": f"{member.mode & 0o777:04o}"}
  return out


def run_pack(args) -> int:
  repo = os.path.abspath(args.tree)
  allowlist = parse_allowlist(args.allowlist)
  source_sha = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()
  if args.source_sha and args.source_sha != source_sha:
    raise SystemExit(f"--source-sha {args.source_sha} does not match the checkout HEAD {source_sha}")

  failures: list[str] = []

  tracked_ok, unexpected = check_tracked_modifications(repo, allowlist)
  if not tracked_ok:
    failures.append(f"tracked modifications outside the allowlist: {', '.join(unexpected)}")
  print(f"tracked modifications: {'clean' if tracked_ok else 'UNEXPECTED'} "
        f"({len(unexpected)} unexpected of the allowlist {len(allowlist)})")

  selected, tracked_outputs, model_artifacts = select_files(repo, allowlist)
  print(f"artifact selection: {len(selected)} untracked files + {len(tracked_outputs)} tracked regenerated files; "
        f"{len(model_artifacts)} untracked model files excluded")
  if model_artifacts:
    print(f"  excluded model artifacts (device compiles these): {model_artifacts[:5]}{' ...' if len(model_artifacts) > 5 else ''}")
  if not selected:
    failures.append("no build outputs were selected")

  ion_ok, ion_evidence = check_ion(repo, set(selected) | set(tracked_outputs))
  print(f"ION gate: {'PASS' if ion_ok else 'FAIL'} {json.dumps(ion_evidence, sort_keys=True)}")
  if not ion_ok:
    failures.append("ION backend strings are missing from the artifact")

  tici_ok, tici_evidence = check_tici_compiles(args.scons_log)
  print(f"-D__TICI__ gate: {'PASS' if tici_ok else 'FAIL'} "
        f"({tici_evidence.get('compile_commands')} commands, {len(tici_evidence.get('offenders', []))} offenders)")
  for offender in tici_evidence.get("offenders", []):
    print(f"  offender: {offender}")
  if not tici_ok:
    failures.append(tici_evidence.get("error", "-D__TICI__ gate failed"))

  elf_ok, elf_evidence = check_elf_arch(repo, sorted(set(selected) | set(tracked_outputs)))
  print(f"ELF gate: {'PASS' if elf_ok else 'FAIL'} "
        f"({elf_evidence['elf_files']} ELF files, {elf_evidence['panda_firmware']} panda firmware skipped)")
  for offender in elf_evidence.get("offenders", []):
    print(f"  offender: {offender}")
  if not elf_ok:
    failures.append("artifact contains a non-AArch64 ELF")

  model_in_artifact = [rel for rel in selected + tracked_outputs if rel.startswith(MODELS_DIR)]
  if model_in_artifact:
    failures.append(f"model artifacts leaked into the artifact: {model_in_artifact[:5]}")
  print(f"model gate: {'PASS' if not model_in_artifact else 'FAIL'} (artifact ships no model artifacts)")

  os.makedirs(args.out_dir, exist_ok=True)
  tarball_path = os.path.join(args.out_dir, f"{source_sha}.tar.xz")
  manifest_path = os.path.join(args.out_dir, f"{source_sha}.json")

  if failures:
    for failure in failures:
      print(f"GATE FAILURE: {failure}")
    print("PARITY GATES FAILED; artifact not packed")
    return 1

  pack_tarball(repo, selected + tracked_outputs, tarball_path)
  packed = tarball_files(tarball_path)
  file_hashes = {rel: {"sha256": sha256_file(os.path.join(repo, rel)),
                       "size": os.path.getsize(os.path.join(repo, rel)),
                       "mode": normalized_mode(os.path.join(repo, rel))}
                 for rel in selected + tracked_outputs}

  hashes_ok = packed == file_hashes
  print(f"manifest hash gate: {'PASS' if hashes_ok else 'FAIL'} ({len(packed)} packed files)")
  if not hashes_ok:
    for rel in sorted(set(packed) | set(file_hashes)):
      if packed.get(rel) != file_hashes.get(rel):
        print(f"  mismatch: {rel}: packed={packed.get(rel)} tree={file_hashes.get(rel)}")
    print("PARITY GATES FAILED; artifact not published")
    return 1

  tarball_sha256 = sha256_file(tarball_path)
  tarball_size = os.path.getsize(tarball_path)
  uncompressed = sum(info["size"] for info in file_hashes.values())
  manifest = {
    "format": 1,
    "source_commit": source_sha,
    "agnos_version": args.agnos_version,
    "arch": args.arch,
    "ion": True,
    "tici": True,
    "tarball": f"{source_sha}.tar.xz",
    "tarball_sha256": tarball_sha256,
    "tarball_size": tarball_size,
    "files": file_hashes,
    "tracked_regenerated": tracked_outputs,
    "gates": {
      "tracked_clean": {"pass": tracked_ok, "allowlist": len(allowlist)},
      "ion": {"pass": ion_ok, "evidence": ion_evidence},
      "tici_compiles": {"pass": tici_ok, "compile_commands": tici_evidence.get("compile_commands"),
                        "offenders": len(tici_evidence.get("offenders", []))},
      "aarch64_elf": {"pass": elf_ok, **elf_evidence},
      "no_model_artifacts": {"pass": not model_in_artifact, "excluded": len(model_artifacts)},
      "manifest_hashes": {"pass": hashes_ok, "files": len(file_hashes)},
      "device_install_simulation": "pending",
    },
  }
  write_json_atomic(manifest_path, manifest)
  print(f"artifact packed: {tarball_path} ({tarball_size} bytes compressed, {uncompressed} bytes uncompressed)")
  print(f"manifest written: {manifest_path} ({len(file_hashes)} files)")

  if args.repro_previous:
    compare_manifests(args.repro_previous, manifest_path)
  return 0


def compare_manifests(previous_path: str, current_path: str) -> None:
  print("=== reproducibility report (two runs with identical inputs) ===")
  try:
    previous = read_json(previous_path)
    current = read_json(current_path)
  except Exception as exc:
    print(f"reproducibility: cannot read a manifest: {exc}")
    return
  if previous.get("source_commit") != current.get("source_commit"):
    print(f"reproducibility: previous run built {str(previous.get('source_commit'))[:12]}, "
          f"current built {str(current.get('source_commit'))[:12]}; inputs differ, comparison skipped")
    return
  prev_files = previous.get("files") or {}
  cur_files = current.get("files") or {}
  added = sorted(set(cur_files) - set(prev_files))
  removed = sorted(set(prev_files) - set(cur_files))
  changed = sorted(rel for rel in set(prev_files) & set(cur_files) if prev_files[rel] != cur_files[rel])
  print(f"reproducibility for {str(current.get('source_commit'))[:12]}: "
        f"{len(added)} added, {len(removed)} removed, {len(changed)} changed, {len(cur_files)} total")
  for rel in added:
    print(f"  added:   {rel}")
  for rel in removed:
    print(f"  removed: {rel}")
  for rel in changed:
    print(f"  changed: {rel}")
  if not (added or removed or changed):
    print("reproducibility: PASS; byte-identical file set and hashes")
  summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
  if summary_path:
    lines = [f"### Reproducibility ({current.get('source_commit', '')[:12]})",
             f"- added: {len(added)}, removed: {len(removed)}, changed: {len(changed)} of {len(cur_files)} files"]
    lines += [f"- changed: `{rel}`" for rel in changed[:20]]
    with open(summary_path, "a", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")


def run_repro(args) -> int:
  compare_manifests(args.previous, args.current)
  return 0


def run_attest(args) -> int:
  manifest = read_json(args.manifest)
  gates = manifest.setdefault("gates", {})
  try:
    gates[args.key] = json.loads(args.value)
  except ValueError:
    gates[args.key] = args.value
  write_json_atomic(args.manifest, manifest)
  print(f"attested {args.key}={gates[args.key]!r} in {args.manifest}")
  return 0


def run_verify(args) -> int:
  manifest = read_json(args.manifest)
  if not os.path.isfile(args.tarball):
    print(f"tarball not found: {args.tarball}")
    return 1
  actual = sha256_file(args.tarball)
  if actual != manifest.get("tarball_sha256"):
    print(f"tarball sha256 mismatch: got {actual}, expected {manifest.get('tarball_sha256')}")
    return 1
  if os.path.getsize(args.tarball) != manifest.get("tarball_size"):
    print("tarball size mismatch")
    return 1
  packed = tarball_files(args.tarball)
  if packed != manifest.get("files"):
    print("tarball contents do not match the manifest")
    for rel in sorted(set(packed) | set(manifest.get("files") or {})):
      if packed.get(rel) != (manifest.get("files") or {}).get(rel):
        print(f"  mismatch: {rel}")
    return 1
  print(f"verify OK: {manifest.get('source_commit', '')[:12]} ({len(packed)} files)")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="command", required=True)

  pack = sub.add_parser("pack")
  pack.add_argument("--tree", required=True)
  pack.add_argument("--source-sha", default="")
  pack.add_argument("--allowlist", required=True)
  pack.add_argument("--scons-log", required=True)
  pack.add_argument("--out-dir", required=True)
  pack.add_argument("--agnos-version", required=True)
  pack.add_argument("--arch", default="larch64")
  pack.add_argument("--repro-previous", default="")

  repro = sub.add_parser("repro")
  repro.add_argument("--previous", required=True)
  repro.add_argument("--current", required=True)

  attest = sub.add_parser("attest")
  attest.add_argument("--manifest", required=True)
  attest.add_argument("--key", required=True)
  attest.add_argument("--value", required=True)

  verify = sub.add_parser("verify")
  verify.add_argument("--manifest", required=True)
  verify.add_argument("--tarball", required=True)

  args = parser.parse_args()
  if args.command == "pack":
    return run_pack(args)
  if args.command == "repro":
    return run_repro(args)
  if args.command == "attest":
    return run_attest(args)
  if args.command == "verify":
    return run_verify(args)
  return 1


if __name__ == "__main__":
  sys.exit(main())
