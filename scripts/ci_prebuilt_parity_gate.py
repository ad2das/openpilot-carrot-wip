#!/usr/bin/env python3
"""Publish-parity gates G1-G6 for the carrot-wip-prebuilt workflow.

Runs against the cleaned build tree before publishing (worktree mode) and
against a freshly fetched published commit after publishing (commit mode).
G7 lives in the workflow because it needs the AGNOS chroot.

Every gate prints raw evidence; the script exits non-zero when any requested
gate fails so the publish step is skipped.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys

MODELS_DIR = "openpilot/selfdrive/modeld/models"
ION_TARGETS = (
  "openpilot/system/camerad/camerad",
  "msgq_repo/msgq/visionipc/visionipc_pyx.so",
)
ION_STRING = b"/dev/ion"
SHM_STRING = b"/dev/shm/msgq_visionbuf"
MODEL_INPUTS_PATHS = (
  "openpilot/selfdrive/modeld",
  "tinygrad_repo",
  "openpilot/common/file_chunker.py",
)

COMPILER_RE = re.compile(
  r"^\s*(?:\S*/)?(?:[a-z0-9_+.-]+-)?(?:gcc|g\+\+|cc|c\+\+|clang|clang\+\+)(?:-\d+(?:\.\d+)*)?\s"
)
SOURCE_RE = re.compile(r"[\w+./-]+\.(?:c|cc|cpp|cxx|C)(?![\w])")


def git_out(repo: str, *args: str) -> str:
  result = subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True)
  return result.stdout


def git_bytes(repo: str, *args: str) -> bytes:
  result = subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)
  return result.stdout


def z_split(text: str) -> list[str]:
  return [p for p in text.split("\0") if p]


def parse_allowlist(path: str) -> list[tuple[str, str]]:
  entries: list[tuple[str, str]] = []
  with open(path, encoding="utf-8") as f:
    for lineno, raw in enumerate(f, 1):
      line = raw.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split(None, 1)
      pattern = parts[0]
      reason = parts[1].strip() if len(parts) > 1 else ""
      if reason.startswith("#"):
        reason = reason[1:].strip()
      if not reason:
        raise SystemExit(f"{path}:{lineno}: allowlist entry needs a one-line reason: {line}")
      entries.append((pattern, reason))
  return entries


def allowlist_reason(path: str, entries: list[tuple[str, str]]) -> str | None:
  for pattern, reason in entries:
    if fnmatch.fnmatchcase(path, pattern):
      return reason
  return None


def source_tracked(repo: str, source_sha: str) -> set[str]:
  return set(z_split(git_out(repo, "ls-tree", "-r", "--name-only", "-z", source_sha)))


def gate_g1(repo: str, source_sha: str, published_sha: str | None, allowlist: list[tuple[str, str]]) -> bool:
  diff_args = ["diff", "--name-only", "-z", "--no-renames"]
  if published_sha:
    label = f"{source_sha[:12]}..{published_sha[:12]}"
    paths = z_split(git_out(repo, *diff_args, source_sha, published_sha))
    tracked = source_tracked(repo, source_sha)
    paths = [p for p in paths if p in tracked]
  else:
    label = "source commit vs built worktree"
    paths = z_split(git_out(repo, *diff_args))

  print(f"G1 tracked superset ({label}): {len(paths)} changed or deleted tracked paths")
  violations = []
  for path in sorted(paths):
    reason = allowlist_reason(path, allowlist)
    if reason:
      print(f"G1   allow  {path}  # {reason}")
    else:
      print(f"G1   FAIL   {path}")
      violations.append(path)
  if not paths:
    print("G1   no tracked changes or deletions")
  print(f"G1 result: {'PASS' if not violations else 'FAIL'}"
        f" ({len(paths) - len(violations)} allowed, {len(violations)} disallowed)")
  return not violations


def gate_g2(repo: str, root: str, source_sha: str, published_sha: str | None) -> bool:
  if published_sha:
    src = set(z_split(git_out(repo, "ls-tree", "-r", "--name-only", "-z", source_sha, "--", MODELS_DIR)))
    pub = set(z_split(git_out(repo, "ls-tree", "-r", "--name-only", "-z", published_sha, "--", MODELS_DIR)))
    extra = sorted(pub - src)
    print(f"G2 models dir (published {published_sha[:12]}): {len(pub)} tracked files, {len(extra)} not in the source commit")
    for path in extra:
      print(f"G2   FAIL   unexpected published file {path}")
    onnx = sorted(p for p in pub if p.endswith(".onnx"))
    print(f"G2   published model inputs (.onnx): {onnx if onnx else 'none'}")
    ok = not extra and bool(onnx)
    print(f"G2 result: {'PASS' if ok else 'FAIL'}")
    return ok

  untracked = z_split(git_out(repo, "ls-files", "--others", "-z", "--", MODELS_DIR))
  tracked = z_split(git_out(repo, "ls-files", "-z", "--", MODELS_DIR))
  onnx = [p for p in tracked if p.endswith(".onnx")]
  missing = [p for p in onnx if not os.path.isfile(os.path.join(root, p))]
  print(f"G2 models dir (worktree): {len(untracked)} untracked files, {len(tracked)} tracked files")
  for path in untracked:
    print(f"G2   FAIL   untracked model artifact {path}")
  print(f"G2   tracked model inputs (.onnx): {onnx if onnx else 'none'}")
  for path in missing:
    print(f"G2   FAIL   tracked model input missing on disk: {path}")
  ok = not untracked and bool(onnx) and not missing
  print(f"G2 result: {'PASS' if ok else 'FAIL'}")
  return ok


def gate_g3(repo: str, root: str, published_sha: str | None) -> bool:
  label = f"published {published_sha[:12]}" if published_sha else "worktree"
  ok = True
  for rel in ION_TARGETS:
    if published_sha:
      data = git_bytes(repo, "show", f"{published_sha}:{rel}")
    else:
      with open(os.path.join(root, rel), "rb") as f:
        data = f.read()
    ion = data.count(ION_STRING)
    shm = data.count(SHM_STRING)
    print(f"G3 ION ({label}): {rel}: /dev/ion x{ion}, /dev/shm/msgq_visionbuf x{shm}, {len(data)} bytes")
    if ion < 1 or shm != 0:
      print(f"G3   FAIL   {rel} must use the ION backend and not the /dev/shm fallback")
      ok = False
  print(f"G3 result: {'PASS' if ok else 'FAIL'}")
  return ok


def gate_g4(log_path: str, min_commands: int) -> bool:
  if not log_path or not os.path.isfile(log_path):
    print(f"G4   FAIL   scons log not found: {log_path}")
    print("G4 result: FAIL")
    return False
  compile_commands: list[tuple[int, str]] = []
  offenders: list[tuple[int, str]] = []
  with open(log_path, encoding="utf-8", errors="replace") as f:
    for lineno, line in enumerate(f, 1):
      line = line.rstrip("\n")
      if not COMPILER_RE.match(line):
        continue
      if " -c " not in f"{line} ":
        continue
      sources = SOURCE_RE.findall(line)
      if not sources:
        continue
      outside = [s for s in sources if not s.startswith("panda/") and "/panda/" not in s]
      if not outside:
        continue
      compile_commands.append((lineno, line))
      if "-D__TICI__" not in line:
        offenders.append((lineno, line))
  print(f"G4 larch64 config: {len(compile_commands)} C/C++ compile commands outside panda/ in {log_path}")
  for lineno, line in offenders:
    print(f"G4   FAIL   {log_path}:{lineno}: missing -D__TICI__: {line}")
  if len(compile_commands) < min_commands:
    print(f"G4   FAIL   expected at least {min_commands} compile commands, found {len(compile_commands)}")
    print("G4 result: FAIL")
    return False
  ok = not offenders
  print(f"G4 result: {'PASS' if ok else 'FAIL'} ({len(offenders)} offenders)")
  return ok


def gate_g5(repo: str, root: str) -> bool:
  untracked = z_split(git_out(repo, "ls-files", "--others", "-z"))
  checked = 0
  offenders: list[tuple[str, str]] = []
  for rel in untracked:
    full = os.path.join(root, rel)
    if os.path.islink(full) or not os.path.isfile(full):
      continue
    try:
      with open(full, "rb") as f:
        magic = f.read(4)
    except OSError as exc:
      offenders.append((rel, f"unreadable: {exc}"))
      continue
    if magic != b"\x7fELF":
      continue
    checked += 1
    result = subprocess.run(["readelf", "-h", full], capture_output=True, text=True, check=False)
    if result.returncode != 0:
      offenders.append((rel, f"readelf failed: {result.stderr.strip()}"))
      continue
    machine = "unknown"
    for line in result.stdout.splitlines():
      if "Machine:" in line:
        machine = line.split("Machine:", 1)[1].strip()
        break
    if machine != "AArch64":
      offenders.append((rel, machine))
  print(f"G5 untracked ELF architecture: {checked} untracked ELF files scanned, {len(offenders)} not AArch64")
  for rel, machine in offenders:
    print(f"G5   FAIL   {rel}: {machine}")
  ok = not offenders and checked > 0
  if checked == 0:
    print("G5   FAIL   no untracked ELF files were found; the built tree looks empty")
  print(f"G5 result: {'PASS' if ok else 'FAIL'}")
  return ok


def expected_model_inputs(repo: str, source_sha: str) -> str:
  parts = git_out(repo, "rev-parse", *[f"{source_sha}:{p}" for p in MODEL_INPUTS_PATHS]).splitlines()
  return "".join(f"{part}:" for part in parts)


def gate_g6(repo: str, root: str, source_sha: str, published_sha: str | None) -> bool:
  if published_sha:
    raw = git_bytes(repo, "show", f"{published_sha}:prebuilt.json")
    label = f"published {published_sha[:12]}"
  else:
    json_path = os.path.join(root, "prebuilt.json")
    label = "worktree"
    if not os.path.isfile(json_path):
      print(f"G6   FAIL   prebuilt.json is missing at the tree root")
      print("G6 result: FAIL")
      return False
    with open(json_path, "rb") as f:
      raw = f.read()

  expected_inputs = expected_model_inputs(repo, source_sha)
  try:
    data = json.loads(raw.decode("utf-8"))
  except ValueError as exc:
    print(f"G6   FAIL   prebuilt.json does not parse ({label}): {exc}")
    print("G6 result: FAIL")
    return False

  failures = []
  if data.get("builder") != "ci":
    failures.append(f"builder={data.get('builder')!r} (expected 'ci')")
  if data.get("models") != "device":
    failures.append(f"models={data.get('models')!r} (expected 'device')")
  if data.get("source_commit") != source_sha:
    failures.append(f"source_commit={data.get('source_commit')!r} (expected {source_sha})")
  if data.get("model_inputs") != expected_inputs:
    failures.append(f"model_inputs={data.get('model_inputs')!r} (expected {expected_inputs!r})")
  print(f"G6 prebuilt.json ({label}): {json.dumps(data, sort_keys=True)}")
  print(f"G6   recomputed model_inputs from {source_sha[:12]}: {expected_inputs}")
  for failure in failures:
    print(f"G6   FAIL   {failure}")
  ok = not failures
  print(f"G6 result: {'PASS' if ok else 'FAIL'}")
  return ok


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=["worktree", "commit"], default="worktree")
  parser.add_argument("--tree", required=True, help="repository holding the tree/commits")
  parser.add_argument("--source-sha", required=True)
  parser.add_argument("--published-sha", default="")
  parser.add_argument("--allowlist", required=True)
  parser.add_argument("--scons-log", default="")
  parser.add_argument("--min-compile-commands", type=int, default=50)
  parser.add_argument("--gates", default="G1,G2,G3,G4,G5,G6")
  args = parser.parse_args()

  if args.mode == "commit" and not args.published_sha:
    parser.error("--mode commit requires --published-sha")

  published = args.published_sha or None
  allowlist = parse_allowlist(args.allowlist)
  requested = [g.strip().upper() for g in args.gates.split(",") if g.strip()]

  results: dict[str, bool] = {}
  for gate in requested:
    print(f"=== {gate} ===")
    if gate == "G1":
      results[gate] = gate_g1(args.tree, args.source_sha, published, allowlist)
    elif gate == "G2":
      results[gate] = gate_g2(args.tree, args.tree, args.source_sha, published)
    elif gate == "G3":
      results[gate] = gate_g3(args.tree, args.tree, published)
    elif gate == "G4":
      results[gate] = gate_g4(args.scons_log, args.min_compile_commands)
    elif gate == "G5":
      results[gate] = gate_g5(args.tree, args.tree)
    elif gate == "G6":
      results[gate] = gate_g6(args.tree, args.tree, args.source_sha, published)
    else:
      print(f"unknown gate {gate}")
      results[gate] = False

  print("=== gate summary ===")
  for gate in requested:
    print(f"  {gate}: {'PASS' if results[gate] else 'FAIL'}")
  failed = [g for g in requested if not results[g]]
  if failed:
    print(f"PARITY GATES FAILED: {', '.join(failed)}")
    return 1
  print("PARITY GATES PASSED")
  return 0


if __name__ == "__main__":
  sys.exit(main())
