"""Shared tool action policy.

Keep dispatcher behavior in dispatcher.py, but keep action names and the
free-form shell command policy in one small module.
"""
from __future__ import annotations

import posixpath
from typing import Iterable, List, Optional, Tuple


KNOWN_TOOL_ACTIONS = {
  "backup_settings",
  "delete_all_logs",
  "delete_all_videos",
  "git_branch_list",
  "git_checkout",
  "git_log",
  "git_pull",
  "git_remote_add",
  "git_remote_set",
  "git_reset",
  "git_reset_repo_checkout",
  "git_reset_repo_fetch",
  "git_sync",
  "reboot",
  "rebuild_all",
  "reset_calib",
  "send_tmux_log",
  "server_tmux_log",
  "shell_cmd",
}

READ_ONLY_SHELL_COMMANDS = {"cat", "df", "echo", "free", "git", "ls", "uptime"}
READ_ONLY_GIT_COMMANDS = {
  "branch",
  "diff",
  "log",
  "remote",
  "rev-parse",
  "show",
  "show-ref",
  "status",
}
_DISALLOWED_GIT_FLAGS = (
  "-c",
  "-o",
  "--config",
  "--git-dir",
  "--work-tree",
  "--exec-path",
  "--upload-pack",
  "--output",
  "--ext-diff",
  "--textconv",
  "--filter",
  "--no-index",
)
_READ_ONLY_BRANCH_FLAGS = {
  "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose",
  "-l", "--list", "--show-current", "-q", "--quiet",
  "--contains", "--no-contains", "--merged", "--no-merged", "--points-at",
  "--sort", "--format", "--color", "--no-color", "--abbrev", "--no-abbrev",
  "--column", "--no-column",
}
# Flags whose value git may take from the NEXT argument. --color/--abbrev/--column
# only accept an attached "--flag=value", so they must not be listed here: git would
# read the following argument as a branch operation (`--color -m --color x` renames).
_BRANCH_VALUE_FLAGS = {"--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--sort", "--format"}
_READ_ONLY_CAT_FLAGS = {"-n", "-b", "-s", "-v", "-E", "-T"}
_ALLOWED_CAT_ABS_PREFIXES = ("/data/openpilot/", "/data/media/", "/tmp/", "/proc/", "/sys/", "/var/log/")
_BLOCKED_CAT_SEGMENTS = {"..", ".ssh", "GithubSshKeys", "id_rsa", "id_ed25519", "authorized_keys", "known_hosts", "shadow", "gshadow", ".netrc", ".git-credentials"}
_BLOCKED_PROC_SEGMENTS = {"root", "cwd", "exe", "fd", "fdinfo", "map_files", "environ", "mem", "maps", "smaps"}


def _is_disallowed_git_flag(arg: str) -> bool:
  if not arg.startswith("-"):
    return False
  for flag in _DISALLOWED_GIT_FLAGS:
    if arg == flag or arg.startswith(f"{flag}="):
      return True
    if len(flag) == 2 and arg.startswith(flag) and len(arg) > 2:
      return True
  return False


def _validate_cat_args(args: List[str]) -> Optional[Tuple[str, str, str]]:
  paths: List[str] = []
  for arg in args:
    if arg == "-":
      return "cat stdin not allowed", "CMD_NOT_ALLOWED", arg
    if arg.startswith("-"):
      if arg not in _READ_ONLY_CAT_FLAGS:
        return f"cat flag not allowed: {arg}", "CMD_NOT_ALLOWED", arg
      continue
    if arg.startswith("~") or "\x00" in arg:
      return f"cat path not allowed: {arg}", "CMD_NOT_ALLOWED", arg
    raw_parts = [p for p in arg.replace("\\", "/").split("/") if p]
    if any(p in _BLOCKED_CAT_SEGMENTS for p in raw_parts):
      return f"cat path not allowed: {arg}", "CMD_NOT_ALLOWED", arg
    norm = posixpath.normpath(arg.replace("\\", "/"))
    norm_parts = [p for p in norm.split("/") if p]
    if any(p in _BLOCKED_CAT_SEGMENTS for p in norm_parts):
      return f"cat path not allowed: {arg}", "CMD_NOT_ALLOWED", arg
    if norm.startswith("/"):
      if not any(norm == p.rstrip("/") or norm.startswith(p) for p in _ALLOWED_CAT_ABS_PREFIXES):
        return f"cat path not allowed: {arg}", "CMD_NOT_ALLOWED", arg
      if norm.startswith("/proc/") and any(p in _BLOCKED_PROC_SEGMENTS for p in norm_parts[1:]):
        return f"cat path not allowed: {arg}", "CMD_NOT_ALLOWED", arg
    elif norm == ".." or norm.startswith("../"):
      return f"cat path not allowed: {arg}", "CMD_NOT_ALLOWED", arg
    paths.append(arg)
  if not paths:
    return "cat requires a file path", "CMD_NOT_ALLOWED", "cat"
  return None


def _validate_git_args(subcommand: str, args: List[str]) -> Optional[Tuple[str, str, str]]:
  for arg in args:
    if _is_disallowed_git_flag(arg):
      return f"git flag not allowed: {arg}", "GIT_CMD_NOT_ALLOWED", arg

  if subcommand == "diff":
    for arg in args:
      if arg == "--" or arg.startswith("-"):
        continue
      if arg.startswith(("/", "~")) or "\x00" in arg:
        return f"git diff path not allowed: {arg}", "GIT_CMD_NOT_ALLOWED", arg
      parts = arg.replace("\\", "/").split("/")
      if ".." in parts:
        return f"git diff path not allowed: {arg}", "GIT_CMD_NOT_ALLOWED", arg
    return None

  if subcommand == "branch":
    allow_positional = False
    expect_value = False
    for arg in args:
      if expect_value:
        expect_value = False
        continue
      if arg.startswith("-"):
        base = arg.split("=", 1)[0]
        if base not in _READ_ONLY_BRANCH_FLAGS:
          return f"git branch flag not allowed: {arg}", "GIT_CMD_NOT_ALLOWED", arg
        if base in ("-l", "--list"):
          allow_positional = True
        elif base in _BRANCH_VALUE_FLAGS and "=" not in arg:
          expect_value = True
      elif not allow_positional:
        return f"git branch mutation not allowed: {arg}", "GIT_CMD_NOT_ALLOWED", arg
    return None

  if subcommand == "remote":
    if not args:
      return None
    first = args[0]
    if first in ("-v", "--verbose") and len(args) == 1:
      return None
    if first == "show":
      for extra in args[1:]:
        if extra.startswith("-") and extra != "-n":
          return f"git remote flag not allowed: {extra}", "GIT_CMD_NOT_ALLOWED", extra
      return None
    if first == "get-url":
      for extra in args[1:]:
        if extra.startswith("-") and extra not in ("--push", "--all"):
          return f"git remote flag not allowed: {extra}", "GIT_CMD_NOT_ALLOWED", extra
      return None
    return f"git remote action not allowed: {first}", "GIT_CMD_NOT_ALLOWED", first

  return None


def normalize_action(action: object) -> str:
  return str(action or "").strip()


def is_known_action(action: object) -> bool:
  return normalize_action(action) in KNOWN_TOOL_ACTIONS


def validate_action(action: object) -> Optional[Tuple[str, str]]:
  normalized = normalize_action(action)
  if normalized:
    if normalized in KNOWN_TOOL_ACTIONS:
      return None
    return f"unknown action: {normalized}", "UNKNOWN_TOOL_ACTION"
  return "missing action", "MISSING_TOOL_ACTION"


def validate_shell_argv(argv: Iterable[str]) -> Optional[Tuple[str, str, str]]:
  parts: List[str] = [str(part) for part in argv]
  if not parts:
    return "empty cmd", "EMPTY_CMD", ""

  top = parts[0]
  if top not in READ_ONLY_SHELL_COMMANDS:
    return f"not allowed: {top}", "CMD_NOT_ALLOWED", top

  if top == "cat":
    return _validate_cat_args(parts[1:])

  if top != "git":
    return None

  if len(parts) < 2:
    return "missing git subcommand", "GIT_CMD_NOT_ALLOWED", ""

  subcommand = parts[1]
  if subcommand not in READ_ONLY_GIT_COMMANDS:
    return f"git subcommand not allowed: {subcommand}", "GIT_CMD_NOT_ALLOWED", subcommand

  return _validate_git_args(subcommand, parts[2:])
