#!/usr/bin/env bash
# Shared prebuilt-tree cleanup for on-device release builds.
# Used by release/build_carrot.sh (--mode device-release).
#
# Only git-untracked files are ever deleted, so tracked build inputs such as
# third_party/raylib/*/libraylib.a survive the cleanup. The intentional
# tracked deletions are models/*.onnx and the release-only Jenkinsfile and
# release/ directory, which are dropped from the published tree. Rejects files
# above GitHub's size limit.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prebuilt_cleanup.sh --mode device-release [TREE]

  --mode device-release  On-device release semantics: drop untracked build
                         junk, models/*.onnx and device-local stamps; keep the
                         device-built model pickles.
EOF
}

MODE=""
TARGET_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MODE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      usage >&2
      exit 2
      ;;
    *)
      [[ -z "$TARGET_DIR" ]] || { usage >&2; exit 2; }
      TARGET_DIR="$1"
      shift
      ;;
  esac
done

[[ "$MODE" == "device-release" ]] || { usage >&2; exit 2; }
TARGET_DIR="${TARGET_DIR:-$PWD}"
cd -- "$TARGET_DIR"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  printf 'prebuilt cleanup requires a Git checkout: %s\n' "$TARGET_DIR" >&2
  exit 1
}

UNTRACKED="$(mktemp)"
cleanup_tmp() { rm -f -- "$UNTRACKED"; }
trap cleanup_tmp EXIT
git ls-files --others -z | tr '\0' '\n' > "$UNTRACKED"

is_untracked() {
  grep -Fxq -- "$1" "$UNTRACKED"
}

# Delete untracked files whose basename matches the historical build-junk
# patterns. Tracked files are never touched, even when they match.
delete_untracked_build_junk() {
  local rel base
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    base="${rel##*/}"
    case "$base" in
      *.a|*.o|*.os|*.pyc|moc_*) rm -f -- "$rel" ;;
    esac
  done < "$UNTRACKED"
}

# Remove cache directories only when no tracked file lives under them.
delete_untracked_cache_dirs() {
  local rel dir
  while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    dir="$rel"
    while [[ "$dir" == */* ]]; do
      dir="${dir%/*}"
      case "${dir##*/}" in
        __pycache__|.pytest_cache|.ruff_cache|.mypy_cache|.hypothesis)
          if [[ -d "$dir" ]] && [[ -z "$(git ls-files -- "$dir")" ]]; then
            rm -rf -- "$dir"
          fi
          ;;
      esac
    done
  done < "$UNTRACKED"
}

# The historical rule removed only the tree-root .sconsign.dblite.
delete_root_sconsign() {
  [[ -e ".sconsign.dblite" ]] || return 0
  if [[ -f ".sconsign.dblite" ]]; then
    is_untracked ".sconsign.dblite" && rm -f -- ".sconsign.dblite"
  elif [[ -z "$(git ls-files -- ".sconsign.dblite")" ]]; then
    rm -rf -- ".sconsign.dblite"
  fi
}

delete_untracked_build_junk
delete_untracked_cache_dirs
delete_root_sconsign

# On-device release keeps the compiled model pickles but the historical
# semantics always dropped the .onnx inputs, the device-local stamps and any
# stale CI manifest that would make the launcher expect a device rebuild.
rm -f -- openpilot/selfdrive/modeld/models/*.onnx
rm -f -- openpilot/selfdrive/modeld/models/.build_stamp \
         openpilot/selfdrive/modeld/models/.big_model_build_stamp
rm -f -- prebuilt.json

# Release-only files are dropped from every published tree.
rm -rf -- Jenkinsfile release/
touch prebuilt

BIG_FILES="$(find . -type f -not -path './.git/*' -size +95M -print)"
if [[ -n "$BIG_FILES" ]]; then
  printf 'files exceeding the GitHub size limit were found:\n%s\n' "$BIG_FILES" >&2
  exit 1
fi
