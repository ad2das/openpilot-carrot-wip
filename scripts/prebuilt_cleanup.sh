#!/usr/bin/env bash
# Shared prebuilt-tree cleanup. Used by release/build_carrot.sh (on-device
# release) and by the carrot-wip-prebuilt CI workflow so the rules never
# diverge. Deletes build-only files, marks the tree prebuilt, and rejects
# files above GitHub's size limit. Pass the tree path as $1 (default: cwd).
set -Eeuo pipefail

TARGET_DIR="${1:-$PWD}"
cd -- "$TARGET_DIR"

find . -type f \( -name '*.a' -o -name '*.o' -o -name '*.os' -o -name '*.pyc' -o -name 'moc_*' \) -delete
find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' -o -name '.hypothesis' \) -prune -exec rm -rf -- {} +
rm -rf -- .sconsign.dblite Jenkinsfile release/
rm -f -- openpilot/selfdrive/modeld/models/*.onnx
# Build-time stamps describe the pre-cleanup tree and are device-local state;
# the launcher refreshes them for the trusted prebuilt checkout.
rm -f -- openpilot/selfdrive/modeld/models/.build_stamp openpilot/selfdrive/modeld/models/.big_model_build_stamp
touch prebuilt

BIG_FILES="$(find . -type f -not -path './.git/*' -size +95M -print)"
if [[ -n "$BIG_FILES" ]]; then
  printf 'files exceeding the GitHub size limit were found:\n%s\n' "$BIG_FILES" >&2
  exit 1
fi
