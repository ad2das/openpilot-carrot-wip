#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CLEANUP="$SCRIPT_DIR/../scripts/prebuilt_cleanup.sh"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT

make_repo() {
  local tree="$1"
  mkdir -p "$tree"
  git -c init.templateDir= init "$tree" >/dev/null
  git -C "$tree" config user.name "prebuilt cleanup test"
  git -C "$tree" config user.email "cleanup-test@example.com"
}

populate_tracked() {
  local tree="$1"
  mkdir -p "$tree/third_party/raylib/larch64" "$tree/third_party/raylib/x86_64" \
           "$tree/openpilot/selfdrive/modeld/models" "$tree/openpilot/common" \
           "$tree/tinygrad_repo" "$tree/release/tests"
  touch "$tree/third_party/raylib/larch64/libraylib.a" \
        "$tree/third_party/raylib/x86_64/libraylib.a" \
        "$tree/openpilot/selfdrive/modeld/models/driving_supercombo.onnx" \
        "$tree/openpilot/selfdrive/modeld/models/README.md" \
        "$tree/openpilot/common/file_chunker.py" \
        "$tree/tinygrad_repo/README.md" \
        "$tree/Jenkinsfile" \
        "$tree/release/build_carrot.sh"
  git -C "$tree" add -A
  git -C "$tree" commit -m tracked >/dev/null
}

populate_untracked_junk() {
  local tree="$1"
  mkdir -p "$tree/pkg/sub" "$tree/pkg/sub/__pycache__" "$tree/pkg/__pycache__" "$tree/pkg/.pytest_cache" "$tree/.ruff_cache" \
           "$tree/.mypy_cache" "$tree/.hypothesis"
  touch "$tree/pkg/a.o" "$tree/pkg/b.a" "$tree/pkg/c.os" "$tree/pkg/d.pyc" "$tree/pkg/sub/moc_widget.cpp"
  touch "$tree/pkg/__pycache__/x.pyc" "$tree/pkg/sub/__pycache__/y.pyc"
  touch "$tree/pkg/.pytest_cache/state" "$tree/.ruff_cache/state" "$tree/.mypy_cache/state" "$tree/.hypothesis/state"
  touch "$tree/.sconsign.dblite" "$tree/pkg/sub/.sconsign.dblite"
}

populate_untracked_models() {
  local tree="$1"
  touch "$tree/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl" \
        "$tree/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl.chunkmanifest" \
        "$tree/openpilot/selfdrive/modeld/models/tg_input_devices.json" \
        "$tree/openpilot/selfdrive/modeld/models/.build_stamp" \
        "$tree/openpilot/selfdrive/modeld/models/.big_model_build_stamp" \
        "$tree/openpilot/selfdrive/modeld/models/extra.onnx"
}

assert_junk_gone() {
  local tree="$1"
  [[ ! -e "$tree/pkg/a.o" && ! -e "$tree/pkg/b.a" && ! -e "$tree/pkg/c.os" && ! -e "$tree/pkg/d.pyc" ]] || {
    echo "build junk survived cleanup" >&2; exit 1; }
  [[ ! -e "$tree/pkg/sub/moc_widget.cpp" && ! -e "$tree/pkg/__pycache__" && ! -e "$tree/pkg/sub/__pycache__" ]] || {
    echo "moc/pycache survived cleanup" >&2; exit 1; }
  [[ ! -e "$tree/.pytest_cache" && ! -e "$tree/.ruff_cache" && ! -e "$tree/.mypy_cache" && ! -e "$tree/.hypothesis" ]] || {
    echo "cache directories survived cleanup" >&2; exit 1; }
  [[ ! -e "$tree/.sconsign.dblite" ]] || { echo "root .sconsign.dblite survived cleanup" >&2; exit 1; }
  # Only the tree-root .sconsign.dblite is removed, matching the historical rule.
  [[ -e "$tree/pkg/sub/.sconsign.dblite" ]] || { echo "nested .sconsign.dblite was removed" >&2; exit 1; }
}

# --- device-release mode: previous semantics, but tracked files survive ---
DEVICE="$TMP_DIR/device"
make_repo "$DEVICE"
populate_tracked "$DEVICE"
populate_untracked_junk "$DEVICE"
populate_untracked_models "$DEVICE"

bash "$CLEANUP" --mode device-release "$DEVICE"

[[ -f "$DEVICE/prebuilt" ]]
[[ ! -e "$DEVICE/prebuilt.json" ]]
# The tracked raylib libraries must survive (they used to be deleted).
[[ -f "$DEVICE/third_party/raylib/larch64/libraylib.a" && -f "$DEVICE/third_party/raylib/x86_64/libraylib.a" ]]
# The device-compiled model pickles stay; the onnx inputs are the intentional drop.
[[ -f "$DEVICE/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl" ]]
[[ ! -e "$DEVICE/openpilot/selfdrive/modeld/models/driving_supercombo.onnx" ]]
[[ ! -e "$DEVICE/openpilot/selfdrive/modeld/models/extra.onnx" ]]
[[ ! -e "$DEVICE/openpilot/selfdrive/modeld/models/.build_stamp" && ! -e "$DEVICE/openpilot/selfdrive/modeld/models/.big_model_build_stamp" ]]
[[ -f "$DEVICE/openpilot/selfdrive/modeld/models/README.md" ]]
[[ ! -e "$DEVICE/Jenkinsfile" && ! -e "$DEVICE/release" ]]
assert_junk_gone "$DEVICE"

echo "device-release cleanup passed"

# --- ci mode: keep the tracked onnx inputs, drop every untracked model artifact ---
CI="$TMP_DIR/ci"
make_repo "$CI"
populate_tracked "$CI"
populate_untracked_junk "$CI"
populate_untracked_models "$CI"

bash "$CLEANUP" --mode ci "$CI"

[[ -f "$CI/prebuilt" ]]
[[ -f "$CI/prebuilt.json" ]]
grep -Fq '"builder":"ci"' "$CI/prebuilt.json"
grep -Fq '"models":"device"' "$CI/prebuilt.json"
SOURCE_COMMIT="$(git -C "$CI" rev-parse HEAD)"
grep -Fq "\"source_commit\":\"$SOURCE_COMMIT\"" "$CI/prebuilt.json"
EXPECTED_INPUTS="$(git -C "$CI" rev-parse HEAD:openpilot/selfdrive/modeld HEAD:tinygrad_repo HEAD:openpilot/common/file_chunker.py | tr '\n' ':')"
grep -Fq "\"model_inputs\":\"$EXPECTED_INPUTS\"" "$CI/prebuilt.json"
# The tracked onnx inputs and libraries stay.
[[ -f "$CI/openpilot/selfdrive/modeld/models/driving_supercombo.onnx" ]]
[[ -f "$CI/third_party/raylib/larch64/libraylib.a" ]]
[[ -f "$CI/openpilot/selfdrive/modeld/models/README.md" ]]
# Every untracked model artifact is gone.
[[ ! -e "$CI/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl" ]]
[[ ! -e "$CI/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl.chunkmanifest" ]]
[[ ! -e "$CI/openpilot/selfdrive/modeld/models/tg_input_devices.json" ]]
[[ ! -e "$CI/openpilot/selfdrive/modeld/models/.build_stamp" && ! -e "$CI/openpilot/selfdrive/modeld/models/.big_model_build_stamp" ]]
[[ ! -e "$CI/openpilot/selfdrive/modeld/models/extra.onnx" ]]
[[ ! -e "$CI/Jenkinsfile" && ! -e "$CI/release" ]]
assert_junk_gone "$CI"

echo "ci cleanup passed"

# --- a file above GitHub's 95 MiB limit is rejected after the marker is written ---
BIG="$TMP_DIR/big"
make_repo "$BIG"
mkdir -p "$BIG/pkg"
dd if=/dev/zero of="$BIG/oversized.bin" bs=1048576 count=96 status=none
if bash "$CLEANUP" --mode device-release "$BIG" 2>"$TMP_DIR/big.err"; then
  echo "expected cleanup to reject a file larger than 95 MiB" >&2
  exit 1
fi
grep -q 'oversized.bin' "$TMP_DIR/big.err"
[[ -f "$BIG/prebuilt" ]]

printf 'prebuilt cleanup test passed\n'
