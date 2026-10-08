#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CLEANUP="$SCRIPT_DIR/../scripts/prebuilt_cleanup.sh"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT

WORK="$TMP_DIR/tree"
mkdir -p "$WORK/openpilot/selfdrive/modeld/models" \
         "$WORK/release/tests" \
         "$WORK/pkg/sub" \
         "$WORK/pkg/__pycache__" \
         "$WORK/pkg/sub/__pycache__"
touch "$WORK/openpilot/selfdrive/modeld/models/driving.onnx"
touch "$WORK/openpilot/selfdrive/modeld/models/keep.pkl"
touch "$WORK/release/build_carrot.sh" "$WORK/Jenkinsfile"
touch "$WORK/pkg/a.o" "$WORK/pkg/b.a" "$WORK/pkg/c.os" "$WORK/pkg/d.pyc"
touch "$WORK/pkg/sub/moc_widget.cpp"
touch "$WORK/pkg/__pycache__/x.pyc" "$WORK/pkg/sub/__pycache__/y.pyc"
touch "$WORK/.sconsign.dblite"
mkdir -p "$WORK/pkg/sub"
touch "$WORK/pkg/sub/.sconsign.dblite"

bash "$CLEANUP" "$WORK"

[[ -f "$WORK/prebuilt" ]]
[[ ! -e "$WORK/openpilot/selfdrive/modeld/models/driving.onnx" ]]
[[ -f "$WORK/openpilot/selfdrive/modeld/models/keep.pkl" ]]
[[ ! -e "$WORK/release" && ! -e "$WORK/Jenkinsfile" ]]
[[ ! -e "$WORK/.sconsign.dblite" ]]
# Only the tree-root .sconsign.dblite is removed, matching the historical rule.
[[ -e "$WORK/pkg/sub/.sconsign.dblite" ]]
[[ ! -e "$WORK/pkg/a.o" && ! -e "$WORK/pkg/b.a" && ! -e "$WORK/pkg/c.os" && ! -e "$WORK/pkg/d.pyc" ]]
[[ ! -e "$WORK/pkg/sub/moc_widget.cpp" ]]
[[ ! -e "$WORK/pkg/__pycache__" && ! -e "$WORK/pkg/sub/__pycache__" ]]

# A file above GitHub's 95 MiB limit must be rejected, after the prebuilt
# marker is written (same order as the historical on-device release rule).
BIG="$TMP_DIR/big"
mkdir -p "$BIG"
dd if=/dev/zero of="$BIG/oversized.bin" bs=1048576 count=96 status=none
if bash "$CLEANUP" "$BIG" 2>"$TMP_DIR/big.err"; then
  echo "expected cleanup to reject a file larger than 95 MiB" >&2
  exit 1
fi
grep -q 'oversized.bin' "$TMP_DIR/big.err"
[[ -f "$BIG/prebuilt" ]]

printf 'prebuilt cleanup test passed\n'
