#!/usr/bin/env bash
# Device-install simulation for the carrot-wip build artifact pipeline.
#
# Runs inside the AGNOS chroot on the artifact for this commit (already packed
# by scripts/ci_build_artifact_gate.py) and drives the real launcher functions
# with the scons/build entry points shimmed, so only the install and decision
# logic is exercised:
#
#   (a) clean checkout + artifact  -> install 0, ARTIFACT_OK=1, FORCE_REBUILD=0,
#       MODEL_REBUILD=1, exactly one model-only build
#   (b) plus stub QCOM model artifacts and a matching .build_stamp -> no build
#   (c) no artifact -> one full build (today's behaviour)
#   (d) /dev/ion absent or AGNOS_VERSION different -> install rejected -> full build
#   (e) one tampered file in the tarball -> install rejected -> full build
#   (f) git status after (a)/(b) shows only the tracked_regenerated paths modified
#
# The /dev/ion placeholder exists ONLY for this simulation and is removed again
# on exit; nothing here opens it.
set -Eeo pipefail

REPO=/data/openpilot
SHA=""
ARTIFACT_DIR=""
WORK=/data/ci_boot_sim

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --sha) SHA="$2"; shift 2 ;;
    --artifact-dir) ARTIFACT_DIR="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$SHA" && -n "$ARTIFACT_DIR" ]] || { echo "--sha and --artifact-dir are required" >&2; exit 2; }
[[ -f /TICI ]] || { echo "::error::simulation requires the /TICI marker" >&2; exit 1; }

TREE="$WORK/tree"
DATA="$WORK/data"
CACHE="$DATA/carrot_build_cache/$SHA"
LOG="$WORK/invocations.log"
STATUS_LOG="$WORK/git_status.log"
SIM_AGNOS_VERSION="${AGNOS_VERSION:-19.8-carrot-bt1}"
TAMPERED_VERSION="simulation-other-version"

fail() { echo "SIM FAIL: $*" >&2; exit 1; }

cleanup() {
  rm -f /dev/ion
  git -C "$REPO" worktree remove --force "$TREE" >/dev/null 2>&1 || true
  git -C "$REPO" worktree prune >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- simulation environment -------------------------------------------------
export AGNOS_VERSION="$SIM_AGNOS_VERSION"
export CARROT_DATA_DIR="$DATA"
export CARROT_BOOT_TIMING_LOG="$WORK/boot_timing.log"
export SCONS_CACHE_DIR="$WORK/scons_cache"
export PYDEPS="$REPO/pydeps"
# The simulated checkout must provide openpilot itself; pydeps backs the
# compiled dependencies, exactly like the launcher's own PYTHONPATH.
export PYTHONPATH="$TREE:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "$DATA" "$TREE" "$WORK/scons_cache"
rm -f "$WORK/invocations.log" "$WORK/git_status.log" "$WORK/boot_timing.log" "$WORK/scons.log" "$WORK/fastpath.stamp"
mkdir -p "$DATA" "$WORK/scons_cache"
touch /dev/ion

# The launcher bypasses the fast path when the stamp is absent; keep every run
# on the full path. carrot_build_if_needed reads this launcher global, which is
# otherwise only set by the real boot flow.
export CARROT_BOOT_STAMP_PATH="$WORK/fastpath.stamp"
fast_boot=0

source "$REPO/launch_env.sh"
source "$REPO/launch_chffrplus.sh"

# The launcher resolves DIR from its own location; point it at the simulated
# device checkout so the artifact install, Params check and model decisions all
# operate on $TREE, exactly like a device where DIR is the installed tree.
DIR="$TREE"

# Shim every startup command (scons and ./build.py run through these) and record
# the invocation instead of executing it.
run_startup_command() {
  printf 'STARTUP %s\n' "$*" >> "$LOG"
  return 0
}
scons() {
  printf 'SCONS %s\n' "$*" >> "$WORK/scons.log"
  return 0
}

build_calls() { grep -c '^STARTUP ' "$LOG" 2>/dev/null || true; }
startup_lines() { cat "$LOG" 2>/dev/null || true; }

reset_tree() {
  cd "$REPO"
  git worktree remove --force "$TREE" >/dev/null 2>&1 || true
  git worktree prune >/dev/null 2>&1 || true
  git worktree add --quiet --detach "$TREE" "$SHA"
  rm -rf "$DATA/carrot_build_artifacts"
  rm -f "$WORK/fastpath.stamp"
  : > "$LOG"
}

stage_artifact() {
  rm -rf "$CACHE"
  mkdir -p "$CACHE"
  cp "$ARTIFACT_DIR/$SHA.json" "$CACHE/$SHA.json"
  cp "$ARTIFACT_DIR/$SHA.tar.xz" "$CACHE/$SHA.tar.xz"
}

# Run the launcher's artifact preparation for the simulated checkout.
run_prepare() {
  ARTIFACT_OK=0
  unset CARROT_BUILD_ARTIFACT
  cd "$TREE"
  carrot_prepare_build_artifact
}

run_decisions() {
  FORCE_REBUILD=0
  MODEL_REBUILD=0
  cd "$TREE"
  invalidate_modeld_build_if_needed
  invalidate_native_build_if_needed
  carrot_build_if_needed
}

expect_build() {
  local label="$1" expected="$2"
  local calls
  calls="$(build_calls)"
  [[ "$calls" = "1" ]] || { startup_lines; fail "$label: expected exactly one build invocation, got $calls"; }
  grep -Fxq "STARTUP $expected" "$LOG" || { startup_lines; fail "$label: expected 'STARTUP $expected'"; }
}

expect_no_build() {
  local label="$1"
  local calls
  calls="$(build_calls)"
  [[ "$calls" = "0" ]] || { startup_lines; fail "$label: expected no build invocation, got $calls"; }
}

check_status_only_regenerated() {
  local label="$1" bad=0
  git -C "$TREE" status --porcelain --untracked-files=no > "$STATUS_LOG" || fail "$label: git status failed"
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    local path="${line:3}"
    local allowed=0
    while IFS= read -r entry; do
      [[ "$entry" == "$path" ]] && allowed=1
    done < <(sed 's/#.*//' "$TREE/scripts/build_artifact_tracked_outputs.txt" | awk 'NF{print $1}')
    if [[ "$allowed" != 1 ]]; then
      echo "SIM FAIL: $label: tracked modification not in the artifact allowlist: $line"
      bad=1
    fi
  done < "$STATUS_LOG"
  [[ "$bad" = 0 ]] || exit 1
  echo "SIM(f) $label: tracked status is limited to the allowlist:"
  sed 's/^/  /' "$STATUS_LOG"
}

echo "=== simulation environment ==="
uname -m
echo "sha:            $SHA"
echo "AGNOS_VERSION:  $AGNOS_VERSION"
echo "artifact:       $ARTIFACT_DIR"
echo "/TICI:          $(test -f /TICI && echo present || echo missing)"
echo "/dev/ion:       $(test -e /dev/ion && echo present || echo missing)"

# --- (a) clean checkout + artifact -----------------------------------------
echo "=== SIM(a): clean checkout + artifact -> model-only build ==="
reset_tree
stage_artifact
run_prepare
echo "SIM(a): ARTIFACT_OK=$ARTIFACT_OK"
[[ "$ARTIFACT_OK" = "1" ]] || fail "SIM(a): ARTIFACT_OK=$ARTIFACT_OK"
[[ "${CARROT_BUILD_ARTIFACT:-}" = "1" ]] || fail "SIM(a): CARROT_BUILD_ARTIFACT is not exported"
run_decisions
echo "SIM(a): FORCE_REBUILD=$FORCE_REBUILD MODEL_REBUILD=$MODEL_REBUILD"
[[ "$FORCE_REBUILD" = "0" ]] || fail "SIM(a): FORCE_REBUILD=$FORCE_REBUILD"
[[ "$MODEL_REBUILD" = "1" ]] || fail "SIM(a): MODEL_REBUILD=$MODEL_REBUILD"
expect_build "SIM(a)" "./build.py --targets openpilot/selfdrive/modeld/models"
check_status_only_regenerated "SIM(a)"
echo "SIM(a) PASS"

# --- (b) stub model artifacts + matching stamp ------------------------------
echo "=== SIM(b): QCOM model artifacts with the matching stamp -> no build ==="
MODELS="$TREE/openpilot/selfdrive/modeld/models"
touch "$MODELS/driving_tinygrad.pkl" "$MODELS/dmonitoring_model_tinygrad.pkl" "$MODELS/dmonitoring_model_metadata.pkl"
printf '%s' '{"openpilot.selfdrive.modeld.modeld": {"default": {"WARP_DEV": "QCOM", "QUEUE_DEV": "QCOM"}, "usbgpu": {"WARP_DEV": "QCOM", "QUEUE_DEV": "AMD"}}, "openpilot.selfdrive.modeld.dmonitoringmodeld": {"default": {"DEV": "QCOM"}}}' > "$MODELS/tg_input_devices.json"
( cd "$TREE" && git rev-parse HEAD:openpilot/selfdrive/modeld HEAD:tinygrad_repo HEAD:openpilot/common/file_chunker.py 2>/dev/null | tr '\n' ':' ) > "$MODELS/.build_stamp"
: > "$LOG"
run_decisions
echo "SIM(b): FORCE_REBUILD=$FORCE_REBUILD MODEL_REBUILD=$MODEL_REBUILD"
[[ "$FORCE_REBUILD" = "0" ]] || fail "SIM(b): FORCE_REBUILD=$FORCE_REBUILD"
[[ "$MODEL_REBUILD" = "0" ]] || fail "SIM(b): MODEL_REBUILD=$MODEL_REBUILD"
expect_no_build "SIM(b)"
check_status_only_regenerated "SIM(b)"
echo "SIM(b) PASS"

# --- (c) no artifact ---------------------------------------------------------
echo "=== SIM(c): no artifact -> full build (today's behaviour) ==="
reset_tree
rm -rf "$CACHE"
export CARROT_BUILD_ARTIFACT_URL="http://127.0.0.1:1/unpublished"
run_prepare
echo "SIM(c): ARTIFACT_OK=$ARTIFACT_OK"
[[ "$ARTIFACT_OK" = "0" ]] || fail "SIM(c): ARTIFACT_OK=$ARTIFACT_OK"
run_decisions
expect_build "SIM(c)" "./build.py"
echo "SIM(c) PASS"

# --- (d) /dev/ion absent or AGNOS_VERSION different -------------------------
echo "=== SIM(d1): /dev/ion absent -> install rejected -> full build ==="
reset_tree
stage_artifact
rm -f /dev/ion
run_prepare
echo "SIM(d1): ARTIFACT_OK=$ARTIFACT_OK"
[[ "$ARTIFACT_OK" = "0" ]] || fail "SIM(d1): ARTIFACT_OK=$ARTIFACT_OK"
run_decisions
expect_build "SIM(d1)" "./build.py"
echo "SIM(d1) PASS"

echo "=== SIM(d2): AGNOS_VERSION different -> install rejected -> full build ==="
reset_tree
stage_artifact
touch /dev/ion
export AGNOS_VERSION="$TAMPERED_VERSION"
run_prepare
echo "SIM(d2): ARTIFACT_OK=$ARTIFACT_OK"
[[ "$ARTIFACT_OK" = "0" ]] || fail "SIM(d2): ARTIFACT_OK=$ARTIFACT_OK"
run_decisions
expect_build "SIM(d2)" "./build.py"
export AGNOS_VERSION="$SIM_AGNOS_VERSION"
echo "SIM(d2) PASS"

# --- (e) one tampered file in the tarball ------------------------------------
echo "=== SIM(e): tampered tarball file -> install rejected -> full build ==="
reset_tree
stage_artifact
python3 - "$CACHE/$SHA.tar.xz" <<'PY'
import sys, tarfile, os
path = sys.argv[1]
tmp = path + ".tampered"
first = None
with tarfile.open(path, "r:xz") as src, tarfile.open(tmp, "w:xz", preset=6) as dst:
  for member in src:
    if not member.isfile() or member.name.endswith(".pyc"):
      continue
    data = src.extractfile(member).read()
    if first is None:
      first = member.name
      data += b"tampered"
    info = member
    dst.addfile(info, __import__("io").BytesIO(data))
if first is None:
  raise SystemExit("no file was tampered")
os.replace(tmp, path)
print(f"tampered {first}")
PY
run_prepare
echo "SIM(e): ARTIFACT_OK=$ARTIFACT_OK"
[[ "$ARTIFACT_OK" = "0" ]] || fail "SIM(e): ARTIFACT_OK=$ARTIFACT_OK"
run_decisions
expect_build "SIM(e)" "./build.py"
echo "SIM(e) PASS"

echo "ALL DEVICE-INSTALL SIMULATIONS PASSED"
