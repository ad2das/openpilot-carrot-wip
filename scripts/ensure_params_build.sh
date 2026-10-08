#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEADER="$ROOT/openpilot/common/params_keys.h"
MODULE="$ROOT/openpilot/common/params_pyx.so"
CACHE_DIR="${SCONS_CACHE_DIR:-/data/scons_cache}"

if ! mkdir -p "$CACHE_DIR" 2>/dev/null; then
  CACHE_DIR="/tmp/scons_cache"
  mkdir -p "$CACHE_DIR"
fi

STAMP="$CACHE_DIR/carrot_params_keys.sha256"
HEADER_HASH="$(sha256sum "$HEADER" | awk '{print $1}')"
BUILT_HASH="$(cat "$STAMP" 2>/dev/null || true)"

record_stamp() {
  printf '%s\n' "$HEADER_HASH" > "$STAMP.tmp"
  mv -f "$STAMP.tmp" "$STAMP"
}

if [ "$HEADER_HASH" = "$BUILT_HASH" ] && [ -f "$MODULE" ]; then
  exit 0
fi

# A prebuilt tree ships params_pyx.so compiled from this checkout's header but
# not the device-local stamp above, so the first boot after an update would
# otherwise drop the module and rebuild it with SCons. Trust the shipped module
# when its full Params registry still matches the header, and only record the
# stamp; any mismatch (or an unloadable module) falls through to the rebuild.
if [ -f "$ROOT/prebuilt" ] && [ -f "$MODULE" ] && \
   PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 "$ROOT/openpilot/system/manager/params_check.py"; then
  echo "Prebuilt Params module matches this checkout; recording the keys stamp."
  record_stamp
  exit 0
fi

echo "Params registry changed; rebuilding params_pyx.so."
rm -f \
  "$ROOT/openpilot/common/params.o" \
  "$ROOT/openpilot/common/params.os" \
  "$ROOT/openpilot/common/libcommon.a" \
  "$ROOT/openpilot/common/common.a" \
  "$MODULE"

cd "$ROOT"
scons -u -j4 openpilot/common/params_pyx.so
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -c \
  'from openpilot.common.params import Params; keys = Params().all_keys(); assert b"EnableRadarTracks" in keys and b"CarrotRadarCutInSensitivity" not in keys and b"CarrotRadarMode" not in keys and b"RadarMotionMode" not in keys and b"RadarDPathMode" not in keys and b"RadarLeadModelMode" not in keys'
printf '%s\n' "$HEADER_HASH" > "$STAMP.tmp"
mv -f "$STAMP.tmp" "$STAMP"
