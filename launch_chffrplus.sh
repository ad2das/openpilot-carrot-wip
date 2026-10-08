#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source "$DIR/launch_env.sh"

function disable_automatic_git_maintenance {
  # Git fetch/pull can otherwise leave a detached repack running into a drive.
  # Inherit this policy in recovery, web, manager and their Git/submodule workers
  # without changing .git/config (which would invalidate the staging overlay).
  local config_count="${GIT_CONFIG_COUNT:-0}"
  local option
  for option in gc.auto=0 gc.autoDetach=false maintenance.auto=false; do
    export "GIT_CONFIG_KEY_${config_count}=${option%%=*}"
    export "GIT_CONFIG_VALUE_${config_count}=${option#*=}"
    config_count=$((config_count + 1))
  done
  export GIT_CONFIG_COUNT="$config_count"
}

disable_automatic_git_maintenance

# --- warm-boot timing + fast path -----------------------------------------
# The fast path skips dependency bootstrap, SCons and the Params checks when
# the checkout fingerprint matches the stamp written by the previous full
# successful boot. Everything here must stay cheap and must never break boot.

function carrot_data_dir {
  printf '%s\n' "${CARROT_DATA_DIR:-/data}"
}

function carrot_timing_now {
  local now
  now="$(date +%s.%3N 2>/dev/null || true)"
  if [ -z "$now" ]; then
    now="$(date +%s 2>/dev/null || true)"
  fi
  [ -n "$now" ] || now="0"
  printf '%s\n' "$now"
}

function carrot_boot_timing_path {
  if [ -z "${CARROT_BOOT_TIMING_LOG:-}" ]; then
    local data_dir candidate
    data_dir="$(carrot_data_dir)"
    candidate="$data_dir/carrot_boot_timing.log"
    if mkdir -p "$data_dir" 2>/dev/null && { [ -e "$candidate" ] || : >> "$candidate" 2>/dev/null; }; then
      CARROT_BOOT_TIMING_LOG="$candidate"
    else
      CARROT_BOOT_TIMING_LOG="/tmp/carrot_boot_timing.log"
    fi
    export CARROT_BOOT_TIMING_LOG
  fi
  printf '%s\n' "$CARROT_BOOT_TIMING_LOG"
}

function boot_timing {
  printf '%s %s\n' "$(carrot_timing_now)" "$1" >> "$(carrot_boot_timing_path)" 2>/dev/null || true
}

function boot_timing_start {
  local log_path uptime="unknown"
  log_path="$(carrot_boot_timing_path)"
  if [ -f "$log_path" ] && command -v tail >/dev/null 2>&1; then
    if tail -n 400 "$log_path" > "$log_path.tmp" 2>/dev/null; then
      mv -f "$log_path.tmp" "$log_path" 2>/dev/null || rm -f "$log_path.tmp"
    else
      rm -f "$log_path.tmp"
    fi
  fi
  if [ -r /proc/uptime ]; then
    uptime="$(cut -d' ' -f1 /proc/uptime 2>/dev/null || true)"
    [ -n "$uptime" ] || uptime="unknown"
  fi
  printf '%s boot_start uptime=%s\n' "$(carrot_timing_now)" "$uptime" >> "$log_path" 2>/dev/null || true
}

function carrot_boot_stamp_path {
  printf '%s\n' "${CARROT_BOOT_STAMP_PATH:-$(carrot_data_dir)/carrot_boot_fastpath.stamp}"
}

function carrot_boot_force_path {
  printf '%s\n' "${CARROT_FORCE_FULL_BOOT_FILE:-$(carrot_data_dir)/carrot_force_full_boot}"
}

function carrot_boot_fingerprint {
  python3 "$DIR/scripts/carrot_boot_fingerprint.py" --root "$DIR" --pydeps "$PYDEPS" 2>/dev/null | tail -n 1
}

# Prints either "fast" or "full <reason>" and never fails.
function carrot_fastpath_decision {
  local stamp_path force_path fingerprint
  stamp_path="$(carrot_boot_stamp_path)"
  force_path="$(carrot_boot_force_path)"

  if [ "${CARROT_FULL_BOOT:-}" = "1" ]; then
    printf 'full CARROT_FULL_BOOT=1\n'
    return 0
  fi
  if [ -e "$force_path" ]; then
    printf 'full force-file\n'
    return 0
  fi
  if [ ! -f "$stamp_path" ]; then
    printf 'full stamp-missing\n'
    return 0
  fi

  fingerprint="$(carrot_boot_fingerprint || true)"
  if [ -z "$fingerprint" ]; then
    printf 'full fingerprint-error\n'
    return 0
  fi
  if [ "$fingerprint" != "$(cat "$stamp_path" 2>/dev/null)" ]; then
    printf 'full fingerprint-mismatch\n'
    return 0
  fi
  case "$fingerprint" in
    *" ready=1") printf 'fast\n' ;;
    *) printf 'full inputs-not-ready\n' ;;
  esac
}

function carrot_drop_fastpath_stamp {
  rm -f "$(carrot_boot_stamp_path)" 2>/dev/null || true
}

function carrot_write_fastpath_stamp {
  local fingerprint stamp_path
  fingerprint="$(carrot_boot_fingerprint || true)"
  [ -n "$fingerprint" ] || return 0
  # Only a boot that produced every native input may opt the next boot into
  # the fast path; otherwise stay on the full path until the build is healthy.
  case "$fingerprint" in
    *" ready=1") ;;
    *) return 0 ;;
  esac
  stamp_path="$(carrot_boot_stamp_path)"
  if printf '%s\n' "$fingerprint" > "$stamp_path.tmp" 2>/dev/null && \
     mv -f "$stamp_path.tmp" "$stamp_path" 2>/dev/null; then
    rm -f "$(carrot_boot_force_path)" 2>/dev/null || true
  fi
}

function cleanup_stale_git_lfs_hooks {
  # Some deployed checkouts still contain hooks installed by git-lfs even
  # though the executable is no longer part of the device image. Those hooks
  # make an otherwise successful pull/checkout fail at the final hook step.
  command -v git-lfs >/dev/null 2>&1 && return

  local git_dir
  local hook
  local hook_path

  git_dir="$(git -C "$DIR" rev-parse --absolute-git-dir 2>/dev/null || true)"
  [ -n "$git_dir" ] || return

  for hook in post-checkout post-commit post-merge pre-push; do
    hook_path="$git_dir/hooks/$hook"
    # Delete only hooks generated by git-lfs. Preserve every unrelated local
    # hook, including similarly named custom hooks.
    if [ -f "$hook_path" ] && grep -q "command -v git-lfs" "$hook_path" && grep -q "git lfs $hook" "$hook_path"; then
      echo "Removing stale git-lfs hook: $hook"
      rm -f -- "$hook_path"
    fi
  done
}

function install_runtime_python_package {
  # uv-created Ubuntu environments may intentionally have no pip module. Use
  # the launcher's interpreter and pydeps target with either installer.
  if python3 -m pip --version > /dev/null 2>&1; then
    python3 -m pip install --disable-pip-version-check --no-input --timeout 15 --retries 2 \
      --target "$PYDEPS" --upgrade "$@"
  elif command -v uv > /dev/null 2>&1; then
    UV_HTTP_TIMEOUT=15 UV_HTTP_RETRIES=2 uv pip install --python "$(command -v python3)" --no-python-downloads \
      --target "$PYDEPS" --upgrade "$@"
  else
    echo "pip is missing; attempting to bootstrap it with ensurepip."
    if python3 -m ensurepip --upgrade && python3 -m pip --version > /dev/null 2>&1; then
      python3 -m pip install --disable-pip-version-check --no-input --timeout 15 --retries 2 \
        --target "$PYDEPS" --upgrade "$@"
    else
      echo "No Python package installer is available. Run tools/setup.sh to prepare the Ubuntu environment."
      return 1
    fi
  fi
}

function ensure_python_package {
  local import_name="$1"
  local package_name="$2"
  local required="${3:-0}"
  local install_dependencies="${4:-0}"
  local wheel_dir="$DIR/third_party/wheels"

  if python3 -c "import ${import_name}" > /dev/null 2>&1; then
    echo "${package_name} already installed."
    return 0
  fi

  echo "${package_name} installing from local wheel."
  if [ "$install_dependencies" = "1" ]; then
    install_runtime_python_package --no-index --find-links "$wheel_dir" "$package_name"
  else
    install_runtime_python_package --no-index --no-deps --find-links "$wheel_dir" "$package_name"
  fi
  if [ "$?" = "0" ] && \
     python3 -c "import ${import_name}" > /dev/null 2>&1; then
    echo "${package_name} installed."
    return 0
  fi

  # Bundled native wheels target AGNOS/aarch64. Desktop Ubuntu may need wheels
  # for another architecture or Python version; allow its installer to fetch
  # compatible packages and dependencies. Vehicle startup stays offline.
  if [ ! -f /TICI ] && [ ! -f /AGNOS ]; then
    echo "${package_name} local installation failed; trying the online package index."
    if install_runtime_python_package "$package_name" && \
       python3 -c "import ${import_name}" > /dev/null 2>&1; then
      echo "${package_name} installed."
      return 0
    fi
  fi

  # Keep the actual import error visible when installation could not repair it.
  python3 -c "import ${import_name}" >&2
  if [ "$required" = "1" ]; then
    echo "Required Python package ${package_name} is unavailable; not starting openpilot."
    return 1
  fi

  echo "Optional Python package ${package_name} is unavailable; continuing without it."
  return 0
}

function bootstrap_runtime_dependencies {
  if ! ensure_python_package serial pyserial 1 || \
     ! ensure_python_package msgpack msgpack 1 || \
     ! ensure_python_package aiohttp aiohttp 1 1 || \
     ! ensure_python_package "aiortc; import av; import pylibsrtp" "aiortc==1.14.0" 1 1 || \
     ! ensure_python_package psutil psutil 1 || \
     ! ensure_python_package crcmod crcmod-plus 1 || \
     ! ensure_python_package jsonrpc json-rpc 1 || \
     ! ensure_python_package qrcode qrcode 1; then
    return 1
  fi

  ensure_python_package brotli brotli 0
  ensure_python_package usb pyusb 0

  # Xiaoge lane/BSD inference uses a pinned, bundled OpenCV wheel. Keep NumPy
  # from AGNOS and install only into pydeps, never the read-only system venv.
  # Prepare it even when ShareData is off so the settings toggle works offline.
  ensure_python_package "cv2; assert cv2.__version__ == '4.13.0'; assert hasattr(cv2.dnn, 'readNetFromONNX')" \
    "opencv-python-headless==4.13.0.92" 0

  # AGNOS 19 follows current comma, which no longer includes the legacy Eigen
  # and libjpeg wrappers used by this branch's rednose and JPEG encoder. Keep
  # those native headers/libraries available from official offline wheels.
  if { [ -f /TICI ] || [ -f /AGNOS ]; } && \
     { ! ensure_python_package eigen eigen 1 || ! ensure_python_package libjpeg libjpeg 1; }; then
    return 1
  fi

  # MPC headers, templates, native libraries and CasADi now come from one wheel.
  if ! ensure_python_package "acados; from acados.acados_template import AcadosOcpSolver; from casadi import SX" \
      "comma-deps-acados==0.2.2.post103" 1; then
    return 1
  fi

  if ! ensure_python_package json11 "comma-deps-json11==20170411.0.post103" 1; then
    return 1
  fi

  # Development checkouts build C++ tests by default, matching SConstruct extras.
  if [ -f "$DIR/.gitattributes" ] && ! ensure_python_package catch2 "comma-deps-catch2==2.13.10.post96" 1; then
    return 1
  fi

  ensure_python_package shapely shapely 0
}

function show_agnos_update_failure {
  local reason="$1"
  local message

  printf -v message 'AGNOS update paused.\n\n%s\n\nOpen the recovery address shown above (port 6999).\nLog: /data/agnos-updater-ui.log\n\nTap Reboot to try the updater again.' "$reason"
  printf '%s\n' "$message"
  python3 "$DIR/openpilot/system/ui/text.py" "$message" || true
}

function agnos_init {
  # TODO: move this to agnos
  sudo rm -f /data/etc/NetworkManager/system-connections/*.nmmeta
  rm -f /data/scons_cache/config.lock

  sudo abctl --set_success

  # TODO: do this without udev in AGNOS
  # udev does this, but sometimes we startup faster
  sudo chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
  sudo chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0

  export AGNOS_UPDATE_CONFIRMATION_FILE="${AGNOS_UPDATE_CONFIRMATION_FILE:-/data/agnos-update-confirmed}"

  # Check if AGNOS update is required
  if [ "$(< /VERSION)" != "$AGNOS_VERSION" ]; then
    AGNOS_PY="$DIR/openpilot/system/hardware/tici/agnos.py"
    MANIFEST="$DIR/openpilot/system/hardware/tici/agnos.json"
    MODEL="$(tr -d '\000\r\n' 2>/dev/null < /sys/firmware/devicetree/base/model | tr '[:upper:]' '[:lower:]')"
    MODEL="${MODEL#comma }"
    if [ "$MODEL" = "c3" ] || [ "$MODEL" = "tici" ]; then
      MANIFEST="$DIR/openpilot/system/hardware/tici/agnos-tici.json"
    fi
    echo "AGNOS_PY=${AGNOS_PY}"
    echo "MANIFEST=${MANIFEST}"
    echo "MODEL=${MODEL}"

    # A completed inactive slot only needs activation and a reboot. Check this
    # before any network or standalone-UI dependency work.
    if python3 "$AGNOS_PY" --verify "$MANIFEST"; then
      echo "Verified AGNOS update activated; rebooting."
      if sudo reboot; then
        while true; do sleep 1; done
      fi
      show_agnos_update_failure "The updated slot is ready, but the reboot command failed."
      return 1
    fi

    if python3 -c "import jeepney" > /dev/null 2>&1; then
      echo "jeepney already installed."
    else
      echo "Installing the AGNOS updater UI dependency."
      if ! python3 -m pip install --target "$PYDEPS" --upgrade --timeout 15 --retries 2 jeepney; then
        show_agnos_update_failure "The updater UI dependency could not be installed. Check the network, then retry."
        return 1
      fi
    fi

    local AGNOS_UI_LOG="/data/agnos-updater-ui.log"
    local ui_result
    : > "$AGNOS_UI_LOG"
    for attempt in 1 2 3; do
      echo "Starting AGNOS updater UI (${attempt}/3)." | tee -a "$AGNOS_UI_LOG"
      python3 "$DIR/openpilot/system/ui/updater.py" "$AGNOS_PY" "$MANIFEST" 2>&1 | tee -a "$AGNOS_UI_LOG"
      ui_result=${PIPESTATUS[0]}
      echo "AGNOS updater UI exited with status ${ui_result}." | tee -a "$AGNOS_UI_LOG"
      sleep 2
    done

    show_agnos_update_failure "The automatic updater stopped three times. Check the updater log for the failure."
    return 1
  else
    rm -f "$AGNOS_UPDATE_CONFIRMATION_FILE"
  fi
}

function start_carrot_recovery {
  local recovery_script="$DIR/openpilot/selfdrive/carrot/recovery/server.py"
  local py_bin

  [ -f "$recovery_script" ] || return
  py_bin="$(command -v python3 || command -v python || true)"
  [ -n "$py_bin" ] || return

  if command -v pgrep >/dev/null 2>&1 && pgrep -f "openpilot/selfdrive/carrot/recovery/server.py" >/dev/null 2>&1; then
    return
  fi

  echo "Starting carrot recovery server on 6999."
  (cd "$DIR" && "$py_bin" "$recovery_script" --port 6999 >> /tmp/carrot_recovery.log 2>&1 &)
}

function start_carrot_web {
  [ "$CARROT_WEB_EXTERNAL" = "1" ] || return

  local watchdog_script="$DIR/scripts/carrot_web_watchdog.sh"
  local pid_file="${CARROT_WEB_PID_FILE:-/tmp/carrot_web_watchdog.pid}"
  local py_bin

  [ -f "$watchdog_script" ] || return

  # The watchdog survives tmux/openpilot restarts. The pid file can be lost or
  # replaced while that old process is still alive, so also check the process
  # table before starting another watchdog.
  if command -v pgrep >/dev/null 2>&1 && pgrep -f '[c]arrot_web_watchdog[.]sh' >/dev/null 2>&1; then
    return
  fi

  if [ -f "$pid_file" ]; then
    local old_pid
    old_pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" >/dev/null 2>&1; then
      return
    fi
  fi

  py_bin="$(command -v python3 || command -v python || true)"
  [ -n "$py_bin" ] || return

  echo "Starting external carrot web server on 7000."
  if command -v setsid >/dev/null 2>&1; then
    setsid bash "$watchdog_script" "$DIR" "$py_bin" >> /tmp/carrot_server.log 2>&1 &
  else
    bash "$watchdog_script" "$DIR" "$py_bin" >> /tmp/carrot_server.log 2>&1 &
  fi
}

function big_model_artifact_ready {
  python3 -c 'from openpilot.selfdrive.modeld.helpers import active_usbgpu_compiled_path; raise SystemExit(0 if active_usbgpu_compiled_path() is not None else 1)' 2>/dev/null
}

function invalidate_modeld_build_if_needed {
  local stamp_path="$DIR/openpilot/selfdrive/modeld/models/.build_stamp"
  local big_stamp_path="$DIR/openpilot/selfdrive/modeld/models/.big_model_build_stamp"
  local tg_devices_path="$DIR/openpilot/selfdrive/modeld/models/tg_input_devices.json"
  local driving_pkl_path="$DIR/openpilot/selfdrive/modeld/models/driving_tinygrad.pkl"
  local old_stamp
  local old_big_stamp

  MODEL_BUILD_STAMP_VALUE="$(git rev-parse HEAD:openpilot/selfdrive/modeld HEAD:tinygrad_repo HEAD:openpilot/common/file_chunker.py 2>/dev/null | tr '\n' ':')"
  if [ -z "$MODEL_BUILD_STAMP_VALUE" ]; then
    MODEL_BUILD_STAMP_VALUE="$(git rev-parse HEAD 2>/dev/null || true)"
  fi

  old_stamp="$(cat "$stamp_path" 2>/dev/null || true)"
  if [ ! -f "$tg_devices_path" ] || { [ ! -f "$driving_pkl_path" ] && [ ! -f "$driving_pkl_path.chunkmanifest" ]; }; then
    echo "Model/tinygrad artifacts are missing; revalidating with SCons."
    # Keep generated artifacts. SCons tracks the compiler, tinygrad and model
    # dependencies and will rebuild only stale targets. Deleting everything
    # here caused unrelated modeld changes to trigger long full recompiles.
    FORCE_REBUILD=1
  elif [ "$MODEL_BUILD_STAMP_VALUE" != "$old_stamp" ]; then
    # The stamp lives inside the hashed modeld tree, so a prebuilt tree cannot
    # ship it (the value would have to describe itself). Trust the shipped
    # artifacts when they are present and refresh the stamp instead; a source
    # checkout keeps forcing the SCons revalidation on any mismatch.
    if [ -f "$DIR/prebuilt" ]; then
      if mkdir -p "$(dirname "$stamp_path")" && \
         printf '%s' "$MODEL_BUILD_STAMP_VALUE" > "$stamp_path.tmp" 2>/dev/null && \
         mv -f "$stamp_path.tmp" "$stamp_path" 2>/dev/null; then
        echo "Prebuilt model artifacts present; refreshed the model build stamp."
      else
        echo "Could not record the model build stamp; revalidating with SCons."
        FORCE_REBUILD=1
      fi
    else
      echo "Model/tinygrad inputs changed or artifacts are missing; revalidating with SCons."
      FORCE_REBUILD=1
    fi
  fi

  if [ -n "$BIG_MODEL_SHA" ]; then
    old_big_stamp="$(cat "$big_stamp_path" 2>/dev/null || true)"
    if [ "$BIG_MODEL_SHA" != "$old_big_stamp" ] || ! big_model_artifact_ready; then
      echo "USB eGPU big model changed or needs compilation."
      FORCE_REBUILD=1
    fi
  fi
}

function prepare_big_model_if_needed {
  BIG_MODEL_SHA=""
  unset CARROT_BIG_MODEL_STARTUP_FAILED

  # With an attached eGPU, install this checkout's selection before building or
  # starting modeld. Offline delivery retries here instead of running an old model.
  if ! python3 -c 'from openpilot.selfdrive.modeld.helpers import usbgpu_present; raise SystemExit(0 if usbgpu_present() else 1)' 2>/dev/null; then
    return
  fi

  if ! flock /tmp/big_model_update.lock python3 -m openpilot.selfdrive.modeld.big_model --prepare-for-startup --retry-network; then
    echo "Selected eGPU model preparation failed; using the internal model this boot."
    export CARROT_BIG_MODEL_STARTUP_FAILED=1
    return
  fi
  BIG_MODEL_SHA="$(python3 -m openpilot.selfdrive.modeld.big_model --active-sha 2>/dev/null || true)"

  # Do not reject compilation from a one-shot 12V check here. During ignition
  # startup the USB bridge can enumerate before switched GPU power and PCIe are
  # ready. Mark the missing target stale and let build_usbgpu_model perform its
  # retrying readiness check while the build screen remains visible.
}

function start_big_model_update {
  local log_path="/tmp/big_model_update.log"
  local lock_path="/tmp/big_model_update.lock"

  if command -v pgrep >/dev/null 2>&1 && pgrep -f '[o]penpilot.selfdrive.modeld.big_model --ensure-if-egpu' >/dev/null 2>&1; then
    return
  fi

  echo "Checking optional eGPU model in background (log: ${log_path})."
  if command -v flock >/dev/null 2>&1; then
    (
      exec 9>"$lock_path"
      flock -n 9 || exit 0
      if command -v ionice >/dev/null 2>&1; then
        PYTHONUNBUFFERED=1 ionice -c 3 nice -n 10 python3 -m openpilot.selfdrive.modeld.big_model --ensure-if-egpu --network-wait-seconds 60 --retry-network
      else
        PYTHONUNBUFFERED=1 nice -n 10 python3 -m openpilot.selfdrive.modeld.big_model --ensure-if-egpu --network-wait-seconds 60 --retry-network
      fi
    ) >> "$log_path" 2>&1 &
  else
    if command -v ionice >/dev/null 2>&1; then
      PYTHONUNBUFFERED=1 ionice -c 3 nice -n 10 python3 -m openpilot.selfdrive.modeld.big_model --ensure-if-egpu --network-wait-seconds 60 --retry-network >> "$log_path" 2>&1 &
    else
      PYTHONUNBUFFERED=1 nice -n 10 python3 -m openpilot.selfdrive.modeld.big_model --ensure-if-egpu --network-wait-seconds 60 --retry-network >> "$log_path" 2>&1 &
    fi
  fi
}

function invalidate_native_build_if_needed {
  local missing=0
  local path

  for path in "$DIR/openpilot/system/loggerd/loggerd" "$DIR/openpilot/system/loggerd/encoderd" "$DIR/openpilot/system/camerad/camerad"; do
    if [ ! -x "$path" ]; then
      echo "Missing native binary: $path"
      missing=1
    fi
  done

  if [ "$missing" = "1" ]; then
    FORCE_REBUILD=1
  fi

  # A prebuilt checkout can retain params_pyx.so from before new keys were
  # added. Check the loaded registry, not just the presence of native binaries.
  if ! python3 "$DIR/openpilot/system/manager/params_check.py"; then
    FORCE_REBUILD=1
  fi
}

function start_manager {
  # A warm start or a long build can leave the launcher on an isolated CPU.
  # Set the manager's initial mask before Python creates threads or forks:
  # ordinary services share CPUs 0-5; camera/model/control keep their explicit
  # affinity overrides. The compiler can still use all eight CPUs separately.
  if [ -f /AGNOS ]; then
    taskset -c 0-5 ./manager.py
  else
    ./manager.py
  fi
}

function carrot_fast_boot_skip_command {
  # Commands that the last full healthy boot already validated; on the fast
  # path they must not run at all.
  [ "${CARROT_FAST_BOOT:-0}" = "1" ] || return 1
  case "$*" in
    *bootstrap_runtime_dependencies*|*ensure_params_build.sh*|*./build.py*|*params_check.py*) return 0 ;;
  esac
  return 1
}

function run_startup_command {
  if carrot_fast_boot_skip_command "$@"; then
    return 0
  fi
  case "$*" in
    *bootstrap_runtime_dependencies*) boot_timing deps ;;
    *ensure_params_build.sh*) boot_timing params_build ;;
    *./build.py*) boot_timing scons ;;
    *params_check.py*) boot_timing params_check ;;
  esac
  "$@" 2>&1 | python3 -m openpilot.common.startup_recovery --capture-log /tmp/carrot_startup_failure.log
  return "${PIPESTATUS[0]}"
}

function show_startup_failure {
  local reason="$1"
  # No build/manager can still hold the checkout when the update button runs.
  flock -u 9
  exec 9>&-
  unset CARROT_BOOT_LOCK_FD
  export CARROT_STARTUP_RECOVERY=1
  start_carrot_recovery
  while true; do
    python3 "$DIR/openpilot/system/ui/startup_recovery.py" --reason "$reason" --log /tmp/carrot_startup_failure.log
    echo "Recovery display stopped; checking for a fix without graphics. Web recovery remains on port 6999."
    python3 -m openpilot.common.startup_recovery --repo "$DIR"
    sleep 30
  done
}

function launch {
  boot_timing_start
  # Protect the checkout throughout bootstrap, SCons and manager initialization.
  # The manager releases this inherited flock after init; background web/recovery
  # servers must not inherit it. Never delete the lock file itself.
  export CARROT_REPO_LOCK_PATH="${CARROT_REPO_LOCK_PATH:-/tmp/carrot_repo_update.lock}"
  exec 9>"$CARROT_REPO_LOCK_PATH"
  if ! flock -w 300 9; then
    echo "Another repository operation is still running; launch deferred."
    exec 9>&-
    start_carrot_recovery
    while true; do sleep 1; done
  fi
  export CARROT_BOOT_LOCK_FD=9
  # The launcher owns startup failure display, including Python import errors.
  export CARROT_STARTUP_RECOVERY=1
  cleanup_stale_git_lfs_hooks

  # Check to see if there's a valid overlay-based update available. Conditions
  # are as follows:
  #
  # 1. The DIR init file has to exist, with a newer modtime than anything in
  #    the DIR Git repo. This checks for local development work or the user
  #    switching branches/forks, which should not be overwritten.
  # 2. The FINALIZED consistent file has to exist, indicating there's an update
  #    that completed successfully and synced to disk.

  if [ -f "${DIR}/.overlay_init" ]; then
    find ${DIR}/.git -newer ${DIR}/.overlay_init | grep -q '.' 2> /dev/null
    if [ $? -eq 0 ]; then
      echo "${DIR} has been modified, skipping overlay update installation"
    else
      if [ -f "${STAGING_ROOT}/finalized/.overlay_consistent" ]; then
        if [ ! -d /data/safe_staging/old_openpilot ]; then
          echo "Valid overlay update found, installing"
          LAUNCHER_LOCATION="${BASH_SOURCE[0]}"

          mv $DIR /data/safe_staging/old_openpilot
          mv "${STAGING_ROOT}/finalized" $DIR
          cd $DIR

          echo "Restarting launch script ${LAUNCHER_LOCATION}"
          unset AGNOS_VERSION
          exec "${LAUNCHER_LOCATION}"
        else
          echo "openpilot backup found, not updating"
          # TODO: restore backup? This means the updater didn't start after swapping
        fi
      fi
    fi
  fi

  # handle pythonpath
  ln -sfn $(pwd) /data/pythonpath
  PYDEPS="$DIR/pydeps"
  mkdir -p "$PYDEPS"
  export PYTHONPATH="$PYDEPS:$PWD${PYTHONPATH:+:$PYTHONPATH}"

  # Keep recovery access available even when an OS update or source build fails.
  if [ ! -f /data/params/d/GithubSshKeys ]; then
    echo -n openpilot > /data/params/d/GithubUsername
    cat /usr/comma/setup_keys > /data/params/d/GithubSshKeys
  fi
  if [ "$(cat /data/params/d/SshEnabled 2>/dev/null)" != "1" ]; then
    echo -n 1 > /data/params/d/SshEnabled
  fi
  (
    exec 9>&-
    unset CARROT_BOOT_LOCK_FD
    start_carrot_recovery
  )

  # hardware specific init
  if [ -f /AGNOS ]; then
    if ! agnos_init; then
      flock -u 9
      while true; do sleep 1; done
    fi
    # Offline wall-clock floor after any OS update, before dependency/bootstrap
    # or Params SCons work. GPS timed starts only after the main build.
    if ! run_startup_command python3 "$DIR/openpilot/common/build_time.py"; then
      show_startup_failure "System clock correction failed before build"
    fi
  fi

  # Warm-boot fast path: an exactly matching stamp means this checkout already
  # completed a full, healthy boot, so dependency bootstrap, the SCons
  # dependency walk and the Params checks can be skipped. Any mismatch,
  # failure or explicit force falls back to the full path.
  local fast_boot=0
  local fast_decision
  fast_decision="$(carrot_fastpath_decision)"
  case "$fast_decision" in
    fast)
      fast_boot=1
      export CARROT_FAST_BOOT=1
      boot_timing "fastpath=1"
      ;;
    *)
      boot_timing "fastpath=0 reason=${fast_decision#full }"
      ;;
  esac

  # AGNOS must be current before installing its matching offline wheels. SCons
  # imports native dependency modules while building Params, so bootstrap them
  # before the first SCons invocation. run_startup_command skips these on the
  # fast path via CARROT_FAST_BOOT.
  if ! run_startup_command bootstrap_runtime_dependencies; then
    carrot_drop_fastpath_stamp
    show_startup_failure "Runtime dependency installation failed"
  fi

  # Build Params before any long-running carrot service imports it.
  if ! run_startup_command bash "$DIR/scripts/ensure_params_build.sh"; then
    echo "Params registry build failed, not starting openpilot."
    carrot_drop_fastpath_stamp
    show_startup_failure "Params registry build failed"
  fi

  # Export in the parent so manager also knows the external watchdog owns the
  # web server. An export inside the subshell never reaches manager and causes
  # a second carrot_server to crash repeatedly on the occupied port 7000.
  export CARROT_WEB_EXTERNAL="${CARROT_WEB_EXTERNAL:-1}"
  (
    exec 9>&-
    unset CARROT_BOOT_LOCK_FD
    start_carrot_web
  )


  FORCE_REBUILD=0
  if [ "$fast_boot" = "0" ]; then
    boot_timing big_model_probe
    prepare_big_model_if_needed
    boot_timing native_check
    invalidate_modeld_build_if_needed
    invalidate_native_build_if_needed
  fi

  rm -f openpilot/selfdrive/pandad/*.so
  # write tmux scrollback to a file
  tmux capture-pane -pq -S-1500 > /tmp/launch_log

  # start manager
  cd openpilot/system/manager
  if [ "$fast_boot" = "0" ] && { [ "$FORCE_REBUILD" = "1" ] || [ ! -f $DIR/prebuilt ]; }; then
    if ! run_startup_command ./build.py; then
      echo "openpilot build failed, not starting manager."
      carrot_drop_fastpath_stamp
      show_startup_failure "openpilot build failed"
    fi
    if [ "$FORCE_REBUILD" = "1" ]; then
      mkdir -p "$DIR/openpilot/selfdrive/modeld/models"
      echo -n "$MODEL_BUILD_STAMP_VALUE" > "$DIR/openpilot/selfdrive/modeld/models/.build_stamp"
      if [ -n "$BIG_MODEL_SHA" ] && big_model_artifact_ready; then
        echo -n "$BIG_MODEL_SHA" > "$DIR/openpilot/selfdrive/modeld/models/.big_model_build_stamp"
      fi
    fi
  fi
  # Never start driving services if a rebuild left the Params registry stale.
  if ! run_startup_command python3 "$DIR/openpilot/system/manager/params_check.py"; then
    echo "Native Params still do not match this checkout; not starting manager."
    carrot_drop_fastpath_stamp
    show_startup_failure "Native Params do not match this checkout"
  fi
  if [ "$fast_boot" = "0" ]; then
    # All build steps succeeded: remember this exact checkout so the next boot
    # can take the fast path. The build has finished, so the fingerprint now
    # reflects the built artifacts.
    carrot_write_fastpath_stamp
  fi
  start_big_model_update
  boot_timing manager_exec
  if ! run_startup_command start_manager; then
    carrot_drop_fastpath_stamp
    show_startup_failure "Manager failed to start"
  fi
  # Also release if manager failed before reaching main()/initialization.
  flock -u 9
  exec 9>&-

  # if broken, keep on screen error
  while true; do sleep 1; done
}

# Run only when executed, so tests can source the functions above.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  launch "$@"
fi
