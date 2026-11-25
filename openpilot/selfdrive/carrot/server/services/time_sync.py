import datetime
import os
import subprocess
import time
from typing import Any


TIME_SYNC_THRESHOLD_SEC = 10
TIME_SYNC_DEBUG_DEFAULT = True
ZONEINFO_ROOT = "/usr/share/zoneinfo"
SUBPROCESS_TIMEOUT_SEC = 10


def resolve_zoneinfo_path(timezone_name: str, zoneinfo_root: str | None = None) -> str | None:
  tz = str(timezone_name or "").strip()
  if not tz or "\x00" in tz or tz.startswith("/") or "\\" in tz:
    return None
  parts = tz.split("/")
  if any(part in ("", ".", "..") for part in parts):
    return None
  root_real = os.path.realpath(ZONEINFO_ROOT if zoneinfo_root is None else zoneinfo_root)
  target_real = os.path.realpath(os.path.join(root_real, tz))
  try:
    if os.path.commonpath([root_real, target_real]) != root_real or target_real == root_real:
      return None
  except ValueError:
    return None
  if not os.path.isfile(target_real):
    return None
  return target_real


def run_cmd_debug(cmd: list[str]) -> dict:
  try:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC)
    return {
      "cmd": cmd,
      "returncode": proc.returncode,
      "stdout": (proc.stdout or "").strip(),
      "stderr": (proc.stderr or "").strip(),
    }
  except Exception as e:
    return {
      "cmd": cmd,
      "returncode": -1,
      "stdout": "",
      "stderr": str(e),
    }


def sync_system_time_from_browser(epoch_ms: int, timezone_name: str, debug: bool = False) -> dict:
  server_epoch = int(time.time())
  target_epoch = int(epoch_ms // 1000)
  diff_sec = target_epoch - server_epoch

  localtime_path = "/data/etc/localtime"
  timezone_name = (timezone_name or "").strip() or "UTC"
  zoneinfo_path = resolve_zoneinfo_path(timezone_name)

  result: dict[str, Any] = {
    "ok": True,
    "applied": False,
    "server_epoch": server_epoch,
    "target_epoch": target_epoch,
    "diff_sec": diff_sec,
    "timezone": timezone_name,
    "threshold_sec": TIME_SYNC_THRESHOLD_SEC,
    "steps": [],
  }

  def log(msg: str):
    if debug or TIME_SYNC_DEBUG_DEFAULT:
      print(f"[time_sync] {msg}")

  log(f"request tz={timezone_name} target_epoch={target_epoch} server_epoch={server_epoch} diff_sec={diff_sec}")

  # 1) timezone 링크 설정
  if timezone_name:
    if zoneinfo_path is None:
      result["ok"] = False
      result["message"] = f"invalid or missing zoneinfo: {timezone_name}"
      log(result["message"])
      return result

    current_target = ""
    try:
      if os.path.exists(localtime_path) or os.path.islink(localtime_path):
        current_target = os.path.realpath(localtime_path)
    except Exception as e:
      log(f"failed to read current localtime link: {e}")

    if current_target == zoneinfo_path:
      result["steps"].append({"timezone": "already matched"})
      log(f"timezone already matched: {timezone_name}")
    else:
      try:
        if os.path.exists(localtime_path) or os.path.islink(localtime_path):
          subprocess.run(["sudo", "rm", "-f", localtime_path], check=True, timeout=SUBPROCESS_TIMEOUT_SEC)
          result["steps"].append({"remove_localtime": localtime_path})
          log(f"removed existing localtime: {localtime_path}")

        subprocess.run(["sudo", "ln", "-s", zoneinfo_path, localtime_path], check=True, timeout=SUBPROCESS_TIMEOUT_SEC)
        result["steps"].append({"set_timezone_link": zoneinfo_path})
        log(f"timezone set to: {timezone_name}")
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        result["ok"] = False
        result["message"] = f"failed to set timezone: {e}"
        log(result["message"])
        return result

  # 2) timezone이 없었는지 확인
  no_timezone = False
  try:
    if os.path.getsize(localtime_path) == 0:
      no_timezone = True
  except (FileNotFoundError, OSError):
    no_timezone = True

  # 3) diff가 작고 timezone도 있으면 스킵
  if abs(diff_sec) <= TIME_SYNC_THRESHOLD_SEC and not no_timezone:
    log(f"skip date set: within threshold ({diff_sec}s) and timezone exists")
    result["message"] = "skip: time diff within threshold"
    return result

  # 4) 시간 세팅 시도
  # epoch는 UTC 기준이므로 UTC로 해석해서 넣는 것이 안전
  try:
    new_time_utc = datetime.datetime.fromtimestamp(target_epoch, tz=datetime.timezone.utc)
  except (OSError, OverflowError, ValueError) as e:
    result["ok"] = False
    result["message"] = f"invalid target epoch: {e}"
    log(result["message"])
    return result
  formatted_time = new_time_utc.strftime("%Y-%m-%d %H:%M:%S")

  try:
    cmd = ["sudo", "env", "TZ=UTC", "date", "-s", formatted_time]
    result["steps"].append({"date_cmd": " ".join(cmd)})
    log(f"run: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, timeout=SUBPROCESS_TIMEOUT_SEC)
    result["applied"] = True
    result["message"] = "time updated"
    result["server_epoch_after"] = int(time.time())
    log(f"time set success: {formatted_time} UTC")
    return result
  except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
    result["ok"] = False
    result["message"] = f"failed to set date: {e}"
    log(result["message"])
    return result
