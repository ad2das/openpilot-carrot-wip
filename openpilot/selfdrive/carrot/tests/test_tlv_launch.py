"""Launcher tests for tlvd (W102 field fix: mid-drive start froze the manager).

Covers:
  - process_config wiring: tlvd -> tlv.tlvd_launch with spawn=True,
    tlv_dsp_probe spawn=True, and no entry importing tlv.tlvd directly.
  - Fresh-interpreter import purity: importing tlvd_launch must not pull
    numpy/tinygrad/backends or touch os.environ (module level stays stdlib-only,
    the manager preimports every process module).
  - tlvd_launch.main() order: env -> CPU policy -> OOM score -> tlvd.main().
  - Linux-only spawn smoke: start the launcher from a parent whose worker
    threads churn a logging handler lock; Process.start() must return in <5 s
    (the deadlock class: a Python 'fork' Process.start() from the multi-threaded
    manager blocked on thread-held locks and never returned).

On a PC the fork's Cython/unix modules are not built, so this module installs
the same minimal shims as the other test_tlv_* files.
"""
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import types
from multiprocessing import get_context
from pathlib import Path

import pytest


def _install_pc_shims() -> None:
  """Stub Unix-only/Cython modules when running on a PC; no-ops on a device."""

  def stub(name: str, attrs: dict) -> None:
    try:
      __import__(name)
      return
    except ImportError:
      pass
    module = types.ModuleType(name)
    for key, value in attrs.items():
      setattr(module, key, value)
    sys.modules[name] = module

  stub("fcntl", {"LOCK_EX": 2, "LOCK_NB": 4, "LOCK_SH": 1, "LOCK_UN": 8,
                 "flock": lambda *a, **k: None, "ioctl": lambda *a, **k: 0, "fcntl": lambda *a, **k: 0})
  stub("termios", {"tcgetattr": lambda *a, **k: [], "tcsetattr": lambda *a, **k: None,
                   "tcflush": lambda *a, **k: None, "tcsendbreak": lambda *a, **k: None,
                   "TCSANOW": 0, "TCSADRAIN": 1, "TCSAFLUSH": 2, "TCIFLUSH": 0, "TCOFLUSH": 1,
                   "TCIOFLUSH": 2, "B9600": 0, "B115200": 0, "B38400": 0, "ISIG": 0, "ICANON": 0})
  stub("setproctitle", {"getproctitle": lambda: "", "setproctitle": lambda *a, **k: None})

  class _ZmqSock:
    def setsockopt(self, *a, **k):
      pass

    def connect(self, *a, **k):
      pass

    def send(self, *a, **k):
      pass

    def close(self, *a, **k):
      pass

  class _ZmqContext:
    def socket(self, *a, **k):
      return _ZmqSock()

    def term(self):
      pass

  class _ZmqAgain(Exception):
    pass

  stub("zmq", {"Context": _ZmqContext, "PUSH": 1, "LINGER": 2, "NOBLOCK": 4,
               "error": types.SimpleNamespace(Again=_ZmqAgain)})

  try:
    import msgq  # noqa: F401
  except ImportError:
    msgq = types.ModuleType("msgq")

    class _IpcError(Exception):
      pass

    for name in ("Context", "Poller", "SubSocket", "PubSocket", "SocketEventHandle"):
      setattr(msgq, name, type(name, (), {}))
    msgq.IpcError = _IpcError
    msgq.MultiplePublishersError = _IpcError
    for name in ("fake_event_handle", "drain_sock_raw", "toggle_fake_events", "set_fake_prefix",
                 "get_fake_prefix", "delete_fake_prefix", "wait_for_one_event"):
      setattr(msgq, name, lambda *args, **kwargs: None)
    sys.modules["msgq"] = msgq

  try:
    import openpilot.common.params_pyx  # noqa: F401
  except ImportError:
    params_pyx = types.ModuleType("openpilot.common.params_pyx")

    class _Params:
      def __init__(self, *a, **k):
        self._store = {}

      def get(self, key, return_default=False, default=None, **k):
        return self._store.get(key, default)

      def get_bool(self, key, **k):
        return bool(self._store.get(key, False))

      def get_int(self, key, **k):
        try:
          return int(self._store.get(key, 0))
        except (TypeError, ValueError):
          return 0

      def get_float(self, key, **k):
        try:
          return float(self._store.get(key, 0.0))
        except (TypeError, ValueError):
          return 0.0

      def put(self, key, value, **k):
        self._store[key] = value

      def put_bool(self, key, value, **k):
        self._store[key] = bool(value)

      def put_int(self, key, value, **k):
        self._store[key] = int(value)

      def put_nonblocking(self, key, value, **k):
        self._store[key] = value

      def remove(self, key, **k):
        self._store.pop(key, None)

      def check_key(self, key):
        return True

    params_pyx.Params = _Params
    params_pyx.ParamKeyFlag = type("ParamKeyFlag", (), {"PERSISTENT": 1, "CLEAR_ON_MANAGER_START": 2,
                                                        "CLEAR_ON_OFFROAD_TRANSITION": 4, "CLEAR_ON_IGNITION_ON": 8})
    params_pyx.ParamKeyType = type("ParamKeyType", (), {"BOOL": 1, "INT": 2, "FLOAT": 3, "STRING": 4, "JSON": 5, "BYTES": 6})
    params_pyx.UnknownKeyName = type("UnknownKeyName", (Exception,), {})
    sys.modules["openpilot.common.params_pyx"] = params_pyx

  stub("openpilot.cereal.car", {"CarParams": type("CarParams", (), {})})
  stub("openpilot.system.sentry", {"set_tag": lambda *a, **k: None, "capture_exception": lambda *a, **k: None})


_install_pc_shims()

from openpilot.selfdrive.carrot.tlv import tlvd_launch  # noqa: E402
from openpilot.system.manager import process_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]


# -- process registration ----------------------------------------------------

def test_tlvd_process_uses_launcher_with_spawn():
  proc = process_config.managed_processes["tlvd"]
  assert proc.module == "openpilot.selfdrive.carrot.tlv.tlvd_launch"
  assert proc.process_factory is get_context("spawn").Process
  assert proc.restart_if_crash is True
  # No entry may import tlv.tlvd directly: that is the module whose import chain
  # (and old module-level env writes) leaked into the manager.
  assert all(getattr(p, "module", None) != "openpilot.selfdrive.carrot.tlv.tlvd" for p in process_config.procs)


def test_tlv_dsp_probe_process_uses_spawn():
  proc = process_config.managed_processes["tlv_dsp_probe"]
  assert proc.module == "openpilot.selfdrive.carrot.tlv.dsp_probe"
  assert proc.process_factory is get_context("spawn").Process


# -- import purity in a fresh interpreter ------------------------------------

_PURITY_SCRIPT = textwrap.dedent(
  """
  import json, os, sys
  before = dict(os.environ)
  import openpilot.selfdrive.carrot.tlv.tlvd_launch
  banned = sorted(
    name for name in sys.modules
    if name == "numpy" or name.startswith("numpy.")
    or name == "tinygrad" or name.startswith("tinygrad.")
    or name in ("openpilot.selfdrive.carrot.tlv.backends",
                "openpilot.selfdrive.carrot.tlv.tlvd")
  )
  env_added = sorted(key for key in os.environ if key not in before)
  env_changed = sorted(key for key in before if os.environ.get(key) != before[key])
  print(json.dumps({"banned": banned, "env_added": env_added, "env_changed": env_changed}))
  """
)


def test_launcher_import_is_stdlib_only_and_env_clean():
  env = dict(os.environ)
  env["PYTHONPATH"] = str(REPO_ROOT)
  done = subprocess.run(
    [sys.executable, "-c", _PURITY_SCRIPT],
    cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60,
  )
  assert done.returncode == 0, done.stderr
  payload = json.loads(done.stdout.strip().splitlines()[-1])
  assert payload["banned"] == [], f"import pulled heavy modules: {payload['banned']}"
  assert payload["env_added"] == [], f"import added env vars: {payload['env_added']}"
  assert payload["env_changed"] == [], f"import changed env vars: {payload['env_changed']}"


# -- main() step order -------------------------------------------------------

def test_main_sets_env_then_policy_then_oom_then_tlvd(monkeypatch):
  calls = []
  env_keys = tuple(tlvd_launch.TLVD_ENV)
  saved = {key: os.environ.get(key) for key in env_keys}
  os.environ["DEV"] = "QCOM"  # inherited GPU hint must be overridden, not kept

  real_set_env = tlvd_launch.set_process_env

  def record_env():
    real_set_env()
    calls.append("env")

  monkeypatch.setattr(tlvd_launch, "set_process_env", record_env)

  def record_policy():
    assert os.environ.get("DEV") == "CPU"  # env really precedes the policy
    calls.append("policy")
    return []

  monkeypatch.setattr(tlvd_launch, "apply_cpu_policy", record_policy)

  def record_oom(value=1000, path="/proc/self/oom_score_adj"):
    assert os.environ.get("DEV") == "CPU"
    calls.append("oom")
    return True

  monkeypatch.setattr(tlvd_launch, "write_oom_score_adj", record_oom)

  class FakeTlvd:
    @staticmethod
    def main():
      assert all(os.environ.get(key) == value for key, value in tlvd_launch.TLVD_ENV.items())
      calls.append("tlvd")

  monkeypatch.setattr(tlvd_launch, "_load_tlvd", lambda: FakeTlvd)

  try:
    tlvd_launch.main()
  finally:
    for key, value in saved.items():
      if value is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = value

  assert calls == ["env", "policy", "oom", "tlvd"]


# -- Linux-only spawn smoke --------------------------------------------------

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="spawn start smoke is Linux-only")
def test_spawn_start_returns_while_parent_threads_churn_logging_lock():
  import logging

  hold = threading.Lock()
  stop = threading.Event()

  class _HoldLockHandler(logging.Handler):
    def emit(self, record):
      del record
      with hold:
        time.sleep(0.001)

  logger = logging.getLogger("tlv-spawn-smoke")
  logger.setLevel(logging.DEBUG)
  logger.propagate = False
  handler = _HoldLockHandler()
  logger.addHandler(handler)

  def churn():
    while not stop.is_set():
      logger.info("churn")

  threads = [threading.Thread(target=churn, daemon=True) for _ in range(4)]
  for thread in threads:
    thread.start()
  time.sleep(0.05)

  proc = get_context("spawn").Process(target=tlvd_launch.main)
  start_begin = time.monotonic()
  try:
    proc.start()
  finally:
    elapsed = time.monotonic() - start_begin
    stop.set()
    logger.removeHandler(handler)
    for thread in threads:
      thread.join(timeout=2.0)

  try:
    assert elapsed < 5.0, f"spawn start blocked the parent for {elapsed:.1f} s"
  finally:
    proc.join(timeout=30.0)
    if proc.is_alive():
      proc.terminate()
      proc.join(timeout=5.0)
    if proc.is_alive():
      proc.kill()
      proc.join(timeout=5.0)
  assert proc.exitcode is not None
