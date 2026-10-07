import inspect
import threading
from types import SimpleNamespace

import pytest

import openpilot.system.manager.manager as manager


class FakeUI:
  enabled = True

  def __init__(self):
    self.started = 0
    self.stopped = 0

  def start(self):
    self.started += 1

  def stop(self, *args, **kwargs):
    self.stopped += 1

  def prepare(self):
    pass


class FakeProcess:
  enabled = True

  def __init__(self, name, events):
    self.name = name
    self.events = events

  def prepare(self):
    self.events.append(("prepare", self.name))

  def start(self):
    pass

  def stop(self, *args, **kwargs):
    pass


class FakeParams:
  def __init__(self):
    self.values = {}

  def clear_all(self, *args):
    pass

  def all_keys(self):
    return []

  def get_default_value(self, key):
    return None

  def get(self, key, *args):
    return self.values.get(key)

  def get_bool(self, key):
    return bool(self.values.get(key))

  def get_int(self, key, *args):
    return int(self.values.get(key, args[0] if args else 0))

  def get_float(self, key, *args):
    return float(self.values.get(key, args[0] if args else 0.0))

  def put(self, key, value):
    self.values[key] = value

  def put_bool(self, key, value):
    self.values[key] = value

  def put_bool_nonblocking(self, key, value):
    self.values[key] = value

  def put_float(self, key, value):
    self.values[key] = value

  def put_int(self, key, value):
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)


def fake_build_metadata():
  openpilot = SimpleNamespace(version="0.9", git_commit="a" * 40, git_commit_date="2026",
                              git_origin="origin", git_normalized_origin="origin", is_dirty=False)
  return SimpleNamespace(openpilot=openpilot, channel="fork", tested_channel=False, release_channel=False)


@pytest.fixture
def manager_env(monkeypatch, tmp_path):
  events = []
  ui = FakeUI()
  procs = {"ui": ui, "other": FakeProcess("other", events)}
  timing = tmp_path / "timing.log"

  monkeypatch.setattr(manager, "managed_processes", procs)
  monkeypatch.setattr(manager, "ParamKeyFlag", SimpleNamespace(
    CLEAR_ON_MANAGER_START=1, CLEAR_ON_ONROAD_TRANSITION=2,
    CLEAR_ON_OFFROAD_TRANSITION=3, CLEAR_ON_IGNITION_ON=4, DEVELOPMENT_ONLY=5))
  monkeypatch.setattr(manager, "save_bootlog", lambda: None)
  monkeypatch.setattr(manager, "get_build_metadata", fake_build_metadata)
  monkeypatch.setattr(manager, "UpdateStatus", lambda basedir: None)
  monkeypatch.setattr(manager, "Params", FakeParams)
  monkeypatch.setattr(manager, "get_stopping_speed", lambda params, blocking: None)
  monkeypatch.setattr(manager, "configure_wide_camera", lambda params: None)
  monkeypatch.setattr(manager, "register", lambda show_spinner: "dongleid")
  monkeypatch.setattr(manager, "sentry",
                      SimpleNamespace(SentryProject=SimpleNamespace(SELFDRIVE=1), init=lambda *a, **k: None))
  monkeypatch.setattr(manager, "cloudlog",
                      SimpleNamespace(bind_global=lambda *a, **k: None, exception=lambda *a, **k: None))
  monkeypatch.setattr(manager, "HARDWARE", SimpleNamespace(get_serial=lambda: "serial", get_device_type=lambda: "pc"))
  monkeypatch.setattr(manager, "Paths", SimpleNamespace(shm_path=lambda: str(tmp_path / "shm")))
  monkeypatch.delenv("PREPAREONLY", raising=False)
  monkeypatch.setenv("CARROT_BOOT_TIMING_LOG", str(timing))
  return SimpleNamespace(events=events, ui=ui, procs=procs, timing=timing)


def test_ui_starts_before_register_and_prepare(manager_env) -> None:
  manager.manager_init()

  assert manager_env.ui.started == 1
  assert ("prepare", "other") in manager_env.events

  phases = [line.split(" ", 1)[1].strip() for line in manager_env.timing.read_text().splitlines()]
  assert phases.index("ui_started") < phases.index("register") < phases.index("prepare")


def test_prepareonly_does_not_start_early_ui(manager_env, monkeypatch) -> None:
  monkeypatch.setenv("PREPAREONLY", "1")
  manager.manager_init()

  assert manager_env.ui.started == 0
  assert ("prepare", "other") in manager_env.events


def test_early_ui_stops_when_manager_init_raises(manager_env, monkeypatch) -> None:
  def fail_registration(show_spinner):
    raise RuntimeError("registration failed")

  monkeypatch.setattr(manager, "register", fail_registration)

  with pytest.raises(RuntimeError):
    manager.manager_init()

  assert manager_env.ui.started == 1
  assert manager_env.ui.stopped == 1


def test_supported_cars_writer_runs_in_daemon_thread(manager_env, monkeypatch) -> None:
  done = threading.Event()
  calls = []

  def fake_write():
    calls.append(True)
    done.set()

  monkeypatch.setattr(manager, "write_supported_cars_files", fake_write)
  thread = manager.start_supported_cars_writer()

  assert thread.daemon
  assert done.wait(5)
  assert calls == [True]


def test_supported_cars_writer_starts_after_first_ensure_running() -> None:
  source = inspect.getsource(manager.manager_thread)
  assert source.index("ensure_running(managed_processes.values(), False") < source.index("start_supported_cars_writer()")
