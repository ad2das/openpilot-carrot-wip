"""Regression tests for cold-boot and warm-boot startup optimizations."""
import ast
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.common import utils
from openpilot.common.basedir import BASEDIR

ROOT = Path(BASEDIR)


def _bash() -> str:
  if os.name == "nt":
    candidate = "C:/Program Files/Git/bin/bash.exe"
    if Path(candidate).exists():
      return candidate
  return shutil.which("bash") or ""


def test_sudo_write_batch_direct_path_uses_zero_sudo(tmp_path: Path, monkeypatch) -> None:
  f1 = tmp_path / "a"
  f2 = tmp_path / "b"
  f1.write_text("", encoding="utf-8")
  f2.write_text("", encoding="utf-8")

  sudo_calls = []
  monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: sudo_calls.append((a, k)))

  utils.sudo_write_batch([("1", str(f1)), ("performance", str(f2))], chmod_paths=[str(f1)])
  assert sudo_calls == []
  assert f1.read_text(encoding="utf-8") == "1"
  assert f2.read_text(encoding="utf-8") == "performance"


def test_sudo_write_batch_preserves_order_and_batches_single_sudo(tmp_path: Path, monkeypatch) -> None:
  # Simulate direct write failing on the second item: item 1 writes directly,
  # then items 2 and 3 (plus chmod_paths) flush in order in ONE sudo sh -c call
  # with zero `su` invocation.
  p1 = tmp_path / "p1"
  p2 = tmp_path / "p2"
  p3 = tmp_path / "p3"
  c1 = tmp_path / "c1"
  for p in (p1, p2, p3, c1):
    p.write_text("", encoding="utf-8")

  real_open = open

  def fake_open(path, mode="r", *args, **kwargs):
    if str(path) == str(p2) and "w" in mode:
      raise PermissionError("root only")
    return real_open(path, mode, *args, **kwargs)

  real_access = os.access

  def fake_access(path, mode):
    if str(path) == str(c1):
      return False
    return real_access(path, mode)

  recorded = []
  monkeypatch.setattr("builtins.open", fake_open)
  monkeypatch.setattr(utils.os, "access", fake_access)
  monkeypatch.setattr(utils.subprocess, "run", lambda cmd, **kw: recorded.append(cmd) or SimpleNamespace(stderr=""))

  utils.sudo_write_batch([("first", str(p1)), ("second", str(p2)), ("third", str(p3))], chmod_paths=[str(c1)])

  assert p1.read_text(encoding="utf-8") == "first"
  assert len(recorded) == 1
  cmd = recorded[0]
  assert cmd[:3] == ["sudo", "sh", "-c"]
  assert "su " not in cmd[3]
  assert cmd[4:] == ["_", "1", str(c1), "second", str(p2), "third", str(p3)]


def test_sudo_write_batch_surfaces_failed_paths_to_stderr(tmp_path: Path, monkeypatch, capsys) -> None:
  p1 = tmp_path / "p1"
  p1.write_text("", encoding="utf-8")

  def deny_open(*_a, **_k):
    raise PermissionError("root only")

  monkeypatch.setattr("builtins.open", deny_open)
  monkeypatch.setattr(
    utils.subprocess,
    "run",
    lambda cmd, **kw: SimpleNamespace(stderr=f"{p1}\n/sys/missing/node\n"),
  )

  utils.sudo_write_batch([("1", str(p1))])
  captured = capsys.readouterr()
  assert "sudo_write_batch failed for:" in captured.err
  assert str(p1) in captured.err
  assert "/sys/missing/node" in captured.err


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX sh/chmod execution test")
def test_sudo_batch_script_executes_cleanly_with_edge_values(tmp_path: Path) -> None:
  c1 = tmp_path / "unlock_me"
  c1.write_text("", encoding="utf-8")
  c1.chmod(0o444)
  w1 = tmp_path / "dash_val"
  w2 = tmp_path / "newline_val"
  w1.write_text("", encoding="utf-8")
  w2.write_text("", encoding="utf-8")
  missing_chmod = tmp_path / "missing_chmod"
  missing_write = tmp_path / "no_such_dir" / "missing_write"

  args = [
    "2", str(missing_chmod), str(c1),
    "-n", str(w1),
    "ignored", str(missing_write),
    "1\n", str(w2),
  ]
  res = subprocess.run(["sh", "-c", utils._SUDO_BATCH_SCRIPT, "_", *args], check=False, capture_output=True, text=True)
  assert res.returncode == 0
  assert str(missing_write) in res.stderr
  assert str(w1) not in res.stderr
  assert str(w2) not in res.stderr
  assert stat.S_IMODE(c1.stat().st_mode) & 0o222 == 0o222
  assert w1.read_text(encoding="utf-8") == "-n"
  assert w2.read_text(encoding="utf-8") == "1\n"


def test_sudo_read_direct_and_fallback(tmp_path: Path, monkeypatch) -> None:
  f = tmp_path / "info"
  f.write_text("  Core state: 1 \n", encoding="utf-8")
  calls = []
  monkeypatch.setattr(utils.subprocess, "check_output", lambda *a, **k: calls.append((a, k)) or "fallback")
  assert utils.sudo_read(str(f)) == "Core state: 1"
  assert calls == []

  def deny_open(*_a, **_k):
    raise PermissionError("denied")

  monkeypatch.setattr("builtins.open", deny_open)
  assert utils.sudo_read(str(f)) == "fallback"
  assert len(calls) == 1
  assert calls[0][0] == (["sudo", "cat", str(f)],)
  assert "shell" not in calls[0][1]


@pytest.mark.skipif(os.name == "nt", reason="TICI hardware modules require Linux")
def test_tici_initialize_hardware_batches_writes_and_dedups_amplifier(monkeypatch) -> None:
  from openpilot.system.hardware.tici import hardware

  class FakeAmp:
    def __init__(self, init_result=True):
      self.init_result = init_result
      self.events = []

    def initialize_configuration(self, model):
      self.events.append(("init", model))
      return self.init_result

    def set_global_shutdown(self, amp_disabled):
      self.events.append(("shutdown", amp_disabled))
      return True

  batches = []
  subprocs = []
  direct_writes = []
  monkeypatch.setattr(hardware, "gpio_init", lambda *a: None)
  monkeypatch.setattr(hardware, "gpio_set", lambda *a: None)
  monkeypatch.setattr(hardware, "get_irqs_for_action", lambda action: ["101"] if action in ("kgsl-3d0", "a5", "spi_geni") else [])
  monkeypatch.setattr(hardware, "sudo_write_batch", lambda writes, chmod_paths=(): batches.append((list(writes), list(chmod_paths))))
  monkeypatch.setattr(hardware, "sudo_write", lambda val, path: direct_writes.append((val, path)))
  monkeypatch.setattr(hardware.subprocess, "check_output", lambda *a, **k: "42\n43\n")
  monkeypatch.setattr(hardware.subprocess, "call", lambda cmd, **k: subprocs.append(cmd) or 0)

  tici = hardware.Tici()
  amp = FakeAmp(init_result=True)
  tici.__dict__["amplifier"] = amp
  monkeypatch.setattr(tici, "get_device_type", lambda: "tizi")

  tici.initialize_hardware()
  assert len(batches) == 1
  writes, chmods = batches[0]
  assert ("f", "/proc/irq/default_smp_affinity") in writes
  assert ("N", "/sys/kernel/debug/msm_vidc/clock_scaling") in writes
  assert ("Y", "/sys/kernel/debug/msm_vidc/disable_thermal_mitigation") in writes
  assert ("3", "/proc/irq/101/smp_affinity_list") in writes
  assert "/dev/kmsg" in chmods
  assert "/sys/devices/system/cpu/cpu4/online" in chmods
  assert "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor" in chmods
  assert "/proc/irq/101/smp_affinity_list" in chmods
  assert len(subprocs) == 1
  assert subprocs[0][:3] == ["sudo", "sh", "-c"]
  assert subprocs[0][-2:] == ["42", "43"]
  assert amp.events == [("init", "tizi")]

  # Immediate set_power_save(False) after successful initialize_hardware skips duplicate amp init.
  tici.set_power_save(False)
  assert amp.events == [("init", "tizi")]

  # Entering powersave shuts down amp; exiting powersave re-initializes it.
  tici.set_power_save(True)
  assert amp.events[-1] == ("shutdown", True)
  tici.set_power_save(False)
  assert amp.events[-2:] == [("shutdown", False), ("init", "tizi")]

  # If initialize_hardware failed amp init, set_power_save(False) retries it.
  tici2 = hardware.Tici()
  amp2 = FakeAmp(init_result=False)
  tici2.__dict__["amplifier"] = amp2
  monkeypatch.setattr(tici2, "get_device_type", lambda: "tizi")
  tici2.initialize_hardware()
  tici2.set_power_save(False)
  assert amp2.events == [("init", "tizi"), ("shutdown", False), ("init", "tizi")]


def test_python_process_skips_preimport_for_spawn_and_opt_out(monkeypatch) -> None:
  path = ROOT / "openpilot/system/manager/process.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PythonProcess")
  imported = []
  ns = {
    "ManagerProcess": type("Base", (), {"proc": None, "shutting_down": False}),
    "Process": lambda *a, **k: SimpleNamespace(start=lambda: None),
    "get_context": lambda _m: SimpleNamespace(Process=lambda *a, **k: SimpleNamespace(start=lambda: None)),
    "launcher": lambda *a: None,
    "cloudlog": SimpleNamespace(info=lambda *a: None),
    "importlib": SimpleNamespace(import_module=lambda m: imported.append(m)),
  }
  exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), ns)

  p_normal = ns["PythonProcess"]("normal", "mod.normal", lambda *_: True)
  p_spawn = ns["PythonProcess"]("spawned", "mod.spawned", lambda *_: True, spawn=True)
  p_skip = ns["PythonProcess"]("debug", "mod.debug", lambda *_: True, preimport=False)

  p_normal.prepare()
  p_spawn.prepare()
  p_skip.prepare()
  assert imported == ["mod.normal"]


def test_process_config_prioritizes_driving_critical_path() -> None:
  path = ROOT / "openpilot/system/manager/process_config.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  names = []
  carrot_cluster_preimport = None
  for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("PythonProcess", "NativeProcess", "DaemonProcess"):
      proc_name = ast.literal_eval(node.args[0])
      names.append(proc_name)
      if proc_name == "carrot_cluster":
        for kw in node.keywords:
          if kw.arg == "preimport":
            carrot_cluster_preimport = ast.literal_eval(kw.value)
  for critical in ("camerad", "modeld", "card", "selfdrived", "controlsd"):
    for low_prio in ("proclogd", "journald", "micd"):
      assert names.index(critical) < names.index(low_prio), f"{critical} must start before {low_prio}"
  assert carrot_cluster_preimport is not False


def test_manager_early_hardware_overlap_cleanup_and_prepare_skip(monkeypatch, tmp_path) -> None:
  import openpilot.system.manager.manager as manager

  events = []

  class FakeProc:
    enabled = True

    def __init__(self, name):
      self.name = name
      self.started = 0
      self.stopped = 0

    def start(self):
      self.started += 1
      events.append(("start", self.name))

    def stop(self, *args, **kwargs):
      self.stopped += 1
      events.append(("stop", self.name))

    def prepare(self):
      events.append(("prepare", self.name))

  class FakeParams:
    def __init__(self):
      self.values = {}

    def clear_all(self, *args): pass
    def all_keys(self): return []
    def get_default_value(self, key): return None
    def get(self, key, *args): return self.values.get(key)
    def get_bool(self, key): return bool(self.values.get(key))
    def get_int(self, key, *args): return int(self.values.get(key, 0))
    def get_float(self, key, *args): return float(self.values.get(key, 0.0))
    def put(self, key, value): self.values[key] = value
    def put_bool(self, key, value):
      events.append(("put_bool", key, value))
      self.values[key] = value
    def put_bool_nonblocking(self, key, value): self.values[key] = value
    def put_float(self, key, value): self.values[key] = value
    def put_int(self, key, value): self.values[key] = value
    def remove(self, key): self.values.pop(key, None)

  procs = {name: FakeProc(name) for name in ("ui", "pandad", "hardwared", "modeld")}
  meta = SimpleNamespace(
    openpilot=SimpleNamespace(version="0.9", git_commit="a" * 40, git_commit_date="2026",
                              git_origin="origin", git_normalized_origin="origin", is_dirty=False),
    channel="fork", tested_channel=False, release_channel=False,
  )
  monkeypatch.setattr(manager, "managed_processes", procs)
  monkeypatch.setattr(manager, "ParamKeyFlag", SimpleNamespace(
    CLEAR_ON_MANAGER_START=1, CLEAR_ON_ONROAD_TRANSITION=2,
    CLEAR_ON_OFFROAD_TRANSITION=3, CLEAR_ON_IGNITION_ON=4, DEVELOPMENT_ONLY=5))
  monkeypatch.setattr(manager, "save_bootlog", lambda: None)
  monkeypatch.setattr(manager, "get_build_metadata", lambda: meta)
  monkeypatch.setattr(manager, "UpdateStatus", lambda _d: None)
  monkeypatch.setattr(manager, "Params", FakeParams)
  monkeypatch.setattr(manager, "get_stopping_speed", lambda p, blocking: None)
  monkeypatch.setattr(manager, "configure_wide_camera", lambda p: None)
  monkeypatch.setattr(manager, "register", lambda show_spinner: "dongle")
  monkeypatch.setattr(manager, "sentry", SimpleNamespace(SentryProject=SimpleNamespace(SELFDRIVE=1), init=lambda *a, **k: None))
  monkeypatch.setattr(manager, "cloudlog", SimpleNamespace(bind_global=lambda *a, **k: None, exception=lambda *a, **k: None))
  monkeypatch.setattr(manager, "HARDWARE", SimpleNamespace(get_serial=lambda: "ser", get_device_type=lambda: "pc"))
  monkeypatch.setattr(manager, "Paths", SimpleNamespace(shm_path=lambda: str(tmp_path / "shm")))
  monkeypatch.delenv("PREPAREONLY", raising=False)
  monkeypatch.delenv("NOBOARD", raising=False)
  monkeypatch.delenv("BLOCK", raising=False)
  monkeypatch.setenv("CARROT_BOOT_TIMING_LOG", str(tmp_path / "timing.log"))

  manager.manager_init()
  # IsOnroad must be cleared to False before pandad starts.
  assert events.index(("put_bool", "IsOnroad", False)) < events.index(("start", "pandad"))
  assert events.index(("start", "pandad")) < events.index(("prepare", "modeld"))
  assert events.index(("start", "hardwared")) < events.index(("prepare", "modeld"))
  # Early-started processes must not be re-imported into manager.
  assert ("prepare", "ui") not in events
  assert ("prepare", "pandad") not in events
  assert ("prepare", "hardwared") not in events

  # Failure during prepare must stop all early-started processes (ui, pandad, hardwared).
  events.clear()
  for p in procs.values():
    p.started = p.stopped = 0
  procs["modeld"].prepare = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
  with pytest.raises(RuntimeError, match="boom"):
    manager.manager_init()
  assert procs["ui"].stopped == 1
  assert procs["pandad"].stopped == 1
  assert procs["hardwared"].stopped == 1


def test_write_supported_cars_caches_clean_commit(monkeypatch, tmp_path: Path) -> None:
  import openpilot.system.manager.manager as manager

  imports = []
  meta_calls = 0

  def fake_import(name):
    imports.append(name)
    brand = name.split(".")[2]
    doc = SimpleNamespace(name=f"{brand}-car")
    return SimpleNamespace(CAR=[SimpleNamespace(config=SimpleNamespace(car_docs=[doc]))])

  meta = SimpleNamespace(openpilot=SimpleNamespace(git_commit="abc123", is_dirty=False))

  def fake_get_build_metadata():
    nonlocal meta_calls
    meta_calls += 1
    return meta

  monkeypatch.setattr(manager, "_boot_build_metadata", None)
  monkeypatch.setattr(manager, "Params", lambda: SimpleNamespace(get_param_path=lambda: str(tmp_path)))
  monkeypatch.setattr(manager, "get_build_metadata", fake_get_build_metadata)
  monkeypatch.setattr(manager.importlib, "import_module", fake_import)

  manager.write_supported_cars_files()
  assert len(imports) == 7
  assert meta_calls == 1
  assert (tmp_path / ".supported_cars_commit").read_text().strip() == "abc123"

  # Cached boot metadata avoids redundant get_build_metadata() git subprocesses.
  imports.clear()
  monkeypatch.setattr(manager, "_boot_build_metadata", meta)
  manager.write_supported_cars_files()
  assert imports == []
  assert meta_calls == 1

  # Missing one file -> rebuilds.
  (tmp_path / "SupportedCars_tesla").unlink()
  manager.write_supported_cars_files()
  assert len(imports) == 7

  # New commit -> rebuilds.
  imports.clear()
  meta.openpilot.git_commit = "def456"
  manager.write_supported_cars_files()
  assert len(imports) == 7
  assert (tmp_path / ".supported_cars_commit").read_text().strip() == "def456"

  # Dirty checkout -> always writes.
  imports.clear()
  meta.openpilot.is_dirty = True
  manager.write_supported_cars_files()
  assert len(imports) == 7


@pytest.mark.skipif(_bash() == "", reason="bash is unavailable")
def test_launcher_bootstrap_fast_precheck_and_agnos_single_sudo(tmp_path: Path) -> None:
  source = (ROOT / "launch_chffrplus.sh").read_text(encoding="utf-8")

  agnos_fn = source[source.index("function agnos_init {"):source.index("function start_carrot_recovery {")]
  assert agnos_fn.count("sudo sh -c") == 1
  assert "sudo abctl" not in agnos_fn
  assert "sudo chgrp" not in agnos_fn
  assert "sudo chmod" not in agnos_fn

  boot_fn = source[source.index("function bootstrap_runtime_dependencies {"):source.index("function show_agnos_update_failure {")]
  harness_hit = f"""
DIR={tmp_path.as_posix()!r}
python3() {{ return 0; }}
ensure_python_package() {{ echo "ENSURE:$1"; return 0; }}
{boot_fn}
bootstrap_runtime_dependencies
"""
  res_hit = subprocess.run([_bash(), "-c", harness_hit], capture_output=True, text=True, timeout=15)
  assert res_hit.returncode == 0, res_hit.stderr
  assert "ENSURE:" not in res_hit.stdout

  harness_miss = f"""
DIR={tmp_path.as_posix()!r}
python3() {{ return 1; }}
ensure_python_package() {{ echo "ENSURE:$1"; return 0; }}
{boot_fn}
bootstrap_runtime_dependencies
"""
  res_miss = subprocess.run([_bash(), "-c", harness_miss], capture_output=True, text=True, timeout=15)
  assert res_miss.returncode == 0, res_miss.stderr
  assert "ENSURE:serial" in res_miss.stdout


def test_main_layout_initializes_settings_layout_eagerly(monkeypatch) -> None:
  # SettingsLayout owns the main UI WifiManager and toggle/developer/device transition callbacks;
  # it must be constructed during MainLayout.__init__ so background Wi-Fi scanning and param
  # transition hooks are live from boot even when the user never opens Settings.
  settings_created = []
  settings_callbacks = []
  settings_panels = []
  settings_events = []

  class Widget:
    def show_event(self): pass
    def hide_event(self): pass

  class DummyLayout(Widget):
    def __init__(self):
      self._setup_widget = SimpleNamespace(set_open_settings_callback=lambda cb: None)
    def set_settings_callback(self, cb): pass
    def set_click_callback(self, cb): pass
    def render(self, rect): pass

  class FakeSidebar(Widget):
    def __init__(self): self.is_visible = True
    def set_callbacks(self, **kw): pass
    def set_visible(self, v): self.is_visible = v
    def render(self, rect): pass

  class FakeSettings(Widget):
    def __init__(self): settings_created.append(self)
    def set_callbacks(self, on_close): settings_callbacks.append(on_close)
    def set_current_panel(self, panel): settings_panels.append(panel)
    def show_event(self): settings_events.append("show")
    def hide_event(self): settings_events.append("hide")
    def render(self, rect): pass

  class FakeOnboarding:
    completed = True

  gui_app = SimpleNamespace(push_widget=lambda w: None, is_recording=lambda: False)
  fake_ui_state = SimpleNamespace(params=SimpleNamespace(get_bool=lambda k: False), started=False, sm=None)
  panel_type = SimpleNamespace(DEVICE=0, TOGGLES=2, FIREHOSE=4)
  stubs = {
    "pyray": SimpleNamespace(Rectangle=lambda *a: a),
    "openpilot.selfdrive.ui.layouts.sidebar": SimpleNamespace(Sidebar=FakeSidebar, SIDEBAR_WIDTH=300),
    "openpilot.selfdrive.ui.layouts.home": SimpleNamespace(HomeLayout=DummyLayout),
    "openpilot.selfdrive.ui.layouts.settings.settings": SimpleNamespace(SettingsLayout=FakeSettings, PanelType=panel_type),
    "openpilot.selfdrive.ui.onroad.augmented_road_view": SimpleNamespace(AugmentedRoadView=DummyLayout),
    "openpilot.selfdrive.ui.ui_state": SimpleNamespace(
      device=SimpleNamespace(add_interactive_timeout_callback=lambda cb: None),
      ui_state=fake_ui_state,
    ),
    "openpilot.system.ui.lib.application": SimpleNamespace(gui_app=gui_app),
    "openpilot.system.ui.widgets": SimpleNamespace(Widget=Widget),
    "openpilot.selfdrive.ui.layouts.onboarding": SimpleNamespace(OnboardingWindow=FakeOnboarding),
    "openpilot.selfdrive.ui.widgets.carrot_web_dialog": SimpleNamespace(CarrotWebDialog=DummyLayout),
  }
  for name, stub in stubs.items():
    monkeypatch.setitem(sys.modules, name, stub)

  spec = importlib.util.spec_from_file_location("_test_eager_main_layout", ROOT / "openpilot/selfdrive/ui/layouts/main.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)

  layout = mod.MainLayout()
  assert len(settings_created) == 1
  assert len(settings_callbacks) == 1

  layout.open_settings(panel_type.TOGGLES)
  assert len(settings_created) == 1
  assert settings_panels == [panel_type.TOGGLES]
  assert settings_events == ["show"]

  settings_callbacks[0]()
  assert settings_events == ["show", "hide"]
  layout.open_settings(panel_type.DEVICE)
  assert len(settings_created) == 1
  assert settings_panels == [panel_type.TOGGLES, panel_type.DEVICE]


def test_modeld_preloads_pkl_and_preserves_vipc_drop_accounting() -> None:
  source = (ROOT / "openpilot/selfdrive/modeld/modeld.py").read_text(encoding="utf-8")
  assert "preloaded_internal_jits = load_oob(open_file_chunked(modeld_pkl_path(False)))" in source
  assert "vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)" in source
  assert "if run_count > 0 else 0" not in source


def test_hardwared_bounds_fast_offline_poll_to_boot_window() -> None:
  path = ROOT / "openpilot/system/hardware/hardwared.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "hw_state_thread")

  def run_ticks(monotonic_val: float, network_type: str, ticks: int) -> list[int]:
    polled_at: list[int] = []
    tick_idx = 0

    class FakeEndEvent:
      def is_set(self) -> bool:
        return tick_idx >= ticks

    class FakeQueue:
      def put_nowait(self, _item) -> None:
        pass

    def fake_sleep(_dt: float) -> None:
      nonlocal tick_idx
      tick_idx += 1

    def get_network_type() -> str:
      polled_at.append(tick_idx)
      return network_type

    ns = {
      "DT_HW": 0.5,
      "AGNOS": False,
      "NetworkType": SimpleNamespace(none="none", wifi="wifi"),
      "HardwareState": lambda **kw: SimpleNamespace(**kw),
      "HARDWARE": SimpleNamespace(
        get_network_type=get_network_type,
        get_modem_temperatures=list,
        get_modem_version=lambda: None,
        get_modem_data_usage=lambda: (0, 0),
        get_network_info=lambda: None,
        get_network_strength=lambda _nt: 0,
        get_network_metered=lambda _nt: False,
      ),
      "time": SimpleNamespace(monotonic=lambda: monotonic_val, sleep=fake_sleep),
      "queue": SimpleNamespace(Full=RuntimeError),
      "cloudlog": SimpleNamespace(event=lambda *a, **k: None, exception=lambda *a, **k: None),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    ns["hw_state_thread"](FakeEndEvent(), FakeQueue())
    return polled_at

  # Boot window (< 120s) while offline: polls every 2s (every 4 ticks at DT_HW=0.5).
  assert run_ticks(monotonic_val=15.0, network_type="none", ticks=10) == [0, 4, 8]
  # After 120s while still offline: falls back to normal 10s cadence (every 20 ticks).
  assert run_ticks(monotonic_val=130.0, network_type="none", ticks=21) == [0, 20]
  # Connected during boot window: switches immediately to 10s cadence after first poll.
  assert run_ticks(monotonic_val=15.0, network_type="wifi", ticks=21) == [0, 20]
