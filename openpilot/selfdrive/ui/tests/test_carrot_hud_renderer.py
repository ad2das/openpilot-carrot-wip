import ast
import importlib.util
import sys
import time as stdlib_time
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


HUD_RENDERER_PATH = Path(__file__).parents[1] / "onroad" / "hud_renderer.py"


@pytest.fixture
def hud_module(monkeypatch):
  @dataclass(frozen=True)
  class Color:
    r: int
    g: int
    b: int
    a: int

  @dataclass
  class Rectangle:
    x: float
    y: float
    width: float
    height: float

  @dataclass
  class Vector2:
    x: float
    y: float

  class Texture:
    pass

  raylib = types.ModuleType("pyray")
  raylib.Color = Color
  raylib.Rectangle = Rectangle
  raylib.Vector2 = Vector2
  raylib.Texture = Texture
  raylib.WHITE = Color(255, 255, 255, 255)
  raylib.BLACK = Color(0, 0, 0, 255)
  raylib.BLANK = Color(0, 0, 0, 0)
  raylib.GREEN = Color(0, 255, 0, 255)
  raylib.YELLOW = Color(255, 255, 0, 255)
  for name in (
    "draw_line_ex",
    "draw_rectangle_gradient_v",
    "draw_rectangle_rounded",
    "draw_rectangle_rounded_lines_ex",
    "draw_text_ex",
    "draw_texture_pro",
    "draw_triangle",
  ):
    setattr(raylib, name, lambda *args, **kwargs: None)

  class Widget:
    def __init__(self):
      pass

  @dataclass
  class Gradient:
    start: tuple[float, float]
    end: tuple[float, float]
    colors: list
    stops: list[float]

  fake_ui_state = SimpleNamespace(
    params=None,
    sm=None,
    started_frame=0,
    status=0,
    is_metric=True,
    usbgpu_present=False,
    usbgpu_compiled=False,
    usbgpu_compile_pending=False,
    usbgpu_loading=False,
    usbgpu_active=False,
    usbgpu_startup_failed=False,
  )
  gui_app = SimpleNamespace(
    font=lambda weight: ("font", weight),
    has_font=lambda weight: False,
    texture=lambda path: ("texture", path),
    width=2160,
    height=1080,
  )

  stubs = {
    "pyray": raylib,
    "openpilot.common.constants": SimpleNamespace(
      CV=SimpleNamespace(MS_TO_KPH=3.6, MS_TO_MPH=2.2369362920544),
    ),
    "openpilot.system.hardware.usbgpu": SimpleNamespace(
      usbgpu_badge_state=lambda compiled, loading, active, failed, compile_pending=False: (
        "error" if failed else "loading" if loading else "compile_pending" if compile_pending
        else "active" if active else "ready" if compiled else "not_compiled"
      ),
    ),
    "openpilot.selfdrive.ui.ui_state": SimpleNamespace(
      ui_state=fake_ui_state,
      UIStatus=SimpleNamespace(ENGAGED=1, DISENGAGED=2, OVERRIDE=3),
    ),
    "openpilot.system.ui.lib.application": SimpleNamespace(
      gui_app=gui_app,
      GL_VERSION=3,
      FontWeight=SimpleNamespace(SEMI_BOLD=1, BOLD=2, MEDIUM=3, DISPLAY=4),
      FONT_DIR=Path("openpilot/system/assets/fonts"),
    ),
    "openpilot.system.ui.lib.multilang": SimpleNamespace(tr=lambda text: text),
    "openpilot.system.ui.lib.shader_polygon": SimpleNamespace(
      draw_polygon=lambda *args, **kwargs: None,
      draw_polygon_solid=lambda *args, **kwargs: None,
      draw_polygons=lambda *args, **kwargs: None,
      Gradient=Gradient,
    ),
    "openpilot.system.ui.lib.text_measure": SimpleNamespace(
      measure_text_cached=lambda font, text, size: Vector2(len(text) * size, size),
    ),
    "openpilot.system.ui.lib.text_draw": SimpleNamespace(draw_text_ui_style=lambda *args, **kwargs: None),
    "openpilot.system.ui.widgets": SimpleNamespace(Widget=Widget),
  }
  for name, stub in stubs.items():
    monkeypatch.setitem(sys.modules, name, stub)

  module_name = f"_test_carrot_hud_renderer_{id(fake_ui_state)}"
  spec = importlib.util.spec_from_file_location(module_name, HUD_RENDERER_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  monkeypatch.setitem(sys.modules, module_name, module)
  spec.loader.exec_module(module)
  return module, fake_ui_state


class FakeParams:
  def __init__(self, values, failures=None):
    self.values = values
    self.failures = dict(failures or {})
    self.calls = []

  def get_int(self, key):
    self.calls.append(key)
    if self.failures.get(key, 0) > 0:
      self.failures[key] -= 1
      raise RuntimeError(f"failed to read {key}")
    return self.values[key]


def make_param_renderer(module, *, next_refresh=0.0):
  renderer = object.__new__(module.HudRenderer)
  renderer._hud_params_next_refresh_time = next_refresh
  renderer._show_device_state = 91
  renderer._show_date_time = 92
  renderer._show_tpms = 93
  renderer._show_plot_mode = 94
  renderer._longitudinal_personality = 95
  return renderer


def test_direct_hud_param_reads_are_isolated_to_refresh():
  tree = ast.parse(HUD_RENDERER_PATH.read_text(encoding="utf-8"))
  reads = []

  for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
    for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
      func = call.func
      if (
        isinstance(func, ast.Attribute)
        and func.attr == "get_int"
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "params"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "ui_state"
      ):
        assert len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
        reads.append((function.name, call.args[0].value))

  assert reads == [
    ("_refresh_hud_params", "ShowDeviceState"),
    ("_refresh_hud_params", "ShowDateTime"),
    ("_refresh_hud_params", "ShowTpms"),
    ("_refresh_hud_params", "ShowPlotMode"),
    ("_refresh_hud_params", "LongitudinalPersonality"),
    ("_refresh_hud_params", "CruiseGapLevels"),
  ]


def test_hud_renderer_has_no_subsection_profiling():
  tree = ast.parse(HUD_RENDERER_PATH.read_text(encoding="utf-8"))

  assert not any(
    isinstance(node, ast.ClassDef) and node.name == "HudRenderTimings"
    for node in ast.walk(tree)
  )
  assert not any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "monotonic_ns"
    for node in ast.walk(tree)
  )


def test_initial_cruise_gap_preserves_legacy_fallback(hud_module):
  module, _ = hud_module

  renderer = module.HudRenderer()

  assert renderer._longitudinal_personality == 7
  assert renderer._get_cruise_gap() == 8


def test_hud_params_are_refreshed_once_per_interval(hud_module):
  module, fake_ui_state = hud_module
  params = FakeParams({
    "ShowDeviceState": 1,
    "ShowDateTime": 2,
    "ShowTpms": 3,
    "ShowPlotMode": 4,
    "LongitudinalPersonality": 4,
    "CruiseGapLevels": 3,
  })
  fake_ui_state.params = params
  renderer = make_param_renderer(module)

  renderer._refresh_hud_params(0.0)
  assert (
    renderer._show_device_state,
    renderer._show_date_time,
    renderer._show_tpms,
    renderer._show_plot_mode,
    renderer._longitudinal_personality,
  ) == (1, 2, 3, 4, 4)
  assert renderer._hud_params_next_refresh_time == 1.0
  assert renderer._cruise_gap_levels == 3
  assert params.calls == [
    "ShowDeviceState",
    "ShowDateTime",
    "ShowTpms",
    "ShowPlotMode",
    "LongitudinalPersonality",
    "CruiseGapLevels",
  ]

  renderer._refresh_hud_params(0.999)
  assert len(params.calls) == 6

  renderer._refresh_hud_params(1.0)
  assert len(params.calls) == 12
  assert renderer._hud_params_next_refresh_time == 2.0


@pytest.mark.parametrize("levels, expected", [(2, 2), (3, 3), (4, 4), (1, 4), (9, 4)])
def test_cruise_gap_levels_outside_two_to_four_show_four(hud_module, levels, expected):
  module, fake_ui_state = hud_module
  fake_ui_state.params = FakeParams({
    "ShowDeviceState": 1,
    "ShowDateTime": 2,
    "ShowTpms": 3,
    "ShowPlotMode": 4,
    "LongitudinalPersonality": 1,
    "CruiseGapLevels": levels,
  })
  renderer = make_param_renderer(module)

  renderer._refresh_hud_params(0.0)

  assert renderer._cruise_gap_levels == expected


def test_cruise_gap_levels_read_failure_shows_four(hud_module):
  module, fake_ui_state = hud_module
  fake_ui_state.params = FakeParams({
    "ShowDeviceState": 1,
    "ShowDateTime": 2,
    "ShowTpms": 3,
    "ShowPlotMode": 4,
    "LongitudinalPersonality": 1,
    "CruiseGapLevels": 2,
  }, failures={"CruiseGapLevels": 1})
  renderer = make_param_renderer(module)

  renderer._refresh_hud_params(0.0)

  assert renderer._cruise_gap_levels == 4
  assert renderer._longitudinal_personality == 1


def test_show_param_failure_keeps_complete_snapshot_and_retries(hud_module):
  module, fake_ui_state = hud_module
  params = FakeParams({
    "ShowDeviceState": 1,
    "ShowDateTime": 2,
    "ShowTpms": 3,
    "ShowPlotMode": 4,
    "LongitudinalPersonality": 4,
  }, failures={"ShowDateTime": 1})
  fake_ui_state.params = params
  renderer = make_param_renderer(module)

  renderer._refresh_hud_params(0.0)

  assert (
    renderer._show_device_state,
    renderer._show_date_time,
    renderer._show_tpms,
    renderer._show_plot_mode,
    renderer._longitudinal_personality,
  ) == (91, 92, 93, 94, 95)
  assert renderer._hud_params_next_refresh_time == 0.0
  assert params.calls == ["ShowDeviceState", "ShowDateTime"]

  renderer._refresh_hud_params(0.1)
  assert (
    renderer._show_device_state,
    renderer._show_date_time,
    renderer._show_tpms,
    renderer._show_plot_mode,
    renderer._longitudinal_personality,
  ) == (1, 2, 3, 4, 4)
  assert renderer._hud_params_next_refresh_time == pytest.approx(1.1)


def test_personality_failure_uses_gap_eight_and_retries_next_frame(hud_module):
  module, fake_ui_state = hud_module
  params = FakeParams({
    "ShowDeviceState": 1,
    "ShowDateTime": 2,
    "ShowTpms": 3,
    "ShowPlotMode": 4,
    "LongitudinalPersonality": 4,
  }, failures={"LongitudinalPersonality": 1})
  fake_ui_state.params = params
  renderer = make_param_renderer(module)

  renderer._refresh_hud_params(0.0)

  assert (renderer._show_device_state, renderer._show_date_time, renderer._show_tpms, renderer._show_plot_mode) == (1, 2, 3, 4)
  assert renderer._get_cruise_gap() == 8
  assert renderer._hud_params_next_refresh_time == 0.0

  renderer._refresh_hud_params(0.1)
  assert renderer._get_cruise_gap() == 5
  assert renderer._hud_params_next_refresh_time == pytest.approx(1.1)
  assert params.calls.count("LongitudinalPersonality") == 2


def test_speed_limit_snapshot_reads_submaster_once(hud_module):
  module, fake_ui_state = hud_module
  carrot_man = SimpleNamespace(xSpdLimit="80", xSignType=2.0, nRoadLimitSpeed=90)

  class CountingSubMaster:
    def __init__(self):
      self.calls = 0

    def __getitem__(self, key):
      assert key == "carrotMan"
      self.calls += 1
      if self.calls > 1:
        raise AssertionError("carrotMan was read more than once")
      return carrot_man

  fake_ui_state.sm = CountingSubMaster()
  renderer = object.__new__(module.HudRenderer)

  assert renderer._get_speed_limit_info() == (80, 2, 90)
  assert fake_ui_state.sm.calls == 1


def test_drive_card_uses_one_limit_snapshot_and_keeps_draw_order(hud_module, monkeypatch):
  module, fake_ui_state = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._debug_speed_panel = False
  renderer.speed = 40.0
  renderer.is_cruise_set = False
  renderer._engaged = False
  renderer._cruise_gap_levels = module.GAP_LEVELS_MAX
  calls = []
  labels = []

  renderer._type = SimpleNamespace(
    width=lambda text, size, weight, spacing=0.0: float(len(text) * 10),
    draw=lambda text, *args, **kwargs: (labels.append(text), 20.0)[1],
    draw_mid=lambda text, *args, **kwargs: 20.0,
  )
  monkeypatch.setattr(renderer, "_cruise_text", lambda: (calls.append("cruise"), "--")[1])
  monkeypatch.setattr(renderer, "_update_cruise_speed_animation", lambda text: calls.append("animation"))
  limit_reads = []

  def limit_state():
    limit_reads.append(True)
    calls.append("limit")
    return 0, False, False

  monkeypatch.setattr(renderer, "_limit_state", limit_state)
  monkeypatch.setattr(renderer, "_traffic_light", lambda: calls.append("light"))
  monkeypatch.setattr(renderer, "_override_chip", lambda: (calls.append("chip"), None)[1])
  monkeypatch.setattr(renderer, "_get_gear_text", lambda: (calls.append("gear"), "D")[1])
  monkeypatch.setattr(renderer, "_get_cruise_gap", lambda: (calls.append("gap"), 3)[1])
  monkeypatch.setattr(renderer, "_get_driving_mode_text_and_color", lambda: (calls.append("mode"), ("", module.hs.TEXT))[1])

  renderer._draw_drive_card(module.rl.Rectangle(10, 20, 1000, 600))

  assert limit_reads == [True]
  assert calls == ["cruise", "animation", "limit", "light", "chip", "gear", "gap", "mode"]
  assert labels == ["SET", "--", "40", "km/h"]


def test_device_info_updates_only_on_first_frame_or_new_service_frame(hud_module, monkeypatch):
  module, fake_ui_state = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._device_info_loaded = False
  renderer._device_info_recv_frames = (-1, -1)
  renderer.is_cruise_set = True
  renderer.set_speed = 10
  renderer.speed = 20
  fake_ui_state.started_frame = 2
  fake_ui_state.sm = SimpleNamespace(
    recv_frame={"carState": 1, "deviceState": 10, "peripheralState": 20},
  )
  refreshes = []

  def update_device_info():
    refreshes.append(True)
    renderer._device_info_loaded = True

  monkeypatch.setattr(renderer, "_update_device_info", update_device_info)

  renderer._update_state()
  assert len(refreshes) == 1

  renderer._update_state()
  assert len(refreshes) == 1

  fake_ui_state.sm.recv_frame["deviceState"] = 11
  renderer._update_state()
  assert len(refreshes) == 2

  fake_ui_state.sm.recv_frame["peripheralState"] = 21
  renderer._update_state()
  assert len(refreshes) == 3


def test_device_display_cache_follows_service_receive_frames(hud_module):
  module, fake_ui_state = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._device_info_loaded = False
  renderer._device_info_recv_frames = (-1, -1)
  renderer.is_cruise_set = True
  renderer.set_speed = 10
  renderer.speed = 20
  device_state = SimpleNamespace(
    freeSpacePercent=55.0,
    memoryUsagePercent=20,
    cpuTempC=[40.0, 42.0],
    cpuUsagePercent=[10.0, 20.0],
  )
  peripheral_state = SimpleNamespace(voltage=12_500)

  class FakeSubMaster(dict):
    pass

  sm = FakeSubMaster(deviceState=device_state, peripheralState=peripheral_state)
  sm.recv_frame = {"carState": 1, "deviceState": 10, "peripheralState": 20}
  fake_ui_state.started_frame = 2
  fake_ui_state.sm = sm

  renderer._update_state()
  assert renderer._device_info_recv_frames == (10, 20)
  assert (
    renderer._cpu_temp_text,
    renderer._memory_usage_text,
    renderer._disk_usage_text,
    renderer._voltage_text,
  ) == ("41°C", "20%", "45%", "12.5V")

  device_state.cpuTempC = [50.0]
  device_state.memoryUsagePercent = 30
  device_state.freeSpacePercent = 40.0
  peripheral_state.voltage = 13_250
  renderer._update_state()
  assert (
    renderer._cpu_temp_text,
    renderer._memory_usage_text,
    renderer._disk_usage_text,
    renderer._voltage_text,
  ) == ("41°C", "20%", "45%", "12.5V")

  sm.recv_frame["deviceState"] = 11
  renderer._update_state()
  assert renderer._device_info_recv_frames == (11, 20)
  assert (
    renderer._cpu_temp_text,
    renderer._memory_usage_text,
    renderer._disk_usage_text,
    renderer._voltage_text,
  ) == ("50°C", "30%", "60%", "13.2V")


def test_incomplete_device_info_is_retried(hud_module):
  module, fake_ui_state = hud_module
  renderer = object.__new__(module.HudRenderer)
  device_state = SimpleNamespace(
    freeSpacePercent=55.0,
    memoryUsagePercent=20,
    cpuUsagePercent=[10.0, 20.0],
  )
  peripheral_state = SimpleNamespace(voltage=12_500)
  fake_ui_state.sm = {
    "deviceState": device_state,
    "peripheralState": peripheral_state,
  }

  renderer._update_device_info()
  assert renderer._device_info_loaded is False

  device_state.cpuTempC = [40.0, 42.0]
  renderer._update_device_info()
  assert renderer._device_info_loaded is True
  assert renderer._cpu_temp == pytest.approx(41.0)
  assert renderer._cpu_usage == pytest.approx(15.0)
  assert renderer._voltage == pytest.approx(12.5)
  assert renderer._cpu_temp_text == "41°C"
  assert renderer._memory_usage_text == "20%"
  assert renderer._disk_usage_text == "45%"
  assert renderer._voltage_text == "12.5V"


def test_round_box_reuses_scratch_rectangle_without_changing_draw_geometry(hud_module, monkeypatch):
  module, _ = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._round_box_rect = module.rl.Rectangle(0, 0, 0, 0)
  fill_color = module.rl.Color(1, 2, 3, 4)
  line_color = module.rl.Color(5, 6, 7, 8)
  calls = []

  def snapshot(kind, rect, *args):
    calls.append((kind, id(rect), (rect.x, rect.y, rect.width, rect.height), args))

  monkeypatch.setattr(module.rl, "draw_rectangle_rounded", lambda rect, *args: snapshot("fill", rect, *args))
  monkeypatch.setattr(module.rl, "draw_rectangle_rounded_lines_ex", lambda rect, *args: snapshot("line", rect, *args))

  renderer._draw_round_box(1, 2, 3, 4, fill_color, line_color, 0.25, 8, 2)
  renderer._draw_round_box(5, 6, 7, 8, fill_color, line_color, 0.5, 4, 3)

  assert [call[0] for call in calls] == ["fill", "line", "fill", "line"]
  assert len({call[1] for call in calls}) == 1
  assert [call[2] for call in calls] == [
    (1.0, 2.0, 3.0, 4.0),
    (1.0, 2.0, 3.0, 4.0),
    (5.0, 6.0, 7.0, 8.0),
    (5.0, 6.0, 7.0, 8.0),
  ]
  assert calls[0][3] == (0.25, 8, fill_color)
  assert calls[1][3] == (0.25, 8, 2.0, line_color)
  assert calls[2][3] == (0.5, 4, fill_color)
  assert calls[3][3] == (0.5, 4, 3.0, line_color)


@pytest.mark.parametrize(("value", "expected_text", "expected_color_name"), (
  (4.9, "  -", "WHITE_220"),
  (5.0, "5", "TPMS_LOW"),
  (5.1, "5", "TPMS_LOW"),
  (30.9, "31", "TPMS_LOW"),
  (31.0, "31", "WHITE_220"),
  (31.1, "31", "WHITE_220"),
  (59.9, "60", "WHITE_220"),
  (60.0, "60", "WHITE_220"),
  (60.1, "  -", "WHITE_220"),
))
def test_tpms_legacy_display_boundaries(hud_module, value, expected_text, expected_color_name):
  module, _ = hud_module
  renderer = object.__new__(module.HudRenderer)

  assert renderer._get_tpms_text(value) == expected_text
  assert renderer._get_tpms_color(value) == getattr(module.COLORS, expected_color_name)


@pytest.mark.parametrize(("show_tpms", "top", "drawn"), (
  (0, True, False), (0, False, False),
  (1, True, True), (1, False, False),
  (2, True, False), (2, False, True),
  (3, True, True), (3, False, True),
))
def test_tpms_draws_only_when_that_hud_edge_is_enabled(hud_module, monkeypatch, show_tpms, top, drawn):
  module, fake_ui_state = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._show_tpms = show_tpms
  fake_ui_state.sm = {
    "carState": SimpleNamespace(tpms=SimpleNamespace(fl=31, fr=32, rl=33, rr=34)),
  }
  module.hs.zones.clear()
  cards = []
  cars = []
  labels = []
  monkeypatch.setattr(module.hs, "glass_card", lambda x, y, w, h, *args, **kwargs: cards.append((x, y, w, h)))
  monkeypatch.setattr(renderer, "_draw_tpms_car", lambda cx, cy, lows: cars.append((cx, cy, lows)))
  renderer._type = SimpleNamespace(
    draw=lambda text, *args, **kwargs: (labels.append(text), 20.0)[1],
    width=lambda *args, **kwargs: 20.0,
    draw_mid=lambda *args, **kwargs: None,
  )
  rect = module.rl.Rectangle(10, 50, 1000, 750)

  renderer._draw_tpms(rect, top)

  assert (len(cards) == 1) == drawn
  if drawn:
    x = rect.x + rect.width - module.M_X - module.TPMS_W
    y = rect.y + module.M_TOP if top else rect.y + rect.height - module.M_BOTTOM - module.TPMS_H
    assert cards[0] == (x, y, module.TPMS_W, module.TPMS_H)
    assert len(cars) == 1
    assert labels == ["31", "32", "33", "34"]
  else:
    assert cars == []


def test_clock_reformats_only_when_minute_key_changes(hud_module, monkeypatch):
  module, _ = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._show_date_time = 1
  renderer._date_time_minute_key = None
  renderer._date_time_text = ""
  renderer._date_text = ""
  module.hs.zones.clear()
  module.hs.top_boxes.clear()
  draws = []
  renderer._type = SimpleNamespace(
    draw=lambda text, *args, **kwargs: (draws.append(text), 20.0)[1],
    width=lambda *args, **kwargs: 20.0,
    draw_mid=lambda *args, **kwargs: None,
  )
  moments = iter((
    stdlib_time.struct_time((2026, 7, 16, 12, 1, 1, 3, 197, 0)),
    stdlib_time.struct_time((2026, 7, 16, 12, 1, 59, 3, 197, 0)),
    stdlib_time.struct_time((2026, 7, 16, 12, 2, 0, 3, 197, 0)),
    stdlib_time.struct_time((2026, 7, 16, 13, 2, 0, 3, 197, 0)),
  ))
  localtime_calls = []
  strftime_calls = []
  real_strftime = stdlib_time.strftime

  def fake_localtime():
    localtime_calls.append(True)
    return next(moments)

  def fake_strftime(fmt, now):
    strftime_calls.append((fmt, now.tm_hour, now.tm_min))
    return real_strftime(fmt, now)

  monkeypatch.setattr(module.time, "localtime", fake_localtime)
  monkeypatch.setattr(module.time, "strftime", fake_strftime)
  rect = module.rl.Rectangle(0, 0, 1000, 600)

  for _ in range(4):
    renderer._draw_clock(rect)

  assert len(localtime_calls) == 4
  # The wall-clock string is rebuilt only when the minute key changes.
  assert strftime_calls == [("%H:%M", 12, 1), ("%H:%M", 12, 2), ("%H:%M", 13, 2)]
  assert draws == ["12:01", "7월 16일 (목)", "12:01", "7월 16일 (목)",
                   "12:02", "7월 16일 (목)", "13:02", "7월 16일 (목)"]
  assert renderer._date_time_minute_key == (2026, 197, 13, 2, 0)
  assert renderer._date_text == "7월 16일 (목)"

  renderer._show_date_time = 0
  monkeypatch.setattr(module.time, "localtime", lambda: pytest.fail("hidden clock read the clock"))
  renderer._draw_clock(rect)
  assert len(draws) == 8


def test_render_draws_each_hud_section_in_order(hud_module, monkeypatch):
  module, _ = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer.is_cruise_available = True
  renderer._show_plot_mode = 6
  renderer._font_display = object()
  renderer._blink_timer = 0
  renderer._disp_timer = 0
  renderer._tlv_dsp_notice = SimpleNamespace(draw=lambda rect: False)
  calls = []

  renderer._plot_renderer = SimpleNamespace(
    draw=lambda rect, font, mode: calls.append(("plot", mode)),
  )
  monkeypatch.setattr(renderer, "_refresh_hud_params", lambda now: calls.append(("params", now)))
  monkeypatch.setattr(renderer, "_get_turn_info_hud_data", lambda: (calls.append("turns"), {})[1])
  monkeypatch.setattr(renderer, "_draw_guidance_card", lambda rect, info: calls.append("guide"))
  monkeypatch.setattr(renderer, "_draw_tpms", lambda rect, top: calls.append(("tpms", top)))
  monkeypatch.setattr(renderer, "_draw_system_card", lambda rect: calls.append("system"))
  monkeypatch.setattr(renderer, "_draw_trip", lambda rect, info: calls.append("trip"))
  monkeypatch.setattr(renderer, "_draw_drive_card", lambda rect: calls.append("drive"))
  monkeypatch.setattr(renderer, "_draw_clock", lambda rect: calls.append("clock"))
  monkeypatch.setattr(renderer, "_draw_egpu_badge", lambda rect: calls.append("egpu"))
  monkeypatch.setattr(renderer, "_draw_cruise_speed_animation", lambda rect: calls.append("animation"))
  monkeypatch.setattr(module.rl, "draw_rectangle_gradient_v", lambda *args: calls.append("shade"))
  monkeypatch.setattr(module.time, "monotonic", lambda: 12.5)

  renderer._render(module.rl.Rectangle(0, 0, 1000, 600))

  assert calls == [
    ("params", 12.5),
    "shade",
    "shade",
    "turns",
    "guide",
    ("tpms", True),
    "system",
    "trip",
    ("tpms", False),
    "drive",
    ("plot", 6),
    "clock",
    "egpu",
    "animation",
  ]


def test_vehicle_navigation_profile_does_not_force_speed_with_cruise_off(hud_module):
  module, _ = hud_module
  sm = {
    "longitudinalPlan": SimpleNamespace(cruiseTarget=0.0),
    "carrotMan": SimpleNamespace(
      vehicleNaviActive=True,
      vehicleNaviSpeed=105,
      desiredSpeed=105,
      desiredSource="hda_section",
    ),
  }

  override = module.SetSpeedOverride().compute(sm, set_speed_kph=105)

  assert not override.active
  assert override.speed_kph == 105
  assert override.label == "MAX"
  assert override.speed_color_mode == 0
  assert not override.force_persist


@pytest.mark.parametrize("dist, text", [
  (0, ""), (7, "10 m"), (83, "90 m"), (120, "120 m"), (299, "300 m"), (300, "300 m"),
  (347, "350 m"), (951, "1.0 km"), (1287, "1.3 km"),
])
def test_turn_distance_counts_down_in_calm_steps(hud_module, dist, text):
  module, fake_ui_state = hud_module
  fake_ui_state.is_metric = True
  renderer = object.__new__(module.HudRenderer)

  assert renderer._format_turn_distance_text(dist) == text


def turn_info(turn=0, dist=0, nxt=0, nxt_dist=0):
  return {"x_turn_info": turn, "x_dist_to_turn": dist, "x_turn_info_next": nxt, "x_dist_to_turn_next": nxt_dist}


@pytest.mark.parametrize("info, expected", [
  (turn_info(2, 350, 1, 530), (2, 350, 1, 180)),  # the turn after next, 180 m past the first
  (turn_info(2, 350), (2, 350, 0, 0)),            # no turn after next
  (turn_info(2, 350, 1, 300), (2, 350, 0, 0)),    # a stale next short of the first is ignored
  (turn_info(2, -10, 1, 170), (1, 170, 0, 0)),    # past the first turn: its successor is promoted
  (turn_info(-1, 0, 1, 170), (1, 170, 0, 0)),
  (turn_info(-1, 0), (0, 0, 0, 0)),
])
def test_turn_resolution_promotes_the_next_turn_once_past(hud_module, info, expected):
  module, _ = hud_module

  assert module.HudRenderer._resolve_turn(info) == expected


def test_a_short_feed_dropout_keeps_the_last_turn(hud_module):
  module, _ = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._guide_turn = None

  assert renderer._held_turn(turn_info(2, 350, 1, 530), True, 10.0) == (2, 350, 1, 180)
  assert renderer._held_turn(turn_info(), True, 11.9) == (2, 350, 1, 180)
  assert renderer._held_turn(turn_info(), False, 11.9) == (2, 350, 1, 180)
  assert renderer._held_turn(turn_info(), True, 12.0) == (0, 0, 0, 0)
  assert renderer._guide_turn is None


@pytest.mark.parametrize("metric, dist, text", [
  (True, 0, ""), (True, 237, "237 m"), (True, 999, "999 m"), (True, 1000, "1.0 km"), (True, 1234, "1.2 km"),
  (False, 100, "328 ft"), (False, 2000, "1.2 mi"),
])
def test_camera_distance_text_shows_the_exact_metres(hud_module, metric, dist, text):
  module, fake_ui_state = hud_module
  fake_ui_state.is_metric = metric
  renderer = object.__new__(module.HudRenderer)

  assert renderer._format_camera_distance_text(dist) == text


def test_camera_distance_is_latched_for_a_second_and_refreshes_at_once_for_a_new_camera(hud_module):
  module, _ = hud_module
  renderer = object.__new__(module.HudRenderer)
  renderer._camera_dist_held = None
  key = ("camera", 2)

  assert renderer._camera_display_dist(key, 300, 0.0) == 300
  assert renderer._camera_display_dist(key, 292, 0.5) == 300  # held: within the refresh second
  assert renderer._camera_display_dist(key, 285, 1.0) == 285  # one second has passed
  assert renderer._camera_display_dist(("camera", 3), 240, 1.2) == 240  # a new camera shows its value at once
  assert renderer._camera_display_dist(("camera", 3), 260, 1.3) == 260  # rising distance (new camera) at once


def held_camera_renderer(module, fake_ui_state):
  renderer = object.__new__(module.HudRenderer)
  renderer._camera_dist_held = None
  renderer._progress_key = None
  renderer._progress_start = module.NAV_PROGRESS_SPAN
  renderer._progress_last = 0
  renderer._guide_turn = None
  fake_ui_state.sm = {"carrotMan": SimpleNamespace(xSpdLimit=80, xSignType=2, nRoadLimitSpeed=90)}
  return renderer


def camera_card_draws(renderer, module, monkeypatch, dist, now):
  numbers = []
  renderer._type = SimpleNamespace(
    draw=lambda text, *args, **kwargs: (numbers.append(text), 20.0)[1],
    width=lambda *args, **kwargs: 20.0,
    draw_mid=lambda *args, **kwargs: None,
    ellipsize=lambda text, *args, **kwargs: text,
  )
  monkeypatch.setattr(module.hs, "glass_card", lambda *args, **kwargs: None)
  monkeypatch.setattr(module.hs, "regulatory_sign", lambda *args, **kwargs: None)
  monkeypatch.setattr(module.time, "monotonic", lambda: now)
  module.hs.zones.clear()
  module.hs.top_boxes.clear()
  info = {
    "x_spd_dist": dist,
    "n_go_pos_dist": 0,
    "n_go_pos_time": 0,
    "x_turn_info": 0,
    "x_dist_to_turn": 0,
    "x_turn_info_next": 0,
    "x_dist_to_turn_next": 0,
    "atc_type": "",
    "sdi_descr": "과속 단속",
  }
  renderer._draw_guidance_card(module.rl.Rectangle(10, 20, 1000, 600), info)
  return numbers[0] if numbers else None


def test_camera_card_holds_the_number_within_a_second_and_changes_after(hud_module, monkeypatch):
  module, fake_ui_state = hud_module
  renderer = held_camera_renderer(module, fake_ui_state)

  assert camera_card_draws(renderer, module, monkeypatch, 300, 0.0) == "300"
  assert camera_card_draws(renderer, module, monkeypatch, 292, 0.5) == "300"
  assert camera_card_draws(renderer, module, monkeypatch, 285, 1.0) == "285"


def test_camera_card_resets_the_latch_when_the_camera_comes_back(hud_module, monkeypatch):
  module, fake_ui_state = hud_module
  renderer = held_camera_renderer(module, fake_ui_state)

  assert camera_card_draws(renderer, module, monkeypatch, 300, 0.0) == "300"
  assert camera_card_draws(renderer, module, monkeypatch, 0, 0.4) is None  # camera gone: nothing drawn
  assert renderer._camera_dist_held is None
  assert camera_card_draws(renderer, module, monkeypatch, 250, 0.6) == "250"  # new camera shows at once


def _drive_card_probe(module, monkeypatch, *, limit, light):
  """Draw the drive card once and return what the speed/sign/light layout used.

  The text drawer is replaced, so the size passed for the speed numerals is the
  value under test rather than a rendered result.
  """
  renderer = object.__new__(module.HudRenderer)
  renderer._debug_speed_panel = False
  renderer.speed = 40.0
  renderer.is_cruise_set = False
  renderer._engaged = False
  renderer._cruise_gap_levels = module.GAP_LEVELS_MAX

  probed = {"speed_sizes": [], "sign_x": [], "chip_right": None, "light_rect": None}

  def draw(text, x, y, size, *args, **kwargs):
    if text == "40":
      probed["speed_sizes"].append(size)
    return 20.0

  # Size-proportional metrics: the speed fit must see a real width, otherwise
  # every branch clamps at the 1.0 ceiling and the shrink is invisible.
  renderer._type = SimpleNamespace(
    width=lambda text, size, weight, spacing=0.0: 0.55 * size * len(text),
    draw=draw,
    draw_mid=lambda text, *args, **kwargs: 20.0,
  )
  monkeypatch.setattr(renderer, "_cruise_text", lambda: "--")
  monkeypatch.setattr(renderer, "_update_cruise_speed_animation", lambda text: None)
  monkeypatch.setattr(renderer, "_limit_state", lambda: (limit, False, False))
  monkeypatch.setattr(renderer, "_traffic_light", lambda: light)
  monkeypatch.setattr(renderer, "_override_chip",
                      lambda: ("accel", "50", module.hs.LIVE_GREEN, False))
  monkeypatch.setattr(renderer, "_get_gear_text", lambda: "D")
  monkeypatch.setattr(renderer, "_get_cruise_gap", lambda: 3)
  monkeypatch.setattr(renderer, "_get_driving_mode_text_and_color", lambda: ("", module.hs.TEXT))

  # Record the sign centre and the light capsule. The stub raylib has no
  # draw_circle_gradient, so the real sign is not replayed; only its x matters.
  real_card = module.hs.card

  def sign(cx, cy, value, type_, k=1.0, glow_alarm=False):
    probed["sign_x"].append(cx)

  # The light's backing capsule is the only card filled with translucent black,
  # which identifies it on both the old and the new layout.
  light_fill = module.hs.rgba(0, 0, 0, 160)

  def card(x, y, w, h, fill=None, *args, **kwargs):
    if light is not None and fill == light_fill:
      probed["light_rect"] = (x, y, w, h)
    return real_card(x, y, w, h, fill, *args, **kwargs)

  monkeypatch.setattr(module.hs, "regulatory_sign", sign)
  monkeypatch.setattr(module.hs, "card", card)
  # The stub raylib has no circle primitives; the light's glow and dots are not
  # under test, only the geometry of the capsule and the chip.
  monkeypatch.setattr(module.hs, "glow", lambda *a, **k: None)
  monkeypatch.setattr(module.hs, "dot", lambda *a, **k: None)

  real_draw_mid = renderer._type.draw_mid
  def draw_mid(text, x, y, size, *args, **kwargs):
    if text == "50":  # the override chip's value: record its right edge
      probed["chip_right"] = x + 0.55 * size * len(text)
    return real_draw_mid(text, x, y, size, *args, **kwargs)
  renderer._type.draw_mid = draw_mid

  renderer._draw_drive_card(module.rl.Rectangle(10, 20, 1000, 600))
  return probed


def test_traffic_light_never_resizes_the_speed_or_moves_the_limit_sign(hud_module, monkeypatch):
  """U2: the light is a capsule in the corner; the sign and speed keep their place and size."""
  module, _ = hud_module
  without = _drive_card_probe(module, monkeypatch, limit=60, light=None)
  with_light = _drive_card_probe(module, monkeypatch, limit=60, light="red")

  assert without["speed_sizes"] and with_light["speed_sizes"]
  assert without["speed_sizes"][-1] == with_light["speed_sizes"][-1], (
    "the speed numerals resized when the traffic light appeared")
  assert without["sign_x"] and with_light["sign_x"]
  assert without["sign_x"][-1] == with_light["sign_x"][-1], (
    "the limit sign moved when the traffic light appeared")


def test_override_chip_stays_left_of_the_light_capsule(hud_module, monkeypatch):
  module, _ = hud_module
  probed = _drive_card_probe(module, monkeypatch, limit=60, light="red")

  assert probed["light_rect"] is not None, "the light capsule was not drawn"
  assert probed["chip_right"] is not None, "the override chip was not drawn"
  chip_right = probed["chip_right"]
  light_x = probed["light_rect"][0]
  assert chip_right <= light_x, f"chip right {chip_right:.0f} overlaps the light at x={light_x:.0f}"


def _system_card_probe(module, fake_ui_state, monkeypatch, *, active_carrot=0, gps_fix=False, vertices=0,
                       device=True, cpu_temp=55.0, memory_usage=48, disk_text="91%", volt_text="12.2V",
                       disp_timer=0, blink_timer=0, zones=None, rect=None):
  """Draw the right-column system card once and return its card box, rules and text runs.

  `active_carrot`, `gps_fix` and `vertices` are the real inputs of `_system_links`, so the
  badges under test are the ones the renderer decides on. The text drawer is replaced with
  size-proportional metrics, so the sizes are the values under test rather than a render.
  """
  fake_ui_state.sm = {
    "carrotMan": SimpleNamespace(activeCarrot=active_carrot, navPathVertexCount=vertices),
    "gpsLocationExternal": SimpleNamespace(hasFix=gps_fix),
  }
  renderer = object.__new__(module.HudRenderer)
  renderer._show_device_state = 1 if device else 0
  renderer._blink_timer = blink_timer
  renderer._disp_timer = disp_timer
  renderer._cpu_temp = cpu_temp
  renderer._memory_usage = memory_usage
  renderer._memory_usage_text = f"{memory_usage}%"
  renderer._disk_usage_text = disk_text
  renderer._voltage_text = volt_text

  texts = []

  def draw_mid(text, x, middle, size, color, weight, align=0.0, spacing=0.0):
    width = 0.55 * size * len(text)
    left = x - width * align
    texts.append(SimpleNamespace(text=text, left=left, right=left + width, size=size, color=color))
    return width

  renderer._type = SimpleNamespace(
    width=lambda text, size, weight=None, spacing=0.0: 0.55 * size * len(text),
    draw_mid=draw_mid,
  )
  module.hs.zones.clear()
  module.hs.top_boxes.clear()
  module.hs.zones.update(zones or {})
  cards, rules = [], []
  monkeypatch.setattr(module.hs, "glass_card", lambda x, y, w, h, *args, **kwargs: cards.append((x, y, w, h)))
  monkeypatch.setattr(module.hs, "card", lambda *args, **kwargs: rules.append(args))

  renderer._draw_system_card(rect if rect is not None else module.rl.Rectangle(10, 50, 1000, 750))
  return SimpleNamespace(cards=cards, rules=rules, texts=texts, renderer=renderer)


def test_system_card_is_exactly_tpms_width_and_right_aligned(hud_module, monkeypatch):
  """The merged card takes the tyre card's width and stacks under it, not beside the road."""
  module, fake_ui_state = hud_module
  rect = module.rl.Rectangle(10, 50, 1000, 750)
  tpms = (rect.x + rect.width - module.M_X - module.TPMS_W, rect.y + module.M_TOP, module.TPMS_W, module.TPMS_H)

  probe = _system_card_probe(module, fake_ui_state, monkeypatch, active_carrot=1, gps_fix=True, vertices=2,
                             device=True, zones={"tpms": tpms}, rect=rect)

  assert len(probe.cards) == 1, "the card was not drawn exactly once"
  x, top, w, h = probe.cards[0]
  assert w == module.TPMS_W
  assert x + w == rect.x + rect.width - module.M_X
  assert top == tpms[1] + tpms[3] + module.STACK_GAP, "the card does not stack under the tyre card"
  assert module.hs.zones["status"] == (x, top, w, h)
  assert module.hs.top_boxes == [(x, top, w, h)]


@pytest.mark.parametrize(("links_on", "device", "rows"), (
  (True, False, "links"),
  (False, True, "device"),
  (True, True, "both"),
  (False, False, None),
))
def test_system_card_height_follows_its_rows(hud_module, monkeypatch, links_on, device, rows):
  module, fake_ui_state = hud_module

  probe = _system_card_probe(module, fake_ui_state, monkeypatch, active_carrot=1 if links_on else 0,
                             gps_fix=links_on, vertices=2 if links_on else 0, device=device)
  expected = {
    "links": module.SYS_LINK_H,
    "device": module.SYS_DEVICE_H,
    "both": module.SYS_LINK_H + module.SYS_DEVICE_H,
  }

  if rows is None:
    assert probe.cards == [], "an empty card was drawn"
    assert "status" not in module.hs.zones
  else:
    assert len(probe.cards) == 1
    assert probe.cards[0][3] == expected[rows]


def test_system_card_text_stays_inside_the_card(hud_module, monkeypatch):
  """Worst case (100°, 100%, 14.1V with every badge lit) still fits the card's padding."""
  module, fake_ui_state = hud_module

  probe = _system_card_probe(module, fake_ui_state, monkeypatch, active_carrot=1, gps_fix=True, vertices=2,
                             device=True, cpu_temp=100.0, memory_usage=100, volt_text="14.1V", disp_timer=32)

  x, _, w, _ = probe.cards[0]
  lo, hi = x + module.SYS_PAD - 1, x + module.SYS_W - module.SYS_PAD + 1
  assert probe.texts
  for item in probe.texts:
    assert lo <= item.left and item.right <= hi, (
      f"{item.text!r} spans {item.left:.0f}..{item.right:.0f}, outside {lo:.0f}..{hi:.0f}")

  values = [item for item in probe.texts if item.text in ("100°", "100%", "14.1V")]
  assert [item.text for item in values] == ["100°", "100%", "14.1V"]
  for i in range(len(values) - 1):
    left, right = values[i], values[i + 1]
    assert right.left - left.right >= module.SYS_MIN_GAP - 1, (
      f"{left.text!r} and {right.text!r} are {right.left - left.right:.1f} px apart")
  assert max(item.size for item in values) < module.SYS_VALUE_SIZE, "the worst case kept the full value size"


def test_normal_readings_keep_the_full_value_size(hud_module, monkeypatch):
  """Everyday readings have room to spare, so the shrinking path never touches them."""
  module, fake_ui_state = hud_module

  probe = _system_card_probe(module, fake_ui_state, monkeypatch, device=True, cpu_temp=48.0,
                             memory_usage=42, disk_text="45%")

  values = [item for item in probe.texts if item.text in ("48°", "42%", "45%")]
  assert [item.text for item in values] == ["48°", "42%", "45%"]
  assert [item.size for item in values] == [module.SYS_VALUE_SIZE] * 3


def test_hot_values_blink_red_and_links_keep_their_colour(hud_module, monkeypatch):
  module, fake_ui_state = hud_module

  probe = _system_card_probe(module, fake_ui_state, monkeypatch, active_carrot=1, gps_fix=True, vertices=2,
                             device=True, cpu_temp=85.0, blink_timer=0)
  drawn = {item.text: item for item in probe.texts}

  assert drawn["85°"].color == module.hs.WARN_RED, "a hot CPU did not blink red"
  assert drawn["APM"].color == module.NAV_BLUE
  assert drawn["GPS"].color == module.hs.LIVE_GREEN
  assert drawn["ROUTE"].color == module.hs.LIVE_GREEN
  for tag in ("CPU", "MEM", "DISK"):
    assert drawn[tag].color == module.hs.TEXT_2, f"the {tag} tag is not drawn bright"

  probe = _system_card_probe(module, fake_ui_state, monkeypatch, active_carrot=2, gps_fix=True, vertices=2,
                             device=True, cpu_temp=85.0, blink_timer=0)
  drawn = {item.text: item for item in probe.texts}

  assert "APM" not in drawn, "the blue APM badge stayed up once APN took over"
  assert drawn["APN"].color == module.hs.LIVE_GREEN
