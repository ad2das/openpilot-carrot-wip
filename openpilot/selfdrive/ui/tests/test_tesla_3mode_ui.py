"""Unit tests for Tesla v15 3-Mode Swipe Hybrid UI (3D FSD default, AR, Quiet)."""
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ONROAD_DIR = Path(__file__).parents[1] / "onroad"


@pytest.fixture
def tesla_env(monkeypatch):
  @dataclass(frozen=True)
  class Color:
    r: int
    g: int
    b: int
    a: int

  @dataclass
  class Rectangle:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

  @dataclass
  class Vector2:
    x: float = 0.0
    y: float = 0.0

  @dataclass
  class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

  class Camera3D:
    def __init__(self, position, target, up, fovy, projection):
      self.position = position
      self.target = target
      self.up = up
      self.fovy = fovy
      self.projection = projection

  raylib = types.ModuleType("pyray")
  raylib.Color = Color
  raylib.Rectangle = Rectangle
  raylib.Vector2 = Vector2
  raylib.Vector3 = Vector3
  raylib.Camera3D = Camera3D
  raylib.Texture = object
  raylib.Font = object
  raylib.Mesh = SimpleNamespace
  raylib.WHITE = Color(255, 255, 255, 255)
  raylib.BLACK = Color(0, 0, 0, 255)
  raylib.CameraProjection = SimpleNamespace(CAMERA_PERSPECTIVE=0)
  raylib.BlendMode = SimpleNamespace(BLEND_ADDITIVE=1)
  raylib.TextureFilter = SimpleNamespace(TEXTURE_FILTER_BILINEAR=1, TEXTURE_FILTER_TRILINEAR=2)
  raylib.ShaderUniformDataType = SimpleNamespace(SHADER_UNIFORM_FLOAT=0, SHADER_UNIFORM_VEC3=2)
  raylib.RL_QUADS = 7
  raylib.RL_TRIANGLES = 4
  for fn_name in (
    "draw_line_ex", "draw_rectangle_gradient_v", "draw_rectangle_rounded",
    "draw_rectangle_rounded_lines_ex", "draw_text_ex", "draw_texture_pro",
    "draw_triangle", "draw_circle", "draw_circle_v", "draw_ring",
  ):
    setattr(raylib, fn_name, lambda *a, **k: None)

  class Widget:
    def __init__(self):
      self._click_callback = None

  fake_ui_state = SimpleNamespace(
    params=SimpleNamespace(get_int=lambda k: 4 if k == "CruiseGapLevels" else 0),
    sm=None,
    started=True,
    started_frame=0,
    status=1,
    lat_active=True,
    is_metric=True,
    usbgpu_present=False,
  )
  gui_app = SimpleNamespace(
    font=lambda w: ("font", w),
    has_font=lambda w: False,
    texture=lambda p: ("tex", p),
    width=2160,
    height=1080,
  )

  stubs = {
    "pyray": raylib,
    "openpilot.common.constants": SimpleNamespace(CV=SimpleNamespace(MS_TO_KPH=3.6, MS_TO_MPH=2.2369363)),
    "openpilot.system.hardware.usbgpu": SimpleNamespace(usbgpu_badge_state=lambda *a, **k: "ready"),
    "openpilot.selfdrive.ui.ui_state": SimpleNamespace(
      ui_state=fake_ui_state,
      UIStatus=SimpleNamespace(ENGAGED=1, DISENGAGED=2, OVERRIDE=3),
    ),
    "openpilot.system.ui.lib.application": SimpleNamespace(
      gui_app=gui_app,
      GL_VERSION="#version 300 es\n",
      FontWeight=SimpleNamespace(SEMI_BOLD=1, BOLD=2, MEDIUM=3, DISPLAY=4),
      FONT_DIR=Path("nonexistent"),
      KR_ATLAS_SCALE=1.25,
      KR_ATLAS_RAISE=0.125,
    ),
    "openpilot.system.ui.lib.shader_polygon": SimpleNamespace(
      draw_polygon=lambda *a, **k: None,
      draw_polygon_solid=lambda *a, **k: None,
      draw_polygons=lambda *a, **k: None,
      Gradient=SimpleNamespace,
    ),
    "openpilot.selfdrive.ui.onroad.plot_renderer": SimpleNamespace(
      PlotRenderer=lambda: SimpleNamespace(draw=lambda *a, **k: None),
    ),
    "openpilot.system.ui.widgets": SimpleNamespace(Widget=Widget),
  }
  for name, stub in stubs.items():
    monkeypatch.setitem(sys.modules, name, stub)
  monkeypatch.setitem(sys.modules, "openpilot.selfdrive.ui.onroad.hud_style", SimpleNamespace(
    rgba=lambda r, g, b, a=255: Color(r, g, b, a),
    with_alpha=lambda c, a: Color(c.r, c.g, c.b, a),
    Type=lambda: SimpleNamespace(width=lambda *a, **k: 20.0, draw=lambda *a, **k: 20.0, draw_mid=lambda *a, **k: 20.0, ellipsize=lambda t, *a, **k: t),
    TEXT=Color(255, 255, 255, 255),
    TILE_AMBER=(Color(255, 179, 42, 255), Color(240, 138, 0, 255)),
    BOLD=2, SEMI=1, MEDIUM=3,
    zones={}, top_boxes=[], panels=[],
  ))

  def load(mod_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(mod_name, ONROAD_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    return mod

  t3d = load("_test_tesla_3d", "tesla_3d_renderer.py")
  thud = load("_test_tesla_hud", "tesla_hud.py")
  yield t3d, thud, fake_ui_state


def test_default_mode_is_3d_fsd_and_stepping_clamps(tesla_env):
  _, thud, _ = tesla_env
  hud = thud.TeslaHudRenderer()
  assert hud.mode == thud.TESLA_MODE_3D

  assert hud.step_mode(-1) is False
  assert hud.mode == thud.TESLA_MODE_3D

  assert hud.step_mode(1) is True
  assert hud.mode == thud.TESLA_MODE_AR

  assert hud.step_mode(1) is True
  assert hud.mode == thud.TESLA_MODE_QUIET

  assert hud.step_mode(1) is False
  assert hud.mode == thud.TESLA_MODE_QUIET

  assert hud.step_mode(-1) is True
  assert hud.mode == thud.TESLA_MODE_AR


def test_dock_hit_testing_tabs_and_handle(tesla_env):
  _, thud, _ = tesla_env
  hud = thud.TeslaHudRenderer()
  hud._dock_tab_rects = [(700.0, 980.0, 154.0, 54.0), (862.0, 980.0, 184.0, 54.0), (1054.0, 980.0, 148.0, 54.0)]
  hud._handle_rect = (1018.0, 1060.0, 124.0, 4.5)

  hud.reveal_dock(5.0)
  assert hud.hit_test_dock(760.0, 1005.0) == thud.TESLA_MODE_3D
  assert hud.hit_test_dock(940.0, 1005.0) == thud.TESLA_MODE_AR
  assert hud.hit_test_dock(1120.0, 1005.0) == thud.TESLA_MODE_QUIET
  assert hud.hit_test_dock(200.0, 200.0) is None

  # When dock auto-hides in 3D mode, bottom swipe handle still registers tap (-1)
  hud._dock_visible_until = 0.0
  assert hud.hit_test_dock(760.0, 1005.0) is None
  assert hud.hit_test_dock(1080.0, 1055.0) == -1


class _FakeSubMaster(dict):
  def __init__(self, data, valid=None):
    super().__init__(data)
    self.valid = valid or {k: True for k in data}
    self.alive = {k: True for k in data}


def test_road_spline_handles_missing_and_curved_model(tesla_env):
  t3d, _, _ = tesla_env
  spline = t3d._RoadSpline()
  spline.update_from_model(None)
  assert spline.x_at_z(-30.0) == pytest.approx(0.0)

  xs = np.linspace(0.0, 100.0, 25, dtype=np.float32)
  ys = 0.001 * xs * xs
  sm = _FakeSubMaster({
    "modelV2": SimpleNamespace(
      position=SimpleNamespace(x=xs, y=ys),
      laneLines=[
        SimpleNamespace(y=[-5.4]),
        SimpleNamespace(y=[-1.8]),
        SimpleNamespace(y=[1.8]),
        SimpleNamespace(y=[5.4]),
      ],
      laneLineProbs=[0.4, 0.9, 0.9, 0.4],
      roadEdges=[SimpleNamespace(y=[-5.5]), SimpleNamespace(y=[5.5])],
    ),
  })
  spline.update_from_model(sm)
  assert spline.x_at_z(-40.0) > 0.5
  assert spline.yaw_at_z(-40.0) < 0.0


def test_collect_traffic_deduplicates_lead_one_from_tracks(tesla_env):
  t3d, _, _ = tesla_env
  lead_one = SimpleNamespace(status=True, dRel=35.0, yRel=0.1, vLeadK=22.0, vRel=-1.0, radarTrackId=7)
  dup_track = SimpleNamespace(dRel=35.2, yRel=0.0, vLeadK=22.0, radarTrackId=7)
  left_car = SimpleNamespace(dRel=18.0, yRel=3.7, vLeadK=25.0, radarTrackId=8)
  right_car = SimpleNamespace(dRel=48.0, yRel=-3.8, vLeadK=26.0, radarTrackId=9)
  rs = SimpleNamespace(
    leadOne=lead_one,
    leadTwo=SimpleNamespace(status=False, dRel=0.0, yRel=0.0, radarTrackId=-1),
    leadsLeft=[left_car],
    leadsRight=[right_car],
    leadsCenter=[dup_track],
  )
  sm = _FakeSubMaster({"radarState": rs})
  lead, sides = t3d.Tesla3DRenderer._collect_traffic(sm)
  assert lead is not None
  assert lead[0] == pytest.approx(35.0)
  assert len(sides) == 2
  assert {round(s[0]) for s in sides} == {18, 48}


def test_hud_snapshot_safe_when_submaster_empty(tesla_env):
  _, thud, fake_ui_state = tesla_env
  fake_ui_state.sm = None
  hud = thud.TeslaHudRenderer()
  snap = hud._read_snapshot(0.05)
  assert snap["speed"] == 0
  assert snap["gear"] == "D"
  assert snap["low_tpms"] == []
  assert snap["active_carrot"] == 0


def test_carrot_link_badge_maps_active_carrot_to_apm_and_apn(tesla_env, monkeypatch):
  _, thud, fake_ui_state = tesla_env
  hud = thud.TeslaHudRenderer()

  # Verify snapshot extraction for 0 / 1 (APM) / 2 (APN)
  for ac in (0, 1, 2, 3):
    fake_ui_state.sm = _FakeSubMaster({"carrotMan": SimpleNamespace(activeCarrot=ac)})
    assert hud._read_snapshot(0.05)["active_carrot"] == ac

  # Verify rendered badge text and accent colors
  drawn_labels = []
  drawn_dots = []
  monkeypatch.setattr(thud.hs, "chip", lambda *a, **k: None, raising=False)
  monkeypatch.setattr(thud.hs, "glow", lambda *a, **k: None, raising=False)
  monkeypatch.setattr(thud.hs, "dot", lambda x, y, r, col: drawn_dots.append(col), raising=False)
  monkeypatch.setattr(hud._type, "draw_mid", lambda txt, *a, **k: drawn_labels.append(txt) or 80.0)

  assert hud._draw_carrot_link_pill(1800.0, 66.0, 0) == 0.0
  assert drawn_labels == []

  w_apm = hud._draw_carrot_link_pill(1800.0, 66.0, 1)
  assert w_apm > 0.0
  assert drawn_labels[-1] == "APM"
  assert drawn_dots[-1] == thud.TESLA_BLUE

  w_apn = hud._draw_carrot_link_pill(1800.0, 66.0, 2)
  assert w_apn > 0.0
  assert drawn_labels[-1] == "APN"
  assert drawn_dots[-1] == thud.TESLA_EMERALD


def test_day_night_controller_snaps_on_first_sample_and_debounces_tunnels(tesla_env):
  _, thud, _ = tesla_env
  ctrl = thud.TeslaDayNightController(default_day=False)
  assert ctrl.is_day is False

  # Sensor unavailable (-1) preserves initial clock-seeded state
  assert ctrl.update(-1.0, now=10.0) is False

  # First valid daylight sample snaps immediately to Day
  assert ctrl.update(72.0, now=11.0) is True

  # Brief highway overpass shadow (1.0 s dark dip) must NOT strobe to Night
  for i in range(1, 11):
    assert ctrl.update(10.0, now=11.0 + i * 0.1) is True

  # Returning to sunlight clears pending transition
  assert ctrl.update(75.0, now=12.2) is True

  # Entering a real tunnel / night (sustained dark > EMA + 1.5s hold) switches cleanly to Night
  t = 13.0
  for _ in range(60):
    t += 0.1
    ctrl.update(8.0, now=t)
  assert ctrl.is_day is False

  # Exiting tunnel into sustained sunlight switches cleanly back to Day
  for _ in range(60):
    t += 0.1
    ctrl.update(78.0, now=t)
  assert ctrl.is_day is True

  # Light HUD flag activates ONLY in Mode 0 (3D FSD) when day_mode is True
  hud = thud.TeslaHudRenderer()
  hud.set_day_mode(True)
  assert hud._is_light_ui is True
  hud.set_mode(thud.TESLA_MODE_AR)
  assert hud._is_light_ui is False
  hud.set_mode(thud.TESLA_MODE_QUIET)
  assert hud._is_light_ui is False
