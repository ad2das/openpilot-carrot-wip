import math
import time
import pyray as rl
from dataclasses import dataclass
from openpilot.common.constants import CV
from openpilot.selfdrive.carrot.deceleration_source import deceleration_source_presentation
from openpilot.selfdrive.ui.onroad import hud_style as hs
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.hardware.usbgpu import usbgpu_badge_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'
CRUISE_SPEED_ANIMATION_START = 120
CRUISE_SPEED_ANIMATION_MAX = 100
CRUISE_SPEED_ANIMATION_STEP = 12
CRUISE_SPEED_ANIMATION_START_SIZE = 240
HUD_PARAM_REFRESH_INTERVAL = 1.0
WEEKDAYS_KO = ("일", "월", "화", "수", "목", "금", "토")

# Layout, in content-rect pixels. Four instrument cards hold the corners (guidance top-left, tyres
# top-right, drive bottom-left, trip bottom-right); the clock floats top-centre over the image.
EDGE_X = 36
M_X = 36
M_TOP = 32
M_BOTTOM = 30
CARD_R = 38
TOP_SHADE_H = 250
BOTTOM_SHADE_H = 390
DRIVE_W = 620
DRIVE_H = 328
SET_SIZE = 60
SET_BASE = 74
CHIP_SIZE = 36
CHIP_H = 56
SPEED_SIZE = 196
SPEED_SIZE_3 = 152
SPEED_BASE = 246
UNIT_SIZE = 42
SIGN_R = 86
SIGN_RIGHT = 118
SIGN_RIGHT_LIGHT = 198
SIGN_Y = 156
LIGHT_RIGHT = 62
STATUS_MID = 286
STATUS_SIZE = 40
NAV_W_MIN = 520
NAV_W_MAX = 900
NAV_H = 276
NAV_H_SHORT = 224
NAV_TILE = 176
NAV_DISTANCE_SIZE = 132
NAV_UNIT_SIZE = 58
NAV_TEXT_SIZE = 54
NAV_PROGRESS_SPAN = 2000.0
CLOCK_SIZE = 76
DATE_SIZE = 40
BADGE_SIZE = 28
BADGE_H = 52
TPMS_W = 320
TPMS_H = 236
TPMS_SIZE = 50
TRIP_STATS_H = 150     # three-column stats row
TRIP_HEAD_H = 66       # road-name header above it
TRIP_COL_MIN = 196
TRIP_COL_PAD = 34
TRIP_SIZE = 64
TRIP_UNIT_SIZE = 38
TRIP_LABEL_SIZE = 30
TRIP_ROAD_SIZE = 38
STACK_GAP = 20
DIM_GREY = hs.rgba(142, 147, 155)
NAV_BLUE = hs.NAV
# SetSpeedOverrideState.speed_color_mode -> chip colour (eco, deceleration, vehicle navigation, external navigation).
OVERRIDE_COLORS = {1: hs.LIVE_GREEN, 2: hs.PROMPT_AMBER, 3: hs.rgba(175, 82, 222), 4: hs.CARROT}
# Speed-override reasons as short Korean words; unknown sources fall back to the raw label.
OVERRIDE_LABELS = {
  "eco": "에코", "apply": "감속", "cam": "단속", "section": "구간", "bump": "방지턱", "police": "경찰",
  "road": "도로", "turn": "회전", "route": "경로", "school": "스쿨존", "gas": "주유", "vturn": "커브",
  "model": "모델", "waze": "WAZE",
}


@dataclass(frozen=True)
class UIConfig:
  header_height: int = 300
  border_size: int = 30
  button_size: int = 192
  set_speed_width_metric: int = 200
  set_speed_width_imperial: int = 172
  set_speed_height: int = 204
  wheel_icon_size: int = 144


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 66
  max_speed: int = 40
  set_speed: int = 90


@dataclass(frozen=True)
class Colors:
  WHITE = rl.WHITE
  DISENGAGED = rl.Color(145, 155, 149, 255)
  OVERRIDE = rl.Color(145, 155, 149, 255)  # Added
  ENGAGED = rl.Color(128, 216, 166, 255)
  DISENGAGED_BG = rl.Color(0, 0, 0, 153)
  OVERRIDE_BG = rl.Color(145, 155, 149, 204)
  ENGAGED_BG = rl.Color(128, 216, 166, 204)
  GREY = rl.Color(166, 166, 166, 255)
  DARK_GREY = rl.Color(114, 114, 114, 255)
  BLACK_TRANSLUCENT = rl.Color(0, 0, 0, 166)
  WHITE_TRANSLUCENT = rl.Color(255, 255, 255, 200)
  BORDER_TRANSLUCENT = rl.Color(255, 255, 255, 75)
  HEADER_GRADIENT_START = rl.Color(0, 0, 0, 114)
  HEADER_GRADIENT_END = rl.BLANK
  CARROT_GREEN = rl.Color(0, 203, 0, 255)
  BLACK_90 = rl.Color(0, 0, 0, 90)
  RED_180 = rl.Color(255, 0, 0, 180)
  GREEN_190 = rl.Color(0, 255, 0, 190)
  GREEN_200 = rl.Color(0, 255, 0, 200)
  GREEN_210 = rl.Color(0, 255, 0, 210)
  BLUE_210 = rl.Color(0, 120, 255, 210)
  VEHICLE_NAVI_LAVENDER = rl.Color(199, 125, 255, 230)
  RED_200 = rl.Color(255, 0, 0, 200)
  RED_210 = rl.Color(255, 0, 0, 210)
  YELLOW_210 = rl.Color(255, 255, 0, 210)
  WHITE_210 = rl.Color(255, 255, 255, 210)
  WHITE_220 = rl.Color(255, 255, 255, 220)
  TPMS_LOW = rl.Color(255, 90, 90, 220)
  ORANGE_200 = rl.Color(255, 165, 0, 200)
  ORANGE_230 = rl.Color(255, 165, 0, 230)
  EXTERNAL_NAVI_ORANGE = rl.Color(244, 172, 54, 230)
  RED_SOLID = rl.Color(255, 0, 0, 255)


UI_CONFIG = UIConfig()
FONT_SIZES = FontSizes()
COLORS = Colors()

@dataclass(frozen=True)
class SetSpeedOverrideState:
  active: bool
  speed_kph: float
  label: str
  speed_color_mode: int # 0: white, 1: eco green, 2: orange, 3: vehicle-navigation blue, 4: external-navigation green
  force_persist: bool


class SetSpeedOverride:

  def compute(self, sm, set_speed_kph: float) -> SetSpeedOverrideState:
    # 1) eco (highest)
    cruise_target = None
    try:
      cruise_target = float(sm['longitudinalPlan'].cruiseTarget)
    except Exception:
      cruise_target = None

    if cruise_target is not None and cruise_target > (set_speed_kph + 0.5):
      return SetSpeedOverrideState(
        active=True,
        speed_kph=cruise_target,
        label="eco",
        speed_color_mode=1,
        force_persist=True,   # eco 조건 유지되는 동안 계속 표시
      )

    # 2) apply_speed (desiredSpeed/source)
    desired_speed = None
    desired_source = ""
    try:
      desired_speed = float(sm['carrotMan'].desiredSpeed)
      desired_source = str(sm['carrotMan'].desiredSource or "")
    except Exception:
      desired_speed = None
      desired_source = ""

    if desired_speed is not None and 0 < desired_speed < 200 and desired_speed < set_speed_kph:
      label, speed_color_mode = deceleration_source_presentation(desired_source)
      return SetSpeedOverrideState(
        active=True,
        speed_kph=desired_speed,
        label=label,
        speed_color_mode=speed_color_mode,
        force_persist=True,   # 조건 유지되는 동안 계속 표시
      )

    # 3) default
    return SetSpeedOverrideState(
      active=False,
      speed_kph=set_speed_kph,
      label=tr("MAX"),
      speed_color_mode=0,
      force_persist=False,
    )


class HudRenderer(Widget):
  def __init__(self):
    super().__init__()
    self.is_cruise_set = False
    self.is_cruise_available = True
    self.set_speed = SET_SPEED_NA
    self.speed = 0.0
    self.v_ego_cluster_seen = False

    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_medium = gui_app.font(FontWeight.MEDIUM)
    self._font_display = gui_app.font(FontWeight.DISPLAY)

    self._type = hs.Type()

    self._set_speed_override = SetSpeedOverride()
    self._debug_speed_panel = False
    self._engaged = False
    self._overspeed = False
    self._progress_key: tuple | None = None
    self._progress_start = NAV_PROGRESS_SPAN
    self._progress_last = 0
    self._set_anchor = (0.0, 0.0)

    # c3-wip cruise-speed animation: center popup -> left Carrot HUD target.
    self._cruise_speed_text_last = ""
    self._cruise_speed_animation_text = ""
    self._cruise_speed_animation_time = -1

    self._blink_timer = 0
    self._disp_timer = 0

    self._cpu_temp = 0.0
    self._cpu_usage = 0.0
    self._memory_usage = 0
    self._free_space = 0.0
    self._voltage = 0.0
    self._device_info_loaded = False
    self._device_info_recv_frames = (-1, -1)
    self._cpu_temp_text = "0°C"
    self._memory_usage_text = "0%"
    self._disk_usage_text = "100%"
    self._voltage_text = "0.0V"
    self._plot_renderer = None
    self._round_box_rect = rl.Rectangle(0.0, 0.0, 0.0, 0.0)

    self._hud_params_next_refresh_time = 0.0
    self._show_device_state = 0
    self._show_date_time = 0
    self._show_tpms = 1
    self._show_plot_mode = 0
    self._longitudinal_personality = 7

    self._date_time_minute_key: tuple[int, int, int, int, int] | None = None
    self._date_time_text = ""
    self._date_text = ""

  def _refresh_hud_params(self, now: float) -> None:
    if now < self._hud_params_next_refresh_time:
      return

    try:
      show_device_state = ui_state.params.get_int("ShowDeviceState")
      show_date_time = ui_state.params.get_int("ShowDateTime")
      show_tpms = ui_state.params.get_int("ShowTpms")
      show_plot_mode = ui_state.params.get_int("ShowPlotMode")
    except Exception:
      # Keep the last complete snapshot and retry on the next frame.
      return

    personality_read_failed = False
    try:
      longitudinal_personality = ui_state.params.get_int("LongitudinalPersonality")
    except Exception:
      # Preserve the legacy fallback (gap 8) and retry on the next frame.
      longitudinal_personality = 7
      personality_read_failed = True

    self._show_device_state = show_device_state
    self._show_date_time = show_date_time
    self._show_tpms = show_tpms
    self._show_plot_mode = show_plot_mode
    self._longitudinal_personality = longitudinal_personality
    self._hud_params_next_refresh_time = now if personality_read_failed else now + HUD_PARAM_REFRESH_INTERVAL

  def _update_state(self) -> None:
    """Update HUD state based on car state and controls state."""
    sm = ui_state.sm
    device_info_recv_frames = (sm.recv_frame["deviceState"], sm.recv_frame["peripheralState"])
    if not self._device_info_loaded or device_info_recv_frames != self._device_info_recv_frames:
      self._update_device_info()
      if self._device_info_loaded:
        self._device_info_recv_frames = device_info_recv_frames

    if sm.recv_frame["carState"] < ui_state.started_frame:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    controls_state = sm['controlsState']
    car_state = sm['carState']

    v_cruise_cluster = car_state.vCruiseCluster
    self.set_speed = (
      controls_state.deprecated.vCruise if v_cruise_cluster == 0.0 else v_cruise_cluster
    )
    self.is_cruise_set = 0 < self.set_speed < SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    #if self.is_cruise_set and not ui_state.is_metric:
    #  self.set_speed *= KM_TO_MILE

    self._engaged = sm['selfdriveState'].enabled

    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo
    speed_conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

  def _render(self, rect: rl.Rectangle) -> None:
    """Render HUD elements to the screen."""
    self._refresh_hud_params(time.monotonic())
    hs.panels.clear()
    hs.zones.clear()
    hs.top_boxes.clear()
    self._blink_timer = (self._blink_timer + 1) % 16
    self._disp_timer = (self._disp_timer + 1) % 64

    # Soft shades under the top and bottom card rows; the cards themselves carry the contrast.
    top, bottom = int(rect.y), int(rect.y + rect.height)
    rl.draw_rectangle_gradient_v(int(rect.x), top, int(rect.width), TOP_SHADE_H, rl.Color(0, 0, 0, 140), rl.BLANK)
    rl.draw_rectangle_gradient_v(int(rect.x), bottom - BOTTOM_SHADE_H, int(rect.width), BOTTOM_SHADE_H,
                                 rl.BLANK, rl.Color(0, 0, 0, 158))

    info = self._get_turn_info_hud_data()
    self._draw_guidance_card(rect, info)
    self._draw_tpms(rect, top=True)
    self._draw_status_capsule(rect)
    self._draw_trip(rect, info)
    self._draw_tpms(rect, top=False)
    if self.is_cruise_available:
      self._draw_drive_card(rect)

    # The C3X HUD has no experimental-mode button; the toggle stays in Settings.

    if self._plot_renderer is None:
      self._plot_renderer = PlotRenderer()
    self._plot_renderer.draw(rect, self._font_display, self._show_plot_mode)

    if not self._banner_alert_active():
      self._draw_clock(rect)
    self._draw_egpu_badge(rect)
    self._draw_cruise_speed_animation(rect)

  def user_interacting(self) -> bool:
    return False

  def _draw_egpu_badge(self, rect: rl.Rectangle) -> None:
    # Keep runtime state visible while the shared USB hub re-enumerates; a
    # transient missing sysfs sample must not hide loading or failure details.
    if not (getattr(ui_state, 'jetlink_badge', None) or getattr(ui_state, 'usbgpu_delivery_badge', None) or
            ui_state.usbgpu_present or ui_state.usbgpu_active or
            ui_state.usbgpu_loading or ui_state.usbgpu_startup_failed):
      return

    state = usbgpu_badge_state(ui_state.usbgpu_compiled, ui_state.usbgpu_loading,
                               ui_state.usbgpu_active, ui_state.usbgpu_startup_failed,
                               ui_state.usbgpu_compile_pending)
    text = "eGPU REBOOT" if state == "compile_pending" else "eGPU ERROR" if state == "error" else "eGPU"
    if (getattr(ui_state, "usbgpu_delivery_badge", None) and
        not (ui_state.usbgpu_active or ui_state.usbgpu_loading or ui_state.usbgpu_startup_failed)):
      text, state = ui_state.usbgpu_delivery_badge
    if getattr(ui_state, 'jetlink_badge', None) and not ui_state.usbgpu_active:
      text, state = ui_state.jetlink_badge
    color = {
      "active": hs.GREEN,
      "loading": hs.AMBER,
      "error": hs.RED,
      "compile_pending": hs.CARROT,
      "not_compiled": hs.CARROT,
      "ready": hs.TEXT_2,
    }[state]
    # A capsule under the tyre card (or in its place), with a status dot. hs.Type swaps in the
    # Hangul atlas for non-Latin badge text (e.g. Jetlink states), so mixed-script labels render.
    t = self._type
    w = t.width(text, BADGE_SIZE, hs.BOLD, 1.0) + 74
    right = rect.x + rect.width - M_X
    tpms = hs.zones.get("tpms")
    y = tpms[1] + tpms[3] + 16 if tpms is not None else rect.y + M_TOP
    hs.chip(right - w, y, w, 56)
    hs.dot(right - w + 30, y + 28, 8, color)
    t.draw_mid(text, right - w + 50, y + 28, BADGE_SIZE, color, hs.BOLD, spacing=1.0)

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw the MAX speed indicator box."""
    set_speed_width = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
    x = rect.x + 60 + (UI_CONFIG.set_speed_width_imperial - set_speed_width) // 2
    y = rect.y + 45

    set_speed_rect = rl.Rectangle(x, y, set_speed_width, UI_CONFIG.set_speed_height)
    rl.draw_rectangle_rounded(set_speed_rect, 0.35, 10, COLORS.BLACK_TRANSLUCENT)
    rl.draw_rectangle_rounded_lines_ex(set_speed_rect, 0.35, 10, 6, COLORS.BORDER_TRANSLUCENT)

    max_color = COLORS.GREY
    set_speed_color = COLORS.DARK_GREY
    if self.is_cruise_set:
      set_speed_color = COLORS.WHITE
      if ui_state.status == UIStatus.ENGAGED:
        max_color = COLORS.ENGAGED
      elif ui_state.status == UIStatus.DISENGAGED:
        max_color = COLORS.DISENGAGED
      elif ui_state.status == UIStatus.OVERRIDE:
        max_color = COLORS.OVERRIDE

    max_text = tr("MAX")
    max_text_width = measure_text_cached(self._font_semi_bold, max_text, FONT_SIZES.max_speed).x
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + (set_speed_width - max_text_width) / 2, y + 27),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

    set_speed_text = CRUISE_DISABLED_CHAR if not self.is_cruise_set else str(round(self.set_speed))
    speed_text_width = measure_text_cached(self._font_bold, set_speed_text, FONT_SIZES.set_speed).x
    rl.draw_text_ex(
      self._font_bold,
      set_speed_text,
      rl.Vector2(x + (set_speed_width - speed_text_width) / 2, y + 77),
      FONT_SIZES.set_speed,
      0,
      set_speed_color,
    )

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    """Draw the current vehicle speed and unit."""
    speed_text = str(round(self.speed))
    speed_text_size = measure_text_cached(self._font_bold, speed_text, FONT_SIZES.current_speed)
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.WHITE)

    unit_text = tr("km/h") if ui_state.is_metric else tr("mph")
    unit_text_size = measure_text_cached(self._font_medium, unit_text, FONT_SIZES.speed_unit)
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.WHITE_TRANSLUCENT)

  def _draw_round_box(self, x, y, w, h, fill_color,
                      line_color=None,
                      roundness=0.25,
                      segments=8,
                      line_thickness=2):
    rect = self._round_box_rect
    rect.x = float(x)
    rect.y = float(y)
    rect.width = float(w)
    rect.height = float(h)
    rl.draw_rectangle_rounded(rect, roundness, segments, fill_color)
    if line_color is not None and line_thickness > 0:
      rl.draw_rectangle_rounded_lines_ex(rect, roundness, segments, float(line_thickness), line_color)


  def _draw_texture_rect(self, tex, x, y, w, h, tint=rl.WHITE):
    if tex is None:
      return
    rl.draw_texture_pro(
      tex,
      rl.Rectangle(0, 0, float(tex.width), float(tex.height)),
      rl.Rectangle(float(x), float(y), float(w), float(h)),
      rl.Vector2(0, 0),
      0.0,
      tint,
    )

  def _get_gear_text(self) -> str:
    sm = ui_state.sm

    try:
      car_state = sm["carState"]
      gear = car_state.gearShifter
    except Exception:
      return "R"

    # cereal enum → 문자열 변환
    try:
      gear_name = str(gear).split('.')[-1]
    except Exception:
      gear_name = str(gear)

    # DRIVE 처리
    if "DRIVE" in gear_name.upper():
      try:
        step = int(car_state.gearStep)
        if step > 0:
          return str(step)
        else:
          return "D"
      except Exception:
        return "D"

    if "PARK" in gear_name.upper():
      return "P"

    if "REVERSE" in gear_name.upper():
      return "R"

    if "NEUTRAL" in gear_name.upper():
      return "N"

    if "SPORT" in gear_name.upper():
      return "S"

    if "LOW" in gear_name.upper():
      return "L"

    if "BRAKE" in gear_name.upper():
      return "B"

    if "ECO" in gear_name.upper():
      return "E"

    if "UNKNOWN" in gear_name.upper():
      return "U"

    return "M"

  def _get_cruise_gap(self) -> int:
    return int(self._longitudinal_personality) + 1

  def _get_driving_mode_text_and_color(self) -> tuple[str, rl.Color]:
    try:
      mode_val = int(ui_state.sm["longitudinalPlan"].myDrivingMode)
    except Exception:
      return "", COLORS.WHITE_TRANSLUCENT

    if mode_val == 1:   # eco
      return tr("eco"), hs.GREEN
    if mode_val == 2:   # safe
      return tr("safe"), hs.AMBER
    if mode_val == 3:   # normal
      return tr("norm"), hs.TEXT
    if mode_val == 4:   # high
      return tr("high"), hs.RED

    return "", COLORS.WHITE_TRANSLUCENT

  def _update_device_info(self):
    sm = ui_state.sm
    loaded = True

    self._cpu_temp = 0.0
    self._cpu_usage = 0.0
    self._memory_usage = 0
    self._free_space = 0.0
    self._voltage = 0.0
    #self._plot_renderer = None

    try:
      device_state = sm["deviceState"]
      self._free_space = float(device_state.freeSpacePercent)
      self._memory_usage = int(device_state.memoryUsagePercent)

      try:
        cpu_temps = list(device_state.cpuTempC)
        if len(cpu_temps) > 0:
          self._cpu_temp = sum(cpu_temps) / len(cpu_temps)
      except Exception:
        loaded = False

      try:
        cpu_usages = [float(v) for v in device_state.cpuUsagePercent if float(v) > 0]
        if len(cpu_usages) > 0:
          self._cpu_usage = sum(cpu_usages) / len(cpu_usages)
      except Exception:
        loaded = False
    except Exception:
      loaded = False

    try:
      peripheral_state = sm["peripheralState"]
      self._voltage = float(peripheral_state.voltage) / 1000.0
    except Exception:
      loaded = False

    self._cpu_temp_text = f"{self._cpu_temp:.0f}°C"
    self._memory_usage_text = f"{self._memory_usage}%"
    self._disk_usage_text = f"{100 - self._free_space:.0f}%"
    self._voltage_text = f"{self._voltage:.1f}V"
    self._device_info_loaded = loaded

  def _get_active_carrot(self) -> int:
    try:
      return int(ui_state.sm["carrotMan"].activeCarrot)
    except Exception:
      return 0


  def _get_nav_path_vertex_count(self) -> int:
    try:
      return int(ui_state.sm["carrotMan"].navPathVertexCount)
    except Exception:
      return 0


  def _get_traffic_state(self) -> int:
    try:
      return int(ui_state.sm["carrotMan"].trafficState)
    except Exception:
      return 0


  def _get_traffic_state_carrot(self) -> int:
    try:
      return int(ui_state.sm["carrotMan"].trafficStateCarrot)
    except Exception:
      return 0


  def _get_speed_limit_info(self) -> tuple[int, int, int]:
    """
    return:
      x_spd_limit, x_sign_type, road_limit_speed
    """
    try:
      cm = ui_state.sm["carrotMan"]
    except Exception:
      return 0, 0, 0

    try:
      x_spd_limit = int(cm.xSpdLimit)
    except Exception:
      x_spd_limit = 0

    try:
      x_sign_type = int(cm.xSignType)
    except Exception:
      x_sign_type = 0

    try:
      road_limit_speed = int(cm.nRoadLimitSpeed)
    except Exception:
      road_limit_speed = 0

    return x_spd_limit, x_sign_type, road_limit_speed


  def _gps_has_fix(self) -> bool:
    sm = ui_state.sm

    try:
      return bool(sm["gpsLocationExternal"].hasFix)
    except Exception:
      pass

    try:
      return bool(sm["gpsLocation"].hasFix)
    except Exception:
      pass

    return False

  # ---- speed limit / signals --------------------------------------------------------------------

  def _limit_state(self) -> tuple[int, bool, bool]:
    """(limit in display units, enforcement camera, camera alarm)."""
    x_spd_limit, x_sign_type, road_limit_speed = self._get_speed_limit_info()
    camera = x_spd_limit > 0 and x_sign_type != 22
    limit = x_spd_limit if camera else road_limit_speed
    limit_display = int(limit if ui_state.is_metric else limit * KM_TO_MILE + 0.5) if limit > 0 else 0
    alarm = x_spd_limit > 0 and x_sign_type not in (22, 4)
    return limit_display, camera, alarm

  def _traffic_light(self) -> str | None:
    traffic_state = self._get_traffic_state()
    traffic_state_carrot = self._get_traffic_state_carrot()
    if traffic_state == 1 or traffic_state_carrot == 1:
      return "red"
    if traffic_state == 2 or traffic_state_carrot == 2:
      return "green"
    return None

  def _cruise_text(self) -> str:
    if self._engaged and self.is_cruise_set:
      set_speed = float(self.set_speed)
      if not ui_state.is_metric:
        set_speed *= KM_TO_MILE
      return str(int(round(set_speed)))
    return "--"

  def _override_chip(self) -> tuple[str, str, rl.Color, bool] | None:
    """(reason, speed, fill, white ink) for the solid pill beside SET, or None."""
    ov = self._set_speed_override.compute(ui_state.sm, float(self.set_speed))
    if not ov.active:
      return None
    ov_speed = float(ov.speed_kph)
    if not ui_state.is_metric:
      ov_speed *= KM_TO_MILE
    label, value, mode = str(ov.label), str(int(round(ov_speed))), ov.speed_color_mode
    if self._debug_speed_panel:
      label, value = "vturn", "111"
    label = OVERRIDE_LABELS.get(label.lower(), label.upper())
    return label, value, OVERRIDE_COLORS.get(mode, hs.LIVE_GREEN), mode == 3

  # ---- drive card (bottom-left): SET row, speed + sign, gear/gap/mode ------------------------

  def _draw_drive_card(self, rect: rl.Rectangle) -> None:
    t = self._type
    x = rect.x + M_X
    y = rect.y + rect.height - M_BOTTOM - DRIVE_H
    right = x + DRIVE_W
    hs.glass_card(x, y, DRIVE_W, DRIVE_H, 40)
    hs.panels.append((x, y, DRIVE_W, DRIVE_H))
    hs.zones["drive"] = (x, y, DRIVE_W, DRIVE_H)

    cruise_text = self._cruise_text()
    self._update_cruise_speed_animation(cruise_text)
    engaged = cruise_text != "--"
    if engaged:
      hs.hgradient_line(x + 60, y, DRIVE_W - 120, 4, hs.LIVE_GREEN)

    # Right column: limit sign, pushed left when a traffic light takes the edge.
    limit, _camera, alarm = self._limit_state()
    light = self._traffic_light()
    sign_x = right - (SIGN_RIGHT_LIGHT if light else SIGN_RIGHT)
    column = sign_x - SIGN_R - 20 if limit > 0 else (right - LIGHT_RIGHT - 50 if light else right - 32)

    # SET row: the label carries the engagement colour; an active override reads as a solid pill.
    sx = x + 32
    sx += t.draw("SET", sx, y + SET_BASE, 30, hs.LIVE_GREEN if engaged else DIM_GREY, hs.BOLD, spacing=2) + 14
    number_w = t.draw(cruise_text, sx, y + SET_BASE, SET_SIZE, hs.TEXT if engaged else hs.with_alpha(hs.TEXT, 128), hs.SEMI)
    self._set_anchor = (sx + number_w / 2, y + SET_BASE)
    chip = self._override_chip()
    if chip is not None:
      label, value, fill, white = chip
      ink = hs.TEXT if white else hs.INK
      # The pill sits above the sign's crown, so only the card edge bounds it; the reason drops before the speed does.
      cx = sx + number_w + 22
      value_w = t.width(value, CHIP_SIZE, hs.BOLD)
      label_w = t.width(label, 32, hs.BOLD) + 14
      if cx + label_w + value_w + 52 > right - 32:
        label, label_w = "", 0.0
      cw = label_w + value_w + 52
      mid = y + SET_BASE - SET_SIZE * hs.INTER_CAP / 2
      hs.card(cx, mid - CHIP_H / 2, cw, CHIP_H, fill, CHIP_H / 2, None)
      if label:
        t.draw_mid(label, cx + 26, mid, 32, ink, hs.BOLD)
      t.draw_mid(value, cx + 26 + label_w, mid, CHIP_SIZE, ink, hs.BOLD)

    # Speed numerals, red over the limit, shrunk rather than run into the sign.
    speed_text = "123" if self._debug_speed_panel else str(int(round(self.speed)))
    size = SPEED_SIZE if len(speed_text) < 3 else SPEED_SIZE_3
    unit = tr("km/h") if ui_state.is_metric else tr("mph")
    room = column - (x + 26)
    need = t.width(speed_text, size, hs.SEMI, -6 * size / SPEED_SIZE) + 18 + t.width(unit, UNIT_SIZE, hs.MEDIUM)
    if need > room:
      size *= max(0.7, room / need)
    self._overspeed = limit > 0 and self.speed > limit + 2
    w = t.draw(speed_text, x + 26, y + SPEED_BASE, size, hs.WARN_RED if self._overspeed else hs.TEXT, hs.SEMI,
               spacing=-6 * size / SPEED_SIZE)
    t.draw(unit, x + 26 + w + 18, y + SPEED_BASE, UNIT_SIZE, hs.with_alpha(hs.TEXT, 178), hs.MEDIUM)

    if limit > 0:
      if alarm:
        pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 2.0 * math.pi)
        hs.glow(sign_x, y + SIGN_Y, 124, hs.rgba(255, 69, 58, int(90 + 90 * pulse)))
      hs.regulatory_sign(sign_x, y + SIGN_Y, str(limit), t)
    if light:
      lx = right - LIGHT_RIGHT
      hs.card(lx - 30, y + SIGN_Y - 78, 60, 156, hs.rgba(0, 0, 0, 140), 30, hs.rgba(255, 255, 255, 38), 1.5)
      for cy, color, on in ((y + SIGN_Y - 34, hs.WARN_RED, light == "red"), (y + SIGN_Y + 34, hs.LIVE_GREEN, light == "green")):
        if on:
          hs.glow(lx, cy, 40, hs.with_alpha(color, 120))
        hs.dot(lx, cy, 21, color if on else hs.with_alpha(color, 46))

    # Status row: gear tile and following-gap segments on the left, driving mode on the right.
    mid = y + STATUS_MID
    hs.card(x + 30, mid - 24, 56, 48, hs.with_alpha(hs.TEXT, 235), 12, None)
    t.draw_mid(self._get_gear_text(), x + 58, mid, 36, hs.INK, hs.BOLD, align=0.5)
    gap = self._get_cruise_gap()
    gx = x + 108
    for i in range(4):
      hs.card(gx + i * 30, mid - 8, 24, 16, hs.TEXT if i < gap else hs.with_alpha(hs.TEXT, 56), 5, None)
    mode_text, mode_color = self._get_driving_mode_text_and_color()
    if self._debug_speed_panel:
      mode_text, mode_color = "safe", hs.AMBER
    if mode_text:
      t.draw_mid(mode_text, right - 32, mid, STATUS_SIZE, mode_color, hs.SEMI, align=1.0)

    self._draw_device_state(x, y)

  def _update_cruise_speed_animation(self, cruise_text: str) -> None:
    if self._cruise_speed_text_last != cruise_text:
      self._cruise_speed_text_last = cruise_text
      if cruise_text == "--":
        return
      self._cruise_speed_animation_text = cruise_text
      self._cruise_speed_animation_time = CRUISE_SPEED_ANIMATION_START

  def _draw_cruise_speed_animation(self, rect: rl.Rectangle) -> None:
    if self._cruise_speed_animation_time <= 0 or not self._cruise_speed_animation_text:
      return

    # c3-wip's integer state machine: a large centre popup that settles onto the SET number.
    self._cruise_speed_animation_time -= CRUISE_SPEED_ANIMATION_STEP
    animation_time = self._cruise_speed_animation_time
    interpolation_time = min(animation_time, CRUISE_SPEED_ANIMATION_MAX)
    t = interpolation_time / CRUISE_SPEED_ANIMATION_MAX

    start_x = rect.x + rect.width / 2.0
    start_y = rect.y + rect.height - 400.0
    target_x, target_y = self._set_anchor
    size = SET_SIZE + (CRUISE_SPEED_ANIMATION_START_SIZE - SET_SIZE) * t

    self._type.draw(self._cruise_speed_animation_text, target_x + (start_x - target_x) * t,
                    target_y + (start_y - target_y) * t, size, hs.LIVE_GREEN, hs.SEMI, align=0.5, shadow=True)

  def _draw_device_state(self, card_x: float, card_y: float) -> None:
    if self._show_device_state <= 0:
      return

    blink = self._blink_timer <= 8
    if self._disp_timer < 32:
      last = ("DISK", self._disk_usage_text, False)
    else:
      last = ("VOLT", self._voltage_text, False)
    items = (("CPU", self._cpu_temp_text, self._cpu_temp > 80), ("MEM", self._memory_usage_text, self._memory_usage > 85), last)

    # One capsule above the drive card.
    t = self._type
    width = 0.0
    for label, value, _hot in items:
      width += t.width(label, 28, hs.BOLD, 1.5) + 10 + t.width(value, 38, hs.SEMI) + 30
    h = 60
    top = card_y - 18 - h
    hs.chip(card_x, top, width + 30, h)
    x = card_x + 30
    mid = top + h / 2
    for label, value, hot in items:
      x += t.draw_mid(label, x, mid, 28, hs.with_alpha(hs.TEXT, 150), hs.BOLD, spacing=1.5) + 10
      x += t.draw_mid(value, x, mid, 38, hs.WARN_RED if hot and blink else hs.TEXT, hs.SEMI) + 30

  # ---- clock and link badges (top-centre) -------------------------------------------------------

  def _draw_clock(self, rect: rl.Rectangle) -> None:
    # Screen centre, nudged only as far as a wide guidance card requires.
    guide, tpms = hs.zones.get("guide"), hs.zones.get("tpms")
    cx = rect.x + rect.width / 2
    if guide is not None:
      cx = max(cx, guide[0] + guide[2] + 260)
    if tpms is not None:
      cx = min(cx, tpms[0] - 260)
    y = rect.y + 30
    show_datetime = self._show_date_time
    if show_datetime > 0:
      self._refresh_date_time_text(time.localtime())
      if show_datetime in (1, 2):
        self._type.draw(self._date_time_text, cx, y + 58, CLOCK_SIZE, hs.TEXT, hs.SEMI, align=0.5, shadow=True)
        y += 58
      if show_datetime in (1, 3):
        y += 52 if show_datetime == 1 else 40
        self._type.draw(self._date_text, cx, y, DATE_SIZE, hs.with_alpha(hs.TEXT, 218), hs.SEMI, align=0.5, shadow=True)
    if y > rect.y + 40:
      hs.top_boxes.append((cx - 240, rect.y, 480, y - rect.y + 16))

  def _draw_status_capsule(self, rect: rl.Rectangle) -> None:
    """Link status in the top-right column, under the TPMS card when it is up there."""
    tpms = hs.zones.get("tpms")
    top = tpms[1] + tpms[3] + STACK_GAP if tpms is not None else rect.y + M_TOP
    box = self._draw_system_badges(rect.x + rect.width - M_X, top)
    if box is not None:
      hs.top_boxes.append(box)

  def _draw_system_badges(self, right: float, top: float) -> tuple[float, float, float, float] | None:
    """One glass capsule, right-aligned: a lit dot per active link (APM keeps its blue); absent when all are off."""
    items = []
    active_carrot = self._get_active_carrot()
    if active_carrot >= 1:
      items.append(("APN", hs.LIVE_GREEN) if active_carrot >= 2 else ("APM", NAV_BLUE))
    if self._gps_has_fix():
      items.append(("GPS", hs.LIVE_GREEN))
    if self._get_nav_path_vertex_count() > 1:
      items.append(("ROUTE", hs.LIVE_GREEN))
    if not items:
      return None

    t = self._type
    dot, pad, sep = 8.5, 24.0, 22.0
    widths = [2 * dot + 11 + t.width(label, BADGE_SIZE, hs.BOLD, 1.5) for label, _ in items]
    w = sum(widths) + 2 * pad + 2 * sep * (len(items) - 1)
    x = right - w
    mid = top + BADGE_H / 2
    hs.chip(x, top, w, BADGE_H)
    x += pad
    for index, ((label, color), item_w) in enumerate(zip(items, widths, strict=True)):
      if index:
        hs.card(x + sep - 1, mid - 11, 2, 22, hs.rgba(255, 255, 255, 40), 1, None)
        x += 2 * sep
      hs.glow(x + dot, mid, dot * 2.8, hs.with_alpha(color, 150))
      hs.dot(x + dot, mid, dot, color)
      t.draw_mid(label, x + 2 * dot + 11, mid, BADGE_SIZE, hs.TEXT_2, hs.BOLD, spacing=1.5)
      x += item_w
    return right - w, top, w, BADGE_H

  def _refresh_date_time_text(self, now: time.struct_time) -> None:
    minute_key = (now.tm_year, now.tm_yday, now.tm_hour, now.tm_min, now.tm_isdst)
    if minute_key == self._date_time_minute_key:
      return

    weekday = WEEKDAYS_KO[(now.tm_wday + 1) % 7]
    self._date_time_text = time.strftime("%H:%M", now)
    self._date_text = f"{now.tm_mon}월 {now.tm_mday}일 {weekday}요일"
    self._date_time_minute_key = minute_key

  # ---- tyre pressure ------------------------------------------------------------------------

  def _get_tpms_color(self, tpms: float) -> rl.Color:
    if tpms < 5 or tpms > 60:
      return COLORS.WHITE_220
    if tpms < 31:
      return COLORS.TPMS_LOW
    return COLORS.WHITE_220

  def _get_tpms_text(self, tpms: float) -> str:
    if tpms < 5 or tpms > 60:
      return '  -'
    return f'{round(tpms):.0f}'

  def _draw_tpms(self, rect: rl.Rectangle, top: bool) -> None:
    if self._show_tpms not in ((1, 3) if top else (2, 3)):
      return

    try:
      tpms = ui_state.sm['carState'].tpms
      values = (float(tpms.fl), float(tpms.fr), float(tpms.rl), float(tpms.rr))
    except Exception:
      return

    x = rect.x + rect.width - M_X - TPMS_W
    if top:
      y = rect.y + M_TOP
      hs.zones["tpms"] = (x, y, TPMS_W, TPMS_H)
      hs.top_boxes.append((x, y, TPMS_W, TPMS_H))
    else:
      # Bottom-right corner, stacked above the trip capsule when it is shown.
      trip = hs.zones.get("trip")
      bottom = trip[1] if trip is not None else rect.y + rect.height - M_BOTTOM + STACK_GAP
      y = bottom - STACK_GAP - TPMS_H
    hs.glass_card(x, y, TPMS_W, TPMS_H, CARD_R)
    if not top:
      hs.panels.append((x, y, TPMS_W, TPMS_H))
    lows = tuple(self._get_tpms_color(value) == COLORS.TPMS_LOW for value in values)
    self._draw_tpms_car(x + 160, y + 118, lows)

    t = self._type
    for value, low, vx, vy, align in zip(values, lows, (x + 102, x + 218, x + 102, x + 218), (y + 86, y + 86, y + 186, y + 186),
                                         (1.0, 0.0, 1.0, 0.0), strict=True):
      t.draw(self._get_tpms_text(value).strip(), vx, vy, TPMS_SIZE, hs.WARN_RED if low else hs.TEXT, hs.SEMI, align=align)

  @staticmethod
  def _draw_tpms_car(cx: float, cy: float, low: tuple[bool, bool, bool, bool]) -> None:
    """Top view: a quiet body outline and four tyres, red where the pressure is low."""
    hs.card(cx - 30, cy - 78, 60, 156, hs.rgba(255, 255, 255, 18), 26, hs.rgba(255, 255, 255, 150), 2.5)
    hs.card(cx - 21, cy - 34, 42, 54, hs.rgba(255, 255, 255, 26), 12, None)
    for (tx, ty), is_low in zip(((cx - 39, cy - 58), (cx + 31, cy - 58), (cx - 39, cy + 30), (cx + 31, cy + 30)), low, strict=True):
      hs.card(tx, ty, 8, 28, hs.WARN_RED if is_low else hs.rgba(255, 255, 255, 200), 3, None)

  # ---- guidance (top-left): turn card or enforcement camera card ------------------------------

  def _get_turn_info_hud_data(self) -> dict:
    try:
      cm = ui_state.sm["carrotMan"]
    except Exception:
      cm = None

    def field(name, cast, default):
      try:
        return cast(getattr(cm, name))
      except Exception:
        return default

    def text(value) -> str:
      return str(value or "")

    return {
      "active_carrot": field("activeCarrot", int, 0),
      "x_turn_info": field("xTurnInfo", int, 0),
      "x_dist_to_turn": field("xDistToTurn", int, 0),
      "x_spd_dist": field("xSpdDist", int, 0),
      "n_go_pos_dist": field("nGoPosDist", int, 0),
      "n_go_pos_time": field("nGoPosTime", int, 0),
      "atc_type": field("atcType", text, ""),
      "sdi_descr": field("szSdiDescr", text, ""),
      "road_name": field("szPosRoadName", text, ""),
      "tbt_main_text": field("szTBTMainText", text, ""),
    }

  def _format_turn_distance_text(self, dist_m: int) -> str:
    if dist_m <= 0:
      return ""

    if ui_state.is_metric:
      if dist_m < 1000:
        return f"{dist_m} m"
      return f"{dist_m / 1000.0:.1f} km"
    else:
      if dist_m < 1609:
        return f"{int(dist_m * 3.28084)} ft"
      return f"{dist_m / 1609.344:.1f} mi"

  def _progress(self, key: tuple, dist: int) -> float:
    """Approach progress: fills over the last two kilometres, or from where a longer approach began."""
    if key != self._progress_key or dist > self._progress_last + 50:
      self._progress_key = key
      self._progress_start = max(float(dist), NAV_PROGRESS_SPAN)
    self._progress_last = dist
    return min(1.0, max(0.0, 1.0 - dist / self._progress_start))

  def _draw_guidance_card(self, rect: rl.Rectangle, info: dict) -> None:
    t = self._type
    x_spd_limit, x_sign_type, _ = self._get_speed_limit_info()
    route = info["n_go_pos_dist"] > 0 and info["n_go_pos_time"] > 0
    turn = route and info["x_turn_info"] > 0
    cam_dist = info["x_spd_dist"]
    camera = x_spd_limit > 0 and x_sign_type != 22 and cam_dist > 0
    if camera and turn and info["x_dist_to_turn"] < cam_dist:
      camera = False
    if not (camera or turn):
      self._progress_key = None
      return

    x, y = rect.x + M_X, rect.y + M_TOP
    if camera:
      limit = x_spd_limit if ui_state.is_metric else int(x_spd_limit * KM_TO_MILE + 0.5)
      dist = cam_dist
      text = info["sdi_descr"] or "과속 단속"
      extra = str(limit)
      body, bar = hs.CARD_WARN, hs.WARN_RED
      key: tuple = ("camera", x_sign_type)
    else:
      dist = info["x_dist_to_turn"]
      text, extra = info["tbt_main_text"], ""
      body, bar = hs.CARD_BODY, hs.LIVE_GREEN if info["atc_type"] else NAV_BLUE
      key = ("turn", info["x_turn_info"])

    # The card hugs its content: no blank tail after the distance or the instruction line.
    number, _, unit = self._format_turn_distance_text(dist).partition(" ")
    content = 0.0
    if number:
      content = t.width(number, NAV_DISTANCE_SIZE, hs.SEMI, -3) + 14 + t.width(unit, NAV_UNIT_SIZE, hs.MEDIUM)
    if text:
      content = max(content, t.width(text, NAV_TEXT_SIZE, hs.SEMI))
    w = min(NAV_W_MAX, max(NAV_W_MIN, 234 + content + 48))
    h = NAV_H if text else NAV_H_SHORT
    hs.glass_card(x, y, w, h, CARD_R, body)
    hs.top_boxes.append((x, y, w, h))
    hs.zones["guide"] = (x, y, w, h)

    gx, gy = x + 118, y + (122 if text else 98)
    if camera:
      hs.regulatory_sign(gx, gy, extra, t, 0.92)
    elif not hs.maneuver_arrow(info["x_turn_info"], gx, gy, 1.12 if text else 0.96, hs.LIVE_GREEN if info["atc_type"] else hs.TEXT):
      t.draw_mid(f"감속:{info['x_turn_info']}", gx, gy, 40, hs.TEXT, hs.BOLD, align=0.5)

    if number:
      # Without an instruction line the distance sits on the arrow's centre.
      base = y + 134 if text else gy + NAV_DISTANCE_SIZE * hs.INTER_CAP / 2
      nw = t.draw(number, x + 234, base, NAV_DISTANCE_SIZE, hs.TEXT, hs.SEMI, spacing=-3)
      t.draw(unit, x + 234 + nw + 14, base, NAV_UNIT_SIZE, hs.with_alpha(hs.TEXT, 184), hs.MEDIUM)
    if text:
      text = t.ellipsize(text, NAV_TEXT_SIZE, w - 234 - 48, hs.SEMI)
      t.draw(text, x + 236, y + 200, NAV_TEXT_SIZE, hs.TEXT, hs.SEMI)

    progress = self._progress(key, dist)
    bar_y = y + h - 40
    hs.card(x + 30, bar_y, w - 60, 8, hs.rgba(255, 255, 255, 33), 4, None)
    if progress > 0.01:
      hs.card(x + 30, bar_y, max(8.0, (w - 60) * progress), 8, bar, 4, None)

  # ---- trip capsule (bottom-right) ----------------------------------------------------------

  def _trip_runs(self, remain_sec: int, dist_m: int) -> list[tuple[list[tuple[str, bool]], str]]:
    """Arrival time, remaining time and distance: ((text, is_number) runs, caption) per column."""
    # Arrival time is wall-clock based; monotonic time cannot be converted to local time.
    eta_tm = time.localtime(time.time() + remain_sec)  # noqa: TID251
    minutes = max(1, round(remain_sec / 60.0))
    if minutes >= 60:
      duration = [(str(minutes // 60), True), ("시간", False)] + ([(str(minutes % 60), True), ("분", False)] if minutes % 60 else [])
    else:
      duration = [(str(minutes), True), ("분", False)]
    if ui_state.is_metric:
      km = dist_m / 1000.0
      distance = [(f"{km:.1f}" if km < 100 else f"{km:.0f}", True), ("km", False)]
    else:
      mi = dist_m / 1609.344
      distance = [(f"{mi:.1f}" if mi < 100 else f"{mi:.0f}", True), ("mi", False)]
    return [([(f"{eta_tm.tm_hour:02d}:{eta_tm.tm_min:02d}", True)], "도착 예정"),
            (duration, "남은 시간"), (distance, "남은 거리")]

  def _trip_run_width(self, runs: list[tuple[str, bool]]) -> float:
    t = self._type
    width = 0.0
    for i, (text, number) in enumerate(runs):
      if i:
        width += 12 if number else 6  # unit hugs its figure; the next figure gets air
      width += t.width(text, TRIP_SIZE, hs.SEMI, -1.0) if number else t.width(text, TRIP_UNIT_SIZE, hs.SEMI)
    return width

  def _draw_trip(self, rect: rl.Rectangle, info: dict) -> None:
    remain_sec, dist_m = info["n_go_pos_time"], info["n_go_pos_dist"]
    if rect.width < 1200 or not (dist_m > 0 and remain_sec > 0):
      return

    t = self._type
    columns = self._trip_runs(remain_sec, dist_m)
    # Each column hugs its own figure, so a long duration does not widen the others.
    widths = [max(TRIP_COL_MIN, max(self._trip_run_width(runs), t.width(caption, TRIP_LABEL_SIZE, hs.MEDIUM)) + TRIP_COL_PAD * 2)
              for runs, caption in columns]
    w = sum(widths)
    road = info["road_name"]
    if road:
      road = t.ellipsize(road, TRIP_ROAD_SIZE, w - 2 * TRIP_COL_PAD - 34, hs.SEMI)
    h = TRIP_STATS_H + (TRIP_HEAD_H if road else 0)
    right = rect.x + rect.width - M_X
    y = rect.y + rect.height - M_BOTTOM - h
    x = right - w
    hs.glass_card(x, y, w, h, CARD_R)
    hs.panels.append((x, y, w, h))
    hs.zones["trip"] = (x, y, w, h)

    top = y
    if road:
      # Header: the road we are on, quiet, above a hairline.
      mid = y + TRIP_HEAD_H / 2 + 4
      hs.dot(x + TRIP_COL_PAD + 7, mid, 7, NAV_BLUE)
      t.draw_mid(road, x + TRIP_COL_PAD + 30, mid, TRIP_ROAD_SIZE, hs.TEXT_2, hs.SEMI)
      hs.card(x + 26, y + TRIP_HEAD_H, w - 52, 2, hs.HAIRLINE, 1, None)
      top = y + TRIP_HEAD_H

    # Stats: big figure with its unit, caption underneath; columns split by hairlines.
    base = top + 84
    caption_mid = top + TRIP_STATS_H - 34
    left = x
    for i, ((runs, caption), col_w) in enumerate(zip(columns, widths, strict=True)):
      cx = left + col_w / 2
      if i:
        hs.card(left - 1, top + 30, 2, TRIP_STATS_H - 60, hs.HAIRLINE, 1, None)
      left += col_w
      tx = cx - self._trip_run_width(runs) / 2
      for j, (text, number) in enumerate(runs):
        if j:
          tx += 12 if number else 6
        if number:
          tx += t.draw(text, tx, base, TRIP_SIZE, hs.TEXT, hs.SEMI, spacing=-1.0)
        else:
          tx += t.draw(text, tx, base, TRIP_UNIT_SIZE, hs.TEXT_3, hs.SEMI)
      t.draw_mid(caption, cx, caption_mid, TRIP_LABEL_SIZE, hs.with_alpha(hs.TEXT, 150), hs.MEDIUM, align=0.5)

  def _banner_alert_active(self) -> bool:
    """Small/mid alerts take the top-centre slot from the clock."""
    try:
      return int(ui_state.sm['selfdriveState'].alertSize.raw) in (1, 2)
    except Exception:
      return False


class PlotRenderer:
  PLOT_MAX = 400

  def __init__(self):
    self._plot_size = 0
    self._plot_index = 0
    self._plot_queue = [[0.0] * self.PLOT_MAX for _ in range(3)]
    self._plot_min = 0.0
    self._plot_max = 0.0
    self._plot_x = 480.0
    self._plot_width = 1000.0
    self._plot_y = 40.0
    self._plot_height = 300.0
    self._plot_dx = 2.0
    self._show_plot_mode_prev = -1
    self._type = hs.Type()

  def _clear(self):
    self._plot_size = 0
    self._plot_index = 0
    self._plot_min = 0.0
    self._plot_max = 0.0
    self._plot_queue = [[0.0] * self.PLOT_MAX for _ in range(3)]

  def _make_plot_data(self, sm, show_plot_mode: int):
    car_state = sm['carState']
    lp = sm['longitudinalPlan']
    car_control = sm['carControl']
    controls_state = sm['controlsState']

    a_ego = float(car_state.aEgo)
    v_ego = float(car_state.vEgo)

    accel = 0.0
    try:
      accel = float(lp.accels[0])
    except Exception:
      pass

    speeds_0 = 0.0
    try:
      speeds_0 = float(lp.speeds[0])
    except Exception:
      pass

    accel_out = 0.0
    try:
      accel_out = float(car_control.actuators.accel)
    except Exception:
      pass

    if show_plot_mode in (0, 1):
      return [a_ego, accel, accel_out], '1.Accel (Y:a_ego, G:a_target, O:a_out)'

    if show_plot_mode == 2:
      return [speeds_0, v_ego, a_ego], '2.Speed/Accel(Y:speed_0, G:v_ego, O:a_ego)'

    if show_plot_mode == 3:
      pos_32 = 0.0
      vel_32 = 0.0
      vel_0 = 0.0
      try:
        pos_32 = float(sm['modelV2'].position.x[32])
      except Exception:
        pass
      try:
        vel_32 = float(sm['modelV2'].velocity.x[32])
      except Exception:
        pass
      try:
        vel_0 = float(sm['modelV2'].velocity.x[0])
      except Exception:
        pass
      return [pos_32, vel_32, vel_0], '3.Model(Y:pos_32, G:vel_32, O:vel_0)'

    if show_plot_mode == 4:
      a_lead_k = 0.0
      v_rel = 0.0
      try:
        a_lead_k = float(sm['radarState'].leadOne.aLeadK)
      except Exception:
        pass
      try:
        v_rel = float(sm['radarState'].leadOne.vRel)
      except Exception:
        pass
      return [accel, a_lead_k, v_rel], '4.Lead(Y:accel, G:a_lead, O:v_rel)'

    if show_plot_mode == 5:
      a_lead = 0.0
      j_lead = 0.0
      try:
        a_lead = float(sm['radarState'].leadOne.aLead)
      except Exception:
        pass
      try:
        j_lead = float(sm['radarState'].leadOne.jLead)
      except Exception:
        pass
      return [a_ego, a_lead, j_lead], '5.Lead(Y:a_ego, G:a_lead, O:j_lead)'

    if show_plot_mode == 6:
      actual_lat_accel = 0.0
      desired_lat_accel = 0.0
      output = 0.0
      try:
        actual_lat_accel = float(controls_state.lateralControlState.torqueState.actualLateralAccel) * 10.0
      except Exception:
        pass
      try:
        desired_lat_accel = float(controls_state.lateralControlState.torqueState.desiredLateralAccel) * 10.0
      except Exception:
        pass
      try:
        output = float(controls_state.lateralControlState.torqueState.output) * 10.0
      except Exception:
        pass
      return [actual_lat_accel, desired_lat_accel, output], '6.Steer(Y:actual, G:desire, O:output)'

    if show_plot_mode == 7:
      actual_angle = float(car_state.steeringAngleDeg)
      target_angle = 0.0
      angle_offset = 0.0
      try:
        target_angle = float(car_control.actuators.steeringAngleDeg)
      except Exception:
        pass
      try:
        angle_offset = float(sm['liveParameters'].angleOffsetDeg) * 10.0
      except Exception:
        pass
      return [actual_angle, target_angle, angle_offset], '7.SteerA (Y:Actual, G:Target, O:Offset*10)'

    if show_plot_mode == 8:
      curvature = 0.0
      try:
        curvature = float(car_control.actuators.curvature) * 10000.0
      except Exception:
        pass
      return [curvature, curvature, curvature], '8.Curvature (*10000)'

    return [0.0, 0.0, 0.0], 'no data'

  def _update_plot_queue(self, plot_data):
    self._plot_index = (self._plot_index + 1) % self.PLOT_MAX

    for i in range(3):
      self._plot_queue[i][self._plot_index] = float(plot_data[i])

    if self._plot_size < self.PLOT_MAX:
      self._plot_size += 1

    self._plot_min = float('inf')
    self._plot_max = float('-inf')
    for i in range(3):
      values = self._plot_queue[i][:self._plot_size] if self._plot_size < self.PLOT_MAX else self._plot_queue[i]
      self._plot_min = min(self._plot_min, min(values))
      self._plot_max = max(self._plot_max, max(values))

    if self._plot_min == float('inf'):
      self._plot_min = -2.0
    if self._plot_max == float('-inf'):
      self._plot_max = 2.0

    if self._plot_min > -2.0:
      self._plot_min = -2.0
    if self._plot_max < 2.0:
      self._plot_max = 2.0

  def _draw_plotting(self, index: int, x_base: float, y_base: float, color, font):
    if self._plot_size <= 0:
      return

    plot_range = self._plot_max - self._plot_min
    plot_ratio = self._plot_height if plot_range < 1.0 else (self._plot_height / plot_range)

    prev = None
    latest_x = None
    latest_y = None
    latest_value = 0.0

    for i in range(self._plot_size):
      data = self._plot_queue[index][(self._plot_index - i + self.PLOT_MAX) % self.PLOT_MAX]
      plot_y = y_base + self._plot_height - (data - self._plot_min) * plot_ratio
      plot_x = x_base + (self._plot_size - i) * self._plot_dx

      pt = rl.Vector2(plot_x, plot_y)
      if prev is not None:
        rl.draw_line_ex(prev, pt, 3.0, color)
      else:
        latest_x = plot_x
        latest_y = plot_y
        latest_value = data
      prev = pt

    if latest_x is not None and latest_y is not None:
      # Latest values stack in a fixed legend column so close traces never overprint.
      self._type.draw(f'{latest_value:.2f}', latest_x + 18, y_base + 44 + index * 44, 36, color, hs.BOLD, shadow=True)

  def draw(self, rect: rl.Rectangle, font, show_plot_mode: int) -> None:
    if show_plot_mode == 0:
      return
    try:
      if not ui_state.sm.alive['carState'] or not ui_state.sm.alive['longitudinalPlan']:
        return
    except Exception:
      return

    if show_plot_mode != self._show_plot_mode_prev:
      self._clear()
      self._show_plot_mode_prev = show_plot_mode

    try:
      plot_data, title = self._make_plot_data(ui_state.sm, show_plot_mode)
    except Exception:
      return

    self._update_plot_queue(plot_data)

    if rect.width < 1200:
      return

    x_base = rect.x + self._plot_x
    y_base = rect.y + self._plot_y
    colors = [hs.AMBER, hs.GREEN, hs.CARROT]

    for i in range(3):
      self._draw_plotting(i, x_base, y_base, colors[i], font)

    self._type.draw(title, x_base, y_base + 6, 32, hs.TEXT_2, hs.SEMI, shadow=True)
