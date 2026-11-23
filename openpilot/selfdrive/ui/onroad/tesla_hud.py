"""Tesla V12 / v15 Unified Borderless HUD + Quiet OLED Mode + 3-Mode Swipe Dock.

Shared visual language across all three onroad display modes on Comma 3X (2160x1080):
  - Mode 0 (TESLA_MODE_3D, default): 3D FSD driving world + borderless V12 left HUD + top-right frosted nav/camera card
  - Mode 1 (TESLA_MODE_AR):          Real road AR camera + borderless V12 left HUD + minimal bottom-right ETA pill
  - Mode 2 (TESLA_MODE_QUIET):       Cybercab OLED obsidian split layout + typographic nav + always-on 3-mode dock

Preserves all critical Carrot driving cues (enforcement cameras, traffic light state,
SoftHold, top-exit manual start, blind-spot crimson edge glow, turn signals, low-TPMS
warning, eGPU/Jetlink badge, and PlotRenderer debug plots) without box clutter.
"""
import math
import time
from typing import Optional

import pyray as rl

from openpilot.common.constants import CV
from openpilot.selfdrive.ui.onroad import hud_style as hs
from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state
from openpilot.system.hardware.usbgpu import usbgpu_badge_state
from openpilot.system.ui.lib.application import FONT_DIR, FontWeight, gui_app
from openpilot.system.ui.widgets import Widget

TESLA_MODE_3D = 0
TESLA_MODE_AR = 1
TESLA_MODE_QUIET = 2
TESLA_MODE_NAMES = ("3D FSD", "AR 카메라", "Quiet")

# Tesla OLED Accent Palette
TESLA_BLUE = hs.rgba(59, 164, 255)
TESLA_BLUE_TEXT = hs.rgba(78, 178, 255)
TESLA_CYAN = hs.rgba(108, 200, 255)
TESLA_EMERALD = hs.rgba(46, 229, 157)
TESLA_AMBER = hs.rgba(255, 182, 46)
TESLA_RED = hs.rgba(255, 68, 62)
TESLA_SIGN_RED = hs.rgba(225, 39, 41)
TESLA_CARD_DARK = (hs.rgba(18, 25, 39, 204), hs.rgba(13, 18, 29, 210), hs.rgba(10, 14, 23, 218))
TESLA_CARD_LIGHT = (hs.rgba(255, 255, 255, 240), hs.rgba(248, 251, 255, 242), hs.rgba(240, 245, 252, 245))
TESLA_CARD_WARN = (hs.rgba(96, 22, 28, 222), hs.rgba(62, 12, 17, 228), hs.rgba(40, 8, 12, 234))
TESLA_CARD_WARN_DAY = (hs.rgba(218, 36, 44, 244), hs.rgba(192, 26, 34, 246), hs.rgba(168, 20, 28, 248))

# Daytime Studio Light HUD Ink & Accents (Mode 0 3D FSD when ambient light is daytime)
TESLA_DAY_INK = hs.rgba(16, 23, 36)
TESLA_DAY_INK_SUB = hs.rgba(72, 88, 112)
TESLA_DAY_BLUE = hs.rgba(14, 108, 242)
TESLA_DAY_EMERALD = hs.rgba(5, 132, 88)
TESLA_DAY_AMBER = hs.rgba(217, 119, 6)
TESLA_DAY_RED = hs.rgba(220, 38, 38)


class TeslaDayNightController:
  """Auto day/night theme resolver driven by C3X ui_state.light_sensor (0..100).

  Uses EMA smoothing + wide hysteresis (day >= 42, night <= 26) + sustained hold
  (1.5 s) so brief highway overpasses or streetlights never strobe the UI, while
  snapping immediately on the first valid camera exposure sample.
  """

  DAY_ON_THRESHOLD = 42.0
  DAY_OFF_THRESHOLD = 26.0
  INIT_SPLIT_THRESHOLD = 34.0
  HOLD_SEC = 1.5
  EMA_TAU = 1.8

  def __init__(self, default_day: Optional[bool] = None):
    if default_day is None:
      hr = time.localtime().tm_hour
      default_day = (7 <= hr < 19)
    self._is_day: bool = bool(default_day)
    self._ema: Optional[float] = None
    self._last_t: Optional[float] = None
    self._pending_state: Optional[bool] = None
    self._pending_since: float = 0.0
    self._force_day: Optional[bool] = None

  @property
  def is_day(self) -> bool:
    return self._force_day if self._force_day is not None else self._is_day

  def set_override(self, force_day: Optional[bool]) -> None:
    self._force_day = None if force_day is None else bool(force_day)

  def update(self, light_sensor: float, now: Optional[float] = None) -> bool:
    if self._force_day is not None:
      return self._force_day
    if now is None:
      now = time.monotonic()

    ls = float(light_sensor)
    if ls < 0.0:
      # Sensor not live yet — keep clock-seeded / current state without drifting
      self._last_t = now
      return self._is_day

    if self._ema is None:
      self._ema = ls
      self._is_day = ls >= self.INIT_SPLIT_THRESHOLD
      self._last_t = now
      self._pending_state = None
      return self._is_day

    dt = max(0.0, min(1.0, now - (self._last_t if self._last_t is not None else now)))
    self._last_t = now
    alpha = 1.0 - math.exp(-dt / self.EMA_TAU) if dt > 0.0 else 0.0
    self._ema += (ls - self._ema) * alpha

    want: Optional[bool] = None
    if not self._is_day and self._ema >= self.DAY_ON_THRESHOLD:
      want = True
    elif self._is_day and self._ema <= self.DAY_OFF_THRESHOLD:
      want = False

    if want is None:
      self._pending_state = None
    elif self._pending_state != want:
      self._pending_state = want
      self._pending_since = now
    elif (now - self._pending_since) >= self.HOLD_SEC:
      self._is_day = want
      self._pending_state = None

    return self._is_day

# Driving mode names & accent dot colors (Carrot myDrivingMode: 1=연비, 2=안전, 3=일반, 4=고속)
DRIVING_MODES = {
  1: ("연비", hs.rgba(46, 229, 157)),
  2: ("안전", hs.rgba(255, 182, 46)),
  3: ("일반", TESLA_BLUE),
  4: ("고속", hs.rgba(255, 92, 84)),
}

# Gear shifter mapping (matching cereal CarState.GearShifter)
GEAR_LABELS = ("P", "R", "N", "D")
_GEAR_MAP = {
  "park": "P", "reverse": "R", "neutral": "N", "drive": "D",
  "sport": "D", "low": "D", "eco": "D", "manumatic": "D",
}

_OVERRIDE_LABELS = {
  1: ("ECO", TESLA_EMERALD),
  2: ("CAM", TESLA_RED),
  3: ("CURVE", TESLA_AMBER),
  4: ("LIMIT", TESLA_AMBER),
  5: ("TURN", TESLA_BLUE),
}


def _format_car_label(fingerprint: str) -> str:
  fp = str(fingerprint or "").strip()
  if not fp:
    return "IONIQ 5"
  u = fp.upper()
  if "IONIQ_5" in u or "IONIQ5" in u:
    return "IONIQ 5"
  if "IONIQ_6" in u or "IONIQ6" in u:
    return "IONIQ 6"
  if "EV6" in u:
    return "EV6"
  if "EV9" in u:
    return "EV9"
  parts = [p for p in fp.replace("_", " ").split() if p.upper() not in ("HYUNDAI", "KIA", "GENESIS", "CAR")]
  return " ".join(parts[:2]).title() if parts else fp[:14]


def _format_dist_parts(dist_m: float) -> tuple[str, str]:
  d = max(0.0, float(dist_m))
  if d >= 1000.0:
    return f"{d / 1000.0:.1f}", "km"
  if d >= 100.0:
    return str(int(round(d / 10.0) * 10)), "m"
  return str(int(round(d))), "m"


def _fit_text(t: hs.Type, text: str, size: float, max_w: float, weight: str) -> str:
  """Truncates text with ASCII '...' (avoids missing Unicode U+2026 '?' glyph in Hangul bitmap atlas)."""
  if not text or t.width(text, size, weight) <= max_w:
    return text
  dots = "..."
  lo, hi = 0, len(text)
  while lo < hi:
    mid = (lo + hi + 1) // 2
    if t.width(text[:mid].rstrip() + dots, size, weight) <= max_w:
      lo = mid
    else:
      hi = mid - 1
  return text[:lo].rstrip() + dots if lo > 0 else ""


def _format_eta(remain_sec: float, now_struct=None) -> tuple[str, str]:
  sec = max(0.0, float(remain_sec))
  mins = max(1, int(round(sec / 60.0)))
  t_now = time.time() if now_struct is None else time.mktime(now_struct)
  arr = time.localtime(t_now + sec)
  return f"{arr.tm_hour:02d}:{arr.tm_min:02d}", str(mins)


class _LightNumeralFont:
  """Loads Inter-Medium.ttf digits at 390px for sunlight-legible speedometer numerals on 6-inch C3X OLED."""

  def __init__(self):
    self._font: Optional[rl.Font] = None
    self._loaded = False
    self._failed = False

  def font(self) -> Optional[rl.Font]:
    if self._loaded:
      return self._font
    if self._failed:
      return None
    self._loaded = True
    try:
      ttf_path = None
      for fname in ("Inter-Medium.ttf", "Inter-SemiBold.ttf", "Inter-Regular.ttf"):
        p1 = FONT_DIR / fname
        if p1.exists():
          ttf_path = p1
          break
        p2 = FONT_DIR.parent.parent / "assets" / "fonts" / fname
        if p2.exists():
          ttf_path = p2
          break
      if ttf_path is None:
        self._failed = True
        return None
      chars = [ord(c) for c in "0123456789-."]
      cps = rl.ffi.new("int[]", chars)
      f = rl.load_font_ex(str(ttf_path), 390, cps, len(chars))
      if int(f.texture.id) <= 0:
        self._failed = True
        return None
      rl.gen_texture_mipmaps(f.texture)
      rl.set_texture_filter(f.texture, rl.TextureFilter.TEXTURE_FILTER_TRILINEAR)
      self._font = f
      return f
    except Exception:
      self._failed = True
      return None


class TeslaHudRenderer(Widget):
  """Unified Tesla V12/v15 HUD with 3-mode swipe state, dock hit-testing, and Quiet OLED mode."""

  def __init__(self):
    super().__init__()
    self._type = hs.Type()
    self._num_font = _LightNumeralFont()
    self._plot_renderer = None
    self._font_display = gui_app.font(FontWeight.DISPLAY) if gui_app.has_font(FontWeight.DISPLAY) else None

    # Active display mode: default = Mode 0 (3D FSD)
    self._mode: int = TESLA_MODE_3D
    self._prev_mode: int = TESLA_MODE_3D
    self._mode_switch_t: float = -10.0
    self._dock_visible_until: float = time.monotonic() + 4.0
    self._drag_offset_x: float = 0.0
    self._day_mode: bool = False

    # Hit rectangles updated each frame for direct dock / handle tap interaction
    self._dock_tab_rects: list[tuple[float, float, float, float]] = []
    self._handle_rect: Optional[tuple[float, float, float, float]] = None

    # External 3D lead screen anchor set by AugmentedRoadView when in Mode 0
    self._lead_3d_anchor: Optional[tuple[float, float, float, float, bool]] = None

    # Smooth EMA states
    self._accel_ema = 0.0
    self._last_t = time.monotonic()
    self._params_next_t = 0.0
    self._show_plot_mode = 0
    self._cruise_gap_levels = 4
    self._longitudinal_personality = 2
    self._car_label = "IONIQ 5"
    self._op_long = True
    self._user_interacting = False

    # Clock cache (only reformat when minute changes)
    self._clock_minute_key = None
    self._clock_text = "14:32"

  @property
  def mode(self) -> int:
    return self._mode

  @property
  def day_mode(self) -> bool:
    return self._day_mode

  def set_day_mode(self, is_day: bool) -> None:
    self._day_mode = bool(is_day)

  @property
  def _is_light_ui(self) -> bool:
    return self._day_mode and self._mode == TESLA_MODE_3D

  def set_mode(self, mode: int, show_dock: bool = True) -> None:
    clamped = max(TESLA_MODE_3D, min(TESLA_MODE_QUIET, int(mode)))
    now = time.monotonic()
    if clamped != self._mode:
      self._prev_mode = self._mode
      self._mode = clamped
      self._mode_switch_t = now
    if show_dock:
      self._dock_visible_until = now + 3.4

  def step_mode(self, delta: int) -> bool:
    """Steps mode by delta (-1 right swipe, +1 left swipe). Returns True if mode changed."""
    target = max(TESLA_MODE_3D, min(TESLA_MODE_QUIET, self._mode + int(delta)))
    now = time.monotonic()
    self._dock_visible_until = now + 3.4
    if target != self._mode:
      self._prev_mode = self._mode
      self._mode = target
      self._mode_switch_t = now
      return True
    return False

  def reveal_dock(self, seconds: float = 4.0) -> None:
    self._dock_visible_until = time.monotonic() + seconds

  def set_drag_offset(self, dx: float) -> None:
    self._drag_offset_x = float(dx)

  def set_lead_3d_anchor(self, anchor: Optional[tuple[float, float, float, float, bool]]) -> None:
    self._lead_3d_anchor = anchor

  def user_interacting(self) -> bool:
    return self._user_interacting

  def dock_is_visible(self, now: Optional[float] = None) -> bool:
    t = time.monotonic() if now is None else now
    return self._mode == TESLA_MODE_QUIET or t < self._dock_visible_until or abs(self._drag_offset_x) > 18.0

  def hit_test_dock(self, x: float, y: float) -> Optional[int]:
    """Returns tapped mode index (0, 1, 2) if (x, y) hits a dock tab, -1 if swipe handle hit, else None."""
    if self.dock_is_visible() and self._dock_tab_rects:
      for idx, (rx, ry, rw, rh) in enumerate(self._dock_tab_rects):
        if rx - 8 <= x <= rx + rw + 8 and ry - 14 <= y <= ry + rh + 14:
          return idx
    if self._handle_rect is not None:
      hx, hy, hw, hh = self._handle_rect
      if hx - 48 <= x <= hx + hw + 48 and hy - 44 <= y <= hy + hh + 24:
        return -1
    return None

  def _refresh_params(self, now: float) -> None:
    if now < self._params_next_t:
      return
    self._params_next_t = now + 1.0
    params = getattr(ui_state, "params", None)
    if params is None:
      return
    try:
      self._show_plot_mode = int(params.get_int("ShowPlotMode"))
    except Exception:
      self._show_plot_mode = 0
    try:
      g = int(params.get_int("CruiseGapLevels"))
      self._cruise_gap_levels = g if 2 <= g <= 4 else 4
    except Exception:
      self._cruise_gap_levels = 4
    try:
      self._longitudinal_personality = int(params.get_int("LongitudinalPersonality"))
    except Exception:
      pass

  def _update_clock(self) -> str:
    now_st = time.localtime()
    key = (now_st.tm_year, now_st.tm_yday, now_st.tm_hour, now_st.tm_min)
    if key != self._clock_minute_key:
      self._clock_minute_key = key
      self._clock_text = f"{now_st.tm_hour:02d}:{now_st.tm_min:02d}"
    return self._clock_text

  def _read_snapshot(self, dt: float) -> dict:
    sm = ui_state.sm
    is_metric = bool(getattr(ui_state, "is_metric", True))
    conv = CV.MS_TO_KPH if is_metric else CV.MS_TO_MPH

    snap = {
      "speed": 0,
      "unit": "km/h" if is_metric else "mph",
      "gear": "D",
      "status": getattr(ui_state, "status", UIStatus.DISENGAGED),
      "engaged": getattr(ui_state, "status", UIStatus.DISENGAGED) == UIStatus.ENGAGED or bool(getattr(ui_state, "lat_active", False)),
      "override": getattr(ui_state, "status", UIStatus.DISENGAGED) == UIStatus.OVERRIDE,
      "set_speed": 0,
      "cruise_set": False,
      "limit_speed": 0,
      "cam_dist": 0.0,
      "cam_alarm": False,
      "override_tag": None,
      "driving_mode": 3,
      "gap": max(1, min(self._cruise_gap_levels, self._longitudinal_personality + 1)),
      "accel": 0.0,
      "left_bsm": False,
      "right_bsm": False,
      "left_blink": False,
      "right_blink": False,
      "soft_hold": False,
      "tlv_top_exit": False,
      "traffic_light": 0,  # 0=none, 1=red, 2=green
      "turn_info": 0,
      "dist_turn": 0.0,
      "tbt_text": "",
      "remain_dist": 0.0,
      "remain_time": 0.0,
      "lead": None,
      "low_tpms": [],
      "gps_ok": True,
      "active_carrot": 0,  # 0=off, 1=APM, >=2=APN
    }

    if sm is None:
      return snap

    if sm.alive.get("carParams", False):
      cp = sm["carParams"]
      self._car_label = _format_car_label(getattr(cp, "carFingerprint", ""))
      self._op_long = bool(getattr(cp, "openpilotLongitudinalControl", True))

    if sm.alive.get("carState", False):
      cs = sm["carState"]
      v_disp = float(getattr(cs, "vEgoCluster", 0.0))
      if v_disp <= 0.01:
        v_disp = float(getattr(cs, "vEgo", 0.0))
      snap["speed"] = max(0, int(round(v_disp * conv)))

      raw_gear = str(getattr(cs, "gearShifter", "drive")).lower()
      for k, gl in _GEAR_MAP.items():
        if k in raw_gear:
          snap["gear"] = gl
          break

      a_ego = float(getattr(cs, "aEgo", 0.0))
      self._accel_ema += (a_ego - self._accel_ema) * min(1.0, dt * 6.5 + 0.16)
      snap["accel"] = self._accel_ema

      cs_state = getattr(cs, "cruiseState", None)
      if cs_state is not None:
        sp = float(getattr(cs_state, "speedCluster", getattr(cs_state, "speed", 0.0)))
        if sp > 0.5 and sp < 250.0:
          snap["set_speed"] = int(round(sp * conv))
          snap["cruise_set"] = True

      snap["left_bsm"] = bool(getattr(cs, "leftBlindspot", False))
      snap["right_bsm"] = bool(getattr(cs, "rightBlindspot", False))
      snap["left_blink"] = bool(getattr(cs, "leftBlinker", False))
      snap["right_blink"] = bool(getattr(cs, "rightBlinker", False))
      snap["soft_hold"] = int(getattr(cs, "softHoldActive", 0) or 0) > 0

      tpms = getattr(cs, "tpms", None)
      if tpms is not None:
        for code, attr in (("FL", "fl"), ("FR", "fr"), ("RL", "rl"), ("RR", "rr")):
          val = float(getattr(tpms, attr, 0.0) or 0.0)
          if 5.0 <= val < 31.0:
            snap["low_tpms"].append((code, int(round(val))))

    if sm.alive.get("selfdriveState", False):
      ss = sm["selfdriveState"]
      if not snap["cruise_set"]:
        mc = float(getattr(ss, "maxCruiseSpeed", 0.0) or 0.0)
        if mc > 0.5 and mc < 250.0:
          snap["set_speed"] = int(round(mc * conv))
          snap["cruise_set"] = True

    if sm.alive.get("carrotMan", False):
      cm = sm["carrotMan"]
      snap["active_carrot"] = int(getattr(cm, "activeCarrot", 0) or 0)
      spd_lim = int(float(getattr(cm, "xSpdLimit", 0) or 0))
      sign_type = int(float(getattr(cm, "xSignType", 0) or 0))
      road_lim = int(float(getattr(cm, "nRoadLimitSpeed", 0) or 0))
      spd_dist = float(getattr(cm, "xSpdDist", 0.0) or 0.0)

      if spd_lim > 0 and sign_type != 22:
        snap["limit_speed"] = spd_lim
        snap["cam_dist"] = spd_dist
        snap["cam_alarm"] = spd_dist > 0.0 or snap["speed"] > spd_lim + 2
      elif road_lim > 0:
        snap["limit_speed"] = road_lim

      # Carrot override source / applied target speed
      app_spd = int(float(getattr(cm, "desiredSpeed", 0) or 0))
      if app_spd > 0 and snap["cruise_set"] and app_spd < snap["set_speed"] - 1:
        snap["set_speed"] = app_spd
        if snap["cam_alarm"]:
          snap["override_tag"] = _OVERRIDE_LABELS[2]
        else:
          snap["override_tag"] = _OVERRIDE_LABELS[1]

      snap["turn_info"] = int(getattr(cm, "xTurnInfo", 0) or 0)
      snap["dist_turn"] = float(getattr(cm, "xDistToTurn", 0.0) or 0.0)
      snap["tbt_text"] = str(getattr(cm, "szTBTMainText", "") or "").strip()
      snap["remain_dist"] = float(getattr(cm, "xDistToDest", 0.0) or 0.0)
      snap["remain_time"] = float(getattr(cm, "xTimeToDest", 0.0) or 0.0)
      snap["tlv_top_exit"] = bool(getattr(cm, "tlvTopExit", False))

      # Driving mode & gap from carrotMan if published
      dm = int(getattr(cm, "myDrivingMode", 0) or 0)
      if dm in DRIVING_MODES:
        snap["driving_mode"] = dm
      tg = int(getattr(cm, "mySafeModeFactor", 0) or 0)
      if 1 <= tg <= self._cruise_gap_levels:
        snap["gap"] = tg

    if sm.alive.get("longitudinalPlan", False):
      lp = sm["longitudinalPlan"]
      ts = int(getattr(lp, "trafficState", 0) or 0)
      # Carrot trafficState: 1=red/stop, 2=green/go
      if ts in (1, 2):
        snap["traffic_light"] = ts
      dm_lp = int(getattr(lp, "myDrivingMode", 0) or 0)
      if dm_lp in DRIVING_MODES:
        snap["driving_mode"] = dm_lp
      gap_lp = int(getattr(lp, "cruiseGap", 0) or 0)
      if 1 <= gap_lp <= self._cruise_gap_levels:
        snap["gap"] = gap_lp

    if sm.valid.get("radarState", False):
      lead = getattr(sm["radarState"], "leadOne", None)
      if lead is not None and bool(getattr(lead, "status", False)):
        d = float(getattr(lead, "dRel", 0.0))
        if d > 1.0:
          v_l = max(0.0, float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0))) * conv)
          v_r = float(getattr(lead, "vRel", 0.0))
          snap["lead"] = (d, v_l, v_r < -3.0)

    return snap

  # ------------------------------------------------------------------
  # Quiet Mode Background (called in place of camera/3D when Mode 2)
  # ------------------------------------------------------------------
  def render_quiet_background(self, rect: rl.Rectangle) -> None:
    """Pure obsidian OLED backdrop with whisper masked cobalt horizon wave."""
    rl.draw_rectangle_rec(rect, hs.rgba(4, 6, 10, 255))

    cx = rect.x + rect.width * 0.5
    bot = rect.y + rect.height
    # Soft sapphire horizon radial glow at bottom center + gentle left cluster wash
    hs.ellipse_glow(cx, bot + 10.0, rect.width * 0.46, 300.0, hs.rgba(28, 114, 255, 62))
    hs.ellipse_glow(rect.x + rect.width * 0.22, rect.y + rect.height * 0.42, 460.0, 360.0, hs.rgba(44, 132, 255, 22))

    # Whisper fading horizon beam & subtle perspective lane hints cleanly above the dock
    hor_y = bot - 178.0
    hs.hgradient_line(rect.x + 180.0, hor_y, rect.width - 360.0, 2.0, hs.rgba(72, 178, 255, 86))
    hs.ellipse_glow(cx, hor_y, 380.0, 24.0, hs.rgba(48, 152, 255, 56))

    # Subtle perspective dashed lane guides dissolving downward
    for side in (-1.0, 1.0):
      x0, y0 = cx + side * 52.0, hor_y + 4.0
      x1, y1 = cx + side * 290.0, bot - 12.0
      for seg in range(5):
        t0 = seg * 0.20
        t1 = t0 + 0.11
        ax = x0 + (x1 - x0) * t0
        ay = y0 + (y1 - y0) * t0
        bx = x0 + (x1 - x0) * t1
        by = y0 + (y1 - y0) * t1
        a = int(34 * (1.0 - t0 * 0.7))
        rl.draw_line_ex(rl.Vector2(ax, ay), rl.Vector2(bx, by), 2.2, hs.rgba(190, 220, 255, a))

  # ------------------------------------------------------------------
  # Shared Tesla V12 Primitives
  # ------------------------------------------------------------------
  def _draw_power_regen_bar(self, rect: rl.Rectangle, accel: float, quiet: bool, light: bool = False) -> None:
    """Extreme-left vertical Tesla EV power/regen bar sized for 6-inch C3X OLED peripheral vision."""
    x = rect.x + 14.0
    top = rect.y + 132.0
    bar_h = 816.0
    w = 18.0
    mid_y = top + bar_h * 0.52

    track_col = hs.rgba(16, 26, 44, 34) if light else hs.rgba(255, 255, 255, 36 if quiet else 48)
    tick_col = hs.rgba(16, 26, 44, 148) if light else hs.rgba(255, 255, 255, 168)
    hs.pill(x, top, w, bar_h, track_col)
    rl.draw_rectangle_rounded(rl.Rectangle(x - 6.0, mid_y - 2.5, w + 12.0, 5.0), 1.0, 4, tick_col)

    if accel >= 0.05:
      frac = min(1.0, accel / 2.2)
      fh = max(24.0, frac * (mid_y - top - 18.0))
      fy = mid_y - fh - 5.0
      if not light:
        hs.ellipse_glow(x + w * 0.5, fy + fh * 0.4, 28.0, fh * 0.58, hs.rgba(130, 206, 255, 108))
      power_grad = (
        (hs.rgba(12, 92, 232, 255), hs.rgba(28, 126, 255, 248), hs.rgba(72, 162, 255, 224))
        if light else
        (hs.rgba(255, 255, 255, 255), hs.rgba(148, 212, 255, 242), hs.rgba(59, 164, 255, 195))
      )
      hs.gradient_rect(x, fy, w, fh, min(w * 0.42, fh * 0.42), power_grad, stops=(0.0, 0.5, 1.0))
      hs.gradient_rect(x + 3.0, mid_y + 12.0, w - 6.0, 96.0, 4.5,
                       (hs.rgba(14, 158, 108, 72) if light else hs.rgba(46, 229, 157, 78), hs.rgba(46, 229, 157, 6)),
                       stops=(0.0, 1.0))
    elif accel <= -0.05:
      frac = min(1.0, (-accel) / 2.8)
      fh = max(24.0, frac * (top + bar_h - mid_y - 18.0))
      fy = mid_y + 6.0
      if not light:
        hs.ellipse_glow(x + w * 0.5, fy + fh * 0.55, 30.0, fh * 0.58, hs.rgba(46, 229, 157, 126))
      regen_grad = (
        (hs.rgba(8, 150, 102, 245), hs.rgba(18, 184, 126, 252), hs.rgba(52, 214, 154, 255))
        if light else
        (hs.rgba(46, 229, 157, 215), hs.rgba(68, 242, 172, 252), hs.rgba(182, 255, 224, 255))
      )
      hs.gradient_rect(x, fy, w, fh, min(w * 0.42, fh * 0.42), regen_grad, stops=(0.0, 0.5, 1.0))
    else:
      hs.gradient_rect(x + 2.5, mid_y - 60.0, w - 5.0, 52.0, 4.5,
                       (hs.rgba(24, 118, 242, 120) if light else hs.rgba(165, 218, 255, 148),
                        hs.rgba(59, 164, 255, 64) if light else hs.rgba(59, 164, 255, 88)),
                       stops=(0.0, 1.0))
      hs.gradient_rect(x + 3.0, mid_y + 12.0, w - 6.0, 78.0, 4.5,
                       (hs.rgba(14, 158, 108, 60) if light else hs.rgba(46, 229, 157, 64), hs.rgba(46, 229, 157, 6)),
                       stops=(0.0, 1.0))

  def _draw_steering_wheel_icon(self, cx: float, cy: float, r: float, color: rl.Color, glow_on: bool) -> None:
    if glow_on:
      hs.glow(cx, cy, r * 1.9, hs.with_alpha(color, 72))
    stroke = max(5.5, r * 0.21)
    rl.draw_ring(rl.Vector2(cx, cy), r - stroke, r, 0.0, 360.0, 36, color)
    rl.draw_circle_v(rl.Vector2(cx, cy), r * 0.26, color)
    hs.curve((cx - r * 0.84, cy - r * 0.12), (cx, cy - r * 0.24), (cx + r * 0.84, cy - r * 0.12), stroke * 0.95, color, steps=8)
    rl.draw_line_ex(rl.Vector2(cx, cy + r * 0.20), rl.Vector2(cx, cy + r * 0.88), stroke, color)

  def _draw_monumental_speed(self, value: str, x: float, baseline: float, size: float,
                             color: rl.Color, shadow: bool = True) -> float:
    f = self._num_font.font()
    if f is None:
      return self._type.draw(value, x, baseline, size, color, hs.SEMI, align=0.0, shadow=shadow, spacing=-size * 0.045)
    spacing = -size * 0.048
    w = float(rl.measure_text_ex(f, value, size, spacing).x)
    top = baseline - size * 0.78
    if shadow:
      pos_s = rl.Vector2(float(x), float(top + 9.0))
      getattr(rl, "_orig_draw_text_ex", rl.draw_text_ex)(f, value, pos_s, size, spacing, hs.rgba(0, 0, 0, 170))
    pos = rl.Vector2(float(x), float(top))
    getattr(rl, "_orig_draw_text_ex", rl.draw_text_ex)(f, value, pos, size, spacing, color)
    return w

  def _draw_gear_strip(self, x: float, baseline: float, active_gear: str, size: float = 54.0,
                       light: bool = False) -> None:
    cx = x
    for g in GEAR_LABELS:
      is_on = (g == active_gear)
      if light:
        col = TESLA_DAY_INK if is_on else hs.rgba(20, 32, 52, 86)
      else:
        col = hs.TEXT if is_on else hs.rgba(255, 255, 255, 92)
      gs = size + 8.0 if is_on else size
      w = self._type.draw(g, cx, baseline, gs, col, hs.BOLD if is_on else hs.SEMI, shadow=not light)
      cx += w + 34.0

  def _draw_max_dial(self, cx: float, cy: float, dia: float, set_speed: int, cruise_set: bool,
                     engaged: bool, override_tag: Optional[tuple[str, rl.Color]],
                     light: bool = False) -> None:
    r = dia * 0.5
    if override_tag is not None:
      lbl, ring_col = override_tag
    elif engaged and cruise_set:
      lbl, ring_col = "MAX", TESLA_DAY_BLUE if light else TESLA_BLUE
    elif cruise_set:
      lbl, ring_col = "MAX", hs.rgba(104, 122, 148, 228) if light else hs.rgba(148, 168, 196, 228)
    else:
      lbl, ring_col = "MAX", hs.rgba(128, 142, 164, 170) if light else hs.rgba(255, 255, 255, 86)

    if engaged and cruise_set and not light:
      hs.glow(cx, cy, r * 1.48, hs.with_alpha(ring_col, 88))
    hs.glow(cx, cy + 7.0, r * 1.18, hs.rgba(16, 28, 48, 62) if light else hs.rgba(0, 0, 0, 140))
    disc_fill = hs.rgba(255, 255, 255, 250) if light else hs.rgba(11, 18, 30, 220)
    rl.draw_circle_v(rl.Vector2(cx, cy), r, disc_fill)
    rl.draw_ring(rl.Vector2(cx, cy), r - max(8.0, dia * 0.048), r, 0.0, 360.0, 48, ring_col)

    lbl_col = hs.with_alpha(ring_col, 252) if cruise_set else (TESLA_DAY_INK_SUB if light else hs.rgba(255, 255, 255, 148))
    lbl_sz = max(32.0, dia * 0.175)
    self._type.draw_mid(lbl, cx, cy - r * 0.39, lbl_sz, lbl_col, hs.BOLD, align=0.5, spacing=1.2)
    val_txt = str(set_speed) if cruise_set else "--"
    val_size = dia * (0.41 if len(val_txt) >= 3 else 0.47)
    val_col = (TESLA_DAY_INK if cruise_set else hs.rgba(96, 112, 136, 190)) if light else (
      hs.TEXT if cruise_set else hs.rgba(255, 255, 255, 145)
    )
    self._type.draw_mid(val_txt, cx, cy + r * 0.20, val_size, val_col, hs.BOLD, align=0.5, spacing=-2.5)

  def _draw_speed_limit_disc(self, cx: float, cy: float, dia: float, limit_speed: int, cam_alarm: bool) -> None:
    if limit_speed <= 0:
      return
    r = dia * 0.5
    if cam_alarm:
      pulse = 0.65 + 0.35 * math.sin(time.monotonic() * 6.5)
      hs.glow(cx, cy, r * 1.72, hs.rgba(255, 58, 52, int(168 * pulse)))
    hs.glow(cx, cy + 7.0, r * 1.18, hs.rgba(0, 0, 0, 148))
    rl.draw_circle_v(rl.Vector2(cx, cy), r, hs.rgba(252, 253, 255, 255))
    rl.draw_ring(rl.Vector2(cx, cy), r - dia * 0.115, r, 0.0, 360.0, 48, TESLA_SIGN_RED)
    txt = str(limit_speed)
    sz = dia * (0.41 if len(txt) >= 3 else 0.49)
    self._type.draw_mid(txt, cx, cy + 1.0, sz, hs.rgba(14, 17, 22, 255), hs.BOLD, align=0.5, spacing=-3.0)

  def _draw_mode_gap_pill(self, x: float, y: float, driving_mode: int, gap: int, max_gap: int,
                          engaged: bool, h: float = 106.0, font_size: float = 52.0,
                          light: bool = False) -> float:
    name, dot_col = DRIVING_MODES.get(driving_mode, DRIVING_MODES[3])
    if light and driving_mode == 3:
      dot_col = TESLA_DAY_BLUE
    elif light and driving_mode == 1:
      dot_col = TESLA_DAY_EMERALD
    if not engaged:
      dot_col = hs.rgba(124, 140, 164, 220) if light else hs.rgba(170, 184, 206, 220)
    w_name = self._type.width(name, font_size, hs.BOLD)
    bar_w = 34.0 if font_size >= 50.0 else 28.0
    bar_h = 15.0 if font_size >= 50.0 else 13.0
    bar_gap = 10.0
    bars_total = max_gap * bar_w + max(0, max_gap - 1) * bar_gap
    pill_w = 36.0 + 22.0 + 20.0 + w_name + 26.0 + 26.0 + bars_total + 34.0

    fill_col = hs.rgba(255, 255, 255, 234) if light else hs.rgba(12, 18, 29, 202)
    ring_col = hs.rgba(18, 32, 56, 40) if light else hs.rgba(255, 255, 255, 62)
    hs.chip(x, y, pill_w, h, fill=fill_col, ring=ring_col, ring_width=2.0)
    cy = y + h * 0.5
    dot_x = x + 42.0
    if not light:
      hs.glow(dot_x, cy, 24.0, hs.with_alpha(dot_col, 112))
    hs.dot(dot_x, cy, 11.0, dot_col)

    name_x = dot_x + 28.0
    self._type.draw_mid(name, name_x, cy, font_size, TESLA_DAY_INK if light else hs.TEXT, hs.BOLD)

    sep_x = name_x + w_name + 24.0
    rl.draw_rectangle(int(sep_x), int(cy - 22.0), 2, 44,
                      hs.rgba(18, 32, 56, 44) if light else hs.rgba(255, 255, 255, 68))

    bx = sep_x + 26.0
    if light:
      active_bar_col = TESLA_DAY_BLUE if engaged else hs.rgba(104, 122, 148, 210)
      off_bar_col = hs.rgba(18, 32, 56, 42)
    else:
      active_bar_col = TESLA_CYAN if engaged else hs.rgba(208, 220, 236, 198)
      off_bar_col = hs.rgba(255, 255, 255, 64)
    for i in range(max_gap):
      on = (i < gap)
      if on and engaged and not light:
        hs.glow(bx + bar_w * 0.5, cy, 22.0, hs.rgba(104, 194, 255, 92))
      rl.draw_rectangle_rounded(
        rl.Rectangle(bx, cy - bar_h * 0.5, bar_w, bar_h), 1.0, 4,
        active_bar_col if on else off_bar_col,
      )
      bx += bar_w + bar_gap
    return pill_w

  def _draw_traffic_pill(self, right_x: float, cy: float, traffic_light: int, light: bool = False) -> float:
    """High-visibility traffic signal pill sized for instant glance on 6-inch C3X OLED."""
    if traffic_light not in (1, 2):
      return 0.0
    is_red = (traffic_light == 1)
    col = (TESLA_DAY_RED if light else TESLA_RED) if is_red else (TESLA_DAY_EMERALD if light else TESLA_EMERALD)
    txt = "정지" if is_red else "진행"
    pw, ph = 208.0, 80.0
    px = right_x - pw
    py = cy - ph * 0.5
    fill = hs.rgba(255, 255, 255, 244) if light else hs.rgba(14, 18, 28, 222)
    hs.chip(px, py, pw, ph, fill=fill, ring=hs.with_alpha(col, 224 if light else 200), ring_width=3.2 if light else 3.0)
    if not light:
      hs.glow(px + 44.0, cy, 28.0, hs.with_alpha(col, 138))
    hs.dot(px + 44.0, cy, 13.5, col)
    self._type.draw_mid(txt, px + 76.0, cy, 50.0, TESLA_DAY_INK if light else hs.TEXT, hs.BOLD)
    return pw + 26.0

  def _draw_carrot_link_pill(self, right_x: float, cy: float, active_carrot: int, light: bool = False) -> float:
    """High-visibility Carrot link pill: green APN (activeCarrot>=2) or blue APM (activeCarrot==1)."""
    if active_carrot < 1:
      return 0.0
    is_apn = active_carrot >= 2
    label = "APN" if is_apn else "APM"
    if light:
      accent = TESLA_DAY_EMERALD if is_apn else TESLA_DAY_BLUE
      fill = hs.rgba(234, 252, 243, 244) if is_apn else hs.rgba(234, 244, 255, 244)
      text_col = hs.rgba(4, 114, 76, 255) if is_apn else hs.rgba(10, 84, 206, 255)
    else:
      accent = TESLA_EMERALD if is_apn else TESLA_BLUE
      fill = hs.rgba(10, 40, 28, 224) if is_apn else hs.rgba(10, 26, 48, 224)
      text_col = hs.rgba(176, 255, 220, 255) if is_apn else hs.rgba(172, 224, 255, 255)

    w_txt = self._type.width(label, 46.0, hs.BOLD, spacing=1.0)
    pw, ph = 70.0 + w_txt + 26.0, 76.0
    px = right_x - pw
    py = cy - ph * 0.5
    hs.chip(px, py, pw, ph, fill=fill, ring=hs.with_alpha(accent, 224), ring_width=3.0)
    if not light:
      hs.glow(px + 38.0, cy, 26.0, hs.with_alpha(accent, 138))
    hs.dot(px + 38.0, cy, 11.5, accent)
    self._type.draw_mid(label, px + 68.0, cy + 1.0, 46.0, text_col, hs.BOLD, spacing=1.0)
    return pw + 22.0

  # ------------------------------------------------------------------
  # Mode 0 (3D FSD) & Mode 1 (AR 카메라) Left Cluster (6-inch C3X Scale)
  # ------------------------------------------------------------------
  def _draw_left_cluster_ab(self, rect: rl.Rectangle, snap: dict, is_ar: bool) -> None:
    light = (not is_ar) and self._day_mode
    x = rect.x + 56.0
    y = rect.y + 38.0

    # Soft radial dark cushion behind borderless type on night 3D mode (AR uses optical scrim; day sky needs none)
    if not is_ar and not light:
      hs.ellipse_glow(rect.x + 320.0, rect.y + 470.0, 680.0, 560.0, hs.rgba(6, 9, 15, 152))

    self._draw_gear_strip(x, y + 54.0, snap["gear"], size=54.0, light=light)

    # Monumental 6-inch C3X speedometer (sunlight-legible Inter-Medium)
    spd_str = str(snap["speed"])
    overspeed = snap["limit_speed"] > 0 and snap["speed"] > snap["limit_speed"] + 4
    if snap["cam_alarm"] and overspeed:
      spd_col = TESLA_DAY_RED if light else TESLA_RED
    else:
      spd_col = TESLA_DAY_INK if light else hs.TEXT
    spd_size = 318.0 if is_ar else 338.0
    spd_base = y + (342.0 if is_ar else 356.0)
    spd_w = self._draw_monumental_speed(spd_str, x - 14.0, spd_base, spd_size, spd_col, shadow=not light)
    unit_col = TESLA_DAY_INK_SUB if light else hs.rgba(255, 255, 255, 212 if is_ar else 168)
    self._type.draw(snap["unit"], x - 14.0 + spd_w + 14.0, spd_base - 22.0, 52.0,
                    unit_col, hs.SEMI, shadow=not light)

    # Steering wheel icon + Self-Driving / Override / Manual
    fsd_y = spd_base + 72.0
    if snap["engaged"]:
      st_col, st_txt = (TESLA_DAY_BLUE if light else TESLA_BLUE_TEXT), "Self-Driving"
    elif snap["override"]:
      st_col, st_txt = (TESLA_DAY_AMBER if light else TESLA_AMBER), "Override"
    else:
      st_col, st_txt = (TESLA_DAY_INK_SUB if light else hs.rgba(182, 196, 216, 232)), "Manual"
    self._draw_steering_wheel_icon(x + 32.0, fsd_y, 31.0, st_col, glow_on=(snap["engaged"] and not light))
    self._type.draw_mid(st_txt, x + 82.0, fsd_y, 56.0, st_col, hs.BOLD, shadow=not light)

    # Speed badges: full C3X-scale MAX circular dial + Korean speed limit sign (204px = real SIGN_R 100)
    badge_cy = fsd_y + 154.0
    dia = 204.0
    self._draw_max_dial(x + dia * 0.5, badge_cy, dia, snap["set_speed"], snap["cruise_set"],
                        snap["engaged"], snap["override_tag"], light=light)
    if snap["limit_speed"] > 0:
      self._draw_speed_limit_disc(x + dia + 28.0 + dia * 0.5, badge_cy, dia,
                                  snap["limit_speed"], snap["cam_alarm"])

    # Driving mode & gap frosted pill
    pill_y = badge_cy + dia * 0.5 + 34.0
    self._draw_mode_gap_pill(x, pill_y, snap["driving_mode"], snap["gap"],
                             self._cruise_gap_levels, snap["engaged"], h=106.0, font_size=52.0, light=light)

    # Only show GPS warning pill if GPS fix is actually lost (zero microscopic car-name fluff on 6" OLED)
    if not snap["gps_ok"]:
      gy = rect.y + rect.height - 96.0
      g_fill = hs.rgba(255, 247, 230, 240) if light else hs.rgba(46, 32, 12, 218)
      g_ring = TESLA_DAY_AMBER if light else TESLA_AMBER
      hs.chip(x, gy, 240.0, 68.0, fill=g_fill, ring=g_ring, ring_width=2.4)
      hs.dot(x + 34.0, gy + 34.0, 9.5, g_ring)
      self._type.draw_mid("GPS 대기", x + 62.0, gy + 34.0, 40.0, TESLA_DAY_INK if light else hs.TEXT, hs.BOLD)

    # Register left safe zone at top banner height (y=32..228; 3-digit speed ends ~485px, km/h sits below)
    hs.zones["guide"] = (rect.x, rect.y, 492.0, rect.height)

  def _draw_right_card_3d(self, rect: rl.Rectangle, snap: dict, clock_txt: str) -> None:
    light = self._day_mode
    right_x = rect.x + rect.width - 42.0
    top_y = rect.y + 66.0

    # Top-right big bold clock + optional traffic light + APN/APM Carrot link badge
    clk_col = TESLA_DAY_INK if light else hs.rgba(255, 255, 255, 236)
    clk_w = self._type.draw_mid(clock_txt, right_x, top_y, 76.0, clk_col, hs.BOLD, align=1.0, shadow=not light)
    cur_x = right_x - clk_w - 26.0
    cur_x -= self._draw_traffic_pill(cur_x, top_y, snap["traffic_light"], light=light)
    cur_x -= self._draw_carrot_link_pill(cur_x, top_y, snap["active_carrot"], light=light)

    card_w = 740.0
    card_x = right_x - card_w
    card_y = rect.y + 132.0

    # Priority 1: Speed Enforcement Camera Card (vivid crimson accent in both day & night!)
    if snap["cam_alarm"] and snap["limit_speed"] > 0:
      card_h = 326.0
      hs.glass_card(card_x, card_y, card_w, card_h, r=38.0,
                    body=TESLA_CARD_WARN_DAY if light else TESLA_CARD_WARN)
      self._draw_speed_limit_disc(card_x + 114.0, card_y + 128.0, 166.0, snap["limit_speed"], cam_alarm=True)

      val_s, unit_s = _format_dist_parts(snap["cam_dist"]) if snap["cam_dist"] > 0 else ("단속", "")
      tx = card_x + 226.0
      dw = self._type.draw(val_s, tx, card_y + 132.0, 118.0, hs.TEXT, hs.BOLD, spacing=-3.0)
      if unit_s:
        self._type.draw(unit_s, tx + dw + 10.0, card_y + 132.0, 58.0, hs.rgba(255, 255, 255, 224), hs.SEMI)
      self._type.draw("과속 단속 구간", tx, card_y + 206.0, 54.0, hs.rgba(255, 236, 232, 252), hs.BOLD)

      prog_y = card_y + 258.0
      prog_w = card_w - 72.0
      hs.pill(card_x + 36.0, prog_y, prog_w, 16.0, hs.rgba(255, 255, 255, 58))
      frac = 1.0 - min(1.0, max(0.0, snap["cam_dist"] / 800.0)) if snap["cam_dist"] > 0 else 0.9
      fill_w = max(40.0, prog_w * frac)
      bar_grad = (hs.rgba(255, 255, 255, 245), hs.rgba(255, 210, 206, 255)) if light else (TESLA_RED, hs.rgba(255, 134, 122, 255))
      hs.gradient_rect(card_x + 36.0, prog_y, fill_w, 16.0, 8.0, bar_grad)
      hs.zones["tpms"] = (card_x, card_y, card_w, card_h)
      return

    # Priority 2: Big Bold Frosted Turn-by-Turn Navigation + ETA Card (real C3X NAV_TEXT/NAV_DIST scale)
    has_nav = (snap["turn_info"] > 0 or snap["dist_turn"] > 0 or bool(snap["tbt_text"]) or snap["remain_dist"] > 0)
    if not has_nav:
      top_left = min(right_x - 360.0, cur_x)
      hs.zones["tpms"] = (top_left, rect.y, right_x - top_left, 126.0)
      return

    has_eta = snap["remain_dist"] > 0 or snap["remain_time"] > 0
    card_h = 362.0 if has_eta else 258.0
    hs.glass_card(card_x, card_y, card_w, card_h, r=38.0,
                  body=TESLA_CARD_LIGHT if light else TESLA_CARD_DARK)
    if light:
      rl.draw_rectangle_rounded_lines_ex(
        rl.Rectangle(card_x, card_y, card_w, card_h),
        min(1.0, 76.0 / card_h), 16, 2.0, hs.rgba(18, 32, 56, 34),
      )

    # Giant maneuver arrow + C3X-scale distance + bold road name
    icon_cx, icon_cy = card_x + 92.0, card_y + 108.0
    arrow_col = TESLA_DAY_BLUE if light else TESLA_CYAN
    if snap["turn_info"] > 0:
      hs.maneuver_arrow(snap["turn_info"], icon_cx, icon_cy, 0.84, arrow_col)
    else:
      hs.polyline([(icon_cx, icon_cy + 40), (icon_cx, icon_cy - 38)], 10.0, arrow_col)
      hs.polyline([(icon_cx - 26, icon_cy - 10), (icon_cx, icon_cy - 38), (icon_cx + 26, icon_cy - 10)], 10.0, arrow_col)

    tx = card_x + 178.0
    primary_ink = TESLA_DAY_INK if light else hs.TEXT
    sub_ink = TESLA_DAY_INK_SUB if light else hs.rgba(255, 255, 255, 190)
    road_ink = hs.rgba(24, 34, 52, 248) if light else hs.rgba(255, 255, 255, 242)
    if snap["dist_turn"] > 0:
      val_s, unit_s = _format_dist_parts(snap["dist_turn"])
      dw = self._type.draw(val_s, tx, card_y + 122.0, 118.0, primary_ink, hs.BOLD, spacing=-3.5)
      self._type.draw(unit_s, tx + dw + 10.0, card_y + 122.0, 58.0, sub_ink, hs.SEMI)
    road_str = snap["tbt_text"] or "경로 안내 중"
    road_el = _fit_text(self._type, road_str, 54.0, card_w - 210.0, hs.BOLD)
    self._type.draw(road_el, tx, card_y + 198.0, 54.0, road_ink, hs.BOLD)

    prog_y = card_y + 234.0
    prog_w = card_w - 72.0
    hs.pill(card_x + 36.0, prog_y, prog_w, 12.0, hs.rgba(18, 32, 56, 28) if light else hs.rgba(255, 255, 255, 36))
    frac = 1.0 - min(1.0, max(0.0, snap["dist_turn"] / 1200.0)) if snap["dist_turn"] > 0 else 0.55
    fill_w = max(40.0, prog_w * max(0.14, frac))
    if not light:
      hs.ellipse_glow(card_x + 36.0 + fill_w * 0.5, prog_y + 6.0, fill_w * 0.55, 22.0, hs.rgba(68, 178, 255, 98))
    hs.gradient_rect(card_x + 36.0, prog_y, fill_w, 12.0, 6.0, (hs.rgba(16, 108, 244, 255), hs.rgba(68, 174, 255, 255)))

    if has_eta:
      eta_y = card_y + 304.0
      arr_str, min_str = _format_eta(snap["remain_time"])
      rem_km = f"{snap['remain_dist'] / 1000.0:.1f}" if snap["remain_dist"] >= 1000.0 else f"{int(round(snap['remain_dist']))}m"
      self._type.draw_mid(arr_str, card_x + 36.0, eta_y, 54.0, primary_ink, hs.BOLD)
      self._type.draw_mid(f"{min_str}분", card_x + card_w * 0.52, eta_y, 54.0,
                          TESLA_DAY_BLUE if light else TESLA_CYAN, hs.BOLD, align=0.5)
      dist_lbl = f"{rem_km} km" if snap["remain_dist"] >= 1000.0 else rem_km
      self._type.draw_mid(dist_lbl, card_x + card_w - 36.0, eta_y, 54.0,
                          hs.rgba(52, 66, 90, 240) if light else hs.rgba(255, 255, 255, 232), hs.BOLD, align=1.0)

    hs.zones["tpms"] = (card_x, card_y, card_w, card_h)

  def _draw_3d_lead_tag(self, rect: rl.Rectangle) -> None:
    anchor = self._lead_3d_anchor
    if anchor is None:
      return
    light = self._day_mode
    sx, sy, d_rel, v_kph, closing = anchor
    unit = "km/h" if ui_state.is_metric else "mph"
    v_disp = v_kph if ui_state.is_metric else v_kph * 0.621371
    spd_num = str(int(round(v_disp)))
    dist_txt = f"{int(round(d_rel))}m"

    w_spd = self._type.width(spd_num, 98.0, hs.BOLD, spacing=-3.0)
    w_unit = self._type.width(unit, 28.0, hs.SEMI)
    w_dist = self._type.width(dist_txt, 44.0, hs.SEMI)
    pw = 32.0 + w_spd + 6.0 + w_unit + 22.0 + 2.0 + 22.0 + w_dist + 30.0
    ph = 118.0

    max_x = rect.x + rect.width - pw - 48.0
    tpms_zone = hs.zones.get("tpms")
    py = max(rect.y + 132.0, sy - ph - 26.0)
    if tpms_zone is not None and py < tpms_zone[1] + tpms_zone[3] + 16.0:
      max_x = min(max_x, tpms_zone[0] - pw - 20.0)
    px = max(rect.x + 620.0, min(max_x, sx - pw * 0.5))

    if light:
      if closing:
        ring_col = hs.rgba(255, 255, 255, 235)
        stem_col = hs.rgba(220, 34, 42, 240)
        fill_col = hs.rgba(218, 34, 42, 244)
        spd_col = hs.rgba(255, 255, 255, 255)
        unit_col = hs.rgba(255, 224, 220, 235)
        dist_col = hs.rgba(255, 232, 228, 245)
        sep_col = hs.rgba(255, 255, 255, 96)
      else:
        ring_col = hs.rgba(20, 118, 246, 238)
        stem_col = ring_col
        fill_col = hs.rgba(255, 255, 255, 246)
        spd_col = hs.rgba(14, 21, 34, 255)
        unit_col = hs.rgba(76, 94, 120, 225)
        dist_col = hs.rgba(66, 94, 130, 240)
        sep_col = hs.rgba(18, 32, 56, 48)
    else:
      ring_col = hs.rgba(255, 86, 78, 248) if closing else hs.rgba(103, 196, 255, 224)
      stem_col = hs.with_alpha(ring_col, 215)
      fill_col = hs.rgba(52, 12, 16, 238) if closing else hs.rgba(10, 17, 30, 236)
      spd_col = hs.rgba(255, 98, 90, 255) if closing else hs.rgba(255, 255, 255, 255)
      unit_col = hs.rgba(255, 178, 172, 220) if closing else hs.rgba(255, 255, 255, 178)
      dist_col = hs.rgba(255, 198, 194, 224) if closing else hs.rgba(164, 198, 228, 224)
      sep_col = hs.rgba(255, 255, 255, 68)

    if closing:
      hs.ellipse_glow(px + pw * 0.5, py + ph * 0.5, pw * 0.62, ph * 0.90, hs.rgba(232, 42, 46, 76 if light else 86))
    rl.draw_line_ex(rl.Vector2(px + pw * 0.5, py + ph), rl.Vector2(sx, sy), 4.0, stem_col)
    hs.chip(px, py, pw, ph, fill=fill_col, ring=ring_col, ring_width=3.8 if closing else 3.4)

    cy = py + ph * 0.5
    tx = px + 32.0
    if not light:
      hs.ellipse_glow(tx + w_spd * 0.5, cy, max(62.0, w_spd * 0.68), 44.0,
                      hs.rgba(255, 68, 62, 64) if closing else hs.rgba(72, 184, 255, 58))
    self._type.draw_mid(spd_num, tx, cy - 2.0, 98.0, spd_col, hs.BOLD, spacing=-3.0)
    tx += w_spd + 6.0
    self._type.draw_mid(unit, tx, cy + 21.0, 28.0, unit_col, hs.SEMI)
    tx += w_unit + 22.0
    rl.draw_rectangle(int(tx), int(cy - 26.0), 2, 52, sep_col)
    tx += 24.0
    self._type.draw_mid(dist_txt, tx, cy + 1.0, 44.0, dist_col, hs.SEMI)

  def _draw_right_ar(self, rect: rl.Rectangle, snap: dict, clock_txt: str) -> None:
    right_x = rect.x + rect.width - 42.0
    top_y = rect.y + 66.0
    clk_w = self._type.draw_mid(clock_txt, right_x, top_y, 76.0, hs.rgba(255, 255, 255, 248), hs.BOLD, align=1.0, shadow=True)
    cur_x = right_x - clk_w - 26.0
    cur_x -= self._draw_traffic_pill(cur_x, top_y, snap["traffic_light"])
    cur_x -= self._draw_carrot_link_pill(cur_x, top_y, snap["active_carrot"])

    # In AR mode, ONLY show a bold top-right crimson card when an enforcement camera is active
    if snap["cam_alarm"] and snap["limit_speed"] > 0:
      cw, ch = 630.0, 212.0
      cx, cy = right_x - cw, rect.y + 132.0
      hs.glass_card(cx, cy, cw, ch, r=36.0, body=TESLA_CARD_WARN)
      self._draw_speed_limit_disc(cx + 96.0, cy + ch * 0.5, 144.0, snap["limit_speed"], cam_alarm=True)
      val_s, unit_s = _format_dist_parts(snap["cam_dist"]) if snap["cam_dist"] > 0 else ("단속", "")
      self._type.draw_mid(f"{val_s} {unit_s}".strip(), cx + 194.0, cy + 72.0, 84.0, hs.TEXT, hs.BOLD)
      self._type.draw_mid("과속 단속 구간", cx + 194.0, cy + 150.0, 48.0, hs.rgba(255, 224, 220, 248), hs.BOLD)
      hs.zones["tpms"] = (cx, cy, cw, ch)
    else:
      top_left = min(right_x - 360.0, cur_x)
      hs.zones["tpms"] = (top_left, rect.y, right_x - top_left, 126.0)

    # Single bold floating bottom-right ETA capsule (compact 3 values, 50px Bold for 6-inch C3X!)
    if snap["remain_dist"] > 0 or snap["remain_time"] > 0:
      arr_str, min_str = _format_eta(snap["remain_time"])
      dist_str = f"{snap['remain_dist'] / 1000.0:.1f}km" if snap["remain_dist"] >= 1000.0 else f"{int(round(snap['remain_dist']))}m"
      p1 = arr_str
      p2 = f"{min_str}분"
      p3 = dist_str
      w1 = self._type.width(p1, 50.0, hs.BOLD)
      w2 = self._type.width(p2, 50.0, hs.BOLD)
      w3 = self._type.width(p3, 50.0, hs.BOLD)
      pw = 72.0 + w1 + w2 + w3 + 96.0
      ph = 96.0
      px = right_x - pw
      py = rect.y + rect.height - 34.0 - ph
      hs.chip(px, py, pw, ph, fill=hs.rgba(10, 15, 24, 212), ring=hs.rgba(255, 255, 255, 68), ring_width=2.2)
      cy = py + ph * 0.5
      tx = px + 36.0
      self._type.draw_mid(p1, tx, cy, 50.0, hs.TEXT, hs.BOLD)
      tx += w1 + 24.0
      rl.draw_rectangle(int(tx), int(cy - 22.0), 2, 44, hs.rgba(255, 255, 255, 76))
      tx += 24.0
      self._type.draw_mid(p2, tx, cy, 50.0, TESLA_CYAN, hs.BOLD)
      tx += w2 + 24.0
      rl.draw_rectangle(int(tx), int(cy - 22.0), 2, 44, hs.rgba(255, 255, 255, 76))
      tx += 24.0
      self._type.draw_mid(p3, tx, cy, 50.0, hs.rgba(255, 255, 255, 236), hs.BOLD)

  # ------------------------------------------------------------------
  # Mode 2 (Quiet — Cybercab Pure OLED Split, 6-inch C3X Giant Scale)
  # ------------------------------------------------------------------
  def _draw_quiet_layout(self, rect: rl.Rectangle, snap: dict, clock_txt: str) -> None:
    left_x = rect.x + 64.0
    right_x = rect.x + rect.width - 52.0
    top_y = rect.y + 66.0

    # Top bar: P R N D on left | APN/APM + Traffic pill + Giant Clock on right
    self._draw_gear_strip(left_x, top_y + 22.0, snap["gear"], size=56.0)

    clk_w = self._type.draw_mid(clock_txt, right_x, top_y, 76.0, hs.rgba(255, 255, 255, 232), hs.BOLD, align=1.0)
    cur_x = right_x - clk_w - 26.0
    cur_x -= self._draw_traffic_pill(cur_x, top_y, snap["traffic_light"])
    cur_x -= self._draw_carrot_link_pill(cur_x, top_y, snap["active_carrot"])
    if not snap["gps_ok"]:
      gw, gh = 220.0, 68.0
      gx, gy = cur_x - gw, top_y - gh * 0.5
      hs.chip(gx, gy, gw, gh, fill=hs.rgba(46, 32, 12, 218), ring=TESLA_AMBER, ring_width=2.4)
      hs.dot(gx + 34.0, top_y, 9.5, TESLA_AMBER)
      self._type.draw_mid("GPS 대기", gx + 60.0, top_y, 38.0, hs.TEXT, hs.BOLD)
      cur_x = gx - 22.0
    quiet_top_left = min(rect.x + rect.width - 480.0, cur_x)

    # Left column: Giant 390px speed + Self-Driving + inline 192px MAX / Limit / Mode pill
    spd_str = str(snap["speed"])
    overspeed = snap["limit_speed"] > 0 and snap["speed"] > snap["limit_speed"] + 4
    spd_col = TESLA_RED if (snap["cam_alarm"] and overspeed) else hs.TEXT
    spd_base = rect.y + 468.0
    spd_w = self._draw_monumental_speed(spd_str, left_x - 16.0, spd_base, 390.0, spd_col, shadow=False)
    self._type.draw(snap["unit"], left_x - 16.0 + spd_w + 18.0, spd_base - 24.0, 56.0,
                    hs.rgba(255, 255, 255, 168), hs.SEMI)

    fsd_y = spd_base + 74.0
    if snap["engaged"]:
      st_col, st_txt = TESLA_BLUE_TEXT, "Self-Driving"
    elif snap["override"]:
      st_col, st_txt = TESLA_AMBER, "Override"
    else:
      st_col, st_txt = hs.rgba(182, 196, 216, 232), "Manual"
    self._draw_steering_wheel_icon(left_x + 34.0, fsd_y, 33.0, st_col, glow_on=snap["engaged"])
    self._type.draw_mid(st_txt, left_x + 86.0, fsd_y, 58.0, st_col, hs.BOLD)

    ctrl_cy = fsd_y + 162.0
    dia = 192.0
    self._draw_max_dial(left_x + dia * 0.5, ctrl_cy, dia, snap["set_speed"], snap["cruise_set"],
                        snap["engaged"], snap["override_tag"])
    cur_x = left_x + dia + 22.0
    if snap["limit_speed"] > 0:
      self._draw_speed_limit_disc(cur_x + dia * 0.5, ctrl_cy, dia, snap["limit_speed"], snap["cam_alarm"])
      cur_x += dia + 22.0
    self._draw_mode_gap_pill(cur_x, ctrl_cy - 52.0, snap["driving_mode"], snap["gap"],
                             self._cruise_gap_levels, snap["engaged"], h=104.0, font_size=48.0)

    # Whisper-thin vertical center glass divider
    div_x = rect.x + 1044.0
    div_top = rect.y + 186.0
    div_h = 626.0
    rl.draw_rectangle_gradient_v(int(div_x), int(div_top), 2, int(div_h * 0.25), hs.rgba(255, 255, 255, 0), hs.rgba(255, 255, 255, 48))
    rl.draw_rectangle(int(div_x), int(div_top + div_h * 0.25), 2, int(div_h * 0.50), hs.rgba(255, 255, 255, 48))
    rl.draw_rectangle_gradient_v(int(div_x), int(div_top + div_h * 0.75), 2, int(div_h * 0.25), hs.rgba(255, 255, 255, 48), hs.rgba(255, 255, 255, 0))

    # Right column: Giant Typographic Navigation + 3-column ETA + Bold ACC Lead Trust Pill
    rx = div_x + 68.0
    avail_w = right_x - rx

    if snap["cam_alarm"] and snap["limit_speed"] > 0:
      self._draw_speed_limit_disc(rx + 88.0, rect.y + 286.0, 176.0, snap["limit_speed"], cam_alarm=True)
      val_s, unit_s = _format_dist_parts(snap["cam_dist"]) if snap["cam_dist"] > 0 else ("단속", "")
      dw = self._type.draw(val_s, rx + 206.0, rect.y + 336.0, 152.0, hs.TEXT, hs.BOLD, spacing=-4.0)
      if unit_s:
        self._type.draw(unit_s, rx + 206.0 + dw + 14.0, rect.y + 336.0, 68.0, hs.rgba(255, 255, 255, 190), hs.SEMI)
      self._type.draw("과속 단속 구간 주의", rx, rect.y + 428.0, 62.0, hs.rgba(255, 210, 206, 250), hs.BOLD)
      prog_y = rect.y + 474.0
      hs.pill(rx, prog_y, min(920.0, avail_w), 14.0, hs.rgba(255, 255, 255, 36))
      frac = 1.0 - min(1.0, max(0.0, snap["cam_dist"] / 800.0)) if snap["cam_dist"] > 0 else 0.9
      hs.gradient_rect(rx, prog_y, max(48.0, min(920.0, avail_w) * frac), 14.0, 7.0, (TESLA_RED, hs.rgba(255, 128, 118, 255)))
    elif snap["turn_info"] > 0 or snap["dist_turn"] > 0 or bool(snap["tbt_text"]):
      icon_cx, icon_cy = rx + 68.0, rect.y + 288.0
      if snap["turn_info"] > 0:
        hs.glow(icon_cx, icon_cy, 96.0, hs.rgba(59, 164, 255, 62))
        hs.maneuver_arrow(snap["turn_info"], icon_cx, icon_cy, 0.98, TESLA_CYAN)
      else:
        hs.polyline([(icon_cx, icon_cy + 48), (icon_cx, icon_cy - 44)], 11.0, TESLA_BLUE)
        hs.polyline([(icon_cx - 32, icon_cy - 14), (icon_cx, icon_cy - 44), (icon_cx + 32, icon_cy - 14)], 11.0, TESLA_CYAN)

      val_s, unit_s = _format_dist_parts(snap["dist_turn"]) if snap["dist_turn"] > 0 else ("직진", "")
      dw = self._type.draw(val_s, rx + 168.0, rect.y + 338.0, 152.0, hs.TEXT, hs.BOLD, spacing=-4.0)
      if unit_s:
        self._type.draw(unit_s, rx + 168.0 + dw + 14.0, rect.y + 338.0, 68.0, hs.rgba(255, 255, 255, 178), hs.SEMI)

      road_el = _fit_text(self._type, snap["tbt_text"] or "경로 안내 중", 62.0, avail_w, hs.BOLD)
      self._type.draw(road_el, rx, rect.y + 428.0, 62.0, hs.rgba(255, 255, 255, 242), hs.BOLD)

      prog_w = min(940.0, avail_w)
      prog_y = rect.y + 472.0
      hs.pill(rx, prog_y, prog_w, 14.0, hs.rgba(255, 255, 255, 32))
      frac = 1.0 - min(1.0, max(0.0, snap["dist_turn"] / 1200.0)) if snap["dist_turn"] > 0 else 0.62
      fill_w = max(48.0, prog_w * max(0.16, frac))
      hs.ellipse_glow(rx + fill_w * 0.5, prog_y + 7.0, fill_w * 0.55, 26.0, hs.rgba(78, 182, 255, 104))
      hs.gradient_rect(rx, prog_y, fill_w, 14.0, 7.0, (hs.rgba(30, 112, 255, 255), hs.rgba(108, 199, 255, 255)))
    else:
      self._type.draw("자유 주행 중", rx, rect.y + 344.0, 96.0, hs.rgba(255, 255, 255, 245), hs.BOLD)
      sub = "차로 유지 · 차간거리 제어" if snap["engaged"] else "크루즈 대기 중"
      self._type.draw(sub, rx, rect.y + 432.0, 52.0, hs.rgba(255, 255, 255, 165), hs.SEMI)

    # 3-Column Giant ETA grid
    if snap["remain_dist"] > 0 or snap["remain_time"] > 0:
      eta_y = rect.y + 592.0
      lbl_y = eta_y + 52.0
      arr_str, min_str = _format_eta(snap["remain_time"])
      km_val = f"{snap['remain_dist'] / 1000.0:.1f}" if snap["remain_dist"] >= 1000.0 else str(int(round(snap["remain_dist"])))
      km_unit = "km" if snap["remain_dist"] >= 1000.0 else "m"

      c1, c2, c3 = rx, rx + 340.0, rx + 634.0
      self._type.draw(arr_str, c1, eta_y, 84.0, hs.TEXT, hs.BOLD)
      self._type.draw("도착", c1, lbl_y, 40.0, hs.rgba(255, 255, 255, 152), hs.SEMI)

      wm = self._type.draw(min_str, c2, eta_y, 84.0, TESLA_CYAN, hs.BOLD)
      self._type.draw("분", c2 + wm + 6.0, eta_y, 50.0, hs.rgba(255, 255, 255, 186), hs.SEMI)
      self._type.draw("남은 시간", c2, lbl_y, 40.0, hs.rgba(255, 255, 255, 152), hs.SEMI)

      wk = self._type.draw(km_val, c3, eta_y, 84.0, hs.TEXT, hs.BOLD)
      self._type.draw(km_unit, c3 + wk + 6.0, eta_y, 50.0, hs.rgba(255, 255, 255, 186), hs.SEMI)
      self._type.draw("남은 거리", c3, lbl_y, 40.0, hs.rgba(255, 255, 255, 152), hs.SEMI)

    # Bold ACC Lead Car Trust Pill (lead speed as unmistakable primary hero figure)
    pill_y = rect.y + 706.0
    if snap["lead"] is not None:
      d_rel, v_l, closing = snap["lead"]
      dot_col = TESLA_RED if closing else TESLA_EMERALD
      lbl_txt = "선행차"
      spd_num = str(int(round(v_l)))
      unit_txt = snap["unit"]
      dist_txt = f"{int(round(d_rel))}m"

      w_lbl = self._type.width(lbl_txt, 40.0, hs.SEMI)
      w_spd = self._type.width(spd_num, 108.0, hs.BOLD, spacing=-3.0)
      w_unit = self._type.width(unit_txt, 32.0, hs.SEMI)
      w_dist = self._type.width(dist_txt, 48.0, hs.SEMI)
      pw = 84.0 + w_lbl + 28.0 + w_spd + 8.0 + w_unit + 28.0 + 2.0 + 28.0 + w_dist + 40.0
      ph = 138.0
      hs.card(rx, pill_y, pw, ph,
              fill=hs.rgba(56, 16, 20, 224) if closing else hs.rgba(16, 26, 42, 215),
              radius=36.0,
              edge=hs.rgba(255, 86, 78, 220) if closing else hs.rgba(103, 196, 255, 128),
              edge_width=2.8)
      cy = pill_y + ph * 0.5
      hs.glow(rx + 48.0, cy, 30.0, hs.with_alpha(dot_col, 138))
      hs.dot(rx + 48.0, cy, 13.0, dot_col)
      tx = rx + 84.0
      self._type.draw_mid(lbl_txt, tx, cy + 2.0, 40.0, hs.rgba(255, 255, 255, 165), hs.SEMI)
      tx += w_lbl + 28.0
      hs.ellipse_glow(tx + w_spd * 0.5, cy, max(68.0, w_spd * 0.68), 48.0,
                      hs.rgba(255, 68, 62, 62) if closing else hs.rgba(72, 184, 255, 52))
      self._type.draw_mid(spd_num, tx, cy - 2.0, 108.0, TESLA_RED if closing else hs.TEXT, hs.BOLD, spacing=-3.0)
      tx += w_spd + 8.0
      self._type.draw_mid(unit_txt, tx, cy + 24.0, 32.0, hs.rgba(255, 255, 255, 178), hs.SEMI)
      tx += w_unit + 28.0
      rl.draw_rectangle(int(tx), int(cy - 30.0), 2, 60, hs.rgba(255, 255, 255, 68))
      tx += 30.0
      self._type.draw_mid(dist_txt, tx, cy + 1.0, 48.0,
                          hs.rgba(255, 198, 194, 228) if closing else hs.rgba(164, 198, 228, 228), hs.SEMI)
    else:
      pw, ph = 460.0, 96.0
      hs.card(rx, pill_y, pw, ph, fill=hs.rgba(255, 255, 255, 16), radius=30.0,
              edge=hs.rgba(255, 255, 255, 34), edge_width=1.8)
      cy = pill_y + ph * 0.5
      hs.dot(rx + 44.0, cy, 10.0, TESLA_BLUE if snap["engaged"] else hs.rgba(255, 255, 255, 124))
      self._type.draw_mid("전방 시야 확보", rx + 78.0, cy, 48.0, hs.rgba(255, 255, 255, 212), hs.BOLD)

    hs.zones["guide"] = (rect.x, rect.y, 480.0, 140.0)
    hs.zones["tpms"] = (quiet_top_left, rect.y, rect.x + rect.width - quiet_top_left, 140.0)

  # ------------------------------------------------------------------
  # 3-Mode Bottom Swipe Dock (`3D FSD | AR 카메라 | Quiet`, 6-inch Finger-Friendly!)
  # ------------------------------------------------------------------
  def _draw_mode_dock(self, rect: rl.Rectangle, now: float) -> None:
    light = self._is_light_ui
    cx = rect.x + rect.width * 0.5
    bot = rect.y + rect.height
    self._dock_tab_rects = []

    # Visible bottom center swipe handle bar
    hw, hh = 220.0, 8.0
    hx, hy = cx - hw * 0.5, bot - 22.0
    self._handle_rect = (hx, hy, hw, hh)
    handle_col = hs.rgba(20, 32, 52, 84) if light else hs.rgba(255, 255, 255, 86 if self._mode == TESLA_MODE_QUIET else 74)
    hs.pill(hx, hy, hw, hh, handle_col)

    if self._mode == TESLA_MODE_QUIET:
      alpha = 1.0
    elif abs(self._drag_offset_x) > 18.0:
      alpha = 1.0
    else:
      rem = self._dock_visible_until - now
      if rem <= 0.0:
        return
      alpha = min(1.0, rem / 0.45)

    pad = 12.0
    tab_h = 88.0
    tab_widths = [266.0, 330.0, 238.0]
    gap = 12.0
    dock_w = pad * 2 + sum(tab_widths) + gap * 2
    dock_h = tab_h + pad * 2
    dx = cx - dock_w * 0.5
    dy = bot - 38.0 - dock_h

    dock_fill = hs.rgba(255, 255, 255, int(238 * alpha)) if light else hs.rgba(14, 20, 32, int(226 * alpha))
    dock_ring = hs.rgba(18, 32, 56, int(40 * alpha)) if light else hs.rgba(255, 255, 255, int(62 * alpha))
    hs.chip(dx, dy, dock_w, dock_h, fill=dock_fill, ring=dock_ring, ring_width=2.2, shadow=(alpha > 0.5))

    tx = dx + pad
    ty = dy + pad
    for idx, (label, tw) in enumerate(zip(TESLA_MODE_NAMES, tab_widths, strict=True)):
      self._dock_tab_rects.append((tx, ty, tw, tab_h))
      active = (idx == self._mode)
      if active:
        if light:
          hs.pill(tx, ty, tw, tab_h, hs.rgba(16, 108, 244, int(28 * alpha)),
                  hs.rgba(16, 108, 244, int(92 * alpha)))
          dot_x = tx + 38.0
          hs.dot(dot_x, ty + tab_h * 0.5, 8.5, hs.rgba(14, 108, 242, int(255 * alpha)))
          self._type.draw_mid(label, tx + 62.0 + (tw - 78.0) * 0.5, ty + tab_h * 0.5, 46.0,
                              hs.rgba(14, 21, 34, int(255 * alpha)), hs.BOLD, align=0.5)
        else:
          hs.pill(tx, ty, tw, tab_h, hs.rgba(255, 255, 255, int(52 * alpha)),
                  hs.rgba(255, 255, 255, int(82 * alpha)))
          dot_x = tx + 38.0
          hs.glow(dot_x, ty + tab_h * 0.5, 20.0, hs.rgba(71, 174, 255, int(138 * alpha)))
          hs.dot(dot_x, ty + tab_h * 0.5, 8.0, hs.rgba(71, 174, 255, int(255 * alpha)))
          self._type.draw_mid(label, tx + 62.0 + (tw - 78.0) * 0.5, ty + tab_h * 0.5, 46.0,
                              hs.rgba(255, 255, 255, int(255 * alpha)), hs.BOLD, align=0.5)
      else:
        idle_col = hs.rgba(72, 88, 112, int(224 * alpha)) if light else hs.rgba(255, 255, 255, int(168 * alpha))
        self._type.draw_mid(label, tx + tw * 0.5, ty + tab_h * 0.5, 46.0, idle_col, hs.SEMI, align=0.5)
      tx += tw + gap

  # ------------------------------------------------------------------
  # Global Safety Overlays (All 3 Modes, 6-inch C3X Scale)
  # ------------------------------------------------------------------
  def _draw_safety_overlays(self, rect: rl.Rectangle, snap: dict, now: float) -> None:
    light = self._is_light_ui
    # 1. Tesla-style Blind-Spot Crimson Screen-Edge Glow
    pulse = 0.72 + 0.28 * math.sin(now * 8.0)
    if snap["left_bsm"]:
      a = int(196 * pulse)
      rl.draw_rectangle_gradient_h(int(rect.x), int(rect.y), 180, int(rect.height),
                                   hs.rgba(255, 36, 44, a), hs.rgba(255, 36, 44, 0))
      rl.draw_rectangle(int(rect.x), int(rect.y), 10, int(rect.height), hs.rgba(255, 62, 66, int(252 * pulse)))
    if snap["right_bsm"]:
      a = int(196 * pulse)
      rl.draw_rectangle_gradient_h(int(rect.x + rect.width - 180), int(rect.y), 180, int(rect.height),
                                   hs.rgba(255, 36, 44, 0), hs.rgba(255, 36, 44, a))
      rl.draw_rectangle(int(rect.x + rect.width - 10), int(rect.y), 10, int(rect.height), hs.rgba(255, 62, 66, int(252 * pulse)))

    # 2. Luminous Turn-Signal Chevron Indicators (giant unmistakable blinkers)
    blink_on = (int(now * 3.0) % 2) == 0
    if blink_on:
      if snap["left_blink"]:
        col = (TESLA_DAY_RED if light else TESLA_RED) if snap["left_bsm"] else (TESLA_DAY_EMERALD if light else TESLA_EMERALD)
        bx, by = rect.x + 686.0, rect.y + 72.0
        if not light:
          hs.glow(bx, by, 58.0, hs.with_alpha(col, 122))
        hs.polyline([(bx + 24, by - 28), (bx - 18, by), (bx + 24, by + 28)], 12.0, col)
        hs.polyline([(bx + 56, by - 28), (bx + 14, by), (bx + 56, by + 28)], 9.0, hs.with_alpha(col, 180))
      if snap["right_blink"]:
        col = (TESLA_DAY_RED if light else TESLA_RED) if snap["right_bsm"] else (TESLA_DAY_EMERALD if light else TESLA_EMERALD)
        bx, by = rect.x + rect.width - 780.0, rect.y + 72.0
        if not light:
          hs.glow(bx, by, 58.0, hs.with_alpha(col, 122))
        hs.polyline([(bx - 24, by - 28), (bx + 18, by), (bx - 24, by + 28)], 12.0, col)
        hs.polyline([(bx - 56, by - 28), (bx - 14, by), (bx - 56, by + 28)], 9.0, hs.with_alpha(col, 180))

    # 3. SoftHold Glowing Emerald Pill (68px Bold HOLD, matching real C3X instant legibility!)
    if snap["soft_hold"]:
      pw, ph = 376.0, 114.0
      if not self._banner_alert_active() and not snap["tlv_top_exit"]:
        px = rect.x + (rect.width - pw) * 0.5
        py = rect.y + 34.0
      else:
        px = rect.x + 56.0
        py = rect.y + (836.0 if self._mode == TESLA_MODE_QUIET else 902.0)
      if light:
        hs.chip(px, py, pw, ph, fill=hs.rgba(232, 252, 242, 246), ring=TESLA_DAY_EMERALD, ring_width=3.8)
        hs.dot(px + 54.0, py + ph * 0.5, 14.0, TESLA_DAY_EMERALD)
        self._type.draw_mid("HOLD", px + 88.0 + (pw - 118.0) * 0.5, py + ph * 0.5, 68.0,
                            hs.rgba(4, 102, 68, 255), hs.BOLD, align=0.5, spacing=3.0)
      else:
        hs.ellipse_glow(px + pw * 0.5, py + ph * 0.5, pw * 0.64, ph * 0.92, hs.rgba(46, 229, 157, 94))
        hs.chip(px, py, pw, ph, fill=hs.rgba(10, 26, 22, 238), ring=TESLA_EMERALD, ring_width=3.6)
        hs.dot(px + 54.0, py + ph * 0.5, 14.0, TESLA_EMERALD)
        self._type.draw_mid("HOLD", px + 88.0 + (pw - 118.0) * 0.5, py + ph * 0.5, 68.0, hs.TEXT, hs.BOLD, align=0.5, spacing=3.0)

    # 4. Traffic-Light Top-Exit Manual Start Prompt (yields to AlertRenderer if banner active)
    if snap["tlv_top_exit"] and not self._banner_alert_active():
      pw, ph = 840.0, 124.0
      px = rect.x + (rect.width - pw) * 0.5
      py = rect.y + 34.0
      hs.tile(px, py, pw, ph, 36.0, hs.TILE_AMBER, edge_alpha=100)
      self._type.draw_mid("출발 시 가속 페달을 밟아주세요", px + pw * 0.5, py + ph * 0.5,
                          56.0, hs.rgba(26, 18, 4, 255), hs.BOLD, align=0.5)

    # 5. Low-TPMS Warning Chip ONLY when a tire is actually under-inflated
    if snap["low_tpms"]:
      txt = "TPMS  " + " ".join(f"{c} {v}" for c, v in snap["low_tpms"])
      pw = self._type.width(txt, 50.0, hs.BOLD) + 76.0
      ph = 84.0
      px = rect.x + 56.0
      py = rect.y + rect.height - 118.0
      fill_tpms = hs.rgba(218, 34, 42, 242) if light else hs.rgba(52, 16, 18, 236)
      ring_tpms = hs.rgba(255, 255, 255, 220) if light else TESLA_RED
      hs.chip(px, py, pw, ph, fill=fill_tpms, ring=ring_tpms, ring_width=3.0)
      self._type.draw_mid(txt, px + pw * 0.5, py + ph * 0.5, 50.0, hs.rgba(255, 244, 242, 255), hs.BOLD, align=0.5)

    # 6. eGPU / Jetlink badge if hardware present
    if getattr(ui_state, "usbgpu_present", False):
      st = usbgpu_badge_state(
        getattr(ui_state, "usbgpu_compiled", False),
        getattr(ui_state, "usbgpu_loading", False),
        getattr(ui_state, "usbgpu_active", False),
        getattr(ui_state, "usbgpu_startup_failed", False),
        getattr(ui_state, "usbgpu_compile_pending", False),
      )
      col = (TESLA_DAY_EMERALD if light else TESLA_EMERALD) if st == "active" else (
        (TESLA_DAY_RED if light else TESLA_RED) if st == "error" else (TESLA_DAY_AMBER if light else TESLA_AMBER)
      )
      lbl = f"eGPU {st.upper()}"
      pw = self._type.width(lbl, 40.0, hs.BOLD) + 86.0
      ph = 68.0
      bx = rect.x + 56.0
      by = rect.y + rect.height - 96.0
      hs.chip(bx, by, pw, ph,
              fill=hs.rgba(255, 255, 255, 236) if light else hs.rgba(14, 20, 30, 212),
              ring=hs.with_alpha(col, 190), ring_width=2.2)
      hs.dot(bx + 32.0, by + ph * 0.5, 9.5, col)
      self._type.draw_mid(lbl, bx + 56.0, by + ph * 0.5, 40.0, TESLA_DAY_INK if light else hs.TEXT, hs.BOLD)

  @staticmethod
  def _banner_alert_active() -> bool:
    sm = ui_state.sm
    if sm is None or not sm.alive.get("selfdriveState", False):
      return False
    ss = sm["selfdriveState"]
    return int(getattr(ss, "alertSize", 0) or 0) > 0

  # ------------------------------------------------------------------
  # Main Widget Render Entry Point
  # ------------------------------------------------------------------
  def _render(self, rect: rl.Rectangle) -> None:
    now = time.monotonic()
    dt = min(0.1, max(0.0, now - self._last_t))
    self._last_t = now

    self._refresh_params(now)
    hs.zones.clear()
    hs.top_boxes.clear()
    hs.panels.clear()

    snap = self._read_snapshot(dt)
    clock_txt = self._update_clock()

    # Extreme-left vertical EV power / regen bar (consistent across all 3 modes)
    self._draw_power_regen_bar(rect, snap["accel"], quiet=(self._mode == TESLA_MODE_QUIET), light=self._is_light_ui)

    if self._mode == TESLA_MODE_QUIET:
      self._draw_quiet_layout(rect, snap, clock_txt)
    elif self._mode == TESLA_MODE_AR:
      self._draw_left_cluster_ab(rect, snap, is_ar=True)
      self._draw_right_ar(rect, snap, clock_txt)
    else:
      self._draw_left_cluster_ab(rect, snap, is_ar=False)
      self._draw_right_card_3d(rect, snap, clock_txt)
      self._draw_3d_lead_tag(rect)

    # Global critical safety overlays (blindspot glow, blinkers, SoftHold, top-exit, low TPMS)
    self._draw_safety_overlays(rect, snap, now)

    # Debug plot overlay if explicitly enabled by user param ShowPlotMode > 0
    if self._show_plot_mode > 0 and self._font_display is not None:
      if self._plot_renderer is None:
        try:
          from openpilot.selfdrive.ui.onroad.hud_renderer import PlotRenderer
          self._plot_renderer = PlotRenderer()
        except Exception:
          self._plot_renderer = False
      if self._plot_renderer:
        self._plot_renderer.draw(rect, self._font_display, self._show_plot_mode)

    # Bottom 3-mode swipe dock
    self._draw_mode_dock(rect, now)

    # Subtle 0.22s ease-out fade shimmer on mode switch
    switch_dt = now - self._mode_switch_t
    if 0.0 <= switch_dt < 0.22:
      fade_a = int(68 * (1.0 - switch_dt / 0.22) ** 2)
      if fade_a > 1:
        shimmer = hs.rgba(235, 240, 248, fade_a) if self._is_light_ui else hs.rgba(4, 6, 12, fade_a)
        rl.draw_rectangle_rec(rect, shimmer)
