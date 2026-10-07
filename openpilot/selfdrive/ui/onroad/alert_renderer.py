import time
import pyray as rl
from dataclasses import dataclass
from openpilot.cereal import messaging, log
from openpilot.selfdrive.ui.onroad import hud_style as hs
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.hardware import TICI
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget

AlertSize = log.SelfdriveState.AlertSize
AlertStatus = log.SelfdriveState.AlertStatus

ALERT_MARGIN = 40
ALERT_BORDER_RADIUS = 40
# Small/mid alerts are a banner in the top row, between the guidance card and the tyre card.
BANNER_MAX_W = 960
BANNER_MIN_W = 640
BANNER_GAP = 30
BANNER_TOP = 32
BANNER_TITLE = 64
BANNER_TITLE_MIN = 44
BANNER_SUB = 42

ALERT_HEIGHTS = {
  AlertSize.small: 136,
  AlertSize.mid: 196,
}

SELFDRIVE_STATE_TIMEOUT = 5  # Seconds
SELFDRIVE_UNRESPONSIVE_TIMEOUT = 10  # Seconds

# Banner fills: lit gradients in the status colour (neutral uses the card body), and their ink.
ALERT_FILLS = {
  AlertStatus.normal: hs.CARD_BODY,
  AlertStatus.userPrompt: hs.TILE_AMBER,
  AlertStatus.critical: hs.TILE_RED,
}
ALERT_TEXT = {
  AlertStatus.normal: hs.TEXT,
  AlertStatus.userPrompt: hs.rgba(26, 18, 4),
  AlertStatus.critical: hs.TEXT,
}
# The full-screen variant keeps deep fills so its large white labels stay readable.
ALERT_FULL_COLORS = {
  AlertStatus.normal: (rl.Color(20, 24, 31, 236), rl.Color(8, 10, 14, 242)),
  AlertStatus.userPrompt: (rl.Color(120, 76, 8, 246), rl.Color(70, 42, 4, 250)),
  AlertStatus.critical: (rl.Color(160, 20, 28, 246), rl.Color(84, 8, 12, 250)),
}
FULL_TEXT = hs.TEXT


@dataclass
class Alert:
  text1: str = ""
  text2: str = ""
  size: int = 0
  status: int = 0


# Pre-defined alert instances
ALERT_STARTUP_PENDING = Alert(
  text1=tr("openpilot Unavailable"),
  text2=tr("Waiting to start"),
  size=AlertSize.mid,
  status=AlertStatus.normal,
)

ALERT_CRITICAL_TIMEOUT = Alert(
  text1=tr("TAKE CONTROL IMMEDIATELY"),
  text2=tr("System Unresponsive"),
  size=AlertSize.full,
  status=AlertStatus.critical,
)

ALERT_CRITICAL_REBOOT = Alert(
  text1=tr("System Unresponsive"),
  text2=tr("Reboot Device"),
  size=AlertSize.mid,
  status=AlertStatus.normal,
)


class AlertRenderer(Widget):
  def __init__(self):
    super().__init__()
    self.font_regular: rl.Font = gui_app.font(FontWeight.NORMAL)
    self.font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._type = hs.Type()


  def get_alert(self, sm: messaging.SubMaster) -> Alert | None:
    """Generate the current alert based on selfdrive state."""
    ss = sm['selfdriveState']

    # Check if selfdriveState messages have stopped arriving
    recv_frame = sm.recv_frame['selfdriveState']
    if not sm.updated['selfdriveState']:
      time_since_onroad = time.monotonic() - ui_state.started_time

      # 1. Never received selfdriveState since going onroad
      waiting_for_startup = recv_frame < ui_state.started_frame
      if waiting_for_startup and time_since_onroad > 5:
        return ALERT_STARTUP_PENDING

      # 2. Lost communication with selfdriveState after receiving it
      if TICI and not waiting_for_startup:
        ss_missing = time.monotonic() - sm.recv_time['selfdriveState']
        if ss_missing > SELFDRIVE_STATE_TIMEOUT:
          if ss.enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < SELFDRIVE_UNRESPONSIVE_TIMEOUT:
            return ALERT_CRITICAL_TIMEOUT
          return ALERT_CRITICAL_REBOOT

    # No alert if size is none
    if ss.alertSize == 0:
      return None

    # Don't get old alert
    if recv_frame < ui_state.started_frame:
      return None

    # Return current alert
    return Alert(text1=ss.alertText1, text2=ss.alertText2, size=ss.alertSize.raw, status=ss.alertStatus.raw)

  def _render(self, rect: rl.Rectangle):
    alert = self.get_alert(ui_state.sm)
    if not alert:
      return

    if alert.size == AlertSize.full:
      top, bottom = ALERT_FULL_COLORS.get(alert.status, ALERT_FULL_COLORS[AlertStatus.normal])
      rl.draw_rectangle_gradient_v(int(rect.x), int(rect.y), int(rect.width), int(rect.height), top, bottom)
      text_rect = rl.Rectangle(rect.x + 60, rect.y + 60, rect.width - 120, rect.height - 120)
      self._draw_full_text(text_rect, alert, FULL_TEXT)
      return

    banner = self._get_alert_rect(rect, alert.size)
    fill = ALERT_FILLS.get(alert.status, ALERT_FILLS[AlertStatus.normal])
    if alert.status == AlertStatus.normal:
      hs.glass_card(banner.x, banner.y, banner.width, banner.height, ALERT_BORDER_RADIUS)
    else:
      hs.soft_shadow(banner.x, banner.y, banner.width, banner.height, ALERT_BORDER_RADIUS)
      hs.tile(banner.x, banner.y, banner.width, banner.height, ALERT_BORDER_RADIUS, fill, 90)
    hs.top_boxes.append((banner.x, banner.y, banner.width, banner.height))
    self._draw_banner_text(banner, alert)

  def _get_alert_rect(self, rect: rl.Rectangle, size: int) -> rl.Rectangle:
    if size == AlertSize.full:
      return rect

    h = ALERT_HEIGHTS.get(size, ALERT_HEIGHTS[AlertSize.mid])
    left = rect.x + ALERT_MARGIN
    right = rect.x + rect.width - ALERT_MARGIN
    guide = hs.zones.get("guide")
    tpms = hs.zones.get("tpms")
    if guide is not None:
      left = guide[0] + guide[2] + BANNER_GAP
    if tpms is not None:
      right = tpms[0] - BANNER_GAP
    w = min(BANNER_MAX_W, right - left)
    if w < BANNER_MIN_W:
      # Not enough room between the cards: take the width and let the banner cover them.
      w = min(BANNER_MAX_W, rect.width - 2 * ALERT_MARGIN)
      left, right = rect.x + (rect.width - w) / 2, rect.x + (rect.width + w) / 2
    # Centred on the screen when that fits between the cards, otherwise centred in the gap.
    x = rect.x + (rect.width - w) / 2
    x = min(max(x, left), right - w)
    return rl.Rectangle(x, rect.y + BANNER_TOP, w, h)

  def _draw_banner_text(self, rect: rl.Rectangle, alert: Alert) -> None:
    ink = ALERT_TEXT.get(alert.status, hs.TEXT)
    max_w = rect.width - 96
    cx = rect.x + rect.width / 2
    title = alert.text1 or alert.text2
    size = float(BANNER_TITLE)
    while size > BANNER_TITLE_MIN and self._type.width(title, size, hs.BOLD) > max_w:
      size -= 4
    title = self._type.ellipsize(title, size, max_w, hs.BOLD)
    if alert.size == AlertSize.small or not (alert.text1 and alert.text2):
      self._type.draw_mid(title, cx, rect.y + rect.height / 2, size, ink, hs.BOLD, align=0.5)
      return
    sub = self._type.ellipsize(alert.text2, BANNER_SUB, max_w, hs.SEMI)
    self._type.draw_mid(title, cx, rect.y + 74, size, ink, hs.BOLD, align=0.5)
    self._type.draw_mid(sub, cx, rect.y + 140, BANNER_SUB, hs.with_alpha(ink, 196), hs.SEMI, align=0.5)

  def _wrap(self, text: str, size: float, weight: FontWeight, max_width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split('\n'):
      line = ""
      for word in paragraph.split():
        candidate = f"{line} {word}" if line else word
        if line and self._type.width(candidate, size, weight) > max_width:
          lines.append(line)
          line = word
        else:
          line = candidate
      if line:
        lines.append(line)
    return lines

  def _draw_full_text(self, rect: rl.Rectangle, alert: Alert, fg: rl.Color) -> None:
    # One centred block in the HUD typeface; the title shrinks until it fits on two lines.
    max_width = rect.width - 2 * 160
    title_size = 150.0
    lines = self._wrap(alert.text1, title_size, hs.BOLD, max_width)
    while len(lines) > 2 and title_size > 96:
      title_size -= 18
      lines = self._wrap(alert.text1, title_size, hs.BOLD, max_width)
    sub_size = 76.0
    sub_lines = self._wrap(alert.text2, sub_size, hs.MEDIUM, max_width) if alert.text2 else []

    title_step = title_size * 1.12
    sub_step = sub_size * 1.3
    gap = 56.0 if sub_lines else 0.0
    top = rect.y + (rect.height - (len(lines) * title_step + gap + len(sub_lines) * sub_step)) / 2

    cx = rect.x + rect.width / 2
    for line in lines:
      self._type.draw_mid(line, cx, top + title_step / 2, title_size, fg, hs.BOLD, align=0.5)
      top += title_step
    top += gap
    sub_color = hs.with_alpha(fg, 215)
    for line in sub_lines:
      self._type.draw_mid(line, cx, top + sub_step / 2, sub_size, sub_color, hs.MEDIUM, align=0.5)
      top += sub_step
