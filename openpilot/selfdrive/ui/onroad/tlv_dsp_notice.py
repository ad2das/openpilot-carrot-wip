"""Top-centre capsule for the parked tlv DSP probe (selfdrive/carrot/tlv/dsp_probe.py).

The probe publishes CarrotTlvDspNotice in params_memory:
  {"state": "arming"|"running"|"aborted"|"done"|"failed", "ts": monotonic,
   "deadline": monotonic (arming only), "step": 1..steps (running), "steps": 3}
arming/running are refreshed at 2 Hz, so a notice older than STALE_S means the probe is gone.
"""
import math
import time

import pyray as rl

from openpilot.common.impact_dashcam import read_object
from openpilot.selfdrive.ui.onroad import hud_style as hs
from openpilot.selfdrive.ui.ui_state import ui_state

NOTICE_KEY = "CarrotTlvDspNotice"
READ_INTERVAL_S = 0.25
STALE_S = 3.0
HOLD_S = {"aborted": 4.0, "done": 6.0, "failed": 6.0}

H = 92
TEXT_SIZE = 40
DETAIL_SIZE = 34
SEP = 22.0
PAD = 38.0
DOT_R = 10.0
SEG_W, SEG_H, SEG_GAP = 34.0, 10.0, 8.0

# (fill, ring/dot accent) per state: navy while it works, green when done, amber when it stops or fails.
NAVY = (hs.rgba(18, 34, 74, 236), hs.NAV)
GREEN = (hs.rgba(14, 60, 34, 236), hs.LIVE_GREEN)
AMBER = (hs.rgba(78, 46, 8, 236), hs.PROMPT_AMBER)
STYLE = {"arming": NAVY, "running": NAVY, "done": GREEN, "aborted": AMBER, "failed": AMBER}


class TlvDspNotice:
  def __init__(self):
    self._type = hs.Type()
    self._notice: dict = {}
    self._last_read = -READ_INTERVAL_S

  def _text(self, notice: dict, state: str, now: float) -> tuple[str, str]:
    # (title, detail); a hairline separates them, so no glyph outside ASCII + Hangul is needed.
    if state == "arming":
      left = max(0, math.ceil(float(notice.get("deadline", now)) - now))
      return "곧 신호등 DSP 시험 시작", f"P 유지 {left}초"
    if state == "running":
      return "신호등 DSP 시험 중", f"{int(notice.get('step', 1))}/{int(notice.get('steps', 3))} 단계"
    if state == "aborted":
      return "DSP 시험 중단", "다음 주차 때 다시 시작"
    if state == "done":
      return "DSP 시험 완료", "Carrot Web에서 결과 확인"
    return "DSP 시험 실패", "Carrot Web에서 결과 확인"

  def draw(self, rect: rl.Rectangle) -> bool:
    """Draws the capsule in the clock slot; True when it took the slot."""
    now = time.monotonic()
    if now - self._last_read >= READ_INTERVAL_S:
      self._last_read = now
      self._notice = read_object(ui_state.params_memory, NOTICE_KEY)
    notice = self._notice
    state = notice.get("state")
    if state not in STYLE or not ui_state.started:
      return False
    try:
      age = now - float(notice.get("ts", 0.0))
    except (TypeError, ValueError):
      return False
    if age < -1.0 or age > HOLD_S.get(state, STALE_S):
      return False

    t = self._type
    title, detail = self._text(notice, state, now)
    steps = max(1, min(6, int(notice.get("steps", 3)))) if state == "running" else 0
    bar_w = steps * SEG_W + (steps - 1) * SEG_GAP if steps else 0.0

    guide, tpms = hs.zones.get("guide"), hs.zones.get("tpms")
    lo = guide[0] + guide[2] + 24 if guide is not None else rect.x
    hi = tpms[0] - 24 if tpms is not None else rect.x + rect.width
    fixed = PAD * 2 + DOT_R * 2 + 18 + (bar_w + 28 if steps else 0.0)
    title_w = t.width(title, TEXT_SIZE, hs.BOLD)
    detail_w = t.width(detail, DETAIL_SIZE, hs.SEMI)
    w = fixed + title_w + 2 * SEP + 2 + detail_w
    if w > hi - lo:
      # Drop the detail before giving up; the title alone still says what is happening.
      detail, w = "", fixed + title_w
    if w > hi - lo:
      return False
    cx = min(max(rect.x + rect.width / 2, lo + w / 2), hi - w / 2)
    x, top = cx - w / 2, rect.y + 32
    mid = top + H / 2

    fill, accent = STYLE[state]
    hs.chip(x, top, w, H, fill, ring=hs.with_alpha(accent, 150), ring_width=2.0)
    x += PAD
    hs.dot(x + DOT_R, mid, DOT_R, accent)
    x += DOT_R * 2 + 18
    x += t.draw_mid(title, x, mid, TEXT_SIZE, hs.TEXT, hs.BOLD)
    if detail:
      hs.card(x + SEP, mid - 22, 2, 44, hs.rgba(255, 255, 255, 56), 1, None)
      x += 2 * SEP + 2
      x += t.draw_mid(detail, x, mid, DETAIL_SIZE, hs.TEXT_3, hs.SEMI)
    if steps:
      # One segment per probe step: finished ones solid, the current one breathing.
      step = max(1, min(steps, int(notice.get("step", 1))))
      x += 28
      pulse = int(110 + 110 * (0.5 + 0.5 * math.sin(now * 2 * math.pi / 1.4)))
      for i in range(steps):
        color = accent if i < step - 1 else hs.with_alpha(accent, pulse) if i == step - 1 else hs.rgba(255, 255, 255, 46)
        hs.card(x, mid - SEG_H / 2, SEG_W, SEG_H, color, SEG_H / 2, None)
        x += SEG_W + SEG_GAP
    hs.top_boxes.append((cx - w / 2, top, w, H))
    return True
