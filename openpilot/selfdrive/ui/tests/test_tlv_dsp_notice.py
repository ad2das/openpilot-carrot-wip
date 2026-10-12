"""Tests for the parked DSP probe HUD capsule (ui/onroad/tlv_dsp_notice.py).

Loaded with stubbed pyray/hud_style/ui_state like test_carrot_hud_renderer.py.
Covers the visibility gates (no notice, offroad, stale, hold expiry) and the
arming countdown string; the module itself is not modified.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTICE_PATH = Path(__file__).parents[1] / "onroad" / "tlv_dsp_notice.py"


class FakeTime:
  def __init__(self, start: float = 1000.0) -> None:
    self.now = float(start)

  def monotonic(self) -> float:
    return self.now

  def advance(self, dt: float) -> None:
    self.now += dt


class FakeType:
  def width(self, text, size, weight):
    return float(len(text) * size * 0.5)

  def draw_mid(self, text, x, mid, size, color, weight):
    return float(len(text) * size * 0.5)


@pytest.fixture
def notice_module(monkeypatch):
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

  fake_time = FakeTime()
  fake_ui_state = SimpleNamespace(params_memory={}, started=True)

  hs = SimpleNamespace(
    rgba=lambda r, g, b, a: Color(r, g, b, a),
    with_alpha=lambda color, alpha: Color(color.r, color.g, color.b, alpha),
    NAV=Color(0, 80, 255, 255),
    LIVE_GREEN=Color(0, 255, 0, 255),
    PROMPT_AMBER=Color(255, 176, 0, 255),
    TEXT=Color(255, 255, 255, 255),
    TEXT_3=Color(160, 160, 160, 255),
    BOLD=1,
    SEMI=2,
    zones={},
    top_boxes=[],
    chip=lambda *args, **kwargs: None,
    card=lambda *args, **kwargs: None,
    dot=lambda *args, **kwargs: None,
    Type=FakeType,
  )
  stubs = {
    "pyray": SimpleNamespace(Rectangle=Rectangle),
    "openpilot.selfdrive.ui.onroad.hud_style": hs,
    "openpilot.selfdrive.ui.ui_state": SimpleNamespace(ui_state=fake_ui_state),
  }
  for name, stub in stubs.items():
    monkeypatch.setitem(sys.modules, name, stub)

  module_name = f"_test_tlv_dsp_notice_{id(fake_ui_state)}"
  spec = importlib.util.spec_from_file_location(module_name, NOTICE_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  monkeypatch.setitem(sys.modules, module_name, module)
  spec.loader.exec_module(module)
  monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=fake_time.monotonic))
  return module, fake_ui_state, fake_time, Rectangle


def _set_notice(ui_state, key, payload):
  ui_state.params_memory = {"CarrotTlvDspNotice": payload}


def test_draw_false_without_notice(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  notice = module.TlvDspNotice()
  assert notice.draw(Rectangle(0, 0, 2160, 1080)) is False


def test_draw_false_when_not_started(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  ui_state.started = False
  _set_notice(ui_state, module.NOTICE_KEY, {"state": "running", "ts": fake_time.now, "step": 1, "steps": 3})
  notice = module.TlvDspNotice()
  assert notice.draw(Rectangle(0, 0, 2160, 1080)) is False


def test_draw_false_for_unknown_state(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  _set_notice(ui_state, module.NOTICE_KEY, {"state": "bogus", "ts": fake_time.now})
  notice = module.TlvDspNotice()
  assert notice.draw(Rectangle(0, 0, 2160, 1080)) is False


def test_draw_false_for_stale_running(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  _set_notice(ui_state, module.NOTICE_KEY,
              {"state": "running", "ts": fake_time.now - 4.0, "step": 1, "steps": 3})
  notice = module.TlvDspNotice()
  assert notice.draw(Rectangle(0, 0, 2160, 1080)) is False


def test_draw_true_for_fresh_running(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  _set_notice(ui_state, module.NOTICE_KEY,
              {"state": "running", "ts": fake_time.now - 0.5, "step": 2, "steps": 3})
  notice = module.TlvDspNotice()
  assert notice.draw(Rectangle(0, 0, 2160, 1080)) is True


def test_draw_false_for_done_older_than_hold(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  notice = module.TlvDspNotice()
  _set_notice(ui_state, module.NOTICE_KEY, {"state": "done", "ts": fake_time.now - 1.0})
  assert notice.draw(Rectangle(0, 0, 2160, 1080)) is True

  stale = module.TlvDspNotice()
  fake_time.advance(10.0)
  _set_notice(ui_state, module.NOTICE_KEY, {"state": "done", "ts": fake_time.now - 6.5})
  assert stale.draw(Rectangle(0, 0, 2160, 1080)) is False


def test_arming_text_countdown_string(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  notice = module.TlvDspNotice()
  title, detail = notice._text({"deadline": fake_time.now + 3.0}, "arming", fake_time.now)
  assert title == "곧 신호등 DSP 시험 시작"
  assert detail == "P 유지 3초"

  title, detail = notice._text({"deadline": fake_time.now + 0.2}, "arming", fake_time.now)
  assert detail == "P 유지 1초"


def test_running_text_reports_step(notice_module):
  module, ui_state, fake_time, Rectangle = notice_module
  notice = module.TlvDspNotice()
  title, detail = notice._text({"step": 2, "steps": 3}, "running", fake_time.now)
  assert title == "신호등 DSP 시험 중"
  assert detail == "2/3 단계"
