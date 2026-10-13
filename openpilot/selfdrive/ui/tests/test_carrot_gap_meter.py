import pytest

from openpilot.selfdrive.ui.onroad import hud_renderer


def rgba(color):
  return (color.r, color.g, color.b, color.a)


@pytest.mark.parametrize("gap, levels", [(1, 4), (2, 4), (4, 4), (2, 3), (1, 2)])
def test_gap_meter_lights_one_pill_per_gap_level_left_of_the_car(monkeypatch, gap, levels):
  cards = []
  monkeypatch.setattr(hud_renderer.hs, "card", lambda x, y, w, h, fill, radius, edge: cards.append((x, w, h, rgba(fill))))
  monkeypatch.setattr(hud_renderer.hs, "quad", lambda *args: None)

  hud_renderer.HudRenderer._draw_gap_meter(100.0, 424.0, 500.0, gap, levels)

  pills = [card for card in cards if card[2] == hud_renderer.GAP_SEG_H]
  lit, dim = rgba(hud_renderer.hs.TEXT), rgba(hud_renderer.hs.with_alpha(hud_renderer.hs.TEXT, 50))
  assert [pill[3] for pill in pills] == [lit] * gap + [dim] * (levels - gap)
  assert [pill[0] for pill in pills] == sorted(pill[0] for pill in pills)
  assert pills[0][0] == 100.0
  assert pills[-1][0] + pills[-1][1] <= 424.0 - hud_renderer.GAP_CAR_W
