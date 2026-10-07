"""Visual language of the C3/C3X onroad HUD: glass cards, typography, colour and glyphs.

Numbers and Latin labels are drawn from the 200 px Inter atlases directly. font_fallback()
swaps every face for the 48 px Korean atlas under the Korean locale, which blurs large
numerals, so only text that contains Hangul (or other non-Latin script) uses that atlas.
"""
import math

import numpy as np
import pyray as rl

from openpilot.system.ui.lib import native_text
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.shader_polygon import Gradient, draw_polygon, draw_polygon_solid


def rgba(r: int, g: int, b: int, a: int = 255) -> rl.Color:
  return rl.Color(r, g, b, a)


def with_alpha(color: rl.Color, alpha: int) -> rl.Color:
  return rl.Color(color.r, color.g, color.b, alpha)


# Surfaces
GLASS = rgba(12, 15, 20, 196)
GLASS_STRONG = rgba(12, 15, 20, 226)
HAIRLINE = rgba(255, 255, 255, 30)
DIVIDER = rgba(255, 255, 255, 22)
SHADOW = rgba(0, 0, 0, 120)
# Corner shade that lets type float directly on the camera image.
VIGNETTE = rgba(4, 6, 10, 228)
ALARM_GLOW = rgba(150, 12, 20, 200)

# Text
TEXT = rgba(255, 255, 255)
TEXT_2 = rgba(255, 255, 255, 235)
TEXT_3 = rgba(255, 255, 255, 190)
INK = rgba(14, 16, 20)

# Signals
GREEN = rgba(52, 211, 110)
AMBER = rgba(255, 179, 36)
RED = rgba(255, 72, 64)
SOFT_RED = rgba(255, 112, 102)
BLUE = rgba(26, 136, 255)
LAVENDER = rgba(196, 128, 255)
CARROT = rgba(255, 128, 32)
SIGN_RED = rgba(222, 34, 40)
CAMERA_FILL = rgba(196, 24, 32, 196)

BOLD = FontWeight.BOLD
SEMI = FontWeight.SEMI_BOLD
MEDIUM = FontWeight.MEDIUM

# Baseline as a fraction of the draw size, measured from the generated atlases:
# Inter caps span 0.195..0.800, KaiGenGothicKR Latin/digits sit on 0.896.
_INTER_BASELINE = 0.80
_HANGUL_BASELINE = 0.896
INTER_CAP = 0.605
# Hangul glyphs fill 0.02..0.98 of the draw size, so their centre is 0.396 above the baseline.
_HANGUL_MIDDLE = _HANGUL_BASELINE - 0.5
_INTER_EXTRA = frozenset("°–•×")
_WIDTH_CACHE_LIMIT = 512


def _latin(text: str) -> bool:
  return text.isascii() or all(c.isascii() or c in _INTER_EXTRA for c in text)


class Type:
  """Baseline-anchored text drawing with per-script face selection."""

  def __init__(self):
    self._fonts = {weight: gui_app.font(weight) for weight in (BOLD, SEMI, MEDIUM)}
    self._hangul = gui_app.font(FontWeight.DISPLAY) if gui_app.has_font(FontWeight.DISPLAY) else self._fonts[BOLD]
    self._widths: dict[tuple, float] = {}
    self._position = rl.Vector2(0.0, 0.0)

  def _face(self, text: str, weight: FontWeight):
    if _latin(text):
      return self._fonts[weight], _INTER_BASELINE
    return self._hangul, _HANGUL_BASELINE

  def width(self, text: str, size: float, weight: FontWeight = BOLD, spacing: float = 0.0) -> float:
    if not text:
      return 0.0
    key = (text, size, weight, spacing)
    width = self._widths.get(key)
    if width is None:
      font, _ = self._face(text, weight)
      width = float(rl.measure_text_ex(font, text, size, spacing).x)  # noqa: TID251  # unscaled atlas metrics, cached above
      if len(self._widths) >= _WIDTH_CACHE_LIMIT:
        self._widths.clear()
      self._widths[key] = width
    return width

  def draw(self, text: str, x: float, baseline: float, size: float, color: rl.Color,
           weight: FontWeight = BOLD, align: float = 0.0, shadow: bool = False, spacing: float = 0.0) -> float:
    """Draws `text` with its baseline at `baseline`; align 0 = left, 0.5 = centre, 1 = right."""
    if not text:
      return 0.0
    font, base = self._face(text, weight)
    width = self.width(text, size, weight, spacing)
    left = x - width * align
    top = baseline - base * size
    if shadow:
      self._raw(font, text, left, top + max(2.0, size * 0.03), size, rl.Color(0, 0, 0, min(170, color.a)), spacing)
    self._raw(font, text, left, top, size, color, spacing)
    return width

  def draw_mid(self, text: str, x: float, middle: float, size: float, color: rl.Color,
               weight: FontWeight = BOLD, align: float = 0.0, shadow: bool = False, spacing: float = 0.0) -> float:
    """Draws `text` optically centred on `middle`: Latin on its cap height, Hangul on its em box."""
    offset = INTER_CAP / 2.0 if _latin(text) else _HANGUL_MIDDLE
    return self.draw(text, x, middle + offset * size, size, color, weight, align, shadow, spacing)

  def _raw(self, font, text: str, x: float, y: float, size: float, color: rl.Color, spacing: float = 0.0) -> None:
    position = self._position
    position.x, position.y = float(x), float(y)
    if native_text.try_plain_text(rl, font, text, position, size, spacing, color):
      return
    getattr(rl, "_orig_draw_text_ex", rl.draw_text_ex)(font, text, position, size, spacing, color)

  def ellipsize(self, text: str, size: float, max_width: float, weight: FontWeight = BOLD) -> str:
    if self.width(text, size, weight) <= max_width:
      return text
    lo, hi = 0, len(text)
    while lo < hi:
      mid = (lo + hi + 1) // 2
      if self.width(text[:mid].rstrip() + "…", size, weight) <= max_width:
        lo = mid
      else:
        hi = mid - 1
    return text[:lo].rstrip() + "…" if lo > 0 else ""


_rect = rl.Rectangle(0.0, 0.0, 0.0, 0.0)

# HUD panels drawn last frame (x, y, w, h). The model layer draws first, so it reads these to keep
# its floating labels off the panels instead of being painted over.
panels: list[tuple[float, float, float, float]] = []


def clear_of(boxes, x: float, y: float, w: float, h: float, gap: float = 12.0) -> float:
  """Return y, lifted above any (x, y, w, h) box that the centred box (x, y, w, h) would land on."""
  for bx, by, bw, bh in sorted(boxes, key=lambda b: -b[1]):
    if x + w / 2 > bx and x - w / 2 < bx + bw and y + h / 2 > by and y - h / 2 < by + bh:
      y = min(y, by - h / 2 - gap)
  return y


# Top-row cards and the clock block (previous frame); floating labels drop below these.
top_boxes: list[tuple[float, float, float, float]] = []


def _hits(boxes, x: float, y: float, w: float, h: float, gap: float):
  for box in boxes:
    bx, by, bw, bh = box
    if x + w / 2 + gap > bx and x - w / 2 - gap < bx + bw and y + h / 2 + gap > by and y - h / 2 - gap < by + bh:
      return box
  return None


def drop_below(x: float, y: float, w: float, h: float, claimed, gap: float = 12.0) -> float:
  """Centre y, moved down past every HUD card or claimed label the centred box (x, y, w, h) would overlap."""
  boxes = list(claimed) + top_boxes + panels
  for _ in range(len(boxes) + 1):
    hit = _hits(boxes, x, y, w, h, gap)
    if hit is None:
      break
    y = hit[1] + hit[3] + gap + h / 2
  return y


def place_label(x: float, y: float, w: float, h: float, claimed, gap: float = 10.0):
  """Centre for a floating label near (x, y) that overlaps no HUD card or claimed label, else None.

  Tries the preferred spot, then directly above the obstacle, then beside it on either side."""
  boxes = list(claimed) + top_boxes + panels
  hit = _hits(boxes, x, y, w, h, gap)
  if hit is None:
    return x, y
  bx, by, bw, bh = hit
  for cx, cy in ((x, by - h / 2 - gap), (bx - w / 2 - gap, y), (bx + bw + w / 2 + gap, y)):
    if cy - h / 2 > 0 and _hits(boxes, cx, cy, w, h, gap) is None:
      return cx, cy
  return None


def clear_of_panels(x: float, y: float, w: float, h: float) -> float:
  """Return y, moved off every HUD card the centred box (x, y, w, h) would land on."""
  for bx, by, bw, bh in top_boxes:
    if x + w / 2 > bx and x - w / 2 < bx + bw and y - h / 2 < by + bh + 12:
      y = max(y, by + bh + 12 + h / 2)
  return clear_of(panels, x, y, w, h)


def card(x: float, y: float, w: float, h: float, fill: rl.Color = GLASS, radius: float = 28.0,
         edge: "rl.Color | None" = HAIRLINE, edge_width: float = 2.0) -> None:
  if w <= 0 or h <= 0:
    return
  _rect.x, _rect.y, _rect.width, _rect.height = float(x), float(y), float(w), float(h)
  roundness = min(1.0, 2.0 * radius / min(w, h))
  rl.draw_rectangle_rounded(_rect, roundness, 12, fill)
  if edge is not None and edge_width > 0:
    rl.draw_rectangle_rounded_lines_ex(_rect, roundness, 12, edge_width, edge)


def pill(x: float, y: float, w: float, h: float, fill: rl.Color, edge: "rl.Color | None" = None) -> None:
  card(x, y, w, h, fill, h / 2.0, edge, 2.0)


def dot(x: float, y: float, radius: float, color: rl.Color) -> None:
  rl.draw_circle(int(round(x)), int(round(y)), radius, color)


def arc(cx: float, cy: float, radius: float, width: float, start: float, end: float, color: rl.Color,
        round_caps: bool = True) -> None:
  """Stroked arc centred on `radius`; angles in degrees, 0 = +x, 90 = down."""
  if end - start <= 0.05:
    return
  center = rl.Vector2(float(cx), float(cy))
  segments = max(6, int((end - start) / 4))
  rl.draw_ring(center, radius - width / 2, radius + width / 2, start, end, segments, color)
  if round_caps:
    for angle in (start, end):
      rad = math.radians(angle)
      rl.draw_circle_v(rl.Vector2(cx + math.cos(rad) * radius, cy + math.sin(rad) * radius), width / 2, color)


def polar(cx: float, cy: float, radius: float, angle: float) -> tuple[float, float]:
  rad = math.radians(angle)
  return cx + math.cos(rad) * radius, cy + math.sin(rad) * radius


def vignette(cx: float, cy: float, radius: float, color: rl.Color) -> None:
  """Radial shade fading from `color` at (cx, cy) to transparent; callers rely on the content scissor."""
  rl.draw_circle_gradient(rl.Vector2(float(cx), float(cy)), float(radius), color, rl.Color(color.r, color.g, color.b, 0))


# ---- instrument cards -------------------------------------------------------------------------
# Cards are lit from above: a vertical body gradient with a faint top sheen, a hairline edge that
# fades downwards and a soft drop shadow. Tiles are opaque gradients in a signal colour.

# (top, sheen end, bottom) body colours, alpha included.
CARD_BODY = (rgba(36, 40, 48, 234), rgba(19, 22, 29, 236), rgba(9, 11, 15, 240))
CARD_WARN = (rgba(87, 32, 35, 240), rgba(57, 12, 14, 241), rgba(34, 8, 10, 242))
CARD_EDGE = (rgba(255, 255, 255, 64), rgba(255, 255, 255, 16), rgba(255, 255, 255, 10))
TILE_RED = (rgba(255, 107, 97), rgba(200, 30, 30))
TILE_AMBER = (rgba(255, 179, 42), rgba(240, 138, 0))
CHIP_FILL = rgba(13, 16, 21, 210)
NAV = rgba(43, 155, 255)
LIVE_GREEN = rgba(52, 199, 89)
WARN_RED = rgba(255, 69, 58)
PROMPT_AMBER = rgba(255, 159, 10)
LABEL_AMBER = rgba(255, 179, 64)
SIGN_RING = rgba(229, 38, 45)
SIGN_FACE = rgba(246, 247, 249)
SIGN_INK = rgba(17, 17, 17)

# Named HUD regions for this frame, so later layers (alerts) can sit between them.
zones: dict[str, tuple[float, float, float, float]] = {}

_RIBBONS: dict[tuple, np.ndarray] = {}
_ORIGIN = rl.Rectangle(0.0, 0.0, 1.0, 1.0)


def _front_facing(pts: np.ndarray) -> np.ndarray:
  """Orders a two-chain ribbon so its triangle strip winds the way raylib does not cull."""
  a, b, c = pts[0], pts[-1], pts[1]
  if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0:
    return np.ascontiguousarray(pts[::-1])
  return pts


def _rounded_ribbon(x: float, y: float, w: float, h: float, r: float, segments: int = 7) -> np.ndarray:
  """Rounded rectangle as a two-chain ribbon (left chain top-down, right chain bottom-up)."""
  key = (round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(r, 1))
  pts = _RIBBONS.get(key)
  if pts is not None:
    return pts
  r = max(0.0, min(r, w / 2.0, h / 2.0))
  theta = np.linspace(0.0, math.pi / 2.0, segments + 1)
  inset = r - r * np.sin(theta)
  top = y + r - r * np.cos(theta)
  bottom = (y + h - r + r * np.cos(theta))[::-1]
  ys = np.concatenate([top, bottom])
  insets = np.concatenate([inset, inset[::-1]])
  left = np.stack([x + insets, ys], axis=1)
  right = np.stack([x + w - insets, ys], axis=1)
  pts = _front_facing(np.ascontiguousarray(np.concatenate([left, right[::-1]]), dtype=np.float32))
  if len(_RIBBONS) > 96:
    _RIBBONS.clear()
  _RIBBONS[key] = pts
  return pts


def vertical_gradient(points: np.ndarray, y0: float, y1: float, colors, stops=None) -> None:
  """Fills a ribbon with colours running from screen y0 (first) to y1 (last)."""
  height = float(gui_app.height)
  # The shader measures gl_FragCoord (y up) from `end` towards `start`.
  gradient = Gradient(start=(0.0, height - y1), end=(0.0, height - y0), colors=list(colors),
                      stops=list(stops) if stops is not None else [])
  draw_polygon(_ORIGIN, points, gradient=gradient)


def gradient_rect(x: float, y: float, w: float, h: float, r: float, colors, stops=None) -> None:
  if w <= 0 or h <= 0:
    return
  vertical_gradient(_rounded_ribbon(x, y, w, h, r), y, y + h, colors, stops)


def soft_shadow(x: float, y: float, w: float, h: float, r: float, dy: float = 10.0, blur: float = 18.0,
                alpha: int = 120) -> None:
  """Drop shadow from a few widening translucent layers."""
  layers = 4
  for i in range(layers, 0, -1):
    grow = blur * i / layers
    card(x - grow * 0.6, y + dy - grow * 0.3, w + grow * 1.2, h + grow * 0.9,
         rgba(0, 0, 0, int(alpha / (layers + 1))), r + grow * 0.6, None)


def glass_card(x: float, y: float, w: float, h: float, r: float = 38.0, body=CARD_BODY, shadow: bool = True,
               alpha: float = 1.0) -> None:
  if w <= 0 or h <= 0:
    return
  if shadow:
    soft_shadow(x, y, w, h, r, alpha=int(120 * alpha))
  if alpha < 1.0:
    body = tuple(with_alpha(c, int(c.a * alpha)) for c in body)
  edge = CARD_EDGE if alpha >= 1.0 else tuple(with_alpha(c, int(c.a * alpha)) for c in CARD_EDGE)
  # The edge is a 1.5 px larger body underneath: bright along the top, almost gone at the bottom.
  gradient_rect(x, y, w, h, r, edge, (0.0, 0.5, 1.0))
  gradient_rect(x + 1.5, y + 1.5, w - 3.0, h - 3.0, r - 1.5, body, (0.0, 0.45, 1.0))


def tile(x: float, y: float, w: float, h: float, r: float, colors, edge_alpha: int = 72) -> None:
  gradient_rect(x, y, w, h, r, colors)
  _rect.x, _rect.y, _rect.width, _rect.height = float(x) + 0.75, float(y) + 0.75, float(w) - 1.5, float(h) - 1.5
  rl.draw_rectangle_rounded_lines_ex(_rect, min(1.0, 2.0 * r / min(w, h)), 12, 1.5, rgba(255, 255, 255, edge_alpha))


def chip(x: float, y: float, w: float, h: float, fill: rl.Color = CHIP_FILL, ring: "rl.Color | None" = None,
         ring_width: float = 2.5, shadow: bool = True) -> None:
  """Floating capsule for labels over the road."""
  if shadow:
    soft_shadow(x, y, w, h, h / 2.0, dy=6.0, blur=12.0, alpha=100)
  card(x, y, w, h, fill, h / 2.0, None)
  if ring is not None:
    _rect.x, _rect.y, _rect.width, _rect.height = float(x) + 1.0, float(y) + 1.0, float(w) - 2.0, float(h) - 2.0
    rl.draw_rectangle_rounded_lines_ex(_rect, 1.0, 16, ring_width, ring)
  else:
    _rect.x, _rect.y, _rect.width, _rect.height = float(x) + 0.75, float(y) + 0.75, float(w) - 1.5, float(h) - 1.5
    rl.draw_rectangle_rounded_lines_ex(_rect, 1.0, 16, 1.5, rgba(255, 255, 255, 46))


def ellipse_glow(cx: float, cy: float, rx: float, ry: float, color: rl.Color) -> None:
  """Radial glow squashed into an ellipse (a light pool on the road)."""
  rl.rl_push_matrix()
  rl.rl_translatef(float(cx), float(cy), 0.0)
  rl.rl_scalef(1.0, float(ry) / float(rx), 1.0)
  rl.draw_circle_gradient(rl.Vector2(0.0, 0.0), float(rx), color, rgba(color.r, color.g, color.b, 0))
  rl.rl_pop_matrix()


def glow(cx: float, cy: float, radius: float, color: rl.Color) -> None:
  rl.draw_circle_gradient(rl.Vector2(float(cx), float(cy)), float(radius), color, rgba(color.r, color.g, color.b, 0))


def hgradient_line(x: float, y: float, w: float, h: float, color: rl.Color) -> None:
  """A line that fades in from both ends."""
  half = int(w / 2)
  clear = rgba(color.r, color.g, color.b, 0)
  rl.draw_rectangle_gradient_h(int(x), int(y), half, int(h), clear, color)
  rl.draw_rectangle_gradient_h(int(x) + half, int(y), int(w) - half, int(h), color, clear)


def curve(p0, p1, p2, width: float, color: rl.Color, steps: int = 12) -> None:
  """Quadratic Bezier stroke with round joins."""
  _stroke([p0] + _quad_points(p0, p1, p2, steps), width, color)


# ---- glyphs: unit box [-.5, .5], y down ----

def _ccw_triangle(a, b, c, color: rl.Color) -> None:
  # raylib culls clockwise triangles; on screen (y down) counter-clockwise means a negative cross product.
  cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
  if cross > 0:
    b, c = c, b
  rl.draw_triangle(rl.Vector2(*a), rl.Vector2(*b), rl.Vector2(*c), color)


def triangle(a, b, c, color: rl.Color) -> None:
  _ccw_triangle(a, b, c, color)


def _quad_points(p0, p1, p2, steps: int = 8):
  return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
           (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
          for t in (i / steps for i in range(1, steps + 1))]


def _stroke(points, width: float, color: rl.Color) -> None:
  radius = width / 2.0
  prev = None
  for x, y in points:
    if prev is not None:
      rl.draw_line_ex(rl.Vector2(*prev), rl.Vector2(x, y), width, color)
    rl.draw_circle_v(rl.Vector2(x, y), radius, color)
    prev = (x, y)


def speed_sign(cx: float, cy: float, diameter: float, value: str, type_: Type, alarm: bool = False) -> None:
  """Korean regulatory speed sign: white disc, red ring, black numerals (inverted while alarming)."""
  radius = diameter / 2.0
  ring = diameter * 0.095
  rl.draw_circle_v(rl.Vector2(cx, cy), radius, SIGN_RED)
  rl.draw_circle_v(rl.Vector2(cx, cy), radius - ring, SIGN_RED if alarm else TEXT)
  size = diameter * (0.40 if len(value) >= 3 else 0.48)
  type_.draw(value, cx, cy + size * INTER_CAP / 2.0, size, TEXT if alarm else INK, BOLD, align=0.5)



# ---- manoeuvre icons ------------------------------------------------------------------------------
# One drawn set (selfdrive/assets/images/nav, source: SVG with round joins and a softened sharp head),
# rendered as white alpha masks and tinted at draw time, so the card and the road use the same shapes.

MANEUVER_ICONS = {1: "turn_left", 2: "turn_right", 3: "keep_left", 4: "keep_right", 5: "rotary",
                  6: "toll", 7: "uturn", 8: "destination"}
# AR signs: (base = white rim + drop shadow, fill = route ramp) layers; foot = shaft end in texture fractions.
AR_FEET = {"turn_right": (0.1985, 0.9229), "turn_left": (0.7985, 0.9229), "keep_right": (0.1982, 0.9339),
           "keep_left": (0.7982, 0.9339), "uturn": (0.8127, 0.9254), "destination": (0.4969, 0.9031)}
AR_ICONS = {1: "turn_left", 2: "turn_right", 3: "keep_left", 4: "keep_right", 7: "uturn", 8: "destination"}

_TEXTURES: dict[str, rl.Texture] = {}


def _texture(name: str) -> rl.Texture:
  tex = _TEXTURES.get(name)
  if tex is None:
    # Bilinear only: the AR layers are not power-of-two sized, and GLES2 cannot mipmap those.
    tex = gui_app.texture(f"images/nav/{name}.png")
    _TEXTURES[name] = tex
  return tex


def _blit(tex: rl.Texture, x: float, y: float, w: float, h: float, color: rl.Color) -> None:
  rl.draw_texture_pro(tex, rl.Rectangle(0, 0, tex.width, tex.height), rl.Rectangle(x, y, w, h), rl.Vector2(0, 0), 0.0, color)


def maneuver_arrow(turn_info: int, cx: float, cy: float, k: float, color: rl.Color = TEXT, context: bool = True) -> bool:
  """Draws the xTurnInfo manoeuvre icon in a 140 * k square centred on (cx, cy); False when there is none."""
  name = MANEUVER_ICONS.get(turn_info)
  if name is None:
    return False
  size = 140.0 * k
  _blit(_texture(f"nav_{name}"), cx - size / 2, cy - size / 2, size, size, color)
  return True


def ar_sign(turn_info: int, foot_x: float, foot_y: float, height: float, tint: rl.Color, alpha: float) -> tuple | None:
  """AR manoeuvre sign standing with its shaft end on (foot_x, foot_y); returns its screen box."""
  name = AR_ICONS.get(turn_info)
  if name is None:
    return None
  base, fill = _texture(f"ar_{name}_base"), _texture(f"ar_{name}_fill")
  w = height * base.width / base.height
  fx, fy = AR_FEET[name]
  x, y = foot_x - fx * w, foot_y - fy * height
  a = max(0, min(255, int(255 * alpha)))
  _blit(base, x, y, w, height, rgba(255, 255, 255, a))
  _blit(fill, x, y, w, height, rgba(tint.r, tint.g, tint.b, a))
  return x, y, w, height


def regulatory_sign(cx: float, cy: float, value: str, type_: Type, k: float = 1.0, glow_alarm: bool = False) -> None:
  """Korean speed-limit disc: soft shadow, white face, red ring, dark numerals."""
  if glow_alarm:
    glow(cx, cy, 124 * k, rgba(255, 69, 58, 150))
  glow(cx, cy + 4 * k, 100 * k, rgba(0, 0, 0, 120))
  rl.draw_circle_v(rl.Vector2(cx, cy), 86 * k, SIGN_FACE)
  rl.draw_ring(rl.Vector2(cx, cy), 64 * k, 84 * k, 0.0, 360.0, 64, SIGN_RING)
  size = (62 if len(value) >= 3 else 76) * k
  type_.draw(value, cx, cy + size * INTER_CAP / 2.0, size, SIGN_INK, BOLD, align=0.5, spacing=-2 * k)


def quad(a, b, c, d, color: rl.Color) -> None:
  _ccw_triangle(a, b, c, color)
  _ccw_triangle(a, c, d, color)


def stroke_ribbon(points, half) -> np.ndarray:
  """Outline of a polyline stroke; `half` is one half-width or one per point."""
  pts = np.asarray(points, dtype=np.float32)
  d = np.gradient(pts, axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
  n = np.stack([-d[:, 1], d[:, 0]], axis=1) * np.reshape(np.asarray(half, dtype=np.float32), (-1, 1))
  return _front_facing(np.ascontiguousarray(np.concatenate([pts + n, (pts - n)[::-1]])))


def polyline(points, width: float, color: rl.Color) -> None:
  if len(points) >= 2:
    draw_polygon_solid(stroke_ribbon(points, width / 2.0), color)
