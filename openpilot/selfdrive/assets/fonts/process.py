#!/usr/bin/env python3
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pyray as rl

FONT_DIR = Path(__file__).resolve().parent
SELFDRIVE_DIR = FONT_DIR.parents[1]
TRANSLATIONS_DIR = SELFDRIVE_DIR / "ui" / "translations"
LANGUAGES_FILE = TRANSLATIONS_DIR / "languages.json"

GLYPH_PADDING = 6
EXTRA_CHARS = "–‑✓×°§•X⚙✕◀▶✔⌫⇧␣○●↳çêüñ–‑✓×°§•€£¥"
UNIFONT_LANGUAGES = {"th", "zh-CHT", "zh-CHS", "ko", "ja"}
# Extra 48 px Hangul atlases generated from a source font: atlas stem -> source stem.
KR_ATLASES = {"PretendardKR-Bold": "Pretendard-Bold"}
# Hangul optical size match (system/ui/lib/application.py KR_ATLAS_SCALE / KR_ATLAS_RAISE):
# glyphs are rasterised KR_SCALE larger than the nominal size and raised by KR_RAISE of it.
KR_SCALE = 1.25
KR_RAISE = 0.125
# Sources kept for other renderers (USB cluster) that the device UI no longer loads.
SKIP_FONTS = {"KaiGenGothicKR-Bold"}

def _languages():
  if not LANGUAGES_FILE.exists():
    return {}
  with LANGUAGES_FILE.open(encoding="utf-8") as f:
    return json.load(f)


def _char_sets():
  base = set(map(chr, range(32, 127))) | set(EXTRA_CHARS)
  hangul = set(map(chr, range(0xAC00, 0xD7A4))) | set(map(chr, range(0x1100, 0x1200))) | set(map(chr, range(0x3130, 0x3190)))
  unifont = base | hangul
  korean = base | hangul
  for language, code in _languages().items():
    unifont.update(language)
    po_path = TRANSLATIONS_DIR / f"app_{code}.po"
    try:
      chars = set(po_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      continue
    (unifont if code in UNIFONT_LANGUAGES else base).update(chars)
    if code == "ko":
      korean.update(chars)

  return tuple(sorted(ord(c) for c in base)), tuple(sorted(ord(c) for c in unifont)), tuple(sorted(ord(c) for c in korean))


def _glyph_metrics(glyphs, rects, glyph_count: int):
  entries = []
  min_offset_y, max_extent = None, 0
  for idx in range(glyph_count):
    glyph = glyphs[idx]
    rect = rects[idx]
    width = int(round(rect.width))
    height = int(round(rect.height))
    offset_y = int(round(glyph.offsetY))
    min_offset_y = offset_y if min_offset_y is None else min(min_offset_y, offset_y)
    max_extent = max(max_extent, offset_y + height)
    entries.append({
      "id": glyph.value,
      "x": int(round(rect.x)),
      "y": int(round(rect.y)),
      "width": width,
      "height": height,
      "xoffset": int(round(glyph.offsetX)),
      "yoffset": offset_y,
      "xadvance": int(round(glyph.advanceX)),
    })

  if min_offset_y is None:
    raise RuntimeError("No glyphs were generated")

  line_height = int(round(max_extent - min_offset_y))
  base = int(round(max_extent))
  return entries, line_height, base


def _shelf_atlas(glyphs, glyph_count: int, padding: int):
  """Shelf-packs the glyph bitmaps by their real sizes into an 8192-wide power-of-two atlas.

  raylib's GenImageFontAtlas reserves a full em per glyph, which doubles the Hangul atlas once
  it is rasterised KR_SCALE larger (8192x8192 instead of 8192x4096)."""
  width = 8192
  sizes = [(int(glyphs[i].image.width), int(glyphs[i].image.height)) for i in range(glyph_count)]
  spots = [(0, 0)] * glyph_count
  x = y = shelf = 0
  for i in sorted(range(glyph_count), key=lambda i: -sizes[i][1]):
    w, h = sizes[i]
    if x + w + 2 * padding > width:
      x, y, shelf = 0, y + shelf, 0
    spots[i] = (x + padding, y + padding)
    x += w + 2 * padding
    shelf = max(shelf, h + 2 * padding)
  height = 256
  while height < y + shelf:
    height *= 2
  alpha = np.zeros((height, width), dtype=np.uint8)
  for i in range(glyph_count):
    w, h = sizes[i]
    data = glyphs[i].image.data
    if w > 0 and h > 0 and data != rl.ffi.NULL:
      gx, gy = spots[i]
      alpha[gy:gy + h, gx:gx + w] = np.frombuffer(rl.ffi.buffer(data, w * h), dtype=np.uint8).reshape(h, w)
  pixels = np.empty((height, width, 2), dtype=np.uint8)
  pixels[..., 0] = 255
  pixels[..., 1] = alpha
  image = rl.gen_image_color(width, height, rl.Color(0, 0, 0, 0))
  rl.image_format(image, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA)
  rl.ffi.memmove(image.data, pixels.tobytes(), pixels.nbytes)
  rects = [SimpleNamespace(x=spots[i][0], y=spots[i][1], width=sizes[i][0], height=sizes[i][1]) for i in range(glyph_count)]
  return image, rects


def _write_bmfont(path: Path, font_size: int, face: str, atlas_name: str, line_height: int, base: int, atlas_size, entries):
  # TODO: why doesn't raylib calculate these metrics correctly?
  if line_height != font_size:
    print("using font size for line height", atlas_name)
    line_height = font_size
  lines = [
    f"info face=\"{face}\" size=-{font_size} bold=0 italic=0 charset=\"\" unicode=1 stretchH=100 smooth=0 aa=1 padding=0,0,0,0 spacing=0,0 outline=0",
    f"common lineHeight={line_height} base={base} scaleW={atlas_size[0]} scaleH={atlas_size[1]} pages=1 packed=0 alphaChnl=0 redChnl=4 greenChnl=4 blueChnl=4",
    f"page id=0 file=\"{atlas_name}\"",
    f"chars count={len(entries)}",
  ]
  for entry in entries:
    lines.append(
      ("char id={id:<4} x={x:<5} y={y:<5} width={width:<5} height={height:<5} " +
       "xoffset={xoffset:<5} yoffset={yoffset:<5} xadvance={xadvance:<5} page=0  chnl=15").format(**entry)
    )
  path.write_text("\n".join(lines) + "\n")


def _process_font(font_path: Path, codepoints: tuple[int, ...], stem: str | None = None):
  stem = stem or font_path.stem
  print(f"Processing {font_path.name} -> {stem}...")

  #font_size = {
  #  "unifont.otf": 16,  # unifont is only 16x8 or 16x16 pixels per glyph
  #}.get(font_path.name, 200)
  font_size = 200
  padding = GLYPH_PADDING

  if font_path.name == "unifont.otf":
    font_size = 16
    padding = GLYPH_PADDING

  raster_size, raise_px = font_size, 0
  if stem in KR_ATLASES:
    font_size = 48
    padding = 2
    raster_size, raise_px = round(font_size * KR_SCALE), round(font_size * KR_RAISE)

  data = font_path.read_bytes()
  file_buf = rl.ffi.new("unsigned char[]", data)
  cp_buffer = rl.ffi.new("int[]", codepoints)
  cp_ptr = rl.ffi.cast("int *", cp_buffer)
  glyph_count = rl.ffi.new("int *", len(codepoints))
  glyphs = rl.load_font_data(
    rl.ffi.cast("unsigned char *", file_buf), len(data), raster_size, cp_ptr, len(codepoints),
    rl.FontType.FONT_DEFAULT, glyph_count
  )
  if glyphs == rl.ffi.NULL:
    raise RuntimeError("raylib failed to load font data")
  for idx in range(glyph_count[0]):
    glyphs[idx].offsetY -= raise_px

  if stem in KR_ATLASES:
    image, rects = _shelf_atlas(glyphs, glyph_count[0], padding)
  else:
    rects_ptr = rl.ffi.new("Rectangle **")
    image = rl.gen_image_font_atlas(glyphs, rects_ptr, glyph_count[0], font_size, padding, 0)
    rects = rects_ptr[0]
  if image.width == 0 or image.height == 0:
    raise RuntimeError("raylib returned an empty atlas")

  atlas_name = f"{stem}.png"
  atlas_path = FONT_DIR / atlas_name
  entries, line_height, base = _glyph_metrics(glyphs, rects, glyph_count[0])

  if not rl.export_image(image, atlas_path.as_posix()):
    raise RuntimeError("Failed to export atlas image")

  _write_bmfont(FONT_DIR / f"{stem}.fnt", font_size, stem, atlas_name, line_height, base, (image.width, image.height), entries)


def main():
  base_cp, unifont_cp, korean_cp = _char_sets()
  fonts = sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf"))
  for font in fonts:
    if "emoji" in font.name.lower() or font.stem in SKIP_FONTS:
      continue
    glyphs = unifont_cp if font.stem.lower().startswith("unifont") else base_cp
    _process_font(font, glyphs)
  for stem, source in KR_ATLASES.items():
    _process_font(FONT_DIR / f"{source}.ttf", korean_cp, stem)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
