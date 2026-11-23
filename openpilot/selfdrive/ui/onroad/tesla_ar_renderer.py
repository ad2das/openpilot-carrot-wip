"""Tesla FSD v15 Minimalist AR Camera Overlay (Mode 1 — AR 카메라).

Projects a single luminous electric-blue FSD trajectory ribbon, subtle active lane-lock
edges, lead-vehicle hologram lock-on arc + floating distance/speed chip, and road-surface
projected perspective directional chevrons onto the real forward camera stream using
ModelRenderer's calibrated camera transform and TlvOverlay.
"""
import math
import time
from typing import Optional

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.onroad import hud_style as hs
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state


def _sample_centerline(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
  if points.shape[0] < 2:
    return np.empty((0, 3), dtype=np.float32)
  px = points[:, 0]
  return np.stack([
    distances,
    np.interp(distances, px, points[:, 1]),
    np.interp(distances, px, points[:, 2]),
  ], axis=1).astype(np.float32)


class TeslaArRenderer:
  """Ultra-clean Tesla FSD v15 AR projection layer over the live forward camera."""

  def __init__(self, model_renderer: ModelRenderer):
    self._model = model_renderer
    self._type = hs.Type()
    self._chev_phase = 0.0
    self._last_t = time.monotonic()
    self._scrim_tex: Optional[rl.Texture] = None

  def _ensure_scrim(self) -> Optional[rl.Texture]:
    """Pre-bakes a 1D silky horizontal optical left-edge scrim (zero hard card borders)."""
    if self._scrim_tex is not None:
      return self._scrim_tex
    try:
      w = 256
      t = np.linspace(0.0, 1.0, w, dtype=np.float32)
      # Smooth optical falloff: 76% alpha at left edge -> 52% at 17% -> 18% at 30% -> 0% at 45%
      alpha = np.interp(t, [0.0, 0.17, 0.30, 0.45, 1.0], [198.0, 142.0, 48.0, 0.0, 0.0])
      pixels = np.zeros((1, w, 4), dtype=np.uint8)
      pixels[0, :, 0] = 6
      pixels[0, :, 1] = 10
      pixels[0, :, 2] = 16
      pixels[0, :, 3] = np.clip(alpha + 0.5, 0, 255).astype(np.uint8)
      img = rl.gen_image_color(w, 1, rl.Color(0, 0, 0, 0))
      rl.ffi.memmove(img.data, np.ascontiguousarray(pixels).tobytes(), pixels.nbytes)
      tex = rl.load_texture_from_image(img)
      rl.unload_image(img)
      rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
      self._scrim_tex = tex
      return tex
    except Exception:
      return None

  def _project_point(self, x: float, y: float, z: float, rect: rl.Rectangle) -> Optional[tuple[float, float]]:
    mat = getattr(self._model, "_car_space_transform", None)
    if mat is None:
      return None
    pt = mat @ np.array([x, y, z], dtype=np.float64)
    if abs(pt[2]) < 1e-5:
      return None
    sx = float(pt[0] / pt[2])
    sy = float(pt[1] / pt[2])
    if not (rect.x - 400 <= sx <= rect.x + rect.width + 400 and rect.y - 200 <= sy <= rect.y + rect.height + 400):
      return None
    return sx, sy

  def _project_polyline(self, pts3d: np.ndarray, z_offset: float, rect: rl.Rectangle) -> np.ndarray:
    mat = getattr(self._model, "_car_space_transform", None)
    if mat is None or pts3d.shape[0] < 2:
      return np.empty((0, 2), dtype=np.float32)
    out = []
    for p in pts3d:
      if p[0] < 1.2:
        continue
      v = mat @ np.array([float(p[0]), float(p[1]), float(p[2]) + z_offset], dtype=np.float64)
      if v[2] <= 1e-4:
        continue
      sx, sy = float(v[0] / v[2]), float(v[1] / v[2])
      if rect.x - 600 <= sx <= rect.x + rect.width + 600 and rect.y - 200 <= sy <= rect.y + rect.height + 500:
        out.append((sx, sy))
    if len(out) < 2:
      return np.empty((0, 2), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)

  def _draw_optical_scrim(self, rect: rl.Rectangle) -> None:
    tex = self._ensure_scrim()
    if tex is not None:
      rl.draw_texture_pro(
        tex,
        rl.Rectangle(0, 0, tex.width, tex.height),
        rect,
        rl.Vector2(0, 0),
        0.0,
        rl.WHITE,
      )
    else:
      rl.draw_rectangle_gradient_h(
        int(rect.x), int(rect.y), int(rect.width * 0.42), int(rect.height),
        hs.rgba(6, 10, 16, 192), hs.rgba(6, 10, 16, 0),
      )
    # Subtle top & bottom sky/bumper polish
    rl.draw_rectangle_gradient_v(
      int(rect.x), int(rect.y), int(rect.width), int(rect.height * 0.14),
      hs.rgba(6, 10, 16, 92), hs.rgba(6, 10, 16, 0),
    )
    bh = int(rect.height * 0.18)
    rl.draw_rectangle_gradient_v(
      int(rect.x), int(rect.y + rect.height - bh), int(rect.width), bh,
      hs.rgba(6, 10, 16, 0), hs.rgba(6, 10, 16, 138),
    )

  def _draw_lane_edges(self, model, rect: rl.Rectangle, engaged: bool, left_bsm: bool, right_bsm: bool) -> None:
    lanes = getattr(model, "laneLines", None)
    probs = getattr(model, "laneLineProbs", None)
    if lanes is None or probs is None or len(lanes) < 3 or len(probs) < 3:
      return

    for idx, is_bsm in ((1, left_bsm), (2, right_bsm)):
      prob = float(probs[idx])
      if prob < 0.25 and not is_bsm:
        continue
      ll = lanes[idx]
      lx = np.asarray(ll.x, dtype=np.float32)
      ly = np.asarray(ll.y, dtype=np.float32)
      lz = np.asarray(ll.z, dtype=np.float32)
      if lx.size < 4:
        continue
      pts3d = np.stack([lx, ly, lz], axis=1)
      pts3d = pts3d[(pts3d[:, 0] >= 2.2) & (pts3d[:, 0] <= 55.0)]
      poly = self._project_polyline(pts3d, 1.22, rect)
      if len(poly) < 3:
        continue

      y_top = float(np.min(poly[:, 1]))
      y_bot = float(np.max(poly[:, 1]))
      if y_bot - y_top < 20.0:
        continue

      taper = 0.28 + 0.72 * np.clip((poly[:, 1] - y_top) / max(y_bot - y_top, 1.0), 0.0, 1.0)
      if is_bsm:
        rgb = (255, 66, 68)
        a_scale = 0.95
      elif engaged:
        rgb = (82, 184, 255)
        a_scale = min(1.0, prob + 0.15) * 0.75
      else:
        rgb = (180, 196, 218)
        a_scale = prob * 0.35

      ribbons = [
        (hs.stroke_ribbon(poly, 8.5 * taper), hs.rgba(255, 255, 255, int(68 * a_scale))),
        (hs.stroke_ribbon(poly, 3.2 * taper), hs.rgba(255, 255, 255, int(190 * a_scale))),
      ]
      hs.vertical_gradients(
        ribbons, y_top, y_bot,
        (hs.rgba(*rgb, 0), hs.rgba(*rgb, 215), hs.rgba(*rgb, 18)),
        (0.0, 0.72, 1.0),
      )

  def _draw_ar_ribbon_and_chevrons(self, model, sm, rect: rl.Rectangle,
                                   engaged: bool, override: bool, braking: bool,
                                   max_dist: float) -> Optional[np.ndarray]:
    pos = getattr(model, "position", None)
    if pos is None:
      return None
    px = np.asarray(pos.x, dtype=np.float32)
    py = np.asarray(pos.y, dtype=np.float32)
    pz = np.asarray(pos.z, dtype=np.float32)
    if px.size < 4:
      return None

    raw = np.stack([px, py, pz], axis=1)
    d_end = min(float(px[-1]) - 1.5, max(8.0, max_dist))
    if d_end <= 4.0:
      return None

    dists = np.geomspace(2.4, d_end, 36).astype(np.float32)
    center3d = _sample_centerline(raw, dists)
    center2d = self._project_polyline(center3d, 1.22, rect)
    if len(center2d) < 4:
      return None

    y_top = float(np.min(center2d[:, 1]))
    y_bot = float(np.max(center2d[:, 1]))
    if y_bot - y_top < 24.0:
      return None

    # Perspective width taper along screen depth (pure feathered laser light, zero boxy borders!)
    depth = np.clip((center2d[:, 1] - y_top) / max(y_bot - y_top, 1.0), 0.0, 1.0)
    taper = 0.14 + 0.86 * (depth ** 1.25)

    if engaged:
      if braking:
        outer_rgb, mid_rgb, core_rgb = (225, 48, 56), (255, 108, 96), (255, 228, 224)
        a_mul = 0.92
      else:
        outer_rgb, mid_rgb, core_rgb = (30, 134, 255), (82, 192, 255), (235, 250, 255)
        a_mul = 0.96
    elif override:
      outer_rgb, mid_rgb, core_rgb = (68, 140, 200), (142, 198, 240), (232, 246, 255)
      a_mul = 0.56
    else:
      outer_rgb, mid_rgb, core_rgb = (100, 118, 146), (168, 182, 204), (235, 240, 248)
      a_mul = 0.34

    # 3-layer feathered glowing trajectory ribbon
    outer_ribbons = [
      (hs.stroke_ribbon(center2d, 155.0 * taper), hs.rgba(255, 255, 255, int(68 * a_mul))),
      (hs.stroke_ribbon(center2d, 84.0 * taper), hs.rgba(255, 255, 255, int(112 * a_mul))),
    ]
    hs.vertical_gradients(
      outer_ribbons, y_top, y_bot,
      (hs.rgba(*outer_rgb, 22), hs.rgba(*outer_rgb, 222), hs.rgba(*outer_rgb, 28)),
      (0.0, 0.72, 1.0),
    )

    mid_ribbons = [
      (hs.stroke_ribbon(center2d, 44.0 * taper), hs.rgba(255, 255, 255, int(184 * a_mul))),
      (hs.stroke_ribbon(center2d, 17.0 * taper), hs.rgba(255, 255, 255, int(242 * a_mul))),
    ]
    hs.vertical_gradients(
      mid_ribbons, y_top, y_bot,
      (hs.rgba(*mid_rgb, 54), hs.rgba(*mid_rgb, 248), hs.rgba(*mid_rgb, 38)),
      (0.0, 0.72, 1.0),
    )

    core_rib = [(hs.stroke_ribbon(center2d, 6.0 * taper), hs.rgba(255, 255, 255, int(224 * a_mul)))]
    hs.vertical_gradients(
      core_rib, y_top, y_bot,
      (hs.rgba(*core_rgb, 64), hs.rgba(*core_rgb, 240), hs.rgba(*core_rgb, 24)),
      (0.0, 0.68, 1.0),
    )

    # FSD v15 road-surface projected perspective AR directional chevrons & lane cue badge
    if engaged and sm.alive.get("carrotMan", False):
      cm = sm["carrotMan"]
      turn_info = int(getattr(cm, "xTurnInfo", 0) or 0)
      dist_turn = float(getattr(cm, "xDistToTurn", 0.0) or 0.0)
      tbt_text = str(getattr(cm, "szTBTMainText", "") or "").strip()
      if dist_turn > 0 or turn_info > 0:
        self._draw_road_chevrons_and_cue(center2d, y_top, y_bot, rect, turn_info, dist_turn, tbt_text)

    return center2d

  def _draw_road_chevrons_and_cue(self, center2d: np.ndarray, y_top: float, y_bot: float, rect: rl.Rectangle,
                                  turn_info: int, dist_turn: float, tbt_text: str) -> None:
    order = np.argsort(center2d[:, 1])
    ys = center2d[order, 1]
    xs = center2d[order, 0]
    y_bot_vis = min(y_bot, rect.y + rect.height - 24.0)
    span = y_bot_vis - y_top
    if span < 120.0:
      return

    # 2 subtle forward-gliding perspective chevrons on visible road surface above cue pill
    for frac_base, alpha_base in ((0.58, 165), (0.42, 224)):
      frac = frac_base - self._chev_phase * 0.10
      if frac < 0.28 or frac > 0.68:
        continue
      ty = y_top + span * frac
      if ty < ys[0] or ty > ys[-1]:
        continue
      cx = float(np.interp(ty, ys, xs))
      scale = 0.45 + 0.55 * frac
      half_w = 112.0 * scale
      drop = 44.0 * scale
      pts = [(cx - half_w, ty + drop), (cx, ty), (cx + half_w, ty + drop)]
      hs.polyline(pts, 18.0 * scale, hs.rgba(56, 168, 255, int(alpha_base * 0.28)))
      hs.polyline(pts, 9.0 * scale, hs.rgba(162, 228, 255, alpha_base))

    # Bold frosted AR road cue pill sized for 6-inch C3X OLED (50px Bold!)
    if tbt_text or dist_turn > 0:
      cue_y = min(rect.y + rect.height - 96.0, max(y_top + span * 0.78, y_top + 140.0))
      cue_x = float(np.interp(np.clip(cue_y, ys[0], ys[-1]), ys, xs))
      dist_str = f"{dist_turn / 1000.0:.1f}km" if dist_turn >= 1000.0 else (f"{int(round(dist_turn / 10.0) * 10)}m" if dist_turn >= 100.0 else (f"{int(round(dist_turn))}m" if dist_turn > 0 else ""))
      label = tbt_text
      if label and self._type.width(label, 50.0, hs.BOLD) > 540.0:
        lo, hi = 0, len(label)
        while lo < hi:
          mid = (lo + hi + 1) // 2
          if self._type.width(label[:mid].rstrip() + "...", 50.0, hs.BOLD) <= 540.0:
            lo = mid
          else:
            hi = mid - 1
        label = label[:lo].rstrip() + "..." if lo > 0 else ""
      w_lbl = self._type.width(label, 50.0, hs.BOLD) if label else 0.0
      w_dist = self._type.width(dist_str, 50.0, hs.BOLD) if dist_str else 0.0
      icon_w = 58.0
      gap = 20.0 if (label and dist_str) else 0.0
      pill_w = 64.0 + icon_w + w_lbl + gap + w_dist
      pill_h = 94.0
      # Keep cleanly clear of bottom-right ETA capsule (~600px) and left edge
      px = max(rect.x + 480.0, min(rect.x + rect.width - pill_w - 640.0, cue_x - pill_w * 0.5))
      py = cue_y - pill_h * 0.5

      hs.chip(px, py, pill_w, pill_h, fill=hs.rgba(10, 22, 42, 212), ring=hs.rgba(102, 196, 255, 190), ring_width=2.8)
      icx, icy = px + 52.0, cue_y
      if turn_info > 0 and hs.maneuver_arrow(turn_info, icx, icy, 0.44, hs.rgba(130, 214, 255, 255)):
        pass
      else:
        hs.polyline([(icx, icy + 18), (icx, icy - 18)], 5.5, hs.rgba(116, 206, 255, 255))
        hs.polyline([(icx - 13, icy - 5), (icx, icy - 18), (icx + 13, icy - 5)], 5.5, hs.rgba(198, 236, 255, 255))
      tx = px + 92.0
      if label:
        self._type.draw_mid(label, tx, cue_y, 50.0, hs.rgba(246, 252, 255, 255), hs.BOLD)
        tx += w_lbl + gap
      if dist_str:
        self._type.draw_mid(dist_str, tx, cue_y, 50.0, hs.rgba(108, 199, 255, 255), hs.BOLD)

  def _draw_lead_ar(self, model, sm, rect: rl.Rectangle) -> None:
    if not sm.valid.get("radarState", False):
      return
    rs = sm["radarState"]
    lead = getattr(rs, "leadOne", None)
    if lead is None or not bool(getattr(lead, "status", False)):
      return
    d_rel = float(getattr(lead, "dRel", 0.0))
    if d_rel < 2.5 or d_rel > 140.0:
      return
    y_rel = float(getattr(lead, "yRel", 0.0))
    v_kph = max(0.0, float(getattr(lead, "vLeadK", getattr(lead, "vLead", 0.0))) * (3.6 if ui_state.is_metric else 2.2369363))
    v_rel = float(getattr(lead, "vRel", 0.0))
    closing = v_rel < -3.0

    # Sample road Z at lead distance
    pos = model.position
    px = np.asarray(pos.x, dtype=np.float32)
    pz = np.asarray(pos.z, dtype=np.float32)
    road_z = float(np.interp(d_rel, px, pz)) if px.size >= 2 else 0.0

    base = self._project_point(d_rel, -y_rel, road_z + 1.22, rect)
    roof = self._project_point(d_rel, -y_rel, road_z - 1.45, rect)
    if base is None:
      return
    bx, by = base
    rx, ry = roof if roof is not None else (bx, by - max(78.0, 2200.0 / max(d_rel, 8.0)))

    # Luminous hologram lock-on crescent beneath lead rear wheels
    arc_half = max(36.0, min(124.0, 1950.0 / max(d_rel, 8.0)))
    arc_col = hs.rgba(255, 92, 84, 248) if closing else hs.rgba(108, 206, 255, 252)
    glow_col = hs.rgba(255, 76, 68, 98) if closing else hs.rgba(56, 168, 255, 104)
    hs.ellipse_glow(bx, by + 4.0, arc_half * 1.15, max(14.0, arc_half * 0.25), glow_col)
    hs.taper_curve((bx - arc_half, by - 2.0), (bx, by + arc_half * 0.25), (bx + arc_half, by - 2.0), 7.5, arc_col)

    # Bold floating lead speed-first hero chip above roof (98px Bold speed | 44px Semi distance)
    spd_num = str(int(round(v_kph)))
    unit = "km/h" if ui_state.is_metric else "mph"
    dist_txt = f"{int(round(d_rel))}m"
    w_spd = self._type.width(spd_num, 98.0, hs.BOLD, spacing=-3.0)
    w_unit = self._type.width(unit, 28.0, hs.SEMI)
    w_dist = self._type.width(dist_txt, 44.0, hs.SEMI)
    pw = 32.0 + w_spd + 6.0 + w_unit + 22.0 + 2.0 + 22.0 + w_dist + 30.0
    ph = 118.0
    max_x = rect.x + rect.width - pw - 48.0
    tpms_zone = hs.zones.get("tpms")
    chip_y = max(rect.y + 84.0, ry - ph - 26.0)
    if tpms_zone is not None and chip_y < tpms_zone[1] + tpms_zone[3] + 16.0:
      max_x = min(max_x, tpms_zone[0] - pw - 20.0)
    chip_x = max(rect.x + 620.0, min(max_x, rx - pw * 0.5))

    ring_col = hs.rgba(255, 86, 78, 248) if closing else hs.rgba(103, 196, 255, 224)
    fill_col = hs.rgba(52, 12, 16, 238) if closing else hs.rgba(10, 17, 30, 236)
    spd_col = hs.rgba(255, 98, 90, 255) if closing else hs.rgba(255, 255, 255, 255)
    unit_col = hs.rgba(255, 178, 172, 220) if closing else hs.rgba(255, 255, 255, 178)
    dist_col = hs.rgba(255, 198, 194, 224) if closing else hs.rgba(164, 198, 228, 224)

    if closing:
      hs.ellipse_glow(chip_x + pw * 0.5, chip_y + ph * 0.5, pw * 0.62, ph * 0.90, hs.rgba(255, 62, 58, 86))
    rl.draw_line_ex(rl.Vector2(chip_x + pw * 0.5, chip_y + ph), rl.Vector2(rx, ry), 4.0, hs.with_alpha(ring_col, 215))
    hs.chip(chip_x, chip_y, pw, ph, fill=fill_col, ring=ring_col, ring_width=3.8 if closing else 3.4)

    cy = chip_y + ph * 0.5
    tx = chip_x + 32.0
    hs.ellipse_glow(tx + w_spd * 0.5, cy, max(62.0, w_spd * 0.68), 44.0,
                    hs.rgba(255, 68, 62, 64) if closing else hs.rgba(72, 184, 255, 58))
    self._type.draw_mid(spd_num, tx, cy - 2.0, 98.0, spd_col, hs.BOLD, spacing=-3.0)
    tx += w_spd + 6.0
    self._type.draw_mid(unit, tx, cy + 21.0, 28.0, unit_col, hs.SEMI)
    tx += w_unit + 22.0
    rl.draw_rectangle(int(tx), int(cy - 26.0), 2, 52, hs.rgba(255, 255, 255, 68))
    tx += 24.0
    self._type.draw_mid(dist_txt, tx, cy + 1.0, 44.0, dist_col, hs.SEMI)

  def render(self, rect: rl.Rectangle) -> None:
    now = time.monotonic()
    dt = min(0.1, max(0.0, now - self._last_t))
    self._last_t = now
    self._chev_phase = (self._chev_phase + dt * 0.85) % 1.0

    # Silky invisible optical edge scrim first so AR lines glow over road and left HUD stays razor-sharp
    self._draw_optical_scrim(rect)

    sm = ui_state.sm
    if sm is None or not sm.valid.get("modelV2", False):
      return

    model = sm["modelV2"]
    status = getattr(ui_state, "status", UIStatus.DISENGAGED)
    engaged = status == UIStatus.ENGAGED or bool(getattr(ui_state, "lat_active", False))
    override = status == UIStatus.OVERRIDE

    braking = False
    left_bsm = False
    right_bsm = False
    if sm.alive.get("carState", False):
      cs = sm["carState"]
      braking = bool(getattr(cs, "brakeLights", False)) or float(getattr(cs, "aEgo", 0.0)) < -1.8
      left_bsm = bool(getattr(cs, "leftBlindspot", False))
      right_bsm = bool(getattr(cs, "rightBlindspot", False))

    max_dist = 52.0
    if sm.valid.get("radarState", False):
      lead = getattr(sm["radarState"], "leadOne", None)
      if lead is not None and bool(getattr(lead, "status", False)):
        d_rel = float(getattr(lead, "dRel", 0.0))
        if 3.0 < d_rel < max_dist:
          max_dist = max(4.0, d_rel - 2.2)

    self._draw_lane_edges(model, rect, engaged, left_bsm, right_bsm)
    self._draw_ar_ribbon_and_chevrons(model, sm, rect, engaged, override, braking, max_dist)
    self._draw_lead_ar(model, sm, rect)

    # Render traffic light vision detection box cleanly when available
    tlv = getattr(self._model, "_tlv_overlay", None)
    if tlv is not None and hasattr(sm, "seen") and "carrotTlv" in sm.seen:
      tlv.render(sm, rect, [])
