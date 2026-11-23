"""Native Raylib 3D FSD v14 driving visualization for Comma 3X (2160x1080 OLED).

Renders a deep-charcoal matte 3D highway world driven live by modelV2, radarState,
and carState: sculpted bevel-shaded Hyundai IONIQ 5 ego vehicle with Parametric Pixel
taillights and electric cyan FSD underglow, luminous single cobalt-cyan trajectory
ribbon, scrolling dashed lane dividers, active lane-lock edges, road barriers, and
sculpted satin-graphite surrounding radar vehicles.
"""
import math
import time
from typing import Optional

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state
from openpilot.system.ui.lib.application import GL_VERSION

# Deep matte Tesla OLED night horizon & twilight palette
SKY_TOP = rl.Color(6, 10, 18, 255)
SKY_HORIZON = rl.Color(18, 28, 46, 255)
FOG_RGB = (11 / 255.0, 16 / 255.0, 27 / 255.0)
FOG_COLOR = rl.Color(11, 16, 27, 255)

# Clean Scandinavian Tesla FSD daytime studio light palette
DAY_SKY_TOP = rl.Color(218, 227, 240, 255)
DAY_SKY_HORIZON = rl.Color(238, 243, 249, 255)
DAY_FOG_RGB = (234 / 255.0, 240 / 255.0, 247 / 255.0)
DAY_FOG_COLOR = rl.Color(234, 240, 247, 255)

# GLSL shaders compatible with OpenGL ES 3.0 (C3X Adreno 630) and Desktop GL 3.3+
_LIT_VS = GL_VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexNormal;
in vec4 vertexColor;

uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;

out vec3 vWorldPos;
out vec3 vNormal;
out vec4 vColor;

void main() {
  vec4 wp = matModel * vec4(vertexPosition, 1.0);
  vWorldPos = wp.xyz;
  vNormal = normalize(mat3(matNormal) * vertexNormal);
  vColor = vertexColor;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

_LIT_FS = GL_VERSION + """
in vec3 vWorldPos;
in vec3 vNormal;
in vec4 vColor;

uniform vec4 colDiffuse;
uniform vec3 uCamPos;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uGlowStrength;
uniform float uSpecular;
uniform float uEmissive;
uniform float uDayMode;

out vec4 finalColor;

void main() {
  vec3 base = colDiffuse.rgb * vColor.rgb;
  float alpha = colDiffuse.a * vColor.a;

  vec3 N = normalize(vNormal);
  vec3 V = normalize(uCamPos - vWorldPos);

  vec3 lit;
  if (uEmissive > 0.5) {
    lit = base;
  } else {
    float hemi = N.y * 0.5 + 0.5;
    vec3 Lkey = normalize(vec3(-0.48, 0.82, 0.42));
    float ndlKey = max(dot(N, Lkey), 0.0);
    vec3 Lrim = normalize(vec3(0.72, 0.34, -0.58));
    float ndlRim = max(dot(N, Lrim), 0.0);

    // Night studio vs bright Scandinavian Tesla daytime studio lighting
    vec3 ambNight = mix(vec3(0.08, 0.11, 0.17), vec3(0.46, 0.56, 0.72), hemi) * 0.58;
    vec3 ambDay = mix(vec3(0.46, 0.51, 0.59), vec3(0.96, 0.98, 1.0), hemi) * 0.72;
    vec3 amb = mix(ambNight, ambDay, uDayMode);

    vec3 diffKeyNight = vec3(0.88, 0.93, 1.0) * ndlKey * 0.78;
    vec3 diffKeyDay = vec3(1.0, 0.99, 0.98) * ndlKey * 0.58;
    vec3 diffKey = mix(diffKeyNight, diffKeyDay, uDayMode);

    vec3 diffRimNight = vec3(0.24, 0.64, 1.0) * ndlRim * 0.42;
    vec3 diffRimDay = vec3(0.58, 0.76, 1.0) * ndlRim * 0.20;
    vec3 diffRim = mix(diffRimNight, diffRimDay, uDayMode);

    // Subtle upward electric-blue bounce from road underglow (muted in daylight)
    float underMask = clamp(1.0 - vWorldPos.y * 0.95, 0.0, 1.0) * clamp(1.0 - length(vWorldPos.xz) * 0.22, 0.0, 1.0);
    float underScale = mix(0.52, 0.14, uDayMode);
    vec3 underBounce = vec3(0.14, 0.58, 1.0) * max(-N.y * 0.55 + 0.45, 0.0) * underMask * uGlowStrength * underScale;

    // Crisp Blinn-Phong + Fresnel clearcoat on bodywork and glossy dark glass
    vec3 Hkey = normalize(Lkey + V);
    float specPow = mix(18.0, 72.0, uSpecular);
    float spec = pow(max(dot(N, Hkey), 0.0), specPow) * uSpecular * mix(0.65, 0.52, uDayMode);
    float fres = pow(1.0 - max(dot(N, V), 0.0), 3.2) * uSpecular * mix(0.42, 0.28, uDayMode);
    vec3 specTint = mix(vec3(0.76, 0.88, 1.0), vec3(1.0, 1.0, 1.0), uDayMode);
    vec3 fresTint = mix(vec3(0.32, 0.68, 1.0), vec3(0.74, 0.86, 1.0), uDayMode);
    vec3 specCol = specTint * spec + fresTint * fres;

    lit = base * (amb + diffKey + diffRim) + underBounce + specCol;
  }

  float dist = length(uCamPos - vWorldPos);
  float fog = 1.0 - exp(-pow(dist * uFogDensity, 2.0));
  vec3 col = mix(lit, uFogColor, clamp(fog, 0.0, 1.0));
  finalColor = vec4(col, alpha);
}
"""

_PATH_VS = GL_VERSION + """
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec4 vertexColor;

uniform mat4 mvp;

out vec2 vUv;
out vec4 vCol;

void main() {
  vUv = vertexTexCoord;
  vCol = vertexColor;
  gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

_PATH_FS = GL_VERSION + """
in vec2 vUv;
in vec4 vCol;

uniform vec3 uOuterColor;
uniform vec3 uMidColor;
uniform vec3 uCoreColor;
uniform float uAlpha;

out vec4 finalColor;

void main() {
  float x = abs(vUv.x - 0.5) * 2.0;
  float y = vUv.y;

  float ambientBloom = exp(-x * x * 3.6) * 0.46;
  float mainBand = smoothstep(0.32, 0.11, x) * 0.88;
  float brightCore = smoothstep(0.085, 0.0, x) * 0.68;

  float fadeNear = smoothstep(0.0, 0.075, y);
  float fadeFar = smoothstep(1.0, 0.58, y);

  float a = clamp((ambientBloom + mainBand + brightCore) * fadeNear * fadeFar * uAlpha, 0.0, 1.0);
  vec3 col = mix(uOuterColor, uMidColor, clamp(mainBand, 0.0, 1.0));
  col = mix(col, uCoreColor, clamp(brightCore, 0.0, 1.0));
  finalColor = vec4(col, a);
}
"""

_EDGE_FS = GL_VERSION + """
in vec2 vUv;
in vec4 vCol;

uniform vec3 uEdgeColor;
uniform float uAlpha;

out vec4 finalColor;

void main() {
  float x = abs(vUv.x - 0.5) * 2.0;
  float y = vUv.y;
  float glow = exp(-x * x * 5.8) * 0.58;
  float fade = smoothstep(0.0, 0.08, y) * smoothstep(1.0, 0.45, y);
  finalColor = vec4(uEdgeColor, clamp(glow * fade * uAlpha, 0.0, 1.0));
}
"""


def _rgba_color(r: int, g: int, b: int, a: int = 255) -> rl.Color:
  return rl.Color(int(r), int(g), int(b), int(a))


def _make_radial_texture(size: int, stops: list[tuple[float, tuple[int, int, int, int]]]) -> rl.Texture:
  """Pre-bakes a smooth radial RGBA glow or contact shadow texture on CPU once."""
  grid = np.linspace(-1.0, 1.0, size, dtype=np.float32)
  yy, xx = np.meshgrid(grid, grid, indexing="ij")
  r = np.clip(np.hypot(xx, yy), 0.0, 1.0)
  ps = np.array([s[0] for s in stops], dtype=np.float32)
  rgba = np.zeros((size, size, 4), dtype=np.uint8)
  for ch in range(4):
    vals = np.array([s[1][ch] for s in stops], dtype=np.float32)
    rgba[..., ch] = np.clip(np.interp(r, ps, vals) + 0.5, 0, 255).astype(np.uint8)
  img = rl.gen_image_color(size, size, rl.Color(0, 0, 0, 0))
  rl.ffi.memmove(img.data, np.ascontiguousarray(rgba).tobytes(), rgba.nbytes)
  tex = rl.load_texture_from_image(img)
  rl.unload_image(img)
  rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
  return tex


def _build_car_body_mesh(profile_zy: list[tuple[float, float]], half_w: float,
                         bevel_w: float = 0.14, bevel_h: float = 0.09,
                         taper_y0: float = 0.96, taper_span: float = 0.66,
                         taper_amt: float = 0.15, end_taper: float = 0.085) -> rl.Mesh:
  """Builds a smooth beveled 3D automotive body mesh from a closed 2D side silhouette (z, y).

  Coordinates: +X right, +Y up, +Z rear (ego faces -Z).
  Produces 4 longitudinal slices (-half_w, -inner_w, +inner_w, +half_w) with smooth
  area-weighted vertex normals and capped left/right side flanks.
  """
  pts = np.asarray(profile_zy, dtype=np.float32)
  n = len(pts)
  cz = float(np.mean(pts[:, 0]))
  cy = float(np.mean(pts[:, 1]))

  # Inset outer silhouette slightly for chamfered side edges
  inward = np.empty_like(pts)
  for i in range(n):
    dz = pts[i, 0] - cz
    dy = pts[i, 1] - cy
    d = max(math.hypot(float(dz), float(dy)), 1e-4)
    inward[i, 0] = pts[i, 0] - (dz / d) * bevel_h
    inward[i, 1] = pts[i, 1] - (dy / d) * bevel_h

  # 4 cross-sectional rings across car width (left flank -> left bevel -> right bevel -> right flank)
  inner_w = max(0.2, half_w - bevel_w)
  ring_specs = [
    (-half_w, inward),
    (-inner_w, pts),
    (inner_w, pts),
    (half_w, inward),
  ]

  rings = []
  for x_base, ring_zy in ring_specs:
    ring = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
      z = float(ring_zy[i, 0])
      y = float(ring_zy[i, 1])
      x = x_base
      if y > taper_y0:
        t = min(1.0, (y - taper_y0) / max(taper_span, 0.01))
        x *= 1.0 - taper_amt * t
      end_f = max(0.0, (abs(z) - 1.65) / 0.72)
      x *= 1.0 - end_taper * end_f * end_f
      ring[i] = (x, y, z)
    rings.append(ring)

  tris: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

  def add_tri(a, b, c):
    tris.append((a, b, c))

  # Stitch longitudinal body & bevel strips
  for r_idx in range(3):
    r0 = rings[r_idx]
    r1 = rings[r_idx + 1]
    for i in range(n):
      j = (i + 1) % n
      a, b, c, d = r0[i], r0[j], r1[j], r1[i]
      add_tri(a, b, c)
      add_tri(a, c, d)

  # Cap left flank (x < 0, normal -X) and right flank (x > 0, normal +X) using centroid fan
  left_center = np.mean(rings[0], axis=0)
  right_center = np.mean(rings[3], axis=0)
  for i in range(n):
    j = (i + 1) % n
    add_tri(left_center, rings[0][j], rings[0][i])
    add_tri(right_center, rings[3][i], rings[3][j])

  # Ensure consistent outward winding relative to body center (0, cy, cz)
  center = np.array([0.0, cy, cz], dtype=np.float32)
  tri_count = len(tris)
  v_count = tri_count * 3
  verts = np.empty((v_count, 3), dtype=np.float32)
  norms = np.empty((v_count, 3), dtype=np.float32)

  # First orient triangles outward and collect quantized-vertex smooth normals
  oriented = []
  accum: dict[tuple[int, int, int], np.ndarray] = {}
  for a, b, c in tris:
    fn = np.cross(b - a, c - a)
    tri_mid = (a + b + c) / 3.0
    if float(np.dot(fn, tri_mid - center)) < 0.0:
      b, c = c, b
      fn = -fn
    oriented.append((a, b, c, fn))
    for p in (a, b, c):
      k = (int(round(float(p[0]) * 500)), int(round(float(p[1]) * 500)), int(round(float(p[2]) * 500)))
      if k in accum:
        accum[k] = accum[k] + fn
      else:
        accum[k] = fn.copy()

  for idx, (a, b, c, fn) in enumerate(oriented):
    base = idx * 3
    fn_len = float(np.linalg.norm(fn))
    fn_unit = fn / max(fn_len, 1e-6)
    for k_off, p in enumerate((a, b, c)):
      verts[base + k_off] = p
      key = (int(round(float(p[0]) * 500)), int(round(float(p[1]) * 500)), int(round(float(p[2]) * 500)))
      sn = accum[key]
      sn_len = float(np.linalg.norm(sn))
      if sn_len > 1e-6:
        sn_unit = sn / sn_len
        # Blend smooth normal with face normal to keep crisp automotive character lines
        blended = sn_unit * 0.72 + fn_unit * 0.28
        norms[base + k_off] = blended / max(float(np.linalg.norm(blended)), 1e-6)
      else:
        norms[base + k_off] = fn_unit

  mesh = rl.Mesh()
  mesh.vertexCount = v_count
  mesh.triangleCount = tri_count

  v_bytes = np.ascontiguousarray(verts, dtype=np.float32).tobytes()
  n_bytes = np.ascontiguousarray(norms, dtype=np.float32).tobytes()
  mesh.vertices = rl.ffi.cast("float *", rl.mem_alloc(len(v_bytes)))
  mesh.normals = rl.ffi.cast("float *", rl.mem_alloc(len(n_bytes)))
  rl.ffi.memmove(mesh.vertices, v_bytes, len(v_bytes))
  rl.ffi.memmove(mesh.normals, n_bytes, len(n_bytes))
  rl.upload_mesh(rl.ffi.addressof(mesh), False)
  return mesh


def _build_ribbon_mesh_slots(max_steps: int) -> rl.Mesh:
  """Allocates a dynamic ribbon rl.Mesh (2 vertices per cross-section) for fast road/path strips."""
  v_count = (max_steps + 1) * 2
  t_count = max_steps * 2
  mesh = rl.Mesh()
  mesh.vertexCount = v_count
  mesh.triangleCount = t_count
  mesh.vertices = rl.ffi.cast("float *", rl.mem_alloc(v_count * 3 * 4))
  mesh.texcoords = rl.ffi.cast("float *", rl.mem_alloc(v_count * 2 * 4))
  mesh.normals = rl.ffi.cast("float *", rl.mem_alloc(v_count * 3 * 4))
  mesh.indices = rl.ffi.cast("unsigned short *", rl.mem_alloc(t_count * 3 * 2))

  norms = np.zeros((v_count, 3), dtype=np.float32)
  norms[:, 1] = 1.0
  idx = np.empty((max_steps, 6), dtype=np.uint16)
  i_arr = np.arange(max_steps, dtype=np.uint16) * 2
  idx[:, 0] = i_arr
  idx[:, 1] = i_arr + 2
  idx[:, 2] = i_arr + 1
  idx[:, 3] = i_arr + 1
  idx[:, 4] = i_arr + 2
  idx[:, 5] = i_arr + 3
  rl.ffi.memmove(mesh.normals, norms.tobytes(), norms.nbytes)
  rl.ffi.memmove(mesh.indices, idx.tobytes(), idx.nbytes)
  rl.upload_mesh(rl.ffi.addressof(mesh), True)
  return mesh


class _RoadSpline:
  """Evaluates smooth highway centerline X(d) and yaw(d) for forward distance d >= 0 (meters).

  3D world convention: ego at (0, 0, 0), forward = -Z, right = +X, up = +Y.
  """

  def __init__(self):
    self._xs = np.linspace(0.0, 170.0, 35, dtype=np.float32)
    self._ys = np.zeros(35, dtype=np.float32)
    self.left_lane_w = 1.88
    self.right_lane_w = 1.88
    self.left_prob = 0.85
    self.right_prob = 0.85
    self.left_edge_w = 5.65
    self.right_edge_w = 5.65

  def update_from_model(self, sm) -> None:
    if sm is None or not sm.valid.get("modelV2", False):
      self._ys.fill(0.0)
      return
    try:
      model = sm["modelV2"]
      pos = model.position
      px = np.asarray(pos.x, dtype=np.float32)
      py = np.asarray(pos.y, dtype=np.float32)
      if px.size >= 5 and py.size == px.size:
        # Extrapolate gently beyond model horizon using terminal slope
        d_max = float(px[-1])
        ys = np.interp(self._xs, px, py).astype(np.float32)
        if d_max > 15.0 and self._xs[-1] > d_max:
          slope = float(py[-1] - py[max(0, px.size - 4)]) / max(float(px[-1] - px[max(0, px.size - 4)]), 1.0)
          slope = max(-0.22, min(0.22, slope))
          beyond = self._xs > d_max
          ys[beyond] = float(py[-1]) + slope * (self._xs[beyond] - d_max)
        # Mild temporal smoothing removes jitter while tracking real highway curves crisply
        self._ys = self._ys * 0.45 + ys * 0.55

      # Extract ego lane width from laneLines[1] (left) and laneLines[2] (right)
      lanes = model.laneLines
      probs = model.laneLineProbs
      if len(lanes) >= 3 and len(probs) >= 3:
        self.left_prob = float(probs[1])
        self.right_prob = float(probs[2])
        if len(lanes[1].y) > 0 and self.left_prob > 0.25:
          w_l = abs(float(lanes[1].y[0]))
          if 1.1 <= w_l <= 3.2:
            self.left_lane_w = self.left_lane_w * 0.7 + w_l * 0.3
        if len(lanes[2].y) > 0 and self.right_prob > 0.25:
          w_r = abs(float(lanes[2].y[0]))
          if 1.1 <= w_r <= 3.2:
            self.right_lane_w = self.right_lane_w * 0.7 + w_r * 0.3

      edges = model.roadEdges
      if len(edges) >= 2:
        if len(edges[0].y) > 0:
          el = abs(float(edges[0].y[0]))
          if 2.2 <= el <= 9.0:
            self.left_edge_w = self.left_edge_w * 0.8 + max(el, self.left_lane_w + 3.4) * 0.2
        if len(edges[1].y) > 0:
          er = abs(float(edges[1].y[0]))
          if 2.2 <= er <= 9.0:
            self.right_edge_w = self.right_edge_w * 0.8 + max(er, self.right_lane_w + 3.4) * 0.2
    except Exception:
      pass

  def x_at_z(self, z: float) -> float:
    if z >= -1.5:
      return 0.0
    d = -z
    return float(np.interp(d, self._xs, self._ys))

  def xs_at_zs(self, zs: np.ndarray) -> np.ndarray:
    ds = np.maximum(0.0, -zs)
    out = np.interp(ds, self._xs, self._ys).astype(np.float32)
    out[zs >= -1.5] = 0.0
    return out

  def yaw_at_z(self, z: float) -> float:
    x0 = self.x_at_z(z + 1.5)
    x1 = self.x_at_z(z - 1.5)
    return math.atan2(-(x1 - x0), 3.0)


class Tesla3DRenderer:
  """Real-time Raylib 3D FSD v14 driving scene renderer."""

  def __init__(self):
    self._initialized = False
    self._failed = False
    self._spline = _RoadSpline()

    # Smooth animation & day/night theme state
    self._dash_phase = 0.0
    self._last_t = time.monotonic()
    self._glow_ema = 1.0
    self._day_mode = False

    # Exported screen anchor for floating lead pill in TeslaHudRenderer:
    # (screen_x, screen_y, d_rel_m, v_lead_kph, is_closing)
    self.lead_screen_anchor: Optional[tuple[float, float, float, float, bool]] = None

    # Perspective camera matching approved Option A composition:
    # Ego IONIQ 5 sits proudly at ~57% X, ~72% Y; left 34% open for Tesla V12 HUD
    self._cam = rl.Camera3D(
      rl.Vector3(-1.95, 3.60, 9.65),
      rl.Vector3(-0.75, 0.72, -22.0),
      rl.Vector3(0.0, 1.0, 0.0),
      40.0,
      rl.CameraProjection.CAMERA_PERSPECTIVE,
    )

    # GPU resources (lazily built once an OpenGL context is active)
    self._lit_shader = None
    self._path_shader = None
    self._edge_shader = None
    self._lit_locs: dict[str, int] = {}
    self._path_locs: dict[str, int] = {}
    self._edge_locs: dict[str, int] = {}

    self._under_tex = None
    self._shadow_tex = None
    self._lead_tex = None
    self._red_glow_tex = None
    self._amber_glow_tex = None

    self._ego_body_model = None
    self._ego_side_glass_model = None
    self._suv_body_model = None
    self._sedan_body_model = None
    self._box_model = None
    self._wheel_model = None

    self._path_steps = 72
    self._path_model = None
    self._edge_steps = 48
    self._edge_model = None

  def set_day_mode(self, is_day: bool) -> None:
    self._day_mode = bool(is_day)

  @property
  def day_mode(self) -> bool:
    return self._day_mode

  def _ensure_init(self) -> bool:
    if self._initialized:
      return True
    if self._failed:
      return False
    try:
      self._init_gpu()
      self._initialized = True
      return True
    except Exception:
      self._failed = True
      return False

  def _init_gpu(self) -> None:
    self._lit_shader = rl.load_shader_from_memory(_LIT_VS, _LIT_FS)
    self._lit_locs = {
      "cam": rl.get_shader_location(self._lit_shader, "uCamPos"),
      "fog_col": rl.get_shader_location(self._lit_shader, "uFogColor"),
      "fog_den": rl.get_shader_location(self._lit_shader, "uFogDensity"),
      "glow": rl.get_shader_location(self._lit_shader, "uGlowStrength"),
      "spec": rl.get_shader_location(self._lit_shader, "uSpecular"),
      "emissive": rl.get_shader_location(self._lit_shader, "uEmissive"),
      "day": rl.get_shader_location(self._lit_shader, "uDayMode"),
    }

    self._path_shader = rl.load_shader_from_memory(_PATH_VS, _PATH_FS)
    self._path_locs = {
      "outer": rl.get_shader_location(self._path_shader, "uOuterColor"),
      "mid": rl.get_shader_location(self._path_shader, "uMidColor"),
      "core": rl.get_shader_location(self._path_shader, "uCoreColor"),
      "alpha": rl.get_shader_location(self._path_shader, "uAlpha"),
    }

    self._edge_shader = rl.load_shader_from_memory(_PATH_VS, _EDGE_FS)
    self._edge_locs = {
      "col": rl.get_shader_location(self._edge_shader, "uEdgeColor"),
      "alpha": rl.get_shader_location(self._edge_shader, "uAlpha"),
    }

    # Pre-bake radial glow & AO contact shadow textures
    self._under_tex = _make_radial_texture(256, [
      (0.0, (98, 214, 255, 255)),
      (0.28, (42, 152, 255, 205)),
      (0.58, (20, 102, 248, 86)),
      (0.84, (10, 62, 210, 22)),
      (1.0, (8, 50, 200, 0)),
    ])
    self._shadow_tex = _make_radial_texture(128, [
      (0.0, (2, 4, 8, 244)),
      (0.52, (2, 4, 8, 206)),
      (0.80, (2, 4, 8, 54)),
      (1.0, (2, 4, 8, 0)),
    ])
    self._lead_tex = _make_radial_texture(128, [
      (0.0, (84, 198, 255, 215)),
      (0.45, (38, 138, 255, 112)),
      (1.0, (18, 90, 240, 0)),
    ])
    self._red_glow_tex = _make_radial_texture(128, [
      (0.0, (255, 36, 54, 215)),
      (0.38, (235, 18, 38, 95)),
      (0.72, (190, 6, 24, 26)),
      (1.0, (160, 0, 18, 0)),
    ])
    self._amber_glow_tex = _make_radial_texture(128, [
      (0.0, (255, 186, 40, 225)),
      (0.40, (245, 138, 18, 102)),
      (1.0, (200, 90, 0, 0)),
    ])

    # Sculpted IONIQ 5 ego profile (z front=-2.24 -> rear=+2.28, y=height)
    ioniq_profile = [
      (-2.22, 0.29), (-2.28, 0.62), (-2.12, 0.92), (-1.12, 1.04),
      (-0.34, 1.56), (1.24, 1.55), (2.06, 1.06), (2.26, 0.93),
      (2.24, 0.45), (2.02, 0.29),
    ]
    self._ego_body_model = rl.load_model_from_mesh(
      _build_car_body_mesh(ioniq_profile, half_w=0.95, bevel_w=0.14, bevel_h=0.075)
    )
    self._ego_body_model.materials[0].shader = self._lit_shader

    side_glass_profile = [
      (-0.98, 1.07), (-0.30, 1.49), (1.16, 1.48), (1.58, 1.07),
    ]
    self._ego_side_glass_model = rl.load_model_from_mesh(
      _build_car_body_mesh(side_glass_profile, half_w=0.945, bevel_w=0.03, bevel_h=0.015,
                           taper_amt=0.14, end_taper=0.0)
    )
    self._ego_side_glass_model.materials[0].shader = self._lit_shader

    # Sculpted traffic SUV & Sedan models
    suv_profile = [
      (-2.15, 0.28), (-2.22, 0.68), (-1.95, 0.96), (-1.05, 1.02),
      (-0.30, 1.66), (1.24, 1.66), (2.02, 1.06), (2.18, 0.96),
      (2.22, 0.40), (2.00, 0.26),
    ]
    self._suv_body_model = rl.load_model_from_mesh(
      _build_car_body_mesh(suv_profile, half_w=0.92, bevel_w=0.13, bevel_h=0.08)
    )
    self._suv_body_model.materials[0].shader = self._lit_shader

    sedan_profile = [
      (-2.16, 0.27), (-2.22, 0.64), (-1.92, 0.90), (-1.02, 0.98),
      (-0.28, 1.42), (1.10, 1.42), (1.58, 1.02), (2.18, 0.94),
      (2.22, 0.38), (2.00, 0.26),
    ]
    self._sedan_body_model = rl.load_model_from_mesh(
      _build_car_body_mesh(sedan_profile, half_w=0.90, bevel_w=0.13, bevel_h=0.08)
    )
    self._sedan_body_model.materials[0].shader = self._lit_shader

    # Shared lit primitives (unit cube + wheel cylinder)
    self._box_model = rl.load_model_from_mesh(rl.gen_mesh_cube(1.0, 1.0, 1.0))
    self._box_model.materials[0].shader = self._lit_shader

    self._wheel_model = rl.load_model_from_mesh(rl.gen_mesh_cylinder(0.5, 1.0, 18))
    self._wheel_model.materials[0].shader = self._lit_shader

    # Dynamic path & lane-edge glow ribbons
    self._path_model = rl.load_model_from_mesh(_build_ribbon_mesh_slots(self._path_steps))
    self._path_model.materials[0].shader = self._path_shader

    self._edge_model = rl.load_model_from_mesh(_build_ribbon_mesh_slots(self._edge_steps))
    self._edge_model.materials[0].shader = self._edge_shader

  def _set_lit_uniforms(self, specular: float = 0.25, emissive: float = 0.0, glow: float | None = None) -> None:
    s = self._lit_shader
    L = self._lit_locs
    if glow is not None:
      rl.set_shader_value(s, L["glow"], rl.ffi.new("float *", float(glow)), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(s, L["spec"], rl.ffi.new("float *", float(specular)), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(s, L["emissive"], rl.ffi.new("float *", float(emissive)), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

  def _draw_lit_box(self, pos: tuple[float, float, float], size: tuple[float, float, float],
                    color: rl.Color, rot_axis=(0.0, 1.0, 0.0), rot_deg: float = 0.0,
                    specular: float = 0.25, emissive: float = 0.0) -> None:
    self._set_lit_uniforms(specular=specular, emissive=emissive)
    rl.draw_model_ex(
      self._box_model,
      rl.Vector3(float(pos[0]), float(pos[1]), float(pos[2])),
      rl.Vector3(float(rot_axis[0]), float(rot_axis[1]), float(rot_axis[2])),
      float(rot_deg),
      rl.Vector3(float(size[0]), float(size[1]), float(size[2])),
      color,
    )

  def _draw_ground_quad(self, tex: rl.Texture, cx: float, y: float, cz: float,
                        w: float, l: float, alpha: float = 1.0, additive: bool = False) -> None:
    a = max(0, min(255, int(255 * alpha)))
    if a <= 1:
      return
    hw, hl = w * 0.5, l * 0.5
    rl.rl_disable_depth_mask()
    rl.rl_disable_backface_culling()
    if additive:
      rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
    rl.rl_set_texture(tex.id)
    rl.rl_begin(rl.RL_QUADS)
    rl.rl_color4ub(255, 255, 255, a)
    rl.rl_tex_coord2f(0.0, 0.0)
    rl.rl_vertex3f(cx - hw, y, cz - hl)
    rl.rl_tex_coord2f(0.0, 1.0)
    rl.rl_vertex3f(cx - hw, y, cz + hl)
    rl.rl_tex_coord2f(1.0, 1.0)
    rl.rl_vertex3f(cx + hw, y, cz + hl)
    rl.rl_tex_coord2f(1.0, 0.0)
    rl.rl_vertex3f(cx + hw, y, cz - hl)
    rl.rl_end()
    rl.rl_set_texture(0)
    if additive:
      rl.end_blend_mode()
    else:
      flush = getattr(rl, "rl_draw_render_batch_active", None)
      if flush is not None:
        flush()
    rl.rl_enable_backface_culling()
    rl.rl_enable_depth_mask()

  def _draw_vertical_glow(self, tex: rl.Texture, cx: float, cy: float, cz: float,
                          w: float, h: float, alpha: float = 1.0) -> None:
    a = max(0, min(255, int(255 * alpha)))
    if a <= 1:
      return
    hw, hh = w * 0.5, h * 0.5
    rl.rl_disable_depth_mask()
    rl.rl_disable_backface_culling()
    rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
    rl.rl_set_texture(tex.id)
    rl.rl_begin(rl.RL_QUADS)
    rl.rl_color4ub(255, 255, 255, a)
    rl.rl_tex_coord2f(0.0, 0.0)
    rl.rl_vertex3f(cx - hw, cy + hh, cz)
    rl.rl_tex_coord2f(0.0, 1.0)
    rl.rl_vertex3f(cx - hw, cy - hh, cz)
    rl.rl_tex_coord2f(1.0, 1.0)
    rl.rl_vertex3f(cx + hw, cy - hh, cz)
    rl.rl_tex_coord2f(1.0, 0.0)
    rl.rl_vertex3f(cx + hw, cy + hh, cz)
    rl.rl_end()
    rl.rl_set_texture(0)
    rl.end_blend_mode()
    rl.rl_enable_backface_culling()
    rl.rl_enable_depth_mask()

  def _draw_road_strip_rlgl(self, zs: np.ndarray, xs_center: np.ndarray,
                            x_offset: float, half_w: float, y: float,
                            color: rl.Color, fog_fade: bool = True,
                            fog_rgb: tuple[float, float, float] = (11.0, 16.0, 27.0)) -> None:
    """Draws a curved flat strip on the road surface with distance fog baked into vertex color."""
    n = len(zs)
    if n < 2:
      return
    rl.rl_disable_backface_culling()
    rl.rl_begin(rl.RL_TRIANGLES)
    fog_r, fog_g, fog_b = fog_rgb
    cam_z = float(self._cam.position.z)
    for i in range(n - 1):
      z0 = float(zs[i])
      z1 = float(zs[i + 1])
      x0 = float(xs_center[i]) + x_offset
      x1 = float(xs_center[i + 1]) + x_offset

      if fog_fade:
        d0 = abs(cam_z - z0)
        d1 = abs(cam_z - z1)
        f0 = 1.0 - math.exp(-((d0 * 0.0155) ** 2))
        f1 = 1.0 - math.exp(-((d1 * 0.0155) ** 2))
        r0 = int(color.r * (1.0 - f0) + fog_r * f0)
        g0 = int(color.g * (1.0 - f0) + fog_g * f0)
        b0 = int(color.b * (1.0 - f0) + fog_b * f0)
        a0 = int(color.a * max(0.0, 1.0 - f0 * 0.92))
        r1 = int(color.r * (1.0 - f1) + fog_r * f1)
        g1 = int(color.g * (1.0 - f1) + fog_g * f1)
        b1 = int(color.b * (1.0 - f1) + fog_b * f1)
        a1 = int(color.a * max(0.0, 1.0 - f1 * 0.92))
      else:
        r0 = r1 = color.r
        g0 = g1 = color.g
        b0 = b1 = color.b
        a0 = a1 = color.a

      if a0 <= 2 and a1 <= 2:
        continue

      # Triangle 1
      rl.rl_color4ub(r0, g0, b0, a0)
      rl.rl_vertex3f(x0 - half_w, y, z0)
      rl.rl_color4ub(r1, g1, b1, a1)
      rl.rl_vertex3f(x1 - half_w, y, z1)
      rl.rl_color4ub(r0, g0, b0, a0)
      rl.rl_vertex3f(x0 + half_w, y, z0)
      # Triangle 2
      rl.rl_color4ub(r0, g0, b0, a0)
      rl.rl_vertex3f(x0 + half_w, y, z0)
      rl.rl_color4ub(r1, g1, b1, a1)
      rl.rl_vertex3f(x1 - half_w, y, z1)
      rl.rl_color4ub(r1, g1, b1, a1)
      rl.rl_vertex3f(x1 + half_w, y, z1)
    rl.rl_end()
    rl.rl_enable_backface_culling()

  def _draw_barrier_wall(self, zs: np.ndarray, xs_center: np.ndarray, x_offset: float,
                         day: bool = False) -> None:
    """Draws sleek low highway side barrier + subtle lit top cap matched to day/night palette."""
    n = len(zs)
    if n < 2:
      return
    wall_r, wall_g, wall_b = (172.0, 184.0, 202.0) if day else (34.0, 44.0, 64.0)
    fog_r, fog_g, fog_b = (234.0, 240.0, 247.0) if day else (11.0, 16.0, 27.0)
    cap_col = _rgba_color(230, 237, 246, 245) if day else _rgba_color(72, 92, 126, 240)
    rl.rl_disable_backface_culling()
    rl.rl_begin(rl.RL_TRIANGLES)
    cam_z = float(self._cam.position.z)
    for i in range(n - 1):
      z0, z1 = float(zs[i]), float(zs[i + 1])
      x0, x1 = float(xs_center[i]) + x_offset, float(xs_center[i + 1]) + x_offset
      f0 = 1.0 - math.exp(-((abs(cam_z - z0) * 0.0155) ** 2))
      f1 = 1.0 - math.exp(-((abs(cam_z - z1) * 0.0155) ** 2))
      r0 = int(wall_r * (1.0 - f0) + fog_r * f0)
      g0 = int(wall_g * (1.0 - f0) + fog_g * f0)
      b0 = int(wall_b * (1.0 - f0) + fog_b * f0)
      r1 = int(wall_r * (1.0 - f1) + fog_r * f1)
      g1 = int(wall_g * (1.0 - f1) + fog_g * f1)
      b1 = int(wall_b * (1.0 - f1) + fog_b * f1)
      rl.rl_color4ub(r0, g0, b0, 255)
      rl.rl_vertex3f(x0, 0.0, z0)
      rl.rl_color4ub(r0, g0, b0, 255)
      rl.rl_vertex3f(x0, 0.50, z0)
      rl.rl_color4ub(r1, g1, b1, 255)
      rl.rl_vertex3f(x1, 0.0, z1)

      rl.rl_color4ub(r0, g0, b0, 255)
      rl.rl_vertex3f(x0, 0.50, z0)
      rl.rl_color4ub(r1, g1, b1, 255)
      rl.rl_vertex3f(x1, 0.50, z1)
      rl.rl_color4ub(r1, g1, b1, 255)
      rl.rl_vertex3f(x1, 0.0, z1)
    rl.rl_end()
    rl.rl_enable_backface_culling()
    self._draw_road_strip_rlgl(zs, xs_center, x_offset, 0.08, 0.51, cap_col, fog_rgb=(fog_r, fog_g, fog_b))

  def _update_and_draw_shader_ribbon(self, model, steps: int, z_start: float, z_end: float,
                                     x_offset: float, half_w: float, y: float,
                                     additive: bool = True) -> None:
    zs = np.linspace(z_start, z_end, steps + 1, dtype=np.float32)
    xs = self._spline.xs_at_zs(zs) + x_offset
    ts = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    verts = np.empty(((steps + 1) * 2, 3), dtype=np.float32)
    uvs = np.empty(((steps + 1) * 2, 2), dtype=np.float32)
    verts[0::2, 0] = xs - half_w
    verts[0::2, 1] = y
    verts[0::2, 2] = zs
    verts[1::2, 0] = xs + half_w
    verts[1::2, 1] = y
    verts[1::2, 2] = zs
    uvs[0::2, 0] = 0.0
    uvs[0::2, 1] = ts
    uvs[1::2, 0] = 1.0
    uvs[1::2, 1] = ts

    mesh = model.meshes[0]
    v_bytes = np.ascontiguousarray(verts).tobytes()
    u_bytes = np.ascontiguousarray(uvs).tobytes()
    v_ptr = rl.ffi.cast("const void *", rl.ffi.from_buffer(v_bytes))
    u_ptr = rl.ffi.cast("const void *", rl.ffi.from_buffer(u_bytes))
    rl.update_mesh_buffer(mesh, 0, v_ptr, len(v_bytes), 0)
    rl.update_mesh_buffer(mesh, 1, u_ptr, len(u_bytes), 0)

    # Disable backface culling BEFORE flushing queued rlgl road/barrier strips so double-sided
    # road geometry renders intact before direct VAO shader ribbons draw on top.
    rl.rl_disable_backface_culling()
    flush = getattr(rl, "rl_draw_render_batch_active", None)
    if flush is not None:
      flush()
    rl.rl_disable_depth_mask()
    if additive:
      rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
    rl.draw_model(model, rl.Vector3(0.0, 0.0, 0.0), 1.0, rl.WHITE)
    if additive:
      rl.end_blend_mode()
    rl.rl_enable_backface_culling()
    rl.rl_enable_depth_mask()

  def _draw_ego_ioniq5(self, glow_strength: float, braking: bool,
                       left_blink_on: bool, right_blink_on: bool, day: bool = False) -> None:
    # Electric cobalt FSD active underglow (alpha-blended in daylight so it never washes out)
    if glow_strength > 0.02:
      self._draw_ground_quad(self._under_tex, 0.0, 0.020, 0.16, 6.1, 8.4,
                             alpha=glow_strength * (0.42 if day else 0.98), additive=not day)

    # Grounded ambient occlusion shadow under wheels & chassis
    self._draw_ground_quad(self._shadow_tex, 0.0, 0.024, 0.0, 2.38, 4.95,
                           alpha=0.58 if day else 0.94, additive=False)

    # Main beveled pearl-silver / pearl-white bodywork
    body = _rgba_color(246, 249, 254) if day else _rgba_color(202, 214, 232)
    glass = _rgba_color(24, 34, 50) if day else _rgba_color(9, 15, 26)
    trim = _rgba_color(44, 54, 70) if day else _rgba_color(22, 30, 44)
    self._set_lit_uniforms(specular=0.45, emissive=0.0, glow=glow_strength)
    rl.draw_model(self._ego_body_model, rl.Vector3(0.0, 0.0, 0.0), 1.0, body)

    # Glossy dark side greenhouse glass (flush across left/right flanks!)
    self._set_lit_uniforms(specular=0.92, emissive=0.0, glow=glow_strength * 0.25)
    rl.draw_model(self._ego_side_glass_model, rl.Vector3(0.0, 0.0, 0.0), 1.0, glass)

    # Lower EV rocker skirt & rear diffuser
    self._draw_lit_box((0.0, 0.34, 0.02), (1.84, 0.21, 4.30), trim, specular=0.22)
    self._draw_lit_box((0.0, 0.35, 2.26), (1.72, 0.15, 0.12), trim, specular=0.22)
    self._draw_lit_box((0.0, 0.62, 2.30), (0.58, 0.13, 0.03),
                       _rgba_color(224, 230, 240) if day else _rgba_color(150, 164, 186), specular=0.18)

    # Flush panoramic dark glass roof (seated seamlessly into roof crown at y=1.555!)
    self._draw_lit_box((0.0, 1.560, 0.36), (1.26, 0.024, 1.48), glass, specular=0.94)

    # Visible sloped IONIQ 5 rear hatchback glass window (seated flush along rear C-pillar slope!)
    self._draw_lit_box((0.0, 1.325, 1.66), (1.36, 0.036, 0.84), trim, rot_axis=(1.0, 0.0, 0.0), rot_deg=32.0, specular=0.30)
    self._draw_lit_box((0.0, 1.335, 1.675), (1.26, 0.040, 0.72), glass, rot_axis=(1.0, 0.0, 0.0), rot_deg=32.0, specular=0.96)

    # Iconic upper rear spoiler + high-mount stop lamp
    self._draw_lit_box((0.0, 1.565, 1.38), (1.42, 0.048, 0.24), body, rot_axis=(1.0, 0.0, 0.0), rot_deg=3.0, specular=0.48)
    hmsl_col = _rgba_color(255, 42, 58) if braking else _rgba_color(148, 22, 34)
    self._draw_lit_box((0.0, 1.542, 1.50), (0.42, 0.024, 0.028), hmsl_col, emissive=1.0)

    # Side mirrors
    for side in (-1.0, 1.0):
      self._draw_lit_box((side * 0.94, 1.09, -0.56), (0.20, 0.095, 0.14), body, specular=0.45)

    # 4 detailed wheels + gunmetal aero alloy rims
    tire_col = _rgba_color(26, 32, 42) if day else _rgba_color(16, 21, 30)
    rim_col = _rgba_color(150, 164, 184) if day else _rgba_color(118, 134, 158)
    for wx, wz in ((-0.87, -1.36), (0.87, -1.36), (-0.87, 1.38), (0.87, 1.38)):
      self._set_lit_uniforms(specular=0.10, emissive=0.0)
      rl.draw_model_ex(
        self._wheel_model, rl.Vector3(wx - 0.125, 0.36, wz),
        rl.Vector3(0.0, 0.0, 1.0), -90.0, rl.Vector3(0.72, 0.25, 0.72), tire_col,
      )
      self._set_lit_uniforms(specular=0.72, emissive=0.0)
      rl.draw_model_ex(
        self._wheel_model, rl.Vector3(wx - 0.130, 0.36, wz),
        rl.Vector3(0.0, 0.0, 1.0), -90.0, rl.Vector3(0.46, 0.260, 0.46), rim_col,
      )

    # Iconic black horizontal rear garnish + Parametric Pixel LED taillights
    self._draw_lit_box((0.0, 0.97, 2.27), (1.74, 0.21, 0.07), trim, specular=0.35)

    red_bright = _rgba_color(255, 42, 58) if braking else _rgba_color(255, 28, 46)
    red_core = _rgba_color(255, 128, 138) if braking else _rgba_color(255, 86, 102)
    red_mid = _rgba_color(235, 34, 48) if braking else _rgba_color(156, 22, 34)
    amber_col = _rgba_color(255, 184, 34)

    for side in (-1.0, 1.0):
      blink_side = (side < 0 and left_blink_on) or (side > 0 and right_blink_on)
      for col in range(5):
        for row in range(2):
          if blink_side and col >= 3:
            pcol = amber_col
          else:
            pcol = red_core if col >= 3 else red_bright
          self._draw_lit_box(
            (side * (0.41 + col * 0.090), 0.935 + row * 0.070, 2.312),
            (0.056, 0.050, 0.032),
            pcol,
            emissive=1.0,
          )
    for c in range(-3, 4):
      self._draw_lit_box((c * 0.092, 0.97, 2.308), (0.050, 0.026, 0.025), red_mid, emissive=1.0)

    # Refined taillight bloom (boosted on braking; softer in daylight so it doesn't wash out)
    red_alpha = (0.52 if braking else 0.24) if day else (1.0 if braking else 0.72)
    for side in (-1.0, 1.0):
      self._draw_vertical_glow(self._red_glow_tex, side * 0.60, 0.97, 2.36, 0.96, 0.40, alpha=red_alpha)
      if (side < 0 and left_blink_on) or (side > 0 and right_blink_on):
        self._draw_vertical_glow(self._amber_glow_tex, side * 0.76, 0.97, 2.38, 0.76, 0.42,
                                 alpha=0.65 if day else 1.0)

  def _draw_traffic_car(self, cx: float, cz: float, yaw_rad: float, is_suv: bool,
                        is_lead: bool, day: bool = False) -> None:
    if is_lead:
      self._draw_ground_quad(self._lead_tex, cx, 0.026, cz, 4.4, 6.8,
                             alpha=0.74 if day else 0.88, additive=not day)
    self._draw_ground_quad(self._shadow_tex, cx, 0.022, cz, 2.5, 5.2,
                           alpha=(0.56 if is_lead else 0.44) if day else 0.90, additive=False)

    yaw_deg = math.degrees(yaw_rad)
    if day:
      body_col = _rgba_color(134, 152, 178) if is_lead else _rgba_color(168, 180, 198)
    else:
      body_col = _rgba_color(156, 172, 196) if is_lead else _rgba_color(110, 126, 150)
    model = self._suv_body_model if is_suv else self._sedan_body_model

    self._set_lit_uniforms(specular=0.35, emissive=0.0, glow=0.0)
    rl.draw_model_ex(model, rl.Vector3(cx, 0.0, cz), rl.Vector3(0.0, 1.0, 0.0), yaw_deg,
                     rl.Vector3(1.0, 1.0, 1.0), body_col)

    # Rotate local offsets by yaw around Y
    cy_v, sy_v = math.cos(yaw_rad), math.sin(yaw_rad)

    def to_world(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
      return (cx + lx * cy_v + lz * sy_v, ly, cz - lx * sy_v + lz * cy_v)

    roof_h = 1.66 if is_suv else 1.42
    rw_z = 1.92 if is_suv else 1.62
    glass_col = _rgba_color(32, 42, 58) if day else _rgba_color(16, 23, 36)
    self._draw_lit_box(to_world(0.0, roof_h - 0.24, rw_z), (1.32, 0.34, 0.08), glass_col,
                       rot_axis=(0.0, 1.0, 0.0), rot_deg=yaw_deg, specular=0.82)

    tl_col = _rgba_color(240, 42, 54) if is_lead else _rgba_color(205, 52, 62)
    for side in (-1.0, 1.0):
      self._draw_lit_box(to_world(side * 0.56, 0.94, 2.25), (0.44, 0.095, 0.05), tl_col,
                         rot_axis=(0.0, 1.0, 0.0), rot_deg=yaw_deg, emissive=1.0)

    tire_col = _rgba_color(28, 34, 44) if day else _rgba_color(15, 20, 29)
    for wx, wz in ((-0.86, -1.30), (0.86, -1.30), (-0.86, 1.30), (0.86, 1.30)):
      wp = to_world(wx, 0.35, wz)
      self._draw_lit_box(wp, (0.24, 0.66, 0.66), tire_col, rot_axis=(0.0, 1.0, 0.0), rot_deg=yaw_deg, specular=0.08)

  @staticmethod
  def _collect_traffic(sm) -> tuple[Optional[tuple[float, float, float, float]], list[tuple[float, float, bool]]]:
    """Extracts deduplicated lead vehicle and surrounding radar vehicles.

    Returns (lead_info, side_cars) where:
      lead_info = (d_rel, y_rel_left, v_lead_kph, v_rel) or None
      side_cars = [(d_rel, y_rel_left, is_suv), ...]
    """
    if sm is None or not sm.valid.get("radarState", False):
      return None, []
    try:
      rs = sm["radarState"]
    except Exception:
      return None, []

    lead_info = None
    lead_d = -1.0
    lead_y = 0.0
    lead_id = -1
    lead_one = getattr(rs, "leadOne", None)
    if lead_one is not None and bool(getattr(lead_one, "status", False)):
      d = float(getattr(lead_one, "dRel", 0.0))
      if 2.5 <= d <= 145.0:
        y_l = float(getattr(lead_one, "yRel", 0.0))
        v_k = float(getattr(lead_one, "vLeadK", getattr(lead_one, "vLead", 0.0)))
        v_r = float(getattr(lead_one, "vRel", 0.0))
        lead_info = (d, y_l, max(0.0, v_k * 3.6), v_r)
        lead_d, lead_y = d, y_l
        lead_id = int(getattr(lead_one, "radarTrackId", -1))

    side_cars: list[tuple[float, float, bool]] = []
    seen: list[tuple[float, float]] = []
    if lead_info is not None:
      seen.append((lead_d, lead_y))

    def try_add(tr, force_side: bool = False):
      if tr is None:
        return
      if not force_side and hasattr(tr, "status") and not bool(tr.status):
        return
      d = float(getattr(tr, "dRel", 0.0))
      if not (3.5 <= d <= 135.0):
        return
      y_l = float(getattr(tr, "yRel", 0.0))
      if abs(y_l) > 8.5:
        return
      tid = int(getattr(tr, "radarTrackId", -1))
      if lead_id >= 1 and tid == lead_id:
        return
      for sd, sy in seen:
        if abs(d - sd) < max(4.2, 0.08 * sd) and abs(y_l - sy) < 1.8:
          return
      seen.append((d, y_l))
      # Deterministic SUV vs Sedan visual variety per track
      is_suv = (abs(int(round(d * 1.3 + y_l * 7.0))) % 2 == 0)
      side_cars.append((d, y_l, is_suv))

    try_add(getattr(rs, "leadTwo", None))
    for attr in ("leadsLeft", "leadsRight", "leadsCenter"):
      for tr in (getattr(rs, attr, None) or ()):
        try_add(tr, force_side=True)
        if len(side_cars) >= 10:
          return lead_info, side_cars
    return lead_info, side_cars

  def render(self, rect: rl.Rectangle) -> None:
    """Renders the full-bleed Tesla FSD v14 3D driving view into `rect`."""
    now = time.monotonic()
    dt = min(0.1, max(0.0, now - self._last_t))
    self._last_t = now
    self.lead_screen_anchor = None

    day = self._day_mode

    # 1. Sky/horizon gradient matched to 3D exponential fog (Daylight studio vs OLED night)
    x, y, w, h = int(rect.x), int(rect.y), int(rect.width), int(rect.height)
    hor_h = int(h * 0.34)
    sky_top = DAY_SKY_TOP if day else SKY_TOP
    sky_hor = DAY_SKY_HORIZON if day else SKY_HORIZON
    fog_c = DAY_FOG_COLOR if day else FOG_COLOR
    rl.draw_rectangle_gradient_v(x, y, w, hor_h, sky_top, sky_hor)
    rl.draw_rectangle_gradient_v(x, y + hor_h, w, h - hor_h, sky_hor, fog_c)

    if not self._ensure_init():
      return

    sm = ui_state.sm
    self._spline.update_from_model(sm)

    # Extract driving state safely
    v_ego = 0.0
    braking = False
    left_blink = False
    right_blink = False
    left_bsm = False
    right_bsm = False
    a_ego = 0.0
    if sm is not None and sm.alive.get("carState", False):
      cs = sm["carState"]
      v_ego = max(0.0, float(getattr(cs, "vEgo", 0.0)))
      a_ego = float(getattr(cs, "aEgo", 0.0))
      braking = bool(getattr(cs, "brakeLights", False)) or a_ego < -1.8
      left_blink = bool(getattr(cs, "leftBlinker", False))
      right_blink = bool(getattr(cs, "rightBlinker", False))
      left_bsm = bool(getattr(cs, "leftBlindspot", False))
      right_bsm = bool(getattr(cs, "rightBlindspot", False))

    status = getattr(ui_state, "status", UIStatus.DISENGAGED)
    engaged = status == UIStatus.ENGAGED or bool(getattr(ui_state, "lat_active", False))
    override = status == UIStatus.OVERRIDE

    target_glow = 1.0 if engaged else (0.35 if override else 0.0)
    self._glow_ema += (target_glow - self._glow_ema) * min(1.0, dt * 8.0 + 0.18)

    # Scroll dashed lane dividers cleanly with vehicle speed
    dash_period = 9.5
    self._dash_phase = (self._dash_phase + v_ego * dt) % dash_period

    blink_phase = (int(now * 3.0) % 2) == 0
    left_blink_on = left_blink and blink_phase
    right_blink_on = right_blink and blink_phase

    lead_info, side_cars = self._collect_traffic(sm)

    # Upload frame-global lighting uniforms once
    s = self._lit_shader
    L = self._lit_locs
    cam_pos = rl.ffi.new("float[3]", [self._cam.position.x, self._cam.position.y, self._cam.position.z])
    fog_rgb_norm = DAY_FOG_RGB if day else FOG_RGB
    fog_rgb_255 = (234.0, 240.0, 247.0) if day else (11.0, 16.0, 27.0)
    fog_col = rl.ffi.new("float[3]", list(fog_rgb_norm))
    rl.set_shader_value(s, L["cam"], cam_pos, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
    rl.set_shader_value(s, L["fog_col"], fog_col, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
    rl.set_shader_value(s, L["fog_den"], rl.ffi.new("float *", 0.0155), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    rl.set_shader_value(s, L["day"], rl.ffi.new("float *", 1.0 if day else 0.0), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    # Compute display Z for lead car (clamped nicely in visual perspective sweet spot)
    lead_vis_z = -52.0
    if lead_info is not None:
      d_rel = lead_info[0]
      # Gentle visual compression at far distances keeps lead car crisp and legible on 6" OLED
      vis_d = d_rel if d_rel <= 28.0 else 28.0 + (d_rel - 28.0) * 0.42
      lead_vis_z = -max(5.8, min(72.0, vis_d))

    rl.begin_mode_3d(self._cam)

    # 2. Asphalt / architectural concrete surface & center ego lane
    road_zs = np.linspace(16.0, -165.0, 56, dtype=np.float32)
    road_xs = self._spline.xs_at_zs(road_zs)
    lw = self._spline.left_lane_w
    rw = self._spline.right_lane_w
    half_road = max(6.2, self._spline.left_edge_w, self._spline.right_edge_w)

    apron_col = _rgba_color(220, 229, 240, 255) if day else _rgba_color(6, 9, 16, 255)
    deck_col = _rgba_color(182, 194, 212, 255) if day else _rgba_color(15, 20, 31, 255)
    ego_lane_col = _rgba_color(196, 208, 224, 255) if day else _rgba_color(21, 29, 44, 255)

    self._draw_road_strip_rlgl(road_zs, road_xs, 0.0, 68.0, -0.025, apron_col, fog_rgb=fog_rgb_255)
    self._draw_road_strip_rlgl(road_zs, road_xs, 0.0, half_road, 0.0, deck_col, fog_rgb=fog_rgb_255)
    self._draw_road_strip_rlgl(road_zs, road_xs, (rw - lw) * 0.5, (lw + rw) * 0.5, 0.005, ego_lane_col, fog_rgb=fog_rgb_255)

    # Solid outer shoulder lines
    shoulder_col = _rgba_color(134, 149, 172, 224) if day else _rgba_color(94, 114, 146, 194)
    self._draw_road_strip_rlgl(road_zs, road_xs, -self._spline.left_edge_w, 0.075, 0.015, shoulder_col, fog_rgb=fog_rgb_255)
    self._draw_road_strip_rlgl(road_zs, road_xs, self._spline.right_edge_w, 0.075, 0.015, shoulder_col, fog_rgb=fog_rgb_255)

    # Scrolling crisp white dashed lane dividers
    if day:
      dash_col_l = _rgba_color(255, 255, 255, int(195 + 60 * min(1.0, self._spline.left_prob)))
      dash_col_r = _rgba_color(255, 255, 255, int(195 + 60 * min(1.0, self._spline.right_prob)))
    else:
      dash_col_l = _rgba_color(224, 234, 248, int(146 + 98 * min(1.0, self._spline.left_prob)))
      dash_col_r = _rgba_color(224, 234, 248, int(146 + 98 * min(1.0, self._spline.right_prob)))
    z_cur = 12.0 + self._dash_phase
    while z_cur > -148.0:
      dz = np.linspace(z_cur, z_cur - 3.8, 4, dtype=np.float32)
      dx = self._spline.xs_at_zs(dz)
      self._draw_road_strip_rlgl(dz, dx, -lw, 0.068, 0.018, dash_col_l, fog_rgb=fog_rgb_255)
      self._draw_road_strip_rlgl(dz, dx, rw, 0.068, 0.018, dash_col_r, fog_rgb=fog_rgb_255)
      z_cur -= dash_period

    # Low sleek highway edge barriers
    bar_zs = np.linspace(14.0, -150.0, 42, dtype=np.float32)
    bar_xs = self._spline.xs_at_zs(bar_zs)
    self._draw_barrier_wall(bar_zs, bar_xs, -(self._spline.left_edge_w + 0.90), day=day)
    self._draw_barrier_wall(bar_zs, bar_xs, self._spline.right_edge_w + 0.90, day=day)

    # 3. Subtle glowing active lane-lock edges (crimson red if blind-spot occupied on that side!)
    for side_x, prob, is_bsm in ((-lw, self._spline.left_prob, left_bsm), (rw, self._spline.right_prob, right_bsm)):
      if is_bsm:
        ecol = rl.ffi.new("float[3]", [0.95, 0.16, 0.20] if day else [1.0, 0.22, 0.24])
        ea = 0.96
      elif engaged and prob > 0.25:
        ecol = rl.ffi.new("float[3]", [0.10, 0.48, 0.98] if day else [0.22, 0.64, 1.0])
        ea = (0.72 if day else 0.55) * min(1.0, prob + 0.25)
      else:
        ecol = rl.ffi.new("float[3]", [0.48, 0.56, 0.68] if day else [0.42, 0.52, 0.68])
        ea = 0.26 if day else 0.16
      rl.set_shader_value(self._edge_shader, self._edge_locs["col"], ecol, rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
      rl.set_shader_value(self._edge_shader, self._edge_locs["alpha"], rl.ffi.new("float *", ea), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
      self._update_and_draw_shader_ribbon(self._edge_model, self._edge_steps, 6.0, -68.0, side_x,
                                          0.24 if is_bsm else 0.20, 0.022, additive=not day)

    # 4. Iconic Tesla FSD luminous single trajectory ribbon
    path_end_z = max(-52.0, lead_vis_z + 2.4) if lead_info is not None else -52.0
    if engaged:
      if braking:
        if day:
          outer_c, mid_c, core_c, pa = [0.88, 0.14, 0.20], [0.96, 0.24, 0.26], [1.0, 0.62, 0.60], 0.96
        else:
          outer_c, mid_c, core_c, pa = [0.85, 0.18, 0.22], [1.0, 0.42, 0.36], [1.0, 0.88, 0.86], 0.94
      else:
        if day:
          outer_c, mid_c, core_c, pa = [0.05, 0.38, 0.96], [0.10, 0.52, 1.0], [0.46, 0.78, 1.0], 0.96
        else:
          outer_c, mid_c, core_c, pa = [0.08, 0.44, 1.0], [0.28, 0.76, 1.0], [0.88, 0.97, 1.0], 0.96
    elif override:
      outer_c, mid_c, core_c, pa = ([0.18, 0.48, 0.86], [0.36, 0.64, 0.94], [0.76, 0.88, 1.0], 0.72) if day else ([0.24, 0.54, 0.78], [0.56, 0.78, 0.95], [0.90, 0.96, 1.0], 0.58)
    else:
      outer_c, mid_c, core_c, pa = ([0.44, 0.52, 0.64], [0.56, 0.64, 0.76], [0.80, 0.86, 0.94], 0.52) if day else ([0.30, 0.36, 0.46], [0.55, 0.62, 0.72], [0.85, 0.90, 0.96], 0.34)

    rl.set_shader_value(self._path_shader, self._path_locs["outer"], rl.ffi.new("float[3]", outer_c), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
    rl.set_shader_value(self._path_shader, self._path_locs["mid"], rl.ffi.new("float[3]", mid_c), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
    rl.set_shader_value(self._path_shader, self._path_locs["core"], rl.ffi.new("float[3]", core_c), rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3)
    rl.set_shader_value(self._path_shader, self._path_locs["alpha"], rl.ffi.new("float *", pa), rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)
    self._update_and_draw_shader_ribbon(self._path_model, self._path_steps, -2.2, path_end_z, 0.0, 1.46, 0.038, additive=not day)

    # 5. Sculpted 3D surrounding radar traffic (back-to-front for crisp alpha shadows)
    draw_list: list[tuple[float, float, float, bool, bool]] = []
    for d_s, y_l, is_suv in side_cars:
      vis_d = d_s if d_s <= 28.0 else 28.0 + (d_s - 28.0) * 0.44
      z_s = -max(5.5, min(82.0, vis_d))
      # radarState yRel is positive LEFT -> 3D world X is positive RIGHT
      x_s = self._spline.x_at_z(z_s) - y_l
      draw_list.append((z_s, x_s, self._spline.yaw_at_z(z_s), is_suv, False))

    lead_world_top = None
    if lead_info is not None:
      d_rel, y_l, v_kph, v_rel = lead_info
      x_lead = self._spline.x_at_z(lead_vis_z) - y_l
      yaw_lead = self._spline.yaw_at_z(lead_vis_z)
      draw_list.append((lead_vis_z, x_lead, yaw_lead, True, True))
      lead_world_top = (x_lead, 2.15, lead_vis_z, d_rel, v_kph, v_rel < -3.0)

    draw_list.sort(key=lambda item: item[0])  # farthest (-Z) first
    for z_c, x_c, yaw_c, is_suv, is_lead in draw_list:
      self._draw_traffic_car(x_c, z_c, yaw_c, is_suv=is_suv, is_lead=is_lead, day=day)

    # 6. Sculpted 3D Hyundai IONIQ 5 ego vehicle (always foreground)
    self._draw_ego_ioniq5(self._glow_ema, braking, left_blink_on, right_blink_on, day=day)

    rl.end_mode_3d()

    # Project 3D lead car roof coordinate to 2D screen space for HUD floating pill
    if lead_world_top is not None:
      lx, ly, lz, d_rel, v_kph, closing = lead_world_top
      p2d = rl.get_world_to_screen_ex(rl.Vector3(lx, ly, lz), self._cam, int(rect.width), int(rect.height))
      sx = float(rect.x + p2d.x)
      sy = float(rect.y + p2d.y)
      if rect.x + 480 <= sx <= rect.x + rect.width - 140 and rect.y + 80 <= sy <= rect.y + rect.height - 120:
        self.lead_screen_anchor = (sx, sy, d_rel, v_kph, closing)
