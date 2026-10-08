import math
import colorsys
import numpy as np

from openpilot.selfdrive.ui.onroad import hud_style as hs
from openpilot.selfdrive.ui.onroad.path_geometry import project_path, sample_path
from openpilot.selfdrive.ui.render_diagnostics import RenderDiagnostics
import pyray as rl
from openpilot.cereal import messaging, car, log
from dataclasses import dataclass, field
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.road_markings import (
  LANE_DASH_LENGTH_M as LANE_DASH_LENGTH_M, LANE_DASH_GAP_M as LANE_DASH_GAP_M,
  lane_dash_segments, project_lane_segments, project_blindspot_barrier,
)
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_draw import draw_text_ui_style
from openpilot.system.ui.lib.shader_polygon import draw_polygon, draw_polygon_solid, draw_polygons, Gradient
from openpilot.system.ui.lib import native_draw, native_geometry, native_text
from openpilot.system.ui.widgets import Widget

CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0
CARROT_PARAM_REFRESH_INTERVAL = 1.0

LaneChangeState = log.LaneChangeState

THROTTLE_COLORS = [
  rl.Color(13, 248, 122, 102),   # HSLF(148/360, 0.94, 0.51, 0.4)
  rl.Color(114, 255, 92, 89),    # HSLF(112/360, 1.0, 0.68, 0.35)
  rl.Color(114, 255, 92, 0),     # HSLF(112/360, 1.0, 0.68, 0.0)
]

NO_THROTTLE_COLORS = [
  rl.Color(242, 242, 242, 102), # HSLF(148/360, 0.0, 0.95, 0.4)
  rl.Color(242, 242, 242, 89),  # HSLF(112/360, 0.0, 0.95, 0.35)
  rl.Color(242, 242, 242, 0),   # HSLF(112/360, 0.0, 0.95, 0.0)
]


# C3X path style. Refined hues in the legacy ShowPathColor index order (red, orange, yellow,
# green, blue, navy, violet, amber, white, graphite), so every saved colour keeps its meaning.
PATH_PALETTE = (
  (255, 69, 58), (255, 149, 0), (255, 204, 0), (48, 209, 88), (10, 132, 255),
  (72, 96, 214), (175, 82, 222), (230, 126, 34), (242, 244, 247), (22, 24, 28),
)
PATH_FILL_ALPHA = 150
BRAKE_RGB = (255, 69, 58)
LANE_WHITE = (240, 243, 247)
LANE_YELLOW = (255, 196, 0)
# Screen-space fade (0 = bottom of the view, 1 = top): opaque underfoot, gone by the horizon.
FADE_STOPS = [0.0, 0.3, 0.5]
LEAD_RED = rl.Color(255, 69, 58, 255)
LEAD_AMBER = rl.Color(255, 159, 10, 255)
# Lead readout capsule: figures, the secondary distance, units, state words and the capsule height.
LEAD_SIZE = 58
LEAD_SIZE_2 = 40
LEAD_UNIT_SIZE = 36
LEAD_BADGE_SIZE = 36
LEAD_BADGE_H = 50.0
LEAD_BADGE_PAD = 14.0
LEAD_STATE_SIZE = 48
LEAD_CAPSULE_H = 86.0
LEAD_BLUE = rl.Color(10, 132, 255, 255)
LEAD_CROSS = rl.Color(255, 255, 255, 150)  # vision cross-check figure beside a radar distance
LEAD_TWO = rl.Color(230, 126, 34, 255)


@dataclass
class ModelPoints:
  raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
  projected_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))


@dataclass
class LeadVehicle:
  glow: list[float] = field(default_factory=list)
  chevron: list[float] = field(default_factory=list)
  fill_alpha: int = 0

@dataclass
class RadarLeadInfo:
  x: float = 0.0
  y: float = 0.0
  ax: float = 0.0
  ay: float = 0.0
  d_rel: float = 0.0
  y_rel: float = 0.0
  v_rel: float = 0.0
  v_lat: float = 0.0
  v_sum: float = 0.0
  radar: bool = False
  model_prob: float = 0.0
  has_future_point: bool = False




class ModelRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_filter = FirstOrderFilter(1.0, 0.25, 1 / gui_app.target_fps)
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self._path_offset_z = HEIGHT_INIT[0]

    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._transform_dirty = True
    self._clip_region = None

    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=[],
      stops=[],
    )

    # Get longitudinal control setting from car parameters
    if car_params := Params().get("CarParams"):
      cp = messaging.log_from_bytes(car_params, car.CarParams)
      self._longitudinal_control = cp.openpilotLongitudinalControl

    self._font_display: rl.Font = gui_app.font(FontWeight.DISPLAY)
    self._init_carrot()

  def set_transform(self, transform: np.ndarray):
    self._car_space_transform = transform.astype(np.float32)
    self._transform_dirty = True

  def _render(self, rect: rl.Rectangle):
    sm = ui_state.sm

    # Check if data is up-to-date
    if (sm.recv_frame["liveCalibration"] < ui_state.started_frame or
        sm.recv_frame["modelV2"] < ui_state.started_frame):
      return

    # Set up clipping region
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # Update state
    self._experimental_mode = sm['selfdriveState'].experimentalMode

    live_calib = sm['liveCalibration']
    self._path_offset_z = live_calib.height[0] if live_calib.height else HEIGHT_INIT[0]

    if sm.updated['carParams']:
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    model = sm['modelV2']
    if sm.updated['modelV2']:
      self._update_raw_points(model)

    if self._path.raw_points.shape[0] == 0:
      return

    # The Carrot renderers below rebuild all projected geometry they consume.
    # The legacy _update_model/_update_leads results only feed the disabled draw
    # calls, and _draw_path_carrot immediately replaces path.projected_points.
    self._transform_dirty = False

    # Legacy renderers are intentionally disabled. Restore their update pipeline
    # together with these draw calls if they are ever enabled again.
    # self._draw_lane_lines()
    # self._draw_path(sm)
    # self._draw_lead_indicator()
    self._draw_carrot_overlays(sm)

  def _draw_carrot_overlays(self, sm) -> None:
    if not hasattr(self, '_render_diagnostics'):
      self._render_diagnostics = RenderDiagnostics('uiModel')
    timing = self._render_diagnostics
    timing.start()
    # Floating labels claim screen boxes in priority order: the lead readout first, then radar tags.
    self._float_boxes = []
    self._ar_box = None
    self._lead_capsule = None
    timing.call('path', self._draw_path_carrot, sm)
    timing.call('lanes', self._draw_lane_lines_carrot, sm)
    timing.call('ar_turn', self._draw_ar_turn_carrot, sm)
    timing.call('blindspot', self._draw_blind_spot_carrot, sm)
    timing.call('radar', self._draw_radar_info_carrot, sm)
    if self._lead_capsule is not None:
      self._draw_lead_capsule_carrot(*self._lead_capsule)
    timing.values['native_draw'] = float(native_draw.active())
    timing.values['native_geometry'] = float(native_geometry.active())
    timing.values['native_text'] = float(native_text.active())
    timing.finish()

  def _update_raw_points(self, model):
    """Update raw 3D points from model data"""
    self._path.raw_points = np.array([model.position.x, model.position.y, model.position.z], dtype=np.float32).T

    for i, lane_line in enumerate(model.laneLines):
      self._lane_lines[i].raw_points = np.array([lane_line.x, lane_line.y, lane_line.z], dtype=np.float32).T

    for i, road_edge in enumerate(model.roadEdges):
      self._road_edges[i].raw_points = np.array([road_edge.x, road_edge.y, road_edge.z], dtype=np.float32).T

    self._lane_line_probs = np.array(model.laneLineProbs, dtype=np.float32)
    self._road_edge_stds = np.array(model.roadEdgeStds, dtype=np.float32)
    self._acceleration_x = np.array(model.acceleration.x, dtype=np.float32)

  def _update_leads(self, radar_state, path_x_array):
    """Update positions of lead vehicles"""
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    leads = [radar_state.leadOne, radar_state.leadTwo]

    for i, lead_data in enumerate(leads):
      if lead_data and lead_data.status:
        d_rel, y_rel, v_rel = lead_data.dRel, lead_data.yRel, lead_data.vRel
        idx = self._get_path_length_idx(path_x_array, d_rel)

        # Get z-coordinate from path at the lead vehicle position
        z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
        point = self._map_to_screen(d_rel, -y_rel, z + self._path_offset_z)
        if point:
          self._lead_vehicles[i] = self._update_lead_vehicle(d_rel, v_rel, point, self._rect)

  def _update_model(self, lead, path_x_array):
    """Update model visualization data based on model message"""
    max_distance = np.clip(path_x_array[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = self._get_path_length_idx(self._lane_lines[0].raw_points[:, 0], max_distance)

    # Update lane lines using raw points
    for i, lane_line in enumerate(self._lane_lines):
      lane_line.projected_points = self._map_line_to_polygon(
        lane_line.raw_points, 0.025 * self._lane_line_probs[i], 0.0, max_idx, max_distance
      )

    # Update road edges using raw points
    for road_edge in self._road_edges:
      road_edge.projected_points = self._map_line_to_polygon(road_edge.raw_points, 0.025, 0.0, max_idx, max_distance)

    # Update path using raw points
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)

    max_idx = self._get_path_length_idx(path_x_array, max_distance)
    self._path.projected_points = self._map_line_to_polygon(
      self._path.raw_points, 0.9, self._path_offset_z, max_idx, max_distance, allow_invert=False
    )

    self._update_experimental_gradient()

  def _update_experimental_gradient(self):
    """Pre-calculate experimental mode gradient colors"""
    if not self._experimental_mode:
      return

    max_len = min(len(self._path.projected_points) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      # Some points (screen space) are out of frame (rect space)
      track_y = self._path.projected_points[i][1]
      if track_y < self._rect.y or track_y > (self._rect.y + self._rect.height):
        i += 1
        continue

      # Calculate color based on acceleration (0 is bottom, 1 is top)
      lin_grad_point = 1 - (track_y - self._rect.y) / self._rect.height

      # speed up: 120, slow down: 0
      path_hue = np.clip(60 + self._acceleration_x[i] * 35, 0, 120)

      saturation = min(abs(self._acceleration_x[i] * 1.5), 1)
      lightness = np.interp(saturation, [0.0, 1.0], [0.95, 0.62])
      alpha = np.interp(lin_grad_point, [0.75 / 2.0, 0.75], [0.4, 0.0])

      # Use HSL to RGB conversion
      color = self._hsla_to_color(path_hue / 360.0, saturation, lightness, alpha)

      gradient_stops.append(lin_grad_point)
      segment_colors.append(color)

      # Skip a point, unless next is last
      i += 1 + (1 if (i + 2) < max_len else 0)

    # Store the gradient in the path object
    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=segment_colors,
      stops=gradient_stops,
    )

  def _update_lead_vehicle(self, d_rel, v_rel, point, rect):
    speed_buff, lead_buff = 10.0, 40.0

    # Calculate fill alpha
    fill_alpha = 0
    if d_rel < lead_buff:
      fill_alpha = 255 * (1.0 - (d_rel / lead_buff))
      if v_rel < 0:
        fill_alpha += 255 * (-1 * (v_rel / speed_buff))
      fill_alpha = min(fill_alpha, 255)

    # Calculate size and position
    sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 2.35
    x = np.clip(point[0], 0.0, rect.width - sz / 2)
    y = min(point[1], rect.height - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    glow = [(x + (sz * 1.35) + g_xo, y + sz + g_yo), (x, y - g_yo), (x - (sz * 1.35) - g_xo, y + sz + g_yo)]
    chevron = [(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)]

    return LeadVehicle(glow=glow, chevron=chevron, fill_alpha=int(fill_alpha))

  def _draw_lane_lines(self):
    """Draw lane lines and road edges"""
    for i, lane_line in enumerate(self._lane_lines):
      if lane_line.projected_points.size == 0:
        continue

      alpha = np.clip(self._lane_line_probs[i], 0.0, 0.7)
      color = rl.Color(255, 255, 255, int(alpha * 255))
      draw_polygon(self._rect, lane_line.projected_points, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size == 0:
        continue

      alpha = np.clip(1.0 - self._road_edge_stds[i], 0.0, 1.0)
      color = rl.Color(255, 0, 0, int(alpha * 255))
      draw_polygon(self._rect, road_edge.projected_points, color)

  def _draw_path(self, sm):
    """Draw path with dynamic coloring based on mode and throttle state."""
    if not self._path.projected_points.size:
      return

    allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control
    self._blend_filter.update(int(allow_throttle))

    if self._experimental_mode:
      # Draw with acceleration coloring
      if len(self._exp_gradient.colors) > 1:
        draw_polygon(self._rect, self._path.projected_points, gradient=self._exp_gradient)
      else:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(255, 255, 255, 30))
    else:
      # Blend throttle/no throttle colors based on transition
      blend_factor = round(self._blend_filter.x * 100) / 100
      blended_colors = self._blend_colors(NO_THROTTLE_COLORS, THROTTLE_COLORS, blend_factor)
      gradient = Gradient(
        start=(0.0, 1.0),  # Bottom of path
        end=(0.0, 0.0),  # Top of path
        colors=blended_colors,
        stops=[0.0, 0.5, 1.0],
      )
      draw_polygon(self._rect, self._path.projected_points, gradient=gradient)

  def _draw_lead_indicator(self):
    # Draw lead vehicles if available
    for lead in self._lead_vehicles:
      if not lead.glow or not lead.chevron:
        continue

      rl.draw_triangle_fan(lead.glow, len(lead.glow), rl.Color(218, 202, 37, 255))
      rl.draw_triangle_fan(lead.chevron, len(lead.chevron), rl.Color(201, 34, 49, lead.fill_alpha))

  @staticmethod
  def _get_path_length_idx(pos_x_array: np.ndarray, path_distance: float) -> int:
    """Get the index corresponding to the given path distance"""
    if len(pos_x_array) == 0:
      return 0
    indices = np.where(pos_x_array <= path_distance)[0]
    return indices[-1] if indices.size > 0 else 0

  def _map_to_screen(self, in_x, in_y, in_z):
    """Project a point in car space to screen space"""
    input_pt = np.array([in_x, in_y, in_z])
    pt = self._car_space_transform @ input_pt

    if abs(pt[2]) < 1e-6:
      return None

    x, y = pt[0] / pt[2], pt[1] / pt[2]

    clip = self._clip_region
    if not (clip.x <= x <= clip.x + clip.width and clip.y <= y <= clip.y + clip.height):
      return None

    return (x, y)

  def _map_line_to_polygon(self, line: np.ndarray, y_off: float, z_off: float, max_idx: int, max_distance: float, allow_invert: bool = True, y_shift: float = 0.0, start_idx: int = 0) -> np.ndarray:
    return native_geometry.project_ribbon(line, y_off, z_off, max_idx, self._car_space_transform,
                                          self._clip_region, allow_invert, max_distance, y_shift, start_idx)

  @staticmethod
  def _hsla_to_color(h, s, l, a):
    rgb = colorsys.hls_to_rgb(h, l, s)
    return rl.Color(
      int(rgb[0] * 255),
      int(rgb[1] * 255),
      int(rgb[2] * 255),
      int(a * 255)
    )

  @staticmethod
  def _blend_colors(begin_colors, end_colors, t):
    if t >= 1.0:
      return end_colors
    if t <= 0.0:
      return begin_colors

    inv_t = 1.0 - t
    return [rl.Color(
      int(inv_t * start.r + t * end.r),
      int(inv_t * start.g + t * end.g),
      int(inv_t * start.b + t * end.b),
      int(inv_t * start.a + t * end.a)
    ) for start, end in zip(begin_colors, end_colors, strict=True)]


  def _init_carrot(self):
    self._carrot_params_next_refresh_time = 0.0
    self._carrot_show_lane_info = 1
    self._carrot_show_radar_info = 0
    self._carrot_radar_lat_factor = 0.5

    self._carrot_show_path_mode_normal = 13
    self._carrot_show_path_color_normal = 14
    self._carrot_show_path_mode_lane = 13
    self._carrot_show_path_color_lane = 14
    self._carrot_show_path_color_cruise_off = 14

    self._carrot_show_path_mode = 13
    self._carrot_show_path_color = 14
    self._carrot_show_path_width = 1.0
    self._carrot_active_lane_line = False
    self._carrot_long_active = False
    self._carrot_use_lane_line_speed_apply = 0
    self._carrot_tire_trajectory = 0

    self._carrot_path_draw_seq = 0.0
    self._carrot_pos_t = 0.0

    self._carrot_path_x = 0
    self._carrot_path_y = 0
    self._carrot_path_width_px = 120
    self._carrot_path_fx = 0.0
    self._carrot_path_fy = 0.0
    self._carrot_path_fwidth = 120.0
    self._carrot_path_alpha = 0.85

    self._carrot_radar_track_id = -1
    self._carrot_lead_status = False
    self._carrot_radar_dist = 0.0
    self._carrot_vision_dist = 0.0
    self._carrot_lead_speed = 0.0
    self._carrot_lead_vrel = 0.0
    self._lead_w = 0.0
    self._float_boxes: list[tuple[float, float, float, float]] = []
    self._lead_capsule = None
    self._carrot_x_state = 0
    self._carrot_traffic_state = 0
    self._carrot_v_ego = 0.0
    self._carrot_brake_hold_active = False
    self._carrot_soft_hold_active = 0
    self._carrot_carrot_cruise = 0
    self._carrot_t_follow = 0.0
    self._carrot_tf_distance = 0.0
    self._carrot_tf_left = None
    self._carrot_tf_right = None

    self._carrot_lead_two_status = 0
    self._carrot_lead_two_xl = 0.0
    self._carrot_lead_two_xr = 0.0
    self._carrot_lead_two_y = 0.0

    self._carrot_colors = [rl.Color(*rgb, PATH_FILL_ALPHA) for rgb in PATH_PALETTE]
    self._carrot_depth_top = 0.0
    self._carrot_model_position = np.empty((0, 3), dtype=np.float32)
    self._carrot_max_distance = MIN_DRAW_DISTANCE
    self._type = hs.Type()


    self._carrot_lane_barrier_vertices = [
      np.empty((0, 2), dtype=np.float32),
      np.empty((0, 2), dtype=np.float32),
    ]


  def _refresh_carrot_params(self, now: float):
    if now < self._carrot_params_next_refresh_time:
      return

    self._carrot_show_lane_info = ui_state.params.get_int("ShowLaneInfo")
    self._carrot_show_radar_info = ui_state.params.get_int("ShowRadarInfo")
    self._carrot_radar_lat_factor = 0.5

    self._carrot_show_path_mode_normal = ui_state.params.get_int("ShowPathMode")
    self._carrot_show_path_color_normal = ui_state.params.get_int("ShowPathColor")
    self._carrot_show_path_mode_lane = ui_state.params.get_int("ShowPathModeLane")
    self._carrot_show_path_color_lane = ui_state.params.get_int("ShowPathColorLane")
    self._carrot_show_path_color_cruise_off = ui_state.params.get_int("ShowPathColorCruiseOff")
    self._carrot_tire_trajectory = ui_state.params.get_int("CarrotTireTrajectory")
    self._carrot_params_next_refresh_time = now + CARROT_PARAM_REFRESH_INTERVAL


  def _carrot_interp(self, x: float, xp, fp) -> float:
    return float(np.interp(x, xp, fp))


  def _draw_polygon_outline_carrot(self, points: np.ndarray, color: rl.Color, thickness: float):
    if points.shape[0] < 2:
      return
    if native_draw.try_outline(rl, points, color, thickness):
      return
    for i in range(points.shape[0] - 1):
      rl.draw_line_ex(
        rl.Vector2(float(points[i][0]), float(points[i][1])),
        rl.Vector2(float(points[i + 1][0]), float(points[i + 1][1])),
        thickness,
        color,
      )
    rl.draw_line_ex(
      rl.Vector2(float(points[-1][0]), float(points[-1][1])),
      rl.Vector2(float(points[0][0]), float(points[0][1])),
      thickness,
      color,
    )


  def _polygon_signed_area_carrot(self, pts: np.ndarray) -> float:
    if pts.shape[0] < 3:
      return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


  def _is_point_in_triangle_carrot(self, p, a, b, c) -> bool:
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])

    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-9:
      return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u >= -1e-6 and v >= -1e-6 and (u + v) <= 1.0 + 1e-6


  def _triangulate_polygon_carrot(self, pts: np.ndarray):
    n = pts.shape[0]
    if n < 3:
      return []
    if n == 3:
      return [(pts[0], pts[1], pts[2])]

    work = pts.copy()
    if self._polygon_signed_area_carrot(work) < 0.0:
      work = work[::-1].copy()

    indices = list(range(len(work)))
    tris = []
    guard = 0
    while len(indices) > 3 and guard < 512:
      guard += 1
      ear_found = False
      m = len(indices)
      for k in range(m):
        i0 = indices[(k - 1) % m]
        i1 = indices[k]
        i2 = indices[(k + 1) % m]
        a = work[i0]
        b = work[i1]
        c = work[i2]

        cross = (float(b[0] - a[0]) * float(c[1] - a[1]) -
                 float(b[1] - a[1]) * float(c[0] - a[0]))
        if cross <= 1e-6:
          continue

        inside = False
        for j in indices:
          if j in (i0, i1, i2):
            continue
          if self._is_point_in_triangle_carrot(work[j], a, b, c):
            inside = True
            break
        if inside:
          continue

        tris.append((a.copy(), b.copy(), c.copy()))
        del indices[k]
        ear_found = True
        break

      if not ear_found:
        return []

    if len(indices) == 3:
      tris.append((work[indices[0]].copy(), work[indices[1]].copy(), work[indices[2]].copy()))
    return tris


  def _draw_quad_fill_carrot(self, p0, p1, p2, p3, color: rl.Color):
    pts = np.array([p0, p1, p2, p3], dtype=np.float32)
    draw_polygon_solid(pts, color)

  def _draw_two_quads_from_6pts_carrot(self, x, y, fill_color: rl.Color, brake_valid: bool, color_idx: int):
    pts = np.array(list(zip(x, y)), dtype=np.float32)
    if pts.shape[0] != 6:
      return

    left_pts = np.array([pts[0], pts[1], pts[2], pts[5]], dtype=np.float32)
    right_pts = np.array([pts[5], pts[2], pts[3], pts[4]], dtype=np.float32)

    fill, edge = self._shape_colors_carrot(pts, fill_color, brake_valid, color_idx)
    draw_polygon_solid(left_pts, fill)
    draw_polygon_solid(right_pts, fill)
    if edge is not None:
      self._draw_polygon_outline_carrot(pts, edge, 2.5)

  def _draw_polygon_from_xy_carrot(self, xs, ys, fill_color: rl.Color, brake_valid: bool, color_idx: int):
    pts = np.column_stack((xs, ys)).astype(np.float32, copy=False)
    self._draw_polygon_points_carrot(pts, fill_color, brake_valid, color_idx)


  def _draw_polygon_points_carrot(self, pts: np.ndarray, fill_color: rl.Color, brake_valid: bool, color_idx: int):
    pts = np.asarray(pts, dtype=np.float32)

    if pts.shape[0] >= 2:
      keep = [0]
      for i in range(1, pts.shape[0]):
        if abs(float(pts[i][0] - pts[keep[-1]][0])) > 1e-3 or abs(float(pts[i][1] - pts[keep[-1]][1])) > 1e-3:
          keep.append(i)
      pts = pts[keep]

    if pts.shape[0] >= 3:
      if abs(float(pts[0][0] - pts[-1][0])) < 1e-3 and abs(float(pts[0][1] - pts[-1][1])) < 1e-3:
        pts = pts[:-1]

    if pts.shape[0] < 3:
      return

    fill, edge = self._shape_colors_carrot(pts, fill_color, brake_valid, color_idx)
    draw_polygon_solid(pts, fill)
    if edge is not None:
      self._draw_polygon_outline_carrot(pts, edge, 2.5)

  def _depth_scale_carrot(self, y: float) -> float:
    """0 at the path's far end, 1 at the bottom of the view."""
    top = self._carrot_depth_top
    bottom = self._rect.y + self._rect.height
    if bottom - top < 1.0:
      return 1.0
    return min(1.0, max(0.0, (y - top) / (bottom - top)))

  def _shape_colors_carrot(self, pts: np.ndarray, fill_color: rl.Color, brake_valid: bool, color_idx: int):
    """Pieces fade with distance and again underfoot; colours >= 10 get a soft white rim, braking a red one."""
    depth = self._depth_scale_carrot(float(pts[:, 1].mean()))
    k = (0.2 + 0.6 * depth) * float(np.interp(depth, [0.65, 1.0], [1.0, 0.3]))
    fill = rl.Color(fill_color.r, fill_color.g, fill_color.b, int(fill_color.a * k))
    if brake_valid:
      return fill, rl.Color(*BRAKE_RGB, int(240 * k))
    if color_idx >= 10:
      return fill, rl.Color(255, 255, 255, int(190 * k))
    return fill, None

  def _fade_fill_carrot(self, pts: np.ndarray, rgb, alpha: int, bell: bool = False) -> None:
    """Long ribbons get a screen-space fade; short pieces a single depth-scaled alpha.

    `bell` also fades the nearest metres, so lane markings do not run as wide bars under the
    speed cluster and the guidance block at the bottom corners.
    """
    if len(pts) < 3:
      return
    if len(pts) > 8:
      batch = getattr(self, '_bell_batch', None)
      if bell and batch is not None:
        # Collected and drawn together by _flush_bell_batch_carrot: one shader pass for every lane.
        batch.append((pts, rl.Color(*rgb, alpha)))
        return
      if bell:
        colors = [rl.Color(*rgb, 0), rl.Color(*rgb, 0), rl.Color(*rgb, alpha), rl.Color(*rgb, int(alpha * 0.45)), rl.Color(*rgb, 0)]
        stops = [0.0, 0.2, 0.32, 0.42, 0.5]
      else:
        colors = [rl.Color(*rgb, alpha), rl.Color(*rgb, int(alpha * 0.55)), rl.Color(*rgb, int(alpha * 0.12))]
        stops = FADE_STOPS
      draw_polygon(self._rect, pts, gradient=Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=stops))
    else:
      # Dashes come by the dozen each frame: plain arithmetic instead of numpy scalar calls.
      depth = self._depth_scale_carrot(float(pts[:, 1].sum()) / len(pts))
      k = 0.2 + 0.8 * depth
      if bell:
        # 1 to 0.3 deep, then down to 0.45 at 0.5 and gone from 0.75 (underfoot).
        k *= 1.0 if depth <= 0.3 else 1.0 - 2.75 * (depth - 0.3) if depth <= 0.5 else max(0.0, 0.45 - 1.8 * (depth - 0.5))
      a = int(alpha * k)
      if a > 0:
        draw_polygon_solid(pts, rl.Color(*rgb, a))

  def _draw_path_carpet_carrot(self, mode: int, color_idx: int, brake_valid: bool) -> None:
    """A soft lane carpet under every path mode."""
    if mode == 0 or mode < 9 or mode in (13, 14, 15):
      carpet = self._path.projected_points
    else:
      carpet = self._build_path_polygon_update_line_data_dist_carrot(
        self._carrot_model_position, self._carrot_show_path_width, 1.22, 1.22, self._carrot_max_distance, False)
    if len(carpet) < 4:
      return
    self._carrot_depth_top = float(carpet[:, 1].min())

    rgb = BRAKE_RGB if brake_valid else PATH_PALETTE[color_idx % 10]
    near = 130 if mode == 0 else (84 if 9 <= mode <= 12 else 56)
    if not brake_valid and color_idx % 10 == 9:
      rgb, near = (255, 255, 255), 44 if mode else 70
    colors = [rl.Color(*rgb, near), rl.Color(*rgb, near * 2 // 5), rl.Color(*rgb, 0)]
    draw_polygon(self._rect, carpet, gradient=Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=FADE_STOPS))


  def _draw_line_segment_carrot(self, p0, p1, color: rl.Color, thickness: float):
    rl.draw_line_ex(
      rl.Vector2(float(p0[0]), float(p0[1])),
      rl.Vector2(float(p1[0]), float(p1[1])),
      thickness,
      color,
    )


  def _draw_rect_fill_outline_carrot(self, x, y, w, h, fill_color: rl.Color, stroke_color: rl.Color, stroke_width: float):
    rect = rl.Rectangle(float(x), float(y), float(w), float(h))
    rl.draw_rectangle_rounded(rect, 0.15, 12, fill_color)
    if stroke_width > 0.0:
      rl.draw_rectangle_rounded_lines_ex(rect, 0.15, 12, stroke_width, stroke_color)


  def _update_path_end_carrot(self, sm):
    if self._path.raw_points.shape[0] == 0:
      return

    line = self._path.raw_points
    model = sm['modelV2']
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state is not None else None
    lead_two = radar_state.leadTwo if radar_state is not None else None
    lp = sm['longitudinalPlan']
    selfdrive_state = sm['selfdriveState']
    car_state = sm['carState']

    self._carrot_v_ego = car_state.vEgo
    self._carrot_brake_hold_active = car_state.brakeHoldActive
    self._carrot_soft_hold_active = car_state.softHoldActive
    self._carrot_carrot_cruise = car_state.carrotCruise
    self._carrot_long_active = selfdrive_state.enabled
    self._carrot_x_state = lp.xState
    self._carrot_traffic_state = lp.trafficState

    max_distance = np.clip(line[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    idx = self._get_path_length_idx(line[:, 0], max_distance)
    y = float(line[idx, 1])
    z = float(line[idx, 2])

    if len(model.leadsV3) > 0 and model.leadsV3[0].prob > 0.5:
      self._carrot_vision_dist = model.leadsV3[0].x[0] - 1.52
    else:
      self._carrot_vision_dist = 0.0

    self._carrot_lead_status_prev = self._carrot_lead_status
    self._carrot_lead_status = False
    self._carrot_radar_track_id = -1
    self._carrot_radar_dist = 0.0

    if lead_one is not None and lead_one.status:
      lead_idx = self._get_path_length_idx(line[:, 0], lead_one.dRel)
      z = float(line[lead_idx, 2])
      max_distance = float(lead_one.dRel)
      y = float(-lead_one.yRel)
      self._carrot_radar_track_id = int(lead_one.radarTrackId)
      self._carrot_radar_dist = float(lead_one.dRel) if lead_one.radar else 0.0
      self._carrot_lead_speed = float(lead_one.vLead) * (3.6 if ui_state.is_metric else 2.2369363)
      vrel = float(lead_one.vRel) * (3.6 if ui_state.is_metric else 2.2369363)
      self._carrot_lead_vrel = vrel if not self._carrot_lead_status_prev else self._carrot_lead_vrel * 0.8 + vrel * 0.2
      self._carrot_lead_status = True

    left_pt = self._map_to_screen(max_distance, y - 1.2, z + 1.22)
    right_pt = self._map_to_screen(max_distance, y + 1.2, z + 1.22)
    if left_pt is not None and right_pt is not None:
      lex, ley = left_pt
      rex, rey = right_pt
      path_width = rex - lex
      path_x = (lex + rex) / 2.0
      path_y = (ley + rey) / 2.0

      path_x = float(np.clip(path_x, 350.0, self._rect.width - 350.0))
      path_y = float(np.clip(path_y, 200.0, self._rect.height - 80.0))

      self._carrot_path_fx = self._carrot_path_fx * self._carrot_path_alpha + path_x * (1.0 - self._carrot_path_alpha)
      self._carrot_path_fy = self._carrot_path_fy * self._carrot_path_alpha + path_y * (1.0 - self._carrot_path_alpha)

      if path_width < 120.0:
        path_width = 120.0
      elif path_width > 800.0:
        path_width = 800.0

      self._carrot_path_fwidth = self._carrot_path_fwidth * self._carrot_path_alpha + path_width * (1.0 - self._carrot_path_alpha)
      self._carrot_path_x = int(self._carrot_path_fx)
      self._carrot_path_y = int(self._carrot_path_fy)
      self._carrot_path_width_px = int(self._carrot_path_fwidth)

    self._carrot_t_follow = float(lp.tFollow)
    self._carrot_tf_distance = float(lp.desiredDistance)
    tf_idx = self._get_path_length_idx(line[:, 0], self._carrot_tf_distance)
    tf_y = float(line[tf_idx, 1])
    tf_z = float(line[tf_idx, 2])
    self._carrot_tf_left = self._map_to_screen(self._carrot_tf_distance, tf_y - 1.0, tf_z + 1.22)
    self._carrot_tf_right = self._map_to_screen(self._carrot_tf_distance, tf_y + 1.0, tf_z + 1.22)

    if lead_two is not None and lead_one is not None and lead_two.radar and lead_two.dRel > lead_one.dRel + 3.0:
      lead_two_idx = self._get_path_length_idx(line[:, 0], lead_two.dRel)
      z2 = float(line[lead_two_idx, 2])
      y2 = float(-lead_two.yRel)
      lead_two_left = self._map_to_screen(lead_two.dRel, y2 - 1.2, z2 + 1.22)
      lead_two_right = self._map_to_screen(lead_two.dRel, y2 + 1.2, z2 + 1.22)

      if lead_two_left is not None and lead_two_right is not None:
        if self._carrot_lead_two_status > 0:
          self._carrot_lead_two_xl = self._carrot_lead_two_xl * 0.8 + lead_two_left[0] * 0.2
          self._carrot_lead_two_xr = self._carrot_lead_two_xr * 0.8 + lead_two_right[0] * 0.2
          self._carrot_lead_two_y = self._carrot_lead_two_y * 0.8 + lead_two_left[1] * 0.2
        else:
          self._carrot_lead_two_xl = lead_two_left[0]
          self._carrot_lead_two_xr = lead_two_right[0]
          self._carrot_lead_two_y = lead_two_left[1]

        src = int(lp.longitudinalPlanSource.raw)
        self._carrot_lead_two_status = 2 if src == 1 else 1
    else:
      self._carrot_lead_two_status = 0


  def _lead_runs_carrot(self) -> list[tuple]:
    """Readout contents on one baseline: ('dot', colour, r), ('num'/'text'/'unit', text, size, colour), ('bar',)
    and ('trend', sign, text, size). The capsule is sized for widest-digit 'num' slots so it does not breathe as digits change."""
    if self._carrot_soft_hold_active or self._carrot_brake_hold_active or self._carrot_carrot_cruise:
      text = "AUTOHOLD" if self._carrot_brake_hold_active else ("SOFTHOLD" if self._carrot_soft_hold_active else "CARROT")
      return [("text", text, LEAD_STATE_SIZE, hs.TEXT)]
    if self._carrot_long_active:
      if self._carrot_x_state in (3, 5):
        if self._carrot_v_ego < 1.0:
          return [("text", "Signal Error" if self._carrot_traffic_state >= 1000 else "Signal Ready", LEAD_STATE_SIZE, hs.TEXT)]
        return [("text", "Signal slowing", LEAD_STATE_SIZE, hs.TEXT)]
      if self._carrot_x_state == 4:
        return [("text", "E2E주행중", LEAD_STATE_SIZE, hs.TEXT)]

    text_color = hs.TEXT if self._carrot_x_state == 0 else (hs.rgba(191, 191, 191) if self._carrot_x_state == 1 else hs.GREEN)
    runs: list[tuple] = []
    radar = self._carrot_radar_dist > 0.0
    vision = self._carrot_vision_dist > 0.0
    # The distance that drives the car leads in its source colour and carries the unit; beside a radar
    # distance the vision estimate is a quiet cross-check that follows it, a size down, unit-less.
    if radar or vision:
      primary = self._carrot_radar_dist if radar else self._carrot_vision_dist
      source = (LEAD_RED if self._carrot_radar_track_id < 1 else LEAD_AMBER) if radar else LEAD_BLUE
      runs += [("dot", source, 8.0), ("num", f"{primary:.1f}", LEAD_SIZE, text_color),
               ("unit", "m", LEAD_UNIT_SIZE, hs.TEXT_3)]
      if radar and vision:
        runs += [("dot", LEAD_BLUE, 5.0), ("num", f"{self._carrot_vision_dist:.1f}", LEAD_SIZE_2, LEAD_CROSS)]
    if self._carrot_lead_status and self._carrot_lead_speed > 0.5:
      if runs:
        runs.append(("bar",))
      runs += [("num", f"{self._carrot_lead_speed:.0f}", LEAD_SIZE, hs.TEXT),
               ("unit", "km/h" if ui_state.is_metric else "mph", LEAD_UNIT_SIZE, hs.TEXT_3)]
      # Speed relative to us as a tinted badge, in the radar tags' language: green up = pulling away,
      # red down = closing.
      rel = round(self._carrot_lead_vrel)
      if abs(rel) >= 2:
        runs.append(("trend", 1 if rel > 0 else -1, f"{abs(rel)}", LEAD_BADGE_SIZE))
    return runs

  _LEAD_GAPS = {"dot": 10.0, "num": 0.0, "text": 0.0, "unit": 8.0, "bar": 24.0, "trend": 18.0}

  def _lead_run_width(self, run, slot: bool = False) -> float:
    kind = run[0]
    if kind == "dot":
      return run[2] * 2.0
    if kind == "bar":
      return 2.0 + 24.0
    if kind == "num" and slot:
      # Widest-digit slot: Inter's figures are proportional.
      return self._type.width("".join("0" if c.isdigit() else c for c in run[1]), run[2], hs.SEMI)
    if kind == "trend":
      # Badge: padding, arrow, gap, figure (widest-digit slot so the badge does not breathe either).
      figure = "".join("0" for _ in run[2]) if slot else run[2]
      return 2 * LEAD_BADGE_PAD + run[3] * 0.5 + 6.0 + self._type.width(figure, run[3], hs.BOLD)
    return self._type.width(run[1], run[2], hs.SEMI)

  def _lead_run_gap(self, prev, kind) -> float:
    if prev is None:
      return 0.0
    if prev == "dot":
      return 9.0
    return (18.0 if prev == "unit" else 24.0) if kind == "dot" else self._LEAD_GAPS[kind]

  def _layout_lead_capsule_carrot(self, cx: float, car_top: float) -> None:
    runs = self._lead_runs_carrot()
    if not runs:
      self._lead_w = 0.0
      return
    content = slots = 0.0
    prev = None
    for run in runs:
      gap = self._lead_run_gap(prev, run[0])
      content += gap + self._lead_run_width(run)
      slots += gap + self._lead_run_width(run, slot=True)
      prev = run[0]
    h = LEAD_CAPSULE_H
    target = slots + 64.0
    # Ease width changes (a figure gaining a digit, the trend cue appearing) instead of snapping.
    self._lead_w = target if self._lead_w <= 0.0 or abs(target - self._lead_w) > 160.0 else self._lead_w * 0.7 + target * 0.3
    w = max(self._lead_w, content + 48.0)
    # The tail tip sits just above the (truck-height) roof estimate.
    mid = hs.clear_of_panels(cx, car_top - 28.0 - h / 2, w, h)
    self._float_boxes.append((cx - w / 2, mid - h / 2, w, h))
    self._lead_capsule = (runs, cx, mid, w, h, car_top, content)

  def _draw_lead_capsule_carrot(self, runs, cx: float, mid: float, w: float, h: float, car_top: float, content: float) -> None:
    t = self._type
    bottom = mid + h / 2
    tail = 12.0
    if bottom + tail < car_top - 40.0:
      # Pushed up by a HUD card: extend the tail with a stem that fades out (the lead's real height is unknown).
      rl.draw_rectangle_gradient_v(int(cx) - 1, int(bottom + tail - 2.0), 3, int(car_top - bottom - tail),
                                   hs.rgba(255, 255, 255, 150), hs.rgba(255, 255, 255, 0))
    hs.chip(cx - w / 2, mid - h / 2, w, h)
    if bottom + tail < car_top:
      hs.triangle((cx - tail, bottom - 1.0), (cx + tail, bottom - 1.0), (cx, bottom + tail), hs.CHIP_FILL)
    x = cx - content / 2
    # Every run shares the large figures' baseline; smaller figures and units sit on it.
    base = mid + max((run[2] for run in runs if run[0] in ("num", "text")), default=LEAD_SIZE) * hs.INTER_CAP / 2
    prev = None
    for i, run in enumerate(runs):
      kind = run[0]
      x += self._lead_run_gap(prev, kind)
      width = self._lead_run_width(run)
      if kind == "dot":
        # A source dot is centred on the cap height of the figure it labels.
        size = runs[i + 1][2] if i + 1 < len(runs) and runs[i + 1][0] == "num" else LEAD_SIZE
        hs.dot(x + run[2], base - size * hs.INTER_CAP / 2, run[2], run[1])
      elif kind == "bar":
        rl.draw_rectangle_rounded(rl.Rectangle(x, mid - 24.0, 2.0, 48.0), 1.0, 4, hs.rgba(255, 255, 255, 56))
      elif kind == "num":
        t.draw(run[1], x, base, run[2], run[3], hs.SEMI)
      elif kind == "trend":
        size = run[3]
        color = hs.LIVE_GREEN if run[1] > 0 else hs.WARN_RED
        bh = LEAD_BADGE_H
        hs.card(x, mid - bh / 2, width, bh, hs.with_alpha(color, 46), bh / 2, hs.with_alpha(color, 90), 1.5)
        g = size * 0.5
        gh = g * 0.86
        gx, gy = x + LEAD_BADGE_PAD, mid
        if run[1] > 0:
          hs.triangle((gx, gy + gh / 2), (gx + g, gy + gh / 2), (gx + g / 2, gy - gh / 2), color)
        else:
          hs.triangle((gx, gy - gh / 2), (gx + g, gy - gh / 2), (gx + g / 2, gy + gh / 2), color)
        t.draw(run[2], gx + g + 6.0, mid + size * hs.INTER_CAP / 2, size, color, hs.BOLD)
      else:
        t.draw(run[1], x, base, run[2], run[3], hs.SEMI)
      x += width
      prev = kind

  def _draw_path_end_overlay_carrot(self):
    x = float(self._carrot_path_x)
    base = float(self._carrot_path_y)
    w = float(self._carrot_path_width_px)

    self._draw_tf_marker_carrot()

    if self._carrot_lead_status:
      if self._carrot_lead_two_status > 0:
        x2 = (self._carrot_lead_two_xl + self._carrot_lead_two_xr) / 2.0
        w2 = self._carrot_lead_two_xr - self._carrot_lead_two_xl
        y2 = self._carrot_lead_two_y
        if self._carrot_lead_two_status == 2:
          hs.ellipse_glow(x2, y2 - 4, w2 * 0.7, w2 * 0.12, hs.with_alpha(LEAD_TWO, 150))
        hs.taper_curve((x2 - w2 * 0.45, y2 - 5), (x2, y2 + 10), (x2 + w2 * 0.45, y2 - 5), 5.0, LEAD_TWO)

      # The lead stands in a pool of light in its source colour (radar red/amber, vision blue),
      # underlined by a crescent that tapers to points at the wheels.
      rcolor = LEAD_RED if self._carrot_radar_track_id < 1 else LEAD_AMBER
      stroke = rcolor if self._carrot_radar_track_id >= 0 else LEAD_BLUE
      hs.ellipse_glow(x, base - 4, w * 0.66, w * 0.11, hs.with_alpha(stroke, 170))
      hs.taper_curve((x - w * 0.52, base - 9), (x, base + 15), (x + w * 0.52, base - 9), 7.0, stroke)

    # Vehicle height is unknown: leave room for a truck so the readout never sits on the lead.
    self._layout_lead_capsule_carrot(x, base - w * 1.3)


  def _draw_tf_marker_carrot(self):
    if self._carrot_tf_distance > 0.0 and self._carrot_tf_left is not None and self._carrot_tf_right is not None:
      # Desired following distance: a quiet rule across the lane.
      lx, ly = float(self._carrot_tf_left[0]), float(self._carrot_tf_left[1])
      rx, ry = float(self._carrot_tf_right[0]), float(self._carrot_tf_right[1])
      inset = (rx - lx) * 0.18
      hs.polyline([(lx + inset, ly), (rx - inset, ry)], 4.0, hs.rgba(255, 255, 255, 230))

  _BELL = Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=[rl.Color(255, 255, 255, 0), rl.Color(255, 255, 255, 0),
                   rl.Color(255, 255, 255, 255), rl.Color(255, 255, 255, 115), rl.Color(255, 255, 255, 0)],
                   stops=[0.0, 0.2, 0.32, 0.42, 0.5])

  def _draw_lane_lines_carrot(self, sm):
    self._bell_batch = []
    try:
      self._draw_lane_lines_body_carrot(sm)
    finally:
      batch, self._bell_batch = self._bell_batch, None
      draw_polygons(self._rect, batch, self._BELL)

  def _draw_lane_lines_body_carrot(self, sm):
    if self._carrot_show_lane_info < 1:
      return
    if not sm.valid['modelV2'] or not sm.valid['carState']:
      return

    car_state = sm['carState']
    lane_line_probs = self._lane_line_probs
    lane_line_visible = lane_line_probs > 0.3
    road_edge_stds = self._road_edge_stds
    lane_zero = self._lane_lines[0].raw_points

    if lane_zero.shape[0] == 0:
      return
    if self._carrot_show_lane_info == 1 and not np.any(lane_line_visible):
      return

    max_distance = float(np.clip(lane_zero[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    max_idx = self._get_path_length_idx(lane_zero[:, 0], max_distance)
    left_lane_line = car_state.leftLaneLine
    right_lane_line = car_state.rightLaneLine
    draw_double_left = left_lane_line % 10 == 4

    lane_vertices = []
    empty_vertices = np.empty((0, 2), dtype=np.float32)
    lane_vertices_double = empty_vertices

    for i, lane_line in enumerate(self._lane_lines):
      # The old path projected and submitted low-confidence lanes with alpha
      # zero. They cannot affect the framebuffer, so avoid both the projection
      # work and the draw call while preserving the > 0.3 visibility threshold.
      if not lane_line_visible[i]:
        lane_vertices.append([])
        continue

      lane_code = left_lane_line if i == 1 else right_lane_line if i == 2 else None
      line_width = 0.025
      if i == 1 and left_lane_line >= 20:
        line_width = 0.05
      # Negative means the vehicle has no lane-type classification. Keep the
      # high-confidence model geometry visible and use the legacy solid style.
      is_dashed = lane_code is not None and lane_code >= 0 and lane_code % 10 == 0
      if is_dashed:
        projected_segments = project_lane_segments(
          lane_dash_segments(lane_line.raw_points, max_distance), line_width, self._car_space_transform, self._clip_region,
        )
      else:
        pts = self._map_line_to_polygon(
          lane_line.raw_points, line_width, 0.0, max_idx, max_distance,
        )
        projected_segments = [pts] if pts.size != 0 else []
      lane_vertices.append(projected_segments)

      if i == 1 and draw_double_left:
        lane_vertices_double = self._map_line_to_polygon(
          lane_line.raw_points,
          line_width,
          0.0,
          max_idx,
          max_distance,
          True,
          -0.3,
        )

    for i in range(4):
      if not lane_vertices[i]:
        continue
      # Confidence sets the strength; yellow keeps the centre-line meaning.
      alpha = int(np.interp(float(lane_line_probs[i]), [0.3, 0.8], [120, 235]))
      yellow = (i == 1 and left_lane_line >= 20) or (i == 2 and right_lane_line >= 20)
      rgb = LANE_YELLOW if yellow else LANE_WHITE

      # Dashes are many short pieces; at full strength they read as a hatch, not a line.
      seg_alpha = alpha if len(lane_vertices[i]) == 1 else int(alpha * 0.6)
      for lane_segment_vertices in lane_vertices[i]:
        self._fade_fill_carrot(lane_segment_vertices, rgb, seg_alpha, bell=True)

      if i == 1 and draw_double_left and lane_vertices_double.size != 0:
        self._fade_fill_carrot(lane_vertices_double, rgb, alpha, bell=True)

    if self._carrot_show_lane_info > 1:
      max_idx_road_edge = self._get_path_length_idx(lane_zero[:, 0], 100.0)
      road_vertices = [
        self._map_line_to_polygon(road_edge.raw_points, 0.025, 0.0, max_idx_road_edge, 100.0)
        for road_edge in self._road_edges
      ]
      for i in range(2):
        if road_vertices[i].size == 0:
          continue
        temp_f = float(np.clip(road_edge_stds[i] / 2.0, 0.0, 1.0))
        rgb = tuple(int(a + (b - a) * temp_f) for a, b in zip(BRAKE_RGB, (10, 132, 255), strict=True))
        self._fade_fill_carrot(road_vertices[i], rgb, 235, bell=True)



  def _build_blind_spot_barrier_carrot(self, model_position: np.ndarray, y_shift: float) -> np.ndarray:
    max_idx = self._get_path_length_idx(model_position[:, 0], 40.0)
    return project_blindspot_barrier(model_position[:max_idx + 1], y_shift, self._car_space_transform, self._clip_region)


  def _update_blind_spot_barriers_carrot(self, sm, update_left: bool = True, update_right: bool = True):
    if not sm.valid['modelV2']:
      self._carrot_lane_barrier_vertices[0] = np.empty((0, 2), dtype=np.float32)
      self._carrot_lane_barrier_vertices[1] = np.empty((0, 2), dtype=np.float32)
      return

    model_position = self._path.raw_points
    if model_position.shape[0] == 0:
      self._carrot_lane_barrier_vertices[0] = np.empty((0, 2), dtype=np.float32)
      self._carrot_lane_barrier_vertices[1] = np.empty((0, 2), dtype=np.float32)
      return

    if update_left:
      self._carrot_lane_barrier_vertices[0] = self._build_blind_spot_barrier_carrot(model_position, -1.7)
    if update_right:
      self._carrot_lane_barrier_vertices[1] = self._build_blind_spot_barrier_carrot(model_position, 1.7)

  def _draw_blind_spot_segments_carrot(self, points: np.ndarray, color: rl.Color):
    # One continuous wall, lit along its far stretch beside the car and faded toward the
    # near end, where the drive card covers it anyway.
    if len(points) < 6:
      return
    pts = hs.front_facing(np.asarray(points, dtype=np.float32))
    top, h = self._rect.y, self._rect.height
    y0 = float(pts[:, 1].min())
    y1 = min(float(pts[:, 1].max()), top + h)
    if y1 - y0 < 4.0:
      return

    def frac(y: float) -> float:
      return min(max(1.0 - (y - top) / h, 0.0), 1.0)

    rgb = (color.r, color.g, color.b)
    stops = [frac(y1), frac(y0 + 0.3 * (y1 - y0)), frac(y0)]
    stops[1] = max(stops[1], stops[0] + 1e-3)
    stops[2] = max(stops[2], stops[1] + 1e-3)
    # A translucent pane: the road stays readable through it, the top rail carries the colour.
    colors = [rl.Color(*rgb, 0), rl.Color(*rgb, int(color.a * 0.5)), rl.Color(*rgb, int(color.a * 0.3))]
    draw_polygon(self._rect, pts, gradient=Gradient(start=(0.0, 1.0), end=(0.0, 0.0), colors=colors, stops=stops))
    half = len(pts) // 2
    rail = min(pts[:half], pts[half:][::-1], key=lambda chain: float(chain[:, 1].mean()))
    near = y0 + 0.75 * (y1 - y0)
    for a, b in zip(rail[:-1], rail[1:], strict=True):
      k = float(np.clip((near - max(a[1], b[1])) / max(near - y0, 1.0) * 2.0, 0.0, 1.0))
      if k > 0.0:
        rl.draw_line_ex(rl.Vector2(float(a[0]), float(a[1])), rl.Vector2(float(b[0]), float(b[1])), 4.0,
                        rl.Color(*rgb, int(235 * k)))


  @staticmethod
  def _blind_spot_draw_state_carrot(car_state, radar_state, meta) -> tuple[bool, bool, bool, bool]:
    left_blindspot = bool(car_state.leftBlindspot)
    right_blindspot = bool(car_state.rightBlindspot)

    lane_change_state = meta.laneChangeState
    lane_change_direction = str(meta.laneChangeDirection).lower()
    right_lane_change = lane_change_state == LaneChangeState.preLaneChange and "right" in lane_change_direction
    left_lane_change = lane_change_state == LaneChangeState.preLaneChange and "left" in lane_change_direction

    assist_distance = float(car_state.vEgo) * 3.0
    left_assist = (
      not left_blindspot and radar_state.leadLeft.status and
      float(radar_state.leadLeft.dRel) < assist_distance and left_lane_change
    )
    right_assist = (
      not right_blindspot and radar_state.leadRight.status and
      float(radar_state.leadRight.dRel) < assist_distance and right_lane_change
    )
    return left_blindspot, right_blindspot, left_assist, right_assist


  def _draw_blind_spot_carrot(self, sm) -> None:
    input_services = ("modelV2", "carState", "radarState")
    if not all(sm.valid[service] for service in input_services):
      return

    car_state = sm['carState']
    radar_state = sm['radarState']
    meta = sm['modelV2'].meta

    left_blindspot, right_blindspot, left_assist, right_assist = self._blind_spot_draw_state_carrot(
      car_state, radar_state, meta,
    )
    if not (left_blindspot or right_blindspot or left_assist or right_assist):
      return

    # Blind-spot barrier: flat amber (or green for an assisted change), no outline.
    warn_color = rl.Color(255, 176, 32, 135)
    assist_color = rl.Color(48, 209, 88, 135)
    self._update_blind_spot_barriers_carrot(
      sm,
      update_left=left_blindspot or left_assist,
      update_right=right_blindspot or right_assist,
    )

    if left_blindspot:
      self._draw_blind_spot_segments_carrot(self._carrot_lane_barrier_vertices[0], warn_color)
    elif left_assist:
      self._draw_blind_spot_segments_carrot(self._carrot_lane_barrier_vertices[0], assist_color)

    if right_blindspot:
      self._draw_blind_spot_segments_carrot(self._carrot_lane_barrier_vertices[1], warn_color)
    elif right_assist:
      self._draw_blind_spot_segments_carrot(self._carrot_lane_barrier_vertices[1], assist_color)


  # ---- AR turn guidance: the card's own manoeuvre icon standing on the road at the corner ------------

  AR_NEAR, AR_FAR, AR_FADE_IN, AR_FADE_OUT = 12.0, 150.0, 25.0, 14.0
  AR_LATERAL = 2.4         # corner offset from our path toward the turn, metres
  AR_KEEP_LATERAL = 1.7    # keep-left/right signs stand on the lane line
  AR_HEIGHT_M = 3.2        # sign height in the world...
  AR_MIN_PX, AR_MAX_PX = 190.0, 360.0  # ...held legible far away and calm up close
  AR_LEAD_HEIGHT_M = 3.2   # lead silhouette height kept clear: a truck, since the vehicle class is unknown
  AR_LEAD_GAP = 18.0       # screen clearance between sign and lead, px
  AR_PUSH_MAX = 4.0        # furthest extra sideways step toward the turn, metres
  AR_PUSH_EASE = 0.2       # per-frame easing of that step

  def _ar_turn_state(self, sm):
    """(turn, side, distance, glyph colour) for an approaching turn on an active route, else None."""
    try:
      cm = sm['carrotMan']
      turn, dist = int(cm.xTurnInfo), float(cm.xDistToTurn)
      route = int(cm.nGoPosDist) > 0 and int(cm.nGoPosTime) > 0
      atc = bool(str(cm.atcType or ""))
    except Exception:
      return None
    if not route or not (self.AR_NEAR < dist <= self.AR_FAR):
      return None
    # Lateral sign follows the device frame (+y is right); keep-lane pins stand at the lane line.
    side = {1: -1.0, 2: 1.0, 3: -1.0, 4: 1.0, 6: -1.0, 7: -1.0, 8: 1.0}.get(turn)
    if side is None:
      return None
    return turn, side, dist, (hs.LIVE_GREEN if atc else hs.TEXT)

  def _project(self, pts: np.ndarray) -> np.ndarray | None:
    """Car-space points (N, 3) to screen (N, 2) without the clip test; None if any lies behind the camera."""
    q = pts.astype(np.float32) @ self._car_space_transform.T
    if np.any(q[:, 2] < 1e-3):
      return None
    return q[:, :2] / q[:, 2:3]

  def _ar_lead_box(self, sm, raw: np.ndarray) -> tuple | None:
    """Screen box (x, y, w, h) of the lead car's silhouette: 2.6 m wide, truck-high, on the road."""
    try:
      lead = sm['radarState'].leadOne
      if not lead.status:
        return None
      d_rel, y_rel = float(lead.dRel), -float(lead.yRel)
    except Exception:
      return None
    z = float(np.interp(d_rel, raw[:, 0], raw[:, 2])) + self._path_offset_z
    pts = self._project(np.array([[d_rel, y_rel - 1.3, z], [d_rel, y_rel + 1.3, z], [d_rel, y_rel, z - self.AR_LEAD_HEIGHT_M]]))
    if pts is None:
      return None
    x0, x1 = float(min(pts[:2, 0])), float(max(pts[:2, 0]))
    y0, y1 = float(pts[2, 1]), float(max(pts[:2, 1]))
    return x0, y0, x1 - x0, y1 - y0

  @staticmethod
  def _boxes_overlap(a: tuple, b: tuple, gap: float) -> bool:
    return a[0] < b[0] + b[2] + gap and b[0] < a[0] + a[2] + gap and a[1] < b[1] + b[3] + gap and b[1] < a[1] + a[3] + gap

  def _draw_ar_turn_carrot(self, sm) -> None:
    state = self._ar_turn_state(sm)
    raw = self._path.raw_points
    if state is None or raw is None or len(raw) < 2:
      self.__dict__.pop('_ar_push', None)  # a new sign starts where it belongs, not mid-glide
      return
    turn, side, dist, color = state
    if dist > float(raw[-1, 0]) or turn not in hs.AR_ICONS:
      return
    alpha = min(1.0, (self.AR_FAR - dist) / self.AR_FADE_IN, (dist - self.AR_NEAR) / self.AR_FADE_OUT)
    offset = self.AR_KEEP_LATERAL if turn in (3, 4) else (0.0 if turn == 8 else self.AR_LATERAL)
    centre = float(np.interp(dist, raw[:, 0], raw[:, 1]))
    road_z = float(np.interp(dist, raw[:, 0], raw[:, 2])) + self._path_offset_z

    def place(lateral: float):
      pts = self._project(np.array([[dist, lateral, road_z], [dist, lateral, road_z - self.AR_HEIGHT_M],
                                    [dist, lateral - 1.1, road_z], [dist, lateral + 1.1, road_z]]))
      if pts is None:
        return None
      foot_x, foot_y = float(pts[0, 0]), float(pts[0, 1])
      return pts, foot_x, foot_y, min(self.AR_MAX_PX, max(self.AR_MIN_PX, foot_y - float(pts[1, 1])))

    # The lead car is what the driver must see: the sign steps out sideways, toward the turn, until it
    # clears the car's silhouette, and eases there so it never jumps.
    lead = self._ar_lead_box(sm, raw)
    push = 0.0
    if lead is not None:
      for step in np.arange(0.0, self.AR_PUSH_MAX + 1e-3, 0.25):
        placed = place(centre + side * (offset + step))
        if placed is None:
          break
        box = hs.ar_sign_box(turn, placed[1], placed[2], placed[3])
        if box is None or not self._boxes_overlap(box, lead, self.AR_LEAD_GAP):
          push = float(step)
          break
      else:
        push = self.AR_PUSH_MAX
    self._ar_push = getattr(self, '_ar_push', push)
    self._ar_push += (push - self._ar_push) * self.AR_PUSH_EASE
    placed = place(centre + side * (offset + self._ar_push))
    if placed is None:
      return
    pts, foot_x, foot_y, height = placed
    tint = hs.LIVE_GREEN if color is hs.LIVE_GREEN else hs.NAV

    # Contact shadow on the road plants the sign. Should it still cover the lead car (no room left beside
    # it), the sign is drawn translucent and without a shadow so the car still reads in front.
    sign_box = hs.ar_sign_box(turn, foot_x, foot_y, height)
    occluded = lead is not None and sign_box is not None and self._boxes_overlap(sign_box, lead, 0.0)
    if occluded:
      alpha *= 0.5
    else:
      rx = max(26.0, abs(float(pts[3, 0] - pts[2, 0])) / 2)
      for grow, a in ((1.0, 70), (0.7, 60), (0.42, 70)):
        rl.draw_ellipse(int(foot_x), int(foot_y), rx * grow, max(6.0, rx * grow * 0.22), rl.Color(0, 0, 0, int(a * alpha)))
    box = hs.ar_sign(turn, foot_x, foot_y, height, tint, alpha)
    if box is not None:
      # Radar tags drawn later keep clear of the sign.
      self._float_boxes.append(box)
      self._ar_box = box

  def _draw_radar_info_carrot(self, sm):
    if self._carrot_show_radar_info <= 0:
      return
    if not sm.valid['radarState'] or not sm.valid['modelV2']:
      return

    radar_state = sm['radarState']
    model = sm['modelV2']
    if len(model.laneLines) < 3:
      return

    lane_line = model.laneLines[2]
    lane_x = np.array(lane_line.x, dtype=np.float32)
    lane_z = np.array(lane_line.z, dtype=np.float32)

    # Nearest first, so the closest cars claim the clean spot above their roofs.
    leads = sorted([*radar_state.leadsLeft, *radar_state.leadsRight, *radar_state.leadsCenter], key=lambda ld: ld.dRel)
    for lead in leads:
      d_rel = float(lead.dRel)
      if d_rel <= 2.5:
        continue

      idx = self._get_path_length_idx(lane_x, d_rel)
      if idx >= len(lane_z):
        continue

      road_z = float(lane_z[idx])
      y_rel = float(lead.yRel)
      side = self._map_to_screen(d_rel, -y_rel, road_z - 0.61)
      if side is None:
        continue

      x, y = side
      v = float(lead.vLeadK)
      v_lat = float(lead.vLat)
      radar = bool(lead.radar)
      model_prob = float(lead.modelProb)
      v_abs = math.sqrt(v * v + v_lat * v_lat)
      v_sum = v_abs if v >= 0.0 else -v_abs

      if v_abs > 3.0:
        speed_text = f"{(v_sum * 3.6):.0f}" if ui_state.is_metric else f"{(v_sum * 2.2369363):.0f}"

        if not radar:
          ring = LEAD_BLUE
        elif model_prob == 0.01:
          ring = hs.LIVE_GREEN
        elif v_sum > 0.0:
          ring = None
        else:
          ring = hs.WARN_RED

        # Same-direction traffic gets a pulling-away / closing cue relative to us.
        trend = 0
        if v >= 0.0:
          rel = v - self._carrot_v_ego
          trend = 1 if rel > 1.0 else (-1 if rel < -1.0 else 0)

        roof = self._map_to_screen(d_rel, -y_rel, road_z - 1.45)
        roof_x, roof_y = roof if roof is not None else (x, y - 60.0)
        box = self._draw_speed_tag_carrot(roof_x, roof_y, speed_text, ring, trend, d_rel)

        if self._carrot_show_radar_info >= 2 and box is not None:
          self._type.draw_mid(f"{d_rel:.1f}m  {y_rel:+.1f}", box[0], box[1] - box[2] / 2 - 20, 28, hs.TEXT_2, hs.BOLD,
                              align=0.5, shadow=True)
      elif self._carrot_show_radar_info >= 3:
        hs.dot(x, y, 7.0, hs.rgba(255, 255, 255, 200))

  def _crosses_ar_sign(self, x0: float, y0: float, x1: float, y1: float) -> bool:
    box = getattr(self, '_ar_box', None)
    if box is None:
      return False
    bx, by, bw, bh = box
    # The sign texture has transparent margins; test against its inner 80%.
    bx, by, bw, bh = bx + bw * 0.1, by + bh * 0.1, bw * 0.8, bh * 0.8
    for t in np.linspace(0.0, 1.0, 16):
      px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
      if bx <= px <= bx + bw and by <= py <= by + bh:
        return True
    return False

  def _draw_speed_tag_carrot(self, x: float, y: float, text: str, ring: "rl.Color | None", trend: int, d_rel: float):
    """Speed callout pinned above a tracked car's roof; the ring keeps the radar-state colour.

    trend 1 = pulling away (green up), -1 = closing (red down). Returns (centre x, centre y, height) or None."""
    near = d_rel < 40.0
    size = 46 if near else 40
    h = 66.0 if near else 58.0
    glyph = 0.0 if trend == 0 else h * 0.36
    text_w = self._type.width(text, size, hs.SEMI)
    inner = text_w + (glyph + 10.0 if glyph else 0.0)
    w = max(h * 1.6, inner + h * 0.8)
    tail = 10.0
    spot = hs.place_label(x, y - tail - h / 2, w, h, self._float_boxes, gap=8.0)
    if spot is None:
      return None
    tx, mid = spot
    left, bottom = tx - w / 2, mid + h / 2
    ex = min(max(x, left + h / 2), left + w - h / 2)
    if abs(tx - x) >= 1.0 and self._crosses_ar_sign(ex, bottom, x, y):
      # A leader drawn across the turn sign would read as part of it; the sign wins.
      return None
    self._float_boxes.append((tx - w / 2, mid - h / 2, w, h))
    fill = hs.CHIP_FILL
    if abs(tx - x) < 1.0:
      # Directly above: a callout tail, extended by a hairline stem if the label had to rise.
      if y - bottom > tail + 2.0:
        rl.draw_line_ex(rl.Vector2(x, bottom + tail - 1.0), rl.Vector2(x, y), 2.0, hs.rgba(255, 255, 255, 110))
      hs.triangle((x - tail * 0.7, bottom - 1.0), (x + tail * 0.7, bottom - 1.0), (x, bottom + tail * 0.7), fill)
    else:
      rl.draw_line_ex(rl.Vector2(ex, bottom - 2.0), rl.Vector2(x, y), 2.5, hs.rgba(255, 255, 255, 150))
      hs.dot(x, y, 4.0, hs.rgba(255, 255, 255, 200))
    hs.chip(left, mid - h / 2, w, h, fill=fill, ring=hs.with_alpha(ring, 230) if ring is not None else None, ring_width=3.0)
    cx = tx - inner / 2
    if glyph:
      gy, gh = mid, glyph * 0.8
      gx = cx + glyph / 2
      if trend > 0:
        hs.triangle((gx - glyph / 2, gy + gh / 2), (gx + glyph / 2, gy + gh / 2), (gx, gy - gh / 2), hs.LIVE_GREEN)
      else:
        hs.triangle((gx - glyph / 2, gy - gh / 2), (gx + glyph / 2, gy - gh / 2), (gx, gy + gh / 2), hs.WARN_RED)
      cx += glyph + 10.0
    self._type.draw_mid(text, cx, mid, size, hs.TEXT, hs.SEMI, align=0.0)
    return tx, mid, h


  def _build_path_polygon_update_line_data2_carrot(self, line: np.ndarray, width_apply: float, z_off_start: float, z_off_end: float, max_idx: int, allow_invert: bool = True) -> np.ndarray:
    points = line[:max_idx + 1]
    points = points[points[:, 0] >= 0]
    return project_path(points, width_apply, z_off_start, z_off_end,
                        self._car_space_transform, self._clip_region, allow_invert)

  def _build_path_polygon_update_line_data_dist_carrot(self, line: np.ndarray, width_apply: float, z_off_start: float, z_off_end: float, max_dist: float, allow_invert: bool = True) -> np.ndarray:
    distances = []
    dist = 2.0
    while dist < max_dist:
      distances.append(dist)
      dist = dist + dist * 0.15
    distances.append(max_dist)
    return project_path(sample_path(line, distances), width_apply, z_off_start, z_off_end,
                        self._car_space_transform, self._clip_region, allow_invert)


  def _build_path_polygon_update_line_data_dist3_carrot(self, sm, line: np.ndarray, width_apply: float, z_off_start: float, z_off_end: float, max_dist: float, allow_invert: bool = True) -> np.ndarray:
    distances = []

    car_state = sm['carState']
    v_ego_kph = float(car_state.vEgoCluster * 3.6)

    dt = min(v_ego_kph * 0.01, 0.6 if self._carrot_show_path_mode >= 10 else 1.0)
    if v_ego_kph < 1.0:
      self._carrot_pos_t = 4.0
    elif dt < 0.2:
      dt = 0.2

    self._carrot_pos_t += dt
    if self._carrot_pos_t > 24.0:
      self._carrot_pos_t -= 24.0

    draw_t = [self._carrot_pos_t]

    def add_time_points(count: int, interval: float):
      for _ in range(count):
        t = draw_t[-1]
        draw_t.append(t + interval - 24.0 if t + interval > 24.0 else t + interval)

    if self._carrot_show_path_mode == 9:
      add_time_points(1, 3.0)
      add_time_points(1, 10.0)
      add_time_points(1, 3.0)
    elif self._carrot_show_path_mode == 10:
      add_time_points(7, 3.0)
    elif self._carrot_show_path_mode == 11:
      add_time_points(5, 3.0)
    elif self._carrot_show_path_mode == 12:
      n = int(np.clip(v_ego_kph * 0.058 - 0.5, 0, 7))
      add_time_points(n, 3.0)

    def dist_function(t: float, max_distance: float) -> float:
      dist = 3.0 * pow(1.2, t)
      return max_distance if dist >= max_distance else dist

    draw_t_idx = int(np.argmin(draw_t))
    exit_flag = False
    i = 0
    while i <= len(draw_t) and not exit_flag:
      t = draw_t[draw_t_idx]
      draw_t_idx = (draw_t_idx + 1) % len(draw_t)
      if t < 3.0:
        i += 1
        continue

      dist = dist_function(t, max_dist)
      if dist == max_dist:
        exit_flag = True

      for j in range(2, -1, -1):
        dist_j = dist_function(100.0, max_dist) if exit_flag else dist_function(t - j * 1.0, max_dist)
        distances.append(dist_j)

        if exit_flag:
          break

      i += 1

    # Animated path order intentionally permits inversions, as before.
    return project_path(sample_path(line, distances), width_apply, z_off_start, z_off_end,
                        self._car_space_transform, self._clip_region)


  def _make_path_data_carrot(self, sm) -> bool:
    if not sm.valid['modelV2'] or not sm.valid['carState']:
      return False

    self._carrot_active_lane_line = sm['controlsState'].activeLaneLine

    if self._carrot_active_lane_line and sm.valid['lateralPlan']:
      model_position = np.array([
        sm['lateralPlan'].position.x,
        sm['lateralPlan'].position.y,
        sm['lateralPlan'].position.z,
      ], dtype=np.float32).T
    else:
      model_position = self._path.raw_points

    if model_position.shape[0] == 0:
      return False

    max_distance = np.clip(model_position[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_distance -= 2.0
    self._carrot_model_position = model_position
    self._carrot_max_distance = float(max_distance)

    max_idx = self._get_path_length_idx(model_position[:, 0], max_distance)
    self._carrot_long_active = sm['selfdriveState'].enabled

    if self._carrot_active_lane_line:
      self._carrot_show_path_mode = self._carrot_show_path_mode_lane
      self._carrot_show_path_color = self._carrot_show_path_color_lane
    else:
      self._carrot_show_path_mode = self._carrot_show_path_mode_normal
      self._carrot_show_path_color = self._carrot_show_path_color_normal

    if not self._carrot_long_active:
      self._carrot_show_path_color = self._carrot_show_path_color_cruise_off

    if self._carrot_show_path_mode == 0:
      self._path.projected_points = self._build_path_polygon_update_line_data2_carrot(model_position, self._carrot_show_path_width, 1.22, 1.22, max_idx)
    elif self._carrot_show_path_mode < 9 or self._carrot_show_path_mode in (13, 14, 15):
      self._path.projected_points = self._build_path_polygon_update_line_data_dist_carrot(model_position, self._carrot_show_path_width, 1.22, 1.22, max_distance, False)
    else:
      self._path.projected_points = self._build_path_polygon_update_line_data_dist3_carrot(sm, model_position, self._carrot_show_path_width, 1.22, 1.22, max_distance, False)

    return self._path.projected_points.size != 0


  def _draw_special_modes_carrot(self, mode: int, color_idx: int, brake_valid: bool):
    track_vertices = self._path.projected_points
    track_vertices_len = len(track_vertices)
    if track_vertices_len < 4:
      return

    g = 0.05
    gc = 0.4
    if mode == 13:
      g = 0.2
      gc = 0.10
    elif mode == 14:
      g = 0.45
      gc = 0.05
    elif mode == 15:
      gc = g

    glen = track_vertices_len // 2 - 1
    if glen <= 0:
      return

    xp = [[0.0] * (glen * 2) for _ in range(3)]
    yp = [[0.0] * (glen * 2) for _ in range(3)]

    for i in range(glen):
      e = track_vertices_len - i - 1
      ge = glen * 2 - 1 - i
      x1 = float(track_vertices[i][0])
      y1 = float(track_vertices[i][1])
      x2 = float(track_vertices[e][0])
      y2 = float(track_vertices[e][1])

      xp[0][i] = x1
      yp[0][i] = y1
      xp[0][ge] = x1 + (x2 - x1) * g
      yp[0][ge] = y1 + (y2 - y1) * g

      xp[1][i] = x1 + (x2 - x1) * (0.5 - gc)
      yp[1][i] = y1 + (y2 - y1) * (0.5 - gc)
      xp[1][ge] = x1 + (x2 - x1) * (0.5 + gc)
      yp[1][ge] = y1 + (y2 - y1) * (0.5 + gc)

      xp[2][i] = x1 + (x2 - x1) * (1.0 - g)
      yp[2][i] = y1 + (y2 - y1) * (1.0 - g)
      xp[2][ge] = x2
      yp[2][ge] = y2

    if mode in (13, 14):
      self._draw_polygon_from_xy_carrot(xp[0], yp[0], self._carrot_colors[color_idx % 10], brake_valid, color_idx)
    if mode in (13, 15):
      self._draw_polygon_from_xy_carrot(xp[1], yp[1], self._carrot_colors[color_idx % 10], brake_valid, color_idx)
    if mode in (13, 14):
      self._draw_polygon_from_xy_carrot(xp[2], yp[2], self._carrot_colors[color_idx % 10], brake_valid, color_idx)


  def _path_rgb_carrot(self, color_idx: int, brake_valid: bool):
    if brake_valid:
      return BRAKE_RGB
    if color_idx % 10 == 9:
      return (255, 255, 255)
    return PATH_PALETTE[color_idx % 10]

  def _draw_complex_path_carrot(self, color_idx: int, brake_valid: bool):
    """Lane edges as soft light rails over the carpet, with three quiet chevrons pointing ahead."""
    pts = self._path.projected_points
    half = len(pts) // 2
    if half < 3:
      return
    left = pts[:half]
    right = pts[::-1][:half]
    rgb = self._path_rgb_carrot(color_idx, brake_valid)
    bottom = self._rect.y + self._rect.height
    top = float(min(left[:, 1].min(), right[:, 1].min()))
    if bottom - top < 40.0:
      return

    # Rails: wide faint, medium, narrow bright; thinner with distance, gone underfoot and at the horizon.
    # All six strokes share one gradient shape, scaled per stroke by its tint: one shader pass.
    rails = []
    for side in (left, right):
      taper = 0.35 + 0.65 * np.clip((side[:, 1] - top) / (bottom - top), 0.0, 1.0)
      for width, alpha in ((18.0, 46), (9.0, 90), (4.0, 255)):
        rails.append((hs.stroke_ribbon(side, width / 2.0 * taper), rl.Color(255, 255, 255, alpha)))
    hs.vertical_gradients(rails, top, bottom, (rl.Color(*rgb, 31), rl.Color(*rgb, 230), rl.Color(*rgb, 0)), (0.0, 0.78, 1.0))

    # Chevrons at fixed screen depths, interpolated between the path's sample rows.
    mid_y = (left[:, 1] + right[:, 1]) / 2.0
    order = np.argsort(mid_y)
    ys = mid_y[order]
    for t, width, alpha in ((0.83, 12.0, 128), (0.55, 9.0, 97), (0.3, 7.0, 66)):
      target = top + (bottom - top) * t
      if target < ys[0] or target > ys[-1]:
        continue
      lx, ly = float(np.interp(target, ys, left[order, 0])), float(np.interp(target, ys, left[order, 1]))
      rx, ry = float(np.interp(target, ys, right[order, 0])), float(np.interp(target, ys, right[order, 1]))
      cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0
      hx, hy = (rx - lx) / 2.0 * 0.8, (ry - ly) / 2.0 * 0.8
      reach = math.hypot(hx, hy)
      if reach < 12.0:
        continue
      hs.polyline([(cx - hx, cy - hy + reach * 0.25), (cx, cy - reach * 0.06), (cx + hx, cy + hy + reach * 0.25)],
                  width, rl.Color(*rgb, alpha))


  def _draw_mode1_to_6_carrot(self, path_draw_seq: float, path_draw_seq2: int, mode: int, color_idx: int, brake_valid: bool):
    track_vertices = self._path.projected_points
    track_vertices_len = len(track_vertices)

    color_n = 0
    for i in range(0, track_vertices_len // 2 - 4, 2):
      x = [0.0] * 4
      y = [0.0] * 4
      x[0] = float(track_vertices[i][0]); y[0] = float(track_vertices[i][1])
      x[1] = float(track_vertices[i + 2][0]); y[1] = float(track_vertices[i + 2][1])
      x[2] = float(track_vertices[track_vertices_len - i - 3][0]); y[2] = float(track_vertices[track_vertices_len - i - 3][1])
      x[3] = float(track_vertices[track_vertices_len - i - 1][0]); y[3] = float(track_vertices[track_vertices_len - i - 1][1])

      draw_seg = (
        int(path_draw_seq) == i // 2 or
        int(path_draw_seq) == i // 2 - 2 or
        path_draw_seq2 == i // 2 or
        path_draw_seq2 == i // 2 - 2
      )
      if track_vertices_len // 2 < 8 or mode in (5, 6):
        draw_seg = True

      if draw_seg:
        idx = color_n if mode in (2, 6) else color_idx % 10
        self._draw_polygon_from_xy_carrot(x, y, self._carrot_colors[idx], brake_valid, color_idx)

      if i > 1:
        color_n += 1
        if color_n > 6:
          color_n = 0


  def _draw_mode3_to_8_carrot(self, path_draw_seq: float, path_draw_seq2: int, mode: int, color_idx: int, brake_valid: bool):
    track_vertices = self._path.projected_points
    track_vertices_len = len(track_vertices)

    color_n = 0
    for i in range(0, track_vertices_len // 2 - 4, 2):
      x = [0.0] * 6
      y = [0.0] * 6
      x[0] = float(track_vertices[i][0]); y[0] = float(track_vertices[i][1])
      x[1] = float(track_vertices[i + 2][0]); y[1] = float(track_vertices[i + 2][1])
      x[2] = (float(track_vertices[i + 4][0]) + float(track_vertices[track_vertices_len - i - 5][0])) / 2.0
      y[2] = (float(track_vertices[i + 4][1]) + float(track_vertices[track_vertices_len - i - 5][1])) / 2.0
      x[3] = float(track_vertices[track_vertices_len - i - 3][0]); y[3] = float(track_vertices[track_vertices_len - i - 3][1])
      x[4] = float(track_vertices[track_vertices_len - i - 1][0]); y[4] = float(track_vertices[track_vertices_len - i - 1][1])
      x[5] = (x[1] + x[3]) / 2.0; y[5] = (y[1] + y[3]) / 2.0

      draw_seg = (
        int(path_draw_seq) == i // 2 or
        int(path_draw_seq) == i // 2 - 2 or
        path_draw_seq2 == i // 2 or
        path_draw_seq2 == i // 2 - 2
      )
      if track_vertices_len // 2 < 8 or mode in (7, 8):
        draw_seg = True

      if draw_seg:
        idx = color_n if mode in (4, 8) else color_idx % 10
        self._draw_two_quads_from_6pts_carrot(x, y, self._carrot_colors[idx], brake_valid, color_idx)

      if i > 1:
        color_n += 1
        if color_n > 6:
          color_n = 0


  def _draw_animated_path_carrot(self, sm, mode: int, color_idx: int, brake_valid: bool):
    track_vertices = self._path.projected_points
    track_vertices_len = len(track_vertices)
    if track_vertices_len < 8:
      return

    car_state = sm['carState']
    accel = float(car_state.aEgo)
    v_ego_kph = float(car_state.vEgo * 3.6)

    seq = max(0.3, v_ego_kph / 100.0)
    forward = True
    if accel < -1.0:
      forward = False
    if accel > -0.5:
      forward = True

    max_seq = min(track_vertices_len // 4 + 3, 16)

    if not hasattr(self, "_carrot_path_draw_seq2"):
      self._carrot_path_draw_seq2 = -1

    if forward:
      self._carrot_path_draw_seq += seq
      if self._carrot_path_draw_seq > max_seq:
        self._carrot_path_draw_seq = self._carrot_path_draw_seq2 if self._carrot_path_draw_seq2 >= 0 else 0.0
    else:
      self._carrot_path_draw_seq -= seq
      if self._carrot_path_draw_seq < 0.0:
        self._carrot_path_draw_seq = self._carrot_path_draw_seq2 if self._carrot_path_draw_seq2 >= 0 else float(max_seq)

    self._carrot_path_draw_seq2 = ((int(self._carrot_path_draw_seq) - max_seq // 2 + max_seq) % max_seq) if max_seq > 15 else -5

    if mode in (1, 2, 5, 6):
      self._draw_mode1_to_6_carrot(self._carrot_path_draw_seq, self._carrot_path_draw_seq2, mode, color_idx, brake_valid)
    else:
      self._draw_mode3_to_8_carrot(self._carrot_path_draw_seq, self._carrot_path_draw_seq2, mode, color_idx, brake_valid)


  def _draw_path_carrot(self, sm):
    self._refresh_carrot_params(rl.get_time())
    if not self._render_diagnostics.call('path_geometry', self._make_path_data_carrot, sm):
      return

    self._update_path_end_carrot(sm)

    car_state = sm['carState']
    temp = int(car_state.useLaneLineSpeed)
    if temp != self._carrot_use_lane_line_speed_apply:
      self._carrot_use_lane_line_speed_apply = temp

    brake_valid = car_state.brakeLights
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state is not None else None
    lp = sm['longitudinalPlan']
    accel = lp.accels[0] if len(lp.accels) > 0 else 0.0

    show_path_color = self._carrot_show_path_color
    if show_path_color >= 20:
      if self._carrot_long_active:
        show_path_color = 13
        if lead_one is not None and lead_one.status:
          if abs(accel) < 0.5:
            show_path_color = 12
          elif accel >= 0.5:
            show_path_color = 11
          else:
            show_path_color = 10
      else:
        show_path_color = 19

    show_path_mode = self._carrot_show_path_mode

    # Mode 0 is the carpet itself; every other mode draws its pieces on top of it.
    self._draw_path_carpet_carrot(show_path_mode, show_path_color, brake_valid)
    if show_path_mode == 0:
      pass
    elif 13 <= show_path_mode <= 15:
      self._draw_special_modes_carrot(show_path_mode, show_path_color, brake_valid)
    elif show_path_mode >= 9:
      self._draw_complex_path_carrot(show_path_color, brake_valid)
    else:
      self._draw_animated_path_carrot(sm, show_path_mode, show_path_color, brake_valid)

    self._render_diagnostics.call('path_labels', self._draw_path_end_overlay_carrot)
    self._render_diagnostics.call('tire', self._draw_tire_trajectory_carrot, sm)

  @staticmethod
  def _tire_grad_color(d: float, a: int) -> rl.Color:
    # danger 0(중앙)->1(차선밟음): 초록 -> 노랑 -> 빨강 멀티스톱
    d = max(0.0, min(1.5, d))
    if d <= 0.10:
      r, g = 0.0, 200.0
    elif d <= 0.60:
      f = (d - 0.10) / 0.50
      r, g = f * 255.0, 200.0
    elif d <= 1.00:
      f = (d - 0.60) / 0.40
      r, g = 255.0, 200.0 - f * 200.0
    else:
      r, g = 255.0, 0.0
    return rl.Color(int(r), int(g), 0, a)

  def _draw_tire_trajectory_carrot(self, sm):
    """타이어 궤적: 모델 예측 경로 위에 차폭(±0.9m) 타이어 트랙을 그라데이션 밴드로 그리고,
    차선 중앙 대비 편차(danger_ratio)에 따라 초록->빨강으로 색을 바꾼다. 이탈 임박 시 펄스.
    하단에 L/R/C 방향 + 편차(m) 표시. CarrotTireTrajectory 토글로만 노출된다."""
    if self._carrot_tire_trajectory <= 0:
      return
    if not sm.valid['modelV2']:
      return

    model = sm['modelV2']
    lane_lines = model.laneLines
    probs = model.laneLineProbs
    if len(lane_lines) < 3 or len(probs) < 3:
      return

    left_y, right_y = lane_lines[1].y, lane_lines[2].y
    if len(left_y) == 0 or len(right_y) == 0:
      return

    # 차량 중심에서 좌/우 차선까지 거리(m). openpilot 모델 +Y=좌측 이므로 우측은 부호 반전.
    left_m = float(left_y[0])
    right_m = -float(right_y[0])
    # 한쪽 차선 인식이 약하면 기본 차선폭(3.0m)으로 반대쪽 추정.
    if probs[1] < 0.3 and probs[2] > 0.3:
      left_m = 3.0 - right_m
    elif probs[2] < 0.3 and probs[1] > 0.3:
      right_m = 3.0 - left_m
    total = left_m + right_m
    if total < 0.5:
      total = 3.0

    # drift>0 => 차량이 좌측 차선쪽으로 치우침(좌측 타이어가 좌측 차선에 근접).
    drift = (right_m - left_m) / 2.0
    limit = max(0.1, total / 2.0 - 0.9)   # 타이어(차폭 절반 0.9m)가 차선을 밟는 지점을 1.0
    danger = abs(drift) / limit

    if drift > 0.02:
      dir_label = "L"
    elif drift < -0.02:
      dir_label = "R"
    else:
      dir_label = "C"
    dist_str = f"{abs(drift):.2f}"

    # 모델 예측 주행경로. 선행차가 있으면 그 거리까지만, 없으면 최대 40m.
    path = self._path.raw_points
    if path.shape[0] < 5:
      return
    max_dist = 40.0
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    if radar_state is not None and radar_state.leadOne.status:
      d_rel = float(radar_state.leadOne.dRel)
      if 3.0 < d_rel < max_dist:
        max_dist = d_rel
    max_idx = self._get_path_length_idx(path[:, 0], max_dist)
    if max_idx < 5:
      return

    # 색상: 치우친 쪽 = danger 그라데이션, 반대쪽 = 초록. 심한 이탈 시 좌/우 밴드 펄스.
    green = rl.Color(0, 200, 0, 255)
    warn = abs(drift) >= 0.5
    pulse = warn and (int(rl.get_time() * 1000) % 800 < 400)
    if warn:
      side = rl.Color(255, 0, 0, 255) if pulse else rl.Color(255, 100, 0, 220)
      color_left = side if drift > 0.0 else green
      color_right = side if drift < 0.0 else green
    elif drift > 0.02:
      color_left, color_right = self._tire_grad_color(danger, 255), green
    elif drift < -0.02:
      color_left, color_right = green, self._tire_grad_color(danger, 255)
    else:
      color_left = color_right = green

    # 타이어 트랙 밴드: 경로중심 ±0.9m 위치에 얇은 리본. 원근투영으로 폭은 자동 축소.
    # 화면 좌측 타이어 = +Y(y_shift=+0.9), 우측 = -Y. 아래(가까움)=진하게 위(멀리)=투명.
    band_half = 0.16
    for y_shift, col in ((0.9, color_left), (-0.9, color_right)):
      ribbon = self._map_line_to_polygon(path, band_half, 1.22, max_idx, max_dist, True, y_shift)
      if ribbon.size == 0:
        continue
      grad = Gradient(
        start=(0.0, 1.0),  # 경로 하단(차량 근처)
        end=(0.0, 0.0),    # 경로 상단(원거리)
        colors=[rl.Color(col.r, col.g, col.b, col.a), rl.Color(col.r, col.g, col.b, 0)],
        stops=[0.0, 1.0],
      )
      draw_polygon(self._rect, ribbon, gradient=grad)

    # 하단 L/R/C + 편차(m) 텍스트 (경로 시작점 부근에 투영).
    if danger < 1.5:
      base = self._map_to_screen(3.0, float(path[0, 1]), float(path[0, 2]) + self._path_offset_z)
      if base is not None:
        tx, ty = base
        white = rl.Color(255, 255, 255, 255)
        draw_text_ui_style(dir_label, tx, ty - 56.0, 72, white, font=self._font_display, align="center", y_offset=0.0)
        draw_text_ui_style(dist_str, tx, ty + 18.0, 52, white, font=self._font_display, align="center", y_offset=0.0)
