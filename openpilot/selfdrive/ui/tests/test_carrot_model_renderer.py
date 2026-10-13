import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.ui.onroad import model_renderer

SOURCE_NOW = 10_000_000_000


class FakeSubMaster(dict):
  def __init__(self, **messages):
    super().__init__(messages)
    self.valid = dict.fromkeys(messages, True)


def source_payload(*, left_oem=False, left_vision=False, right_oem=False, right_vision=False):
  return json.dumps({
    "type": "xiaogeBlindspotSources",
    "version": 1,
    "receivedMonoTimeNanos": SOURCE_NOW,
    "left": {"oem": left_oem, "vision": left_vision},
    "right": {"oem": right_oem, "vision": right_vision},
  }).encode()


def blind_spot_messages(
  *,
  left_blindspot: bool = False,
  right_blindspot: bool = False,
  left_status: bool = False,
  right_status: bool = False,
  left_d_rel: float = 100.0,
  right_d_rel: float = 100.0,
  v_ego: float = 10.0,
  lane_change_state=None,
  lane_change_direction: str = "none",
):
  if lane_change_state is None:
    lane_change_state = model_renderer.LaneChangeState.off
  car_state = SimpleNamespace(
    leftBlindspot=left_blindspot,
    rightBlindspot=right_blindspot,
    vEgo=v_ego,
  )
  radar_state = SimpleNamespace(
    leadLeft=SimpleNamespace(status=left_status, dRel=left_d_rel),
    leadRight=SimpleNamespace(status=right_status, dRel=right_d_rel),
  )
  meta = SimpleNamespace(
    laneChangeState=lane_change_state,
    laneChangeDirection=lane_change_direction,
  )
  return car_state, radar_state, meta


@pytest.mark.parametrize(
  "messages, expected",
  [
    (blind_spot_messages(), (False, False, False, False)),
    (blind_spot_messages(left_blindspot=True), (True, False, False, False)),
    (blind_spot_messages(right_blindspot=True), (False, True, False, False)),
    (
      blind_spot_messages(
        left_status=True,
        left_d_rel=29.9,
        lane_change_state=model_renderer.LaneChangeState.preLaneChange,
        lane_change_direction="left",
      ),
      (False, False, True, False),
    ),
    (
      blind_spot_messages(
        right_status=True,
        right_d_rel=30.0,
        lane_change_state=model_renderer.LaneChangeState.preLaneChange,
        lane_change_direction="right",
      ),
      (False, False, False, False),
    ),
    (
      blind_spot_messages(
        left_blindspot=True,
        left_status=True,
        left_d_rel=5.0,
        lane_change_state=model_renderer.LaneChangeState.preLaneChange,
        lane_change_direction="left",
      ),
      (True, False, False, False),
    ),
  ],
)
def test_blind_spot_draw_state(messages, expected):
  assert model_renderer.ModelRenderer._blind_spot_draw_state_carrot(*messages) == expected


def test_blind_spot_inactive_skips_barrier_update():
  renderer = object.__new__(model_renderer.ModelRenderer)
  car_state, radar_state, meta = blind_spot_messages()
  sm = FakeSubMaster(modelV2=SimpleNamespace(meta=meta), carState=car_state, radarState=radar_state)
  updates = []
  renderer._update_blind_spot_barriers_carrot = lambda *args, **kwargs: updates.append((args, kwargs))

  renderer._draw_blind_spot_carrot(sm)

  assert updates == []


def test_blind_spot_does_not_require_radar_for_confirmed_warning():
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_lane_barrier_vertices = [np.ones((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32)]
  car_state, radar_state, meta = blind_spot_messages(left_blindspot=True)
  sm = FakeSubMaster(modelV2=SimpleNamespace(meta=meta), carState=car_state, radarState=radar_state)
  sm.valid["radarState"] = False
  updates = []
  renderer._update_blind_spot_barriers_carrot = lambda *args, **kwargs: updates.append((args, kwargs))
  renderer._draw_blind_spot_segments_carrot = lambda *_args: None

  renderer._draw_blind_spot_carrot(sm)

  assert len(updates) == 1
  assert updates[0][1] == {"update_left": True, "update_right": False}


def test_blind_spot_updates_only_active_side():
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_lane_barrier_vertices = [np.ones((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32)]
  car_state, radar_state, meta = blind_spot_messages(left_blindspot=True)
  sm = FakeSubMaster(modelV2=SimpleNamespace(meta=meta), carState=car_state, radarState=radar_state)
  updates = []
  draws = []
  renderer._update_blind_spot_barriers_carrot = lambda _, **kwargs: updates.append(kwargs)
  renderer._draw_blind_spot_segments_carrot = lambda points, color: draws.append((points, color))

  renderer._draw_blind_spot_carrot(sm)

  assert updates == [{"update_left": True, "update_right": False}]
  assert len(draws) == 1
  assert draws[0][0] is renderer._carrot_lane_barrier_vertices[0]


@pytest.mark.parametrize(
  "messages,sources,expected_rgb",
  [
    (blind_spot_messages(left_blindspot=True), source_payload(left_oem=True), (255, 179, 36)),
    (blind_spot_messages(left_blindspot=True), source_payload(left_vision=True), (26, 136, 255)),
    (blind_spot_messages(left_blindspot=True), source_payload(left_oem=True, left_vision=True), (255, 72, 64)),
  ],
)
def test_blind_spot_wall_color_reflects_detection_source(monkeypatch, messages, sources, expected_rgb):
  monkeypatch.setattr(model_renderer.time, "monotonic_ns", lambda: SOURCE_NOW)
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_lane_barrier_vertices = [np.ones((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32)]
  car_state, radar_state, meta = messages
  sm = FakeSubMaster(modelV2=SimpleNamespace(meta=meta), carState=car_state, radarState=radar_state,
                     customReservedRawData1=sources)
  draws = []
  renderer._update_blind_spot_barriers_carrot = lambda *_args, **_kwargs: None
  renderer._draw_blind_spot_segments_carrot = lambda _points, color: draws.append((color.r, color.g, color.b))

  renderer._draw_blind_spot_carrot(sm)

  assert draws == [expected_rgb]


def test_blind_spot_wall_colors_are_independent_by_side(monkeypatch):
  monkeypatch.setattr(model_renderer.time, "monotonic_ns", lambda: SOURCE_NOW)
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_lane_barrier_vertices = [np.zeros((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32)]
  car_state, radar_state, meta = blind_spot_messages(
    left_blindspot=True, right_blindspot=True,
  )
  sm = FakeSubMaster(
    modelV2=SimpleNamespace(meta=meta), carState=car_state, radarState=radar_state,
    customReservedRawData1=source_payload(left_vision=True, right_oem=True, right_vision=True),
  )
  draws = []
  renderer._update_blind_spot_barriers_carrot = lambda *_args, **_kwargs: None
  renderer._draw_blind_spot_segments_carrot = lambda points, color: draws.append(
    (points, (color.r, color.g, color.b))
  )

  renderer._draw_blind_spot_carrot(sm)

  assert draws == [
    (renderer._carrot_lane_barrier_vertices[0], (26, 136, 255)),
    (renderer._carrot_lane_barrier_vertices[1], (255, 72, 64)),
  ]


def test_blind_spot_wall_color_prefers_carstate_source_fields_over_stale_packet(monkeypatch):
  monkeypatch.setattr(model_renderer.time, "monotonic_ns", lambda: SOURCE_NOW)
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_lane_barrier_vertices = [np.ones((4, 2), dtype=np.float32), np.ones((4, 2), dtype=np.float32)]
  car_state, radar_state, meta = blind_spot_messages(left_blindspot=True)
  car_state.leftBlindspotOem = True
  car_state.leftBlindspotOnnx = True
  car_state.blindspotSplitSourcesValid = True
  stale_sources = source_payload(left_oem=False, left_vision=False)
  sm = FakeSubMaster(modelV2=SimpleNamespace(meta=meta), carState=car_state, radarState=radar_state,
                     customReservedRawData1=stale_sources)
  sm.valid["customReservedRawData1"] = False
  draws = []
  renderer._update_blind_spot_barriers_carrot = lambda *_args, **_kwargs: None
  renderer._draw_blind_spot_segments_carrot = lambda _points, color: draws.append((color.r, color.g, color.b))

  renderer._draw_blind_spot_carrot(sm)

  assert draws == [(255, 72, 64)]


class FakeParams:
  def __init__(self):
    self.values = {
      "ShowLaneInfo": 1,
      "ShowRadarInfo": 0,
      "ShowPathMode": 9,
      "ShowPathColor": 20,
      "ShowPathModeLane": 14,
      "ShowPathColorLane": 20,
      "ShowPathColorCruiseOff": 19,
      "CarrotTireTrajectory": 1,
    }
    self.calls = []

  def get_int(self, key):
    self.calls.append(key)
    return int(self.values[key])


def test_carrot_params_refresh_is_throttled(monkeypatch):
  params = FakeParams()
  monkeypatch.setattr(model_renderer.ui_state, "params", params)
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_params_next_refresh_time = 0.0

  renderer._refresh_carrot_params(0.0)
  assert len(params.calls) == len(params.values)
  assert renderer._carrot_show_path_mode_normal == 9

  params.values["ShowPathMode"] = 13
  renderer._refresh_carrot_params(0.999)
  assert len(params.calls) == len(params.values)
  assert renderer._carrot_show_path_mode_normal == 9

  renderer._refresh_carrot_params(1.0)
  assert len(params.calls) == 2 * len(params.values)
  assert renderer._carrot_show_path_mode_normal == 13


def test_carrot_params_failed_refresh_retries(monkeypatch):
  params = FakeParams()
  original_get_int = params.get_int
  failed = False

  def fail_once(key):
    nonlocal failed
    if key == "ShowPathColor" and not failed:
      failed = True
      raise OSError("temporary params read failure")
    return original_get_int(key)

  params.get_int = fail_once
  monkeypatch.setattr(model_renderer.ui_state, "params", params)
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_params_next_refresh_time = 0.0

  with pytest.raises(OSError):
    renderer._refresh_carrot_params(0.0)
  assert renderer._carrot_params_next_refresh_time == 0.0

  renderer._refresh_carrot_params(0.1)
  assert renderer._carrot_params_next_refresh_time == pytest.approx(1.1)


def test_lane_draw_reuses_cached_raw_points():
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_show_lane_info = 1
  renderer._lane_line_probs = np.ones(4, dtype=np.float32)
  renderer._road_edge_stds = np.zeros(2, dtype=np.float32)
  renderer._lane_lines = [
    model_renderer.ModelPoints(raw_points=np.array([[0.0, i, 0.0], [20.0, i, 0.0]], dtype=np.float32))
    for i in range(4)
  ]
  renderer._road_edges = [
    model_renderer.ModelPoints(raw_points=np.array([[0.0, i, 0.0], [20.0, i, 0.0]], dtype=np.float32))
    for i in range(2)
  ]
  renderer._get_path_length_idx = lambda *_: 1
  renderer._rect = model_renderer.rl.Rectangle(0.0, 0.0, 2160.0, 1080.0)
  projected_inputs = []

  def project(line, *_args, **_kwargs):
    projected_inputs.append(line)
    return np.empty((0, 2), dtype=np.float32)

  renderer._map_line_to_polygon = project
  car_state = SimpleNamespace(leftLaneLine=1, rightLaneLine=1)

  class LaneSubMaster(FakeSubMaster):
    def __getitem__(self, key):
      if key == "modelV2":
        raise AssertionError("lane draw must reuse cached model arrays")
      return super().__getitem__(key)

  sm = LaneSubMaster(modelV2=object(), carState=car_state)
  renderer._draw_lane_lines_carrot(sm)

  expected = [
    renderer._lane_lines[0].raw_points,
    renderer._lane_lines[1].raw_points,
    renderer._lane_lines[2].raw_points,
    renderer._lane_lines[3].raw_points,
  ]
  assert len(projected_inputs) == len(expected)
  assert all(actual is cached for actual, cached in zip(projected_inputs, expected, strict=True))


def test_lane_draw_builds_cached_optional_geometry():
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._carrot_show_lane_info = 2
  renderer._lane_line_probs = np.ones(4, dtype=np.float32)
  renderer._road_edge_stds = np.zeros(2, dtype=np.float32)
  renderer._lane_lines = [
    model_renderer.ModelPoints(raw_points=np.array([[0.0, i, 0.0], [20.0, i, 0.0]], dtype=np.float32))
    for i in range(4)
  ]
  renderer._road_edges = [
    model_renderer.ModelPoints(raw_points=np.array([[0.0, i, 0.0], [20.0, i, 0.0]], dtype=np.float32))
    for i in range(2)
  ]
  renderer._get_path_length_idx = lambda *_: 1
  renderer._rect = model_renderer.rl.Rectangle(0.0, 0.0, 2160.0, 1080.0)
  projected_inputs = []

  def project(line, *_args, **_kwargs):
    projected_inputs.append(line)
    return np.empty((0, 2), dtype=np.float32)

  renderer._map_line_to_polygon = project
  car_state = SimpleNamespace(leftLaneLine=24, rightLaneLine=1)

  class LaneSubMaster(FakeSubMaster):
    def __getitem__(self, key):
      if key == "modelV2":
        raise AssertionError("lane draw must reuse cached model arrays")
      return super().__getitem__(key)

  sm = LaneSubMaster(modelV2=object(), carState=car_state)
  renderer._draw_lane_lines_carrot(sm)

  expected = [
    renderer._lane_lines[0].raw_points,
    renderer._lane_lines[1].raw_points,
    renderer._lane_lines[1].raw_points,
    renderer._lane_lines[2].raw_points,
    renderer._lane_lines[3].raw_points,
    renderer._road_edges[0].raw_points,
    renderer._road_edges[1].raw_points,
  ]
  assert len(projected_inputs) == len(expected)
  assert all(actual is cached for actual, cached in zip(projected_inputs, expected, strict=True))


def test_model_overlay_draw_order():
  renderer = object.__new__(model_renderer.ModelRenderer)
  calls = []
  renderer._draw_path_carrot = lambda _: calls.append("path")
  renderer._draw_lane_lines_carrot = lambda _: calls.append("laneLines")
  renderer._draw_ar_turn_carrot = lambda _: calls.append("arTurn")
  renderer._draw_blind_spot_carrot = lambda _: calls.append("blindSpot")
  renderer._draw_radar_info_carrot = lambda _: calls.append("radar")

  renderer._draw_carrot_overlays(object())

  assert calls == ["path", "laneLines", "arTurn", "blindSpot", "radar"]


def test_render_stale_data_skips_overlays(monkeypatch):
  renderer = object.__new__(model_renderer.ModelRenderer)
  calls = []
  renderer._draw_carrot_overlays = lambda _: calls.append("overlays")
  sm = SimpleNamespace(recv_frame={"liveCalibration": 0, "modelV2": 0})
  monkeypatch.setattr(model_renderer.ui_state, "sm", sm, raising=False)
  monkeypatch.setattr(model_renderer.ui_state, "started_frame", 1, raising=False)

  renderer._render(object())

  assert calls == []


AR_ASSETS = Path(__file__).parents[2] / "assets" / "images" / "nav"


def png_size(name: str) -> tuple[int, int]:
  with open(AR_ASSETS / f"{name}.png", "rb") as f:
    header = f.read(24)
  assert header[:8] == b"\x89PNG\r\n\x1a\n"
  return struct.unpack(">II", header[16:24])


def ar_submaster(*, turn=2, dist=15.0, lead=True, d_rel=12.0, y_rel=-0.6, route=True):
  carrot_man = SimpleNamespace(
    xTurnInfo=turn,
    xDistToTurn=dist,
    nGoPosDist=100 if route else 0,
    nGoPosTime=10,
    atcType="",
  )
  lead_one = SimpleNamespace(status=lead, dRel=d_rel, yRel=y_rel)
  return FakeSubMaster(carrotMan=carrot_man, radarState=SimpleNamespace(leadOne=lead_one))


def ar_renderer(monkeypatch):
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._path = SimpleNamespace(raw_points=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32))
  renderer._path_offset_z = 0.0
  renderer._float_boxes = []

  def project(pts):
    pts = np.asarray(pts, dtype=np.float64)
    return np.column_stack((960.0 + 1000.0 * pts[:, 1] / pts[:, 0], 540.0 + 1000.0 * pts[:, 2] / pts[:, 0]))

  def texture(name):
    width, height = png_size(name)
    return SimpleNamespace(width=width, height=height, id=0)

  monkeypatch.setattr(renderer, "_project", project)
  monkeypatch.setattr(model_renderer.hs, "_texture", texture)
  monkeypatch.setattr(model_renderer.hs, "_blit", lambda *args, **kwargs: None)
  monkeypatch.setattr(model_renderer.rl, "draw_ellipse", lambda *args, **kwargs: None)
  return renderer


def ar_sign_box_for(renderer, turn, dist, lateral):
  pts = renderer._project(np.array([[dist, lateral, 0.0], [dist, lateral, -renderer.AR_HEIGHT_M],
                                    [dist, lateral - 1.1, 0.0], [dist, lateral + 1.1, 0.0]]))
  foot_x, foot_y = (float(v) for v in pts[0])
  height = min(renderer.AR_MAX_PX, max(renderer.AR_MIN_PX, foot_y - float(pts[1, 1])))
  return model_renderer.hs.ar_sign_box(turn, foot_x, foot_y, height)


def test_boxes_overlap_true_false_and_edge_gap():
  overlap = model_renderer.ModelRenderer._boxes_overlap
  a = (0.0, 0.0, 100.0, 50.0)

  assert overlap(a, (60.0, 20.0, 100.0, 50.0), 0.0)
  assert overlap((60.0, 20.0, 100.0, 50.0), a, 0.0)

  right = (120.0, 20.0, 100.0, 50.0)
  assert not overlap(a, right, 0.0)
  assert not overlap(a, right, 20.0)
  assert overlap(a, right, 21.0)

  below = (0.0, 60.0, 100.0, 50.0)
  assert not overlap(a, below, 0.0)
  assert not overlap(a, below, 10.0)
  assert overlap(a, below, 11.0)


def test_ar_turn_sign_pushes_toward_turn_side_until_lead_clears(monkeypatch):
  renderer = ar_renderer(monkeypatch)
  sm = ar_submaster()

  lead = renderer._ar_lead_box(sm, renderer._path.raw_points)
  assert lead is not None
  unpushed = ar_sign_box_for(renderer, turn=2, dist=15.0, lateral=renderer.AR_LATERAL)
  assert renderer._boxes_overlap(unpushed, lead, renderer.AR_LEAD_GAP)

  for _ in range(10):
    renderer._draw_ar_turn_carrot(sm)

  assert renderer._ar_push == pytest.approx(1.0)
  assert renderer._ar_box is not None
  assert not renderer._boxes_overlap(renderer._ar_box, lead, 0.0)
  # The right-turn sign (side +1) ends clear of the lead's right edge.
  assert renderer._ar_box[0] > lead[0] + lead[2]


def test_ar_turn_sign_without_lead_keeps_the_old_placement(monkeypatch):
  renderer = ar_renderer(monkeypatch)
  sm = ar_submaster(lead=False)
  expected = ar_sign_box_for(renderer, turn=2, dist=15.0, lateral=renderer.AR_LATERAL)

  renderer._draw_ar_turn_carrot(sm)

  assert renderer._ar_push == 0.0
  assert renderer._ar_box == pytest.approx(expected)
  assert len(renderer._float_boxes) == 1
  assert renderer._float_boxes[0] == pytest.approx(expected)


def test_ar_push_is_cleared_when_the_turn_state_disappears(monkeypatch):
  renderer = ar_renderer(monkeypatch)
  renderer._draw_ar_turn_carrot(ar_submaster())
  assert renderer._ar_push > 0.0
  assert "_ar_push" in renderer.__dict__

  renderer._draw_ar_turn_carrot(ar_submaster(route=False))

  assert "_ar_push" not in renderer.__dict__


def lead_renderer(**overrides):
  renderer = object.__new__(model_renderer.ModelRenderer)
  defaults = {
    "_carrot_soft_hold_active": False,
    "_carrot_brake_hold_active": False,
    "_carrot_carrot_cruise": False,
    "_carrot_long_active": False,
    "_carrot_x_state": 0,
    "_carrot_radar_dist": 0.0,
    "_carrot_vision_dist": 0.0,
    "_carrot_radar_track_id": -1,
    "_carrot_lead_status": False,
    "_carrot_lead_speed": 0.0,
    "_carrot_lead_closing": False,
  }
  for key, value in {**defaults, **overrides}.items():
    setattr(renderer, key, value)
  return renderer


def test_lead_runs_are_speed_then_quieter_gap_without_trend(monkeypatch):
  monkeypatch.setattr(model_renderer.ui_state, "is_metric", True)
  renderer = lead_renderer(
    _carrot_radar_dist=12.5,
    _carrot_vision_dist=14.3,
    _carrot_radar_track_id=0,
    _carrot_lead_status=True,
    _carrot_lead_speed=30.0,
    _carrot_lead_closing=True,
  )

  # No source dots, no vision cross-check and no relative-speed arrow: the lead speed leads, then the radar gap.
  assert renderer._lead_runs_carrot() == [
    ("num", "30", model_renderer.LEAD_SIZE, model_renderer.hs.TEXT),
    ("unit", "km/h", model_renderer.LEAD_UNIT_SIZE, model_renderer.hs.TEXT_3),
    ("bar",),
    ("num", "12", model_renderer.LEAD_GAP_SIZE, model_renderer.hs.TEXT),
    ("unit", "m", model_renderer.LEAD_UNIT_SIZE, model_renderer.hs.TEXT_3),
  ]


@pytest.mark.parametrize("run, weight", [
  (("num", "80", model_renderer.LEAD_SIZE, model_renderer.hs.TEXT), model_renderer.hs.BOLD),
  (("num", "49", model_renderer.LEAD_GAP_SIZE, model_renderer.hs.TEXT), model_renderer.hs.SEMI),
  (("unit", "m", model_renderer.LEAD_UNIT_SIZE, model_renderer.hs.TEXT_3), model_renderer.hs.SEMI),
])
def test_only_the_leading_figure_is_bold(run, weight):
  assert model_renderer.ModelRenderer._lead_run_weight(run) is weight


def test_lead_runs_vision_only_shows_vision_distance_with_tenths_up_close(monkeypatch):
  monkeypatch.setattr(model_renderer.ui_state, "is_metric", True)
  renderer = lead_renderer(_carrot_vision_dist=8.7)

  assert renderer._lead_runs_carrot() == [
    ("num", "8.7", model_renderer.LEAD_SIZE, model_renderer.hs.TEXT),
    ("unit", "m", model_renderer.LEAD_UNIT_SIZE, model_renderer.hs.TEXT_3),
  ]


@pytest.mark.parametrize("dist, text", [(9.94, "9.9"), (9.96, "10"), (10.0, "10"), (49.6, "50")])
def test_lead_distance_uses_tenths_only_below_ten_metres(monkeypatch, dist, text):
  monkeypatch.setattr(model_renderer.ui_state, "is_metric", True)
  renderer = lead_renderer(_carrot_radar_dist=dist, _carrot_radar_track_id=7)

  runs = renderer._lead_runs_carrot()

  assert runs[0] == ("num", text, model_renderer.LEAD_SIZE, model_renderer.hs.TEXT)
  assert not any(run[0] == "dot" for run in runs)


@pytest.mark.parametrize("track_id, closing, colour", [
  (0, False, model_renderer.LEAD_RADAR),
  (7, False, model_renderer.LEAD_TRACK),
  (-1, False, model_renderer.LEAD_BLUE),
  (0, True, model_renderer.LEAD_RED),
  (-1, True, model_renderer.LEAD_RED),
])
def test_lead_crescent_neutral_radar_amber_track_blue_camera_red_only_when_closing(track_id, closing, colour):
  renderer = lead_renderer(_carrot_radar_track_id=track_id, _carrot_lead_closing=closing)

  assert renderer._lead_stroke_carrot() is colour


def test_lead_closing_needs_short_time_to_contact_and_holds_with_hysteresis():
  closing_next = model_renderer.ModelRenderer._lead_closing_next

  # Standing or pulling away never reads as closing, nor does a crawl below the minimum closing speed.
  assert not closing_next(False, 5.0, 0.0)
  assert not closing_next(False, 5.0, 2.0)
  assert not closing_next(True, 2.0, -0.9)
  # 20 m at 8 m/s is 2.5 s to contact: on. 30 m at 8 m/s is 3.75 s: off when idle, held once on.
  assert closing_next(False, 20.0, -8.0)
  assert not closing_next(False, 30.0, -8.0)
  assert closing_next(True, 30.0, -8.0)
  # Beyond the release threshold it clears.
  assert not closing_next(True, 40.0, -8.0)


def speed_tag_renderer():
  renderer = object.__new__(model_renderer.ModelRenderer)
  renderer._float_boxes = []
  renderer._type = SimpleNamespace(
    width=lambda text, size, weight: 10.0,
    draw_mid=lambda *args, **kwargs: None,
  )
  return renderer


def test_speed_tag_drops_when_label_lands_beyond_reach(monkeypatch):
  renderer = speed_tag_renderer()
  placed = []
  monkeypatch.setattr(
    model_renderer.hs,
    "place_label",
    lambda x, y, w, h, claimed, gap=0.0: placed.append((x, y, w, h)) or (x + 300.0, y),
  )
  monkeypatch.setattr(model_renderer.hs, "chip", lambda *args, **kwargs: None)
  monkeypatch.setattr(model_renderer.hs, "triangle", lambda *args, **kwargs: None)
  monkeypatch.setattr(model_renderer.hs, "dot", lambda *args, **kwargs: None)
  monkeypatch.setattr(model_renderer.rl, "draw_line_ex", lambda *args, **kwargs: None)

  result = renderer._draw_speed_tag_carrot(960.0, 540.0, "80", None, 60.0)

  assert result is None
  assert renderer._float_boxes == []
  assert len(placed) == 1


def test_speed_tag_keeps_label_within_reach(monkeypatch):
  renderer = speed_tag_renderer()
  spot = (960.0 + 120.0, 501.0)
  monkeypatch.setattr(model_renderer.hs, "place_label", lambda *args, **kwargs: spot)
  monkeypatch.setattr(model_renderer.hs, "chip", lambda *args, **kwargs: None)
  monkeypatch.setattr(model_renderer.hs, "triangle", lambda *args, **kwargs: None)
  monkeypatch.setattr(model_renderer.hs, "dot", lambda *args, **kwargs: None)
  monkeypatch.setattr(model_renderer.rl, "draw_line_ex", lambda *args, **kwargs: None)

  result = renderer._draw_speed_tag_carrot(960.0, 540.0, "80", None, 60.0)

  assert result is not None
  assert result[:2] == spot
  assert len(renderer._float_boxes) == 1


@pytest.mark.parametrize("track, expected", [
  (SimpleNamespace(radarTrackId=7, dRel=80.0, yRel=3.5), True),   # the lead's own radar track, wherever it is
  (SimpleNamespace(radarTrackId=3, dRel=49.6, yRel=0.4), True),   # sits where the lead is
  (SimpleNamespace(radarTrackId=3, dRel=49.6, yRel=3.4), False),  # next lane at the same distance
  (SimpleNamespace(radarTrackId=3, dRel=70.0, yRel=0.0), False),  # further up the same lane
])
def test_the_lead_track_gets_no_second_speed_tag(track, expected):
  lead_one = SimpleNamespace(radarTrackId=7, dRel=49.0, yRel=0.0, status=True)

  assert model_renderer.ModelRenderer._is_lead_one_track(track, lead_one) is expected


def test_a_stock_radar_lead_matches_its_track_by_position():
  lead_one = SimpleNamespace(radarTrackId=0, dRel=49.0, yRel=0.0, status=True)

  assert model_renderer.ModelRenderer._is_lead_one_track(SimpleNamespace(radarTrackId=12, dRel=50.5, yRel=-0.6), lead_one)
  assert not model_renderer.ModelRenderer._is_lead_one_track(SimpleNamespace(radarTrackId=0, dRel=20.0, yRel=0.0), lead_one)
