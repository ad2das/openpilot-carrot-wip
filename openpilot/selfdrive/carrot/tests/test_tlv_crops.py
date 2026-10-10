"""Tests for TLV crop geometry and NV12 conversion (PC, numpy + cv2 reference)."""
import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import crops

W, H = 1928, 1208


def test_focal_scaling():
  assert crops.focal_px_for_width(1928) == pytest.approx(2648.0)
  assert crops.focal_px_for_width(964) == pytest.approx(1324.0)


def test_horizon_row_centre_without_calibration():
  assert crops.horizon_row(None, 2648.0, W, H) == pytest.approx(H / 2)
  assert crops.horizon_row([0.0, 0.0, 0.0], 2648.0, W, H) == pytest.approx(H / 2)
  assert crops.horizon_row([0.0, 0.0], 2648.0, W, H) == pytest.approx(H / 2)


def test_horizon_row_responds_to_pitch():
  up = crops.horizon_row([0.0, 0.02, 0.0], 2648.0, W, H)
  down = crops.horizon_row([0.0, -0.02, 0.0], 2648.0, W, H)
  assert abs(up - down) > 50.0
  assert min(up, down) < H / 2 < max(up, down)


def test_far_geometry_puts_horizon_at_75_percent():
  geom = crops.far_geometry(600.0, W, H, 2648.0)
  assert geom.view == "far"
  assert (geom.w, geom.h) == (640, 320)
  assert geom.scale == 1
  assert geom.focal_px == pytest.approx(2648.0)
  assert geom.x0 == 644 and geom.x0 % 2 == 0
  assert geom.y0 == 360 and geom.y0 % 2 == 0
  assert geom.y0 + crops.HORIZON_CROP_FRACTION * geom.h == pytest.approx(600.0)


def test_near_geometry_downsamples_x2():
  geom = crops.near_geometry(600.0, W, H, 2648.0)
  assert geom.view == "near"
  assert (geom.w, geom.h) == (1280, 640)
  assert geom.scale == 2
  assert geom.focal_px == pytest.approx(1324.0)
  assert (geom.model_w, geom.model_h) == (640, 320)
  assert geom.x0 == 324 and geom.y0 == 120
  assert geom.y0 + crops.HORIZON_CROP_FRACTION * geom.h == pytest.approx(600.0)


def test_geometry_clamps_to_image():
  top = crops.far_geometry(100.0, W, H, 2648.0)
  assert top.y0 == 0
  bottom = crops.far_geometry(1200.0, W, H, 2648.0)
  assert bottom.y0 == H - bottom.h
  side = crops.near_geometry(600.0, 1000, H, 2648.0)
  assert side.w <= 1000 and side.x0 >= 0


def _synthetic_frame(width=1928, height=1208, stride_pad=0):
  stride = width + stride_pad
  rng = np.random.default_rng(7)
  y = np.full((height, stride), 100, dtype=np.uint8)
  y[:, :width] = rng.integers(0, 255, size=(height, width), dtype=np.uint8)
  uv = np.full((height // 2, stride), 128, dtype=np.uint8)
  uv[:, :width] = rng.integers(0, 255, size=(height // 2, width), dtype=np.uint8)
  buf = np.concatenate([y.reshape(-1), uv.reshape(-1)]).tobytes()
  return buf, stride, stride * height


def test_nv12_planes_with_padding():
  buf, stride, uv_offset = _synthetic_frame(stride_pad=64)
  y, uv = crops.nv12_planes(buf, 1928, 1208, stride, uv_offset)
  assert y.shape == (1208, stride)
  assert uv.shape == (604, stride)
  assert uv_offset == stride * 1208


def test_nv12_planes_rejects_short_frames():
  with pytest.raises(ValueError):
    crops.nv12_planes(b"\x00" * 100, 1928, 1208, 1928, 1928 * 1208)


def test_nv12_to_rgb_matches_cv2_reference():
  cv2 = pytest.importorskip("cv2")
  h, w = 64, 96
  y = np.random.default_rng(3).integers(16, 236, size=(h, w), dtype=np.uint8)
  uv = np.random.default_rng(4).integers(0, 256, size=(h // 2, w), dtype=np.uint8)
  ours = crops.nv12_to_rgb(y, uv)
  nv12 = np.vstack((y, uv))
  ref = cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
  assert np.abs(ours.astype(np.int16) - ref.astype(np.int16)).max() <= 3


def test_model_input_from_frame_shapes():
  buf, stride, uv_offset = _synthetic_frame(stride_pad=32)
  geom = crops.near_geometry(600.0, 1928, 1208, 2648.0)
  model_input = crops.model_input_from_frame(buf, 1928, 1208, stride, uv_offset, geom)
  assert model_input.shape == (1, 3, 320, 640)
  assert model_input.dtype == np.uint8
  assert model_input.flags["C_CONTIGUOUS"]


def test_crop_downsample_matches_nearest_decimation():
  buf, stride, uv_offset = _synthetic_frame()
  y, uv = crops.nv12_planes(buf, 1928, 1208, stride, uv_offset)
  near = crops.near_geometry(600.0, 1928, 1208, 2648.0)
  y_small, uv_small = crops.nv12_crop(y, uv, near)
  x0, y0 = near.x0, near.y0
  assert y_small.shape == (320, 640)
  assert uv_small.shape == (160, 640)
  assert np.array_equal(y_small, np.ascontiguousarray(y[y0:y0 + 640:2, x0:x0 + 1280:2]))


def test_map_model_box_to_full():
  far = crops.far_geometry(600.0, W, H, 2648.0)
  assert crops.map_model_box_to_full((10.0, 20.0, 30.0, 60.0), far) == (654.0, 380.0, 674.0, 420.0)
  near = crops.near_geometry(600.0, W, H, 2648.0)
  assert crops.map_model_box_to_full((10.0, 20.0, 30.0, 60.0), near) == (344.0, 160.0, 384.0, 240.0)


def _allocating_reference(buf, stride, uv_offset, geom):
  """The pre-FramePreprocessor path: allocate the whole chain per call."""
  y, uv = crops.nv12_planes(buf, W, H, stride, uv_offset)
  y_small, uv_small = crops.nv12_crop(y, uv, geom)
  rgb = crops.nv12_to_rgb(y_small, uv_small)
  return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def test_frame_preprocessor_matches_allocating_path():
  buf, stride, uv_offset = _synthetic_frame(stride_pad=64)
  pre = crops.FramePreprocessor()
  for geom in (crops.far_geometry(600.0, W, H, 2648.0), crops.near_geometry(600.0, W, H, 2648.0)):
    ref = _allocating_reference(buf, stride, uv_offset, geom)
    got = pre.model_input_from_frame(buf, W, H, stride, uv_offset, geom)
    assert got is pre.model_input
    assert np.array_equal(got, ref), f"view {geom.view} differs from the allocating path"
    # Workspace reuse must not change the result.
    got2 = pre.model_input_from_frame(buf, W, H, stride, uv_offset, geom)
    assert got2 is got and np.array_equal(got2, ref)


def test_frame_preprocessor_output_does_not_alias_frame_data():
  buf, stride, uv_offset = _synthetic_frame()
  pre = crops.FramePreprocessor()
  geom = crops.far_geometry(600.0, W, H, 2648.0)
  out = pre.model_input_from_frame(buf, W, H, stride, uv_offset, geom).copy()
  mutated = bytearray(buf)
  mutated[:] = b"\x00" * len(mutated)
  again = pre.model_input_from_frame(bytes(mutated), W, H, stride, uv_offset, geom)
  assert not np.array_equal(out, again)
  assert np.array_equal(out, pre.model_input_from_frame(buf, W, H, stride, uv_offset, geom))


def test_frame_preprocessor_rejects_mismatched_geometry():
  pre = crops.FramePreprocessor()
  geom = crops.ViewGeometry(view="far", x0=0, y0=0, w=64, h=32, scale=1, focal_px=100.0)
  with pytest.raises(ValueError):
    pre.model_input_from_frame(b"\x00" * (64 * 32 * 3 // 2), 64, 32, 64, 64 * 32, geom)
