"""Road-camera crop geometry and NV12 -> RGB conversion for the TLV detector.

Pure numpy (no messaging imports), unit-testable on a PC. All public coordinates
are full-frame road-camera pixels; the model input is a 320x640 (H x W) RGB
uint8 NCHW tensor per CONTRACT.md.

Geometry per CONTRACT:
  FAR  = native-resolution 640x320 crop, effective focal 2648 px.
  NEAR = 1280x640 region downsampled x2, effective focal 1324 px.
The calibrated horizon sits at 75% of the crop height (clamped to the image).
Focal values scale with the actual frame width (1928 px reference).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openpilot.common.transformations.camera import get_view_frame_from_calib_frame

REFERENCE_WIDTH = 1928
REFERENCE_HEIGHT = 1208
FOCAL_REFERENCE_PX = 2648.0

MODEL_W = 640
MODEL_H = 320
HORIZON_CROP_FRACTION = 0.75

FAR_CROP = (640, 320)
NEAR_CROP = (1280, 640)


@dataclass(frozen=True)
class ViewGeometry:
  view: str
  x0: int
  y0: int
  w: int
  h: int
  scale: int
  focal_px: float

  @property
  def model_w(self) -> int:
    return self.w // self.scale

  @property
  def model_h(self) -> int:
    return self.h // self.scale


def focal_px_for_width(frame_width: int, reference_focal: float = FOCAL_REFERENCE_PX,
                       reference_width: int = REFERENCE_WIDTH) -> float:
  return float(reference_focal) * float(frame_width) / float(reference_width)


def horizon_row(rpy_calib, focal_px: float, width: int, height: int) -> float:
  """Pixel row where the calibrated road horizon projects (image centre fallback)."""
  centre = 0.5 * float(height)
  if rpy_calib is None or len(rpy_calib) != 3:
    return centre
  view_from_calib = get_view_frame_from_calib_frame(float(rpy_calib[0]), float(rpy_calib[1]), float(rpy_calib[2]), 0.0)[:, :3]
  forward_view = view_from_calib.dot(np.array([1.0, 0.0, 0.0]))
  if forward_view[2] <= 1e-3:
    return centre
  row = centre + focal_px * float(forward_view[1]) / float(forward_view[2])
  return float(row)


def _even_down(value: int) -> int:
  return int(value) - (int(value) & 1)


def _clamp_even(value: int, lo: int, hi: int) -> int:
  return max(_even_down(lo), min(_even_down(value), _even_down(hi)))


def _geometry(view: str, crop: tuple[int, int], horizon_y: float, width: int, height: int,
              focal_px: float, scale: int) -> ViewGeometry:
  w = min(_even_down(crop[0]), _even_down(width))
  h = min(_even_down(crop[1]), _even_down(height))
  x0 = _clamp_even((width - w) // 2, 0, width - w)
  y0 = _clamp_even(int(round(horizon_y - HORIZON_CROP_FRACTION * h)), 0, height - h)
  effective_focal = focal_px / scale if scale > 1 else focal_px
  return ViewGeometry(view=view, x0=x0, y0=y0, w=w, h=h, scale=scale, focal_px=float(effective_focal))


def far_geometry(horizon_y: float, width: int, height: int, focal_px: float) -> ViewGeometry:
  return _geometry("far", FAR_CROP, horizon_y, width, height, focal_px, scale=1)


def near_geometry(horizon_y: float, width: int, height: int, focal_px: float) -> ViewGeometry:
  return _geometry("near", NEAR_CROP, horizon_y, width, height, focal_px, scale=2)


def nv12_planes(data, width: int, height: int, stride: int, uv_offset: int):
  """Validated Y and UV plane views of a padded NV12 allocation."""
  if width <= 0 or height <= 0 or stride < width:
    raise ValueError("invalid NV12 dimensions or stride")
  if width % 2 or height % 2:
    raise ValueError("NV12 frame dimensions must be even")
  raw = np.frombuffer(data, dtype=np.uint8)
  y_size = stride * height
  uv_end = uv_offset + stride * (height // 2)
  if uv_offset < y_size or raw.size < uv_end:
    raise ValueError(f"short NV12 frame: {raw.size} < {uv_end}")
  y = raw[:y_size].reshape(height, stride)
  uv = raw[uv_offset:uv_end].reshape(height // 2, stride)
  return y, uv


def nv12_crop(y: np.ndarray, uv: np.ndarray, geom: ViewGeometry):
  """Visible crop planes; UV stays a byte view with interleaved U/V pairs."""
  y_roi = y[geom.y0:geom.y0 + geom.h, geom.x0:geom.x0 + geom.w]
  uv_roi = uv[geom.y0 // 2:(geom.y0 + geom.h) // 2, geom.x0:geom.x0 + geom.w]
  if geom.scale == 1:
    return np.ascontiguousarray(y_roi), np.ascontiguousarray(uv_roi)
  y_small = np.ascontiguousarray(y_roi[::2, ::2])
  h2, w2 = uv_roi.shape
  uv_pairs = uv_roi.reshape(h2, w2 // 2, 2)[::2, ::2, :]
  uv_small = np.ascontiguousarray(uv_pairs.reshape(h2 // 2, w2 // 2))
  return y_small, uv_small


def nv12_to_rgb(y: np.ndarray, uv: np.ndarray) -> np.ndarray:
  """NV12 (2x2 subsampled chroma, BT.601 limited range) to uint8 RGB (H, W, 3)."""
  y_f = y.astype(np.float32)
  h, w = y_f.shape
  pairs = uv.reshape(h // 2, w // 2, 2).astype(np.float32)
  u = np.repeat(np.repeat(pairs[:, :, 0], 2, axis=0), 2, axis=1) - 128.0
  v = np.repeat(np.repeat(pairs[:, :, 1], 2, axis=0), 2, axis=1) - 128.0
  c = np.maximum(y_f - 16.0, 0.0) * 1.164383
  r = c + 1.596027 * v
  g = c - 0.391762 * u - 0.812968 * v
  b = c + 2.017232 * u
  rgb = np.stack((r, g, b), axis=-1)
  return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def model_input_from_frame(data, width: int, height: int, stride: int, uv_offset: int,
                           geom: ViewGeometry) -> np.ndarray:
  """Crop + convert one NV12 frame into the [1, 3, 320, 640] uint8 NCHW input."""
  y, uv = nv12_planes(data, width, height, stride, uv_offset)
  y_small, uv_small = nv12_crop(y, uv, geom)
  rgb = nv12_to_rgb(y_small, uv_small)
  return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def map_model_box_to_full(box: tuple[float, float, float, float], geom: ViewGeometry) -> tuple[float, float, float, float]:
  """Model-space xyxy (640x320) to full-frame pixel xyxy."""
  scale = float(geom.scale)
  return (
    geom.x0 + box[0] * scale,
    geom.y0 + box[1] * scale,
    geom.x0 + box[2] * scale,
    geom.y0 + box[3] * scale,
  )
