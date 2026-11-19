"""Monocular traffic-light range estimation for the C3X road camera.

Fuses two independent range cues already available on the device:
  1. Horizontal housing width w (px) calibrated on 58 stop-at-line approaches
     and verified against 1,505 top-exit approaches (W-TLD1/W-TLD2).
  2. Calibrated elevation ray intersected with the modelV2 roadEdges z-profile
     (liveCalibration rpyCalib + height[0] + roadEdges[0/1].z), which keeps
     long-range accuracy (>= 150 m) across uphill/downhill grades and does not
     shrink when the driving model's position.x trajectory shortens.

Pure numpy/math (no messaging imports) so it is unit-testable on a PC.
"""
from __future__ import annotations

import math

import numpy as np

from openpilot.common.transformations.camera import get_view_frame_from_calib_frame
from openpilot.selfdrive.carrot.tlv.crops import (
  FOCAL_REFERENCE_PX,
  REFERENCE_HEIGHT,
  REFERENCE_WIDTH,
)

# Horizontal box width fit: d = K / max(w_ref - B, 0.5) + X (metres).
BOX_W_K = 1789.0
BOX_W_B = 8.25
BOX_W_X = 6.0

RANGE_MIN_M = 10.0
RANGE_MAX_M = 420.0
RANGE_FAR_FALLBACK_M = 350.0

# Overhead Korean traffic-light housing centre height above road surface (m).
HOUSING_MOUNT_HEIGHT_M = 4.6
DEFAULT_CAM_HEIGHT_M = 1.3
MIN_HEIGHT_ABOVE_CAM_M = 2.5

MAX_END_SLOPE = 0.08
MAX_ROAD_Z_M = 40.0
_RAY_DS = np.linspace(15.0, RANGE_MAX_M, 82, dtype=np.float64)


def box_width_range_m(box_w_px: float, focal_px: float = FOCAL_REFERENCE_PX) -> float:
  """Range estimate in metres from horizontal housing width (full-frame px)."""
  if not (math.isfinite(box_w_px) and math.isfinite(focal_px) and focal_px > 0.0):
    return float("nan")
  w_ref = float(box_w_px) * (FOCAL_REFERENCE_PX / float(focal_px))
  if w_ref <= 1.0:
    return float("nan")
  d = BOX_W_K / max(w_ref - BOX_W_B, 0.5) + BOX_W_X
  return float(min(RANGE_MAX_M, max(RANGE_MIN_M, d)))


def road_edges_mean_z(road_edges) -> tuple[np.ndarray | None, np.ndarray | None]:
  """Extract mean (x, z) road elevation profile from modelV2.roadEdges."""
  try:
    if road_edges is None or len(road_edges) != 2:
      return None, None
    x0 = np.asarray(road_edges[0].x, dtype=np.float64)
    z0 = np.asarray(road_edges[0].z, dtype=np.float64)
    x1 = np.asarray(road_edges[1].x, dtype=np.float64)
    z1 = np.asarray(road_edges[1].z, dtype=np.float64)
    if (x0.ndim != 1 or x1.ndim != 1 or len(x0) < 2 or len(x1) < 2
        or z0.shape != x0.shape or z1.shape != x1.shape):
      return None, None
    if not (np.all(np.isfinite(x0)) and np.all(np.isfinite(z0))
            and np.all(np.isfinite(x1)) and np.all(np.isfinite(z1))):
      return None, None
    if np.any(np.diff(x0) <= 0.0) or np.any(np.diff(x1) <= 0.0):
      return None, None
    if x0.shape == x1.shape and np.allclose(x0, x1, atol=1e-3):
      zm = 0.5 * (z0 + z1)
      return x0, zm
    z1_on_0 = np.interp(x0, x1, z1)
    return x0, 0.5 * (z0 + z1_on_0)
  except (AttributeError, TypeError, ValueError, IndexError):
    return None, None


def ray_upward_slope(cx: float, cy: float, rpy_calib, focal_px: float,
                     width: int = REFERENCE_WIDTH, height: int = REFERENCE_HEIGHT) -> float:
  """Calibrated road-frame upward ray slope (-z / x) for full-frame pixel (cx, cy)."""
  if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(focal_px) and focal_px > 0.0):
    return float("nan")
  cx0 = 0.5 * float(width)
  cy0 = 0.5 * float(height)
  if rpy_calib is None or len(rpy_calib) != 3 or not all(math.isfinite(float(v)) for v in rpy_calib):
    return float((cy0 - cy) / focal_px)
  vfc = get_view_frame_from_calib_frame(
    float(rpy_calib[0]), float(rpy_calib[1]), float(rpy_calib[2]), 0.0,
  )[:, :3]
  ray_v = np.array([(cx - cx0) / focal_px, (cy - cy0) / focal_px, 1.0], dtype=np.float64)
  ray_c = vfc.T @ ray_v
  if ray_c[0] <= 1e-3:
    return float("nan")
  # Calib frame has +x forward, +y right, +z down => upward slope is -z / x.
  return float(-ray_c[2] / ray_c[0])


def elevation_grade_range_m(cx: float, cy: float, rpy_calib, focal_px: float = FOCAL_REFERENCE_PX,
                            width: int = REFERENCE_WIDTH, height: int = REFERENCE_HEIGHT,
                            h_cam_m: float | None = None,
                            road_x=None, road_z=None) -> float:
  """Range estimate in metres by intersecting the calibrated ray with road grade."""
  slope_up = ray_upward_slope(cx, cy, rpy_calib, focal_px, width, height)
  if not math.isfinite(slope_up) or slope_up < -0.06:
    return float("nan")
  hc = float(h_cam_m) if (h_cam_m is not None and math.isfinite(float(h_cam_m))) else DEFAULT_CAM_HEIGHT_M
  h_tgt = max(HOUSING_MOUNT_HEIGHT_M - hc, MIN_HEIGHT_ABOVE_CAM_M)

  use_grade = False
  if road_x is not None and road_z is not None:
    xg = np.asarray(road_x, dtype=np.float64)
    zg = np.asarray(road_z, dtype=np.float64)
    if (xg.ndim == 1 and zg.shape == xg.shape and len(xg) >= 2
        and np.all(np.isfinite(xg)) and np.all(np.isfinite(zg))
        and np.all(np.diff(xg) > 0.0) and float(np.max(np.abs(zg))) <= MAX_ROAD_Z_M):
      use_grade = True

  if not use_grade:
    if slope_up <= 0.002:
      return RANGE_FAR_FALLBACK_M
    return float(min(RANGE_MAX_M, max(RANGE_MIN_M, h_tgt / slope_up)))

  end_slope = float(np.clip((zg[-1] - zg[-2]) / max(float(xg[-1] - xg[-2]), 1.0),
                            -MAX_END_SLOPE, MAX_END_SLOPE))
  zd = np.interp(_RAY_DS, xg, zg)
  beyond = _RAY_DS > xg[-1]
  if np.any(beyond):
    zd[beyond] = zg[-1] + end_slope * (_RAY_DS[beyond] - xg[-1])
  below = _RAY_DS < xg[0]
  if np.any(below) and xg[0] > 1.0:
    zd[below] = (zg[0] / xg[0]) * _RAY_DS[below]

  # Height above road surface equals h_tgt when slope_up * d + road_z(d) == h_tgt.
  f = slope_up * _RAY_DS + zd - h_tgt
  crossings = np.where((f[:-1] <= 0.0) & (f[1:] > 0.0))[0]
  if len(crossings) == 0:
    if f[-1] <= 0.0:
      return RANGE_FAR_FALLBACK_M
    return float("nan")
  k = int(crossings[0])
  frac = float(-f[k] / max(float(f[k + 1] - f[k]), 1e-9))
  return float(_RAY_DS[k] + frac * (_RAY_DS[k + 1] - _RAY_DS[k]))


def fused_tl_range_m(box, rpy_calib, focal_px: float = FOCAL_REFERENCE_PX,
                     width: int = REFERENCE_WIDTH, height: int = REFERENCE_HEIGHT,
                     h_cam_m: float | None = None,
                     road_x=None, road_z=None) -> float:
  """Fused monocular range in metres (0.0 when unusable).

  Combines horizontal housing width and grade-compensated elevation via their
  geometric mean (verified ~50/50 optimal in log-space LORO across Palisade+K5).
  """
  try:
    x1, y1, x2, y2 = (float(v) for v in box)
  except (TypeError, ValueError):
    return 0.0
  if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
    return 0.0
  d_w = box_width_range_m(x2 - x1, focal_px)
  d_el = elevation_grade_range_m(
    0.5 * (x1 + x2), 0.5 * (y1 + y2), rpy_calib, focal_px, width, height,
    h_cam_m=h_cam_m, road_x=road_x, road_z=road_z,
  )
  if math.isfinite(d_w) and d_w > 0.0 and math.isfinite(d_el) and d_el > 0.0:
    return float(math.sqrt(d_w * d_el))
  if math.isfinite(d_w) and d_w > 0.0:
    return float(d_w)
  return 0.0
