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
# Standstill/creep (CONTRACT.md): NEAR crop shifted up so the horizon sits at
# 90% of the crop height, covering overhead lights near the stop line.
STANDSTILL_HORIZON_CROP_FRACTION = 0.90

FAR_CROP = (640, 320)
NEAR_CROP = (1280, 640)

# Lamp-classifier crop (CONTRACT.md): housing box padded 25%, resized to 32x64.
LAMP_CROP_H = 32
LAMP_CROP_W = 64
LAMP_PAD_FRACTION = 0.25
MAX_LAMP_CROPS = 4


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
              focal_px: float, scale: int,
              horizon_fraction: float = HORIZON_CROP_FRACTION) -> ViewGeometry:
  w = min(_even_down(crop[0]), _even_down(width))
  h = min(_even_down(crop[1]), _even_down(height))
  x0 = _clamp_even((width - w) // 2, 0, width - w)
  y0 = _clamp_even(int(round(horizon_y - horizon_fraction * h)), 0, height - h)
  effective_focal = focal_px / scale if scale > 1 else focal_px
  return ViewGeometry(view=view, x0=x0, y0=y0, w=w, h=h, scale=scale, focal_px=float(effective_focal))


def far_geometry(horizon_y: float, width: int, height: int, focal_px: float,
                 horizon_fraction: float = HORIZON_CROP_FRACTION) -> ViewGeometry:
  return _geometry("far", FAR_CROP, horizon_y, width, height, focal_px, scale=1,
                   horizon_fraction=horizon_fraction)


def near_geometry(horizon_y: float, width: int, height: int, focal_px: float,
                  horizon_fraction: float = HORIZON_CROP_FRACTION) -> ViewGeometry:
  return _geometry("near", NEAR_CROP, horizon_y, width, height, focal_px, scale=2,
                   horizon_fraction=horizon_fraction)


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


class FramePreprocessor:
  """Preallocated workspace for crop + NV12->RGB + NCHW packing.

  RESOURCE_BUDGET.md: every per-frame buffer is allocated once here; the hot
  path only writes into these arrays (ufunc out=, slice copies), so no per-frame
  numpy allocation happens. Instantiate once per process and reuse.

  Layout: the model input is MODEL_H x MODEL_W (320x640) for both views - FAR is
  a native crop, NEAR is a x2 decimation.
  """

  def __init__(self, model_h: int = MODEL_H, model_w: int = MODEL_W) -> None:
    self.model_h = int(model_h)
    self.model_w = int(model_w)
    h, w = self.model_h, self.model_w
    self.y_crop = np.empty((h, w), dtype=np.uint8)
    self.uv_pairs = np.empty((h // 2, w // 2, 2), dtype=np.uint8)
    self.model_input = np.empty((1, 3, h, w), dtype=np.uint8)
    self._yf = np.empty((h, w), dtype=np.float32)
    self._uf = np.empty((h, w), dtype=np.float32)
    self._vf = np.empty((h, w), dtype=np.float32)
    self._cf = np.empty((h, w), dtype=np.float32)
    self._rf = np.empty((h, w), dtype=np.float32)
    self._gf = np.empty((h, w), dtype=np.float32)
    self._bf = np.empty((h, w), dtype=np.float32)
    self._tf = np.empty((h, w), dtype=np.float32)

  def model_input_from_frame(self, data, width: int, height: int, stride: int, uv_offset: int,
                             geom: ViewGeometry) -> np.ndarray:
    """Copy the crop out of `data` and write the NCHW input into the workspace.

    The returned array never references `data`, so the caller can release the
    VisionIPC buffer as soon as this returns (RESOURCE_BUDGET.md: VisionIPC).
    """
    if geom.model_h != self.model_h or geom.model_w != self.model_w:
      raise ValueError(f"geometry {geom.model_w}x{geom.model_h} does not fit workspace {self.model_w}x{self.model_h}")
    y, uv = nv12_planes(data, width, height, stride, uv_offset)
    self._crop_into(y, uv, geom)
    self._nv12_to_nchw_into()
    return self.model_input

  def _crop_into(self, y: np.ndarray, uv: np.ndarray, geom: ViewGeometry) -> None:
    h, w = geom.h, geom.w
    y_roi = y[geom.y0:geom.y0 + h, geom.x0:geom.x0 + w]
    if geom.scale == 1:
      np.copyto(self.y_crop, y_roi)
    else:
      np.copyto(self.y_crop, y_roi[::2, ::2])
    uv3 = uv.reshape(uv.shape[0], uv.shape[1] // 2, 2)
    p0y, p0x = geom.y0 // 2, geom.x0 // 2
    ph, pw = h // 2, w // 2
    if geom.scale == 1:
      np.copyto(self.uv_pairs, uv3[p0y:p0y + ph, p0x:p0x + pw, :])
    else:
      np.copyto(self.uv_pairs, uv3[p0y:p0y + ph:2, p0x:p0x + pw:2, :])

  def _nv12_to_nchw_into(self) -> None:
    yf, uf, vf, cf = self._yf, self._uf, self._vf, self._cf
    np.copyto(yf, self.y_crop, casting="unsafe")
    yf -= 16.0
    np.maximum(yf, 0.0, out=yf)
    np.multiply(yf, 1.164383, out=cf)
    for buf, src in ((uf, self.uv_pairs[:, :, 0]), (vf, self.uv_pairs[:, :, 1])):
      np.copyto(buf[0::2, 0::2], src, casting="unsafe")
      np.copyto(buf[0::2, 1::2], src, casting="unsafe")
      np.copyto(buf[1::2, 0::2], src, casting="unsafe")
      np.copyto(buf[1::2, 1::2], src, casting="unsafe")
      buf -= 128.0
    rf, gf, bf, tf = self._rf, self._gf, self._bf, self._tf
    np.multiply(vf, 1.596027, out=rf)
    np.add(rf, cf, out=rf)
    np.multiply(uf, -0.391762, out=gf)
    np.multiply(vf, -0.812968, out=tf)
    np.add(gf, tf, out=gf)
    np.add(gf, cf, out=gf)
    np.multiply(uf, 2.017232, out=bf)
    np.add(bf, cf, out=bf)
    np.clip(rf, 0.0, 255.0, out=rf)
    np.clip(gf, 0.0, 255.0, out=gf)
    np.clip(bf, 0.0, 255.0, out=bf)
    np.copyto(self.model_input[0, 0], rf, casting="unsafe")
    np.copyto(self.model_input[0, 1], gf, casting="unsafe")
    np.copyto(self.model_input[0, 2], bf, casting="unsafe")


class LampCropBuffer:
  """Preallocated [MAX_LAMP_CROPS, 3, 32, 64] uint8 builder for the classifier.

  Each crop is the track box padded by 25% and resized nearest into 32x64, in
  the CURRENT view's model space (640x320). Buffers are allocated once; build()
  only writes into them (np.take with out=).
  """

  def __init__(self, max_tracks: int = MAX_LAMP_CROPS) -> None:
    self.max_tracks = int(max_tracks)
    self.crops = np.zeros((self.max_tracks, 3, LAMP_CROP_H, LAMP_CROP_W), dtype=np.uint8)
    self._rows = np.empty((3, LAMP_CROP_H, MODEL_W), dtype=np.uint8)
    self._ys = np.empty(LAMP_CROP_H, dtype=np.intp)
    self._xs = np.empty(LAMP_CROP_W, dtype=np.intp)

  def build(self, model_input: np.ndarray, geom: ViewGeometry,
            boxes) -> tuple[np.ndarray, int]:
    """Write up to max_tracks lamp crops; returns (crops [4,3,32,64], count)."""
    if model_input.shape != (1, 3, MODEL_H, MODEL_W):
      raise ValueError(f"unexpected model input shape {model_input.shape}")
    count = 0
    for box in boxes:
      if count >= self.max_tracks:
        break
      if self._fill_crop(model_input[0], geom, box, count):
        count += 1
    return self.crops, count

  def _fill_crop(self, image: np.ndarray, geom: ViewGeometry, box, slot: int) -> bool:
    x1 = (box[0] - geom.x0) / geom.scale
    y1 = (box[1] - geom.y0) / geom.scale
    x2 = (box[2] - geom.x0) / geom.scale
    y2 = (box[3] - geom.y0) / geom.scale
    pad_x = LAMP_PAD_FRACTION * max(0.0, x2 - x1)
    pad_y = LAMP_PAD_FRACTION * max(0.0, y2 - y1)
    x1, x2 = x1 - pad_x, x2 + pad_x
    y1, y2 = y1 - pad_y, y2 + pad_y
    x1 = min(max(x1, 0.0), MODEL_W - 1.0)
    y1 = min(max(y1, 0.0), MODEL_H - 1.0)
    x2 = min(max(x2, x1 + 1.0), MODEL_W - 1.0)
    y2 = min(max(y2, y1 + 1.0), MODEL_H - 1.0)
    for i in range(LAMP_CROP_H):
      self._ys[i] = int(y1 + (i + 0.5) * (y2 - y1) / LAMP_CROP_H)
    for j in range(LAMP_CROP_W):
      self._xs[j] = int(x1 + (j + 0.5) * (x2 - x1) / LAMP_CROP_W)
    np.take(image, self._ys, axis=1, out=self._rows)
    np.take(self._rows, self._xs, axis=2, out=self.crops[slot])
    return True


def model_input_from_frame(data, width: int, height: int, stride: int, uv_offset: int,
                           geom: ViewGeometry, workspace: FramePreprocessor | None = None) -> np.ndarray:
  """Crop + convert one NV12 frame into the [1, 3, 320, 640] uint8 NCHW input.

  With `workspace` the result is written into its preallocated arrays (no
  per-frame allocation); without one a fresh workspace is allocated per call
  (PC convenience only - never do that in the tlvd hot path).
  """
  if workspace is None:
    workspace = FramePreprocessor()
  return workspace.model_input_from_frame(data, width, height, stride, uv_offset, geom)


def map_model_box_to_full(box: tuple[float, float, float, float], geom: ViewGeometry) -> tuple[float, float, float, float]:
  """Model-space xyxy (640x320) to full-frame pixel xyxy."""
  scale = float(geom.scale)
  return (
    geom.x0 + box[0] * scale,
    geom.y0 + box[1] * scale,
    geom.x0 + box[2] * scale,
    geom.y0 + box[3] * scale,
  )
