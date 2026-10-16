"""W136: tlv_pixel_transform maps fcam full-frame pixels to the on-screen stream.

Far traffic lights the TLV overlay boxes come from the road (fcam) camera; when
the wide (ecam) stream is on screen the box must be re-projected. The wide case
is a rotation-only homography fcam -> ecam composed with the on-screen transform.
"""
import sys
import types

import numpy as np
import pytest

if "msgq.visionipc" not in sys.modules:
  _vipc = types.ModuleType("msgq.visionipc")

  class _VisionStreamType:
    VISION_STREAM_ROAD = 0
    VISION_STREAM_WIDE_ROAD = 1

  class _Stub:
    def __init__(self, *args, **kwargs):
      pass

    def __getattr__(self, _name):
      return lambda *a, **k: None

    def __call__(self, *a, **k):
      return None

  _vipc.VisionStreamType = _VisionStreamType
  _vipc.__getattr__ = lambda _name: _Stub()
  sys.modules["msgq.visionipc"] = _vipc

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.ui.onroad.augmented_road_view import tlv_pixel_transform


def test_road_returns_video_transform_unchanged():
  video_transform = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0], [0.0, 0.0, 1.0]])
  out = tlv_pixel_transform(video_transform, np.eye(3), np.eye(3), np.eye(3), np.eye(3), is_wide=False)
  assert out is video_transform


def test_wide_homography_matches_fcam_to_ecam_projection():
  cam = DEVICE_CAMERAS[("tici", "ox03c10")]
  k_f = cam.fcam.intrinsics
  k_e = cam.ecam.intrinsics

  rpy_calib = np.array([0.03, -0.02, 0.01])
  wide_from_device_euler = np.array([0.01, 0.005, -0.008])
  device_from_calib = rot_from_euler(rpy_calib)
  view_from_device = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
  view_from_calib = view_from_device @ device_from_calib
  wide_from_device = rot_from_euler(wide_from_device_euler)
  view_from_wide_calib = view_from_device @ wide_from_device @ device_from_calib

  video_transform = np.array([[1.3, 0.0, 120.0], [0.0, 1.3, 40.0], [0.0, 0.0, 1.0]])

  h = tlv_pixel_transform(
    video_transform, k_f, k_e, view_from_calib, view_from_wide_calib, is_wide=True
  )

  # A point 300 m ahead in the calib frame.
  pt_calib = np.array([300.0, 0.4, -0.2])
  pt_ecam = k_e @ view_from_wide_calib @ pt_calib
  pt_ecam = pt_ecam[:2] / pt_ecam[2]

  pt_fcam = k_f @ view_from_calib @ pt_calib
  pt_fcam = np.array([pt_fcam[0] / pt_fcam[2], pt_fcam[1] / pt_fcam[2], 1.0])

  mapped = h @ pt_fcam
  mapped = mapped[:2] / mapped[2]
  on_screen = video_transform @ np.array([pt_ecam[0], pt_ecam[1], 1.0])
  on_screen = on_screen[:2] / on_screen[2]

  assert mapped == pytest.approx(on_screen, abs=0.5)
