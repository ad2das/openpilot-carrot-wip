"""W136: TlvOverlay logic (no drawing). The screen box follows carrotTlv.relevantBox.

These tests never draw: _draw is monkeypatched to capture the eased screen box,
pyray is the w84 stub on the desktop gate, and time.monotonic is driven manually.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.ui.onroad import tlv_overlay
from openpilot.selfdrive.ui.onroad.tlv_overlay import TlvOverlay


class _FakeType:
  def __init__(self, *args, **kwargs):
    pass

  def width(self, text, size, weight):
    return float(len(str(text))) * float(size) * 0.5

  def draw_mid(self, *args, **kwargs):
    return None


def _overlay(monkeypatch):
  """TlvOverlay without touching gui_app fonts (hs.Type needs them)."""
  monkeypatch.setattr(tlv_overlay.hs, "Type", _FakeType, raising=False)
  return TlvOverlay()


class FakeSM:
  def __init__(self, msg=None, *, seen=True, age=0.0, now=0.0):
    self.seen = {"carrotTlv": seen}
    self.recv_time = {"carrotTlv": now - age}
    self._msg = msg

  def __getitem__(self, _service):
    return self._msg


def box_msg(x1, y1, x2, y2, *, state=1, track_id=5, status=1):
  return SimpleNamespace(
    status=SimpleNamespace(raw=status),
    relevantTrackId=track_id,
    state=SimpleNamespace(raw=state),
    relevantBox=SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2),
  )


def test_current_returns_none_when_unseen(monkeypatch):
  ov = _overlay(monkeypatch)
  assert ov._current(FakeSM(box_msg(0, 0, 10, 10), seen=False)) is None


def test_current_returns_none_when_stale(monkeypatch):
  monkeypatch.setattr(tlv_overlay.time, "monotonic", lambda: 100.0)
  ov = _overlay(monkeypatch)
  sm = FakeSM(box_msg(0, 0, 10, 10), age=tlv_overlay.STALE_S + 0.01, now=100.0)
  assert ov._current(sm) is None


def test_current_returns_none_when_status_not_running(monkeypatch):
  ov = _overlay(monkeypatch)
  assert ov._current(FakeSM(box_msg(0, 0, 10, 10, status=0))) is None


def test_current_returns_none_when_no_track(monkeypatch):
  ov = _overlay(monkeypatch)
  assert ov._current(FakeSM(box_msg(0, 0, 10, 10, track_id=-1))) is None


def test_current_returns_none_for_zero_or_inverted_box(monkeypatch):
  ov = _overlay(monkeypatch)
  assert ov._current(FakeSM(box_msg(10, 10, 10, 20))) is None  # zero width
  assert ov._current(FakeSM(box_msg(20, 10, 10, 20))) is None  # inverted x


def test_current_returns_box_state_trackid(monkeypatch):
  monkeypatch.setattr(tlv_overlay.time, "monotonic", lambda: 50.0)
  ov = _overlay(monkeypatch)
  cur = ov._current(FakeSM(box_msg(1.0, 2.0, 3.0, 4.0, state=2, track_id=9), now=50.0))
  assert cur == ((1.0, 2.0, 3.0, 4.0), 2, 9)


def test_project_scale_and_offset_homography(monkeypatch):
  ov = _overlay(monkeypatch)
  ov.set_pixel_transform(np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 20.0], [0.0, 0.0, 1.0]]))
  out = ov._project((1.0, 2.0, 3.0, 4.0))
  assert out == pytest.approx((12.0, 24.0, 16.0, 28.0))


def test_render_eases_same_track_and_snaps_on_new_track(monkeypatch):
  clock = {"t": 0.0}
  monkeypatch.setattr(tlv_overlay.time, "monotonic", lambda: clock["t"])
  ov = _overlay(monkeypatch)
  ov.set_pixel_transform(np.eye(3))
  captured = []
  monkeypatch.setattr(ov, "_draw", lambda scr, *a, **k: captured.append(scr))

  # Frame 1: first sighting snaps to the target.
  clock["t"] = 1.0
  ov.render(FakeSM(box_msg(0, 0, 100, 100, track_id=5), now=1.0), None, [])
  assert len(captured) == 1
  first = captured[-1]

  # Frame 2: same track, box moved -> eased part of the way (not a snap).
  clock["t"] = 1.08
  ov.render(FakeSM(box_msg(200, 0, 300, 100, track_id=5), now=1.08), None, [])
  eased = captured[-1]
  assert first[0] < eased[0] < 200.0

  # Frame 3: new track -> snaps to the target box.
  clock["t"] = 1.16
  ov.render(FakeSM(box_msg(500, 0, 600, 100, track_id=6), now=1.16), None, [])
  snapped = captured[-1]
  assert snapped[0] == pytest.approx(500.0)
