from __future__ import annotations

import threading
from types import SimpleNamespace

from openpilot.selfdrive.carrot.carrot_man import CarrotMan


class _SpyLock:
  def __init__(self, on_enter=None, on_exit=None):
    self._lock = threading.Lock()
    self._on_enter = on_enter
    self._on_exit = on_exit

  def __enter__(self):
    self._lock.acquire()
    if self._on_enter:
      self._on_enter()
    return self

  def __exit__(self, exc_type, exc, tb):
    try:
      if self._on_exit:
        self._on_exit()
    finally:
      self._lock.release()


def test_handle_route_updates_navd_active_atomically_under_navi_lock():
  cm = CarrotMan.__new__(CarrotMan)
  observed_inside_lock = []

  def on_exit():
    observed_inside_lock.append((cm.navi_points_active, cm.navd_active, len(cm.navi_points)))

  cm._navi_points_lock = _SpyLock(on_exit=on_exit)
  cm.navi_points = []
  cm.navi_points_start_index = 0
  cm.navi_points_active = False
  cm.navd_active = False
  cm.pm = SimpleNamespace(send=lambda *a, **k: None)
  cm.params = SimpleNamespace(put=lambda *a, **k: None, remove=lambda *a, **k: None)

  cm.handle_route([{"x": 127.0, "y": 37.5}, {"x": 127.01, "y": 37.51}])
  assert observed_inside_lock == [(True, True, 2)]

  cm.handle_route([])
  assert observed_inside_lock[-1] == (False, False, 0)
