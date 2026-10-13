"""Tests for the TLV shadow-mode recorder (openpilot/selfdrive/carrot/tlv/recorder.py).

Sampling policy, ring-full drops, storage caps (fake caps + fake statvfs), the
writer's exception containment, the no-allocation hot path and the tlvd mode
gating are covered here. The Carrot Web endpoints live in
openpilot/selfdrive/carrot/server/tests/test_tlv_rec.py.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import time
import tracemalloc
from types import SimpleNamespace

import cv2
import numpy as np

from openpilot.selfdrive.carrot.tlv import crops, recorder, tlvd, tracker

GB = 1024 * 1024 * 1024
GEOM = crops.far_geometry(600.0, crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT, crops.FOCAL_REFERENCE_PX)


class FakeDisk:
  """Injected statvfs: free space in bytes (frsize 1)."""

  def __init__(self, free_bytes: int) -> None:
    self.free_bytes = int(free_bytes)

  def __call__(self, _path: str) -> SimpleNamespace:
    return SimpleNamespace(f_bavail=self.free_bytes, f_frsize=1, f_blocks=self.free_bytes)


def make_input(fill: int = 16) -> np.ndarray:
  arr = np.empty((1, 3, recorder.MODEL_H, recorder.MODEL_W), dtype=np.uint8)
  arr.fill(fill)
  return arr


def make_det(cls: int = 0, score: float = 0.9) -> tracker.Detection:
  return tracker.Detection(cls=cls, score=score, x1=300.0, y1=200.0, x2=340.0, y2=240.0)


def make_track(vote=None) -> tracker.Track:
  track = tracker.Track(track_id=1, first_frame=0, last_frame=10, cls=0, score=0.8,
                        box=(300.0, 200.0, 340.0, 240.0), view="far", focal_px=2648.0)
  if vote is not None:
    track.votes.append(vote)
  return track


def offer(rec, now, *, v_ego=10.0, state=0, dets=None, tracks=None, red_likely=False,
          state_conf=0.0, relevant_track_id=-1, model_input=None, gated=()) -> bool:
  return rec.offer(
    model_input if model_input is not None else make_input(),
    now=now, frame_id=1, view="far", raised=False, geom=GEOM,
    horizon_y=600.0, calib_rpy=[0.0, 0.0, 0.0],
    v_ego=v_ego, a_ego=0.0, yaw_rate=0.0, steering_angle_deg=0.0,
    left_blinker=False, right_blinker=False,
    dets=list(dets or []), tracks=list(tracks or []),
    relevant_track_id=relevant_track_id, state=state, state_conf=state_conf,
    red_likely=red_likely, gated=list(gated), wall_time_iso="2026-01-01T00:00:00.000+09:00",
  )


def make_recorder(tmp_path, **kwargs) -> recorder.TlvRecorder:
  kwargs.setdefault("statvfs_fn", FakeDisk(64 * GB))
  return recorder.TlvRecorder(root=str(tmp_path), **kwargs)


def session_dirs(root: str) -> list[str]:
  return sorted(d for d in os.listdir(root)
                if recorder.SESSION_NAME_RE.match(d) and os.path.isdir(os.path.join(root, d)))


def read_index(root: str) -> list[dict]:
  dirs = session_dirs(root)
  assert len(dirs) == 1, dirs
  path = os.path.join(root, dirs[0], "index.jsonl")
  with open(path, encoding="utf-8") as f:
    return [json.loads(line) for line in f if line.strip()]


def wait_until(predicate, timeout_s: float = 5.0) -> bool:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return False


# ---------------------------------------------------------------------------
# sampling policy
# ---------------------------------------------------------------------------

def test_light_frames_throttled_to_half_second(tmp_path):
  rec = make_recorder(tmp_path)
  det = [make_det()]
  assert offer(rec, 100.0, dets=det) is True
  assert offer(rec, 100.1, dets=det) is False
  assert offer(rec, 100.49, dets=det) is False
  assert offer(rec, 100.5, dets=det) is True
  assert offer(rec, 101.0, dets=det) is True
  assert rec.wait_idle()
  stats = rec.stats()
  assert stats["saved"] == 3
  assert stats["dropped_rate"] == 2
  assert stats["dropped_full"] == 0
  assert [m["reason"] for m in read_index(str(tmp_path))] == ["light"] * 3
  rec.close()


def test_light_frame_also_when_track_or_state_or_red_likely(tmp_path):
  rec = make_recorder(tmp_path)
  assert offer(rec, 100.0, tracks=[make_track()]) is True
  assert offer(rec, 101.0, state=1) is True
  assert offer(rec, 102.0, red_likely=True) is True
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 3
  rec.close()


def test_standstill_light_throttled_to_three_seconds(tmp_path):
  rec = make_recorder(tmp_path)
  det = [make_det()]
  assert offer(rec, 100.0, v_ego=0.5, dets=det) is True
  assert offer(rec, 101.0, v_ego=0.5, dets=det) is False
  assert offer(rec, 102.9, v_ego=0.5, dets=det) is False
  assert offer(rec, 103.0, v_ego=0.5, dets=det) is True
  assert rec.wait_idle()
  stats = rec.stats()
  assert stats["saved"] == 2
  assert stats["dropped_rate"] == 2
  rec.close()


def test_background_frames_throttled_and_speed_gated(tmp_path):
  rec = make_recorder(tmp_path)
  assert offer(rec, 200.0, v_ego=10.0) is True
  assert offer(rec, 204.0, v_ego=10.0) is False
  assert offer(rec, 205.0, v_ego=10.0) is True
  # below 3 m/s a background frame is not a candidate at all
  assert offer(rec, 206.0, v_ego=2.0) is False
  assert offer(rec, 300.0, v_ego=2.0) is False
  assert rec.wait_idle()
  stats = rec.stats()
  assert stats["saved"] == 2
  assert stats["dropped_rate"] == 1
  reasons = [m["reason"] for m in read_index(str(tmp_path))]
  assert reasons == ["background", "background"]
  rec.close()


def test_state_transition_bypasses_intervals_with_cap(tmp_path):
  rec = make_recorder(tmp_path)
  det = [make_det()]
  # four transitions 0.1 s apart: each saves immediately (0.5 s interval bypassed)
  for i, state in enumerate((1, 2, 1, 2)):
    assert offer(rec, 300.0 + 0.1 * i, dets=det, state=state) is True
  assert rec.wait_idle()
  # fifth transition inside the 10 s window: no bypass and 0.1 s since the last
  # save -> rate drop
  assert offer(rec, 300.4, dets=det, state=1) is False
  assert rec.wait_idle()
  # after the window has passed the next transition saves again
  assert offer(rec, 310.5, dets=det, state=2) is True
  assert rec.wait_idle()
  stats = rec.stats()
  assert stats["saved"] == 5
  assert stats["dropped_rate"] == 1
  assert [m["reason"] for m in read_index(str(tmp_path))] == ["transition"] * 5
  rec.close()


# ---------------------------------------------------------------------------
# ring-full drop path
# ---------------------------------------------------------------------------

def test_ring_full_drops_and_counts(tmp_path):
  gate = SimpleNamespace(event=None)

  def slow_encode(_rgb):
    gate.event.wait(5.0)
    return b"jpeg"

  import threading
  gate.event = threading.Event()
  rec = make_recorder(tmp_path, encode_fn=slow_encode)
  for i in range(4):
    assert offer(rec, 100.0 + 1.0 * i, dets=[make_det()]) is True
  # all four slots are stuck in the blocked writer
  assert offer(rec, 105.0, dets=[make_det()]) is False
  assert rec.stats()["dropped_full"] == 1
  gate.event.set()
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 4
  rec.close()


# ---------------------------------------------------------------------------
# mode gating (tlvd integration)
# ---------------------------------------------------------------------------

class FakeParams:
  def __init__(self, values) -> None:
    self._values = dict(values)

  def get_int(self, key):
    value = self._values.get(key)
    return None if value is None else int(value)

  def get(self, key):
    return self._values.get(key)

  def put_int(self, key, value) -> None:
    self._values[key] = value

  def put(self, key, value) -> None:
    self._values[key] = value


def test_mode2_never_constructs_recorder(tmp_path):
  runner = tlvd.TlvRunner(FakeParams({"CarrotTlvMode": 2, "CarrotTlvBackend": "null"}),
                          recorder_root=str(tmp_path))
  try:
    assert runner.recorder is None
  finally:
    runner.close_recorder()


def test_mode1_constructs_recorder_with_injected_root(tmp_path):
  runner = tlvd.TlvRunner(FakeParams({"CarrotTlvMode": 1, "CarrotTlvBackend": "null"}),
                          recorder_root=str(tmp_path))
  try:
    assert runner.recorder is not None
  finally:
    runner.close_recorder()


def test_mode1_without_root_stays_off_on_pc():
  runner = tlvd.TlvRunner(FakeParams({"CarrotTlvMode": 1, "CarrotTlvBackend": "null"}))
  try:
    assert runner.recorder is None
  finally:
    runner.close_recorder()


def test_mode_1_to_2_stops_recording(tmp_path):
  """A live 1 -> 2 switch must stop recording at the next 1 s poll."""
  params = FakeParams({"CarrotTlvMode": 1, "CarrotTlvBackend": "null"})
  runner = tlvd.TlvRunner(params, recorder_root=str(tmp_path))
  try:
    rec = runner.recorder
    assert rec is not None and rec.active is True
    t = time.monotonic() + 60.0
    runner._poll_system(t)
    assert offer(rec, 100.0, dets=[make_det()]) is True
    assert rec.wait_idle()
    assert rec.stats()["saved"] == 1

    params._values["CarrotTlvMode"] = 2
    runner._poll_system(t + 2.0)
    assert rec.active is False
    assert offer(rec, 101.0, dets=[make_det()]) is False
    assert rec.wait_idle()
    assert rec.stats()["saved"] == 1
  finally:
    runner.close_recorder()


def test_mode_2_to_1_starts_recording(tmp_path):
  """A tlvd started in mode 2 constructs the recorder lazily on a mode-1 poll."""
  params = FakeParams({"CarrotTlvMode": 2, "CarrotTlvBackend": "null"})
  runner = tlvd.TlvRunner(params, recorder_root=str(tmp_path))
  try:
    assert runner.recorder is None
    t = time.monotonic() + 60.0
    runner._poll_system(t)
    assert runner.recorder is None

    params._values["CarrotTlvMode"] = 1
    runner._poll_system(t + 2.0)
    rec = runner.recorder
    assert rec is not None and rec.active is True
    assert offer(rec, 100.0, dets=[make_det()]) is True
    assert rec.wait_idle()
    assert rec.stats()["saved"] == 1
  finally:
    runner.close_recorder()


def test_mode_1_to_2_to_1_roundtrip(tmp_path):
  """Reactivation records again without a spurious state-transition save."""
  params = FakeParams({"CarrotTlvMode": 1, "CarrotTlvBackend": "null"})
  runner = tlvd.TlvRunner(params, recorder_root=str(tmp_path))
  try:
    rec = runner.recorder
    assert rec is not None
    t = time.monotonic() + 60.0

    assert offer(rec, 100.0, dets=[make_det()]) is True
    assert rec.wait_idle()

    params._values["CarrotTlvMode"] = 2
    runner._poll_system(t)
    assert rec.active is False
    assert offer(rec, 101.0, dets=[make_det()]) is False

    params._values["CarrotTlvMode"] = 1
    runner._poll_system(t + 2.0)
    assert rec.active is True
    assert offer(rec, 102.0, dets=[make_det()]) is True
    assert rec.wait_idle()
    assert rec.stats()["saved"] == 2
    assert [m["reason"] for m in read_index(str(tmp_path))] == ["light", "light"]
  finally:
    runner.close_recorder()


# ---------------------------------------------------------------------------
# file layout and metadata
# ---------------------------------------------------------------------------

def test_jpeg_and_index_metadata(tmp_path):
  rec = make_recorder(tmp_path)
  assert offer(rec, 100.0, dets=[make_det(cls=3, score=0.75)], tracks=[make_track(vote=(123.0, 0, 0.8))],
               state=2, state_conf=0.66, red_likely=True, relevant_track_id=1) is True
  assert rec.wait_idle()
  rec.close()

  entries = read_index(str(tmp_path))
  assert len(entries) == 1
  meta = entries[0]
  expected = {"file", "t_mono", "wall_time_iso", "frame_id", "view", "raised", "geom", "horizon_y",
              "calib_rpy", "v_ego", "a_ego", "yaw_rate", "steering_angle_deg", "left_blinker",
              "right_blinker", "dets", "gated", "tracks", "relevant_track_id", "state", "state_conf",
              "red_likely", "reason"}
  assert expected <= set(meta), sorted(expected - set(meta))
  assert meta["gated"] == []  # no tail-gate drops without the gate
  # first processed frame with state != none is a state transition (0 -> 2)
  assert meta["reason"] == "transition"
  assert meta["state"] == 2
  assert meta["red_likely"] is True
  assert meta["relevant_track_id"] == 1
  assert meta["view"] == "far" and meta["raised"] is False
  assert meta["geom"]["model_w"] == 640 and meta["geom"]["model_h"] == 320
  assert meta["geom"]["x0"] == GEOM.x0 and meta["geom"]["scale"] == 1
  assert meta["dets"] == [{"cls": 3, "score": 0.75, "x1": 300.0, "y1": 200.0, "x2": 340.0, "y2": 240.0}]
  assert meta["tracks"][0]["id"] == 1
  assert meta["tracks"][0]["vote_cls"] == 0
  assert abs(meta["tracks"][0]["vote_prob"] - 0.8) < 1e-6
  assert meta["left_blinker"] is False and meta["right_blinker"] is False

  banned = {"gps", "latitude", "longitude", "lat", "lon", "location", "coords"}
  keys: set[str] = set()

  def walk(obj):
    if isinstance(obj, dict):
      for key, value in obj.items():
        keys.add(key.lower())
        walk(value)
    elif isinstance(obj, list):
      for value in obj:
        walk(value)

  walk(meta)
  assert not (keys & banned), sorted(keys & banned)

  # the referenced JPEG exists and is a 320x640 3-channel image
  image_path = os.path.join(str(tmp_path), session_dirs(str(tmp_path))[0], meta["file"])
  image = cv2.imread(image_path)
  assert image is not None
  assert image.shape == (320, 640, 3)
  assert abs(float(image.mean()) - 16.0) < 6.0


def test_gated_dets_recorded_in_meta(tmp_path):
  """Flat-road tail-light gate drops are auditable: "gated" mirrors "dets"."""
  rec = make_recorder(tmp_path)
  kept = make_det(cls=0, score=0.9)
  dropped = tracker.Detection(cls=2, score=0.4, x1=600.0, y1=800.0, x2=620.0, y2=820.0)
  assert offer(rec, 100.0, dets=[kept], gated=[dropped]) is True
  assert rec.wait_idle()
  rec.close()
  meta = read_index(str(tmp_path))[0]
  assert meta["dets"] == [{"cls": 0, "score": 0.9, "x1": 300.0, "y1": 200.0, "x2": 340.0, "y2": 240.0}]
  assert meta["gated"] == [{"cls": 2, "score": 0.4, "x1": 600.0, "y1": 800.0, "x2": 620.0, "y2": 820.0}]


def test_offer_never_raises_and_recovers(tmp_path):
  rec = make_recorder(tmp_path)
  bad = np.zeros((1, 3, 8, 8), dtype=np.uint8)
  assert offer(rec, 100.0, dets=[make_det()], model_input=bad) is False
  assert rec.offer(None, now=101.0, frame_id=1, view="far", raised=False, geom=GEOM,
                   horizon_y=600.0, calib_rpy=None, v_ego=10.0, a_ego=None, yaw_rate=None,
                   steering_angle_deg=None, left_blinker=None, right_blinker=None,
                   dets=[make_det()], tracks=[], relevant_track_id=-1, state=0,
                   state_conf=0.0, red_likely=False) is False
  assert offer(rec, 102.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 1
  rec.close()


# ---------------------------------------------------------------------------
# storage caps
# ---------------------------------------------------------------------------

def test_size_cap_deletes_oldest_session_first(tmp_path):
  (tmp_path / "20240101-000000").mkdir()
  (tmp_path / "20240101-000000" / "a.jpg").write_bytes(b"x" * 6000)
  (tmp_path / "20240102-000000").mkdir()
  (tmp_path / "20240102-000000" / "b.jpg").write_bytes(b"x" * 2000)
  rec = recorder.TlvRecorder(root=str(tmp_path), statvfs_fn=FakeDisk(64 * GB),
                             max_root_bytes=4000, encode_fn=lambda _rgb: b"j" * 500)
  assert offer(rec, 100.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert not (tmp_path / "20240101-000000").exists()
  assert (tmp_path / "20240102-000000").exists()
  assert rec.stats()["paused"] is None
  rec.close()


def test_current_session_over_cap_stops_recording(tmp_path):
  rec = recorder.TlvRecorder(root=str(tmp_path), statvfs_fn=FakeDisk(64 * GB),
                             max_root_bytes=100, encode_fn=lambda _rgb: b"j" * 5000)
  assert offer(rec, 100.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 1
  assert rec.stats()["paused"] == "size_cap"
  assert offer(rec, 101.0, dets=[make_det()]) is False
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 1
  rec.close()


def test_free_space_pause_and_resume(tmp_path):
  disk = FakeDisk(4 * GB)
  rec = recorder.TlvRecorder(root=str(tmp_path), statvfs_fn=disk)
  assert wait_until(lambda: rec.stats()["paused"] == "disk_low")
  assert offer(rec, 100.0, dets=[make_det()]) is False
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 0

  disk.free_bytes = 7 * GB
  assert wait_until(lambda: rec.stats()["paused"] is None)
  assert offer(rec, 101.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 1
  rec.close()


# ---------------------------------------------------------------------------
# writer exception containment
# ---------------------------------------------------------------------------

def test_writer_exception_does_not_propagate(tmp_path):
  calls = {"n": 0}

  def flaky_encode(_rgb):
    calls["n"] += 1
    if calls["n"] <= 2:
      raise RuntimeError("boom")
    return b"jpeg-data"

  rec = make_recorder(tmp_path, encode_fn=flaky_encode)
  assert offer(rec, 100.0, dets=[make_det()]) is True
  assert offer(rec, 101.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 0
  assert offer(rec, 102.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 1
  rec.close()


def test_close_drains_queue_within_timeout(tmp_path):
  def slow_encode(_rgb):
    time.sleep(0.01)
    return b"jpeg"

  rec = make_recorder(tmp_path, encode_fn=slow_encode)
  for i in range(4):
    offer(rec, 100.0 + 1.0 * i, dets=[make_det()])
  started = time.monotonic()
  rec.close(timeout_s=1.0)
  assert time.monotonic() - started < 1.0 + 0.5
  assert rec.stats()["saved"] == 4


# ---------------------------------------------------------------------------
# hot-path allocation
# ---------------------------------------------------------------------------

def test_offer_no_allocation_growth(tmp_path):
  rec = make_recorder(tmp_path, encode_fn=lambda _rgb: b"jpeg")
  det = make_det()

  def drive(count: int, t0: float) -> None:
    for i in range(count):
      if i % 4 == 0:
        # keep all four ring slots free so every offer takes the full hot path
        rec.wait_idle(2.0)
      offer(rec, t0 + i * 0.5, dets=[det])

  drive(50, 0.0)
  rec.wait_idle()
  gc.collect()
  tracemalloc.start()
  drive(50, 1000.0)  # tracemalloc amortizes its own tables over these
  rec.wait_idle()
  gc.collect()
  before = tracemalloc.get_traced_memory()[0]
  drive(500, 2000.0)
  assert rec.wait_idle()
  gc.collect()
  current, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  growth = current - before
  print(f"\nalloc[tlv-recorder]: offers=500 saved={rec.stats()['saved']} net_growth={growth / 1024:.1f} KB peak_delta={(peak - before) / 1024:.1f} KB")
  assert rec.stats()["saved"] == 50 + 50 + 500
  assert growth < 256 * 1024, f"net growth {growth / 1024:.1f} KB >= 256 KB"
  rec.close()


# ---------------------------------------------------------------------------
# session rotation and .active heartbeat
# ---------------------------------------------------------------------------

def test_session_rotates_after_ten_minutes(tmp_path):
  now = [1000.0]
  names = iter(["20260101-000000", "20260101-010000"])
  rec = make_recorder(tmp_path, clock=lambda: now[0], session_name_fn=lambda: next(names),
                      encode_fn=lambda _rgb: b"jpeg")
  assert offer(rec, 100.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["session"] == "20260101-000000"

  now[0] += recorder.SESSION_ROTATE_S
  assert offer(rec, 200.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["session"] == "20260101-010000"
  assert rec.stats()["saved"] == 2
  assert session_dirs(str(tmp_path)) == ["20260101-000000", "20260101-010000"]
  assert os.path.isfile(os.path.join(str(tmp_path), "20260101-000000", "0000000.jpg"))
  assert os.path.isfile(os.path.join(str(tmp_path), "20260101-010000", "0000001.jpg"))
  rec.close()


def test_vanished_session_is_recreated(tmp_path):
  now = [1000.0]
  names = iter(["20260101-000000", "20260101-000001"])
  rec = make_recorder(tmp_path, clock=lambda: now[0], session_name_fn=lambda: next(names),
                      encode_fn=lambda _rgb: b"jpeg")
  assert offer(rec, 100.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  shutil.rmtree(os.path.join(str(tmp_path), "20260101-000000"))

  assert offer(rec, 101.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["saved"] == 2
  assert rec.stats()["session"] == "20260101-000001"
  assert session_dirs(str(tmp_path)) == ["20260101-000001"]
  rec.close()


def test_active_heartbeat_written_refreshed_and_removed(tmp_path):
  now = [1000.0]
  rec = make_recorder(tmp_path, clock=lambda: now[0], session_name_fn=lambda: "20260101-000000",
                      encode_fn=lambda _rgb: b"jpeg")
  assert offer(rec, 100.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  active_path = os.path.join(str(tmp_path), recorder.ACTIVE_NAME)
  with open(active_path, encoding="utf-8") as f:
    assert f.read() == "20260101-000000"
  first_mtime = os.path.getmtime(active_path)

  time.sleep(0.1)
  now[0] += recorder.ACTIVE_HEARTBEAT_S + 1.0
  assert offer(rec, 101.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  with open(active_path, encoding="utf-8") as f:
    assert f.read() == "20260101-000000"
  assert os.path.getmtime(active_path) > first_mtime

  rec.close()
  assert not os.path.exists(active_path)


def test_session_name_collision_bumps_second(tmp_path):
  (tmp_path / "20260101-000000").mkdir()
  rec = make_recorder(tmp_path, session_name_fn=lambda: "20260101-000000",
                      encode_fn=lambda _rgb: b"jpeg")
  assert offer(rec, 100.0, dets=[make_det()]) is True
  assert rec.wait_idle()
  assert rec.stats()["session"] == "20260101-000001"
  assert session_dirs(str(tmp_path)) == ["20260101-000000", "20260101-000001"]
  assert os.listdir(str(tmp_path / "20260101-000000")) == []
  rec.close()
