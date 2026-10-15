"""W125 real-schema tests: tlvd must run against actual cereal messages.

Field route 00000441: tlvd crash-looped on the device because
  (1) the launcher inherited launch_env.sh's QCOM_PRIORITY=12 and
      backends.gpu_env_reason() refused to build any backend, and
  (2) _feed_watchdog called len() on modelV2.modelExecutionTime, which is a
      scalar Float32 (TypeError -> crash -> crash-loop guard cleared the mode).
The old tests faked modelV2 with a list, so neither bug was caught. These tests
build REAL messages with cereal.messaging.new_message and run the runner
helpers the way the device loop does, plus the launcher env scrub.
"""
import time

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.carrot.tlv import backends, crops, pipeline, tlvd, tlvd_launch, tracker


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


class FakeSM:
  """Minimal SubMaster: service readers plus updated/seen/recv_time/logMonoTime."""

  def __init__(self, services, now: float) -> None:
    self._services = dict(services)
    names = tuple(services)
    self.seen = dict.fromkeys(names, True)
    self.updated = dict.fromkeys(names, True)
    self.valid = dict.fromkeys(names, True)
    self.alive = dict.fromkeys(names, True)
    self.recv_time = dict.fromkeys(names, now)
    self.logMonoTime = {"controlsState": 950_000_000}

  def __getitem__(self, service):
    return self._services[service]


def build_runner():
  return tlvd.TlvRunner(FakeParams({"CarrotTlvBackend": "null"}))


def real_services(now: float):
  model = messaging.new_message("modelV2")
  m = model.modelV2
  m.modelExecutionTime = 0.031
  m.frameDropPerc = 0.5
  m.position.init("x", 33)
  m.position.init("y", 33)
  for i in range(33):
    m.position.x[i] = float(i)
    m.position.y[i] = 0.0

  device = messaging.new_message("deviceState")
  device.deviceState.thermalStatus = "green"

  controls = messaging.new_message("controlsState")
  controls.logMonoTime = 950_000_000

  camera = messaging.new_message("roadCameraState")
  camera.roadCameraState.frameId = 123

  car = messaging.new_message("carState")
  car.carState.vEgo = 10.0  # m/s -> 36 km/h
  car.carState.aEgo = 0.0
  car.carState.yawRate = 0.01
  car.carState.steeringAngleDeg = 0.0
  car.carState.leftBlinker = True
  car.carState.rightBlinker = False

  carrot_man = messaging.new_message("carrotMan")
  carrot_man.carrotMan.atcType = "atc left"

  calib = messaging.new_message("liveCalibration")
  calib.liveCalibration.rpyCalib = [0.0, 0.1, 0.0]
  calib.liveCalibration.calStatus = "calibrated"

  return FakeSM({
    "modelV2": model.modelV2,
    "deviceState": device.deviceState,
    "controlsState": controls.controlsState,
    "roadCameraState": camera.roadCameraState,
    "carState": car.carState,
    "carrotMan": carrot_man.carrotMan,
    "liveCalibration": calib.liveCalibration,
  }, now)


def test_runner_helpers_accept_real_cereal_messages(monkeypatch):
  runner = build_runner()
  now = time.monotonic()
  runner.sm = real_services(now)

  updates = []
  real_update = runner.watchdog.update
  monkeypatch.setattr(runner.watchdog, "update",
                      lambda t, **kw: (updates.append(kw), real_update(t, **kw))[1])

  # Every helper the device loop calls that reads sm[...] must not raise.
  horizon = runner._update_horizon()
  assert horizon == pytest.approx(horizon)  # finite
  assert runner._path_x() == tuple(float(i) for i in range(33))
  assert runner._path_y() == tuple(0.0 for _ in range(33))
  assert runner._left_intent() is True
  assert runner._v_ego_kph() == pytest.approx(36.0)
  runner._update_odometry(now)
  assert runner.odo._v == pytest.approx(10.0)
  runner._feed_watchdog(now)

  # The scalar Float32 must reach the watchdog as seconds (modeld.py:546
  # publishes mt3 - mt1 from perf_counter -> seconds).
  assert updates and updates[-1]["exec_time"] == pytest.approx(0.031)
  assert updates[-1]["frame_drop"] == pytest.approx(0.5)
  assert updates[-1]["thermal"] == 0  # real _DynamicEnum -> ordinal, not int()
  assert updates[-1]["camera_frame_id"] == 123

  # Second feed exercises the controls logMonoTime delta branch.
  runner.sm.logMonoTime["controlsState"] = 960_000_000
  runner._feed_watchdog(now + 0.05)
  assert updates[-1]["controls_interval"] == pytest.approx(0.01)


def test_offer_recorder_accepts_real_cereal_messages():
  runner = build_runner()
  now = time.monotonic()
  runner.sm = real_services(now)

  offers = []

  class FakeRecorder:
    def offer(self, model_input, **kw):
      offers.append((model_input, kw))

  runner.recorder = FakeRecorder()
  horizon = runner._update_horizon()
  width, height = crops.REFERENCE_WIDTH, crops.REFERENCE_HEIGHT
  focal = crops.focal_px_for_width(width)
  geom = pipeline.FramePipeline.geometry("far", horizon, width, height, focal)
  out = pipeline.FrameOutput(view="far", dets=[], matches=[],
                             state=tracker.SignalState.none, state_conf=0.0,
                             track_id=-1, infer_ms=0.0, proc_ms=0.0)
  runner._offer_recorder(out, geom, "far", False, horizon)

  assert len(offers) == 1
  _model_input, kw = offers[0]
  assert kw["frame_id"] == 123  # roadCameraState.frameId, the real UInt32
  assert kw["v_ego"] == pytest.approx(10.0)
  assert kw["left_blinker"] is True


# ---------------------------------------------------------------------------
# launcher env scrub (field crash: inherited QCOM_PRIORITY=12)
# ---------------------------------------------------------------------------

def test_scrub_constants_cover_the_backend_guard():
  assert set(tlvd_launch.SCRUB_ENV_PREFIXES) == set(backends.GPU_ENV_PREFIXES)
  for key in backends.GPU_ENV_KEYS:
    if not any(key.startswith(p) for p in backends.GPU_ENV_PREFIXES):
      assert key in tlvd_launch.SCRUB_ENV_KEYS, f"{key} not covered by a prefix or SCRUB_ENV_KEYS"


def test_set_process_env_scrubs_gpu_env_and_reenables_backend(monkeypatch):
  monkeypatch.setenv("QCOM_PRIORITY", "12")
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
  monkeypatch.setenv("DEV", "GPU")
  assert backends.gpu_env_reason() is not None  # the field failure state

  tlvd_launch.set_process_env()

  assert "QCOM_PRIORITY" not in __import__("os").environ
  assert "CUDA_VISIBLE_DEVICES" not in __import__("os").environ
  assert __import__("os").environ["DEV"] == "CPU"
  assert backends.gpu_env_reason() is None
  assert tlvd_launch.SCRUB_ENV_KEYS  # sanity: the exact-key list stays
