from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest

from openpilot.cereal import car
from opendbc.car.interfaces import CarInterfaceBase
from openpilot.selfdrive.car.cruise import VCruiseCarrot
from openpilot.selfdrive.carrot.carrot_controls import CarrotControls
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque


class _CountingParams:
  def __init__(self, values: dict):
    self.values = dict(values)
    self.reads: list[str] = []
    self.writes: list[tuple[str, object]] = []

  def get_int(self, key: str) -> int:
    self.reads.append(key)
    return int(self.values.get(key, 0))

  def get_float(self, key: str) -> float:
    self.reads.append(key)
    return float(self.values.get(key, 0.0))

  def get_bool(self, key: str) -> bool:
    self.reads.append(key)
    return bool(self.values.get(key, False))

  def put_bool_nonblocking(self, key: str, val: bool) -> None:
    self.writes.append((key, val))
    self.values[key] = val


def _make_torque_cp():
  cp = car.CarParams.new_message(brand="hyundai", steerRatio=13.0)
  torque = cp.lateralTuning.init("torque")
  torque.kp = 1.0
  torque.ki = 0.1
  torque.kf = 1.0
  torque.latAccelFactor = 2.5
  torque.latAccelOffset = 0.05
  torque.friction = 0.08
  torque.useSteeringAngle = True
  return cp.as_reader()


def test_latcontrol_torque_custom_1_to_0_resets_torque_and_pid_gains_and_throttles():
  cp = _make_torque_cp()
  ci = SimpleNamespace(
    use_nnff=False,
    use_nnff_lite=False,
    torque_from_lateral_accel=lambda: partial(CarInterfaceBase.torque_from_lateral_accel_linear, None),
  )
  lac = LatControlTorque(cp, ci)
  fake_params = _CountingParams({
    "LateralTorqueCustom": 1,
    "LateralTorqueAccelFactor": 3500.0,
    "LateralTorqueFriction": 150.0,
    "LateralTorqueKpV": 200.0,
    "LateralTorqueKiV": 30.0,
    "LateralTorqueKf": 120.0,
    "LateralTorqueKd": 15.0,
  })
  lac.params = fake_params

  # Enable custom mode (0 -> 1)
  lac._refresh_custom_torque_params()
  assert lac.lateralTorqueCustom == 1
  assert lac.torque_params.latAccelFactor == pytest.approx(3.5)
  assert lac.torque_params.friction == pytest.approx(0.15)
  assert lac.pid._k_p == [[0], [pytest.approx(2.0)]]
  assert lac.pid._k_i == [[0], [pytest.approx(0.3)]]
  assert lac.pid.k_f == pytest.approx(1.2)
  assert lac.pid._k_d == [[0], [pytest.approx(0.15)]]

  # Turn custom mode OFF (1 -> 0): previously `elif self.lateralTorqueCustom > 1:` skipped reset!
  fake_params.values["LateralTorqueCustom"] = 0
  lac._refresh_custom_torque_params()
  assert lac.lateralTorqueCustom == 0
  assert lac.torque_params.latAccelFactor == pytest.approx(2.5)
  assert lac.torque_params.friction == pytest.approx(0.08)
  assert lac.torque_params.latAccelOffset == pytest.approx(0.05)
  assert lac.pid._k_p == [[0], [pytest.approx(1.0)]]
  assert lac.pid._k_i == [[0], [pytest.approx(0.1)]]
  assert lac.pid.k_f == pytest.approx(1.0)
  assert lac.pid._k_d == [[0], [pytest.approx(0.0)]]

  # Verify 1 Hz throttling inside update() (99 ticks -> 0 reads; 100th tick -> 1 read)
  fake_params.reads.clear()
  lac.frame = 0
  cs = SimpleNamespace(steeringAngleDeg=0.0, vEgo=10.0)
  cc = SimpleNamespace(angularVelocity=[], orientationNED=[])
  for _ in range(99):
    lac.update(False, cs, None, None, False, 0.0, cc, False)
  assert fake_params.reads == []
  lac.update(False, cs, None, None, False, 0.0, cc, False)
  assert fake_params.reads == ["LateralTorqueCustom"]


def test_carrot_controls_caches_suspend_angle_at_1hz():
  cc = CarrotControls.__new__(CarrotControls)
  cc.CP = None
  cc.params = _CountingParams({"LatSuspendAngleDeg": 90})
  cc.lat_suspend_active = False
  cc.lat_suspend_enter_t = 0.0
  cc.lat_suspend_hold_t = 0.0
  cc._suspend_angle = float(cc.params.get_int("LatSuspendAngleDeg"))
  cc._param_frame = 0
  cc.params.reads.clear()

  cs = SimpleNamespace(steeringPressed=False, steeringAngleDeg=0.0)
  for _ in range(99):
    assert cc.lat_suspend_control(cs, True) is True
  assert cc.params.reads == []

  cc.params.values["LatSuspendAngleDeg"] = 45
  assert cc.lat_suspend_control(cs, True) is True
  assert cc.params.reads == ["LatSuspendAngleDeg"]
  assert cc._suspend_angle == 45.0


def test_cruise_skips_gm_activate_after_brake_read_on_non_gm_and_throttles_params():
  def _make_cruise(brand: str, params: _CountingParams) -> VCruiseCarrot:
    vc = VCruiseCarrot.__new__(VCruiseCarrot)
    vc.CP = SimpleNamespace(brand=brand)
    vc.params = params
    vc.frame = 0
    vc._brake_pressed_count = 0
    vc._gas_pressed_count = 0
    vc._gas_tok_timer = 40
    vc._soft_hold_active = 0
    vc.disengage_on_accelerator = True
    vc._gas_tok = False
    vc._resume_pending = None
    vc._update_resume_pending = lambda cs, cc: None
    vc._cruise_control = lambda *a, **k: None
    vc.useLaneLineSpeed = 0
    vc.useLaneLineSpeedApply = 0
    vc.nRoadLimitSpeed = 60
    vc._cruise_ready = False
    vc._auto_speed_up = lambda v: v
    return vc

  cs = SimpleNamespace(steeringAngleDeg=0.0, gasPressed=False, brakePressed=False, aEgo=0.0, vEgo=10.0)
  cc = SimpleNamespace(enabled=False)

  hyundai_params = _CountingParams({"ActivateCruiseAfterBrake": True})
  hyundai_vc = _make_cruise("hyundai", hyundai_params)
  hyundai_vc._update_cruise_state(cs, cc, 60.0)
  assert "ActivateCruiseAfterBrake" not in hyundai_params.reads

  gm_params = _CountingParams({"ActivateCruiseAfterBrake": True})
  gm_vc = _make_cruise("gm", gm_params)
  gm_vc._update_cruise_state(cs, cc, 60.0)
  assert "ActivateCruiseAfterBrake" in gm_params.reads
  assert ("ActivateCruiseAfterBrake", False) in gm_params.writes

  # update_params throttles to every 50 frames (2 Hz)
  hyundai_params.reads.clear()
  for f in range(1, 50):
    hyundai_vc.frame = f
    hyundai_vc.update_params(True)
  assert hyundai_params.reads == []
  hyundai_vc.frame = 50
  hyundai_vc.update_params(True)
  assert len(hyundai_params.reads) > 0
