from openpilot.system.hardware.hardwared import (
  BOOT_THERMAL_GRACE_S, OFFROAD_DANGER_TEMP, ThermalStatus, update_thermal_status,
)


def test_boot_load_does_not_trip_the_offroad_override():
  # Boot heats the SoC just past 75 C; within the grace window only the normal bands apply.
  status = update_thermal_status(ThermalStatus.yellow, 78., 78., True, 5.)
  assert status == ThermalStatus.yellow


def test_offroad_override_returns_after_the_grace_window():
  assert update_thermal_status(ThermalStatus.yellow, OFFROAD_DANGER_TEMP + 1., 78., True, BOOT_THERMAL_GRACE_S) == ThermalStatus.danger
  assert update_thermal_status(ThermalStatus.yellow, OFFROAD_DANGER_TEMP + 1., 78., True, BOOT_THERMAL_GRACE_S - 0.5) == ThermalStatus.yellow


def test_real_overheat_during_boot_still_escalates():
  status = update_thermal_status(ThermalStatus.yellow, 97., 97., True, 5.)
  assert status == ThermalStatus.red
  status = update_thermal_status(status, 108., 108., True, 5.5)
  assert status == ThermalStatus.danger


def test_bands_are_unchanged_onroad():
  assert update_thermal_status(ThermalStatus.green, 82., 82., False, 0.) == ThermalStatus.yellow
  assert update_thermal_status(ThermalStatus.yellow, 90., 90., False, 0.) == ThermalStatus.yellow
  assert update_thermal_status(ThermalStatus.yellow, 74., 74., False, 0.) == ThermalStatus.green


def test_danger_steps_down_one_band_per_update():
  status = ThermalStatus.danger
  seen = []
  for _ in range(3):
    status = update_thermal_status(status, 70., 70., False, 1000.)
    seen.append(status)
  assert seen == [ThermalStatus.red, ThermalStatus.yellow, ThermalStatus.green]
