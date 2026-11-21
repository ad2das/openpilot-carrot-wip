from unittest.mock import patch

from opendbc.car import gen_empty_fingerprint
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, HyundaiSafetyFlags


class Settings:
  def get_int(self, key):
    return 0

  def get_bool(self, key):
    return False

  def get(self, key):
    return ""

  def put_bool(self, key, value):
    pass


def car_params(candidate, cgw1):
  fingerprint = gen_empty_fingerprint()
  if cgw1:
    fingerprint[0][0x541] = 8
  with patch("opendbc.car.hyundai.interface.Params", Settings):
    return CarInterface.get_params(candidate, fingerprint, [], False, False, False)


def flag(candidate, cgw1):
  return car_params(candidate, cgw1).safetyConfigs[-1].safetyParam & HyundaiSafetyFlags.PALISADE_CGW_TEST


def test_palisade_with_a_stock_cgw1_gets_the_flag():
  assert flag(CAR.HYUNDAI_PALISADE, True) == HyundaiSafetyFlags.PALISADE_CGW_TEST


def test_palisade_without_a_stock_cgw1_does_not():
  assert not flag(CAR.HYUNDAI_PALISADE, False)


def test_other_classic_cars_keep_their_param():
  assert not flag(CAR.KIA_K5_2021, True)


def test_legacy_cars_never_get_it():
  assert not flag(CAR.HYUNDAI_VELOSTER, True)
