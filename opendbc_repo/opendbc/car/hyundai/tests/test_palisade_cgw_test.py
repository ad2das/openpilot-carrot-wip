"""Host side of the Palisade CGW1 parked blinker test (palisade_cgw_test.py, remove after phase 1).

100 Hz ticks of 10 ms; every timestamp equals the tick time. STOCK is the neutral stock CGW1 frame. A flash is byte 4
bit 0 (CF_Gway_HeadLampHigh, HB) or byte 6 bit 3 (CF_Gway_PassingSW, PS) on for 0.02 to 1.5 s; on this car only the
lamp bit was ever seen. Every number below is the model run of the reference module, W-BLK5/raw/opus_speccheck.txt.
"""
from types import SimpleNamespace

import pytest

from opendbc.car import Bus, structs
from opendbc.car.hyundai import carcontroller
from opendbc.car.hyundai.palisade_cgw_test import PalisadeCgwTest, cgw_frame
from opendbc.car.hyundai.values import CAR, HyundaiSafetyFlags

STOCK = bytes([0x01, 0, 0, 0x80, 0, 0, 0, 0])
HB = bytes([0x01, 0, 0, 0x80, 0x01, 0, 0, 0])      # byte4 0x01 HeadLampHigh
PS = bytes([0x01, 0, 0, 0x80, 0, 0, 0x08, 0])      # byte6 0x08 PassingSW
BOTH = bytes([0x01, 0, 0, 0x80, 0x01, 0, 0x08, 0])
HAZ_HB = bytes([0x01, 0, 0, 0x80, 0x03, 0, 0, 0])  # hazard switch 0x02 + high beam
MS = 1_000_000
TICK = 10 * MS
STALE = 300 * MS


class Run:
  """Ticks a PalisadeCgwTest at 10 ms from 10 ms; records (ms since start, frame) for every frame it returns."""

  def __init__(self):
    self.m = PalisadeCgwTest()
    self.t = 10 * MS
    self.out = []
    self.parked = lambda ms: True
    self.inactive = lambda ms: True
    self.stale = lambda ms: False

  def feed(self, raw, ms):
    for _ in range(ms // 10):
      ms_now = self.t // MS
      ts = (self.t - STALE,) * 3 if self.stale(ms_now) else (self.t,) * 3
      for f in self.m.update(self.t, raw, ts, self.parked(ms_now), self.inactive(ms_now)):
        self.out.append((ms_now, f))
      self.t += TICK
    return self

  def frames(self):
    return [f for _, f in self.out]

  def starts(self):  # trial starts = first frame, or a frame >= 1 s after the previous one
    s, prev = [], None
    for ms, _ in self.out:
      if prev is None or ms - prev >= 1000:
        s.append(ms)
      prev = ms
    return s

  def last(self):
    return self.out[-1][0]


def flash(raw=HB, on=140, pre=1500, post=4000):
  return Run().feed(STOCK, pre).feed(raw, on).feed(STOCK, post)


def test_n1_flash_on_the_lamp_bit():
  # S1: 50 frames, 1650..4590, every gap 60 ms, lamp 27 on / 23 off, byte5 0x04, nothing else changed
  run = flash()
  assert len(run.out) == 50
  assert run.out[0][0] == 1650
  assert run.last() == 4590
  assert sorted({b[0] - a[0] for a, b in zip(run.out, run.out[1:])}) == [60]
  assert {f[1][5] for f in run.frames()} == {0x04}
  lamps = [f[1][2] & 0x08 for f in run.frames()]
  assert (lamps.count(0x08), lamps.count(0)) == (27, 23)
  for t, frame in run.out:
    lamp = (t - 1650) // 350 % 2 == 0
    assert frame == cgw_frame(STOCK, 1, lamp)
    data = frame[1]
    assert frame[0] == 0x541
    # every byte is STOCK except byte 2 bit 3 (the lamp) and byte 5 (the lever side)
    assert data[5] == 0x04
    assert data[:2] == STOCK[:2] and data[3:5] == STOCK[3:5] and data[6:] == STOCK[6:]
    assert data[2] & ~0x08 == STOCK[2]


def test_n2_flash_on_the_switch_bit():
  # S2: the switch bit fires the same 50 frames
  assert len(flash(PS).out) == 50


def test_n3_both_bits_fire_once():
  # S3: both bits on for 140 ms, then the switch bit alone 10 ms longer: one trial, from the switch's release at 1660 ms
  run = Run().feed(STOCK, 1500).feed(BOTH, 140).feed(PS, 10).feed(STOCK, 4000)
  assert run.starts() == [1660]


@pytest.mark.parametrize("on_ms,frames", ((10, 0), (20, 50), (30, 50), (1500, 50), (1510, 0), (1600, 0)))
def test_n4_pulse_width(on_ms, frames):
  # S4/S4b: 0.02-1.5 s fires, 10 ms is too short, over 1.5 s is a switch-on and never fires
  assert len(flash(on=on_ms).out) == frames


def test_n5_held_beam_never_fires_and_needs_1s_off():
  # S5a: a 1.6 s beam, then a flash 0.5 s after it: nothing
  assert len(Run().feed(STOCK, 1500).feed(HB, 1600).feed(STOCK, 500).feed(HB, 140).feed(STOCK, 3000).out) == 0
  # S5b: 1.1 s after it: the flash fires, one trial
  run = Run().feed(STOCK, 1500).feed(HB, 1600).feed(STOCK, 1100).feed(HB, 140).feed(STOCK, 3500)
  assert run.starts() == [4350]


def test_n6_beam_on_at_start_or_off_under_1s():
  # S6: the beam is on at the first frame and released at 0.5 s: nothing
  assert len(Run().feed(HB, 500).feed(STOCK, 3000).out) == 0
  # S6b: off for only 0.6 s before the flash: nothing
  assert len(Run().feed(STOCK, 600).feed(HB, 140).feed(STOCK, 3000).out) == 0


@pytest.mark.parametrize("gate", ("parked_late", "parked_mid", "inactive_mid"))
def test_n7_gate_must_hold_for_the_whole_flash(gate):
  # S7a/b/c: not parked at the rising edge, parked dropping mid-flash, engaged mid-flash: nothing fires
  run = Run()
  if gate == "parked_late":
    run.parked = lambda ms: ms >= 1580
  elif gate == "parked_mid":
    run.parked = lambda ms: not (1540 <= ms < 1560)
  else:
    run.inactive = lambda ms: ms < 1550
  assert len(run.feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 3000).out) == 0


def test_n8_flash_during_or_right_after_a_trial():
  # S8a: a flash during the trial is ignored; the trial's last frame is still at 4590 ms
  run = Run().feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 1000).feed(HB, 140).feed(STOCK, 2500)
  assert run.starts() == [1650]
  assert run.last() == 4590
  # S8b: 0.5 s after the trial's last frame: ignored
  assert Run().feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 3500).feed(HB, 140).feed(STOCK, 4000).starts() == [1650]
  # S8c: 1.1 s after it: a second trial, 1.3 s after the first trial's last frame
  run = Run().feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 4100).feed(HB, 140).feed(STOCK, 4000)
  assert run.starts() == [1650, 5890]


def test_n9_hazard_during_the_flash():
  # S9: hazard + high beam from 1500 ms: nothing
  assert len(Run().feed(STOCK, 1500).feed(HAZ_HB, 140).feed(STOCK, 3000).out) == 0


def test_n10_stale():
  # S10a: stale until 1.3 s, so the trigger is only 0.2 s fresh when the flash starts: nothing
  run = Run()
  run.stale = lambda ms: 1000 <= ms < 1300
  assert len(run.feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 3000).out) == 0
  # S10b: 0.1 s stale mid-trial: the trial is dropped at 2490 ms
  run = Run()
  run.stale = lambda ms: 2500 <= ms < 2600
  run.feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 3000)
  assert len(run.out) == 15
  assert run.last() == 2490
  # S13: never fresh: nothing
  run = Run()
  run.stale = lambda ms: ms >= 10
  assert len(run.feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 3000).out) == 0


@pytest.mark.parametrize("parked", (lambda ms: ms < 2500, lambda ms: not 2500 <= ms < 2700))
def test_n11_parked_lost_mid_trial(parked):
  # S11: parked lost from 2500 ms, or only 2500-2700 ms: the trial stops at 2490 ms
  run = Run()
  run.parked = parked
  run.feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 4000)
  assert len(run.out) == 15
  assert run.last() == 2490


@pytest.mark.parametrize("index,bit", ((5, 0x02), (5, 0x04), (4, 0x02), (4, 0x04), (2, 0x08), (7, 0x40)))
def test_n12_stock_bits_stop_a_trial(index, bit):
  # S12: a stock turn switch (byte 5 0x02/0x04), hazard bit (byte 4 0x02/0x04) or lamp bit (byte 2 0x08, byte 7 0x40) at 2500-2550 ms
  bad = bytes(STOCK[:index] + bytes([STOCK[index] | bit]) + STOCK[index + 1:])
  run = Run().feed(STOCK, 1500).feed(HB, 140).feed(STOCK, 850).feed(bad, 50).feed(STOCK, 3000)
  assert len(run.out) == 15
  assert run.last() == 2490


@pytest.mark.parametrize("flag_set", (True, False))
def test_t7_carcontroller_builds_the_test_only_with_the_flag(monkeypatch, flag_set):
  params = SimpleNamespace(get_int=lambda key: 0, get_bool=lambda key: False, get_float=lambda key: 0.0)
  monkeypatch.setattr(carcontroller, "Params", lambda: params)
  safety_config = structs.CarParams.SafetyConfig()
  if flag_set:
    safety_config.safetyParam = int(HyundaiSafetyFlags.PALISADE_CGW_TEST)
  cp = structs.CarParams(carFingerprint=CAR.HYUNDAI_PALISADE, wheelbase=2.9, steerRatio=14.0,
                         safetyConfigs=[safety_config])
  controller = carcontroller.CarController({Bus.pt: "hyundai_kia_generic"}, cp)
  if flag_set:
    assert isinstance(controller.cgw_test, PalisadeCgwTest)
  else:
    assert controller.cgw_test is None
