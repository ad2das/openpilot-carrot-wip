# Palisade CGW1 blinker test, phase 1 (parked only): a 0x541 frame the host sends is held by the safety model and sent
# right behind the next stock CGW1 on bus 0 (safety_rx_tx_hook), never on its own. Only while parked, for one 3 s trial,
# re-armed after 1 s of quiet.
from opendbc.car.structs import CarParams
from opendbc.safety.tests.common import CANPackerPanda
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.test_hyundai import checksum

PARAM = 32768  # PALISADE_CGW_TEST
HOLD_US = 150_000
CGW1 = 0x541
STOCK = bytes([0x01, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00])


def frame(direction, lamp):
  data = bytearray(STOCK)
  data[2] |= 0x08 if direction == 1 and lamp else 0
  data[7] |= 0x40 if direction == 2 and lamp else 0
  data[5] |= 0x04 if direction == 1 else 0x02
  return bytes(data)


def patched(base, byte, value):
  data = bytearray(base)
  data[byte] |= value
  return bytes(data)


class Device:
  def __init__(self, param=PARAM, model=CarParams.SafetyModel.hyundai):
    self.s = libsafety_py.libsafety
    assert self.s.set_safety_hooks(model, param) == 0
    self.s.init_tests()
    self.packer = CANPackerPanda("hyundai_kia_generic")
    self.cnt_speed = 0
    self.t = 1_000_000
    self.s.set_timer(self.t)

  def advance(self, us):
    self.t = (self.t + us) & 0xFFFFFFFF
    self.s.set_timer(self.t)

  def rx_tx(self, pkt):
    out = libsafety_py.ffi.new("CANPacket_t *")
    if not self.s.safety_rx_tx_hook(pkt, out):
      return None
    assert out[0].addr == CGW1 and out[0].bus == 0
    return bytes(out[0].data)[:8]

  def stock(self, data=STOCK):
    """a stock CGW1 arrives on bus 0: what panda sends right behind it, or None"""
    pkt = libsafety_py.make_CANPacket(CGW1, 0, data)
    assert self.s.safety_rx_hook(pkt)
    return self.rx_tx(pkt)

  def gear(self, code):
    data = bytearray(8)
    data[4] = code & 0xF  # LVR12 CF_Lvr_Gear, bits 32..35
    assert self.s.safety_rx_hook(libsafety_py.make_CANPacket(0x367, 0, bytes(data)))

  def wheels(self, speeds=(0, 0, 0, 0)):
    values = {"WHL_SPD_FL": speeds[0], "WHL_SPD_FR": speeds[1], "WHL_SPD_RL": speeds[2], "WHL_SPD_RR": speeds[3]}
    values["WHL_SPD_AliveCounter_LSB"] = (self.cnt_speed % 16) & 0x3
    values["WHL_SPD_AliveCounter_MSB"] = (self.cnt_speed % 16) >> 2
    self.cnt_speed += 1
    assert self.s.safety_rx_hook(self.packer.make_can_msg_panda("WHL_SPD11", 0, values, fix_checksum=checksum))

  def park(self, data=STOCK):
    self.stock(data)
    self.gear(0)
    self.wheels()

  def tx(self, data, bus=0):
    pkt = libsafety_py.make_CANPacket(CGW1, bus, data)
    self.s.safety_tx_buffered_for_fwd = False
    return bool(self.s.safety_tx_hook(pkt)), bool(self.s.safety_tx_buffered_for_fwd)


def test_k1_parked_frame_is_held_and_sent_behind_the_next_stock_cgw1():
  for side in (1, 2):
    d = Device()
    d.park()
    d.advance(30_000)
    assert d.tx(frame(side, True)) == (True, True)
    d.advance(70_000)
    assert d.stock() == frame(side, True)
    assert d.stock() is None


def test_k2_hold_window_is_150_ms():
  d = Device()
  d.park()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (True, True)
  d.advance(HOLD_US - 1)
  assert d.stock() == frame(1, True)
  d.park()
  d.advance(60_000)
  assert d.tx(frame(2, True)) == (True, True)
  d.advance(HOLD_US)
  assert d.stock() is None


def test_k3_not_parked_rejects():
  for wheel in range(4):
    d = Device()
    d.park()
    speeds = [0, 0, 0, 0]
    speeds[wheel] = 1.0  # 1 km/h
    d.wheels(tuple(speeds))
    d.advance(30_000)
    assert d.tx(frame(1, True)) == (False, False), wheel
  for code in (5, 6, 7):  # D, N, R
    d = Device()
    d.park()
    d.gear(code)
    d.advance(30_000)
    assert d.tx(frame(1, True)) == (False, False), code

  d = Device()  # no gear frame at all
  d.stock()
  d.wheels()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (False, False)

  d = Device()  # gear older than 250 ms, everything else fresh
  d.park()
  d.advance(300_000)
  d.stock()
  d.wheels()
  assert d.tx(frame(1, True)) == (False, False)

  d = Device()  # wheels older than 250 ms, everything else fresh
  d.park()
  d.advance(300_000)
  d.stock()
  d.gear(0)
  assert d.tx(frame(1, True)) == (False, False)

  d = Device()  # engaged
  d.park()
  d.advance(30_000)
  d.s.set_controls_allowed(True)
  assert d.tx(frame(1, True)) == (False, False)


def test_k4_stock_frame_gates_reject():
  d = Device()  # no stock frame yet
  d.gear(0)
  d.wheels()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (False, False)

  d = Device()  # stock older than 250 ms
  d.park()
  d.advance(300_000)
  assert d.tx(frame(1, True)) == (False, False)

  for byte, bit in ((4, 0x02), (4, 0x04), (5, 0x04), (5, 0x02)):  # hazard, lever
    d = Device()
    d.park(patched(STOCK, byte, bit))
    d.advance(30_000)
    assert d.tx(frame(1, True)) == (False, False), (byte, bit)
  for byte, bit in ((2, 0x08), (7, 0x40)):  # a stock lamp already on
    d = Device()
    d.park(patched(STOCK, byte, bit))
    d.advance(30_000)
    assert d.tx(frame(1, True)) == (False, False), (byte, bit)


def test_k5_frame_shape_rejects():
  both = bytearray(frame(1, True))
  both[7] |= 0x40
  lever3 = patched(patched(STOCK, 5, 0x06), 2, 0x08)
  lever0 = patched(STOCK, 2, 0x08)
  for bad in (bytes(both), lever3, lever0):
    d = Device()
    d.park()
    d.advance(30_000)
    assert d.tx(bad) == (False, False), bad.hex()

  for byte in (0, 1, 3, 4, 6):
    for value in (1, 2, 4, 8, 16, 32, 64, 128):
      if STOCK[byte] & value:
        continue
      d = Device()
      d.park()
      d.advance(30_000)
      assert d.tx(patched(frame(1, True), byte, value)) == (False, False), (byte, value)


def test_k6_held_frame_is_rechecked_against_the_stock_frame_it_rides_behind():
  for byte, bit in ((5, 0x02), (4, 0x02), (7, 0x40), (0, 0x10)):
    d = Device()
    d.park()
    d.advance(30_000)
    assert d.tx(frame(1, True)) == (True, True)
    d.advance(20_000)
    assert d.stock(patched(STOCK, byte, bit)) is None, (byte, bit)

  d = Device()  # the BCM echoing our own lamp bit is fine
  d.park()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (True, True)
  d.advance(20_000)
  assert d.stock(patched(STOCK, 2, 0x08)) == frame(1, True)


def test_k7_rx_tx_ignores_other_traffic():
  def held(d):
    d.park()
    d.advance(30_000)
    assert d.tx(frame(1, True)) == (True, True)

  d = Device()
  held(d)
  other_bus = libsafety_py.make_CANPacket(CGW1, 2, STOCK)
  assert d.s.safety_rx_hook(other_bus)
  assert d.rx_tx(other_bus) is None
  assert d.stock() == frame(1, True)  # the hold survived

  d = Device()
  held(d)
  extended = libsafety_py.make_CANPacket(CGW1, 0, STOCK)
  extended[0].extended = 1
  assert d.rx_tx(extended) is None
  assert d.stock() == frame(1, True)

  d = Device()
  held(d)
  other_addr = libsafety_py.make_CANPacket(0x367, 0, bytes(8))
  assert d.s.safety_rx_hook(other_addr)
  assert d.rx_tx(other_addr) is None
  assert d.stock() == frame(1, True)

  d = Device()
  held(d)
  d.s.set_relay_malfunction(True)
  assert d.stock() is None


def test_k8_one_trial_then_rearm_after_a_quiet_second():
  for start in (1_000_000, 0xFFFF0000):
    d = Device()
    d.t = start
    d.s.set_timer(start)
    accepted = 0
    for i in range(101):  # 20 Hz requests for ~5 s
      d.park()
      accepted += int(d.tx(frame(1, True))[0])
      d.advance(50_000)
    assert accepted == 60, start
    d.advance(1_000_000 - 50_000)  # one quiet second, no requests
    again = 0
    for i in range(101):
      d.park()
      again += int(d.tx(frame(2, True))[0])
      d.advance(50_000)
    assert again == 60, start


def test_k9_rate_limit_is_50_ms():
  d = Device()
  d.park()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (True, True)
  d.stock()
  d.advance(49_999)
  assert d.tx(frame(1, True)) == (False, False)
  d.advance(1)
  assert d.tx(frame(1, True)) == (True, True)


def test_k10_flags():
  d = Device(param=0)
  d.park()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (False, False)

  d = Device(param=PARAM, model=CarParams.SafetyModel.hyundaiLegacy)
  d.park()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (False, False)

  d = Device(param=PARAM | 1024)  # no direct path here
  d.park()
  d.advance(30_000)
  assert d.tx(frame(1, True)) == (True, True)
  assert d.stock() == frame(1, True)

  d = Device()  # a different address is still refused
  d.park()
  d.advance(30_000)
  d.s.safety_tx_buffered_for_fwd = False
  pkt = libsafety_py.make_CANPacket(0x5B6, 0, bytes(8))
  assert not d.s.safety_tx_hook(pkt)
  assert not d.s.safety_tx_buffered_for_fwd


def test_k11_forwarding_is_unchanged():
  d = Device()
  assert d.s.safety_fwd_hook(0, CGW1) == 2
  assert d.s.safety_fwd_hook(2, CGW1) == 0
  assert d.s.safety_fwd_hook(1, CGW1) == -1


def test_k12_other_hyundai_modes_have_no_rx_tx():
  d = Device(param=0, model=CarParams.SafetyModel.hyundaiCanfd)
  pkt = libsafety_py.make_CANPacket(CGW1, 0, STOCK)
  assert d.rx_tx(pkt) is None
