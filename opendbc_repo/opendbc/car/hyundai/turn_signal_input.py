"""Receive-only Palisade CGW1 candidates; no actuator authorization.

Historical logs establish these bit positions, but cannot establish that the
switches remain independent while ICU diagnostic output control is active.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnSignalInput:
  valid: bool = False
  sampleMonoTime: int = 0
  leftSwitch: int = 0
  rightSwitch: int = 0
  hazardSwitch: int = 0
  leftLamp: int = 0
  rightLamp: int = 0


def decode_turn_signal_input(data: bytes, sample_ns: int, now_ns: int, *, bus: int, can_valid: bool, max_age_ns: int = 200_000_000) -> TurnSignalInput:
  if (
    not isinstance(data, bytes)
    or len(data) != 8
    or bus != 0
    or type(sample_ns) is not int
    or type(now_ns) is not int
    or sample_ns <= 0
    or not 0 <= now_ns - sample_ns <= max_age_ns
    or can_valid is not True
  ):
    return TurnSignalInput()
  raw = int.from_bytes(data, "little")
  left, right, hazard = (raw >> 42) & 1, (raw >> 41) & 1, (raw >> 33) & 3
  lamp_left, lamp_right = (raw >> 19) & 3, (raw >> 62) & 3
  valid = hazard in (0, 1) and lamp_left in (0, 1) and lamp_right in (0, 1) and not (left and right)
  return TurnSignalInput(valid, sample_ns, left, right, hazard, lamp_left, lamp_right)


def from_parser(cp) -> TurnSignalInput:
  stamp = cp.ts_nanos.get("CGW1", {}).get("CF_Gway_TurnSigLh", 0)
  # CANParser.can_valid consumes its invalid debounce counter when read. The
  # interface owns that read; the consumer separately gates on CarState.canValid.
  return decode_turn_signal_input(cp.dat.get(0x541, b""), stamp, cp._last_update_nanos, bus=cp.bus, can_valid=not cp.bus_timeout)
