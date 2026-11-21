# Palisade CGW1 parked blinker test (phase 1 of the joongyu01 CGW1 blinker port, Neon 11-21). Remove after phase 1.
# Parked in P, the driver flashes the high beam (pulls flash-to-pass and lets go): the car gets one 3 s LEFT one-touch
# blinker forged on CGW1 0x541. panda (safety_hyundai.h hyundai_cgw_test) holds each frame and sends it right behind the
# next stock CGW1, parked only.
# Trigger = CF_Gway_PassingSW (byte 6 0x08) OR CF_Gway_HeadLampHigh (byte 4 0x01): on the user's Palisade the switch bit
# was 0 in all 1 522 939 logged CGW1 frames and a flash showed only as the lamp bit (W-BLK4, Neon NEON_Q_BLK3 11-21).
# Stock CGW1 (joongyu01 10-26 rlog 00000e40): 10 Hz, lamp bit 0.35 s on / 0.35 s off, byte 5 bits 1..2 = lever side.
TX_PERIOD_NS = 60_000_000      # panda's 50 ms rate limit would bounce frames that arrive a little early
FLASH_HALF_NS = 350_000_000
TRIAL_NS = 3_000_000_000       # panda cuts a parked trial at 3 s
QUIET_NS = 1_000_000_000       # re-arm: trigger off for 1 s, and panda's 1 s with no request at all
FRESH_NS = 250_000_000
PULSE_MIN_NS = 20_000_000      # shortest logged flash 0.039 s (one ~40 ms CGW1 event frame)
PULSE_MAX_NS = 1_500_000_000   # longer = the high beam switched on, not a flash
PASSING = 0x08                 # byte 6: CF_Gway_PassingSW
HIGH_BEAM = 0x01               # byte 4: CF_Gway_HeadLampHigh


def cgw_frame(raw, direction, lamp):
  """The stock CGW1 bytes with the lever side (1 left, 2 right) and, when lamp, that side's lamp bit set."""
  data = bytearray(raw)
  data[2] |= 0x08 if direction == 1 and lamp else 0
  data[7] |= 0x40 if direction == 2 and lamp else 0
  data[5] |= 0x04 if direction == 1 else 0x02
  return (0x541, bytes(data), 0)


class PalisadeCgwTest:
  """One 3 s left blinker trial per high-beam flash, parked and disengaged; the lamp result is unverified."""

  def __init__(self):
    self.off_since = None    # since when the trigger is off (None: on, or not yet seen off while fresh)
    self.pulse_start = None  # rising edge of a flash that can still fire
    self.started = None      # trial start; None: no trial running
    self.last_tx = None

  def update(self, now, raw, timestamps, parked, inactive):
    fresh = len(raw) == 8 and len(timestamps) == 3 and all(0 < ts <= now and now - ts <= FRESH_NS for ts in timestamps)
    if not fresh:
      self.off_since = self.pulse_start = self.started = None
      return []
    trig = bool(raw[6] & PASSING or raw[4] & HIGH_BEAM)
    # stop for stock lamps, hazard or either physical turn switch
    ok = parked and inactive and not (raw[2] & 0x18 or raw[7] & 0xC0 or raw[4] & 0x06 or raw[5] & 0x06)
    if self.started is not None and (not ok or now - self.started >= TRIAL_NS):
      self.started = None
    fire = False
    if trig:
      if self.off_since is not None:  # rising edge: armed only after 1 s off, no trial, 1 s since our last frame
        armed = (now - self.off_since >= QUIET_NS and self.started is None and
                 (self.last_tx is None or now - self.last_tx >= QUIET_NS))
        self.pulse_start = now if armed and ok else None
        self.off_since = None
      elif self.pulse_start is not None and (not ok or now - self.pulse_start > PULSE_MAX_NS):
        self.pulse_start = None  # a gate dropped mid-flash, or the beam is held on: this one never fires
    elif self.off_since is None:  # falling edge, or the first fresh frame
      fire = self.pulse_start is not None and ok and PULSE_MIN_NS <= now - self.pulse_start <= PULSE_MAX_NS
      self.pulse_start = None
      self.off_since = now
    if fire and self.started is None:
      self.started = now
    if self.started is None or (self.last_tx is not None and now - self.last_tx < TX_PERIOD_NS):
      return []
    lamp = (now - self.started) // FLASH_HALF_NS % 2 == 0
    self.last_tx = now
    return [cgw_frame(raw, 1, lamp)]
