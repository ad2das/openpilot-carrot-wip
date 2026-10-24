"""Hyundai CAN FD: StopReq stop sequence (ButterStop decel until the ESC holds, then zero aReq), without automatic release/retry.

Thresholds below are experimental, not OEM acceptance conditions. This controller
cannot guarantee stopping; ECU response must be measured on the vehicle.
"""
from dataclasses import dataclass
from enum import StrEnum


DT = 0.02  # SCC_CONTROL is transmitted at 50 Hz
ENTRY_SPEED = 0.7  # m/s; retain approach braking above the low-speed stop region
STOP_SPEED = 0.05
MOVING_SPEED = 0.10
STOP_CONFIRM_TIME = 0.2
APPROACH_ACCEL = -0.5
STOP_LOWER_BAND = 0.20  # fixed experimental value; never copied from the stock SCC
RELEASE_CONFIRM_FRAMES = 10  # 0.2 s at 50 Hz; a held/requesting stop lets go only after the request stays off this long (departure flicker, route 6d1)
PASS_ACCEL_MIN = -0.45  # approach (StopReq 0): ButterStop accel passes only while softer than this; harder keeps the -0.5 floor
ZERO_SPEED = 0.01  # m/s; all wheels truly stopped
ZERO_CONFIRM_FRAMES = 50  # 1.0 s at 50 Hz truly stopped without AVH -> stock zeros anyway


class StopPhase(StrEnum):
  idle = "idle"
  approach = "approach"
  request = "request"
  held = "held"


@dataclass(frozen=True)
class StopCommand:
  stop_req: int
  raw: float
  value: float
  lower: float


class CanfdStopping:
  def __init__(self):
    # Packet history survives episode resets; it follows the final StopReq,
    # including approach and interlock overrides.
    self.last_scc_stop_req = 0
    self.last_scc_jerk_upper = 1.0
    # ButterStop on: keep sending the openpilot request through approach/request
    # so the soft taper reaches the ESC.
    self.pass_accel = False
    self.reset()

  def limit_scc_jerk_upper(self, stop_req: int, jerk_upper: float) -> float:
    """Separate an Upper increase from the first StopReq frame, not its onset."""
    if stop_req == 1 and self.last_scc_stop_req != 1:
      jerk_upper = min(jerk_upper, self.last_scc_jerk_upper)
    self.last_scc_stop_req = stop_req
    self.last_scc_jerk_upper = jerk_upper
    return jerk_upper

  def reset(self):
    self.phase = StopPhase.idle
    self.reason = "inactive"
    self.stopped_time = 0.0
    self.release_frames = 0
    self.zero_frames = 0

  def enter(self, phase: StopPhase, reason: str):
    self.phase = phase
    self.reason = reason

  def _pass_accel(self, accel: float, previous_value: float, jerk_u: float, jerk_l: float) -> StopCommand:
    raw = min(accel, 0.0)
    value = min(0.0, max(previous_value - jerk_l * DT, min(raw, previous_value + jerk_u * DT)))
    return StopCommand(1, raw, value, 0.0)

  def update(self, *, active: bool, requested: bool, speed: float, held: bool,
             accel: float, previous_value: float, jerk_u: float, jerk_l: float) -> StopCommand | None:
    # Caller validates sensor values and applies pedal/CAN/hold interlocks.
    if not active:
      self.reset()
      return None
    if not requested:
      # Debounce the release only at rest in request/held; pedals, gear, CAN faults and ACC off
      # arrive as active=False above and still release at once.
      if self.phase in (StopPhase.request, StopPhase.held) and speed <= MOVING_SPEED:
        self.release_frames += 1
        if self.release_frames < RELEASE_CONFIRM_FRAMES:
          return StopCommand(1, 0.0, 0.0, STOP_LOWER_BAND)
      self.reset()
      return None
    self.release_frames = 0

    if self.phase == StopPhase.idle:
      self.enter(StopPhase.approach if speed > ENTRY_SPEED else StopPhase.request, "stop_requested")
    if self.phase == StopPhase.approach and speed <= ENTRY_SPEED:
      self.enter(StopPhase.request, "entry_speed")

    self.stopped_time = self.stopped_time + DT if speed <= STOP_SPEED else 0.0
    self.zero_frames = self.zero_frames + 1 if speed <= ZERO_SPEED else 0
    stopped = (held and speed <= MOVING_SPEED) or self.stopped_time >= STOP_CONFIRM_TIME
    if stopped:
      if self.phase != StopPhase.held:
        self.enter(StopPhase.held, "stop_observed")
    elif self.phase == StopPhase.held:
      # Motion changes diagnostics only. Never release StopReq to retry, even
      # when speed rises above the initial entry threshold after a request.
      self.enter(StopPhase.request, "motion_after_hold")

    # ButterStop: keep OP's decel (taper, stop ramp, creep guard) under StopReq until the ESC holds.
    # Zero aReq while the car still rolled at 0.03 m/s never reached a hold and let a K5 creep 3-4 m
    # (routes 6ce/6d1/6d2, 10-24); the stock zero entry follows once AVH holds or after 1 s truly stopped.
    hold_confirmed = held or self.zero_frames >= ZERO_CONFIRM_FRAMES
    if self.phase in (StopPhase.held, StopPhase.request):
      if self.pass_accel and not hold_confirmed:
        return self._pass_accel(accel, previous_value, jerk_u, jerk_l)
      return StopCommand(1, 0.0, 0.0, STOP_LOWER_BAND)

    # Preserve the existing approach deceleration until initial low-speed entry.
    soft = self.pass_accel and accel > PASS_ACCEL_MIN
    raw = min(accel, 0.0) if soft else min(accel, APPROACH_ACCEL)
    value = min(0.0, max(previous_value - jerk_l * DT, min(raw, previous_value + jerk_u * DT)))
    return StopCommand(0, raw, value, 0.0)
