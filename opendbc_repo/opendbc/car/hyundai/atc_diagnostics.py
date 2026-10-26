"""Palisade ICU transaction state machine, with transport owned by its caller.

The ScanDoc PALISADE (LX2) ICU definitions bind 0x770/0x778, session 1003,
and left/right button commands 2fbc15/16 with controls 03 and 00. Provenance:
https://scandoc.online/2610210643/base/kia.dat.gz
  decompressed SHA256: 857f99af96c62075e494a7860e642dd7fddc2ac87a28195d676d42387ca0e114
  reader ELF SHA256: 10428b5a1aea94ecc2f5d98b6bb73eaa3a708e92244955bcf4a8b64fde16a4b7
  English labels SHA256: 843af90b212663aa99d77619f8741b323f8e3d89b97e013a2d6638bfdf4beb31
Archived review: 2026-10-26-atc-implementation/scandoc/independent_protocol_review.md.

These are diagnostic button definitions, not proof of blinking, physical lamp
state, moving-vehicle suitability, bus routing, or cleanup reliability. The
Korean/General profiles also have runtime capability gating, not implemented
here. No capability queries, tester-present schedule, or global close command
are inferred. Timing values are software bounds, not measured ECU guarantees.
"""

from dataclasses import dataclass
from enum import Enum
import math


REQUEST_ADDRESS = 0x770
# Panda drops a same-side release sooner than 10 ms after the previous one;
# keep retries clear of that bound so every counted attempt can reach the bus.
RELEASE_RETRY_MIN_S = 0.02
RESPONSE_ADDRESS = 0x778


class Side(Enum):
  LEFT = 0x15
  RIGHT = 0x16


class Phase(Enum):
  UNKNOWN = "unknown"
  SESSION = "session"
  RELEASING = "releasing"
  READY = "ready"
  ACQUIRING = "acquiring"
  ACKNOWLEDGED = "acknowledged"
  FAULT = "fault"


class Operation(Enum):
  SESSION = "session"
  ACQUIRE = "acquire"
  RELEASE = "release"


def _finite_positive(value: float) -> bool:
  return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class Timing:
  """Explicit software bounds, never vehicle-verified timing guarantees."""
  ack_timeout_s: float
  max_active_s: float
  release_attempts: int

  def __post_init__(self) -> None:
    if not _finite_positive(self.ack_timeout_s) or not _finite_positive(self.max_active_s):
      raise ValueError("simulation timeouts must be finite and positive")
    if type(self.release_attempts) is not int or not 1 <= self.release_attempts <= 16:
      raise ValueError("release_attempts must be an integer in 1..16 (offline resource bound)")


@dataclass(frozen=True)
class CanReply:
  address: int
  data: bytes
  src: int
  timestamp: float
  service: str = "can"


@dataclass(frozen=True)
class PlannedCan:
  address: int
  bus: int
  data: bytes
  operation: Operation
  side: Side | None
  attempt: int

  @property
  def payload(self) -> bytes:
    return self.data[1:1 + self.data[0]]


@dataclass(frozen=True)
class Status:
  phase: Phase
  acknowledged_side: Side | None
  uncertain_sides: frozenset[Side]
  release_acknowledged: frozenset[Side]
  fault: str | None


@dataclass(frozen=True)
class _Pending:
  frame: PlannedCan
  sent_at: float
  deadline: float


class PalisadeAtcDiagnosticPlanner:
  """Transaction state machine; call times must be monotonic seconds.

  initialize() explicitly starts session then requests release on both sides.
  Successful initialization means matching diagnostic replies, not lamps off.
  A fresh request() is necessary after initialization; earlier requests are
  discarded. Direction changes and cancellation release BOTH sides first.

  A fault clears queued acquisition, attempts bounded release on both sides,
  then remains latched. There is no reset/rearm or automatic ON retry. Replacing
  this object starts UNKNOWN again, without restoring a request or ownership.
  Planned frames are treated as possibly sent, even without an acknowledgement.
  atc_blinker supplies the runtime transport and cancellation gates.
  """

  def __init__(self, *, bus: int, timing: Timing):
    if type(bus) is not int or not 0 <= bus < 64:
      raise ValueError("bus must be a physical bus in 0..63, without return/reject flags")
    if not isinstance(timing, Timing):
      raise TypeError("an explicit Timing is required")
    self.bus = bus
    self.timing = timing
    self._phase = Phase.UNKNOWN
    self._acknowledged_side: Side | None = None
    self._uncertain = set(Side)
    self._released: set[Side] = set()
    self._fault: str | None = None
    self._pending: _Pending | None = None
    self._release_queue: list[Side] = []
    self._deferred_retry: tuple[Side, int, float] | None = None
    self._after_release: Side | None = None
    self._active_since: float | None = None
    self._last_now: float | None = None

  @property
  def status(self) -> Status:
    return Status(self._phase, self._acknowledged_side, frozenset(self._uncertain), frozenset(self._released), self._fault)

  def _time(self, now: float) -> None:
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now) or now < 0:
      raise ValueError("now must be finite nonnegative monotonic seconds")
    if self._last_now is not None and now < self._last_now:
      raise ValueError("time moved backwards")
    self._last_now = now

  def _emit(self, operation: Operation, side: Side | None, now: float, attempt: int = 1) -> tuple[PlannedCan, ...]:
    if operation is Operation.SESSION:
      payload = b"\x10\x03"
    else:
      assert side is not None
      payload = bytes((0x2f, 0xbc, side.value, 3 if operation is Operation.ACQUIRE else 0))
    frame = PlannedCan(REQUEST_ADDRESS, self.bus, (bytes((len(payload),)) + payload).ljust(8, b"\x00"), operation, side, attempt)
    self._pending = _Pending(frame, now, now + self.timing.ack_timeout_s)
    return (frame,)

  def initialize(self, now: float) -> tuple[PlannedCan, ...]:
    self._time(now)
    expired = self._expire(now)
    if expired or self._phase is not Phase.UNKNOWN:
      return expired
    self._phase = Phase.SESSION
    return self._emit(Operation.SESSION, None, now)

  def request(self, side: Side | None, now: float) -> tuple[PlannedCan, ...]:
    if side is not None and not isinstance(side, Side):
      raise TypeError("side must be Side or None")
    self._time(now)
    expired = self._expire(now)
    if expired or self._fault is not None:
      return expired
    if self._phase is Phase.READY:
      return self._acquire(side, now) if side is not None else ()
    if self._phase is Phase.RELEASING:
      if side != self._after_release:
        self._after_release = None
      return ()  # Do not queue fresh requests during initialization/cleanup.
    if self._phase in (Phase.ACQUIRING, Phase.ACKNOWLEDGED):
      current = self._pending.frame.side if self._pending is not None else self._acknowledged_side
      if side != current:
        return self._begin_release(now, after=side)
    return ()

  def _acquire(self, side: Side, now: float) -> tuple[PlannedCan, ...]:
    assert self._fault is None and not self._uncertain
    self._phase = Phase.ACQUIRING
    self._active_since = now  # Bound from the first possibly sent command, not its ACK.
    self._uncertain.add(side)
    self._released.discard(side)
    return self._emit(Operation.ACQUIRE, side, now)

  def _begin_release(self, now: float, after: Side | None = None) -> tuple[PlannedCan, ...]:
    previous = self._acknowledged_side
    if self._pending is not None and self._pending.frame.operation is Operation.ACQUIRE:
      previous = self._pending.frame.side
    self._phase = Phase.RELEASING
    self._acknowledged_side = None
    self._active_since = None
    self._after_release = after if self._fault is None else None
    self._uncertain = set(Side)
    self._released.clear()
    order = [previous] if previous is not None else []
    order.extend(side for side in Side if side not in order)
    self._release_queue = order
    return self._next_release(now)

  def _next_release(self, now: float) -> tuple[PlannedCan, ...]:
    self._pending = None
    self._deferred_retry = None
    if self._release_queue:
      return self._emit(Operation.RELEASE, self._release_queue.pop(0), now)
    after, self._after_release = self._after_release, None
    self._phase = Phase.FAULT if self._fault is not None else Phase.READY
    if after is not None and self._fault is None:
      return self._acquire(after, now)
    return ()

  def _fail(self, reason: str, now: float) -> tuple[PlannedCan, ...]:
    self._fault = self._fault or reason
    self._after_release = None
    if self._phase is Phase.RELEASING:
      assert self._pending is not None
      frame = self._pending.frame
      if frame.attempt < self.timing.release_attempts:
        due = self._pending.sent_at + RELEASE_RETRY_MIN_S
        if now < due:
          self._pending = None
          self._deferred_retry = (frame.side, frame.attempt + 1, due)
          return ()
        return self._emit(Operation.RELEASE, frame.side, now, frame.attempt + 1)
      return self._next_release(now)  # Keep this side uncertain; still try the other.
    return self._begin_release(now)

  def _expire(self, now: float) -> tuple[PlannedCan, ...]:
    if self._deferred_retry is not None:
      side, attempt, due = self._deferred_retry
      if now < due:
        return ()
      self._deferred_retry = None
      return self._emit(Operation.RELEASE, side, now, attempt)
    if self._active_since is not None and now >= self._active_since + self.timing.max_active_s:
      return self._fail("maximum_active_duration", now)
    if self._pending is not None and now >= self._pending.deadline:
      return self._fail("ack_timeout", now)
    return ()

  def tick(self, now: float) -> tuple[PlannedCan, ...]:
    self._time(now)
    return self._expire(now)

  def abort(self, reason: str, now: float) -> tuple[PlannedCan, ...]:
    """Latch a fault and attempt bounded release without renewing retries."""
    self._time(now)
    if self._fault is not None:
      return self._expire(now)
    self._fault = reason
    self._after_release = None
    if self._phase is Phase.RELEASING:
      return self._expire(now)
    return self._begin_release(now)

  def receive(self, reply: CanReply, now: float) -> tuple[PlannedCan, ...]:
    self._time(now)
    previous_pending = self._pending
    expired = self._expire(now)
    if previous_pending is not self._pending:
      return expired  # A late reply cannot acknowledge the new cleanup transaction.
    if (self._pending is None or reply.service != "can" or type(reply.src) is not int or reply.src != self.bus or
        type(reply.address) is not int or reply.address != RESPONSE_ADDRESS):
      return ()  # sendcan, returned/rejected frames, other buses/ECUs are not ACKs.
    if isinstance(reply.timestamp, bool) or not isinstance(reply.timestamp, (float, int)) or not math.isfinite(reply.timestamp):
      return self._fail("invalid_reply_time", now)
    if reply.timestamp < self._pending.sent_at:
      return ()  # Recorded stale reply, including a prior transaction's ACK.
    if reply.timestamp > now:
      return self._fail("future_reply", now)
    data = reply.data
    if not isinstance(data, bytes) or not 2 <= len(data) <= 8 or not 1 <= data[0] <= 7 or data[0] + 1 > len(data):
      return self._fail("malformed_single_frame", now)
    payload = data[1:1 + data[0]]  # Padding need not be zero (ECUs also use 0xaa).
    frame = self._pending.frame
    if payload[:1] == b"\x7f":
      return self._fail("negative_response", now)  # Includes NRC 0x78; no unbounded extension.
    if frame.operation is Operation.SESSION:
      if payload[:2] != b"\x50\x03" or len(payload) not in (2, 6):
        return self._fail("unexpected_session_response", now)
      return self._begin_release(now)
    assert frame.side is not None
    expected = bytes((0x6f, 0xbc, frame.side.value, 3 if frame.operation is Operation.ACQUIRE else 0))
    if payload != expected:
      return self._fail("unexpected_control_response", now)
    self._pending = None
    if frame.operation is Operation.RELEASE:
      self._uncertain.discard(frame.side)
      self._released.add(frame.side)
      return self._next_release(now)
    self._phase = Phase.ACKNOWLEDGED
    self._acknowledged_side = frame.side
    return ()
