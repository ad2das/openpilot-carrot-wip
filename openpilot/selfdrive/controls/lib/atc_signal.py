"""ATC signal *requests*, independent of vehicle output or physical verification.

All timestamps are nanoseconds on the same boot's monotonic clock. A request's
source_mono_time is the actual navigation receipt time, never its publication
time. ``valid`` describes parsed, fresh input; it does not mean a lamp operated.
"""

from dataclasses import dataclass, replace
from enum import IntEnum
from math import isfinite
from time import monotonic_ns


class Direction(IntEnum):
  NONE = 0
  LEFT = 1
  RIGHT = 2


@dataclass(frozen=True)
class AtcRequest:
  source_id: int = 0
  episode: int = 0
  source_mono_time: int = 0
  direction: Direction = Direction.NONE
  phase: str = "none"
  valid: bool = False
  reason: str = "uninitialized"


@dataclass(frozen=True)
class RequestGates:
  enabled: bool = False
  lateral_active: bool = False
  can_valid: bool = False
  model_fresh: bool = False
  navigation_fresh: bool = False
  driver_input_known: bool = False
  driver_direction: Direction = Direction.NONE
  hazard: bool = False
  trailer_connected: bool = False
  opposite_steering_torque: bool = False
  explicit_cancel: bool = False


_UINT64_MAX = (1 << 64) - 1
_ACTIVE_PHASES = ("turn", "fork_pending", "fork_active")
_INACTIVE_PHASES = ("none", "prepare", "canceled")
_TOKENS = {
  "turn left": (Direction.LEFT, "turn"),
  "turn right": (Direction.RIGHT, "turn"),
  "fork left": (Direction.LEFT, "fork_pending"),
  "fork right": (Direction.RIGHT, "fork_pending"),
  "atc left": (Direction.LEFT, "fork_pending"),
  "atc right": (Direction.RIGHT, "fork_pending"),
}
_BOOL_GATES = tuple(name for name in RequestGates.__dataclass_fields__ if name != "driver_direction")


def _uint(value, minimum=0):
  return type(value) is int and minimum <= value <= _UINT64_MAX


def valid_direction(value):
  return type(value) in (int, Direction) and value in (Direction.NONE, Direction.LEFT, Direction.RIGHT)


def _duration_ns(seconds):
  if type(seconds) not in (int, float) or not isfinite(seconds) or seconds <= 0:
    raise ValueError("duration must be a finite positive number")
  if seconds > _UINT64_MAX / 1_000_000_000:
    raise ValueError("duration is outside the monotonic timestamp range")
  result = int(seconds * 1_000_000_000)
  if not 0 < result <= _UINT64_MAX:
    raise ValueError("duration is outside the monotonic timestamp range")
  return result


def _fresh(timestamp, now_ns, max_age_ns):
  return _uint(timestamp, 1) and _uint(now_ns, 1) and 0 <= now_ns - timestamp <= max_age_ns


def _token(atc_type):
  # No prefix/substring matching: misspelled or extra-suffixed tokens fail closed.
  if type(atc_type) is not str:
    return None
  if atc_type == "none":
    return Direction.NONE, "none"
  if atc_type in ("straight", "straight prepare", "none prepare"):
    return Direction.NONE, "canceled"
  if atc_type in _TOKENS:
    return _TOKENS[atc_type]
  for suffix, phase in ((" prepare", "prepare"), (" canceled", "canceled")):
    if atc_type.endswith(suffix) and atc_type[: -len(suffix)] in _TOKENS:
      return _TOKENS[atc_type[: -len(suffix)]][0], phase
  return None


class EpisodeProducer:
  """Create one identity for a maneuver; only a fresh none interval rearms it.

  An explicit source_id is useful for deterministic replay. In production the
  default is this instance's monotonic creation time, distinct across restarts.
  Disconnect, invalid input, direction reversal and cancellation are sticky for
  the episode. Changing turn/fork/prepare tokens cannot clear that cancellation.
  """

  def __init__(self, source_id=None, idle_seconds=1.0, max_age_seconds=2.0):
    self.source_id = monotonic_ns() if source_id is None else source_id
    if not _uint(self.source_id, 1):
      raise ValueError("source_id must be a positive uint64")
    self._idle_ns = _duration_ns(idle_seconds)
    self._max_age_ns = _duration_ns(max_age_seconds)
    self.episode = 0
    self._last_now = 0
    self._last_source_time = 0
    self._idle_since = None
    self._armed = False
    self._canceled = True
    self._direction = Direction.NONE

  def _cancel(self):
    self._canceled = True
    self._armed = False
    self._idle_since = None

  def _result(self, timestamp, phase, direction=Direction.NONE, valid=True, reason=""):
    return AtcRequest(self.source_id, self.episode, timestamp, direction, phase, valid, reason)

  def update(self, atc_type, *, now_ns, source_mono_time, source_valid, source_alive=True, cancel=False):
    token = _token(atc_type)
    timing_ok = (
      _fresh(source_mono_time, now_ns, self._max_age_ns)
      and now_ns >= self._last_now
      and source_mono_time >= self._last_source_time
      and self.source_id <= now_ns
    )
    flags_ok = type(source_valid) is bool and type(source_alive) is bool and type(cancel) is bool
    if not timing_ok or not flags_ok or not source_valid or not source_alive or token is None:
      self._cancel()
      return self._result(self._last_source_time, "canceled", valid=False, reason="invalid_or_stale_navigation")

    if self._last_now and now_ns - self._last_now > self._max_age_ns:
      self._cancel()
    self._last_now = now_ns
    self._last_source_time = source_mono_time
    direction, phase = token
    if cancel:
      self._cancel()
      return self._result(source_mono_time, "canceled", reason="explicit_cancel")

    if phase == "none":
      self._canceled = True
      self._direction = Direction.NONE
      if self._idle_since is None:
        self._idle_since = now_ns
      if now_ns - self._idle_since >= self._idle_ns:
        self._armed = True
      return self._result(source_mono_time, "none", reason="idle_ready" if self._armed else "idle_wait")

    self._idle_since = None
    if self._armed and phase != "canceled":
      if self.episode == _UINT64_MAX:
        self._cancel()
        return self._result(source_mono_time, "canceled", valid=False, reason="episode_exhausted")
      self.episode += 1
      self._armed = False
      self._canceled = False
      self._direction = direction

    if phase == "canceled" or direction != self._direction:
      self._cancel()
    if self._canceled:
      return self._result(source_mono_time, "canceled", reason="await_fresh_idle")
    return self._result(
      source_mono_time,
      phase,
      direction if phase in _ACTIVE_PHASES else Direction.NONE,
      reason="navigation_request" if phase in _ACTIVE_PHASES else "navigation_prepare",
    )


def request_is_well_formed(request):
  if not isinstance(request, AtcRequest):
    return False
  if not (
    _uint(request.source_id, 1)
    and _uint(request.episode)
    and _uint(request.source_mono_time, 1)
    and valid_direction(request.direction)
    and type(request.phase) is str
    and type(request.valid) is bool
    and type(request.reason) is str
  ):
    return False
  if request.phase in _ACTIVE_PHASES:
    return request.episode > 0 and request.direction != Direction.NONE
  if request.phase in _INACTIVE_PHASES:
    return request.direction == Direction.NONE and (request.episode > 0 or request.phase in ("none", "canceled"))
  return False


class RequestArbiter:
  """Arbitrate request telemetry only; never operate or acknowledge hardware.

  A fresh none or initial prepare is an inactive baseline. An already-active
  episode seen at startup/restart is quarantined until a none baseline and a
  newer episode. Prepare cannot revive a canceled or previously-active episode.
  Live stop gates apply even when the accompanying request is old or malformed.
  """

  def __init__(self, max_age_seconds=2.0):
    self._max_age_ns = _duration_ns(max_age_seconds)
    self._request = AtcRequest()
    self._last_now = 0
    self._armed = False
    self._blocked = True
    self._active_seen = False
    self._direction = Direction.NONE

  @property
  def source_id(self):
    return self._request.source_id

  @property
  def episode(self):
    return self._request.episode

  def _stop(self, reason, valid=False):
    return replace(self._request, direction=Direction.NONE, phase="canceled", valid=valid, reason=reason)

  def update(self, request, gates, *, now_ns):
    gates_valid = (
      isinstance(gates, RequestGates) and all(type(getattr(gates, name)) is bool for name in _BOOL_GATES) and valid_direction(gates.driver_direction)
    )
    unsafe = (
      not gates_valid
      or not gates.enabled
      or not gates.lateral_active
      or not gates.can_valid
      or not gates.model_fresh
      or not gates.navigation_fresh
      or not gates.driver_input_known
      or gates.driver_direction != Direction.NONE
      or gates.hazard
      or gates.trailer_connected
      or gates.opposite_steering_torque
      or gates.explicit_cancel
    )
    if unsafe:
      self._blocked = True

    if not _uint(now_ns, 1) or now_ns < self._last_now:
      self._blocked = True
      self._armed = False
      return self._stop("invalid_clock")
    if self._last_now and now_ns - self._last_now > self._max_age_ns:
      self._blocked = True
      self._armed = False
    self._last_now = now_ns

    # Validate before either identity high-water mark can move.
    if not request_is_well_formed(request) or not request.valid or request.source_id > now_ns:
      self._blocked = True
      self._armed = False
      return self._stop("invalid_request")
    if not _fresh(request.source_mono_time, now_ns, self._max_age_ns) or not gates_valid or not gates.navigation_fresh or not gates.model_fresh:
      self._blocked = True
      self._armed = False
      return self._stop("stale_or_invalid_input")

    if request.source_id < self.source_id or (request.source_id == self.source_id and request.episode < self.episode):
      return self._stop("old_identity", valid=True)
    if request.source_id == self.source_id and request.source_mono_time < self._request.source_mono_time:
      return self._stop("old_navigation_time", valid=True)

    if request.source_id > self.source_id:
      self._request = AtcRequest(source_id=request.source_id)
      self._armed = False
      self._blocked = True
      self._active_seen = False
      self._direction = Direction.NONE

    if request.episode > self.episode:
      initial_prepare = self.episode == 0 and request.phase == "prepare" and request.reason != "helper_not_ready"
      self._blocked = not (self._armed or initial_prepare)
      self._armed = False
      self._active_seen = False
      self._direction = Direction.NONE
    self._request = request

    if request.phase == "none":
      self._blocked = True
      self._armed = True
      return replace(request, reason="inactive_baseline")
    if request.phase == "canceled":
      self._blocked = True
      self._armed = False
      return replace(request, reason="producer_canceled")
    if unsafe:
      self._blocked = True
    if request.phase == "prepare":
      if self._active_seen:
        self._blocked = True
      if self._blocked:
        return self._stop("episode_blocked", valid=True)
      # Preserve this distinction across multiple consumers (modeld/controlsd).
      return replace(request, reason="helper_not_ready" if request.reason == "helper_not_ready" else "prepare_request")

    self._active_seen = True
    if self._direction not in (Direction.NONE, request.direction):
      self._blocked = True
    self._direction = request.direction
    if self._blocked:
      return self._stop("live_gate_stop" if unsafe else "episode_blocked", valid=True)
    return replace(request, reason="request_allowed")
