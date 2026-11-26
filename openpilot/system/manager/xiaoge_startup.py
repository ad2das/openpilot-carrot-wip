"""Keep optional Xiaoge inference out of onroad startup."""
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True, slots=True)
class _SelfdriveStateSnapshot:
  engageable: bool = False


@dataclass(frozen=True, slots=True)
class _CarStateSnapshot:
  canValid: bool = False
  canTimeout: bool = True


@dataclass(frozen=True, slots=True)
class HighRateSnapshot:
  log_mono_time: dict[str, int]
  checks: dict[str, bool]
  selfdrive_state: _SelfdriveStateSnapshot
  car_state: _CarStateSnapshot
  updated_at: float | None


class XiaogeStartupGate:
  MAIN_SERVICES = ('deviceState', 'pandaStates')
  HIGH_RATE_SERVICES = ('selfdriveState', 'modelV2', 'carState')
  SERVICES = ('deviceState', 'selfdriveState', 'modelV2', 'carState', 'pandaStates')
  SETTLE_SECONDS = 0.5

  def __init__(self):
    self.start_messages = None
    self.ready_since = None
    self.last_check = None
    self.ready = False

  def update(self, started, sm, CP, controls_ready, now):
    if not started:
      self.__init__()
      return False
    if self.start_messages is None:
      # Compare each publisher against itself; publishers may use different
      # monotonic clock domains across suspend. Require new onroad observations.
      self.start_messages = {s: sm.logMonoTime[s] for s in self.SERVICES}
      return False
    if self.ready:
      return True

    # Do not count a gap in manager observations as continuous readiness.
    if self.last_check is not None and now - self.last_check > 1.0:
      self.ready_since = None
    self.last_check = now
    fresh = (sm.all_checks(self.SERVICES) and
             all(sm.logMonoTime[s] > self.start_messages[s] for s in self.SERVICES))
    pandas = sm['pandaStates']
    configs = CP.safetyConfigs
    safety_ready = bool(configs) and len(pandas) >= len(configs) and all(
      (p.safetyModel == configs[i].safetyModel and p.safetyParam == configs[i].safetyParam)
      if i < len(configs) else str(p.safetyModel) in ('silent', 'noOutput')
      for i, p in enumerate(pandas))
    healthy = (controls_ready and fresh and safety_ready and sm['selfdriveState'].engageable
               and sm['carState'].canValid and not sm['carState'].canTimeout)
    if not healthy:
      self.ready_since = None
      return False
    if self.ready_since is None:
      self.ready_since = now
    self.ready = now - self.ready_since >= self.SETTLE_SECONDS
    return self.ready


_EMPTY_SNAPSHOT = HighRateSnapshot(
  log_mono_time=dict.fromkeys(XiaogeStartupGate.HIGH_RATE_SERVICES, 0),
  checks=dict.fromkeys(XiaogeStartupGate.HIGH_RATE_SERVICES, False),
  selfdrive_state=_SelfdriveStateSnapshot(),
  car_state=_CarStateSnapshot(),
  updated_at=None,
)


def _snapshot_from_sm(sm, now: float) -> HighRateSnapshot:
  return HighRateSnapshot(
    log_mono_time={s: int(sm.logMonoTime[s]) for s in XiaogeStartupGate.HIGH_RATE_SERVICES},
    checks={s: bool(sm.all_checks([s])) for s in XiaogeStartupGate.HIGH_RATE_SERVICES},
    selfdrive_state=_SelfdriveStateSnapshot(engageable=bool(sm['selfdriveState'].engageable)),
    car_state=_CarStateSnapshot(
      canValid=bool(sm['carState'].canValid),
      canTimeout=bool(sm['carState'].canTimeout),
    ),
    updated_at=now,
  )


class _MergedSubMaster:
  __slots__ = ('_main_sm', '_snapshot', '_stale', 'logMonoTime')

  def __init__(self, main_sm, snapshot: HighRateSnapshot, *, stale: bool):
    self._main_sm = main_sm
    self._snapshot = snapshot
    self._stale = stale
    self.logMonoTime = {
      **{s: main_sm.logMonoTime[s] for s in XiaogeStartupGate.MAIN_SERVICES},
      **snapshot.log_mono_time,
    }

  def all_checks(self, services) -> bool:
    main_services = [s for s in services if s in XiaogeStartupGate.MAIN_SERVICES]
    if main_services and not self._main_sm.all_checks(main_services):
      return False
    high_rate_services = [s for s in services if s in XiaogeStartupGate.HIGH_RATE_SERVICES]
    if high_rate_services:
      if self._stale:
        return False
      return all(self._snapshot.checks[s] for s in high_rate_services)
    return True

  def __getitem__(self, service: str):
    if service == 'selfdriveState':
      return self._snapshot.selfdrive_state
    if service == 'carState':
      return self._snapshot.car_state
    return self._main_sm[service]


def _default_submaster_factory():
  import openpilot.cereal.messaging as messaging
  return messaging.SubMaster(list(XiaogeStartupMonitor.SUBSCRIBED_SERVICES), poll='deviceState')


def _default_log_exception(msg: str) -> None:
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.exception(msg)
  except Exception:
    pass


class XiaogeStartupMonitor:
  """Keep high-rate Xiaoge readiness subscriptions off the manager main thread.

  msgq directs tkill(SIGUSR2) to the thread that created each SubSocket and
  does not clear read_uids[] on close. Creating the high-rate SubMaster lazily
  once inside a dedicated helper thread (only when ShareData is enabled) keeps
  ~220 Hz SIGUSR2 traffic off the forking manager thread without leaking reader
  slots across toggles or drives.
  """

  SUBSCRIBED_SERVICES = ('deviceState', 'selfdriveState', 'modelV2', 'carState')
  POLL_TIMEOUT_MS = 1000
  SNAPSHOT_MAX_AGE_S = 2.5
  INIT_WAIT_S = 1.0
  START_RETRY_S = 5.0

  def __init__(
    self,
    *,
    submaster_factory=None,
    clock=time.monotonic,
    poll_timeout_ms: int = POLL_TIMEOUT_MS,
    snapshot_max_age_s: float = SNAPSHOT_MAX_AGE_S,
    log_exception=None,
  ):
    self.gate = XiaogeStartupGate()
    self._submaster_factory = submaster_factory or _default_submaster_factory
    self._clock = clock
    self._poll_timeout_ms = poll_timeout_ms
    self._snapshot_max_age_s = snapshot_max_age_s
    self._log_exception = log_exception or _default_log_exception

    self._lock = threading.Lock()
    self._snapshot = _EMPTY_SNAPSHOT
    self._stop_event = threading.Event()
    self._init_done = threading.Event()
    self._thread: threading.Thread | None = None
    self._next_start_at = float('-inf')

  @property
  def ready(self) -> bool:
    return self.gate.ready

  def _ensure_started(self, now: float) -> bool:
    with self._lock:
      thread = self._thread
      next_start_at = self._next_start_at
    if thread is not None and thread.is_alive():
      return True
    if self._stop_event.is_set() or now < next_start_at:
      return False

    self._init_done = threading.Event()
    thread = threading.Thread(target=self._run, name='xiaoge_startup_sm', daemon=True)
    with self._lock:
      self._thread = thread
    thread.start()
    # Wait for one-time SubMaster initialization before manager forks onroad
    # processes so no child can inherit a mid-init allocator or import lock.
    self._init_done.wait(timeout=self.INIT_WAIT_S)
    with self._lock:
      return self._thread is thread and thread.is_alive()

  def _run(self) -> None:
    try:
      sm = self._submaster_factory()
    except Exception:
      self._log_exception("xiaoge startup monitor failed to initialize SubMaster")
      with self._lock:
        self._next_start_at = self._clock() + self.START_RETRY_S
        self._thread = None
      return
    finally:
      self._init_done.set()

    while not self._stop_event.is_set():
      try:
        sm.update(self._poll_timeout_ms)
        snapshot = _snapshot_from_sm(sm, self._clock())
      except Exception:
        with self._lock:
          self._snapshot = _EMPTY_SNAPSHOT
        self._log_exception("xiaoge startup monitor update failed")
        self._stop_event.wait(0.5)
        continue
      with self._lock:
        self._snapshot = snapshot

  def update(self, started: bool, main_sm, CP, controls_ready: bool, share_data: bool, now: float) -> bool:
    if not started:
      return self.gate.update(False, main_sm, CP, controls_ready, now)

    with self._lock:
      thread = self._thread
    if thread is None:
      if not share_data:
        return False
      self._ensure_started(now)
    elif share_data and not thread.is_alive():
      self._ensure_started(now)

    with self._lock:
      snapshot = self._snapshot

    stale = snapshot.updated_at is None or (now - snapshot.updated_at) > self._snapshot_max_age_s
    merged_sm = _MergedSubMaster(main_sm, snapshot, stale=stale)
    return self.gate.update(True, merged_sm, CP, controls_ready, now)

  def stop(self, timeout: float = 0.0) -> None:
    self._stop_event.set()
    thread = self._thread
    if timeout > 0 and thread is not None and thread.is_alive():
      thread.join(timeout=timeout)
