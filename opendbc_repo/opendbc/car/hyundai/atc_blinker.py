"""Opt-in classic Palisade ATC diagnostic sender.

Commands and matching ECU replies are observable; lamp operation is not inferred
from either. The pulse timing below is a software policy awaiting vehicle tests.
"""

from opendbc.car import structs
from opendbc.car.hyundai.atc_diagnostics import CanReply, Operation, PalisadeAtcDiagnosticPlanner, Phase, Side, Timing
from opendbc.car.hyundai.values import CAR, HyundaiFlags, HyundaiSafetyFlags
from openpilot.selfdrive.controls.lib.atc_signal import AtcRequest, RequestArbiter, RequestGates
from openpilot.selfdrive.controls.lib.atc_signal_integration import candidate_gates, recent, request_from_message

CONTROL_MAX_AGE_NS = 150_000_000
ON_SECONDS = 0.35
OFF_SECONDS = 0.35
MAX_EPISODE_SECONDS = 45.0
TIMING = Timing(ack_timeout_s=0.25, max_active_s=1.0, release_attempts=3)

# Web-requested stationary check: one long left ON (lamp steady or ICU flash?),
# normal pulses, release, then watch that the lamp goes out. Driver/vehicle
# gates are stricter than a navigation request; Panda gates are unchanged.
TEST_MAX_AGE_NS = 2_000_000_000
TEST_FIRST_ON_SECONDS = 0.9  # below TIMING.max_active_s
TEST_SECONDS = 3.0
TEST_AFTER_SECONDS = 1.0
TEST_TIMEOUT_SECONDS = 8.0
TEST_MAX_SAMPLES = 400


class AtcSignalTest:
  def __init__(self, nonce, now):
    self.nonce = nonce
    self.started = now
    self.ended = None
    self.pulses = 0
    self.first_on = None
    self.first_off = None
    self.released = None
    self.phases = set()
    self.positive = 0
    self.negative = 0
    self.first_reply = None
    self.samples = []
    self.cancel = ''

  def sample(self, now, cs, phase):
    raw = cs.turnSignalInput
    key = (int(raw.leftLamp), int(cs.leftBlinker), int(raw.leftSwitch), phase)
    if (not self.samples or self.samples[-1][1:] != key) and len(self.samples) < TEST_MAX_SAMPLES:
      self.samples.append((round(now - self.started, 3), *key))

  def lamp(self, start, end):
    """CGW1 left lamp in [start, end): (on fraction after it first lit, re-lights).

    Measuring from the first lit sample tolerates CGW1 latency; a steady lamp
    has no re-light, an ICU-driven flash goes out and lights again.
    """
    if start is None or end is None or end <= start:
      return 0.0, 0
    points = [(t + self.started, lamp) for t, lamp, *_ in self.samples]
    lit, on, rises = None, 0.0, 0
    for i, (t, lamp) in enumerate(points):
      nxt = points[i + 1][0] if i + 1 < len(points) else end
      lo, hi = max(t, start), min(nxt, end)
      if lamp > 0 and hi > lo:
        if lit is None:
          lit = lo
        elif i > 0 and points[i - 1][1] == 0 and t >= start:
          rises += 1
        on += hi - lo
    if lit is None:
      return 0.0, 0
    return round(on / (end - lit), 2), rises

  def result(self, fault):
    on_ratio, rises = self.lamp(self.first_on, self.first_off)
    if self.cancel:
      verdict = 'canceled'
    elif self.positive == 0:
      verdict = 'no_icu_response'
    elif fault:
      verdict = 'fault'
    elif self.first_on is None:
      verdict = 'not_acquired'
    elif not any(lamp for _, lamp, *_ in self.samples):
      verdict = 'lamp_not_seen'
    elif rises > 0 or on_ratio < 0.8:
      verdict = 'icu_flashes'
    else:
      verdict = 'steady'
    return dict(request=self.nonce, verdict=verdict, cancel=self.cancel, fault=fault or '',
                icuPositive=self.positive, icuNegative=self.negative,
                firstReplyMs=None if self.first_reply is None else round((self.first_reply - self.started) * 1000),
                pulses=self.pulses, firstOnLampRatio=on_ratio, firstOnLampRises=rises,
                lampAfter=self.samples[-1][1] if self.samples else None,
                phases=sorted(self.phases), samples=[list(x) for x in self.samples])


def check_block(cc, cs, accepted, now_ns, control_valid):
  """Reason the stationary check may not run now, or ''."""
  raw = cs.turnSignalInput
  if cc is None or not (cc.enabled and control_valid):
    return 'not_engaged'
  if not cs.standstill or cs.vEgo > 0.1:
    return 'not_stopped'
  if cs.gearShifter not in (structs.CarState.GearShifter.drive, structs.CarState.GearShifter.low):
    return 'gear'
  if cs.steerFaultTemporary or cs.steerFaultPermanent or not cs.canValid:
    return 'car_fault'
  if (not raw.valid or not recent(raw.sampleMonoTime, now_ns, 200_000_000) or raw.leftSwitch or raw.rightSwitch
      or raw.hazardSwitch or recent(raw.lastManualPressMonoTime, now_ns, 200_000_000)):
    return 'driver_signal'
  if cs.steeringPressed:
    return 'steering'
  if accepted.direction:
    return 'navigation_request'
  return ''


def classic_atc_enabled(cp):
  return bool(cp.carFingerprint == CAR.HYUNDAI_PALISADE
              and cp.flags & HyundaiFlags.CLASSIC_ATC
              and not cp.flags & (HyundaiFlags.CANFD | HyundaiFlags.LEGACY)
              and len(cp.safetyConfigs) == 1
              and cp.safetyConfigs[0].safetyModel == structs.CarParams.SafetyModel.hyundai
              and cp.safetyConfigs[0].safetyParam & HyundaiSafetyFlags.CLASSIC_ATC)


class PalisadeAtcBlinker:
  def __init__(self):
    self.arbiter = RequestArbiter()
    self.planner = None
    self._started = 0.0
    self._last_session = float('-inf')
    self._on_since = None
    self._next_on = 0.0
    self._ending = False
    self._identity = None
    self._last_now_ns = 0
    self.last_command_ns = 0
    self.last_response_ns = 0
    self.tx_count = 0
    self.test = None
    self._test_nonce = 0
    self.test_result = None  # replaced, never mutated, when a check ends

  @property
  def status(self):
    state = self.planner.status if self.planner is not None else None
    side = state.acknowledged_side if state else None
    return dict(enabled=True, phase=state.phase.value if state else 'idle',
                acknowledgedDirection=1 if side is Side.LEFT else 2 if side is Side.RIGHT else 0,
                fault=state.fault or '' if state else '', lastCommandMonoTime=self.last_command_ns,
                lastResponseMonoTime=self.last_response_ns, txCount=self.tx_count)

  def update(self, cc, cs, replies, now_ns, *, control_valid=True, test_request=0):
    # Clamping only permits cleanup if a caller's clock regresses. Never rearm.
    clock_valid = type(now_ns) is int and now_ns > 0 and now_ns >= self._last_now_ns
    if not clock_valid:
      now_ns = self._last_now_ns
      control_valid = False
    self._last_now_ns = now_ns
    now = now_ns / 1e9
    if cc is None:
      accepted = self.arbiter.update(AtcRequest(), RequestGates(), now_ns=now_ns)
    else:
      req = request_from_message(cc.atcSignal)
      fresh = control_valid and recent(cc.atcSignal.requestMonoTime, now_ns, CONTROL_MAX_AGE_NS)
      gates = candidate_gates(cs, req, now_ns=now_ns,
                              enabled=bool(cc.enabled and control_valid and
                                           not (self.planner is None and (cs.leftBlinker or cs.rightBlinker)) and
                                           cs.gearShifter in (structs.CarState.GearShifter.drive, structs.CarState.GearShifter.low) and
                                           not cs.steerFaultTemporary and not cs.steerFaultPermanent),
                              lateral_active=cc.latActive, model_fresh=bool(fresh), navigation_fresh=bool(fresh))
      accepted = self.arbiter.update(req, gates, now_ns=now_ns)
    desired = Side.LEFT if accepted.direction == 1 else Side.RIGHT if accepted.direction == 2 else None
    identity = (accepted.source_id, accepted.episode)
    frames = []

    if type(test_request) is int and test_request > self._test_nonce and clock_valid:
      self._test_nonce = test_request
      if not recent(test_request, now_ns, TEST_MAX_AGE_NS):
        self._finish_test(AtcSignalTest(test_request, now), 'stale_request')
      elif (self.test is not None or self.planner is not None or now - self._last_session < 1.0
            or cs.leftBlinker or cs.rightBlinker):
        self._finish_test(AtcSignalTest(test_request, now), 'busy')
      elif blocked := check_block(cc, cs, accepted, now_ns, control_valid):
        self._finish_test(AtcSignalTest(test_request, now), blocked)
      else:
        self.test = AtcSignalTest(test_request, now)
    test = self.test
    if test is not None:
      for address, data, src, _ in replies:
        if address == 0x778 and src == 0 and isinstance(data, bytes) and len(data) == 8:
          test.positive += data[1] in (0x50, 0x6f)
          test.negative += data[1] == 0x7f
          test.first_reply = now if test.first_reply is None else test.first_reply
      if test.ended is None:
        test.cancel = check_block(cc, cs, accepted, now_ns, control_valid)
        if test.cancel or now - test.started >= TEST_SECONDS:
          test.ended = now
        else:
          desired, identity = Side.LEFT, ('test', test.nonce)
      if test.ended is not None:
        desired = None

    if self.planner is None:
      if desired is None or now - self._last_session < 1.0:
        if test is not None:
          self._test_tick(test, cs, now)
        return []
      self.planner = PalisadeAtcDiagnosticPlanner(bus=0, timing=TIMING)
      self._started = self._last_session = now
      self._identity = identity
      self._ending = False
      self._on_since = None
      self._next_on = now
      frames.extend(self.planner.initialize(now))

    if desired is None or identity != self._identity:
      self._ending = True
    if not clock_valid:
      frames.extend(self.planner.abort('invalid_clock', now))
    elif now - self._started >= MAX_EPISODE_SECONDS:
      frames.extend(self.planner.abort('maximum_episode_duration', now))

    # Cancel before consuming a late ACK, so it cannot authorize another ON.
    on_seconds = TEST_FIRST_ON_SECONDS if test is not None and test.pulses == 1 else ON_SECONDS
    pulse_finished = self._on_since is not None and now - self._on_since >= on_seconds
    if self._ending or pulse_finished:
      frames.extend(self.planner.request(None, now))

    # This receive batch predates every command generated by this call. A
    # reply in it must never ACK a newly emitted initialization/cleanup frame.
    if not frames:
      for address, data, src, timestamp_ns in replies:
        if (type(timestamp_ns) is not int or timestamp_ns <= 0 or timestamp_ns == now_ns
            or not isinstance(data, bytes) or len(data) != 8):
          continue
        emitted = self.planner.receive(CanReply(address, data, src, timestamp_ns / 1e9), now)
        frames.extend(emitted)
        if address == 0x778 and src == 0 and timestamp_ns < now_ns:
          self.last_response_ns = max(self.last_response_ns, timestamp_ns)
        if emitted:
          break  # The rest of this batch also predates this new transaction.
    frames.extend(self.planner.tick(now))

    state = self.planner.status
    if state.phase is Phase.READY:
      if self._on_since is not None:
        self._next_on = now + OFF_SECONDS  # Start OFF interval after both return-control ACKs.
        self._on_since = None
      if self._ending:
        self.planner = None
      elif desired is not None and now >= self._next_on:
        frames.extend(self.planner.request(desired, now))

    for frame in frames:
      if frame.operation is Operation.ACQUIRE:
        self._on_since = now
        if test is not None and test.ended is None:
          test.pulses += 1
          test.first_on = now if test.first_on is None else test.first_on
      elif frame.operation is Operation.RELEASE and test is not None and test.first_on is not None and test.first_off is None:
        test.first_off = now
      self.last_command_ns = now_ns
      self.tx_count = min(self.tx_count + 1, (1 << 32) - 1)
    if test is not None:
      self._test_tick(test, cs, now)
    return [(frame.address, frame.data, frame.bus) for frame in frames]

  def _test_tick(self, test, cs, now):
    state = self.planner.status if self.planner is not None else None
    phase = state.phase.value if state else 'idle'
    test.phases.add(phase)
    test.sample(now, cs, phase)
    if state is not None and state.phase is Phase.FAULT:
      self._finish_test(test, '', fault=state.fault)
    elif test.ended is not None and self.planner is None:
      test.released = now if test.released is None else test.released
      if now - test.released >= TEST_AFTER_SECONDS:
        self._finish_test(test, '')
    elif now - test.started >= TEST_TIMEOUT_SECONDS:
      self._finish_test(test, '', fault='test_timeout')

  def _finish_test(self, test, cancel, fault=None):
    test.cancel = test.cancel or cancel
    self.test_result = test.result(fault)
    if test is self.test:
      self.test = None
