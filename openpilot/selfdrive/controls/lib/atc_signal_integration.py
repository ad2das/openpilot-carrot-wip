"""ATC request adapters shared by runtime and offline replay.

The vehicle controller separately checks opt-in, request age and live inputs.
Diagnostic output and switch independence still require vehicle measurements.
"""

from dataclasses import replace

from openpilot.selfdrive.controls.lib.atc_signal import AtcRequest, Direction, RequestGates

NAV_MAX_AGE_NS = 2_000_000_000


class NavigationSourceTracker:
  def __init__(self):
    self.previous = None

  def changed(self, sm, *, uses_nav_instruction, carrot_navi_active, remote_id):
    key = ('navInstruction',) if uses_nav_instruction else (('carrotNavi', sm['carrotNavi'].sessionId) if carrot_navi_active else ('legacy', remote_id))
    changed = self.previous is not None and key != self.previous
    self.previous = key
    return changed


def recent(stamp_ns, now_ns, max_age_ns):
  return type(stamp_ns) is int and type(now_ns) is int and stamp_ns > 0 and 0 <= now_ns - stamp_ns <= max_age_ns


def service_fresh(sm, service, now_ns, max_age_ns):
  return (
    sm.seen[service]
    and sm.alive[service]
    and sm.valid[service]
    and recent(sm.logMonoTime[service], now_ns, max_age_ns)
    and recent(int(sm.recv_time[service] * 1e9), now_ns, max_age_ns)
  )


def request_from_message(msg):
  return AtcRequest(msg.sourceId, msg.episode, msg.sourceMonoTime, msg.direction, msg.phase, msg.valid, msg.reason)


def request_as_dict(req):
  return {
    'sourceId': req.source_id,
    'episode': req.episode,
    'sourceMonoTime': req.source_mono_time,
    'direction': int(req.direction),
    'phase': req.phase,
    'valid': req.valid,
    'reason': req.reason,
  }


def driver_signal_state(cs, now_ns):
  """Raw switch state for opt-in cars, with 3 blocking on hazard/unknown.

  Lamp feedback must not masquerade as a driver request during our own pulses.
  The original lamp-based path remains in use when the feature is disabled.
  """
  raw = cs.turnSignalInput
  if (not raw.valid or not recent(raw.sampleMonoTime, now_ns, 200_000_000)
      or raw.leftSwitch not in (0, 1) or raw.rightSwitch not in (0, 1)
      or raw.hazardSwitch != 0 or (raw.leftSwitch and raw.rightSwitch)):
    return 3
  return raw.leftSwitch + 2 * raw.rightSwitch


class AtcDriverInput:
  """Suppress only our lamp feedback; preserve manual one-touch lamp holds."""
  def __init__(self):
    self._last_command = 0
    self._last_manual_press = 0
    self._suppress_lamps = False
    self._manual_lamps = False
    self._manual_dir = 0

  def update(self, cs, now_ns):
    raw = driver_signal_state(cs, now_ns)
    status = cs.atcSignalStatus
    lamps = cs.leftBlinker or cs.rightBlinker
    owned = status.enabled and status.phase not in ('', 'idle')
    if owned or status.lastCommandMonoTime > self._last_command:
      self._suppress_lamps = True
    self._last_command = max(self._last_command, status.lastCommandMonoTime)
    press = cs.turnSignalInput.lastManualPressMonoTime
    new_press = press > self._last_manual_press and recent(press, now_ns, 500_000_000)
    self._last_manual_press = max(self._last_manual_press, press)
    if not owned and not lamps:
      self._suppress_lamps = False
    if raw in (1, 2):
      self._manual_lamps = True
      self._manual_dir = raw
    elif new_press:
      self._manual_lamps = True
    elif raw == 0 and not lamps:
      self._manual_lamps = False
      self._manual_dir = 0
    if not self._suppress_lamps:
      # None selects the original DesireHelper lamp-based driver path.
      return None
    if self._manual_lamps:
      # Our lamp feedback may still be retained on the other side; follow the
      # driver's lever (and its one-touch hold) instead of the lamp flags.
      return self._manual_dir if self._manual_dir else None
    return raw


def navigation_candidate(producer, sm, now_ns, *, atc_type, uses_nav_instruction, carrot_navi_active, legacy_source_ns, source_changed):
  # Heartbeats alone cannot renew an old guidance item.
  if uses_nav_instruction:
    stamp = sm.logMonoTime['navInstruction']
    valid = service_fresh(sm, 'navInstruction', now_ns, NAV_MAX_AGE_NS)
  elif carrot_navi_active:
    stamp = sm['carrotNavi'].guidanceCurrent.meta.receivedMonoTimeNanos
    valid = service_fresh(sm, 'carrotNavi', now_ns, NAV_MAX_AGE_NS)
  else:
    stamp = legacy_source_ns
    valid = recent(stamp, now_ns, NAV_MAX_AGE_NS)
  return request_as_dict(producer.update(atc_type, now_ns=now_ns, source_mono_time=stamp, source_valid=valid, cancel=source_changed))


def candidate_gates(cs, request, *, now_ns, enabled, lateral_active, model_fresh, navigation_fresh):
  raw = cs.turnSignalInput
  known = (
    raw.valid
    and recent(raw.sampleMonoTime, now_ns, 200_000_000)
    and raw.leftSwitch in (0, 1)
    and raw.rightSwitch in (0, 1)
    and raw.hazardSwitch in (0, 1)
    and not (raw.leftSwitch and raw.rightSwitch)
  )
  driver = Direction.LEFT if raw.leftSwitch == 1 else Direction.RIGHT if raw.rightSwitch == 1 else Direction.NONE
  opposite = cs.steeringPressed and (
    (request.direction == Direction.LEFT and cs.steeringTorque < 0) or (request.direction == Direction.RIGHT and cs.steeringTorque > 0)
  )
  return RequestGates(
    enabled=enabled,
    lateral_active=lateral_active,
    can_valid=cs.canValid,
    model_fresh=model_fresh,
    navigation_fresh=navigation_fresh,
    driver_input_known=bool(known),
    driver_direction=driver,
    hazard=raw.hazardSwitch != 0,
    trailer_connected=cs.trailerConnected,
    opposite_steering_torque=opposite,
    explicit_cancel=recent(raw.lastManualPressMonoTime, now_ns, 200_000_000),
  )


def model_candidate(arbiter, helper, sm, now_ns):
  req = request_from_message(sm['carrotMan'].atcSignal)
  intent = req
  if req.direction:
    # The producer identifies a maneuver; the actual DesireHelper decides if
    # it is eligible now. Bluetooth/remote desire never acquires provenance.
    if helper.atc_signal_direction == req.direction:
      req = replace(req, phase=helper.atc_signal_phase)
    else:
      req = replace(req, direction=Direction.NONE, phase='prepare', reason='helper_not_ready')
  gates = candidate_gates(
    sm['carState'],
    intent,
    now_ns=now_ns,
    enabled=sm['carControl'].enabled,
    lateral_active=sm['carControl'].latActive,
    model_fresh=(service_fresh(sm, 'carState', now_ns, 200_000_000) and service_fresh(sm, 'carControl', now_ns, 200_000_000)),
    navigation_fresh=service_fresh(sm, 'carrotMan', now_ns, 500_000_000),
  )
  return request_as_dict(arbiter.update(req, gates, now_ns=now_ns))


def control_candidate(arbiter, sm, cc, now_ns):
  req = request_from_message(sm['modelV2'].meta.atcSignal)
  nav = sm['carrotMan'].atcSignal
  same_navigation = req.source_id == nav.sourceId and req.episode == nav.episode and nav.valid and nav.phase not in ('canceled', 'none')
  same_navigation = same_navigation and (req.direction == 0 or req.direction == nav.direction)
  # Inactive baseline must also cross the consumer on startup/episode end.
  if req.direction == 0 and nav.phase == 'none':
    same_navigation = req.source_id == nav.sourceId and req.episode == nav.episode and nav.valid
  gates = candidate_gates(
    sm['carState'],
    request_from_message(nav),
    now_ns=now_ns,
    enabled=cc.enabled,
    lateral_active=cc.latActive,
    model_fresh=(service_fresh(sm, 'modelV2', now_ns, 250_000_000) and service_fresh(sm, 'carState', now_ns, 200_000_000)),
    navigation_fresh=(same_navigation and service_fresh(sm, 'carrotMan', now_ns, 500_000_000) and recent(nav.sourceMonoTime, now_ns, NAV_MAX_AGE_NS)),
  )
  result = request_as_dict(arbiter.update(req, gates, now_ns=now_ns))
  result['requestMonoTime'] = now_ns
  return result
