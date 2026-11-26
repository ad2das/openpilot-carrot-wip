import ast
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from openpilot.cereal import car, log
from openpilot.system.manager.xiaoge_startup import XiaogeStartupGate, XiaogeStartupMonitor


class Messages(dict):
  def __init__(self):
    super().__init__(deviceState=SimpleNamespace(started=True),
                     selfdriveState=SimpleNamespace(engageable=True, enabled=False),
                     modelV2=SimpleNamespace(), carState=SimpleNamespace(canValid=True, canTimeout=False),
                     pandaStates=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)])
    self.logMonoTime = dict.fromkeys(XiaogeStartupGate.SERVICES, 0)
    self.bad = set()

  def all_checks(self, services):
    return not self.bad.intersection(services)

  def tick(self, now):
    self.logMonoTime = dict.fromkeys(self.logMonoTime, int(now * 1e9))


@pytest.fixture
def setup():
  gate, sm = XiaogeStartupGate(), Messages()
  cp = SimpleNamespace(safetyConfigs=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)])
  sm.tick(-1)
  assert not gate.update(True, sm, cp, True, -1)
  return gate, sm, cp


def update(setup, now, *, started=True, controls=True):
  gate, sm, cp = setup
  sm.tick(now)
  return gate.update(started, sm, cp, controls, now)


def test_timeout_and_controls_ready_do_not_bypass_model_or_can_readiness(setup):
  gate, sm, _ = setup
  sm.bad.add('modelV2')
  for t in (0, 6.01, 7, 8, 9):
    assert not update(setup, t)
  sm.bad.clear()
  sm['carState'].canTimeout = True
  assert not update(setup, 9.5)
  sm['carState'].canTimeout = False
  assert not update(setup, 10)
  assert not update(setup, 10.49)
  assert update(setup, 10.5)
  assert gate.ready
  assert not sm['selfdriveState'].enabled  # No actual engagement is needed.


@pytest.mark.parametrize('service', XiaogeStartupGate.SERVICES)
def test_unhealthy_or_missing_service_resets_settling(setup, service):
  _, sm, _ = setup
  assert not update(setup, 0)
  sm.bad.add(service)
  assert not update(setup, 0.4)
  sm.bad.clear()
  assert not update(setup, 0.5)
  assert update(setup, 1)


@pytest.mark.parametrize('service', XiaogeStartupGate.SERVICES)
def test_previous_session_messages_cannot_release_startup(setup, service):
  gate, sm, cp = setup
  assert not gate.update(False, sm, cp, True, 9)
  sm.tick(9)
  assert not gate.update(True, sm, cp, True, 9)
  sm.tick(10)
  sm.logMonoTime[service] = int(9e9)
  assert not gate.update(True, sm, cp, True, 10)
  assert not gate.update(True, sm, cp, True, 10.5)
  assert not update(setup, 11)
  assert update(setup, 11.5)


@pytest.mark.parametrize('case', ['controls', 'engageable', 'can_valid', 'safety_model', 'safety_param', 'missing_panda', 'missing_config', 'extra_elm'])
def test_incomplete_initialization_is_not_ready(setup, case):
  _, sm, cp = setup
  if case == 'engageable':
    sm['selfdriveState'].engageable = False
  if case == 'can_valid':
    sm['carState'].canValid = False
  if case == 'safety_model':
    sm['pandaStates'][0].safetyModel = 'elm327'
  if case == 'safety_param':
    sm['pandaStates'][0].safetyParam = 0
  if case == 'missing_panda':
    sm['pandaStates'] = []
  if case == 'missing_config':
    cp.safetyConfigs = []
  if case == 'extra_elm':
    sm['pandaStates'].append(SimpleNamespace(safetyModel='elm327'))
  for t in (0, 0.5, 1):
    assert not update(setup, t, controls=case != 'controls')


def test_latch_avoids_restart_on_later_fault_but_resets_offroad(setup):
  _, sm, _ = setup
  assert not update(setup, 0)
  assert update(setup, 0.5)
  sm.bad.add('modelV2')
  sm['selfdriveState'].engageable = False
  assert update(setup, 1)
  assert not update(setup, 2, started=False)
  assert not update(setup, 3)
  sm.bad.clear()
  sm['selfdriveState'].engageable = True
  assert not update(setup, 3.5)
  assert update(setup, 4)


def test_missing_observations_do_not_count_as_settling(setup):
  assert not update(setup, 0)
  assert not update(setup, 5)
  assert update(setup, 5.5)


def test_manager_restart_has_no_persistent_ready_latch(setup):
  _, sm, cp = setup
  assert not update(setup, 0)
  assert update(setup, 0.5)
  gate = XiaogeStartupGate()
  assert not gate.update(True, sm, cp, True, 5)


def test_real_cereal_safety_configs_and_extra_silent_panda():
  cp = car.CarParams.new_message(safetyConfigs=[{'safetyModel': 'hyundai', 'safetyParam': 13}])
  ps = log.Event.new_message()
  ps.init('pandaStates', 2)
  ps.pandaStates[0].safetyModel = 'hyundai'
  ps.pandaStates[0].safetyParam = 13
  ps.pandaStates[1].safetyModel = 'silent'
  sm = Messages()
  sm['pandaStates'] = ps.pandaStates
  setup = XiaogeStartupGate(), sm, cp
  assert not update(setup, -1)
  assert not update(setup, 0)
  assert update(setup, 0.5)


def test_publisher_clock_offsets_do_not_release_old_messages(setup):
  gate, sm, cp = setup
  assert not gate.update(False, sm, cp, True, 1)
  sm.logMonoTime = {s: int((100 + i) * 1e9) for i, s in enumerate(gate.SERVICES)}
  assert not gate.update(True, sm, cp, True, 2)
  assert not gate.update(True, sm, cp, True, 3)
  for s in gate.SERVICES:
    sm.logMonoTime[s] += int(0.5e9)
  assert not gate.update(True, sm, cp, True, 3.5)
  for s in gate.SERVICES:
    sm.logMonoTime[s] += int(0.5e9)
  assert gate.update(True, sm, cp, True, 4)


def test_process_selection_retains_live_sharedata_and_excludes_offroad():
  # Load the production predicate without importing Windows-unavailable Params.
  path = Path(__file__).parents[1] / 'process_config.py'
  tree = ast.parse(path.read_text(encoding='utf-8'))
  predicate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'enable_xiaoge_data')
  namespace = {'car': car}
  exec(compile(ast.Module(body=[predicate], type_ignores=[]), str(path), 'exec'), namespace)
  check = namespace['enable_xiaoge_data']
  cp = car.CarParams.new_message()
  for enabled in (False, True):
    params = SimpleNamespace(get_bool=lambda key, enabled=enabled: enabled)
    assert not check(False, params, cp)
    assert check(True, params, cp) == enabled


class _MainMessages(dict):
  def __init__(self):
    super().__init__(
      deviceState=SimpleNamespace(started=True),
      pandaStates=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)],
    )
    self.logMonoTime = dict.fromkeys(XiaogeStartupGate.MAIN_SERVICES, 0)
    self.bad = set()

  def all_checks(self, services):
    return not self.bad.intersection(services)

  def tick(self, now):
    self.logMonoTime = dict.fromkeys(self.logMonoTime, int(now * 1e9))


class _ControlledHelperSm(Messages):
  def __init__(self):
    super().__init__()
    self._waiting = threading.Event()
    self._allow = threading.Event()
    self.update_count = 0
    self.next_error: Exception | None = None
    self.closing = False

  def update(self, _timeout_ms):
    self._waiting.set()
    self._allow.wait(timeout=5.0)
    self._allow.clear()
    if self.closing:
      return
    self.update_count += 1
    if self.next_error is not None:
      err = self.next_error
      self.next_error = None
      raise err

  def step(self):
    assert self._waiting.wait(timeout=5.0)
    self._waiting.clear()
    self._allow.set()
    assert self._waiting.wait(timeout=5.0)

  def release_for_stop(self):
    self.closing = True
    self._allow.set()


def test_manager_main_submaster_matches_upstream_services():
  path = Path(__file__).parents[1] / 'manager.py'
  tree = ast.parse(path.read_text(encoding='utf-8'))
  fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'manager_thread')
  sm_calls = [
    n for n in ast.walk(fn)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'SubMaster'
  ]
  assert len(sm_calls) == 1
  services = ast.literal_eval(sm_calls[0].args[0])
  poll_kw = next(kw.value for kw in sm_calls[0].keywords if kw.arg == 'poll')
  assert services == ['deviceState', 'carParams', 'pandaStates']
  assert ast.literal_eval(poll_kw) == 'deviceState'


def test_monitor_stays_idle_without_share_data():
  created = []
  monitor = XiaogeStartupMonitor(submaster_factory=lambda: created.append(True))
  main_sm = _MainMessages()
  cp = SimpleNamespace(safetyConfigs=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)])
  for t in (0.0, 0.5, 1.0, 2.0):
    main_sm.tick(t)
    assert not monitor.update(True, main_sm, cp, True, False, t)
  assert created == []
  assert monitor._thread is None


def test_monitor_runs_high_rate_submaster_on_helper_thread_and_releases():
  main_ident = threading.get_ident()
  created_idents = []
  helper_sm = _ControlledHelperSm()
  now = [0.0]

  def factory():
    created_idents.append(threading.get_ident())
    return helper_sm

  monitor = XiaogeStartupMonitor(submaster_factory=factory, clock=lambda: now[0])
  main_sm = _MainMessages()
  cp = SimpleNamespace(safetyConfigs=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)])
  try:
    # First onroad tick with ShareData enabled initializes helper SubMaster off main thread.
    now[0] = 1.0
    main_sm.tick(1.0)
    assert not monitor.update(True, main_sm, cp, True, True, 1.0)
    assert len(created_idents) == 1
    assert created_idents[0] != main_ident

    # Helper polls healthy fresh messages; gate settles over 0.5 s.
    now[0] = 2.0
    helper_sm.tick(2.0)
    helper_sm.step()
    main_sm.tick(2.0)
    assert not monitor.update(True, main_sm, cp, True, True, 2.0)

    now[0] = 2.5
    helper_sm.tick(2.5)
    helper_sm.step()
    main_sm.tick(2.5)
    assert monitor.update(True, main_sm, cp, True, True, 2.5)

    # Offroad resets latch; next onroad drive reuses the existing helper thread
    # without reopening msgq sockets and rejects previous-session timestamps.
    assert not monitor.update(False, main_sm, cp, True, True, 3.0)
    main_sm.tick(4.0)
    assert not monitor.update(True, main_sm, cp, True, True, 4.0)
    assert not monitor.update(True, main_sm, cp, True, True, 4.5)  # stale 2.5 s helper logMonoTime

    now[0] = 5.0
    helper_sm.tick(5.0)
    helper_sm.step()
    main_sm.tick(5.0)
    assert not monitor.update(True, main_sm, cp, True, True, 5.0)

    now[0] = 5.5
    helper_sm.tick(5.5)
    helper_sm.step()
    main_sm.tick(5.5)
    assert monitor.update(True, main_sm, cp, True, True, 5.5)
    assert len(created_idents) == 1
  finally:
    monitor.stop()
    helper_sm.release_for_stop()
    monitor.stop(timeout=2.0)


def test_monitor_mid_drive_share_data_toggle_releases_cleanly():
  helper_sm = _ControlledHelperSm()
  now = [0.0]
  monitor = XiaogeStartupMonitor(submaster_factory=lambda: helper_sm, clock=lambda: now[0])
  main_sm = _MainMessages()
  cp = SimpleNamespace(safetyConfigs=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)])
  try:
    main_sm.tick(10.0)
    assert not monitor.update(True, main_sm, cp, True, False, 10.0)
    assert monitor._thread is None

    # User enables ShareData mid-drive.
    now[0] = 11.0
    main_sm.tick(11.0)
    assert not monitor.update(True, main_sm, cp, True, True, 11.0)
    assert monitor._thread is not None

    now[0] = 11.5
    helper_sm.tick(11.5)
    helper_sm.step()
    main_sm.tick(11.5)
    assert not monitor.update(True, main_sm, cp, True, True, 11.5)

    now[0] = 12.0
    helper_sm.tick(12.0)
    helper_sm.step()
    main_sm.tick(12.0)
    assert monitor.update(True, main_sm, cp, True, True, 12.0)
  finally:
    monitor.stop()
    helper_sm.release_for_stop()
    monitor.stop(timeout=2.0)


def test_monitor_rejects_stale_and_erroring_helper_snapshots():
  helper_sm = _ControlledHelperSm()
  now = [0.0]
  logged = threading.Event()
  monitor = XiaogeStartupMonitor(
    submaster_factory=lambda: helper_sm,
    clock=lambda: now[0],
    snapshot_max_age_s=2.0,
    log_exception=lambda _msg: logged.set(),
  )
  main_sm = _MainMessages()
  cp = SimpleNamespace(safetyConfigs=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)])
  try:
    now[0] = 1.0
    main_sm.tick(1.0)
    assert not monitor.update(True, main_sm, cp, True, True, 1.0)

    # Fresh helper poll at t=2.0 starts settling.
    now[0] = 2.0
    helper_sm.tick(2.0)
    helper_sm.step()
    main_sm.tick(2.0)
    assert not monitor.update(True, main_sm, cp, True, True, 2.0)

    # Snapshot becomes stale (>2.0 s without helper poll) -> cannot release.
    main_sm.tick(4.5)
    assert not monitor.update(True, main_sm, cp, True, True, 4.5)

    # Helper raises inside update -> snapshot cleared immediately.
    helper_sm.next_error = RuntimeError("ipc glitch")
    assert helper_sm._waiting.wait(timeout=5.0)
    helper_sm._waiting.clear()
    helper_sm._allow.set()
    assert logged.wait(timeout=5.0)
    main_sm.tick(4.6)
    assert not monitor.update(True, main_sm, cp, True, True, 4.6)
    assert monitor._snapshot.updated_at is None
  finally:
    monitor.stop()
    helper_sm.release_for_stop()
    monitor.stop(timeout=2.0)


def test_monitor_recovers_after_submaster_init_failure():
  helper_sm = _ControlledHelperSm()
  now = [0.0]
  attempts = []
  errors = []

  def factory():
    attempts.append(now[0])
    if len(attempts) == 1:
      raise RuntimeError("transient shm error")
    return helper_sm

  monitor = XiaogeStartupMonitor(
    submaster_factory=factory,
    clock=lambda: now[0],
    log_exception=errors.append,
  )
  main_sm = _MainMessages()
  cp = SimpleNamespace(safetyConfigs=[SimpleNamespace(safetyModel='hyundai', safetyParam=13)])
  try:
    now[0] = 1.0
    main_sm.tick(1.0)
    assert not monitor.update(True, main_sm, cp, True, True, 1.0)
    assert len(attempts) == 1 and len(errors) == 1

    # Backoff prevents retry spam before START_RETRY_S.
    now[0] = 3.0
    main_sm.tick(3.0)
    assert not monitor.update(True, main_sm, cp, True, True, 3.0)
    assert len(attempts) == 1

    # After backoff expires, factory succeeds and gate settles normally.
    now[0] = 6.5
    main_sm.tick(6.5)
    assert not monitor.update(True, main_sm, cp, True, True, 6.5)
    assert len(attempts) == 2

    now[0] = 7.0
    helper_sm.tick(7.0)
    helper_sm.step()
    main_sm.tick(7.0)
    assert not monitor.update(True, main_sm, cp, True, True, 7.0)

    now[0] = 7.5
    helper_sm.tick(7.5)
    helper_sm.step()
    main_sm.tick(7.5)
    assert monitor.update(True, main_sm, cp, True, True, 7.5)
  finally:
    monitor.stop()
    helper_sm.release_for_stop()
    monitor.stop(timeout=2.0)
