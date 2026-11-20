from functools import partial

from openpilot.cereal import car, log
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, long_control_state_trans


# Supply the current controller inputs for these no-lead state transitions.
transition = partial(long_control_state_trans, a_ego=0.0, stopping_accel=-0.5, radarState=log.RadarState.new_message())
class TestLongControlStateTransition:

  def test_stay_stopped(self):
    CP = car.CarParams.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = transition(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = transition(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

def test_engage():
  CP = car.CarParams.new_message()
  active = True
  current_state = LongCtrlState.off
  next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
  assert next_state == LongCtrlState.stopping
  next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
  assert next_state == LongCtrlState.stopping
  next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid

def test_starting():
  CP = car.CarParams.new_message(startingState=True, vEgoStarting=0.5)
  active = True
  current_state = LongCtrlState.starting
  next_state = transition(CP, active, current_state, v_ego=0.1,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.starting
  next_state = transition(CP, active, current_state, v_ego=1.0,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
  assert next_state == LongCtrlState.pid

# --- L7 B1: the taper floor must not re-limit a law / firm-cap / soft-release hand-off ---

def _l7_control(monkeypatch):
  import openpilot.selfdrive.controls.lib.longcontrol as lcmod
  from openpilot.selfdrive.controls.tests.test_butter_stop import DictParams, make_cp

  stored = {"StoppingAccel": -50, "StopDistanceCarrot": 550, "ButterStop": 1,
            "LongTuningKpV": 100, "LongTuningKiV": 200, "LongTuningKf": 100}
  params = DictParams(stored)
  monkeypatch.setattr(lcmod, "Params", lambda: params)
  # cloudlog is bound at import; replace just the .event attribute for the tick.
  events = []
  monkeypatch.setattr(lcmod.cloudlog, "event", lambda *a, **k: events.append((a, k)))
  return lcmod.LongControl(make_cp())


def _l7_stopping_tick(control, handed, *, law=False, firm=False, release=False, floor=-0.25):
  """One lcs -> stopping tick behind a relaxed taper floor, with the pid hand-off flags set the way
  the pid branch leaves them. The floor slew is already spent (prev seeded at the raw floor, as on
  4d7--2 where cc_u -0.695 was lifted to -0.249), so the floor binds at its raw value."""
  from openpilot.selfdrive.controls.lib import longcontrol as lcmod
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
  from openpilot.selfdrive.controls.tests.test_butter_stop import stopping_inputs
  control.long_control_state = LongCtrlState.stopping
  control.last_output_accel = handed
  control.stop_law_last = law
  control._firm_cap = handed if firm else None
  control._stop_release = handed if release else None
  control._butter_standstill_confirmed = False
  control._butter_rest_floor = None
  control._butter_v_min = 1.0                  # rest_ramp_floor == 0.0 here, so only the raw floor binds
  control._butter_floor_prev = floor           # background slew already spent = the field case 4d7--2
  cs, plan, radar = stopping_inputs(v_ego=1.0)
  orig = lcmod.butter_floor                    # the module-level name longcontrol actually calls
  lcmod.butter_floor = lambda *a, **k: floor
  try:
    out = float(control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0])
  finally:
    lcmod.butter_floor = orig
  return out


def test_l7_b1_handoff_command_is_not_relimited_by_the_taper(monkeypatch):
  # B1. On 4d7--2 the stopping branch applied the background-slewed taper floor after the law
  # handed over, moving cc_u -0.695 -> -0.249 on the lcs 1->2 tick. The stopping tick must
  # reproduce the handed command for each of the three hand-off owners.
  for kind in ("law", "firm", "release"):
    control = _l7_control(monkeypatch)
    out = _l7_stopping_tick(control, -0.695, **{kind: True})
    assert control._butter_no_taper_stop is True, kind
    assert out == -0.695, (kind, out)          # base lifted this to the -0.25 floor

  # A non-law hand-off keeps the taper: bit-identical to the floor-limited command.
  control = _l7_control(monkeypatch)
  out = _l7_stopping_tick(control, -0.695)
  assert control._butter_no_taper_stop is False
  assert out == -0.25, out


def test_l7_b1_floor_prev_reseeds_after_an_override(monkeypatch):
  # B1d. A pid tick on which the soft release owns the output drops the floor slew seed, so the
  # next taper re-seeds from last_output_accel instead of resuming a stale background slew.
  # v_ego 2.5 keeps the taper live (butter_eligible needs STANDSTILL_V < v < V_TAPER 3.0), the plan
  # is flat at speed so should_stop is False and the state stays pid, and the ordinary command
  # (0.0) sits above the released command so the ramp owns the tick.
  import numpy as np
  from types import SimpleNamespace
  from openpilot.selfdrive.controls.lib import longcontrol as lcmod
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
  from openpilot.selfdrive.controls.tests.test_butter_stop import PLAN_T

  control = _l7_control(monkeypatch)
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = -0.5
  control.stop_law_last = False
  control._firm_cap = None
  control._stop_release = -0.5
  control._butter_floor_prev = -0.25

  v = 2.5
  cs = SimpleNamespace(vEgo=v, aEgo=-0.3, brakePressed=False, gasPressed=False, softHoldActive=0,
                       standstill=False, cruiseState=SimpleNamespace(standstill=False), vCruise=0.0,
                       carrotCruise=0)
  plan = SimpleNamespace(aTarget=0.0, vTargetNow=v, jTargetNow=0.0, shouldStop=False, fcw=False,
                         speeds=np.full(PLAN_T.shape, v), longitudinalPlanSource="cruise",
                         cruiseTarget=0.0, cruiseCoastingTarget=0.0, cruiseCoastingPercent=0)
  radar = SimpleNamespace(leadOne=SimpleNamespace(status=False, dRel=80.0, aLeadK=0.0, vLead=0.0),
                          leadTwo=SimpleNamespace(status=False), leadCutInRisk=SimpleNamespace(status=False))

  orig = lcmod.butter_floor
  lcmod.butter_floor = lambda *a, **k: -0.25
  try:
    out = float(control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0])
  finally:
    lcmod.butter_floor = orig

  assert control.long_control_state == LongCtrlState.pid
  assert out == -0.5 + lcmod.STOP_LAW_RELEASE_JERK * lcmod.DT_CTRL, out
  assert control._butter_floor_prev is None, control._butter_floor_prev


# --- L7 Q11: an owned stop's held command eases up to stoppingAccel once standstill is confirmed ---

def _l7_hold_run(monkeypatch, *, owned=True, held=-1.6, floor=-0.25, n=300, v=None, soft_hold=None,
                 creep_latched_at=None, rate=None):
  """The B1 hand-off tick at 1.0 m/s (owner: firm cap when owned), then n stopping ticks. v and soft_hold
  map a tick index to vEgo / softHoldActive (default 0.0 / 0). Returns the hand-off output, the n outputs
  and the n _butter_standstill_confirmed flags read after each tick."""
  from openpilot.selfdrive.controls.lib import longcontrol as lcmod
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
  from openpilot.selfdrive.controls.tests.test_butter_stop import stopping_inputs
  if rate is not None:
    monkeypatch.setattr(lcmod, "B1_HOLD_RELEASE_RATE", rate)
  control = _l7_control(monkeypatch)
  first = _l7_stopping_tick(control, held, firm=owned, floor=floor)
  assert control._butter_no_taper_stop is owned
  outs, flags = [], []
  for i in range(n):
    if creep_latched_at is not None and i == creep_latched_at:
      control._butter_creep_latched = True
    cs, plan, radar = stopping_inputs(v_ego=(v or {}).get(i, 0.0), standstill=True)
    cs.softHoldActive = (soft_hold or {}).get(i, 0)
    outs.append(float(control.update(True, cs, plan, (-3.5, 2.0), 0.05, radar)[0]))
    flags.append(bool(control._butter_standstill_confirmed))
    assert control.long_control_state == LongCtrlState.stopping, i
  return first, outs, flags


def test_l7_q11_owned_hold_is_flat_until_standstill_is_confirmed_then_eases_to_stopping_accel(monkeypatch):
  from openpilot.selfdrive.controls.lib import longcontrol as lcmod
  first, outs, flags = _l7_hold_run(monkeypatch)
  assert first == -1.6
  c = flags.index(True)
  assert c >= round(lcmod.STANDSTILL_T / lcmod.DT_CTRL) - 2, c
  assert all(o == -1.6 for o in outs[:c]), outs[:c + 2]                 # (a) flat until confirmed
  step = lcmod.B1_HOLD_RELEASE_RATE * lcmod.DT_CTRL
  assert abs(outs[c] - (-1.6 + step)) < 1e-12, outs[c]                 # (b) the ease starts on the confirming tick
  steps = [b - a for a, b in zip(outs[c:], outs[c + 1:])]
  assert all(0.0 <= s <= step + 1e-12 for s in steps)                  # (b) up only, at the rate
  assert outs[-1] == -0.5 and max(outs) == -0.5                        # (b) exact clamp at stoppingAccel, no overshoot


def test_l7_q11_ease_pauses_while_rolling_creep_latched_or_soft_hold(monkeypatch):
  from openpilot.selfdrive.controls.lib import longcontrol as lcmod
  step = lcmod.B1_HOLD_RELEASE_RATE * lcmod.DT_CTRL
  # (c1) rolling at STANDSTILL_V or more pauses the ease; it resumes at rest (the confirmation is kept)
  _, outs, _ = _l7_hold_run(monkeypatch, n=120, v={i: 0.06 for i in range(60, 80)})
  assert outs[59] > -1.6
  assert all(o == outs[59] for o in outs[60:80]), outs[58:82]
  assert abs(outs[80] - (outs[79] + step)) < 1e-12, outs[78:82]
  # (c2) the creep guard owns the stop: no ease
  _, outs, _ = _l7_hold_run(monkeypatch, n=120, creep_latched_at=0)
  assert all(o == -1.6 for o in outs), outs[:5]
  # (c3) soft hold: identical to the same run with the ease rate at 0
  hold = {i: 1 for i in range(120)}
  got = _l7_hold_run(monkeypatch, n=120, soft_hold=hold)[1]
  ref = _l7_hold_run(monkeypatch, n=120, soft_hold=hold, rate=0.0)[1]
  assert got == ref


def test_l7_q11_non_owned_stop_is_unchanged(monkeypatch):
  # (d) no hand-off owner: no latch, the taper bookkeeping as before. With no taper floor the command stays
  # below stoppingAccel at confirmed standstill, so a missing owner gate would ease it.
  got = _l7_hold_run(monkeypatch, owned=False, floor=None)
  ref = _l7_hold_run(monkeypatch, owned=False, floor=None, rate=0.0)
  assert got[0] == -1.6
  assert got[1] == ref[1]
  assert all(o == -1.6 for o in got[1])
