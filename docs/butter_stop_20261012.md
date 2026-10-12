# ButterStop: speed-tapered final stop with hold creep guard

On 2026-10-12 the user requested that the final metres of a stop be "butter
smooth" and that the standstill hold must never release or creep into the car
ahead. The implementation is brand-agnostic and lives in LongControl plus the
longitudinal MPC, on branch `butter-stop`. The existing standstill hold/release
logic is not changed.

Setting: `ButterStop` int, 0=off 1=on, default 1. It is registered in
`params_keys.h`, `carrot_settings.json` (Korean/English labels, Carrot Web
CRUISE_STOPGO menu) and read in the same 100-frame cadence as `StoppingAccel`,
plus once per planner parameter refresh. All new behavior is behind that flag.

Mechanism (`openpilot/selfdrive/controls/lib/butter_stop.py`):

- Taper floor below `V_TAPER` 3.0 m/s: `f(v) = -(A_END + K_TAPER*v)` with
  `A_END` 0.25 and `K_TAPER` 0.4 m/s^2.
- The MPC plans stops `D_EXTRA` 0.8 m earlier (the taper may roll into it).
- `butter_floor(v_ego, plan_t, plan_speeds, lead, stop_distance_orig)` computes
  the trapezoid distance to the plan's first sub-0.05 m/s sample, subtracts an
  actuator-lag margin `0.2 + 0.15*v`, is additionally bounded by the lead
  (`dRel - max(3.0 m, stop_distance - 0.3 m)`), and returns the more negative
  of the taper and the kinematic limit `-v^2 / (2*avail)`; `None` when no stop
  is planned or availability is at most 0.05 m.
- Eligibility (all required): setting on, active, `STANDSTILL_V < vEgo <
  V_TAPER`, no brake/gas, no FCW, fresh plan (0 to `MAX_PLAN_AGE`), finite
  inputs, and for a lead: gap at least 3.0 m, lead decel at most 1.5 m/s^2 and
  no closing time-under-2.5 s.
- LongControl applies the floor as a braking limit only: PID output and the
  stopping-state ramp are clamped so they can never exceed (be more negative
  than) the floor, the integrator stays frozen while limited, and the floor
  never adds deceleration or positive output.
- After standstill is confirmed (`<=0.05 m/s` for 0.2 s or `CS.standstill`,
  latched per stop episode) the hold ramps toward `StoppingAccel` at the faster
  `HOLD_RAMP` 2.0 m/s^3 instead of `stoppingDecelRate`, and the creep guard
  takes over: with the setting on, in stopping state after confirmation, any
  `vEgo > 0.15 m/s` without gas immediately restores at least -1.5 m/s^2 until
  the episode ends (latched, logged once).
- Unchanged: `long_control_state_trans`, starting conditions, `should_stop`,
  soft hold, StopReq generation in `hyundaican.py`, jerk limits and brake-hold
  handling.

ButterStop=0 is output-identical to the pre-change controller: a 2,000-frame
synthetic stop/hold/soft-hold/release/cruise/off replay is compared every frame
against a verbatim copy of the 80a20196f `LongControl.update` body, for both
starting-state variants, including the state sequence.

Validation:

- Windows unit tests: `openpilot/selfdrive/controls/tests/test_butter_stop.py`
  has 31 tests, all passing, covering floor math including the lead bound,
  every eligibility gate disabling, stopping-state clamping and ramp behavior,
  the standstill latch with the faster hold ramp, the creep guard firing /
  latching / logging once, identical state sequences ON vs OFF, and the
  ButterStop=0 identity replay.
- Existing suites on the upstream-merge-1012 baseline: controls suite
  17 failed / 926 passed / 58 errors vs baseline 17/895/58 (same failures and
  collection errors; +31 passed are the new tests). Hyundai opendbc tests
  56/1020/2 skipped/1 error, identical to baseline. The pre-existing
  `test_longitudinal_stopping.py::test_planner_uses_current_mpc_or_model_stop_without_early_override`
  failure (8 parameters) is kept and reported.
- ruff: the new module, LongControl, long_mpc and the new test file are clean.
  `carrot_functions.py` retains 5 pre-existing findings that are identical in
  the HEAD baseline copy (whitespace, one long line, one banned `time.time`).
- Closed-loop: the Windows MPC shim was abandoned; validation moved to real
  acados in CI (the `butter-sim` close-loop maneuver step and artifact in
  `.github/workflows/tests.yaml`). This document reports simulation only; there
  is no vehicle test.

User-facing documentation: the Korean and English `cruise-gap` guides and the
settings catalogs gained the ButterStop row plus an explanation, and the
generated Wiki carries the same text in its `CARROT:MANUAL` region.

Limits: the constants and thresholds come from the task brief and are covered
by the unit tests and the CI closed-loop maneuvers only. Physical vehicle
behavior, comfort and interaction with real ECUs are not validated. The
hold/release logic itself is deliberately unchanged, so any pre-existing
release behavior is out of scope.
