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
- The MPC plans stops `D_EXTRA` 1.0 m earlier (the taper may roll into it).
- `butter_floor(v_ego, plan_t, plan_speeds, lead, stop_distance_orig)` computes
  the trapezoid distance to the plan's first sub-0.05 m/s sample, subtracts an
  actuator-lag margin `0.2 + 0.15*v`, is additionally bounded by the lead
  (`dRel - max(3.0 m, stop_distance - 0.6 m)`), and returns the more negative
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
- Closed-loop: real acados runs in CI (the `butter-sim` maneuver step and
  artifact in `.github/workflows/tests.yaml`; sweep run 38209084764, tuned run
  38211150171). The sweep grid was D_EXTRA {0.8, 1.0, 1.2} x
  MIN_FINAL_GAP_SLACK {0.3, 0.6, 0.8}; 0.8/0.3 failed the 40/60 km/h softness
  check because the final gap hit the lead bound; chosen 1.0/0.6. Tuned-run
  results:

  | scenario | OFF final gap | ON final gap | OFF last-0.5s | ON last-0.5s | ON hold creep |
  | --- | --- | --- | --- | --- | --- |
  | stopped lead 20 km/h gap 30 m | 5.054 | 5.672 | 0.5031 | 0.2986 | 0.004 |
  | stopped lead 40 km/h gap 60 m | 4.569 | 5.313 | 0.5072 | 0.2987 | 0.004 |
  | stopped lead 60 km/h gap 90 m | 4.354 | 5.05 | 0.5076 | 0.2984 | 0.003 |
  | lead 40 km/h braking to stop at 2 m/s^2 gap 25 m | 3.568 | 4.111 | 1.0154 | 0.505 | 0.002 |
  | approach stopped car at 25m/s, initial distance: 120m | 11.354 | 11.854 | 0.5043 | 0.2986 | 0.004 |
  | approach stopped car at 20m/s, initial distance 90m | 10.08 | 10.274 | 0.5039 | 0.2991 | 0.004 |
  | steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2 | 5.593 | 6.194 | 0.5045 | 0.298 | 0.003 |
  | steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2 | 2.549 | 3.36 | 1.162 | 1.1588 | 0.001 |
  | steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2 | 2.153 | 2.95 | 1.3598 | 1.3545 | 0.0 |

  No vehicle test.

User-facing documentation: the Korean and English `cruise-gap` guides and the
settings catalogs gained the ButterStop row plus an explanation. Wiki
`CARROT:MANUAL` text for KO/EN/ZH is staged and is published after the branch
lands on carrot-wip.

Limits: the constants and thresholds come from the task brief and are covered
by the unit tests and the CI closed-loop maneuvers only. Physical vehicle
behavior, comfort and interaction with real ECUs are not validated. The
hold/release logic itself is deliberately unchanged, so any pre-existing
release behavior is out of scope.
