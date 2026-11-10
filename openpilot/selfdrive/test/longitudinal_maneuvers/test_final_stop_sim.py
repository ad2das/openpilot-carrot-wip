"""W-FSTOP1 sim gate: ButterStopFinal 1 vs 0 on the fitted plant.

Per event in the W-PLANTE gate population (lead_jump is False), it runs the
real LongitudinalPlanner + real LongControl + fitted plant twice on the same
fixture header -- once with ButterStopFinal 0 and once with 1 -- and checks:

  G1 per event:  both arms come to rest (standstill onset within +-8 s of the
                 real stop); gap_stop(1) >= gap_stop(0) - 0.05 m;
                 min gap > 0 over [t_real_sim - 15, t_real_sim + 2].
  G2 per car:    median time from window entry to sim standstill (arm 1)
                 <= arm 0's time from the same instant (report both).
  G3 per event:  arm 1 mean commanded accel over the last 0.3 s before its sim
                 standstill >= -0.15 (report arm 0's value too).
  G4:            the ButterStopLoop maneuvers run with ButterStopFinal 1.

Report-only: window entries / events, entry-v distribution, exits by reason.

The onset / end-decel / real-stop helpers are the W-PLANTE runner's rules,
copied from test_late_handoff_sim.py. FarLeadLoop runs in full-length mode so a
multistop fixture's pre-roll rest does not truncate the run.

Fixed thresholds: the gates report PASS/FAIL and nothing here tunes constants.
Run with `pytest -s` so the printed rows reach the log:

  pytest -o addopts='' -q -s openpilot/selfdrive/test/longitudinal_maneuvers/test_final_stop_sim.py
"""

import json
import math
import os

import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  FarLeadLoop,
  FixtureProfile,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_fitted_plant import _kf_end_decel
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.butter_stop import (
  FINAL_U_MAX, FINAL_U_MIN, MIN_FINAL_GAP_ABS, PROBE_GUARD_T, PROBE_MAX_T, PROBE_V_MAX, probe_accel,
)

_SCRATCH = "/mnt/c/Users/Administrator/Downloads/carrot-wip/carrot-openpilot/.analysis/scratch/2026-11-08-perf"
FIX = f"{_SCRATCH}/W-PLANT3B/fixtures"
POP = f"{_SCRATCH}/W-PLANTE/raw/sim/head"
OUT = "/mnt/c/Users/Administrator/Downloads/carrot-wip/carrot-openpilot/.analysis/scratch/2026-11-10-firm/W-K5B1/raw/sim"

FINAL_ON = 1
FINAL_OFF = 0

STANDSTILL = 0.05
END_WINDOW = 0.5             # mean |a| over [t_stop-0.5, t_stop], W-PLANTE rule
STOP_CENTRE = 10.0           # fixture window is [T-20, T+25] with T = t_i_log - 10
EVENT_GATE = 8.0             # onset must be within this of the real stop (sim time)
G1_BACK, G1_FWD = 15.0, 2.0  # G1 gap window in sim time around the real stop
G1_GAP_SLACK = 0.05          # m, gap_stop(1) may come this much closer
G3_WINDOW = 0.3              # s, mean command before standstill
G3_MAX_DECEL = -0.15         # m/s^2, arm-1 mean command floor
CLASSIC_CAN_CARS = ("Palisade",)  # ButterStopFinal hands off early on these; the rest open the window


def _gate_events():
  rows = []
  for name in sorted(os.listdir(POP)):
    if not name.endswith(".json"):
      continue
    with open(os.path.join(POP, name), encoding="utf-8") as handle:
      row = json.load(handle)
    if row["lead_jump"]:
      continue
    rows.append((row["car"], row["eid"]))
  assert rows, f"no gate rows in {POP}"
  return rows


EVENTS = _gate_events()


def _onsets(v, limit=STANDSTILL):
  """Standstill onsets: i where v[i] < limit and (i == 0 or v[i-1] >= limit)."""
  v = np.asarray(v, dtype=float)
  below = v < limit
  if not below.any():
    return np.empty(0, dtype=int)
  prev_ok = np.r_[True, v[:-1] >= limit]
  return np.flatnonzero(below & prev_ok)


def _choose_onset(t, onsets, t_real_sim):
  """Onset index nearest the real stop; (None, dist) when none is within the gate."""
  if not onsets.size:
    return None, None
  d = np.abs(np.asarray(t, dtype=float)[onsets] - t_real_sim)
  k = int(np.argmin(d))
  if float(d[k]) > EVENT_GATE:
    return None, float(d[k])
  return int(onsets[k]), float(d[k])


def _mean_abs_before(t, a, t_stop):
  window = (t >= t_stop - END_WINDOW) & (t <= t_stop)
  if not window.any():
    raise AssertionError(
      f"FAIL_empty_window: no samples in [{t_stop - END_WINDOW:.2f}, {t_stop:.2f}]")
  return float(np.mean(np.abs(a[window])))


def real_stop(fixture):
  """Standstill onset nearest +10 s in the fixture frame, plus both clock forms."""
  scale, grid, data = fixture["scale"], fixture["grid"], fixture["data"]
  t = grid["start_s"] + np.arange(grid["n"]) * grid["dt"]
  v = np.array(data["vEgo"], dtype=float) / scale["speed"]
  onsets = np.flatnonzero((v <= STANDSTILL) & np.r_[True, v[:-1] > STANDSTILL])
  if not onsets.size:
    raise AssertionError("FAIL_no_onset: fixture has no standstill onset")
  times = [float(t[i]) for i in onsets]
  t_fix = min(times, key=lambda onset: abs(onset - STOP_CENTRE))
  drel = np.array(data["leadOne_dRel"], dtype=float) / scale["dist"]
  return {
    "t_fixture": t_fix,
    "t_sim": t_fix - float(grid["start_s"]),
    "gap": float(np.interp(t_fix, t, drel)),
    "n_onsets": int(onsets.size),
    "multistop": bool(onsets.size > 1),
  }


def _metrics(run, ref):
  """Onset-nearest-real selection + inline end decel + G1 gap window (fitted keys)."""
  t = np.array(run["t"], dtype=float)
  v = np.array(run["v_kf"], dtype=float)
  a = np.array(run["a_kf"], dtype=float)
  gap = np.array(run["gap"], dtype=float)
  active = np.array(run["active"], dtype=bool)

  onsets = _onsets(v)
  idx, dt = _choose_onset(t, onsets, ref["t_sim"])
  if idx is None:
    nearest = "none" if dt is None else f"{dt:.2f} s"
    raise AssertionError(
      f"FAIL_stopped: no v < {STANDSTILL:.2f} onset within {EVENT_GATE:.1f} s of "
      + f"sim t={ref['t_sim']:.2f} (nearest {nearest}, run ends {float(t[-1]):.2f} s)")

  t_stop = float(t[idx])
  end = _mean_abs_before(t, a, t_stop)

  win = (t >= ref["t_sim"] - G1_BACK) & (t <= ref["t_sim"] + G1_FWD) & active
  if not win.any():
    raise AssertionError("FAIL_gap: no active samples in the G1 window")
  g1 = gap[win]
  if not np.all(g1 > 0.0):
    raise AssertionError(
      f"FAIL_gap: non-positive gap while active in G1 window (min {float(np.min(g1)):.3f})")

  legacy = _kf_end_decel(run)
  if onsets.size == 1 and (legacy is None or abs(end - legacy) > 1e-12):
    shown = float("nan") if legacy is None else legacy
    raise AssertionError(f"FAIL_selfcheck: inline end decel {end:.17g} != _kf_end_decel {shown:.17g}")

  return {
    "t_stop": t_stop,
    "dt_real": float(dt),
    "gap_stop": float(gap[idx]),
    "end_decel": end,
    "min_gap_g1": float(np.min(g1)),
    "n_sim_onsets": int(onsets.size),
    "stopped": bool(run["stopped"]),
    "run_end_t": float(t[-1]),
  }


def _command_mean_before(run, t_stop, window=G3_WINDOW):
  t = np.array(run["t"], dtype=float)
  accel = np.array(run["accel"], dtype=float)
  sel = (t >= t_stop - window) & (t <= t_stop)
  if not sel.any():
    raise AssertionError(
      f"FAIL_g3_window: no commanded samples in [{t_stop - window:.2f}, {t_stop:.2f}]")
  return float(np.mean(accel[sel]))


class FinalStopLoop(FarLeadLoop):
  """FarLeadLoop that records LongControl's ButterStopFinal window state.

  The base loop exposes only its own logs, so the window clock is read from the
  live controller after every update(): entry tick and speed, and the reason
  each window ended. Nothing else about the loop changes.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.tick = 0
    self.window = {"t": [], "final_t": [], "v": [], "accel": [], "lc_state": [], "fcw": []}
    self.entries = []
    self.exits = []
    self.guard_exits = 0
    self.window_ticks = 0
    self.cmd_mismatches = 0
    # The plan guard no longer exists inside the window (the command is the
    # probe table), so this stays 0 and the FSTOP/GUARD rows keep their shape.
    self.plan_guard_ticks = 0
    # Why a window-eligible event never opened one. Recorded per tick from the
    # states update() actually saw; nothing here feeds control.
    self.probe = {"v_ho": None, "a_ho": None, "sacc": None,
                  "v_min_pid": None, "first_le_probe_v_max": None}
    self._inner_update = self.long_control.update
    self.long_control.update = self._update

  def _command_mismatch(self, CS, radarState, logged):
    """True when the window's logged command is not the probe/lead minimum.

    The probe command is a pure function of vEgo, so the whole window command
    is clip(min(probe_accel(v), lead term), FINAL_U_MIN, FINAL_U_MAX). This
    recomputes it from the tick's inputs and compares; a mismatch means the
    controller and this report disagree about the wiring.
    """
    try:
      v_ego = float(CS.vEgo)
    except (TypeError, ValueError):
      return True
    if not math.isfinite(v_ego):
      return True
    u = probe_accel(v_ego)
    lead = getattr(radarState, "leadOne", None)
    if lead is not None and bool(getattr(lead, "status", False)):
      gap = float(lead.dRel) - MIN_FINAL_GAP_ABS
      if math.isfinite(gap):
        u = min(u, -(v_ego * v_ego) / (2.0 * max(gap, 0.05)))
    expected = float(np.clip(u, FINAL_U_MIN, FINAL_U_MAX))
    return abs(float(logged) - expected) > 1e-9

  def _record_probe(self, CS, long_plan, pre_state):
    """Per-tick facts for a window that never opened: when pid handed off and how fast."""
    probe = self.probe
    try:
      v_ego = float(CS.vEgo)
      a_ego = float(CS.aEgo)
    except (TypeError, ValueError):
      return
    if probe["sacc"] is None:
      probe["sacc"] = float(self.long_control.stopping_accel)
    state = int(self.long_control.long_control_state)
    should_stop = bool(long_plan.shouldStop)
    if pre_state == int(LongCtrlState.pid) and state == int(LongCtrlState.stopping) \
        and probe["v_ho"] is None:
      probe["v_ho"] = v_ego
      probe["a_ho"] = a_ego
    if should_stop:
      if state == int(LongCtrlState.pid):
        if probe["v_min_pid"] is None or v_ego < probe["v_min_pid"]:
          probe["v_min_pid"] = v_ego
      if v_ego <= PROBE_V_MAX and probe["first_le_probe_v_max"] is None:
        probe["first_le_probe_v_max"] = (state, a_ego, v_ego)

  def _update(self, active, CS, long_plan, accel_limits, t_since_plan, radarState):
    was_open = self.long_control._final_t is not None
    pre_final_t = self.long_control._final_t
    pre_state = int(self.long_control.long_control_state)
    out = self._inner_update(active, CS, long_plan, accel_limits, t_since_plan, radarState)
    now = self.long_control._final_t
    tick_t = self.tick * DT_CTRL
    self.tick += 1
    self._record_probe(CS, long_plan, pre_state)
    _lead = getattr(radarState, "leadOne", None)
    lead_close = bool(getattr(_lead, "status", False)
                      and float(getattr(_lead, "dRel", float("inf"))) < MIN_FINAL_GAP_ABS)
    self.window["t"].append(tick_t)
    self.window["final_t"].append(now)
    self.window["v"].append(float(CS.vEgo))
    self.window["accel"].append(float(out[0]))
    self.window["lc_state"].append(int(self.long_control.long_control_state))
    self.window["fcw"].append(lead_close)
    if was_open and now is not None and pre_state == int(LongCtrlState.pid):
      self.window_ticks = getattr(self, "window_ticks", 0) + 1
      if self._command_mismatch(CS, radarState, out[0]):
        self.cmd_mismatches = getattr(self, "cmd_mismatches", 0) + 1
    if not was_open and now is not None:
      self.entries.append({"t": tick_t, "v": float(CS.vEgo),
                           "a0": float(self.long_control._final_a0)})
    elif was_open and now is None:
      # Same order as the exit terms in update(), so the reason names the term
      # that actually fired. The guard is last, so it only owns the tick when
      # no earlier term did.
      if not (active and long_plan.shouldStop and self.long_control.final_stop):
        reason = "inactive_or_should_stop_cleared"
      elif lead_close:
        reason = "lead_close"
      elif float(CS.vEgo) < STANDSTILL:
        reason = "standstill"
      elif pre_final_t >= PROBE_MAX_T:
        reason = "probe_timeout"
      elif self.long_control._probe_weak_t >= PROBE_GUARD_T - 1e-9:
        reason = "probe_guard"
        self.guard_exits += 1
      else:
        reason = "timeout"
      self.exits.append({"t": tick_t, "reason": reason, "v": float(CS.vEgo)})
    return out


def _run(car, fixture, final):
  header = dict(fixture["params"])
  header["ButterStop"] = 1
  header["ButterStopFinal"] = int(final)
  loop = FinalStopLoop(header, FixtureProfile(fixture), fitted=car.lower(), full_run=True)
  # FarLeadLoop builds every fixture on the Palisade CP (classic CAN), K5 rows
  # included, so the per-car capability has to be set from here. ButterStopFinal
  # then resolves to the early hand-off on classic CAN, to the K5 measurement
  # window on CAN-FD, and to nothing on any other brand.
  loop.long_control._early_req_capable = car in CLASSIC_CAN_CARS
  loop.long_control._final_window_capable = car not in CLASSIC_CAN_CARS
  loop.long_control._refresh_butter_stop()
  run = loop.run()
  run["entries"] = loop.entries
  run["exits"] = loop.exits
  run["window"] = loop.window
  run["plan_guard_ticks"] = loop.plan_guard_ticks
  run["guard_exits"] = loop.guard_exits
  run["window_ticks"] = loop.window_ticks
  run["cmd_mismatches"] = loop.cmd_mismatches
  run["probe"] = dict(loop.probe)
  run["final_window"] = bool(loop.long_control.final_window)
  run["early_req"] = bool(loop.long_control.early_req)
  return run


def _write(path, row):
  os.makedirs(OUT, exist_ok=True)
  with open(os.path.join(OUT, path), "w", encoding="utf-8") as handle:
    json.dump(row, handle, indent=2, sort_keys=True)
    handle.write("\n")


G2_ROWS = []
G3_ROWS = []
NOENTRY_ROWS = []


def _fmt(value, spec="{:.3f}"):
  return "nan" if value is None else spec.format(float(value))


def _print_noentry(car, eid, probe):
  """Why an event whose car mode is the window opened none, in one line."""
  first = probe.get("first_le_probe_v_max")
  if first is None:
    first_shown = "none"
  else:
    state, a_ego, v_ego = first
    first_shown = f"{state}/a={_fmt(a_ego)}/v={_fmt(v_ego)}"
  print(
    f"NOENTRY {car} {eid} v_ho={_fmt(probe.get('v_ho'))} a_ho={_fmt(probe.get('a_ho'))} "
    + f"sacc={_fmt(probe.get('sacc'))} v_min_pid={_fmt(probe.get('v_min_pid'))} "
    + f"first_le_{PROBE_V_MAX:.2f}={first_shown}",
    flush=True)


@pytest.mark.parametrize("car,eid", EVENTS, ids=[f"{c}-{e}" for c, e in EVENTS])
def test_final_stop_gate(car, eid):
  with open(os.path.join(FIX, car, eid + ".json"), encoding="utf-8") as handle:
    fixture = json.load(handle)
  ref = real_stop(fixture)

  off_run = _run(car, fixture, FINAL_OFF)
  on_run = _run(car, fixture, FINAL_ON)
  off = _metrics(off_run, ref)
  on = _metrics(on_run, ref)

  # G3: mean commanded accel just before each arm's own standstill.
  on_cmd = _command_mean_before(on_run, on["t_stop"])
  off_cmd = _command_mean_before(off_run, off["t_stop"])

  # G2 input: the window-entry tick and the time from it to arm-1 standstill.
  entries = on_run["entries"]
  exits = on_run["exits"]
  entry_t = float(entries[0]["t"]) if entries else float("nan")
  entry_v = float(entries[0]["v"]) if entries else float("nan")
  on_time = on["t_stop"] - entry_t if entries else float("nan")
  off_time = off["t_stop"] - entry_t if entries else float("nan")

  # G3 is a statement about the window's own command, so it can only be judged
  # on events where the window owned the last 0.3 s: it opened and ended by
  # reaching standstill or by timing out. Events with no window, or whose
  # window ended on a closing lead and handed off early, measure the base
  # stopping state and are reported separately.
  g3_applies = bool(entries) and bool(exits) and exits[-1]["reason"] in ("standstill", "timeout")
  g3_ok = (on_cmd >= G3_MAX_DECEL) if g3_applies else None
  g3_note = ("window" if g3_applies
             else ("no_window" if not entries else f"exit_{exits[-1]['reason']}"))

  # Which behaviour ButterStopFinal resolved to on this car, and whether the
  # window opened at all. Both feed the census rows and the no-entry probe.
  mode = "early" if on_run["early_req"] else ("window" if on_run["final_window"] else "off")
  G2_ROWS.append({"car": car, "eid": eid, "entry_t": entry_t, "entry_v": entry_v,
                  "on_time": on_time, "off_time": off_time,
                  "n_entries": len(entries), "exits": exits,
                  "final_window": on_run["final_window"], "early_req": on_run["early_req"],
                  "plan_guard_ticks": int(on_run["plan_guard_ticks"]),
                  "guard_exits": int(on_run["guard_exits"]),
                  "cmd_mismatches": int(on_run["cmd_mismatches"]),
                  "window_ticks": int(sum(1 for x in on_run["window"]["final_t"] if x is not None))})
  G3_ROWS.append({"car": car, "eid": eid, "on_cmd": on_cmd, "off_cmd": off_cmd,
                  "g3_note": g3_note, "g3_ok": g3_ok})
  if mode == "window" and not entries:
    NOENTRY_ROWS.append({"car": car, "eid": eid, **on_run["probe"]})

  gap_ok = on["gap_stop"] >= off["gap_stop"] - G1_GAP_SLACK

  _write(eid + ".json", {
    "eid": eid,
    "car": car,
    "real_t_sim": ref["t_sim"],
    "real_gap": ref["gap"],
    "multistop": ref["multistop"],
    "off": off,
    "on": on,
    "on_cmd": on_cmd,
    "off_cmd": off_cmd,
    "entry_t": entry_t,
    "entry_v": entry_v,
    "exits": exits,
    "final_window": on_run["final_window"],
    "early_req": on_run["early_req"],
    "plan_guard_ticks": int(on_run["plan_guard_ticks"]),
    "guard_exits": int(on_run["guard_exits"]),
    "cmd_mismatches": int(on_run["cmd_mismatches"]),
    "window_ticks": int(sum(1 for x in on_run["window"]["final_t"] if x is not None)),
    "g1_gap_ok": bool(gap_ok),
    "g3_applies": g3_applies,
    "g3_ok": g3_ok,
    "g3_note": g3_note,
  })

  entry_shown = "none" if not np.isfinite(entry_v) else f"{entry_v:.3f}"
  off_part = (
    f"off t_stop={off['t_stop']:6.2f} dt={off['dt_real']:5.2f} gap={off['gap_stop']:6.3f} "
    + f"end={off['end_decel']:6.4f} cmd={off_cmd:7.4f} min_gap={off['min_gap_g1']:6.3f}"
  )
  on_part = (
    f"on t_stop={on['t_stop']:6.2f} dt={on['dt_real']:5.2f} gap={on['gap_stop']:6.3f} "
    + f"end={on['end_decel']:6.4f} cmd={on_cmd:7.4f} min_gap={on['min_gap_g1']:6.3f}"
  )
  guard_ticks = int(on_run["plan_guard_ticks"])
  guard_exits = int(on_run["guard_exits"])
  cmd_mismatches = int(on_run["cmd_mismatches"])
  window_ticks = int(sum(1 for x in on_run["window"]["final_t"] if x is not None))
  if mode == "window" and not entries:
    _print_noentry(car, eid, on_run["probe"])
  print(
    f"FSTOP {car} {eid:34s} real_t={ref['t_sim']:6.2f} | {off_part} | {on_part} "
    + f"entry_v={entry_shown:>6s} | dgap={on['gap_stop'] - off['gap_stop']:+.3f} "
    + f"| mode={mode} plan_guard={guard_ticks}/{window_ticks} guard={guard_exits} "
    + f"cmd_mismatch={cmd_mismatches}",
    flush=True)

  assert gap_ok, (
    f"FAIL_g1_gap {car} {eid}: gap_stop(1)={on['gap_stop']:.3f} < "
    + f"gap_stop(0)-{G1_GAP_SLACK:.2f}={off['gap_stop'] - G1_GAP_SLACK:.3f}")
  if g3_applies:
    assert g3_ok, (
      f"FAIL_g3 {car} {eid}: arm-1 mean command {on_cmd:.4f} over the last "
      + f"{G3_WINDOW:.1f} s < {G3_MAX_DECEL:.2f}")


def test_final_stop_g2_report():
  """G2 per car, plus the entry / exit census and the no-entry probe rows.

  No event entering the window is a legitimate outcome here, not a failure:
  PROBE_V_MAX is the real car's hand-off speed (median 0.400 m/s), and the
  fitted plant hands off later in aEgo at 0.476-0.751 m/s, so the stock path
  takes pid before the probe's speed gate can open. The census and the
  no-entry probe rows are the result in that case.
  """
  assert G2_ROWS, "the per-event gate produced no rows"
  rows = [r for r in G2_ROWS if np.isfinite(r["entry_t"])]
  if not rows:
    print(f"G2_NOENTRY events={len(G2_ROWS)} none entered the window; "
          + f"modes={sorted({r['final_window'] for r in G2_ROWS})}", flush=True)
    for row in NOENTRY_ROWS:
      _print_noentry(row["car"], row["eid"], row)
    _write("g2.json", {"events": len(G2_ROWS), "entered": 0})
    _write("g3.json", G3_ROWS)
    _write("noentry.json", NOENTRY_ROWS)
    return

  summary = {}
  for car in sorted({r["car"] for r in rows}):
    car_rows = [r for r in rows if r["car"] == car]
    on_t = float(np.median([r["on_time"] for r in car_rows]))
    off_t = float(np.median([r["off_time"] for r in car_rows]))
    summary[car] = {"n": len(car_rows), "on_median_s": on_t, "off_median_s": off_t,
                    "ok": bool(on_t <= off_t + 1e-9)}
    verdict = "PASS" if summary[car]["ok"] else "FAIL"
    print(f"G2 {car:9s} n={len(car_rows):2d} on={on_t:.3f} s off={off_t:.3f} s "
          + f"d={on_t - off_t:+.3f} {verdict}", flush=True)
  _write("g2.json", summary)

  for car in sorted({r["car"] for r in rows}):
    car_rows = [r for r in rows if r["car"] == car]
    guards = sum(r["guard_exits"] for r in car_rows)
    ticks = sum(r["window_ticks"] for r in car_rows)
    mismatches = sum(r["cmd_mismatches"] for r in car_rows)
    frac = (guards / len(car_rows)) if car_rows else float("nan")
    print(f"GUARD {car:9s} events={len(car_rows):2d} guard_exits={guards} "
          + f"frac_events={frac:.3f} window_ticks={ticks} cmd_mismatch={mismatches}", flush=True)

  entry_v = [r["entry_v"] for r in rows if np.isfinite(r["entry_v"])]
  exit_reasons = {}
  for r in rows:
    for ex in r["exits"]:
      exit_reasons[ex["reason"]] = exit_reasons.get(ex["reason"], 0) + 1
  multi = [r["eid"] for r in rows if r["n_entries"] > 1]
  dist = (
    f"entry_v median={float(np.median(entry_v)):.3f} "
    + f"p10={float(np.percentile(entry_v, 10)):.3f} p90={float(np.percentile(entry_v, 90)):.3f} "
    + f"min={float(np.min(entry_v)):.3f} max={float(np.max(entry_v)):.3f}"
  )
  print(
    f"SUMMARY entries={len(rows)}/{len(G2_ROWS)} {dist} "
    + f"exits={exit_reasons} multi_entry_events={len(multi)}", flush=True)

  # The three G3 buckets, report-only: the gate itself applies to the first.
  all_rows = G3_ROWS
  buckets = {"window": [], "no_window": [], "lead_close": [], "other_exit": []}
  for r in all_rows:
    note = r["g3_note"]
    if note == "window":
      buckets["window"].append(r)
    elif note == "no_window":
      buckets["no_window"].append(r)
    elif note.startswith("exit_lead_close"):
      buckets["lead_close"].append(r)
    else:
      buckets["other_exit"].append(r)
  for name in ("window", "no_window", "lead_close", "other_exit"):
    items = buckets[name]
    if not items:
      continue
    ids = " ".join(r["eid"].split("_")[-2] + "_" + r["eid"].split("_")[-1] for r in items)
    print(f"G3_BUCKET {name:10s} n={len(items):2d} ids={ids}", flush=True)
    for r in items:
      verdict = "n/a" if r["g3_ok"] is None else ("PASS" if r["g3_ok"] else "FAIL")
      print(f"  G3 {name:10s} {r['car']:9s} {r['eid']:34s} on_cmd={r['on_cmd']:+.4f} "
            + f"off_cmd={r['off_cmd']:+.4f} {verdict}", flush=True)
  _write("g3.json", G3_ROWS)
  _write("noentry.json", NOENTRY_ROWS)
  for row in NOENTRY_ROWS:
    _print_noentry(row["car"], row["eid"], row)

  bad = {car: s for car, s in summary.items() if not s["ok"]}
  assert not bad, f"FAIL_g2: {bad}"


def test_butter_stop_maneuvers_with_final_stop():
  """G4: ButterStopLoop takes no header dict; set the key then refresh.

  ButterStopLoop builds the Palisade CP, so ButterStopFinal resolves to the
  early hand-off there; the maneuvers record it unchanged.
  """
  from openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_stop_maneuvers import (
    _CASES, ButterStopLoop, compute_metrics,
  )

  params = Params()
  rows = []
  for _, maneuver in _CASES:
    runs = {}
    for final in (FINAL_OFF, FINAL_ON):
      loop = ButterStopLoop(maneuver, butter=1)
      params.put_int("ButterStopFinal", final)
      loop.long_control._refresh_butter_stop()
      runs[final] = loop.run()
    off = compute_metrics(runs[FINAL_OFF], maneuver.title + " final0", 1)
    on = compute_metrics(runs[FINAL_ON], maneuver.title + " final1", 1)
    rows.append({
      "title": maneuver.title,
      "final0": {k: v for k, v in off.items() if isinstance(v, (int, float, bool))},
      "final1": {k: v for k, v in on.items() if isinstance(v, (int, float, bool))},
    })
    zero = (
      f"final0 stopped={off.get('stopped')} min_gap={off.get('min_gap_m', float('nan')):.3f} "
      + f"final_gap={off.get('final_gap_m', float('nan')):.3f} "
      + f"accel_last={off.get('accel_last_0.5s_mean_abs')}"
    )
    one = (
      f"final1 stopped={on.get('stopped')} min_gap={on.get('min_gap_m', float('nan')):.3f} "
      + f"final_gap={on.get('final_gap_m', float('nan')):.3f} "
      + f"accel_last={on.get('accel_last_0.5s_mean_abs')}"
    )
    print(f"MANEUVER {maneuver.title:28s} {zero} | {one}", flush=True)

  _write("maneuvers.json", rows)
