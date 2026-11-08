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

_SCRATCH = "/mnt/c/Users/Administrator/Downloads/carrot-wip/carrot-openpilot/.analysis/scratch/2026-11-08-perf"
FIX = f"{_SCRATCH}/W-PLANT3B/fixtures"
POP = f"{_SCRATCH}/W-PLANTE/raw/sim/head"
OUT = f"{_SCRATCH}/W-FSTOP1/raw/sim"

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
    self._inner_update = self.long_control.update
    self.long_control.update = self._update

  def _update(self, active, CS, long_plan, accel_limits, t_since_plan, radarState):
    was_open = self.long_control._final_t is not None
    out = self._inner_update(active, CS, long_plan, accel_limits, t_since_plan, radarState)
    now = self.long_control._final_t
    tick_t = self.tick * DT_CTRL
    self.tick += 1
    fcw = bool(radarState.leadOne.status and radarState.leadOne.dRel < 4.0)
    self.window["t"].append(tick_t)
    self.window["final_t"].append(now)
    self.window["v"].append(float(CS.vEgo))
    self.window["accel"].append(float(out[0]))
    self.window["lc_state"].append(int(self.long_control.long_control_state))
    self.window["fcw"].append(fcw)
    if not was_open and now is not None:
      self.entries.append({"t": tick_t, "v": float(CS.vEgo),
                           "a0": float(self.long_control._final_a0)})
    elif was_open and now is None:
      if not (active and long_plan.shouldStop and self.long_control.final_stop):
        reason = "inactive_or_should_stop_cleared"
      elif fcw:
        reason = "fcw"
      elif float(CS.vEgo) < STANDSTILL:
        reason = "standstill"
      else:
        reason = "timeout"
      self.exits.append({"t": tick_t, "reason": reason, "v": float(CS.vEgo)})
    return out


def _run(car, fixture, final):
  header = dict(fixture["params"])
  header["ButterStop"] = 1
  header["ButterStopFinal"] = int(final)
  loop = FinalStopLoop(header, FixtureProfile(fixture), fitted=car.lower(), full_run=True)
  run = loop.run()
  run["entries"] = loop.entries
  run["exits"] = loop.exits
  return run


def _write(path, row):
  os.makedirs(OUT, exist_ok=True)
  with open(os.path.join(OUT, path), "w", encoding="utf-8") as handle:
    json.dump(row, handle, indent=2, sort_keys=True)
    handle.write("\n")


G2_ROWS = []
G3_ROWS = []


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
  entry_t = float(entries[0]["t"]) if entries else float("nan")
  entry_v = float(entries[0]["v"]) if entries else float("nan")
  on_time = on["t_stop"] - entry_t if entries else float("nan")
  off_time = off["t_stop"] - entry_t if entries else float("nan")
  G2_ROWS.append({"car": car, "eid": eid, "entry_t": entry_t, "entry_v": entry_v,
                  "on_time": on_time, "off_time": off_time,
                  "n_entries": len(entries), "exits": on_run["exits"]})
  G3_ROWS.append({"car": car, "eid": eid, "on_cmd": on_cmd, "off_cmd": off_cmd})

  gap_ok = on["gap_stop"] >= off["gap_stop"] - G1_GAP_SLACK
  g3_ok = on_cmd >= G3_MAX_DECEL
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
    "exits": on_run["exits"],
    "g1_gap_ok": bool(gap_ok),
    "g3_ok": bool(g3_ok),
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
  print(
    f"FSTOP {car} {eid:34s} real_t={ref['t_sim']:6.2f} | {off_part} | {on_part} "
    + f"entry_v={entry_shown:>6s} | dgap={on['gap_stop'] - off['gap_stop']:+.3f}",
    flush=True)

  assert gap_ok, (
    f"FAIL_g1_gap {car} {eid}: gap_stop(1)={on['gap_stop']:.3f} < "
    + f"gap_stop(0)-{G1_GAP_SLACK:.2f}={off['gap_stop'] - G1_GAP_SLACK:.3f}")
  assert g3_ok, (
    f"FAIL_g3 {car} {eid}: arm-1 mean command {on_cmd:.4f} over the last "
    + f"{G3_WINDOW:.1f} s < {G3_MAX_DECEL:.2f}")


def test_final_stop_g2_report():
  """G2 per car, plus the entry / exit census."""
  assert G2_ROWS, "the per-event gate produced no rows"
  rows = [r for r in G2_ROWS if np.isfinite(r["entry_t"])]
  assert rows, "FAIL_g2: no event entered the window"

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
  _write("g3.json", G3_ROWS)

  bad = {car: s for car, s in summary.items() if not s["ok"]}
  assert not bad, f"FAIL_g2: {bad}"


def test_butter_stop_maneuvers_with_final_stop():
  """G4: ButterStopLoop takes no header dict; set the key then refresh."""
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
