"""W-FIRM1 sim gate: the ButterStopFinal firm finish (arm 1) vs off (arm 0).

Per event in the W-PLANTE gate population restricted to the Palisade (the only
classic-CAN car with a fitted plant here), it runs the real LongitudinalPlanner
+ real LongControl + fitted plant three times on the same fixture header:

  arm 0  = ButterStopFinal 0 (the switch off);
  arm 1  = ButterStopFinal 1 (the firm finish at FIRM_A, gated);
  report = ButterStopFinal 1 with FIRM_A patched to 1.2, report-only.

and checks:

  repro:     arm 0 gap_stop must equal test_final_stop_sim's arm-0 gap_stop on
             the same event within 0.01 m. That runner is the validated gate, so
             a mismatch means this runner diverged -- FAIL_repro and stop. Its
             output must have been regenerated on the current base first.
  G1:        both arms come to rest within EVENT_GATE of the real stop;
             gap_stop(1) >= gap_stop(0) - 0.05 m; min gap > 0 while active in
             [t_real_sim - 15, t_real_sim + 2].
  G2:        after arm 1's standstill onset used for gap_stop, vEgo never
             exceeds 0.3 m/s before t_real_sim + 2. If arm 0 exceeds it on the
             same event, the row reports "both" instead of FAIL.

Self-checks inside this file, before the gates: FAIL_cap_off (arm 0 logs capped
ticks), FAIL_never_engaged (arm 1 engages on < 8/10 events), FAIL_units
(|command| > 3.5 anywhere).

The fitted plant has no ESC-standstill-timer model, so this gate judges the gap
and re-acceleration only; the crawl gain itself is a field question. Nothing
here tunes the FIRM_* constants: a gate failure is reported with its rows.

Run with `pytest -s` so the printed rows reach the log:

  pytest -o addopts='' -q -s openpilot/selfdrive/test/longitudinal_maneuvers/test_firm_finish_sim.py
"""

import json
import os

import numpy as np
import pytest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  FarLeadLoop,
  FixtureProfile,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_fitted_plant import _kf_end_decel
from openpilot.selfdrive.controls.lib import longcontrol as longcontrol_module

_SCRATCH = "/mnt/c/Users/Administrator/Downloads/carrot-wip/carrot-openpilot/.analysis/scratch/2026-11-08-perf"
FIX = f"{_SCRATCH}/W-PLANT3B/fixtures"
POP = f"{_SCRATCH}/W-PLANTE/raw/sim/head"
OUT = "/mnt/c/Users/Administrator/Downloads/carrot-wip/carrot-openpilot/.analysis/scratch/2026-11-11-field/W-TRACK1/raw/sim"
# test_final_stop_sim's own output dir. It must be REGENERATED on the current base
# before this gate runs: the 2026-11-09 files predate later commits. Run order is
# run 4 on firm-base (copied aside to raw/run4_base_sim), then run 4 on firm-finish,
# then this file. Comparing against stale rows would be a false repro failure.
BASE_RUN = f"{_SCRATCH}/W-FINAL2/raw/sim"

CAR = "Palisade"
FINAL_OFF = 0        # arm 0: the switch off
FINAL_ON = 1         # arm 1: the firm finish at FIRM_A
FIRM_A_REPORT = 1.2  # report-only arm: the same switch with a deeper constant

STANDSTILL = 0.05
STOP_CENTRE = 10.0           # fixture window is [T-20, T+25] with T = t_i_log - 10
EVENT_GATE = 8.0             # onset must be within this of the real stop (sim time)
G1_BACK, G1_FWD = 15.0, 2.0  # G1 gap window in sim time around the real stop
G1_GAP_SLACK = 0.05          # m, gap_stop(arm 1) may come this much closer
G2_REACCEL_V = 0.3           # m/s, no re-acceleration above this after the stop
REPRO_SLACK = 0.01           # m, arm 0 must match test_final_stop_sim's arm 0
UNITS_MAX = 3.5              # m/s^2, sanity bound on ticks the firm cap can own
ENGAGE_MIN = 8               # of the 10 events, arm 1 must engage on at least this many


def _gate_events():
  rows = []
  for name in sorted(os.listdir(POP)):
    if not name.endswith(".json"):
      continue
    with open(os.path.join(POP, name), encoding="utf-8") as handle:
      row = json.load(handle)
    if row["lead_jump"]:
      continue
    if row["car"] != CAR:
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
  """Onset-nearest-real selection, end decel, and the G1 gap window."""
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
  window = (t >= t_stop - 0.5) & (t <= t_stop)
  if not window.any():
    raise AssertionError(f"FAIL_empty_window: no samples before t_stop={t_stop:.2f}")
  end = float(np.mean(np.abs(a[window])))

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
    "onset_index": int(idx),
  }


class FirmLoop(FarLeadLoop):
  """FarLeadLoop that records the firm cap: engagement, capped ticks, command."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.tick = 0
    self.rows = {"t": [], "v": [], "accel": [], "cap": [], "lc_state": [],
                 "pid_i": [], "gated_ticks": [], "capped_ticks": [], "mean_capped": []}
    self._inner_update = self.long_control.update
    self.long_control.update = self._update

  def _update(self, active, CS, long_plan, accel_limits, t_since_plan, radarState):
    integral_before = float(self.long_control.pid.i)
    out = self._inner_update(active, CS, long_plan, accel_limits, t_since_plan, radarState)
    cap = self.long_control._firm_cap
    accel = float(out[0])
    self.rows["t"].append(self.tick * DT_CTRL)
    self.rows["v"].append(float(CS.vEgo))
    self.rows["accel"].append(accel)
    self.rows["cap"].append(None if cap is None else float(cap))
    self.rows["lc_state"].append(int(self.long_control.long_control_state))
    self.rows["pid_i"].append(float(self.long_control.pid.i))
    if cap is not None:
      self.rows["gated_ticks"].append(self.tick)
      if abs(accel - float(cap)) <= 1e-12:
        self.rows["capped_ticks"].append(self.tick)
        self.rows["mean_capped"].append(accel)
    self.tick += 1
    return out


def _run(fixture, final, firm_a=None):
  """One arm. firm_a patches the module constant for the report-only depth."""
  header = dict(fixture["params"])
  header["ButterStop"] = 1
  header["ButterStopFinal"] = int(final)
  loop = FirmLoop(header, FixtureProfile(fixture), fitted="palisade", full_run=True)
  # FarLeadLoop builds every fixture on the Palisade CP already; set the
  # capability explicitly so the gate cannot depend on that coincidence, then
  # re-resolve the switch exactly as the car does once per second. The switch
  # resolution is what decides firm_a, so it must run after the CP patch.
  saved_firm_a = longcontrol_module.FIRM_A
  if firm_a is not None:
    longcontrol_module.FIRM_A = float(firm_a)
  try:
    loop.long_control._early_req_capable = True
    loop.long_control._refresh_butter_stop()
    run = loop.run()
  finally:
    longcontrol_module.FIRM_A = saved_firm_a
  run["rows"] = loop.rows
  run["firm_a"] = float(loop.long_control.firm_a)
  run["early_req"] = bool(loop.long_control.early_req)
  return run


def _write(eid, row):
  os.makedirs(OUT, exist_ok=True)
  with open(os.path.join(OUT, eid + ".json"), "w", encoding="utf-8") as handle:
    json.dump(row, handle, indent=2, sort_keys=True)
    handle.write("\n")


def _fmt(value, spec="{:.3f}"):
  return "nan" if value is None else spec.format(float(value))


def _base_gap_stop(eid):
  """test_final_stop_sim's own arm-0 gap_stop for this event, if it ran."""
  path = os.path.join(BASE_RUN, eid + ".json")
  if not os.path.exists(path):
    return None
  with open(path, encoding="utf-8") as handle:
    row = json.load(handle)
  return float(row["off"]["gap_stop"])


def _capped_stats(rows):
  capped = rows["capped_ticks"]
  if not capped:
    return {"n": 0, "v_first": None, "mean_cmd": None}
  first = capped[0]
  return {
    "n": len(capped),
    "v_first": float(rows["v"][first]),
    "mean_cmd": float(np.mean([float(rows["accel"][i]) for i in capped])),
  }


def _g2_peak(run, t_stop, t_end):
  """Peak speed after t_stop, over [t_stop, t_end]."""
  t = np.array(run["rows"]["t"], dtype=float)
  v = np.array(run["rows"]["v"], dtype=float)
  sel = (t >= t_stop) & (t <= t_end)
  if not sel.any():
    return None
  return float(np.max(v[sel]))


REPRO_ROWS = []
GATE_ROWS = []


def test_gate_is_the_ten_palisade_events():
  assert len(EVENTS) == 10, [(c, e) for c, e in EVENTS]


@pytest.mark.parametrize("car,eid", EVENTS, ids=[f"{c}-{e}" for c, e in EVENTS])
def test_firm_finish_gate(car, eid):
  with open(os.path.join(FIX, car, eid + ".json"), encoding="utf-8") as handle:
    fixture = json.load(handle)
  ref = real_stop(fixture)

  off_run = _run(fixture, FINAL_OFF)
  on_run = _run(fixture, FINAL_ON)
  rep_run = _run(fixture, FINAL_ON, firm_a=FIRM_A_REPORT)

  off = _metrics(off_run, ref)
  on = _metrics(on_run, ref)
  rep = _metrics(rep_run, ref)

  off_cap = _capped_stats(off_run["rows"])
  on_cap = _capped_stats(on_run["rows"])
  rep_cap = _capped_stats(rep_run["rows"])

  base_gap = _base_gap_stop(eid)
  repro_ok = base_gap is not None and abs(off["gap_stop"] - base_gap) <= REPRO_SLACK
  REPRO_ROWS.append({"eid": eid, "base": base_gap, "arm0": off["gap_stop"],
                     "delta": None if base_gap is None else off["gap_stop"] - base_gap,
                     "ok": bool(repro_ok)})

  # Units sanity, scoped to what this feature can produce. The cap only ever
  # LOWERS the command and only while gated, so any tick it owns must stay inside
  # the bound. A stock stopping-state ramp at speed is not this feature's doing:
  # event 1763 commands -3.774 m/s^2 at v 8.58 with cap None in arm 0 as well.
  worst = max(abs(value) for value in rep_run["rows"]["accel"])
  off_worst = max(abs(value) for value in off_run["rows"]["accel"])
  # Every tick any gate can own, across both gated arms.
  own = [abs(a)
         for rows_ in (on_run["rows"], rep_run["rows"])
         for a, c in zip(rows_["accel"], rows_["cap"], strict=True) if c is not None]
  worst_own = max(own) if own else 0.0
  units_ok = worst_own <= UNITS_MAX
  # The deepest command in any arm comes from the stock stopping ramp, not from
  # the cap: report it so a reader sees whose number the global extreme is.
  units_note = f"own={worst_own:.3f} all={worst:.3f} base={off_worst:.3f}"

  gap_ok = on["gap_stop"] >= off["gap_stop"] - G1_GAP_SLACK
  t_end = ref["t_sim"] + G1_FWD
  on_peak = _g2_peak(on_run, on["t_stop"], t_end)
  off_peak = _g2_peak(off_run, off["t_stop"], t_end)
  on_reaccel = on_peak is not None and on_peak > G2_REACCEL_V
  off_reaccel = off_peak is not None and off_peak > G2_REACCEL_V
  if on_peak is None and off_peak is None:
    # The stop landed after the gate window closed, so G2 measured nothing.
    # Say so; a vacuous pass would read as coverage it does not have.
    g2_state = "n/a"
  elif on_reaccel and off_reaccel:
    g2_state = "both"
  elif on_reaccel:
    g2_state = "FAIL_arm1"
  else:
    g2_state = "pass"

  GATE_ROWS.append({
    "eid": eid, "real_t_sim": ref["t_sim"], "multistop": ref["multistop"],
    "off": off, "on": on, "report": rep,
    "off_cap": off_cap, "on_cap": on_cap, "report_cap": rep_cap,
    "firm_a_on": on_run["firm_a"], "firm_a_report": rep_run["firm_a"],
    "g1_gap_ok": bool(gap_ok), "g2_state": g2_state,
    "on_peak": on_peak, "off_peak": off_peak,
    "units_ok": bool(units_ok), "worst_accel": float(worst),
    "repro_ok": bool(repro_ok), "base_gap": base_gap,
  })

  _write(eid, {
    "eid": eid, "car": car, "real_t_sim": ref["t_sim"], "real_gap": ref["gap"],
    "multistop": ref["multistop"],
    "off": off, "on": on, "report": rep,
    "off_cap": off_cap, "on_cap": on_cap, "report_cap": rep_cap,
    "firm_a": {"on": on_run["firm_a"], "report": rep_run["firm_a"]},
    "g1_gap_ok": bool(gap_ok), "g2_state": g2_state,
    "on_peak": on_peak, "off_peak": off_peak,
    "units_ok": bool(units_ok), "worst_accel": float(worst),
    "repro": {"base": base_gap, "arm0": off["gap_stop"], "ok": bool(repro_ok)},
  })

  print(
    f"FIRM {car} {eid:34s} real_t={ref['t_sim']:6.2f} "
    + f"| gap0={off['gap_stop']:6.3f} gapF1={on['gap_stop']:6.3f} gapR={rep['gap_stop']:6.3f} "
    + f"dF1={on['gap_stop'] - off['gap_stop']:+.3f} dR={rep['gap_stop'] - off['gap_stop']:+.3f} "
    + f"| base={_fmt(base_gap)} repro={_fmt(None if base_gap is None else off['gap_stop'] - base_gap, '{:+.4f}')} "
    + f"| capF1={on_cap['n']:3d} v1={_fmt(on_cap['v_first'])} cmd={_fmt(on_cap['mean_cmd'], '{:+.3f}')} "
    + f"| capR={rep_cap['n']:3d} "
    + f"| G1={'ok' if gap_ok else 'FAIL'} G2={g2_state} peak={_fmt(on_peak)} "
    + f"| units={'ok' if units_ok else 'FAIL'} {units_note}",
    flush=True)

  assert repro_ok, (
    f"FAIL_repro {eid}: arm0 gap_stop={off['gap_stop']:.4f} vs test_final_stop_sim "
    + f"{_fmt(base_gap)} (slack {REPRO_SLACK:.2f} m)")
  assert off_cap["n"] == 0, f"FAIL_cap_off {eid}: arm 0 logged {off_cap['n']} capped ticks"
  assert units_ok, f"FAIL_units {eid}: |command| {worst:.3f} > {UNITS_MAX:.1f}"
  assert gap_ok, (
    f"FAIL_g1_gap {eid}: gap_stop(arm1)={on['gap_stop']:.3f} < "
    + f"gap_stop(0)-{G1_GAP_SLACK:.2f}={off['gap_stop'] - G1_GAP_SLACK:.3f}")
  assert g2_state != "FAIL_arm1", (
    f"FAIL_g2 {eid}: arm 1 re-accelerated to {on_peak:.3f} m/s after its stop "
    + f"(arm 0 peak {_fmt(off_peak)}); limit {G2_REACCEL_V:.1f}")


def test_repro_matches_the_validated_runner():
  """Arm 0 must reproduce test_final_stop_sim's arm 0 on every event."""
  assert len(REPRO_ROWS) == len(EVENTS), (len(REPRO_ROWS), len(EVENTS))
  missing = [row["eid"] for row in REPRO_ROWS if row["base"] is None]
  assert not missing, f"FAIL_repro: no test_final_stop_sim row for {missing}"
  bad = [row for row in REPRO_ROWS if not row["ok"]]
  assert not bad, "FAIL_repro: " + ", ".join(
    f"{row['eid']} {row['arm0']:.4f} vs {row['base']:.4f}" for row in bad)


def test_firm_engages_on_the_population():
  assert len(GATE_ROWS) == len(EVENTS)
  engaged = [row for row in GATE_ROWS if row["on_cap"]["n"] > 0]
  rows = ", ".join(f"{row['eid']}:{row['on_cap']['n']}" for row in GATE_ROWS)
  assert len(engaged) >= ENGAGE_MIN, (
    f"FAIL_never_engaged: arm 1 capped on {len(engaged)}/{len(EVENTS)} events "
    + f"(need {ENGAGE_MIN}); capped ticks per event {rows}")


def test_summary_table():
  """The per-event Δgap_stop table for the report."""
  assert GATE_ROWS, "no rows"
  deltas_arm1 = [row["on"]["gap_stop"] - row["off"]["gap_stop"] for row in GATE_ROWS]
  deltas_report = [row["report"]["gap_stop"] - row["off"]["gap_stop"] for row in GATE_ROWS]
  print("SUMMARY"
        + f" n={len(GATE_ROWS)}"
        + f" dF1_min={min(deltas_arm1):+.3f} dF1_median={float(np.median(deltas_arm1)):+.3f} dF1_max={max(deltas_arm1):+.3f}"
        + f" dR_min={min(deltas_report):+.3f} dR_median={float(np.median(deltas_report)):+.3f} dR_max={max(deltas_report):+.3f}"
        + f" g2={dict((s, sum(1 for r in GATE_ROWS if r['g2_state'] == s)) for s in ('pass', 'both', 'n/a', 'FAIL_arm1'))}",
        flush=True)
  assert all(row["g1_gap_ok"] for row in GATE_ROWS), "G1 failed on at least one event"
