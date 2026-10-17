"""STOP-NOD step 1: real-car low-speed plant calibration + baseline (test only).

Diagnostic / calibration harness for the "car nods at the stop instant" complaint.
It fits a candidate low-speed plant (`plant_real_step`: command dead time L, a
first-order lag tau, and a speed-dependent creep term) to six real auto-stops
(a_imu_fwd truth, see data/butter_nod/real_stops_nod_50hz.json) with ONE
parameter set, reports per-stop fidelity against fixed PASS targets, and runs a
closed-loop baseline with the best plant ("real") over the b1 fixture and every
maneuver scenario of test_butter_stop_maneuvers (plus the start scenarios).

Step 1b adds the regime-split plant mode "real_ls": the OLD validated plant
(test_far_lead_replay.py: PLANT_TAU 0.3, creep, PLANT_SCC_BRAKE stopping
override) governs v >= 4 m/s and the NEW low-speed plant (dead time L, lag tau,
creep c(v)) governs v <= 3 m/s, blended linearly between. The fit is restricted
to the low-speed window (v < 4 m/s, after the last gas row + 0.2 s, up to
t_rel = 0) because the single global lag plant was only plausible there. The
"real" mode/plant code is kept, unused by default.

This file changes no production code. Existing plants (ideal / creep / scc) are
untouched. Simulation only; no vehicle validation. Assertions are limited to the
fixture sha256, that the fit ran and no collision; fidelity pass/fail is
REPORTED, not asserted.
"""
import hashlib
import json
import math
import os
import types

import numpy as np
import pytest

from opendbc.car.hyundai.interface import CarInterface
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, CONTROL_N_T_IDX
from openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_stop_maneuvers import (
  ButterStopLoop, _A_SCENARIOS, _B2_PROXY, _B_SCENARIO, _CASES, _S1, _S3, _S3B, _S3C,
  _START_SCENARIOS, _FakeSM, PLANNER_PERIOD,
)
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers
from openpilot.selfdrive.test.longitudinal_maneuvers.test_far_lead_replay import (
  FarLeadLoop, FixtureProfile, PLANT_A_GATE, PLANT_SCC_BRAKE, PLANT_TAU,
  load_fixture, plant_creep,
)

NOD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "butter_nod",
                        "real_stops_nod_50hz.json")
NOD_SHA256 = "07f34cdd72e1eab0638c4fb75df557ef3cd3cfc3e17787842d24fab86a8d1933"
FIT_DT = 0.02
FIT_START_S = -4.0
COLLISION_GAP_M = 0.4
GAS_COL = 9
V_SW = 4.0  # real_ls switch speed (m/s): fully OLD at v >= 4.0, fully NEW at v <= 3.0
LS_V_MAX = 4.0  # low-speed fit window: rows with v_ego < this
LS_GAS_PAD_S = 0.2

# Grid axes (brief step 1.2).
P_L = [round(0.1 * i, 1) for i in range(7)]            # 0.0 .. 0.6
P_TAU = [round(0.1 * (i + 1), 1) for i in range(10)]   # 0.1 .. 1.0
P_CMAX = [round(0.05 * i, 2) for i in range(11)]       # 0.0 .. 0.5
P_VHI = [round(0.2 * (i + 1), 1) for i in range(8)]    # 0.2 .. 1.6
P_CLOW = [round(-0.3 + 0.1 * i, 1) for i in range(7)]  # -0.3 .. 0.3

# Low-speed fit grid axes (brief step 1b.2).
LS_L = [round(0.1 * i, 1) for i in range(13)]            # 0.0 .. 1.2
LS_TAU = [round(0.1 * (i + 1), 1) for i in range(16)]    # 0.1 .. 1.6
LS_CMAX = [round(0.05 * i, 2) for i in range(11)]        # 0.0 .. 0.5
LS_VHI = [round(0.2 * (i + 1), 1) for i in range(8)]     # 0.2 .. 1.6
LS_CLOW = [round(-0.3 + 0.1 * i, 1) for i in range(10)]  # -0.3 .. 0.6

# real_sr (regime-split + SCC stop) window and grid (brief step 1c.2).
SR_WIN_END = 0.6   # window ends at t_rel +0.6
SR_STOPPING = 2    # fixture long_state value for LongCtrlState.stopping
# Reference set (vfit3.py); the implementation must reproduce it within tolerance.
SR_REF = {"L": 0.6, "tau": 0.8, "SCC_A": -0.4, "C_MAX": 0.1, "V_HI": 1.0, "C_LOW": 0.3}
SR_L = [round(0.2 + 0.1 * i, 1) for i in range(9)]          # 0.2 .. 1.0
SR_TAU = [round(0.3 + 0.1 * i, 1) for i in range(10)]       # 0.3 .. 1.2
SR_SCC = [round(-0.10 - 0.05 * i, 2) for i in range(15)]    # -0.10 .. -0.80
SR_CMAX = [round(0.05 * i, 2) for i in range(7)]            # 0.0 .. 0.30
SR_VHI = [0.25, 0.5, 1.0, 1.5, 2.0]
SR_CLOW = [round(0.05 * i, 2) for i in range(13)]           # 0.0 .. 0.60

# PASS targets (brief step 1.3).
T_RMS = 0.12
T_STOP_DECEL = 0.15   # decel error at t_rel = -0.3 s (replaces t = 0)
T_STOP_TIME = 0.4
T_DIST = 0.25
DECEL_AT_S = -0.3     # gated fidelity time point
DECEL_INFO_S = (-1.0, -0.6, -0.3, 0.0)

# real_sr fidelity targets (brief step 1c.4).
SR_T_DIST = 0.25
SR_T_STOP_TIME = 0.4
SR_T_VRMS = 0.08
SR_T_AWRMS = 0.12
SR_CROSSCHECK_DIST = 0.02   # |ours - vfit3| distance match tolerance
SR_CROSSCHECK_TIME = 0.05   # |ours - vfit3| stop-time match tolerance

# real_sg (real_sr + road-grade term) window/grid and reference (brief 1d).
G_ACCEL = 9.81
# Per-stop raw standstill grade signal: -accelerometer.z median over t_rel
# 1.5-3.0 s, BEFORE the fixture's offset removal (parent CP-D review). More
# negative = more downhill. Keyed "NNN@MM" from route[5:8] + "@" + minute.
SR_HOLD = {"443@8.82": -0.895, "445@1.43": -0.455, "445@7.27": -0.548,
           "445@8.52": -0.848, "447@7.07": -0.922, "44c@6.21": -0.810}
SRG_REF = {"M0": -0.8, "K": 0.25, "L": 0.6, "tau": 0.5, "SCC_A": -0.5,
           "C_MAX": 0.2, "V_HI": 0.5, "C_LOW": 0.2}
SRG_M0 = [round(-1.0 + 0.1 * i, 1) for i in range(6)]     # -1.0 .. -0.5
SRG_K = [round(0.125 * i, 3) for i in range(9)]           # 0.0 .. 1.0
SRG_L = [0.5, 0.6, 0.7]
SRG_TAU = [0.4, 0.5, 0.6, 0.8]
SRG_SCC = [round(-0.30 - 0.05 * i, 2) for i in range(7)]  # -0.30 .. -0.60
SRG_CMAX = [round(0.05 * i, 2) for i in range(7)]         # 0.0 .. 0.30
SRG_VHI = [0.5, 1.0, 1.5]
SRG_CLOW = [round(0.05 * i, 2) for i in range(11)]        # 0.0 .. 0.50
SRG_GRADES = (-0.03, 0.03)  # closed-loop scenario grades (downhill, uphill)
SRG_GAP_ORDER_M = 0.05      # flag if B1 stops closer than B0 by more than this

# The CHOSEN real_sg plant from step 1d: the reference plant for all later steps.
PLANT_REAL = {"M0": -0.5, "K": 0.375, "L": 0.7, "tau": 0.4,
              "SCC_A": -0.6, "C_MAX": 0.1, "V_HI": 0.5, "C_LOW": 0.0}

BASELINE_PARAMS = {"ButterStop": 1, "VEgoStopping": 10, "StoppingAccel": -50}


def _trapz(y, x):
  return float(np.trapezoid(y, x))


# --- candidate plant -------------------------------------------------------

def plant_real_step(state, cmd, v, dt, p):
  """One step of the candidate real-car plant.

  state: mutable dict (holds the command history and the lag state).
  cmd:   commanded acceleration (m/s^2) at this step.
  v:     current speed (m/s). p: {"L","tau","C_MAX","V_HI","C_LOW"}.

  The command is delayed by L, then first-order lagged with tau; a creep term
  c(v) = C_MAX for v >= V_HI else linear C_LOW..C_MAX is added only while the
  lagged command is < 0.3 and v > 0.01; at v == 0 the car holds (a_act = 0).
  Returns (a_act, v_next).
  """
  L = float(p["L"])
  tau = float(p["tau"])
  C_MAX = float(p["C_MAX"])
  V_HI = float(p["V_HI"])
  C_LOW = float(p["C_LOW"])

  hist = state.setdefault("hist", [])
  hist.append(float(cmd))
  n = int(round(L / dt)) if dt > 0.0 else 0
  if n <= 0:
    cmd_d = hist[-1]
  else:
    idx = len(hist) - 1 - n
    cmd_d = hist[idx] if idx >= 0 else hist[0]

  a_lag = float(state.get("a_lag", cmd_d))
  a_lag += (dt / tau) * (cmd_d - a_lag) if tau > 0.0 else (cmd_d - a_lag)
  state["a_lag"] = a_lag

  if v >= V_HI:
    c = C_MAX
  else:
    c = C_LOW + (C_MAX - C_LOW) * (v / V_HI if V_HI > 0.0 else 0.0)
  a_act = a_lag
  if a_lag < 0.3 and v > 0.01:
    a_act += c
  if v <= 0.0:
    a_act = 0.0
  v_next = max(0.0, v + a_act * dt)
  return a_act, v_next


def old_plant_step(state, cmd, v, dt, scc_stop=False, lc_stopping=False):
  """The OLD validated plant (test_far_lead_replay.py), verbatim.

  First-order lag PLANT_TAU on the command, plant_creep(v) added while the
  command is below PLANT_A_GATE and v > 0.01, and the PLANT_SCC_BRAKE stopping
  override when scc_stop and LongControl is in the stopping state. `cmd` is the
  raw LongControl output (the creep gate uses it, not the lagged value). Returns
  the applied acceleration.
  """
  a_lag = float(state.get("a_lag", 0.0))
  a_lag += (dt / PLANT_TAU) * (cmd - a_lag)
  state["a_lag"] = a_lag
  applied = a_lag
  if cmd < PLANT_A_GATE and v > 0.01:
    applied += plant_creep(v)
  if scc_stop and lc_stopping:
    applied = 0.0 if v <= 0.01 else min(applied, PLANT_SCC_BRAKE)
  return applied


def real_ls_step(state, cmd, v, dt, p, scc_stop=False, lc_stopping=False):
  """Regime-split plant: OLD above v >= 4 m/s, NEW below v <= 3 m/s, blended.

  Both plants keep their own internal state in `state` across steps; no reset at
  the switch. w = clip((V_SW - v) / 1.0, 0, 1); applied = w*a_new + (1-w)*a_old.
  """
  old_state = state.setdefault("old", {})
  new_state = state.setdefault("new", {})
  a_old = old_plant_step(old_state, cmd, v, dt, scc_stop=scc_stop, lc_stopping=lc_stopping)
  a_new, _ = plant_real_step(new_state, cmd, v, dt, p)
  w = min(1.0, max(0.0, (V_SW - v) / 1.0))
  return w * a_new + (1.0 - w) * a_old


def sr_new_step(state, cmd, v, dt, p, long_state, diag=None):
  """NEW low-speed plant with the SCC stop regime (brief 1c.1).

  The command is delayed by L. While the (delayed) LongControl state is
  `stopping`, the target is SCC_A (the SCC transmits its own StopReq decel, not
  cmd); otherwise the target is the delayed cmd. The lag follows
  a_lag += (target - a_lag) * min(1, dt / tau); creep c(v) = C_LOW + (C_MAX -
  C_LOW) * min(1, v / V_HI) is added only while a_lag < 0.3 and v > 0.01; at
  v <= 0 a negative accel is clamped to 0. Returns (a_act, v_next).
  """
  L = float(p["L"]); tau = float(p["tau"]); scc_a = float(p["SCC_A"])
  C_MAX = float(p["C_MAX"]); V_HI = float(p["V_HI"]); C_LOW = float(p["C_LOW"])

  hist = state.setdefault("hist", [])
  hist.append((float(cmd), float(long_state)))
  n = int(round(L / dt)) if dt > 0.0 else 0
  idx = len(hist) - 1 - n
  cmd_d, st_d = hist[idx] if idx >= 0 else hist[0]

  target = scc_a if int(st_d) == SR_STOPPING else cmd_d
  a_lag = float(state.get("a_lag", target))
  a_lag += (target - a_lag) * min(1.0, dt / tau) if tau > 0.0 else (target - a_lag)
  state["a_lag"] = a_lag

  c = C_LOW + (C_MAX - C_LOW) * min(1.0, v / V_HI) if V_HI > 0.0 else C_MAX
  a_act = a_lag + (c if (a_lag < 0.3 and v > 0.01) else 0.0)
  if v <= 0.0:
    a_act = 0.0 if a_act < 0.0 else a_act
  v_next = max(0.0, v + a_act * dt)
  if diag is not None:
    diag["target"] = float(target)
    diag["delayed_state"] = int(st_d)
    diag["a_lag"] = float(a_lag)
    diag["creep"] = float(c) if (a_lag < 0.3 and v > 0.01) else 0.0
  return a_act, v_next


def real_sr_step(state, cmd, v, dt, p, long_state):
  """Regime-split plant: OLD above v >= 4 m/s, NEW (real_sr) below v <= 3 m/s.

  Same speed blend as real_ls. `long_state` is the controller's own
  longControlState at this step (feeds the delayed SCC stop regime).
  """
  old_state = state.setdefault("old", {})
  new_state = state.setdefault("new", {})
  lc_stopping = int(long_state) == SR_STOPPING
  a_old = old_plant_step(old_state, cmd, v, dt, scc_stop=False, lc_stopping=lc_stopping)
  a_new, _ = sr_new_step(new_state, cmd, v, dt, p, long_state)
  w = min(1.0, max(0.0, (V_SW - v) / 1.0))
  return w * a_new + (1.0 - w) * a_old


def sg_new_step(state, cmd, v, dt, p, long_state, a_grade, diag=None):
  """real_sr NEW plant plus a road-grade acceleration term (brief 1d.1).

  Identical to sr_new_step except `a_grade` (m/s^2) is added to the applied
  accel. The grade term is applied in the NEW (low-speed) plant only.
  """
  a_act, v_next = sr_new_step(state, cmd, v, dt, p, long_state, diag=diag)
  if v > 0.0 or a_act + a_grade > 0.0:
    a_act = a_act + a_grade
    if v <= 0.0 and a_act < 0.0:
      a_act = 0.0
    v_next = max(0.0, v + a_act * dt)
  if diag is not None:
    diag["grade"] = float(a_grade)
  return a_act, v_next


def real_sg_step(state, cmd, v, dt, p, long_state, a_grade, diag=None):
  """Regime-split plant: OLD above v >= 4 m/s, NEW (real_sg) below v <= 3 m/s.

  `a_grade` is added inside the NEW plant only; the OLD plant is unchanged.
  """
  old_state = state.setdefault("old", {})
  new_state = state.setdefault("new", {})
  lc_stopping = int(long_state) == SR_STOPPING
  a_old = old_plant_step(old_state, cmd, v, dt, scc_stop=False, lc_stopping=lc_stopping)
  a_new, _ = sg_new_step(new_state, cmd, v, dt, p, long_state, a_grade, diag=diag)
  w = min(1.0, max(0.0, (V_SW - v) / 1.0))
  if diag is not None:
    diag["w_new"] = float(w)
    diag["a_old"] = float(a_old)
    diag["a_new"] = float(a_new)
  return w * a_new + (1.0 - w) * a_old


# --- fixture load ----------------------------------------------------------

def _read_nod_raw():
  with open(NOD_PATH, "rb") as handle:
    return handle.read()


def load_stops():
  d = json.loads(_read_nod_raw().decode("utf-8"))
  stops = []
  for entry in d["stops"]:
    rows = np.array(entry["rows"], dtype=float)
    gas_cut_t = None
    if rows[:, GAS_COL].max() > 0.5:
      # Stop 445@1.43 (and any gas-tainted stop): use only rows after the last gas.
      gas = np.flatnonzero(rows[:, GAS_COL] > 0.5)
      gas_cut_t = float(rows[gas[-1], 0])
      rows = rows[gas[-1] + 1:]
    stops.append({"route": entry["route"], "minute": entry["minute"], "rows": rows,
                  "gas_cut_t": gas_cut_t})
  return d, stops


def _prepare_stop(stop):
  """Native fixture samples from t_rel -4.0 on (irregular ~50 Hz timestamps)."""
  rows = stop["rows"]
  t = rows[:, 0]
  i0 = int(np.flatnonzero(t >= FIT_START_S)[0])
  tt = t[i0:]
  dts = np.diff(tt, prepend=tt[0] - (tt[1] - tt[0]) if tt.size > 1 else FIT_DT)
  dts = np.maximum(dts, 1e-6)
  return {
    "t_full": t, "v_full": rows[:, 1], "imu_full": rows[:, 3],
    "t": tt, "dts": dts,
    "v0": float(rows[i0, 1]),
    "cmd": rows[i0:, 4].copy(),
    "imu": rows[i0:, 3].copy(),
  }


# --- open-loop fit (vectorized over the creep axes) -----------------------

def _vec_rms(prep, L, tau, CM, VH, CL):
  cmd = prep["cmd"]
  imu = prep["imu"]
  dts = prep["dts"]
  N = cmd.size
  a_lag = np.empty(N)
  a_lag[0] = cmd[0]
  for i in range(1, N):
    n = int(round(L / dts[i]))
    cmd_d = cmd[max(0, i - n)]
    a_lag[i] = a_lag[i - 1] + (dts[i] / tau) * (cmd_d - a_lag[i - 1])

  K = CM.size
  v = np.full(K, prep["v0"])
  done = np.zeros(K, dtype=bool)
  err2 = np.zeros(K)
  cnt = np.zeros(K)
  for i in range(N):
    c = np.where(v >= VH, CM, CL + (CM - CL) * (v / VH))
    gate = (a_lag[i] < 0.3) & (v > 0.01)
    a_act = a_lag[i] + np.where(gate, c, 0.0)
    a_act = np.where(v <= 0.0, 0.0, a_act)
    m = ~done
    err2 += np.where(m, (a_act - imu[i]) ** 2, 0.0)
    cnt += m
    v_next = np.maximum(0.0, v + a_act * dts[i])
    done = done | (m & (v_next < 0.01))
    v = v_next
  rms = np.sqrt(err2 / np.maximum(cnt, 1.0))
  rms[cnt == 0.0] = np.nan
  return rms


def _mesh(cmax_list, vhi_list, clow_list):
  cm, vh, cl = np.meshgrid(np.array(cmax_list), np.array(vhi_list), np.array(clow_list),
                           indexing="ij")
  return cm.ravel(), vh.ravel(), cl.ravel()


def fit_real_plant(stops, L_list, tau_list, cmax_list, vhi_list, clow_list):
  """Grid-search ONE parameter set for all stops; returns (loss, params) sorted."""
  preps = [_prepare_stop(s) for s in stops]
  CM, VH, CL = _mesh(cmax_list, vhi_list, clow_list)
  K = CM.size
  results = []
  for L in L_list:
    for tau in tau_list:
      per_stop = np.empty((len(preps), K))
      for si, prep in enumerate(preps):
        per_stop[si] = _vec_rms(prep, L, tau, CM, VH, CL)
      avg = np.nanmean(per_stop, axis=0)
      for k in range(K):
        if np.isfinite(avg[k]):
          results.append((float(avg[k]), {"L": L, "tau": tau, "C_MAX": float(CM[k]),
                                          "V_HI": float(VH[k]), "C_LOW": float(CL[k])}))
  results.sort(key=lambda item: item[0])
  return results


def _simulate_scalar(prep, p):
  cmd = prep["cmd"]
  dts = prep["dts"]
  N = cmd.size
  v = prep["v0"]
  a_lag = cmd[0]
  vs = np.empty(N)
  aas = np.empty(N)
  stop_i = N - 1
  for i in range(N):
    if i > 0:
      n = int(round(p["L"] / dts[i]))
      cmd_d = cmd[max(0, i - n)]
      a_lag += (dts[i] / p["tau"]) * (cmd_d - a_lag)
    c = p["C_MAX"] if v >= p["V_HI"] else p["C_LOW"] + (p["C_MAX"] - p["C_LOW"]) * (v / p["V_HI"])
    a_act = a_lag + (c if (a_lag < 0.3 and v > 0.01) else 0.0)
    if v <= 0.0:
      a_act = 0.0
    vs[i] = v
    aas[i] = a_act
    v = max(0.0, v + a_act * dts[i])
    if v < 0.01:
      stop_i = i
      break
  return vs[:stop_i + 1], aas[:stop_i + 1], stop_i


def fidelity_stop(stop, prep, p):
  vs, aas, stop_i = _simulate_scalar(prep, p)
  t = prep["t"]
  imu = prep["imu"]
  k = stop_i
  out = {"route": stop["route"], "minute": stop["minute"]}
  rms = float(np.sqrt(np.mean((aas - imu[:k + 1]) ** 2)))
  w = max(0, np.searchsorted(t[:k + 1], t[k] - 0.6))
  sim_decel = float(np.min(aas[w:k + 1]))
  tf = prep["t_full"]
  imuf = prep["imu_full"]
  mask = (tf >= -0.6) & (tf <= 0.0)
  imu_decel = float(np.min(imuf[mask])) if mask.any() else float(np.min(imuf))
  stop_time = float(t[k])
  sim_dist = _trapz(vs, t[:k + 1])
  log_dist = _trapz(np.interp(t[:k + 1], tf, prep["v_full"]), t[:k + 1])
  out.update({
    "accel_rms": round(rms, 4),
    "stop_decel_sim": round(sim_decel, 4),
    "stop_decel_imu": round(imu_decel, 4),
    "stop_decel_err": round(abs(sim_decel - imu_decel), 4),
    "stop_time_sim_s": round(stop_time, 3),
    "stop_time_err_s": round(abs(stop_time), 3),
    "dist_sim_m": round(sim_dist, 3),
    "dist_log_m": round(log_dist, 3),
    "dist_err_m": round(abs(sim_dist - log_dist), 3),
  })
  out["pass"] = bool(rms <= T_RMS and out["stop_decel_err"] <= T_STOP_DECEL
                     and out["stop_time_err_s"] <= T_STOP_TIME and out["dist_err_m"] <= T_DIST)
  return out


def fidelity_all(stops, p):
  preps = [_prepare_stop(s) for s in stops]
  rows = [fidelity_stop(stop, prep, p) for stop, prep in zip(stops, preps, strict=True)]
  return {"stops": rows,
          "targets": {"accel_rms": T_RMS, "stop_decel_err": T_STOP_DECEL,
                      "stop_time_err_s": T_STOP_TIME, "dist_err_m": T_DIST},
          "all_pass": all(r["pass"] for r in rows)}


# --- low-speed (regime-split) open-loop fit -------------------------------

def _prepare_ls_stop(stop):
  """Low-speed window: v_ego < 4.0, after (last gas + 0.2 s), up to t_rel = 0.

  Keeps the full fixture rows so the NEW plant's dead-time buffer can be filled
  from the logged command before the window start.
  """
  rows = stop["rows"]
  t = rows[:, 0]
  v = rows[:, 1]
  cond = (v < LS_V_MAX) & (t <= 0.0)
  if stop["gas_cut_t"] is not None:
    cond &= (t >= stop["gas_cut_t"] + LS_GAS_PAD_S)
  idx = np.flatnonzero(cond)
  i0, i1 = int(idx[0]), int(idx[-1])
  return {"t_full": t, "v_full": v, "imu_full": rows[:, 3], "cmd_full": rows[:, 4],
          "i0": i0, "i1": i1,
          "t": t[i0:i1 + 1].copy(), "v0": float(v[i0]),
          "imu": rows[i0:i1 + 1, 3].copy()}


def _ls_lag(prep, L, tau):
  """NEW-plant lag over the window from the logged command; lag init = imu[0]."""
  t = prep["t_full"]
  cmd = prep["cmd_full"]
  i0, i1 = prep["i0"], prep["i1"]
  a_lag = np.empty(i1 + 1 - i0)
  a_lag[0] = prep["imu"][0]
  for i in range(i0 + 1, i1 + 1):
    dt = t[i] - t[i - 1]
    n = int(round(L / dt)) if dt > 0.0 else 0
    cmd_d = cmd[max(0, i - n)]
    a_lag[i - i0] = a_lag[i - i0 - 1] + (dt / tau) * (cmd_d - a_lag[i - i0 - 1])
  return a_lag


def _ls_vec_rms(prep, L, tau, CM, VH, CL):
  """Window RMS over the creep grid; simulation starts at the window v0."""
  a_lag = _ls_lag(prep, L, tau)
  dts = np.diff(prep["t"], prepend=0.0)
  dts = np.maximum(dts, 1e-6)
  imu = prep["imu"]
  N = a_lag.size
  K = CM.size
  v = np.full(K, prep["v0"])
  err2 = np.zeros(K)
  for i in range(N):
    c = np.where(v >= VH, CM, CL + (CM - CL) * (v / VH))
    gate = (a_lag[i] < 0.3) & (v > 0.01)
    a_act = a_lag[i] + np.where(gate, c, 0.0)
    a_act = np.where(v <= 0.0, 0.0, a_act)
    err2 += (a_act - imu[i]) ** 2
    v = np.maximum(0.0, v + a_act * dts[i])
  return np.sqrt(err2 / N)


def fit_low_speed_plant(stops, L_list, tau_list, cmax_list, vhi_list, clow_list):
  """Grid-search ONE low-speed parameter set over all stops; sorted (loss, params)."""
  preps = [_prepare_ls_stop(s) for s in stops]
  CM, VH, CL = _mesh(cmax_list, vhi_list, clow_list)
  K = CM.size
  results = []
  for L in L_list:
    for tau in tau_list:
      per_stop = np.empty((len(preps), K))
      for si, prep in enumerate(preps):
        per_stop[si] = _ls_vec_rms(prep, L, tau, CM, VH, CL)
      avg = per_stop.mean(axis=0)
      for k in range(K):
        if np.isfinite(avg[k]):
          results.append((float(avg[k]), {"L": L, "tau": tau, "C_MAX": float(CM[k]),
                                          "V_HI": float(VH[k]), "C_LOW": float(CL[k])}))
  results.sort(key=lambda item: item[0])
  return results


def _ls_simulate_scalar(prep, p):
  """Full-window scalar sim (holds at v=0, never breaks); returns arrays + stop index."""
  a_lag = _ls_lag(prep, p["L"], p["tau"])
  t = prep["t"]
  dts = np.diff(t, prepend=0.0)
  dts = np.maximum(dts, 1e-6)
  v = prep["v0"]
  vs = np.empty(a_lag.size)
  aas = np.empty(a_lag.size)
  stop_i = a_lag.size - 1
  seen_stop = False
  for i in range(a_lag.size):
    c = p["C_MAX"] if v >= p["V_HI"] else p["C_LOW"] + (p["C_MAX"] - p["C_LOW"]) * (v / p["V_HI"])
    a_act = a_lag[i] + (c if (a_lag[i] < 0.3 and v > 0.01) else 0.0)
    if v <= 0.0:
      a_act = 0.0
    vs[i] = v
    aas[i] = a_act
    v = max(0.0, v + a_act * dts[i])
    if not seen_stop and v < 0.01:
      stop_i = i
      seen_stop = True
  return vs, aas, stop_i


def _closest_index(t, target):
  return int(np.argmin(np.abs(np.asarray(t) - target)))


def fidelity_low_speed(stop, prep, p):
  vs, aas, stop_i = _ls_simulate_scalar(prep, p)
  t = prep["t"]
  imu = prep["imu"]
  out = {"route": stop["route"], "minute": stop["minute"],
         "window_s": [round(float(t[0]), 3), round(float(t[-1]), 3)]}
  rms = float(np.sqrt(np.mean((aas - imu[:vs.size]) ** 2)))
  decel_at = {}
  for ts in DECEL_INFO_S:
    j = _closest_index(t, ts)
    if j < aas.size:
      decel_at[ts] = {"sim": round(float(aas[j]), 4), "imu": round(float(imu[j]), 4),
                      "err": round(abs(float(aas[j]) - float(imu[j])), 4)}
    else:
      decel_at[ts] = {"sim": None, "imu": None, "err": None}
  stop_time = float(t[min(stop_i, t.size - 1)])
  sim_dist = _trapz(vs, t[:vs.size])
  log_dist = _trapz(np.interp(t[:vs.size], prep["t_full"], prep["v_full"]), t[:vs.size])
  out.update({
    "accel_rms": round(rms, 4),
    "decel_err_at_minus0.3": decel_at[DECEL_AT_S]["err"],
    "decel_info": decel_at,
    "stop_time_sim_s": round(stop_time, 3),
    "stop_time_err_s": round(abs(stop_time), 3),
    "dist_sim_m": round(sim_dist, 3),
    "dist_log_m": round(log_dist, 3),
    "dist_err_m": round(abs(sim_dist - log_dist), 3),
  })
  gate = decel_at[DECEL_AT_S]["err"]
  out["pass"] = bool(rms <= T_RMS and gate is not None and gate <= T_STOP_DECEL
                     and out["stop_time_err_s"] <= T_STOP_TIME and out["dist_err_m"] <= T_DIST)
  return out


def fidelity_low_speed_all(stops, p):
  preps = [_prepare_ls_stop(s) for s in stops]
  rows = [fidelity_low_speed(stop, prep, p) for stop, prep in zip(stops, preps, strict=True)]
  return {"stops": rows,
          "targets": {"accel_rms": T_RMS, "decel_err_at_minus0.3": T_STOP_DECEL,
                      "stop_time_err_s": T_STOP_TIME, "dist_err_m": T_DIST},
          "all_pass": all(r["pass"] for r in rows)}


# --- real_sr (SCC stop regime) fit to wheel speed / distance ---------------

def _prepare_sr_stop(stop):
  """real_sr window t0 = max(-4.0, last gas + 0.2) .. t_rel +0.6 (brief 1c.2)."""
  rows = stop["rows"]
  t = rows[:, 0]
  t0 = -4.0
  if stop["gas_cut_t"] is not None:
    t0 = max(t0, stop["gas_cut_t"] + LS_GAS_PAD_S)
  w = (t >= t0) & (t <= SR_WIN_END)
  return {"t": t[w].copy(), "cmd": rows[w, 4].copy(), "st": rows[w, 5].copy(),
          "v": rows[w, 1].copy(), "aw": rows[w, 2].copy(),
          "v0": float(rows[w, 1][0]), "aw0": float(rows[w, 2][0])}


def _sr_vec_metrics(prep, L, tau, scc_a, CM, VH, CL):
  """Vectorized-over-creep metrics for the real_sr NEW plant.

  Returns (vRMS, dist_err, stop_time_err, awRMS), each length K. The lag state
  is scalar (independent of creep) so it is computed once.
  """
  t = prep["t"]; cmd = prep["cmd"]; st = prep["st"]; v_real = prep["v"]; aw = prep["aw"]
  N = t.size; K = CM.size
  cd = np.interp(t - L, t, cmd)
  sd = np.interp(t - L, t, st)
  a_lag = np.empty(N)
  a_lag[0] = prep["aw0"]
  for i in range(1, N):
    dt = t[i] - t[i - 1]
    target = scc_a if sd[i] > (SR_STOPPING - 0.5) else cd[i]
    a_lag[i] = a_lag[i - 1] + (target - a_lag[i - 1]) * min(1.0, dt / tau) if tau > 0 else target
  v = np.full(K, prep["v0"]); x = np.zeros(K)
  vs = np.empty((N, K)); ys = np.empty((N, K))
  for i in range(N):
    dt = t[i] - t[i - 1] if i else 0.0
    c = CL + (CM - CL) * np.minimum(1.0, v / VH)
    a = a_lag[i] + np.where((a_lag[i] < 0.3) & (v > 0.01), c, 0.0)
    a = np.where((v <= 0.0) & (a < 0.0), 0.0, a)
    v = np.maximum(0.0, v + a * dt)
    x = x + v * dt
    vs[i] = v; ys[i] = a
  xr = np.concatenate([[0], np.cumsum(v_real[1:] * np.diff(t))])
  dist_err = x - xr[-1]
  mask = vs < 0.05
  any_stop = mask.any(axis=0)
  ts = np.where(any_stop, t[np.argmax(mask, axis=0)], 9.0)
  tr = t[np.argmax(v_real < 0.05)]
  stopt_err = ts - tr
  vRMS = np.sqrt(np.mean((vs - v_real[:, None]) ** 2, axis=0))
  m = t < -0.2
  awRMS = np.sqrt(np.mean((ys[m] - aw[m][:, None]) ** 2, axis=0))
  return vRMS, dist_err, stopt_err, awRMS


def _sr_scalar_metrics(prep, p):
  """Scalar real_sr metrics for one stop (used for fidelity / traces / xcheck)."""
  CM = np.array([p["C_MAX"]]); VH = np.array([p["V_HI"]]); CL = np.array([p["C_LOW"]])
  vr, de, te, ar = _sr_vec_metrics(prep, p["L"], p["tau"], p["SCC_A"], CM, VH, CL)
  return float(vr[0]), float(de[0]), float(te[0]), float(ar[0])


def fit_sr_plant(stops, L_list, tau_list, scc_list, cmax_list, vhi_list, clow_list):
  """Grid-search ONE real_sr set; loss = mean|dist_err| + mean vRMS. Sorted."""
  preps = [_prepare_sr_stop(s) for s in stops]
  CM, VH, CL = _mesh(cmax_list, vhi_list, clow_list)
  K = CM.size
  results = []
  for L in L_list:
    for tau in tau_list:
      for scc_a in scc_list:
        per_vr = np.empty((len(preps), K))
        per_de = np.empty((len(preps), K))
        for si, prep in enumerate(preps):
          vr, de, _te, _ar = _sr_vec_metrics(prep, L, tau, scc_a, CM, VH, CL)
          per_vr[si] = vr
          per_de[si] = de
        loss = np.mean(np.abs(per_de), axis=0) + np.mean(per_vr, axis=0)
        for k in range(K):
          results.append((float(loss[k]), {"L": L, "tau": tau, "SCC_A": scc_a,
                                           "C_MAX": float(CM[k]), "V_HI": float(VH[k]),
                                           "C_LOW": float(CL[k])}))
  results.sort(key=lambda item: item[0])
  return results


def fidelity_sr_stop(stop, prep, p):
  vr, de, te, ar = _sr_scalar_metrics(prep, p)
  out = {"route": stop["route"], "minute": stop["minute"],
         "window_s": [round(float(prep["t"][0]), 3), round(float(prep["t"][-1]), 3)],
         "vRMS": round(vr, 4), "dist_err_m": round(de, 4), "dist_sign": "+" if de >= 0 else "-",
         "stop_time_err_s": round(te, 4), "aWheel_rms": round(ar, 4)}
  out["pass"] = bool(abs(de) <= SR_T_DIST and abs(te) <= SR_T_STOP_TIME
                     and vr <= SR_T_VRMS and ar <= SR_T_AWRMS)
  return out


def fidelity_sr_all(stops, p):
  preps = [_prepare_sr_stop(s) for s in stops]
  rows = [fidelity_sr_stop(stop, prep, p) for stop, prep in zip(stops, preps, strict=True)]
  return {"stops": rows,
          "targets": {"dist_err_m": SR_T_DIST, "stop_time_err_s": SR_T_STOP_TIME,
                      "vRMS": SR_T_VRMS, "aWheel_rms": SR_T_AWRMS},
          "pass_count": sum(r["pass"] for r in rows),
          "all_pass": all(r["pass"] for r in rows)}


# --- real_sg (real_sr + road-grade term) fit to wheel speed / distance ------

def _stop_key(stop):
  """Fixture key for the HOLD table: route[5:8] + '@' + minute."""
  return f"{stop['route'][5:8]}@{stop['minute']:g}"


def _sr_grade(stop, M0, K):
  """Fit-time grade accel a_grade = -K*(HOLD[stop] - M0) (brief 1d.1)."""
  return -K * (SR_HOLD[_stop_key(stop)] - M0)


def _sg_scalar_metrics(prep, p, a_grade):
  """real_sg scalar metrics: real_sr metrics with a_grade added to sim accel."""
  L = p["L"]; tau = p["tau"]; scc_a = p["SCC_A"]
  C_MAX = p["C_MAX"]; V_HI = p["V_HI"]; C_LOW = p["C_LOW"]
  t = prep["t"]; cmd = prep["cmd"]; st = prep["st"]; v_real = prep["v"]
  cd = np.interp(t - L, t, cmd); sd = np.interp(t - L, t, st)
  v = prep["v0"]; a_lag = prep["aw0"]; x = 0.0
  vs = np.empty(t.size); ys = np.empty(t.size); stop_i = t.size - 1; seen = False
  for i in range(t.size):
    dt = t[i] - t[i - 1] if i else 0.0
    target = scc_a if int(round(sd[i])) == SR_STOPPING else cd[i]
    a_lag = a_lag + (target - a_lag) * min(1.0, dt / tau) if tau > 0 else target
    c = C_LOW + (C_MAX - C_LOW) * min(1.0, v / V_HI)
    a = a_lag + (c if (a_lag < 0.3 and v > 0.01) else 0.0) + a_grade
    if v <= 0.0 and a < 0:
      a = 0.0
    vs[i] = v; ys[i] = a
    v = max(0.0, v + a * dt); x += v * dt
    if not seen and v < 0.05:
      stop_i = i; seen = True
  xr = np.concatenate([[0], np.cumsum(v_real[1:] * np.diff(t))])
  dist_err = x - xr[-1]
  ts = t[stop_i] if seen else 9.0
  tr = t[np.argmax(v_real < 0.05)]
  stopt_err = ts - tr
  vRMS = float(np.sqrt(np.mean((vs - v_real) ** 2)))
  m = t < -0.2
  awRMS = float(np.sqrt(np.mean((ys[m] - prep["aw"][m]) ** 2)))
  return vRMS, float(dist_err), float(stopt_err), awRMS


def _sg_vec_metrics(prep, L, tau, scc_a, a_grade, CM, VH, CL):
  """Vectorized-over-creep real_sg metrics (grade is a per-stop scalar)."""
  t = prep["t"]; cmd = prep["cmd"]; st = prep["st"]; v_real = prep["v"]; aw = prep["aw"]
  N = t.size; K = CM.size
  cd = np.interp(t - L, t, cmd); sd = np.interp(t - L, t, st)
  a_lag = np.empty(N); a_lag[0] = prep["aw0"]
  for i in range(1, N):
    dt = t[i] - t[i - 1]
    target = scc_a if sd[i] > (SR_STOPPING - 0.5) else cd[i]
    a_lag[i] = a_lag[i - 1] + (target - a_lag[i - 1]) * min(1.0, dt / tau) if tau > 0 else target
  v = np.full(K, prep["v0"]); x = np.zeros(K)
  vs = np.empty((N, K)); ys = np.empty((N, K))
  for i in range(N):
    dt = t[i] - t[i - 1] if i else 0.0
    c = CL + (CM - CL) * np.minimum(1.0, v / VH)
    a = a_lag[i] + np.where((a_lag[i] < 0.3) & (v > 0.01), c, 0.0) + a_grade
    a = np.where((v <= 0.0) & (a < 0.0), 0.0, a)
    v = np.maximum(0.0, v + a * dt)
    x = x + v * dt
    vs[i] = v; ys[i] = a
  xr = np.concatenate([[0], np.cumsum(v_real[1:] * np.diff(t))])
  dist_err = x - xr[-1]
  mask = vs < 0.05
  any_stop = mask.any(axis=0)
  ts = np.where(any_stop, t[np.argmax(mask, axis=0)], 9.0)
  tr = t[np.argmax(v_real < 0.05)]
  stopt_err = ts - tr
  vRMS = np.sqrt(np.mean((vs - v_real[:, None]) ** 2, axis=0))
  m = t < -0.2
  awRMS = np.sqrt(np.mean((ys[m] - aw[m][:, None]) ** 2, axis=0))
  return vRMS, dist_err, stopt_err, awRMS


def fit_sg_plant(stops, M0_list, K_list, L_list, tau_list, scc_list,
                 cmax_list, vhi_list, clow_list):
  """Grid-search ONE real_sg set; loss = mean|dist_err| + mean vRMS. Sorted."""
  preps = [_prepare_sr_stop(s) for s in stops]
  CM, VH, CL = _mesh(cmax_list, vhi_list, clow_list)
  K = CM.size
  results = []
  for M0 in M0_list:
    for Kg in K_list:
      grades = [_sr_grade(s, M0, Kg) for s in stops]
      for L in L_list:
        for tau in tau_list:
          for scc_a in scc_list:
            per_vr = np.empty((len(preps), K))
            per_de = np.empty((len(preps), K))
            for si, prep in enumerate(preps):
              vr, de, _te, _ar = _sg_vec_metrics(prep, L, tau, scc_a, grades[si], CM, VH, CL)
              per_vr[si] = vr
              per_de[si] = de
            loss = np.mean(np.abs(per_de), axis=0) + np.mean(per_vr, axis=0)
            for k in range(K):
              results.append((float(loss[k]), {"M0": M0, "K": Kg, "L": L, "tau": tau,
                                               "SCC_A": scc_a, "C_MAX": float(CM[k]),
                                               "V_HI": float(VH[k]), "C_LOW": float(CL[k])}))
  results.sort(key=lambda item: item[0])
  return results


def fidelity_sg_stop(stop, prep, p):
  a_grade = _sr_grade(stop, p["M0"], p["K"])
  vr, de, te, ar = _sg_scalar_metrics(prep, p, a_grade)
  out = {"route": stop["route"], "minute": stop["minute"],
         "window_s": [round(float(prep["t"][0]), 3), round(float(prep["t"][-1]), 3)],
         "a_grade": round(a_grade, 4),
         "vRMS": round(vr, 4), "dist_err_m": round(de, 4), "dist_sign": "+" if de >= 0 else "-",
         "stop_time_err_s": round(te, 4), "aWheel_rms": round(ar, 4)}
  out["pass"] = bool(abs(de) <= SR_T_DIST and abs(te) <= SR_T_STOP_TIME
                     and vr <= SR_T_VRMS and ar <= SR_T_AWRMS)
  return out


def fidelity_sg_all(stops, p):
  preps = [_prepare_sr_stop(s) for s in stops]
  rows = [fidelity_sg_stop(stop, prep, p) for stop, prep in zip(stops, preps, strict=True)]
  return {"stops": rows,
          "targets": {"dist_err_m": SR_T_DIST, "stop_time_err_s": SR_T_STOP_TIME,
                      "vRMS": SR_T_VRMS, "aWheel_rms": SR_T_AWRMS},
          "pass_count": sum(r["pass"] for r in rows),
          "all_pass": all(r["pass"] for r in rows)}


def sr_trace(stop, prep, p, t_lo=-3.0, t_hi=0.0, step=0.2):
  """0.2 s trace over [t_lo, t_hi]: t, v_real, v_sim, cmd, long_state, aWheel, a_sim."""
  L = p["L"]; tau = p["tau"]; scc_a = p["SCC_A"]
  C_MAX = p["C_MAX"]; V_HI = p["V_HI"]; C_LOW = p["C_LOW"]
  t = prep["t"]; cmd = prep["cmd"]; st = prep["st"]
  cd = np.interp(t - L, t, cmd); sd = np.interp(t - L, t, st)
  v = prep["v0"]; a_lag = prep["aw0"]; rows_out = []
  sim_v = np.empty(t.size); sim_a = np.empty(t.size)
  for i in range(t.size):
    dt = t[i] - t[i - 1] if i else 0.0
    target = scc_a if int(round(sd[i])) == SR_STOPPING else cd[i]
    a_lag = a_lag + (target - a_lag) * min(1.0, dt / tau) if tau > 0 else target
    c = C_LOW + (C_MAX - C_LOW) * min(1.0, v / V_HI)
    a = a_lag + (c if (a_lag < 0.3 and v > 0.01) else 0.0)
    if v <= 0.0 and a < 0:
      a = 0.0
    sim_v[i] = v; sim_a[i] = a
    v = max(0.0, v + a * dt)
  for tt in np.arange(t_lo, t_hi + 1e-9, step):
    j = _closest_index(t, tt)
    rows_out.append({"t": round(float(t[j]), 2), "v_real": round(float(prep["v"][j]), 3),
                     "v_sim": round(float(sim_v[j]), 3), "cmd": round(float(cmd[j]), 3),
                     "long_state": int(st[j]), "aWheel_real": round(float(prep["aw"][j]), 3),
                     "a_sim": round(float(sim_a[j]), 3)})
  return {"route": stop["route"], "minute": stop["minute"], "trace": rows_out}


# --- closed-loop baseline with the real plant ------------------------------

def _run_loop_real(loop):
  """Shared body of ButterStopLoop._run_loop / FarLeadLoop.run with the real plant.

  `loop` is a ButterStopLoop- or FarLeadLoop-shaped instance whose plant / scc
  flags are off and which carries `real_p` and `_real_state`. Returns the same
  array / dict shapes those loops return.
  """
  maneuver = getattr(loop, "maneuver", None)
  fixture_mode = maneuver is None
  dt = DT_CTRL
  if fixture_mode:
    profile = loop.profile
    max_frames = profile.n_steps * PLANNER_PERIOD
    maneuver_enabled = True
  else:
    frames = int(round(maneuver.duration / dt))
    max_frames = frames + 1
    maneuver_enabled = bool(maneuver.enabled)

  fw_rows = []
  log_t, log_v, log_accel, log_x = [], [], [], []
  log_gap, log_active, log_vlead, log_atarget = [], [], [], []
  log_a_actual, log_should_stop, log_lc_state = [], [], []
  log_desired_distance = []
  log_stock, log_fcw_t = [], []
  log_floor_raw, log_floor_applied, log_bound = [], [], []
  fcw_steps = 0
  fcw = 0.0
  msgs = None
  plan_view = None
  if fixture_mode:
    v_cruise_kph = profile.v_cruise_kph(0.0)
  standstill_frames = 0
  stopped = False
  for frame in range(max_frames):
    t = frame * dt
    if fixture_mode:
      v_lead = None
      standstill = loop.ego_v < 0.01
      v_cruise_kph = profile.v_cruise_kph(t)
      v_cruise_ms = v_cruise_kph / 3.6
      cs_cruise = v_cruise_kph  # FarLeadLoop._cs_view takes km/h
    else:
      v_lead = float(np.interp(t, maneuver.breakpoints, maneuver.speed_lead_values))
      v_cruise_ms = float(np.interp(t, maneuver.breakpoints, maneuver.cruise_values))
      cs_cruise = v_cruise_ms  # ButterStopLoop._cs_view takes m/s
      pitch = float(np.interp(t, maneuver.breakpoints, maneuver.pitch_values))
      prob_lead = float(np.interp(t, maneuver.breakpoints, maneuver.prob_lead_values))
      prob_throttle = float(np.interp(t, maneuver.breakpoints, maneuver.prob_throttle_values))
      standstill = loop.ego_v < 0.01

    if frame % PLANNER_PERIOD == 0:
      if fixture_mode:
        specs = profile.sample(t, loop.ego_x)
        radar_state = loop._radar_message(specs)
        msgs = {"radarState": radar_state}
        msgs.update(loop._sm_messages(v_cruise_kph, standstill))
        loop.planner.update(_FakeSM(msgs, frame), loop.carrot)
        plan_view = loop._plan_view()
        if loop.planner.mpc.crash_cnt > 0:
          fcw_steps += 1
      else:
        a_lead = (v_lead - loop.v_lead_prev) / (PLANNER_PERIOD * dt)
        loop.v_lead_prev = v_lead
        if maneuver.lead_relevancy:
          status = True if maneuver.only_radar else prob_lead > 0.5
          d_rel = max(0.0, loop.lead_x - loop.ego_x)
          if loop.radar_noise_std > 0.0:
            d_rel += float(loop._radar_rng.normal(0.0, loop.radar_noise_std))
          lead = {"d_rel": d_rel, "v_rel": v_lead - loop.ego_v, "v_lead": v_lead,
                  "a_lead": a_lead if not np.isnan(a_lead) else 0.0,
                  "prob": prob_lead if status else 0.0}
        else:
          lead = None
        radar_state = loop._radar_message(lead, lead_two=bool(getattr(maneuver, "only_lead2", False)))
        msgs = {"radarState": radar_state}
        msgs.update(loop._sm_messages(v_cruise_ms, pitch, prob_throttle, standstill))
        loop.planner.update(_FakeSM(msgs, frame), loop.carrot)
        plan_view = loop._plan_view()
        fcw = 1.0 if loop.planner.mpc.crash_cnt > 0 else 0.0

    t_since_plan = (frame % PLANNER_PERIOD) * dt
    limits = CarInterface.get_pid_accel_limits(loop.cp, loop.ego_v, v_cruise_ms)
    radar_view = types.SimpleNamespace(leadOne=msgs["radarState"].leadOne,
                                       leadTwo=msgs["radarState"].leadTwo)
    accel, _, _ = loop.long_control.update(
      maneuver_enabled, loop._cs_view(cs_cruise, standstill), plan_view,
      limits, t_since_plan, radar_view)
    accel = float(accel)
    lc_state_now = loop.long_control.long_control_state
    fixed_grade = getattr(loop, "fixed_a_grade", None)
    plant_diag = {}
    if loop.plant_mode == "real_ls":
      lc_stopping = lc_state_now == LongCtrlState.stopping
      applied = real_ls_step(loop._real_state, accel, loop.ego_v, dt, loop.real_p,
                             scc_stop=False, lc_stopping=lc_stopping)
    elif loop.plant_mode == "real_sr":
      applied = real_sr_step(loop._real_state, accel, loop.ego_v, dt, loop.real_p,
                             int(lc_state_now))
    elif loop.plant_mode == "real_sg":
      a_grade = fixed_grade if fixed_grade is not None else -loop.real_p["K"] * G_ACCEL * math.sin(loop.grade_rad)
      applied = real_sg_step(loop._real_state, accel, loop.ego_v, dt, loop.real_p,
                             int(lc_state_now), a_grade, diag=plant_diag)
    else:
      applied, _ = plant_real_step(loop._real_state, accel, loop.ego_v, dt, loop.real_p)

    stock_now = getattr(loop.long_control, "butter_stock_last", None)

    if getattr(loop, "_trace", None) is not None:
      # Gap to the lead where the profile exposes one (replica/a/b2); else NaN.
      gap_now = float("nan")
      v_lead_now = float("nan")
      try:
        if fixture_mode:
          for spec in profile.sample(t, loop.ego_x):
            if spec is not None and spec[3]:
              g = spec[0] - loop.ego_x
              if not np.isfinite(gap_now) or g < gap_now:
                gap_now, v_lead_now = g, spec[1]
        elif getattr(loop, "maneuver", None) is not None and maneuver.lead_relevancy:
          gap_now = max(0.0, loop.lead_x - loop.ego_x)
          v_lead_now = float(v_lead) if v_lead is not None else float("nan")
      except (NameError, TypeError):
        gap_now = float("nan")
      lc = loop.long_control
      # CP-H0 diagnosis columns.
      import openpilot.selfdrive.controls.lib.butter_stop as _bs
      try:
        plan_speeds_now = getattr(plan_view, "speeds", None)
        plan_stop_d = _bs.plan_stop_distance(CONTROL_N_T_IDX, plan_speeds_now)
      except Exception:
        plan_stop_d = None
      lead_avail = None
      if fixture_mode is False and getattr(loop, "maneuver", None) is not None and maneuver.lead_relevancy:
        lead_avail = max(0.0, loop.lead_x - loop.ego_x) - max(
          _bs.MIN_FINAL_GAP_ABS, (lc.stop_distance_orig or 0.0) - _bs.LEAD_SLACK)
      loop._trace.append({
        "t": t, "v": loop.ego_v, "stock_out": stock_now,
        "floor_raw": lc.butter_floor_raw_last, "floor_applied": lc.butter_floor_last,
        "out": accel, "long_state": int(lc_state_now), "bound": lc.butter_bound_last,
        "a_end": lc.butter_a_end_last,
        "a_ego": float(getattr(loop.long_control, "last_output_accel", float("nan"))),
        "plan_stop_d": plan_stop_d, "lead_avail": lead_avail,
        "x": float(loop.ego_x),
        "creep_latched": bool(getattr(lc, "_butter_creep_latched", False)),
        "standstill_confirmed": bool(getattr(lc, "_butter_standstill_confirmed", False)),
        "should_stop": bool(getattr(plan_view, "shouldStop", False)),
        "v_lead": v_lead_now,
        "v_cruise_ms": (float(v_cruise_ms) if not fixture_mode else float(v_cruise_kph) / 3.6),
        "action_t": float(getattr(plan_view, "actionTime", float("nan"))),
        "stop_delay": float(getattr(plan_view, "stopDelay", float("nan"))),
        "stop_delay_w": float(getattr(plan_view, "stopDelayW", 0.0)),
        "d_rel": float(getattr(radar_view.leadOne, "dRel", float("nan"))),
        "fcw": bool(getattr(plan_view, "fcw", False)) or loop.planner.mpc.crash_cnt > 0,
        "plant_target": plant_diag.get("target"), "plant_a": float(applied),
        "plant_grade": plant_diag.get("grade"), "plant_w_new": plant_diag.get("w_new"),
        "gap": gap_now,
      })

    if fixture_mode:
      gap = np.nan
      v_lead_now = np.nan
      active = False
      for spec in profile.sample(t, loop.ego_x):
        if spec is not None and spec[3]:
          g = spec[0] - loop.ego_x
          if not active or g < gap:
            gap, v_lead_now, active = g, spec[1], True
      log_t.append(t)
      log_v.append(loop.ego_v)
      log_accel.append(accel)
      log_x.append(loop.ego_x)
      log_gap.append(gap)
      log_active.append(active)
      log_vlead.append(v_lead_now)
      log_atarget.append(float(plan_view.aTarget))
      log_a_actual.append(applied)
      log_should_stop.append(bool(plan_view.shouldStop))
      log_lc_state.append(int(loop.long_control.long_control_state))
      log_desired_distance.append(float(plan_view.desiredDistance))
      log_stock.append(float(stock_now) if stock_now is not None else float("nan"))
      lc_f = loop.long_control
      log_floor_raw.append(float(lc_f.butter_floor_raw_last) if lc_f.butter_floor_raw_last is not None else float("nan"))
      log_floor_applied.append(float(lc_f.butter_floor_last) if lc_f.butter_floor_last is not None else float("nan"))
      log_bound.append(lc_f.butter_bound_last if lc_f.butter_bound_last is not None else "")
      if loop.planner.mpc.crash_cnt > 0:
        log_fcw_t.append(t)
    else:
      floor_active = 1.0 if loop.long_control.butter_floor_last is not None else 0.0
      floor_val = loop.long_control.butter_floor_last
      floor_raw = loop.long_control.butter_floor_raw_last
      bound = loop.long_control.butter_bound_last
      bound_code = -1.0 if floor_active == 0.0 else (1.0 if bound == "lead" else 0.0)
      a_end = loop.long_control.butter_a_end_last
      fw_rows.append((t, loop.ego_v, accel, loop.ego_x, loop.lead_x, floor_active,
                      floor_val if floor_val is not None else 0.0, bound_code,
                      floor_raw if floor_raw is not None else 0.0,
                      a_end if a_end is not None else 0.0,
                      applied, float(loop.long_control.long_control_state), fcw,
                      float(stock_now) if stock_now is not None else float("nan"),
                      0.0))

    loop.ego_v = max(0.0, loop.ego_v + applied * dt)
    loop.ego_x = loop.ego_x + loop.ego_v * dt
    if fixture_mode:
      if loop.ego_v < 0.05:
        standstill_frames += 1
        if standstill_frames >= 300:
          stopped = True
          break
      else:
        standstill_frames = 0
    else:
      loop.lead_x = loop.lead_x + v_lead * dt

  if fixture_mode:
    return {
      "t": log_t, "v": log_v, "accel": log_accel, "x": log_x,
      "gap": log_gap, "active": log_active, "v_lead": log_vlead, "a_target": log_atarget,
      "a_actual": log_a_actual, "should_stop": log_should_stop, "lc_state": log_lc_state,
      "desired_distance": log_desired_distance, "stock": log_stock,
      "floor_raw": log_floor_raw, "floor_applied": log_floor_applied, "bound": log_bound,
      "fcw_times": log_fcw_t,
      "fcw_steps": fcw_steps, "stopped": bool(stopped or loop.ego_v < 0.05),
    }
  return np.array(fw_rows)


class _RealScenarioLoop(ButterStopLoop):
  def __init__(self, maneuver, butter, p, plant_mode="real", grade_rad=0.0):
    super().__init__(maneuver, butter=butter, plant=False)
    self.real_p = dict(p)
    self.plant_mode = plant_mode
    self.grade_rad = float(grade_rad)
    self.fixed_a_grade = None
    self._real_state = None

  def run(self):
    self._real_state = {"old": {}, "new": {"hist": []}}
    return _run_loop_real(self)


class _RealFixtureLoop(FarLeadLoop):
  def __init__(self, header, profile, p, plant_mode="real", grade_rad=0.0):
    super().__init__(header, profile, plant=False)
    self.real_p = dict(p)
    self.plant_mode = plant_mode
    self.grade_rad = float(grade_rad)
    self.fixed_a_grade = None
    self._real_state = None

  def run(self):
    self._real_state = {"old": {}, "new": {"hist": []}}
    return _run_loop_real(self)


class RealStopReplicaProfile:
  """Real-stop replica source (brief 2 scenarios).

  Replays the LOGGED lead trajectory in world coordinates while the ego is driven
  by the plant from the logged initial speed/accel. The logged lead is moving at
  the window start (a stationary lead at the start position would be overtaken
  immediately), so lead_x(t) = dRel(t) + (x_ego_log(t) - x_ego_log(t0)); the lead
  is reported stationary only where the log shows it stopped. The ego starts at
  x=0. This reproduces the real approach geometry in closed loop.
  """

  is_replica = True

  def __init__(self, stop, t_start=-8.0, lead_drel=None):
    self.stop = stop
    rows = stop["rows"]
    t = rows[:, 0]
    i0 = int(np.flatnonzero(t >= t_start)[0]) if (t >= t_start).any() else 0
    self.t0 = float(t[i0])
    sub = rows[i0:]
    self.dt = float(np.median(np.diff(t[i0:]))) if sub.shape[0] > 1 else 0.02
    self.dt = min(max(self.dt, 0.02), 0.05)
    self.n_steps = int((float(sub[-1, 0]) - self.t0) / self.dt) + 1
    self.duration = self.n_steps * self.dt
    self.t = np.arange(self.n_steps) * self.dt
    self.t_rel = self.t + self.t0
    self.initial_speed = float(sub[0, 1])
    self.initial_accel = float(sub[0, 2])
    # Ego odometry from the log (cumulative integral of v_ego), zeroed at t0.
    x_ego_full = np.concatenate([[0.0], np.cumsum(sub[1:, 1] * np.diff(sub[:, 0]))])
    x0 = x_ego_full[0]
    # Lead world position over the window (dRel is ego-relative).
    d_rel_log = sub[:, 7]
    lead_x_log = d_rel_log + (x_ego_full - x0)
    self.lead_status_log = sub[:, 6] > 0.5
    self.lead_v_log = sub[:, 8]
    # Resample onto the uniform replica grid.
    tt = sub[:, 0]
    self.lead_x = np.interp(self.t_rel, tt, lead_x_log)
    self.lead_v = np.interp(self.t_rel, tt, self.lead_v_log)
    self.lead_status = np.interp(self.t_rel, tt, self.lead_status_log.astype(float)) > 0.5
    self.lead_a = np.gradient(self.lead_v, self.dt) if self.t.size > 1 else np.zeros_like(self.lead_v)
    self.has_lead = bool(self.lead_status.any())
    # Log stop reference for the sanity check.
    stand = np.flatnonzero(rows[:, 1] < 0.05)
    self.log_stop_t_rel = float(rows[stand[0], 0]) if stand.size else None
    self.log_final_gap = float(rows[stand[0], 7]) if stand.size else None
    # Per-stop grade accel (HOLD table) in the real_sg fit form, using PLANT_REAL.
    self.grade = _sr_grade(stop, PLANT_REAL["M0"], PLANT_REAL["K"])

  def _index(self, t):
    return min(max(int(round(t / self.dt)), 0), self.n_steps - 1)

  def v_cruise_kph(self, _t):
    return float(max(30.0, self.initial_speed * 3.6 + 10.0))

  def sample(self, t, _s_ego):
    i = self._index(t)
    if not self.lead_status[i]:
      return [None, None]
    return [(float(self.lead_x[i]), float(self.lead_v[i]), float(self.lead_a[i]),
             True, True, 7, 1.0), None]

  def reset(self):
    pass


class _RealReplicaLoop(FarLeadLoop):
  def __init__(self, header, profile, p, plant_mode="real_sg", grade_rad=0.0):
    super().__init__(header, profile, plant=False)
    self.real_p = dict(p)
    self.plant_mode = plant_mode
    self.grade_rad = float(grade_rad)
    self.fixed_a_grade = getattr(profile, "grade", None)
    self._real_state = None

  def run(self):
    self._real_state = {"old": {}, "new": {"hist": []}}
    return _run_loop_real(self)


def _stop_row(name, group, butter, t, v, applied, gap, fcw_steps, start_delay, grade=0.0,
              fcw_times=None):
  t = np.asarray(t, dtype=float)
  v = np.asarray(v, dtype=float)
  applied = np.asarray(applied, dtype=float)
  gap = np.asarray(gap, dtype=float)
  finite = np.isfinite(gap)
  stand = np.flatnonzero(v < 0.01)
  stopped = bool(stand.size)
  fcw_first_t = fcw_last_t = None
  if fcw_times is not None and len(fcw_times):
    fcw_first_t = round(float(np.min(fcw_times)), 3)
    fcw_last_t = round(float(np.max(fcw_times)), 3)
  row = {"name": name, "group": group, "butter": int(butter), "grade": grade, "stopped": stopped,
         "stop_decel_at_stop": None, "stop_decel_at_minus0.3": None, "nod_swing": None,
         "tail_s": None, "final_gap_m": None,
         "min_gap_m": None if not finite.any() else round(float(np.nanmin(gap)), 3),
         "collision": bool(finite.any() and np.nanmin(gap) < COLLISION_GAP_M),
         "fcw_steps": int(fcw_steps), "fcw_first_t": fcw_first_t, "fcw_last_t": fcw_last_t,
         "start_delay_s": start_delay}
  if stopped:
    i = int(stand[0])
    step = float(t[1] - t[0]) if t.size > 1 else DT_CTRL
    w = (t >= t[i] - 0.6) & (t <= t[i])
    decel = float(np.min(applied[w])) if w.any() else None
    j = _closest_index(t, t[i] - 0.3)
    decel_m03 = float(applied[j]) if j < applied.size else None
    w2 = (t > t[i]) & (t <= t[i] + 0.3)
    swing = float(np.max(applied[w2]) - decel) if (w2.any() and decel is not None) else None
    tail_w = (t >= max(0.0, t[i] - 6.0)) & (t <= t[i])
    row["stop_decel_at_stop"] = None if decel is None else round(decel, 4)
    row["stop_decel_at_minus0.3"] = None if decel_m03 is None else round(decel_m03, 4)
    row["nod_swing"] = None if swing is None else round(swing, 4)
    row["tail_s"] = round(float(np.sum((v > 0.05) & (v < 0.5) & tail_w)) * step, 3)
    row["final_gap_m"] = None if not finite.any() else round(float(gap[i]), 3)
  return row


def _start_delay(t, v):
  v = np.asarray(v, dtype=float)
  moving = np.flatnonzero(v > 0.3)
  return None if not moving.size else round(float(np.asarray(t, dtype=float)[moving[0]] - 5.0), 3)


def _apply_baseline_params(params, loop):
  """Seed the brief's baseline params (VEgoStopping 0.10, StoppingAccel -0.5) and
  force LongControl to re-read them (it caches stopping_accel at construction)."""
  params.put_int("VEgoStopping", BASELINE_PARAMS["VEgoStopping"])
  params.put_int("StoppingAccel", BASELINE_PARAMS["StoppingAccel"])
  loop.long_control._refresh_stopping_accel()
  loop.long_control._refresh_butter_stop()


# Every "stopped lead" scenario (brief 1d.5): the a-group, the short-gap and
# noisy-radar s-cases, and the no-lead b2 proxy. Grade variants use these + b1.
_STOPPED_LEAD = ([("a", m) for m in _A_SCENARIOS]
                 + [("s1", _S1), ("s3", _S3), ("s3b", _S3B), ("s3c", _S3C),
                    ("b2", _B2_PROXY)])


def _run_one(loop, params):
  _apply_baseline_params(params, loop)
  return loop.run()


def _iter_rows(p, plant_mode="real", with_grade=False):
  """Yield baseline rows (one per scenario x butter); flat first, then grades."""
  params = Params()

  def scenario_rows(maneuver, group, grade, gtag, startdelay=False):
    grade_rad = math.asin(grade)
    suffix = f" [{gtag}]" if gtag else ""
    for butter in (1, 0):
      loop = _RealScenarioLoop(maneuver, butter, p, plant_mode=plant_mode,
                               grade_rad=grade_rad)
      logs = _run_one(loop, params)
      gap = logs[:, 4] - logs[:, 3]
      fcw_mask = logs[:, 12] > 0.5
      yield _stop_row(maneuver.title + suffix, group, butter, logs[:, 0], logs[:, 1],
                      logs[:, 10], gap, int(np.sum(fcw_mask)),
                      _start_delay(logs[:, 0], logs[:, 1]) if startdelay else None,
                      grade, logs[fcw_mask, 0])

  def b1_rows(grade, gtag):
    grade_rad = math.asin(grade)
    suffix = f" [{gtag}]" if gtag else ""
    fixture = load_fixture("b1")
    for butter in (1, 0):
      header = dict(fixture["params"])
      if butter == 0:
        header["ButterStop"] = 0
      loop = _RealFixtureLoop(header, FixtureProfile(fixture), p, plant_mode=plant_mode,
                              grade_rad=grade_rad)
      run = _run_one(loop, params)
      yield _stop_row("b1" + suffix, "b1", butter, run["t"], run["v"], run["a_actual"],
                      run["gap"], run["fcw_steps"], None, grade)

  yield from b1_rows(0.0, "")
  for group, maneuver in _CASES:
    yield from scenario_rows(maneuver, group, 0.0, "")
  for maneuver in _START_SCENARIOS:
    yield from scenario_rows(maneuver, "start", 0.0, "", startdelay=True)
  if with_grade:
    for grade, gtag in ((-0.03, "dn3"), (0.03, "up3")):
      yield from b1_rows(grade, gtag)
      for group, maneuver in _STOPPED_LEAD:
        yield from scenario_rows(maneuver, group, grade, gtag)


def run_baseline(p, plant_mode="real", with_grade=False):
  return list(_iter_rows(p, plant_mode=plant_mode, with_grade=with_grade))


# Only this FCW is accepted: s1's pre-existing one at t < 0.1 s (OLD regime).
_ACCEPTED_FCW_PREFIX = "s1 short-gap stopped lead 15 km/h gap 9 m"


def _gate_failure(row):
  """Return a reason string if `row` violates the 1d.4/1d.6 gate, else None."""
  if row["collision"]:
    return f"collision {row['name']} b{row['butter']} min_gap={row['min_gap_m']}"
  if row["fcw_steps"] > 0:
    accepted = (row["name"].startswith(_ACCEPTED_FCW_PREFIX)
                and row["fcw_last_t"] is not None and row["fcw_last_t"] < 0.1)
    if not accepted:
      return (f"fcw {row['name']} b{row['butter']} steps={row['fcw_steps']} "
              f"t=[{row['fcw_first_t']},{row['fcw_last_t']}]")
  return None


def gate_check(p, plant_mode="real", with_grade=True):
  """Evaluate all scenarios; return (failed, reason, failed_at, rows_until_stop)."""
  partial = []
  for row in _iter_rows(p, plant_mode=plant_mode, with_grade=with_grade):
    partial.append(row)
    reason = _gate_failure(row)
    if reason is not None:
      return True, reason, row["name"] + (f" [b{row['butter']}]"), partial
  return False, None, None, partial


# --- tests -----------------------------------------------------------------

def test_butter_nod_fixture_sha256():
  assert hashlib.sha256(_read_nod_raw()).hexdigest() == NOD_SHA256


def test_butter_nod_smoke():
  """real_sg smoke (brief 1d.7): reference set on 2 stops + 2 closed-loop scenarios
  including one -3% grade. Every field finite; min/max printed."""
  _d, stops = load_stops()
  subset = [stops[0], stops[4]]
  p = dict(SRG_REF)
  fid = fidelity_sg_all(subset, p)
  print(f"NOD-SMOKE ref={p}")
  print(f"NOD-SMOKE fidelity={json.dumps(fid['stops'], sort_keys=True)}")
  metrics = []
  for row in fid["stops"]:
    for key in ("vRMS", "dist_err_m", "stop_time_err_s", "aWheel_rms", "a_grade"):
      assert row[key] is not None and math.isfinite(row[key]), f"{key} not finite: {row[key]}"
      metrics.append(row[key])
  vec = _sg_scalar_metrics(_prepare_sr_stop(subset[0]), p, _sr_grade(subset[0], p["M0"], p["K"]))
  assert all(math.isfinite(x) for x in vec), f"vec metrics not finite: {vec}"
  print(f"NOD-SMOKE metrics min={min(metrics):.4f} max={max(metrics):.4f} n={len(metrics)}")
  assert max(metrics) - min(metrics) > 1e-6, "metrics constant"

  # 2 closed-loop scenarios: b1 flat and the a-group 20 km/h stopped lead at -3%.
  b1_rows = run_baseline(p, plant_mode="real_sg")
  grade_rows = [r for r in _iter_rows(p, plant_mode="real_sg", with_grade=True)
                if r["grade"] == -0.03 and r["name"].startswith("stopped lead 20 km/h gap 30 m")]
  sample = [r for r in b1_rows if r["name"] == "b1" and r["butter"] == 1] + grade_rows
  for r in sample:
    for key in ("stop_decel_at_stop", "stop_decel_at_minus0.3", "nod_swing", "tail_s",
                "final_gap_m", "min_gap_m"):
      assert r[key] is not None and math.isfinite(r[key]), f"{key} not finite: {r[key]}"
  print(f"NOD-SMOKE closed-loop rows={len(b1_rows)} sample={sample}")


def test_butter_nod_sr_crosscheck():
  """Our real_sr metrics at SR_REF must match vfit3.py within 0.02 m / 0.05 s (1c.3)."""
  _d, stops = load_stops()
  p = dict(SR_REF)
  # vfit3.py reference output for set (L 0.6, tau 0.8, SCC_A -0.4, C_MAX 0.1, V_HI 1.0, C_LOW 0.3)
  ref = {
    "00000443--e1be7fac84": {"minute": 8.82, "dist": -0.117066, "stopt": 0.140400},
    "00000445--7bd76114e3": [{"minute": 1.43, "dist": 0.056362, "stopt": 0.183000},
                             {"minute": 7.27, "dist": 0.047173, "stopt": -0.179100},
                             {"minute": 8.52, "dist": -0.584369, "stopt": -0.859800}],
    "00000447--8ec1dcb6ff": {"minute": 7.07, "dist": -0.589841, "stopt": -1.080000},
    "0000044c--4d25f7747b": {"minute": 6.21, "dist": -0.024030, "stopt": -0.181400},
  }
  used = {}
  for stop in stops:
    vr, de, te, ar = _sr_scalar_metrics(_prepare_sr_stop(stop), p)
    entry = ref.get(stop["route"])
    if entry is None:
      continue
    if isinstance(entry, list):
      idx = used.get(stop["route"], 0)
      entry = entry[idx]
      used[stop["route"]] = idx + 1
    assert entry["minute"] == stop["minute"], f"minute mismatch {entry} {stop['minute']}"
    assert abs(de - entry["dist"]) <= SR_CROSSCHECK_DIST, \
      f"{stop['route']}@{stop['minute']} dist {de:.4f} vs {entry['dist']:.4f}"
    assert abs(te - entry["stopt"]) <= SR_CROSSCHECK_TIME, \
      f"{stop['route']}@{stop['minute']} stopt {te:.4f} vs {entry['stopt']:.4f}"
  print("NOD-XCHECK OK (real_sr matches vfit3 within 0.02 m / 0.05 s)")


def test_butter_nod_sg_crosscheck():
  """Our real_sg metrics at SRG_REF must match vfit6.py within 0.02 m / 0.05 s (1d.3)."""
  _d, stops = load_stops()
  p = dict(SRG_REF)
  # vfit6.py output for (M0 -0.8, K 0.25, L 0.6, tau 0.5, SCC_A -0.5, C_MAX 0.2, V_HI 0.5, C_LOW 0.2)
  ref = [
    (8.82, -0.04, -0.42), (1.43, +0.08, -0.40), (7.27, +0.22, +0.08),
    (8.52, -0.09, -0.44), (7.07, -0.05, -0.34), (6.21, +0.08, -0.56),
  ]
  by_minute = {}
  for stop in stops:
    vr, de, te, ar = _sg_scalar_metrics(_prepare_sr_stop(stop), p,
                                        _sr_grade(stop, p["M0"], p["K"]))
    by_minute[stop["minute"]] = (de, te)
  for minute, dist, stopt in ref:
    de, te = by_minute[minute]
    assert abs(de - dist) <= SR_CROSSCHECK_DIST, \
      f"@{minute} dist {de:.4f} vs {dist:.4f}"
    assert abs(te - stopt) <= SR_CROSSCHECK_TIME, \
      f"@{minute} stopt {te:.4f} vs {stopt:.4f}"
  print("NOD-XCHECK-SG OK (real_sg matches vfit6 within 0.02 m / 0.05 s)")


def test_butter_nod_fit_baseline():
  if os.environ.get("BUTTER_NOD_FULL") != "1":
    pytest.skip("heavy real_sg fit/grid; set BUTTER_NOD_FULL=1 to run")
  out_path = os.environ.get("BUTTER_NOD_OUT", "/tmp/butter_nod.json")
  raw = _read_nod_raw()
  assert hashlib.sha256(raw).hexdigest() == NOD_SHA256

  _d, stops = load_stops()
  fits = fit_sg_plant(stops, SRG_M0, SRG_K, SRG_L, SRG_TAU, SRG_SCC, SRG_CMAX, SRG_VHI, SRG_CLOW)
  assert fits, "fit produced no finite parameter sets"
  top5 = [{"loss": round(loss, 6), "params": params} for loss, params in fits[:5]]
  best_loss, best_p = fits[0]
  edge = {axis: best_p[axis] in (lo, hi) for axis, (lo, hi) in (
    ("M0", (SRG_M0[0], SRG_M0[-1])), ("K", (SRG_K[0], SRG_K[-1])),
    ("L", (SRG_L[0], SRG_L[-1])), ("tau", (SRG_TAU[0], SRG_TAU[-1])),
    ("SCC_A", (SRG_SCC[0], SRG_SCC[-1])), ("C_MAX", (SRG_CMAX[0], SRG_CMAX[-1])),
    ("V_HI", (SRG_VHI[0], SRG_VHI[-1])), ("C_LOW", (SRG_CLOW[0], SRG_CLOW[-1])))}

  # 1d.4 selection: closed-loop gate the top-20 sets in loss order.
  chosen = None
  chosen_rank = None
  rejections = []
  gate_rows = None
  for rank, (loss, params) in enumerate(fits[:20], start=1):
    failed, reason, where, partial = gate_check(params, plant_mode="real_sg", with_grade=True)
    if not failed:
      chosen, chosen_rank = params, rank
      gate_rows = partial
      break
    rejections.append({"rank": rank, "loss": round(loss, 6), "params": params,
                       "reason": reason, "failed_at": where})
  assert chosen is not None, "no set in the top-20 passed the closed-loop gate"

  fid = fidelity_sg_all(stops, chosen)
  fid_ref = fidelity_sg_all(stops, dict(SRG_REF))
  base = list(gate_rows)

  collisions = [r for r in base if r["collision"]]
  # ButterStop-order safety check (1d.5): B1 stopping closer than B0 by > 0.05 m.
  by_key = {}
  for r in base:
    by_key.setdefault((r["name"], r["grade"]), {})[r["butter"]] = r
  gap_order_flags = []
  for (name, grade), pair in by_key.items():
    if 0 in pair and 1 in pair:
      f0, f1 = pair[0]["final_gap_m"], pair[1]["final_gap_m"]
      if f0 is not None and f1 is not None:
        delta = f1 - f0
        if delta < -SRG_GAP_ORDER_M:
          gap_order_flags.append({"name": name, "grade": grade, "final_gap_b0": f0,
                                  "final_gap_b1": f1, "delta_m": round(delta, 3)})

  table = {
    "fixture_sha256": NOD_SHA256,
    "hold_table": SR_HOLD,
    "fit": {
      "mode": "real_sg",
      "grid": {"M0": SRG_M0, "K": SRG_K, "L": SRG_L, "tau": SRG_TAU, "SCC_A": SRG_SCC,
               "C_MAX": SRG_CMAX, "V_HI": SRG_VHI, "C_LOW": SRG_CLOW,
               "v_switch": V_SW, "stopping_state": SR_STOPPING, "window_end": SR_WIN_END,
               "grades": list(SRG_GRADES), "n_combo": len(fits)},
      "loss": round(best_loss, 6),
      "best": {"loss": round(best_loss, 6), "params": best_p},
      "edge": edge,
      "top5": top5,
      "selection": {"chosen_rank": chosen_rank, "chosen": chosen,
                    "rejections": rejections},
    },
    "fidelity": fid,
    "fidelity_reference_set": fid_ref,
    "baseline": {"params": {"plant": "real_sg", **BASELINE_PARAMS}, "rows": base,
                 "collisions": [f"{r['name']} b{r['butter']} min_gap={r['min_gap_m']}"
                                for r in collisions],
                 "gap_order_flags": gap_order_flags},
    "note": "desktop open-loop fit / closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  print(f"NOD-FIT best loss={best_loss:.6f} params={best_p} n_combo={len(fits)} edge={edge}")
  print(f"NOD-CHOSEN rank={chosen_rank} loss={round(fits[chosen_rank - 1][0], 6)} params={chosen}")
  for rj in rejections:
    print(f"NOD-REJECT rank={rj['rank']} loss={rj['loss']} at={rj['failed_at']} {rj['reason']}")
  print(f"NOD-FIDELITY pass={fid['pass_count']}/{len(fid['stops'])}")
  for i, row in enumerate(fid["stops"]):
    print(f"NOD-FIDELITY stop{i} {row['route'][-8:]}@{row['minute']} a_g={row['a_grade']:+.3f} "
          f"vRMS={row['vRMS']:.4f} dist={row['dist_sign']}{abs(row['dist_err_m']):.4f} "
          f"stopt={row['stop_time_err_s']:+.4f} awRMS={row['aWheel_rms']:.4f} pass={row['pass']}")
  print(f"NOD-FIDELITY-REF pass={fid_ref['pass_count']}/{len(fid_ref['stops'])}")
  for i, row in enumerate(fid_ref["stops"]):
    print(f"NOD-FIDELITY-REF stop{i} {row['route'][-8:]}@{row['minute']} "
          f"vRMS={row['vRMS']:.4f} dist={row['dist_sign']}{abs(row['dist_err_m']):.4f} "
          f"stopt={row['stop_time_err_s']:+.4f} awRMS={row['aWheel_rms']:.4f} pass={row['pass']}")
  print(f"NOD-BASELINE rows={len(base)} collisions={len(collisions)}")
  for r in collisions:
    print(f"NOD-COLLISION {r['name']} b{r['butter']} min_gap={r['min_gap_m']}")
  for fl in gap_order_flags:
    print(f"NOD-GAPORDER {fl['name']} grade={fl['grade']} b0={fl['final_gap_b0']} "
          f"b1={fl['final_gap_b1']} delta={fl['delta_m']}")
  assert not collisions, "closed-loop collision in CHOSEN set"
  assert not gap_order_flags, "ButterStop 1 stops closer than ButterStop 0 by > 0.05 m"


# ===========================================================================
# STOP-NOD chauffeur-stop (CS): diagnosis + sweep tooling
# ===========================================================================
# B1 = ButterStop as landed (floor + creep guard), unchanged. CS adds a
# lag-compensated taper floor; the harness patches CS constants onto butter_stop
# at run time. Nothing here changes production code.

# The CP-H0 scenarios (replicas + a-group stopped leads).
CS_SCENARIOS = [
  ("replica 445@7.27", "replica", "445@7.27"),
  ("replica 445@8.52", "replica", "445@8.52"),
  ("replica 44c@6.21", "replica", "44c@6.21"),
  ("stopped lead 20 km/h gap 30 m", "a", None),
  ("stopped lead 40 km/h gap 60 m", "a", None),
  ("stopped lead 60 km/h gap 90 m", "a", None),
]

CS_REPLICA_KEYS = ["445@7.27", "445@8.52", "44c@6.21"]
CS_NOLEAD_REPLICA_KEYS = ["443@8.82", "447@7.07"]


def _cs_find_stop(stops, key):
  for s in stops:
    if _stop_key(s) == key:
      return s
  return None


def _cs_scenario_runner(name, group, key, plant_p=None, plant_mode="real_sg"):
  """Return callable() -> trace list for one scenario, driven by the real plant."""
  p = dict(PLANT_REAL) if plant_p is None else dict(plant_p)
  if group == "replica":
    _d, stops = load_stops()
    profile = RealStopReplicaProfile(_cs_find_stop(stops, key))

    def run():
      loop = _RealReplicaLoop({}, profile, p, plant_mode=plant_mode)
      loop.fixed_a_grade = profile.grade
      _apply_baseline_params(Params(), loop)
      loop._trace = []
      loop.run()
      return loop._trace
    return run
  maneuver = next((m for _g, m in _CASES if m.title == name), None)
  assert maneuver is not None, f"scenario not found: {name}"

  def run():
    loop = _RealScenarioLoop(maneuver, 1, p, plant_mode=plant_mode)
    _apply_baseline_params(Params(), loop)
    loop._trace = []
    loop.run()
    return loop._trace
  return run


def cs_trace_metrics(trace):
  """Summary metrics from a CS trace (same shape family as CP-G)."""
  t = np.array([r["t"] for r in trace], dtype=float)
  v = np.array([r["v"] for r in trace], dtype=float)
  a = np.array([r["plant_a"] if r["plant_a"] is not None else np.nan for r in trace], dtype=float)
  out = np.array([r["out"] if r["out"] is not None else np.nan for r in trace], dtype=float)
  finite = np.isfinite(a)
  stand = np.flatnonzero(v < 0.05)
  stopped = bool(stand.size)
  m = {"stopped": stopped, "a_stop": None, "nod_swing": None, "peak_decel": None,
       "peak_jerk": None, "final_gap_m": None, "min_gap_m": None,
       "post_stop_travel": None, "floor_none_steps": 0, "floor_active_steps": 0}
  if t.size > 1:
    jerk = np.gradient(out, t)
    m["peak_jerk"] = float(np.max(np.abs(jerk)))
  if finite.any():
    m["peak_decel"] = float(np.min(a[finite]))
  g = np.array([r.get("gap", np.nan) for r in trace], dtype=float)
  if np.isfinite(g).any():
    m["min_gap_m"] = float(np.nanmin(g))
  for r in trace:
    if r.get("floor_applied") is None:
      m["floor_none_steps"] += 1
    else:
      m["floor_active_steps"] += 1
  if stopped:
    i = int(stand[0])
    m["a_stop"] = float(a[i])
    w = (t >= t[i] - 1.0) & (t <= t[i] + 1.0)
    if w.any() and np.isfinite(a[w]).any():
      m["nod_swing"] = float(np.nanmax(a[w]) - np.nanmin(a[w]))
    if np.isfinite(g[i]):
      m["final_gap_m"] = float(g[i])
    x = np.array([r.get("x", np.nan) for r in trace], dtype=float)
    if np.isfinite(x).any() and np.isfinite(g).any():
      m["post_stop_travel"] = None
  return m
# ===========================================================================
# STOP-NOD chauffeur-stop (CS): CP-H2 sweep, gates H1-H5 / S1-S4, selection
# ===========================================================================
# CS patches butter_stop.V_TAPER_CS/T_LAG/K_CS/D_EXTRA/LEAD_SLACK_CS/
# FLOOR_RELAX_JERK AND butter_stop.BUTTER_CS at run time; nothing here changes
# production code. B1 = the landed ButterStop with BUTTER_CS False.

import copy as _copy  # noqa: E402
import openpilot.selfdrive.controls.lib.butter_stop as _bs  # noqa: E402

_CS_GRID = {
  "V_TAPER_CS": [3.0, 4.0, 5.0],
  "T_LAG": [0.6, 0.9, 1.2],
  "K_CS": [0.3, 0.5],
  "D_EXTRA": [0.4, 0.8, 1.2],
  "LEAD_SLACK_CS": [1.2, 0.6],
  "FLOOR_RELAX_JERK": [0.8, 1.5],
}

_CS_DEFAULTS = {"V_TAPER_CS": 3.0, "T_LAG": 0.0, "K_CS": 0.5, "D_EXTRA": 0.4,
                "LEAD_SLACK_CS": 1.2, "FLOOR_RELAX_JERK": 0.8}
_CS_ATTRS = ("V_TAPER_CS", "T_LAG", "K_CS", "D_EXTRA", "LEAD_SLACK_CS", "FLOOR_RELAX_JERK")

CS_EPS = 0.005
CS_POST_STOP_M = 0.02
CS_DELAY_M = 0.05
CS_DECEL_ADD = 0.4
CS_JERK_ADD = 0.5
CS_GAP_ADD = 1.0
CS_NOD_REDUCE = 0.30
CS_NOD_WORSE = 0.05

CS_ROBUST_VARIANTS = {"L": [0.5, 0.9], "tau": [0.2, 0.6], "SCC_A": [-0.4, -0.8]}


def cs_apply_params(p):
  if p is None:
    cs_restore_defaults()
    return
  _bs.BUTTER_CS = True
  for k in _CS_ATTRS:
    if k in p:
      setattr(_bs, k, p[k])


def cs_restore_defaults():
  _bs.BUTTER_CS = False
  for k, v in _CS_DEFAULTS.items():
    setattr(_bs, k, v)


def cs_grid():
  return [{"V_TAPER_CS": vt, "T_LAG": tl, "K_CS": kc, "D_EXTRA": de,
           "LEAD_SLACK_CS": ls, "FLOOR_RELAX_JERK": fr}
          for vt in _CS_GRID["V_TAPER_CS"] for tl in _CS_GRID["T_LAG"]
          for kc in _CS_GRID["K_CS"] for de in _CS_GRID["D_EXTRA"]
          for ls in _CS_GRID["LEAD_SLACK_CS"] for fr in _CS_GRID["FLOOR_RELAX_JERK"]]


def _cs_plant(plant_p=None, **over):
  base = dict(PLANT_REAL) if plant_p is None else dict(plant_p)
  base.update(over)
  return base


def _cs_row(name, group, tag, t, v, applied, cmd, gap, stock, floor_raw,
            floor_applied, bound, fcw_steps, fcw_times, start_delay, x,
            creep_latched, standstill_confirmed, plan_stop_d, grade=0.0):
  t = np.asarray(t, dtype=float)
  v = np.asarray(v, dtype=float)
  applied = np.asarray(applied, dtype=float)
  cmd = np.asarray(cmd, dtype=float)
  gap = np.asarray(gap, dtype=float)
  stock = np.asarray(stock, dtype=float)
  fraw = np.asarray(floor_raw, dtype=float)
  psd = np.array([np.nan if d is None else float(d) for d in plan_stop_d], dtype=float)
  finite = np.isfinite(gap)
  stand = np.flatnonzero(v < 0.05)
  stopped = bool(stand.size)
  fcw_first_t = fcw_last_t = None
  if fcw_times is not None and len(fcw_times):
    fcw_first_t = round(float(np.min(fcw_times)), 3)
    fcw_last_t = round(float(np.max(fcw_times)), 3)
  step = float(t[1] - t[0]) if t.size > 1 else DT_CTRL
  row = {"name": name, "group": group, "tag": tag, "grade": grade, "stopped": stopped,
         "a_stop": None, "a_at_m03": None, "peak_decel": None, "nod_swing": None,
         "tail_s": None, "final_gap_m": None, "min_gap_after_stop": None,
         "peak_jerk_cmd": None, "post_stop_travel": None,
         "min_gap_m": None if not finite.any() else round(float(np.nanmin(gap)), 3),
         "collision": bool(finite.any() and np.nanmin(gap) < COLLISION_GAP_M),
         "fcw_steps": int(fcw_steps), "fcw_first_t": fcw_first_t, "fcw_last_t": fcw_last_t,
         "start_delay_s": start_delay, "floor_bind_steps": 0, "floor_active_steps": 0,
         "bind_v_lo": None, "bind_v_hi": None, "peak_jerk_plant": None,
         "creep_latched": bool(np.any(creep_latched)),
         "hold_reached": bool(np.any(standstill_confirmed)),
         "ran_stop": bool(np.any(np.isfinite(psd)))}
  active = np.isfinite(fraw)
  binding = active & np.isfinite(stock) & np.isfinite(cmd) & (cmd > stock + 1e-4)
  row["floor_active_steps"] = int(np.sum(active))
  row["floor_bind_steps"] = int(np.sum(binding))
  if binding.any():
    vb = v[binding]
    row["bind_v_lo"] = round(float(np.min(vb)), 3)
    row["bind_v_hi"] = round(float(np.max(vb)), 3)
  if t.size > 1:
    jerk = np.gradient(cmd, t)
    if stand.size:
      aw = (t >= max(0.0, t[stand[0]] - 6.0)) & (t <= t[stand[0]])
      jj = jerk[aw] if aw.any() else jerk
    else:
      jj = jerk[1:] if jerk.size > 1 else jerk
    row["peak_jerk_cmd"] = round(float(np.max(np.abs(jj))) if jj.size else 0.0, 4)
    pj = np.gradient(applied, t)
    row["peak_jerk_plant"] = round(float(np.max(np.abs(pj))) if pj.size else 0.0, 4)
  if stopped:
    i = int(stand[0])
    w = (t >= t[i] - 0.6) & (t <= t[i])
    if w.any():
      row["peak_decel"] = round(float(np.min(applied[w])), 4)
    row["a_stop"] = round(float(applied[i]), 4)
    j = _closest_index(t, t[i] - 0.3)
    row["a_at_m03"] = round(float(applied[j]), 4) if j < applied.size else None
    w2 = (t > t[i]) & (t <= t[i] + 1.0)
    if w2.any():
      row["nod_swing"] = round(float(np.max(applied[w2]) - row["a_stop"]), 4)
    tail_w = (t >= max(0.0, t[i] - 6.0)) & (t <= t[i])
    row["tail_s"] = round(float(np.sum((v > 0.05) & (v < 0.5) & tail_w)) * step, 3)
    if finite.any():
      row["final_gap_m"] = round(float(gap[i]), 3)
      after = (t >= t[i]) & finite
      row["min_gap_after_stop"] = round(float(np.nanmin(gap[after])), 3) if after.any() else None
    xx = np.asarray(x, dtype=float)
    row["post_stop_travel"] = round(float(xx[-1] - xx[i]), 4) if xx.size == t.size else 0.0
  row["_cmd_digest"] = hashlib.sha256(np.nan_to_num(cmd).tobytes()).hexdigest()
  # Private per-step traces for the redefined H3 (stripped from serialized rows).
  row["_gap"] = gap
  row["_applied"] = applied
  row["_final_v"] = round(float(v[-1]), 4) if v.size else None
  return row


def _cs_scenario_row(maneuver, group, butter, p, plant_mode, grade, gtag,
                     plant_p=None, with_startdelay=False):
  grade_rad = math.asin(grade)
  suffix = f" [{gtag}]" if gtag else ""
  loop = _RealScenarioLoop(maneuver, butter, _cs_plant(plant_p), plant_mode=plant_mode,
                           grade_rad=grade_rad)
  loop._trace = []
  logs = _run_one(loop, Params())
  tr = loop._trace
  gap = logs[:, 4] - logs[:, 3]
  fcw_mask = logs[:, 12] > 0.5
  sd = _start_delay(logs[:, 0], logs[:, 1]) if with_startdelay else None
  return _cs_row(maneuver.title + suffix, group, group, logs[:, 0], logs[:, 1], logs[:, 10],
                 logs[:, 2], gap, logs[:, 13], [r["floor_raw"] for r in tr],
                 [r["floor_applied"] for r in tr], [r["bound"] for r in tr],
                 int(np.sum(fcw_mask)), logs[fcw_mask, 0], sd, [r["x"] for r in tr],
                 np.array([r["creep_latched"] for r in tr]),
                 np.array([r["standstill_confirmed"] for r in tr]),
                 [r["plan_stop_d"] for r in tr], grade)


def _cs_fixture_row(profile, group, butter, p, plant_mode, grade, gtag, name, tag,
                    grade_rad=0.0, fixed_grade=None, header=None, plant_p=None):
  suffix = f" [{gtag}]" if gtag else ""
  loop = _RealFixtureLoop(dict(header or {}), profile, _cs_plant(plant_p),
                          plant_mode=plant_mode, grade_rad=grade_rad)
  loop.fixed_a_grade = fixed_grade
  loop._trace = []
  run = _run_one(loop, Params())
  tr = loop._trace
  return _cs_row(name + suffix, group, tag, run["t"], run["v"], run["a_actual"], run["accel"],
                 run["gap"], run["stock"], run["floor_raw"], run["floor_applied"],
                 run["bound"], run["fcw_steps"], run.get("fcw_times"), None,
                 [r["x"] for r in tr], np.array([r["creep_latched"] for r in tr]),
                 np.array([r["standstill_confirmed"] for r in tr]),
                 [r["plan_stop_d"] for r in tr], grade)


_CS_B1_CACHE = {}


def _cs_b1(key, plant_mode, factory):
  ck = (key, plant_mode)
  if ck not in _CS_B1_CACHE:
    cs_apply_params(None)
    _CS_B1_CACHE[ck] = factory()
  return _CS_B1_CACHE[ck]


def _cs_iter_pairs(p, plant_mode="real_sg"):
  fixture = load_fixture("b1")

  def cs_run(factory):
    cs_apply_params(p)
    return factory()

  for group, maneuver in _CASES:
    name = maneuver.title
    factory = (lambda m=maneuver, g=group: _cs_scenario_row(m, g, 1, p, plant_mode, 0.0, ""))
    b1 = _cs_b1(f"man:{name}", plant_mode, factory)
    yield name, b1, cs_run(factory), group

  b1_factory = (lambda: _cs_fixture_row(FixtureProfile(fixture), "b1", 1, p, plant_mode, 0.0, "",
                                        "b1", "b1", header=fixture.get("params")))
  b1 = _cs_b1("b1", plant_mode, b1_factory)
  yield "b1", b1, cs_run(b1_factory), "b1"

  for maneuver in _START_SCENARIOS:
    factory = (lambda m=maneuver: _cs_scenario_row(m, "start", 1, p, plant_mode, 0.0, "",
                                                   with_startdelay=True))
    b1 = _cs_b1(f"start:{maneuver.title}", plant_mode, factory)
    yield maneuver.title, b1, cs_run(factory), "start"

  for grade, gtag in ((SRG_GRADES[0], "dn3"), (SRG_GRADES[1], "up3")):
    gb1_factory = (lambda gr=grade, gr2=grade, t=gtag: _cs_fixture_row(
      FixtureProfile(fixture), "b1", 1, p, plant_mode, gr, t, "b1", "b1",
      grade_rad=math.asin(gr2), header=fixture.get("params")))
    b1 = _cs_b1(f"b1:{gtag}", plant_mode, gb1_factory)
    yield f"b1 [{gtag}]", b1, cs_run(gb1_factory), "b1"
    for group, maneuver in _STOPPED_LEAD:
      factory = (lambda m=maneuver, g=group, gr=grade, t=gtag: _cs_scenario_row(
        m, g, 1, p, plant_mode, gr, t))
      b1 = _cs_b1(f"man:{maneuver.title}:{gtag}", plant_mode, factory)
      yield maneuver.title + f" [{gtag}]", b1, cs_run(factory), group

  _d, stops = load_stops()
  for key in CS_REPLICA_KEYS:
    profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
    name = f"replica {key}"
    factory = (lambda pr=profile, n=name: _cs_fixture_row(
      pr, "replica", 1, p, plant_mode, 0.0, "", n, "replica", fixed_grade=pr.grade))
    b1 = _cs_b1(f"rep:{key}", plant_mode, factory)
    yield name, b1, cs_run(factory), "replica"

  for key in CS_NOLEAD_REPLICA_KEYS:
    profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
    name = f"replica {key}"
    factory = (lambda pr=profile, n=name: _cs_fixture_row(
      pr, "replica_nolead", 1, p, plant_mode, 0.0, "", n, "replica_nolead",
      fixed_grade=pr.grade))
    b1 = _cs_b1(f"repnl:{key}", plant_mode, factory)
    yield name, b1, cs_run(factory), "replica_nolead"

  for axis, vals in CS_ROBUST_VARIANTS.items():
    for val in vals:
      vtag = f"{axis}={val}"
      pv = _cs_plant(**{axis: val})
      for key in CS_REPLICA_KEYS:
        profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
        name = f"replica {key} [{vtag}]"
        factory = (lambda pr=profile, n=name, nv=pv: _cs_fixture_row(
          pr, "replica", 1, p, plant_mode, 0.0, "", n, "replica",
          fixed_grade=pr.grade, plant_p=nv))
        b1 = _cs_b1(f"rep:{key}:{vtag}", plant_mode, factory)
        yield name, b1, cs_run(factory), "replica"
      man40 = next(m for g, m in _CASES if m.title == "stopped lead 40 km/h gap 60 m")
      name = f"stopped lead 40 km/h gap 60 m [{vtag}]"
      factory = (lambda m=man40, nv=pv: _cs_scenario_row(m, "a", 1, p, plant_mode, 0.0, "", plant_p=nv))
      b1 = _cs_b1(f"man40:{vtag}", plant_mode, factory)
      yield name, b1, cs_run(factory), "a"


def _cs_h3_new(b1, cs):
  """Redefined H3 for no-planned-stop scenarios: same or safer, never harsher.

  At every step gap_cs >= gap_b1 - 0.005 m; peak decel <= b1 + 0.1; peak plant
  |jerk| <= b1 + 0.2; no extra stop (both must end still not stopping, or the
  final speed states match). Returns a list of violation strings.
  """
  out = []
  g1, g2 = b1.get("_gap"), cs.get("_gap")
  if g1 is not None and g2 is not None and len(g1) == len(g2) and len(g1):
    worst = float(np.nanmin(np.asarray(g2) - np.asarray(g1)))
    if worst < -CS_EPS:
      out.append(f"H3 gap shrink {worst:.4f} m")
  b1d, c2d = b1["peak_decel"], cs["peak_decel"]
  if b1d is not None and c2d is not None and c2d < b1d - 0.1:
    out.append(f"H3 decel {b1d}->{c2d}")

  b1jp, c2jp = b1.get("peak_jerk_plant"), cs.get("peak_jerk_plant")
  if b1jp is not None and c2jp is not None and c2jp > b1jp + 0.2 + 1e-9:
    out.append(f"H3 plant jerk {b1jp}->{c2jp}")
  # No extra stop: if B1 did not stop, CS must not either.
  if not b1["stopped"] and cs["stopped"]:
    out.append("H3 extra stop where B1 did not stop")
  return out


def _cs_gate_failures(b1, cs, tag, name):
  out = []
  if b1["final_gap_m"] is not None and cs["final_gap_m"] is not None:
    if cs["final_gap_m"] < b1["final_gap_m"] - CS_EPS:
      out.append(f"H1 gap {cs['final_gap_m']} < b1 {b1['final_gap_m']}")
  if b1["min_gap_m"] is not None and cs["min_gap_m"] is not None:
    if cs["min_gap_m"] < b1["min_gap_m"] - CS_EPS:
      out.append(f"H1 min_gap {cs['min_gap_m']} < b1 {b1['min_gap_m']}")
  if cs["collision"]:
    out.append(f"H2 collision min_gap={cs['min_gap_m']}")
  if cs["fcw_steps"] > b1["fcw_steps"]:
    out.append(f"H2 fcw {cs['fcw_steps']} > b1 {b1['fcw_steps']}")
  # H3 (redefined): a scenario with no planned stop must be same-or-safer, never
  # harsher. Bit-identity is impossible when D_EXTRA != 0.4 because D_EXTRA is
  # part of the MPC stop distance and shifts the following distance everywhere.
  # Old bit-identity H3 is kept only as the informational h3_old field.
  if not b1["ran_stop"]:
    out.extend(_cs_h3_new(b1, cs))
  if b1["post_stop_travel"] is not None and cs["post_stop_travel"] is not None:
    if cs["post_stop_travel"] > b1["post_stop_travel"] + CS_POST_STOP_M + 1e-9:
      out.append(f"H4 post_stop {cs['post_stop_travel']} > b1 {b1['post_stop_travel']}+{CS_POST_STOP_M}")
  if cs["creep_latched"] and not b1["creep_latched"]:
    out.append("H4 creep latched where B1 did not")
  if b1["start_delay_s"] is not None and cs["start_delay_s"] is not None:
    if abs(cs["start_delay_s"] - b1["start_delay_s"]) > CS_DELAY_M:
      out.append(f"H5 start_delay {cs['start_delay_s']} vs {b1['start_delay_s']}")
  return out


def _cs_comfort_failures(b1, cs):
  out = []
  if b1["peak_decel"] is not None and cs["peak_decel"] is not None:
    if cs["peak_decel"] < b1["peak_decel"] - CS_DECEL_ADD:
      out.append(f"S2 decel {b1['peak_decel']}->{cs['peak_decel']}")
  if cs["peak_jerk_cmd"] is not None:
    b1j = b1["peak_jerk_cmd"] if b1["peak_jerk_cmd"] is not None else 0.0
    if cs["peak_jerk_cmd"] > b1j + CS_JERK_ADD + 1e-9:
      out.append(f"S3 jerk {b1j}->{cs['peak_jerk_cmd']}")
  if b1["final_gap_m"] is not None and cs["final_gap_m"] is not None:
    if cs["final_gap_m"] - b1["final_gap_m"] > CS_GAP_ADD:
      out.append(f"S4 gap +{cs['final_gap_m'] - b1['final_gap_m']:.3f}")
  return out


def _cs_nod_reductions(pairs):
  """(reductions list, worst) over the 3 replicas + a-group."""
  reds = []
  for name, b1, cs, tag in pairs:
    if tag in ("replica", "a") and b1["nod_swing"] is not None and cs["nod_swing"] is not None:
      denom = b1["nod_swing"] if abs(b1["nod_swing"]) > 1e-6 else 1e-6
      reds.append((name, (b1["nod_swing"] - cs["nod_swing"]) / denom))
  worst = min((r for _n, r in reds), default=float("-inf"))
  mean = float(np.mean([r for _n, r in reds])) if reds else float("-inf")
  return reds, mean, worst


def _cs_eval_params(p, plant_mode="real_sg"):
  pairs, hard, comfort = [], [], []
  h3_old = []
  try:
    for name, b1, cs, tag in _cs_iter_pairs(p, plant_mode=plant_mode):
      pairs.append((name, b1, cs, tag))
      for reason in _cs_gate_failures(b1, cs, tag, name):
        hard.append({"name": name, "tag": tag, "reason": reason})
      for reason in _cs_comfort_failures(b1, cs):
        comfort.append({"name": name, "tag": tag, "reason": reason})
      # Informational: old bit-identity H3 on no-planned-stop scenarios.
      if not b1["ran_stop"]:
        h3_old.append({"name": name,
                       "identical": bool(b1.get("_cmd_digest") == cs.get("_cmd_digest"))})
  finally:
    cs_restore_defaults()
  reds, mean, worst = _cs_nod_reductions(pairs)
  s1_ok = mean >= CS_NOD_REDUCE and worst >= -CS_NOD_WORSE
  analysis = {"pairs": pairs, "nod_mean": mean, "nod_worst": worst,
              "s1_ok": bool(s1_ok), "nod_reductions": reds, "h3_old": h3_old}
  return analysis, hard, comfort


def cs_sweep(plant_mode="real_sg", only=None, shard=None, nshards=None):
  grid = only if only is not None else cs_grid()
  if shard is not None and nshards is not None:
    grid = [g for i, g in enumerate(grid) if i % nshards == shard]
  results = []
  for p in grid:
    analysis, hard, comfort = _cs_eval_params(p, plant_mode=plant_mode)
    passes = not hard and not comfort and analysis["s1_ok"]
    results.append({"params": dict(p), "hard": hard, "comfort": comfort,
                    "nod_mean": analysis["nod_mean"], "nod_worst": analysis["nod_worst"],
                    "s1_ok": analysis["s1_ok"], "passes": passes, "analysis": analysis})
  results.sort(key=lambda r: (not r["passes"], -r["nod_mean"]))
  return results


def _cs_replica_sanity(plant_mode="real_sg"):
  _d, stops = load_stops()
  out = []
  for key in CS_REPLICA_KEYS:
    stop = _cs_find_stop(stops, key)
    profile = RealStopReplicaProfile(stop)
    cs_apply_params(None)
    try:
      row = _cs_fixture_row(profile, "replica", 1, _CS_DEFAULTS, plant_mode, 0.0, "",
                            f"replica {key}", "replica", fixed_grade=profile.grade)
    finally:
      cs_restore_defaults()
    out.append({"key": key, "has_lead": bool(profile.has_lead),
                "log_final_gap_m": None if profile.log_final_gap is None else round(profile.log_final_gap, 3),
                "sim_final_gap_m": row["final_gap_m"], "sim_stopped": row["stopped"],
                "sim_a_stop": row["a_stop"]})
  return out


def test_cs_fixture_uses_real_harness_defaults():
  """CS defaults must equal the landed B1 constants (no accidental drift)."""
  assert _CS_DEFAULTS["V_TAPER_CS"] == _bs.V_TAPER
  assert _CS_DEFAULTS["K_CS"] == _bs.K_TAPER
  assert _CS_DEFAULTS["D_EXTRA"] == 0.4
  assert _CS_DEFAULTS["LEAD_SLACK_CS"] == _bs.LEAD_SLACK
  assert _CS_DEFAULTS["T_LAG"] == 0.0
  assert not getattr(_bs, "BUTTER_CS")


def test_cs_replica_sanity_and_smoke():
  if os.environ.get("BUTTER_NOD_FULL") != "1":
    pytest.skip("heavy closed-loop CS smoke; set BUTTER_NOD_FULL=1 to run")
  rows = _cs_replica_sanity()
  print(f"NOD-CS-SANITY {json.dumps(rows, sort_keys=True)}")
  for r in rows:
    if r["has_lead"]:
      assert r["sim_stopped"] and r["sim_final_gap_m"] is not None
  # Smoke: one param set x {1 replica, 1 a-scenario, 1 grade} - every metric finite.
  p = {"V_TAPER_CS": 4.0, "T_LAG": 0.9, "K_CS": 0.5, "D_EXTRA": 0.8,
       "LEAD_SLACK_CS": 0.6, "FLOOR_RELAX_JERK": 1.5}
  pairs = list(_cs_iter_pairs(p, plant_mode="real_sg"))
  sample = [pr for pr in pairs if pr[3] in ("replica", "a")][:6]
  metrics = []
  for name, b1, cs, tag in sample:
    for row in (b1, cs):
      for key in ("a_stop", "peak_decel", "peak_jerk_cmd", "final_gap_m", "min_gap_m", "tail_s"):
        v = row[key]
        assert v is None or math.isfinite(v), f"{name}:{key}={v}"
        if v is not None:
          metrics.append(v)
  print(f"NOD-CS-SMOKE n={len(sample)} metrics_min={min(metrics):.4f} max={max(metrics):.4f}")
  assert max(metrics) - min(metrics) > 1e-6


def test_cs_sweep():
  if os.environ.get("BUTTER_NOD_FULL") != "1":
    pytest.skip("heavy CS grid sweep; set BUTTER_NOD_FULL=1 to run")
  out_path = os.environ.get("BUTTER_CS_OUT", "/tmp/butter_cs.json")
  results = cs_sweep()
  passing = [r for r in results if r["passes"]]
  print(f"NOD-CS-SWEEP n_combo={len(results)} n_passing={len(passing)}")
  for i, r in enumerate(results[:5], start=1):
    print(f"NOD-CS-TOP{i} nod_mean={r['nod_mean']:.4f} params={r['params']} "
          f"hard={len(r['hard'])} comfort={len(r['comfort'])} s1={r['s1_ok']}")
  assert results

# ===========================================================================
# LongActuatorDelay sweep (AD): analysis/sim only, no product change
# ===========================================================================
# B1 = the user's real config at LongActuatorDelay 0.25 (Params INT x0.01) with
# landed ButterStop (BUTTER_CS off). The sweep varies LongActuatorDelay and
# checks, per value, H1-H5 hard gates and S1-S6 comfort, all vs B1.

AD_DELAYS = [0.25, 0.4, 0.5, 0.65, 0.8, 1.0]
AD_B1_DELAY = 0.25
# The user's full parameter set (the defaults already match except the delay).
AD_USER_PARAMS = {"ButterStop": 1, "VEgoStopping": 10, "StoppingAccel": -50,
                  "StopDistanceCarrot": 550, "LongTuningKpV": 100,
                  "LongTuningKiV": 0, "LongTuningKf": 100}

AD_EPS = 0.005
AD_POST_STOP_M = 0.02
AD_DELAY_M = 0.05
AD_DECEL_ADD = 0.4
AD_JERK_ADD = 0.5
AD_GAP_ADD = 1.5
AD_FOLLOW_STD_ADD = 0.05
AD_CRUISE_OVER_ADD = 0.3   # km/h
AD_CREEP_V = _bs.CREEP_V


def ad_set_params(params, delay):
  """Seed the user's params + LongActuatorDelay (Params INT = seconds x100)."""
  params.put_int("ButterStop", AD_USER_PARAMS["ButterStop"])
  params.put_int("VEgoStopping", AD_USER_PARAMS["VEgoStopping"])
  params.put_int("StoppingAccel", AD_USER_PARAMS["StoppingAccel"])
  params.put_int("StopDistanceCarrot", AD_USER_PARAMS["StopDistanceCarrot"])
  params.put_int("LongTuningKpV", AD_USER_PARAMS["LongTuningKpV"])
  params.put_int("LongTuningKiV", AD_USER_PARAMS["LongTuningKiV"])
  params.put_int("LongTuningKf", AD_USER_PARAMS["LongTuningKf"])
  params.put_int("LongActuatorDelay", int(round(delay * 100)))


def _ad_apply(loop, params, delay):
  ad_set_params(params, delay)
  loop.long_control._refresh_stopping_accel()
  loop.long_control._refresh_butter_stop()
  loop.long_control._refresh_longitudinal_tuning()


def ad_action_t_of(loop):
  return float(getattr(loop.planner, "action_t_last", float("nan")))


def _ad_metrics(trace, tag):
  """Metric dict from an AD trace. `tag` selects scenario-class extra metrics."""
  t = np.array([r["t"] for r in trace], dtype=float)
  v = np.array([r["v"] for r in trace], dtype=float)
  a = np.array([r["plant_a"] if r["plant_a"] is not None else np.nan for r in trace], dtype=float)
  cmd = np.array([r["out"] if r["out"] is not None else np.nan for r in trace], dtype=float)
  gap = np.array([r.get("gap", np.nan) for r in trace], dtype=float)
  vlead = np.array([r.get("v_lead", np.nan) for r in trace], dtype=float)
  vcru = np.array([r.get("v_cruise_ms", np.nan) for r in trace], dtype=float)
  x = np.array([r.get("x", np.nan) for r in trace], dtype=float)
  finite = np.isfinite(gap)
  stand = np.flatnonzero(v < 0.05)
  stopped = bool(stand.size)
  step = float(t[1] - t[0]) if t.size > 1 else DT_CTRL
  m = {"stopped": stopped, "a_stop": None, "nod_swing": None, "peak_decel": None,
       "peak_jerk_plant": None, "p95_jerk_plant": None, "final_gap_m": None,
       "min_gap_m": None, "min_ttc": None, "post_stop_travel": None,
       "creep_latched": bool(np.any([r.get("creep_latched", False) for r in trace])),
       "hold_reached": bool(np.any([r.get("standstill_confirmed", False) for r in trace])),
       "start_delay_s": None, "follow_std": None, "cruise_over_kph": None,
       "collision": bool(finite.any() and np.nanmin(gap) < COLLISION_GAP_M),
       "fcw_steps": int(sum(1 for r in trace if r.get("fcw", False))),
       "action_t": float(np.nanmax([r.get("action_t", np.nan) for r in trace])) if trace else None}
  if finite.any():
    m["min_gap_m"] = round(float(np.nanmin(gap)), 3)
  # Private per-step traces (stripped from serialized rows); used by the SD H0
  # identity check to compare the disabled SD path against the landed AD path.
  m["_t"] = t
  m["_v"] = v
  m["_out"] = cmd
  m["_plant"] = a
  if t.size > 1:
    jp = np.abs(np.gradient(a, t))
    m["peak_jerk_plant"] = round(float(np.nanmax(jp)) if np.isfinite(jp).any() else 0.0, 4)
    m["p95_jerk_plant"] = round(float(np.nanpercentile(jp, 95)) if np.isfinite(jp).any() else 0.0, 4)
  if finite.any() and np.isfinite(vlead).any():
    closing = v - vlead
    approach = finite & (closing > 0.1) & (gap > 0.1)
    if approach.any():
      ttc = gap[approach] / closing[approach]
      m["min_ttc"] = round(float(np.min(ttc)), 3)
  # start delay: first moving sample relative to t=0 (maneuvers start at 0, but
  # the CS harness uses a 5 s offset; use the same convention as _start_delay).
  moving = np.flatnonzero(v > 0.3)
  if moving.size:
    m["start_delay_s"] = round(float(t[moving[0]] - 5.0), 3)
  if t.size > 1 and np.isfinite(a).any():
    # steady-following accel std over the pre-decel window (t in [5, 14]).
    w = (t >= 5.0) & (t <= 14.0) & np.isfinite(a)
    if w.sum() > 20:
      m["follow_std"] = round(float(np.nanstd(a[w])), 4)
  if np.isfinite(vcru).any() and np.isfinite(v).any():
    # cruise speed overshoot above the commanded cruise target (km/h).
    over = (v - vcru) * 3.6
    over = over[np.isfinite(over)]
    if over.size:
      m["cruise_over_kph"] = round(float(np.max(over)), 3)
  if stopped:
    i = int(stand[0])
    m["a_stop"] = round(float(a[i]), 4)
    m["peak_decel"] = round(float(np.nanmin(a[max(0, i - 30):i + 1])), 4)
    w2 = (t > t[i]) & (t <= t[i] + 1.0)
    if w2.any():
      m["nod_swing"] = round(float(np.nanmax(a[w2]) - m["a_stop"]), 4)
    if finite.any():
      m["final_gap_m"] = round(float(gap[i]), 3)
    if np.isfinite(x).any():
      m["post_stop_travel"] = round(float(x[-1] - x[i]), 4)
  return m


def _ad_scenario_row(maneuver, group, tag, delay, plant_p=None, plant_mode="real_sg",
                     grade=0.0, gtag="", with_startdelay=False):
  suffix = f" [{gtag}]" if gtag else ""
  loop = _RealScenarioLoop(maneuver, 1, _cs_plant(plant_p), plant_mode=plant_mode,
                           grade_rad=math.asin(grade))
  loop._trace = []
  params = Params()
  _ad_apply(loop, params, delay)
  loop.run()
  m = _ad_metrics(loop._trace, tag)
  m.update({"name": maneuver.title + suffix, "group": group, "tag": tag, "grade": grade,
            "fcw_steps": int(loop.planner.mpc.crash_cnt > 0)})
  return m


def _ad_fixture_row(profile, group, tag, delay, name, plant_p=None, plant_mode="real_sg",
                    fixed_grade=None, header=None, grade=0.0):
  loop = _RealFixtureLoop(dict(header or {}), profile, _cs_plant(plant_p),
                          plant_mode=plant_mode, grade_rad=math.asin(grade))
  loop.fixed_a_grade = fixed_grade
  loop._trace = []
  params = Params()
  _ad_apply(loop, params, delay)
  loop.run()
  m = _ad_metrics(loop._trace, tag)
  m.update({"name": name, "group": group, "tag": tag, "grade": grade,
            "fcw_steps": int(loop.planner.mpc.crash_cnt > 0)})
  return m

# --- AD scenario registry ---------------------------------------------------

def _ad_class_for(name, tag):
  """Gate class: 'stop' (H1/H4), 'start' (H5), else non-stop (H3)."""
  if tag in ("start",):
    return "start"
  if "following" in name or "cruise" in name.lower() or "disabled" in name or "cut-in" in name:
    return "follow"
  return "stop"


def _ad_build_specs():
  """Deterministic list of AD scenario specs.

  Every spec is a callable(delay) -> metrics row. B1 is the same factory run with
  AD_B1_DELAY, so each pair shares the scenario definition.
  """
  specs = []
  fixture = load_fixture("b1")

  # CS scenario set (a/b/c groups, s1/s2/s3*), flat + grades.
  for group, maneuver in _CASES:
    specs.append((maneuver.title, group, "stop",
                  lambda d, m=maneuver, g=group: _ad_scenario_row(m, g, g, d)))
  # b1 fixture (flat + grades).
  specs.append(("b1", "b1", "stop",
                lambda d: _ad_fixture_row(FixtureProfile(fixture), "b1", "b1", d, "b1",
                                          header=fixture.get("params"))))
  for grade, gtag in ((SRG_GRADES[0], "dn3"), (SRG_GRADES[1], "up3")):
    specs.append((f"b1 [{gtag}]", "b1", "stop",
                  lambda d, gr=grade, t=gtag: _ad_fixture_row(
                    FixtureProfile(fixture), "b1", "b1", d, "b1", header=fixture.get("params"),
                    grade=gr)))
    for group, maneuver in _STOPPED_LEAD:
      specs.append((maneuver.title + f" [{gtag}]", group, "stop",
                    lambda d, m=maneuver, g=group, gr=grade, t=gtag: _ad_scenario_row(
                      m, g, g, d, grade=gr, gtag=t)))
  # Starts.
  for maneuver in _START_SCENARIOS:
    specs.append((maneuver.title, "start", "start",
                  lambda d, m=maneuver: _ad_scenario_row(m, "start", "start", d,
                                                         with_startdelay=True)))
  # Replicas (lead) + no-lead.
  _d, stops = load_stops()
  for key in CS_REPLICA_KEYS:
    profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
    specs.append((f"replica {key}", "replica", "stop",
                  lambda d, pr=profile, n=f"replica {key}": _ad_fixture_row(
                    pr, "replica", "stop", d, n, fixed_grade=pr.grade)))
  for key in CS_NOLEAD_REPLICA_KEYS:
    profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
    specs.append((f"replica {key}", "replica_nolead", "stop",
                  lambda d, pr=profile, n=f"replica {key}": _ad_fixture_row(
                    pr, "replica_nolead", "stop", d, n, fixed_grade=pr.grade)))
  # Whole standard longitudinal maneuver list.
  for m in create_maneuvers({"e2e": False, "force_decel": False}):
    cls = _ad_class_for(m.title, None)
    specs.append((m.title, "std", cls,
                  lambda d, m=m, c=cls: _ad_scenario_row(m, "std", c, d)))

  # Plant-robustness variants for 3 replicas + 2 named maneuvers + 1 following
  # + 1 cut-in.
  man40 = next(m for g, m in _CASES if m.title == "stopped lead 40 km/h gap 60 m")
  man_lead_brake = _B_SCENARIO
  man_follow = next(m for m in create_maneuvers({"e2e": False, "force_decel": False})
                    if m.title.startswith("steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2"))
  man_cutin = next(m for m in create_maneuvers({"e2e": False, "force_decel": False})
                   if m.title.startswith("approach slower cut-in"))
  for axis, vals in CS_ROBUST_VARIANTS.items():
    for val in vals:
      vtag = f"{axis}={val}"
      pv = _cs_plant(**{axis: val})
      for key in CS_REPLICA_KEYS:
        profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
        specs.append((f"replica {key} [{vtag}]", "replica", "stop",
                      lambda d, pr=profile, n=f"replica {key} [{vtag}]", nv=pv: _ad_fixture_row(
                        pr, "replica", "stop", d, n, fixed_grade=pr.grade, plant_p=nv)))
      specs.append((man40.title + f" [{vtag}]", "a", "stop",
                    lambda d, m=man40, t=vtag, nv=pv: _ad_scenario_row(
                      m, "a", "stop", d, plant_p=nv, gtag=t)))
      specs.append((man_lead_brake.title + f" [{vtag}]", "b", "stop",
                    lambda d, m=man_lead_brake, t=vtag, nv=pv: _ad_scenario_row(
                      m, "b", "stop", d, plant_p=nv, gtag=t)))
      specs.append((man_follow.title + f" [{vtag}]", "std", "follow",
                    lambda d, m=man_follow, t=vtag, nv=pv: _ad_scenario_row(
                      m, "std", "follow", d, plant_p=nv, gtag=t)))
      specs.append((man_cutin.title + f" [{vtag}]", "std", "follow",
                    lambda d, m=man_cutin, t=vtag, nv=pv: _ad_scenario_row(
                      m, "std", "follow", d, plant_p=nv, gtag=t)))
  return specs

# --- AD gates --------------------------------------------------------------

def _ad_pair_failures(b1, ad, cls):
  """Hard gate violations for one (B1, delay) pair; list of reason strings."""
  out = []
  if cls == "stop":
    if b1["final_gap_m"] is not None and ad["final_gap_m"] is not None:
      if ad["final_gap_m"] < b1["final_gap_m"] - AD_EPS:
        out.append(f"H1 gap {ad['final_gap_m']} < b1 {b1['final_gap_m']}")
    if b1["min_gap_m"] is not None and ad["min_gap_m"] is not None:
      if ad["min_gap_m"] < b1["min_gap_m"] - AD_EPS:
        out.append(f"H1 min_gap {ad['min_gap_m']} < b1 {b1['min_gap_m']}")
    if b1["post_stop_travel"] is not None and ad["post_stop_travel"] is not None:
      if ad["post_stop_travel"] > b1["post_stop_travel"] + AD_POST_STOP_M + 1e-9:
        out.append(f"H4 post_stop {ad['post_stop_travel']} > b1 {b1['post_stop_travel']}+{AD_POST_STOP_M}")
    if ad["creep_latched"] and not b1["creep_latched"]:
      out.append("H4 creep latched where B1 did not")
    if not ad["hold_reached"] and b1["hold_reached"]:
      out.append("H4 hold not reached where B1 reached it")
  if cls == "start":
    if b1["start_delay_s"] is not None and ad["start_delay_s"] is not None:
      if ad["start_delay_s"] > b1["start_delay_s"] + AD_DELAY_M:
        out.append(f"H5 start_delay {ad['start_delay_s']} > b1 {b1['start_delay_s']}+{AD_DELAY_M}")
  # H2 + H3 apply to all classes.
  if ad["collision"]:
    out.append(f"H2 collision min_gap={ad['min_gap_m']}")
  if ad["fcw_steps"] > b1["fcw_steps"]:
    out.append(f"H2 fcw {ad['fcw_steps']} > b1 {b1['fcw_steps']}")
  if cls != "stop":
    if b1["min_gap_m"] is not None and ad["min_gap_m"] is not None:
      if ad["min_gap_m"] < b1["min_gap_m"] - AD_EPS:
        out.append(f"H3 min_gap {ad['min_gap_m']} < b1 {b1['min_gap_m']}")
    if b1["min_ttc"] is not None and ad["min_ttc"] is not None:
      if ad["min_ttc"] < b1["min_ttc"] - 1e-9:
        out.append(f"H3 min_ttc {ad['min_ttc']} < b1 {b1['min_ttc']}")
  return out


def _ad_comfort_failures(b1, ad, cls):
  """Comfort gate violations (S2-S6); S1 reported separately."""
  out = []
  if b1["peak_decel"] is not None and ad["peak_decel"] is not None:
    if ad["peak_decel"] < b1["peak_decel"] - AD_DECEL_ADD:
      out.append(f"S2 decel {b1['peak_decel']}->{ad['peak_decel']}")
  if ad["peak_jerk_plant"] is not None:
    bj = b1["peak_jerk_plant"] if b1["peak_jerk_plant"] is not None else 0.0
    if ad["peak_jerk_plant"] > bj + AD_JERK_ADD + 1e-9:
      out.append(f"S3 jerk {bj}->{ad['peak_jerk_plant']}")
  if b1["final_gap_m"] is not None and ad["final_gap_m"] is not None:
    if ad["final_gap_m"] - b1["final_gap_m"] > AD_GAP_ADD:
      out.append(f"S4 gap +{ad['final_gap_m'] - b1['final_gap_m']:.3f}")
  if cls == "follow":
    if b1["follow_std"] is not None and ad["follow_std"] is not None:
      if ad["follow_std"] > b1["follow_std"] + AD_FOLLOW_STD_ADD:
        out.append(f"S5 follow_std {b1['follow_std']}->{ad['follow_std']}")
    if b1["cruise_over_kph"] is not None and ad["cruise_over_kph"] is not None:
      if ad["cruise_over_kph"] > b1["cruise_over_kph"] + AD_CRUISE_OVER_ADD:
        out.append(f"S6 cruise_over {b1['cruise_over_kph']}->{ad['cruise_over_kph']}")
  return out


def _ad_nod_reductions(pairs):
  """S1: per-stop-scenario nod reduction over replicas + a-group."""
  reds = []
  for name, b1, ad, cls in pairs:
    if cls == "stop" and b1["nod_swing"] is not None and ad["nod_swing"] is not None:
      if b1["group"] in ("replica", "a"):
        denom = b1["nod_swing"] if abs(b1["nod_swing"]) > 1e-6 else 1e-6
        reds.append((name, (b1["nod_swing"] - ad["nod_swing"]) / denom))
  mean = float(np.mean([r for _n, r in reds])) if reds else float("inf")
  return reds, mean


def _ad_eval_delay(specs, b1_cache, delay):
  """Run all specs for `delay`; return (pairs, hard, comfort). Pair = tuple.

  A scenario that raises is recorded as an H-error and skipped, so one bad spec
  cannot abort the whole delay.
  """
  pairs, hard, comfort = [], [], []
  for name, group, cls, factory in specs:
    try:
      if name not in b1_cache:
        b1_cache[name] = factory(AD_B1_DELAY)
      b1 = b1_cache[name]
      ad = factory(delay)
    except Exception as exc:  # noqa: BLE001
      hard.append({"name": name, "class": cls, "reason": f"HERR {type(exc).__name__}: {exc}"})
      continue
    pairs.append((name, b1, ad, cls))
    for reason in _ad_pair_failures(b1, ad, cls):
      hard.append({"name": name, "class": cls, "reason": reason})
    for reason in _ad_comfort_failures(b1, ad, cls):
      comfort.append({"name": name, "class": cls, "reason": reason})
  return pairs, hard, comfort


def ad_b1_cache(specs):
  cache = {}
  for name, group, cls, factory in specs:
    try:
      cache[name] = factory(AD_B1_DELAY)
    except Exception:  # noqa: BLE001
      pass
  return cache


# ===========================================================================
# Stop-phase actuator delay sweep (SD): analysis/sim only, no product change
# ===========================================================================
# CP-SD: stop-phase-only longer sampling delay. B1 = STOP_DELAY_ENABLED False
# (the landed ButterStop, LongActuatorDelay 0.25). Each set patches the
# stop_delay module constants and re-runs the whole AD scenario set (97 specs).

import openpilot.selfdrive.controls.lib.stop_delay as _sd  # noqa: E402

SD_GRID = {
  "DELAY_LO": [0.4, 0.5, 0.65, 0.8],
  "V_LO": [1.0, 2.0, 3.0, 5.0, 8.0],
  "A_REQ_MAX": [0.6, 0.8, 1.0],
  "MARGIN_SD": [0.3, 0.6],
  "APPLY_TO_LATCH": [False, True],
}
SD_DEFAULTS = {"DELAY_LO": 0.5, "V_LO": 2.0, "A_REQ_MAX": 0.8, "MARGIN_SD": 0.3,
               "APPLY_TO_LATCH": False}
SD_V_FADE = 1.0
SD_SLEW_UP = 0.5
SD_B1_DELAY = AD_B1_DELAY

SD_EPS = 0.005
SD_POST_STOP_M = 0.02
SD_DELAY_M = 0.05
SD_DECEL_ADD = 0.4
SD_JERK_ADD = 0.5
SD_GAP_ABS = 3.0              # absolute final/min gap floor (m)
SD_GAP_CLOSER = 0.5          # reject if closer than B1 by more than this (m)
SD_NOD_REDUCE = 0.15         # S1a big-nod mean reduction floor
SD_NOD_WORSE = 0.02          # S1b max absolute nod-swing increase (m/s^2)
SD_BIG_NOD = 0.7             # a stop scenario enters the big-nod group at/above this
SD_BIG_NOD_REPLICAS = ("445@7.27", "445@8.52")


def _sd_class_for(name, tag):
  """Gate class: 'start' (H5), 'follow' (H3 min gap/TTC), else 'stop' (H1/H4).

  Delegates to the AD title-based classifier (c-group lead-decel maneuvers are
  'follow'; the stopped-car approaches stay 'stop').
  """
  return _ad_class_for(name, tag)


def sd_grid():
  return [{"DELAY_LO": dl, "V_LO": vl, "A_REQ_MAX": ar, "MARGIN_SD": mg,
           "APPLY_TO_LATCH": bool(at)}
          for dl in SD_GRID["DELAY_LO"] for vl in SD_GRID["V_LO"]
          for ar in SD_GRID["A_REQ_MAX"] for mg in SD_GRID["MARGIN_SD"]
          for at in SD_GRID["APPLY_TO_LATCH"]]


def sd_apply_params(p):
  """Enable stop delay and patch the module constants for one candidate set."""
  _sd.STOP_DELAY_ENABLED = True
  for k, v in SD_DEFAULTS.items():
    setattr(_sd, k, p.get(k, v))
  _sd.V_FADE = SD_V_FADE
  _sd.SLEW_UP = SD_SLEW_UP


def sd_restore_defaults():
  _sd.STOP_DELAY_ENABLED = False
  for k, v in SD_DEFAULTS.items():
    setattr(_sd, k, v)
  _sd.V_FADE = SD_V_FADE
  _sd.SLEW_UP = SD_SLEW_UP


def _sd_run(loop):
  """Run a loop at the base LongActuatorDelay 0.25 with the current stop_delay patch."""
  loop._trace = []
  params = Params()
  _ad_apply(loop, params, SD_B1_DELAY)
  loop.run()
  return loop._trace


def _sd_metrics(trace, tag):
  m = _ad_metrics(trace, tag)
  t = np.array([r["t"] for r in trace], dtype=float)
  w = np.array([r.get("stop_delay_w", 0.0) for r in trace], dtype=float)
  sdly = np.array([r.get("stop_delay", np.nan) for r in trace], dtype=float)
  m["_t"] = t
  m["_v"] = np.array([r["v"] for r in trace], dtype=float)
  m["_out"] = np.array([r["out"] if r["out"] is not None else np.nan for r in trace], dtype=float)
  m["_plant"] = np.array([r["plant_a"] if r["plant_a"] is not None else np.nan for r in trace], dtype=float)
  m["_w"] = w
  m["_d_rel"] = np.array([r.get("d_rel", np.nan) for r in trace], dtype=float)
  m["delay_eff_max"] = round(float(np.nanmax(sdly)), 4) if np.isfinite(sdly).any() else None
  m["w_max"] = round(float(np.nanmax(w)), 4)
  onset = np.flatnonzero(w > 0.0)
  if onset.size:
    i = int(onset[0])
    m["w_onset_idx"] = i
    m["w_onset_t"] = round(float(t[i]), 3)
    m["w_onset_v"] = round(float(m["_v"][i]), 3)
  else:
    m["w_onset_idx"] = None
    m["w_onset_t"] = None
    m["w_onset_v"] = None
  return m


def _sd_scenario_row(maneuver, group, tag, plant_p=None, plant_mode="real_sg",
                     grade=0.0, gtag=""):
  suffix = f" [{gtag}]" if gtag else ""
  loop = _RealScenarioLoop(maneuver, 1, _cs_plant(plant_p), plant_mode=plant_mode,
                           grade_rad=math.asin(grade))
  trace = _sd_run(loop)
  m = _sd_metrics(trace, tag)
  m.update({"name": maneuver.title + suffix, "group": group, "tag": tag, "grade": grade,
            "fcw_steps": int(loop.planner.mpc.crash_cnt > 0)})
  return m


def _sd_fixture_row(profile, group, tag, name, plant_p=None, plant_mode="real_sg",
                    fixed_grade=None, header=None, grade=0.0):
  loop = _RealFixtureLoop(dict(header or {}), profile, _cs_plant(plant_p),
                          plant_mode=plant_mode, grade_rad=math.asin(grade))
  loop.fixed_a_grade = fixed_grade
  trace = _sd_run(loop)
  m = _sd_metrics(trace, tag)
  m.update({"name": name, "group": group, "tag": tag, "grade": grade,
            "fcw_steps": int(loop.planner.mpc.crash_cnt > 0)})
  return m


def _sd_build_specs(plant_mode="real_sg"):
  """Same 97-spec scenario set as AD, built with the SD row builders."""
  specs = []
  fixture = load_fixture("b1")

  for group, maneuver in _CASES:
    cls = _sd_class_for(maneuver.title, group)
    specs.append((maneuver.title, group, cls,
                  lambda m=maneuver, g=group, c=cls: _sd_scenario_row(m, g, c,
                                                                       plant_mode=plant_mode)))
  specs.append(("b1", "b1", "stop",
                lambda: _sd_fixture_row(FixtureProfile(fixture), "b1", "b1", "b1",
                                        header=fixture.get("params"), plant_mode=plant_mode)))
  for grade, gtag in ((SRG_GRADES[0], "dn3"), (SRG_GRADES[1], "up3")):
    specs.append((f"b1 [{gtag}]", "b1", "stop",
                  lambda gr=grade, t=gtag: _sd_fixture_row(
                    FixtureProfile(fixture), "b1", "b1", "b1", header=fixture.get("params"),
                    grade=gr, plant_mode=plant_mode)))
    for group, maneuver in _STOPPED_LEAD:
      specs.append((maneuver.title + f" [{gtag}]", group, "stop",
                    lambda m=maneuver, g=group, gr=grade, t=gtag: _sd_scenario_row(
                      m, g, g, grade=gr, gtag=t, plant_mode=plant_mode)))
  for maneuver in _START_SCENARIOS:
    specs.append((maneuver.title, "start", "start",
                  lambda m=maneuver: _sd_scenario_row(m, "start", "start",
                                                      plant_mode=plant_mode)))
  _d, stops = load_stops()
  for key in CS_REPLICA_KEYS:
    profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
    specs.append((f"replica {key}", "replica", "stop",
                  lambda pr=profile, n=f"replica {key}": _sd_fixture_row(
                    pr, "replica", "stop", n, fixed_grade=pr.grade, plant_mode=plant_mode)))
  for key in CS_NOLEAD_REPLICA_KEYS:
    profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
    specs.append((f"replica {key}", "replica_nolead", "stop",
                  lambda pr=profile, n=f"replica {key}": _sd_fixture_row(
                    pr, "replica_nolead", "stop", n, fixed_grade=pr.grade, plant_mode=plant_mode)))
  for m in create_maneuvers({"e2e": False, "force_decel": False}):
    cls = _ad_class_for(m.title, None)
    specs.append((m.title, "std", cls,
                  lambda m=m, c=cls: _sd_scenario_row(m, "std", c, plant_mode=plant_mode)))

  man40 = next(m for g, m in _CASES if m.title == "stopped lead 40 km/h gap 60 m")
  man_lead_brake = _B_SCENARIO
  man_follow = next(m for m in create_maneuvers({"e2e": False, "force_decel": False})
                    if m.title.startswith("steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2"))
  man_cutin = next(m for m in create_maneuvers({"e2e": False, "force_decel": False})
                   if m.title.startswith("approach slower cut-in"))
  for axis, vals in CS_ROBUST_VARIANTS.items():
    for val in vals:
      vtag = f"{axis}={val}"
      pv = _cs_plant(**{axis: val})
      for key in CS_REPLICA_KEYS:
        profile = RealStopReplicaProfile(_cs_find_stop(stops, key))
        specs.append((f"replica {key} [{vtag}]", "replica", "stop",
                      lambda pr=profile, n=f"replica {key} [{vtag}]", nv=pv: _sd_fixture_row(
                        pr, "replica", "stop", n, fixed_grade=pr.grade, plant_p=nv,
                        plant_mode=plant_mode)))
      specs.append((man40.title + f" [{vtag}]", "a", "stop",
                    lambda m=man40, t=vtag, nv=pv: _sd_scenario_row(
                      m, "a", "stop", plant_p=nv, gtag=t, plant_mode=plant_mode)))
      specs.append((man_lead_brake.title + f" [{vtag}]", "b", "stop",
                    lambda m=man_lead_brake, t=vtag, nv=pv: _sd_scenario_row(
                      m, "b", "stop", plant_p=nv, gtag=t, plant_mode=plant_mode)))
      specs.append((man_follow.title + f" [{vtag}]", "std", "follow",
                    lambda m=man_follow, t=vtag, nv=pv: _sd_scenario_row(
                      m, "std", "follow", plant_p=nv, gtag=t, plant_mode=plant_mode)))
      specs.append((man_cutin.title + f" [{vtag}]", "std", "follow",
                    lambda m=man_cutin, t=vtag, nv=pv: _sd_scenario_row(
                      m, "std", "follow", plant_p=nv, gtag=t, plant_mode=plant_mode)))
  return specs


def _sd_speed_phase_identity(b1, sd):
  """H3 speed-phase identity: steps before the first w>0 must equal B1 exactly.

  Returns a list of violation strings; also None-safe on short traces.
  """
  out = []
  onset = sd.get("w_onset_idx")
  b1_out, sd_out = b1.get("_out"), sd.get("_out")
  b1_plant, sd_plant = b1.get("_plant"), sd.get("_plant")
  if b1_out is None or sd_out is None or b1_plant is None or sd_plant is None:
    return out
  n = len(sd_out) if onset is None else int(onset)
  if n <= 0:
    return out
  if not np.array_equal(b1_out[:n], sd_out[:n]):
    out.append("H3 pre-onset command not identical to B1")
  if not np.array_equal(b1_plant[:n], sd_plant[:n]):
    out.append("H3 pre-onset plant not identical to B1")
  return out


def _sd_pair_failures(b1, sd, cls):
  out = []
  if cls == "stop":
    # H1 abs floor (option b): the >= 3.0 m absolute floor applies where B1
    # itself reaches it; where B1 < 3.0 (short-gap scenarios) require no closer
    # than B1 - 0.005 (the d_avail guard should keep w = 0 there).
    b1_f, b1_m = b1["final_gap_m"], b1["min_gap_m"]
    if b1_f is not None and sd["final_gap_m"] is not None:
      if b1_f >= SD_GAP_ABS - 1e-9:
        if sd["final_gap_m"] < SD_GAP_ABS - 1e-9:
          out.append(f"H1 abs final gap {sd['final_gap_m']} < {SD_GAP_ABS}")
      elif sd["final_gap_m"] < b1_f - SD_EPS:
        out.append(f"H1 final gap {sd['final_gap_m']} < b1 {b1_f}-{SD_EPS}")
    if b1_m is not None and sd["min_gap_m"] is not None:
      if b1_m >= SD_GAP_ABS - 1e-9:
        if sd["min_gap_m"] < SD_GAP_ABS - 1e-9:
          out.append(f"H1 abs min gap {sd['min_gap_m']} < {SD_GAP_ABS}")
      elif sd["min_gap_m"] < b1_m - SD_EPS:
        out.append(f"H1 min gap {sd['min_gap_m']} < b1 {b1_m}-{SD_EPS}")
    if b1_f is not None and sd["final_gap_m"] is not None:
      if sd["final_gap_m"] < b1_f - SD_GAP_CLOSER:
        out.append(f"H1 closer {sd['final_gap_m']} < b1 {b1_f}-{SD_GAP_CLOSER}")
    if b1["post_stop_travel"] is not None and sd["post_stop_travel"] is not None:
      if sd["post_stop_travel"] > b1["post_stop_travel"] + SD_POST_STOP_M + 1e-9:
        out.append(f"H4 post_stop {sd['post_stop_travel']} > b1 {b1['post_stop_travel']}+{SD_POST_STOP_M}")
    if sd["creep_latched"] and not b1["creep_latched"]:
      out.append("H4 creep latched where B1 did not")
    if not sd["hold_reached"] and b1["hold_reached"]:
      out.append("H4 hold not reached where B1 reached it")
  if cls == "start":
    if b1["start_delay_s"] is not None and sd["start_delay_s"] is not None:
      if sd["start_delay_s"] > b1["start_delay_s"] + SD_DELAY_M:
        out.append(f"H5 start_delay {sd['start_delay_s']} > b1 {b1['start_delay_s']}+{SD_DELAY_M}")
  # H3 speed-phase identity + the w-onset must be in the low-speed band.
  out.extend(_sd_speed_phase_identity(b1, sd))
  if sd.get("w_onset_v") is not None and sd["w_onset_v"] > _sd.V_LO + _sd.V_FADE + 1e-6:
    out.append(f"H3 w-onset v {sd['w_onset_v']} > V_LO+V_FADE {_sd.V_LO + _sd.V_FADE}")
  # H2 all classes.
  if sd["collision"]:
    out.append(f"H2 collision min_gap={sd['min_gap_m']}")
  if sd["fcw_steps"] > b1["fcw_steps"]:
    out.append(f"H2 fcw {sd['fcw_steps']} > b1 {b1['fcw_steps']}")
  # H3 for follow scenarios: min gap / TTC >= B1 - 0.005 and must not end closer.
  if cls == "follow":
    if b1["min_gap_m"] is not None and sd["min_gap_m"] is not None:
      if sd["min_gap_m"] < b1["min_gap_m"] - SD_EPS:
        out.append(f"H3 follow min_gap {sd['min_gap_m']} < b1 {b1['min_gap_m']}")
    if b1["min_ttc"] is not None and sd["min_ttc"] is not None:
      if sd["min_ttc"] < b1["min_ttc"] - SD_EPS:
        out.append(f"H3 follow min_ttc {sd['min_ttc']} < b1 {b1['min_ttc']}")
    if b1["final_gap_m"] is not None and sd["final_gap_m"] is not None:
      if sd["final_gap_m"] < b1["final_gap_m"] - SD_EPS:
        out.append(f"H3 follow final_gap {sd['final_gap_m']} < b1 {b1['final_gap_m']}")
  return out


def _sd_comfort_failures(b1, sd, cls):
  out = []
  if b1["peak_decel"] is not None and sd["peak_decel"] is not None:
    if sd["peak_decel"] < b1["peak_decel"] - SD_DECEL_ADD:
      out.append(f"S2 decel {b1['peak_decel']}->{sd['peak_decel']}")
  if sd["peak_jerk_plant"] is not None:
    bj = b1["peak_jerk_plant"] if b1["peak_jerk_plant"] is not None else 0.0
    if sd["peak_jerk_plant"] > bj + SD_JERK_ADD + 1e-9:
      out.append(f"S3 jerk {bj}->{sd['peak_jerk_plant']}")
  return out


def _sd_nod_groups(pairs):
  """S1a big-nod mean reduction and S1b worst absolute nod increase."""
  big = []
  stop_worse = float("-inf")
  for name, b1, sd, cls in pairs:
    if cls != "stop":
      continue
    b1n, sdn = b1["nod_swing"], sd["nod_swing"]
    if b1n is not None and sdn is not None:
      stop_worse = max(stop_worse, sdn - b1n)
    if b1n is None or sdn is None:
      continue
    in_big = (b1["group"] == "replica"
              and any(f"replica {k}" == name for k in SD_BIG_NOD_REPLICAS))
    if in_big or b1n >= SD_BIG_NOD:
      denom = b1n if abs(b1n) > 1e-6 else 1e-6
      big.append((name, (b1n - sdn) / denom))
  mean = float(np.mean([r for _n, r in big])) if big else float("-inf")
  return big, mean, stop_worse


def _sd_nod_report(pairs):
  """Per-set B1 vs set nod_swing for the named replicas, b1, and the a-group mean."""
  wanted = [f"replica {k}" for k in ("445@7.27", "445@8.52", "44c@6.21")] + ["b1"]
  report = {}
  for nm in wanted:
    report[nm] = None
  a_b1, a_sd = [], []
  for name, b1, sd, cls in pairs:
    if name in report and b1["nod_swing"] is not None and sd["nod_swing"] is not None:
      report[name] = {"b1": b1["nod_swing"], "sd": sd["nod_swing"]}
    if b1["group"] == "a" and b1["nod_swing"] is not None and sd["nod_swing"] is not None:
      a_b1.append(b1["nod_swing"])
      a_sd.append(sd["nod_swing"])
  report["a_group"] = {
    "b1": round(float(np.mean(a_b1)), 4) if a_b1 else None,
    "sd": round(float(np.mean(a_sd)), 4) if a_sd else None,
  }
  return report


def _sd_eval_params(p, plant_mode="real_sg", b1_cache=None, specs=None):
  if specs is None:
    specs = _sd_build_specs(plant_mode)
  if b1_cache is None:
    b1_cache = _sd_b1_cache(specs)
  pairs, hard, comfort = [], [], []
  sd_apply_params(p)
  try:
    for name, group, cls, factory in specs:
      try:
        b1 = b1_cache[name]
        sd = factory()
      except Exception as exc:  # noqa: BLE001
        hard.append({"name": name, "class": cls, "reason": f"HERR {type(exc).__name__}: {exc}"})
        continue
      pairs.append((name, b1, sd, cls))
      for reason in _sd_pair_failures(b1, sd, cls):
        hard.append({"name": name, "class": cls, "reason": reason})
      for reason in _sd_comfort_failures(b1, sd, cls):
        comfort.append({"name": name, "class": cls, "reason": reason})
  finally:
    sd_restore_defaults()
  big, mean, stop_worse = _sd_nod_groups(pairs)
  s1a_ok = mean >= SD_NOD_REDUCE
  s1b_ok = stop_worse <= SD_NOD_WORSE + 1e-9
  nod_report = _sd_nod_report(pairs)
  min_final = float("inf")
  for _n, _b1, sd, cls in pairs:
    if cls == "stop" and sd["final_gap_m"] is not None:
      min_final = min(min_final, sd["final_gap_m"])
  if not np.isfinite(min_final):
    min_final = None
  analysis = {"pairs": pairs, "big_nod": big, "s1a_mean": mean, "stop_worse": stop_worse,
              "s1a_ok": bool(s1a_ok), "s1b_ok": bool(s1b_ok), "min_final_gap": min_final,
              "nod_report": nod_report}
  passes = (not hard) and (not comfort) and s1a_ok and s1b_ok
  return analysis, hard, comfort, passes


def _sd_b1_cache(specs):
  """B1 = stop_delay disabled; cache the reference rows keyed by scenario name."""
  cache = {}
  sd_restore_defaults()
  for name, group, cls, factory in specs:
    try:
      cache[name] = factory()
    except Exception:  # noqa: BLE001
      pass
  return cache


def sd_sweep(plant_mode="real_sg", only=None, shard=None, nshards=None):
  grid = only if only is not None else sd_grid()
  if shard is not None and nshards is not None:
    grid = [g for i, g in enumerate(grid) if i % nshards == shard]
  specs = _sd_build_specs(plant_mode)
  b1_cache = _sd_b1_cache(specs)
  results = []
  for p in grid:
    analysis, hard, comfort, passes = _sd_eval_params(
      p, plant_mode=plant_mode, b1_cache=b1_cache, specs=specs)
    results.append({"params": dict(p), "hard": hard, "comfort": comfort,
                    "s1a_mean": analysis["s1a_mean"], "s1b_ok": analysis["s1b_ok"],
                    "min_final_gap": analysis["min_final_gap"],
                    "passes": passes, "analysis": analysis})
  results.sort(key=lambda r: (not r["passes"], -r["s1a_mean"]))
  return results


def sd_h0_check(plant_mode="real_sg"):
  """H0: STOP_DELAY_ENABLED False is bit-identical to the landed (AD) path.

  Runs EVERY scenario+variant once through the SD builder with the module
  disabled and through the landed AD builder at the same base delay, and
  compares the per-step command/plant/vehicle traces exactly. Returns a list of
  {"name", "identical"} rows; the sweep aborts if any is not identical.
  """
  ad_cache = ad_b1_cache(_ad_build_specs())          # landed path at delay 0.25
  sd_restore_defaults()
  sd_cache = _sd_b1_cache(_sd_build_specs(plant_mode))  # SD path, module disabled
  out = []
  for name in ad_cache:
    ad_row = ad_cache.get(name)
    sd_row = sd_cache.get(name)
    if ad_row is None or sd_row is None:
      out.append({"name": name, "identical": False, "skipped": True})
      continue
    out.append({"name": name, "identical": _sd_rows_trace_identical(sd_row, ad_row)})
  return out


def _sd_rows_trace_identical(sd_row, ad_row):
  """Compare the SD builder (disabled) row against the landed AD row traces."""
  for k in ("_out", "_plant", "_v", "_t"):
    a, b = sd_row.get(k), ad_row.get(k)
    if a is None or b is None:
      if a is not None or b is not None:
        return False
      continue
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
      return False
    if not np.array_equal(a, b):
      return False
  return True
