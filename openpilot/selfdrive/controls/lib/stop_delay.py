"""Stop-phase-only longer sampling delay (CP-SD experiment, default OFF).

The planner samples its plan at `action_t = longitudinalActuatorDelay + DT_MDL`.
This module schedules a longer delay ONLY in the last low-speed phase of a stop
behind a (near) stationary lead, where extra sampling delay softens the final
braking transition without weakening braking at speed or in hard-brake
following. When STOP_DELAY_ENABLED is False the planner is bit-identical to the
landed B1 behaviour.

Constants are module-level and read at call time so the sweep driver can patch
them per candidate set.
"""
import numpy as np

STOP_DELAY_ENABLED = False   # default OFF (= B1, bit-identical)
DELAY_LO = 0.5               # s, delay used at/below V_LO
V_LO = 2.0                   # m/s, full DELAY_LO at/below
V_FADE = 1.0                 # m/s, linear fade to base delay at V_LO + V_FADE
LEAD_V_MAX = 0.5             # m/s, lead must be (near) stationary
LEAD_A_MIN = -0.5            # m/s^2, lead aLeadK must be >= this (not braking hard)
GAP_ABS = 3.0                # m, absolute final-gap floor
MARGIN_SD = 0.3              # m, extra margin on top of GAP_ABS
A_REQ_MAX = 0.8              # m/s^2, kinematic guard
SLEW_UP = 0.5                # s per s, max rate the effective delay may GROW
APPLY_TO_LATCH = False       # also use the scheduled action_t for plan_stop_speeds/stop_latch


def stop_phase_weight(v_ego, lead_status, d_rel, v_lead, a_lead, fcw, gas, brake):
  """Return w in [0, 1]; 0 means "use the base delay exactly".

  Gate order: disabled / FCW / pedal / no lead / not-stationary lead / braking
  lead / no room / would need real braking / standstill, then a linear fade.
  """
  if not STOP_DELAY_ENABLED or fcw or gas or brake or not lead_status:
    return 0.0
  if v_lead > LEAD_V_MAX or a_lead < LEAD_A_MIN:
    return 0.0
  d_avail = d_rel - GAP_ABS - MARGIN_SD
  if d_avail <= 0.0:
    return 0.0
  if v_ego * v_ego / (2.0 * d_avail) > A_REQ_MAX:
    return 0.0
  if v_ego <= 0.05:
    return 0.0
  return float(np.clip((V_LO + V_FADE - v_ego) / V_FADE, 0.0, 1.0))
