"""W-GREENFIX1: front-of-queue TLV green start when the model stop cue is spurious.

Evidence: 2026-10-26 6e3 20:19:52 (model x[-1] 3.65 m, no lead, green overhead, driver had to
press gas after 3.5 s). W-GREEN2 replay of 33 blocked starts: hold 1.5 s + no radar lead
< 30 m + leadsV3[0].prob < 0.30 opens 6e3 584.71 (+1.33 s) and 1010.59 (+1.19 s) and stays
shut on 6d6 2502.81 (pedestrian + crossing SUV, leadsV3 prob >= 0.36) and 6cf 771.49 (lead 13 m).
"""
GREEN_HOLD_S = 1.5            # TLV green/left held this long before the override may start
GREEN_GAP_S = 0.3             # a green gap shorter than this does not restart the hold
LEAD_CLEAR_M = 30.0           # no radar leadOne/leadTwo closer than this
MODEL_LEAD_PROB_MAX = 0.30    # modelV2.leadsV3[0].prob must stay below this (6d6 ep14 floor 0.36)
STANDSTILL_KPH = 1.0


class GreenHold:
  def __init__(self):
    self.since = None
    self.last = -1e9

  def update(self, now: float, tlv_green: bool):
    if tlv_green:
      if self.since is None or now - self.last > GREEN_GAP_S:
        self.since = now
      self.last = now
    elif now - self.last > GREEN_GAP_S:
      self.since = None
    return self.since


def override_ok(now, since, v_ego_kph, lead1_status, lead1_d, lead2_status, lead2_d, model_lead_prob) -> bool:
  return (since is not None and now - since >= GREEN_HOLD_S
          and v_ego_kph < STANDSTILL_KPH
          and not (lead1_status and lead1_d < LEAD_CLEAR_M)
          and not (lead2_status and lead2_d < LEAD_CLEAR_M)
          and model_lead_prob < MODEL_LEAD_PROB_MAX)
