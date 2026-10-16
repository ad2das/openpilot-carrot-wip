"""Quick BSE probe: evaluate a small list of param sets (post-G4-fix) and print
their gate failures + per-scenario detail. Run inside WSL in the venv.
"""
import json
import sys

from openpilot.common.prefix import OpenpilotPrefix
import openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_nod_plant as H


SETS = [
  {"K_E": 0.5, "L_E": 0.0, "MARGIN_E": 0.0, "EXTRA_MAX": 0.2, "V_E": 5.0},
  {"K_E": 0.5, "L_E": 0.0, "MARGIN_E": 0.0, "EXTRA_MAX": 0.2, "V_E": 4.0},
  {"K_E": 0.8, "L_E": 0.3, "MARGIN_E": 0.0, "EXTRA_MAX": 0.2, "V_E": 4.0},
  {"K_E": 0.8, "L_E": 0.3, "MARGIN_E": 0.2, "EXTRA_MAX": 0.2, "V_E": 4.0},
  {"K_E": 0.5, "L_E": 0.5, "MARGIN_E": 0.2, "EXTRA_MAX": 0.2, "V_E": 3.0},
  {"K_E": 0.8, "L_E": 0.5, "MARGIN_E": 0.2, "EXTRA_MAX": 0.2, "V_E": 3.0},
  {"K_E": 0.5, "L_E": 0.7, "MARGIN_E": 0.2, "EXTRA_MAX": 0.2, "V_E": 3.0},
  {"K_E": 1.2, "L_E": 0.7, "MARGIN_E": 0.2, "EXTRA_MAX": 0.2, "V_E": 3.0},
]


def main():
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    for p in SETS:
      analysis, failures = H._eval_params(p, plant_mode="real_sg")
      print(f"NOD-PROBE {json.dumps(p, sort_keys=True)} score={analysis['score']:.5f} "
            f"nfail={len(failures)}")
      for f in failures:
        print(f"NOD-PROBE-FAIL {f['tag']} {f['name']}: {f['reason']}")
      # Report a_stop improvement on lead replicas for this set.
      for name, b1, bse in analysis["pairs"]:
        if bse["group"] == "replica":
          print(f"NOD-PROBE-REP {name} a_stop b1={b1['a_stop']} bse={bse['a_stop']} "
                f"gap b1={b1['final_gap_m']} bse={bse['final_gap_m']}")


if __name__ == "__main__":
  main()
