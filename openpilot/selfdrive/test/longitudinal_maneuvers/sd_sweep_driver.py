"""CP-SD stop-phase delay sweep driver (16 workers, WSL openpilot venv).

144 sets = DELAY_LO {0.4,0.5,0.65,0.8} x V_LO {1,2,3} x A_REQ_MAX {0.6,0.8,1.0}
x MARGIN_SD {0.3,0.6} x APPLY_TO_LATCH {False,True}. Each set re-runs the whole
97-spec AD scenario set behind a shared B1 cache. Sets are sharded across the
16-process pool; each worker prints NOD-SD-PROGRESS per finished set.

Usage: BUTTER_SD_OUT=... python sd_sweep_driver.py [nproc]
"""
import json
import os
import sys
import time
import multiprocessing as mp

from openpilot.common.prefix import OpenpilotPrefix
import openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_nod_plant as H


def _clean(m):
  return {k: v for k, v in m.items() if k[0] != "_"}


def _serialize(p, analysis, hard, comfort, passes):
  return {
    "params": dict(p),
    "hard": hard,
    "comfort": comfort,
    "passes": bool(passes),
    "s1a_mean": analysis["s1a_mean"],
    "s1b_ok": analysis["s1b_ok"],
    "stop_worse": analysis["stop_worse"],
    "min_final_gap": analysis["min_final_gap"],
    "s1a_ok": analysis["s1a_ok"],
    "big_nod": [[n, round(r, 5)] for n, r in analysis["big_nod"]],
    "pairs": [{"name": n, "group": b1.get("group"), "class": cls,
               "b1": _clean(b1), "sd": _clean(sd)} for n, b1, sd, cls in analysis["pairs"]],
  }


def worker(args):
  shard, nshards, plant_mode = args
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    grid = [g for i, g in enumerate(H.sd_grid()) if i % nshards == shard]
    specs = H._sd_build_specs(plant_mode)
    b1_cache = H._sd_b1_cache(specs)
    out = []
    t0 = time.monotonic()
    for k, p in enumerate(grid):
      analysis, hard, comfort, passes = H._sd_eval_params(
        p, plant_mode=plant_mode, b1_cache=b1_cache, specs=specs)
      out.append(_serialize(p, analysis, hard, comfort, passes))
      print(f"NOD-SD-PROGRESS shard={shard} done={k + 1}/{len(grid)} "
            f"elapsed_s={time.monotonic() - t0:.1f}", flush=True)
  return out


def _eval_sets(sets, plant_mode):
  out = []
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    specs = H._sd_build_specs(plant_mode)
    b1_cache = H._sd_b1_cache(specs)
    for p in sets:
      analysis, hard, comfort, passes = H._sd_eval_params(
        p, plant_mode=plant_mode, b1_cache=b1_cache, specs=specs)
      out.append(_serialize(p, analysis, hard, comfort, passes))
  return out


def _histogram(results):
  by_gate, by_scen = {}, {}
  for r in results:
    gates = set()
    for f in r["hard"] + r["comfort"]:
      g = f["reason"].split(" ")[0]
      gates.add(g)
      by_scen[f["name"]] = by_scen.get(f["name"], 0) + 1
    for g in gates:
      by_gate[g] = by_gate.get(g, 0) + 1
  return {"by_gate": by_gate, "by_scenario": by_scen, "n_sets": len(results)}


def main():
  nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
  out_path = os.environ.get("BUTTER_SD_OUT", "/tmp/stop_delay.json")
  plant_mode = os.environ.get("BUTTER_SD_PLANT", "real_sg")
  t0 = time.monotonic()

  h0 = H.sd_h0_check(plant_mode=plant_mode)
  h0_ok = all(r["identical"] for r in h0)
  print(f"NOD-SD-H0 ok={h0_ok} rows={h0}", flush=True)

  grid = H.sd_grid()
  with mp.Pool(nproc) as pool:
    shards = pool.map(worker, [(i, nproc, plant_mode) for i in range(nproc)])
  results = [r for shard in shards for r in shard]

  def rank(r):
    mf = r["min_final_gap"] if r["min_final_gap"] is not None else 1e9
    return (-r["s1a_mean"], -mf, r["params"]["DELAY_LO"])

  passing = sorted([r for r in results if r["passes"]], key=rank)
  print(f"NOD-SD-SWEEP-DONE n_combo={len(results)} n_passing={len(passing)} "
        f"elapsed_s={time.monotonic() - t0:.1f}", flush=True)

  det_sets = [grid[0], grid[len(grid) // 2]]
  serial = _eval_sets(det_sets, plant_mode)
  pool_by_params = {json.dumps(r["params"], sort_keys=True): r for r in results}
  det_ok = True
  for d in serial:
    key = json.dumps(d["params"], sort_keys=True)
    other = pool_by_params.get(key)
    if (other is None or other["s1a_mean"] != d["s1a_mean"]
        or other["hard"] != d["hard"] or other["comfort"] != d["comfort"]):
      det_ok = False
      print(f"NOD-SD-DET MISMATCH {key}", flush=True)
  print(f"NOD-SD-DET ok={det_ok} sets={len(serial)}", flush=True)

  chosen = passing[0] if passing else None
  table = {
    "plant_real": H.PLANT_REAL,
    "b1_delay": H.SD_B1_DELAY,
    "grid": {**H.SD_GRID, "n_combo": len(results), "V_FADE": H.SD_V_FADE,
             "SLEW_UP": H.SD_SLEW_UP},
    "h0_ok": h0_ok, "h0": h0,
    "determinism_ok": det_ok,
    "npassing": len(passing),
    "chosen": None if chosen is None else {"params": chosen["params"],
                                           "s1a_mean": round(chosen["s1a_mean"], 6),
                                           "min_final_gap": chosen["min_final_gap"],
                                           "pairs": chosen["pairs"]},
    "top5": [{"params": r["params"], "s1a_mean": round(r["s1a_mean"], 6),
              "min_final_gap": r["min_final_gap"], "passes": r["passes"],
              "hard": r["hard"][:5], "comfort": r["comfort"][:5], "s1b_ok": r["s1b_ok"]}
             for r in sorted(results, key=lambda r: rank(r))[:5]],
    "all_results": [{"params": r["params"], "s1a_mean": round(r["s1a_mean"], 5),
                     "min_final_gap": r["min_final_gap"], "s1b_ok": r["s1b_ok"],
                     "stop_worse": round(r["stop_worse"], 5) if r["stop_worse"] != float("-inf") else None,
                     "passes": r["passes"],
                     "nfail": len(r["hard"]) + len(r["comfort"]) + (0 if r["s1b_ok"] else 1),
                     "hard": r["hard"], "comfort": r["comfort"]}
                    for r in results],
    "histogram": _histogram(results),
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  for i, r in enumerate(sorted(results, key=lambda r: rank(r))[:5], 1):
    print(f"NOD-SD-TOP{i} s1a={r['s1a_mean']:.4f} min_final={r['min_final_gap']} "
          f"passes={r['passes']} hard={len(r['hard'])} comfort={len(r['comfort'])} "
          f"params={r['params']}")
  if chosen is None:
    print("NOD-SD-RESULT no set passed all gates")
  else:
    print(f"NOD-SD-CHOSEN {chosen['params']} s1a={chosen['s1a_mean']:.4f} "
          f"min_final={chosen['min_final_gap']}")
  print("NOD-SD-FINISHED", flush=True)


if __name__ == "__main__":
  main()
