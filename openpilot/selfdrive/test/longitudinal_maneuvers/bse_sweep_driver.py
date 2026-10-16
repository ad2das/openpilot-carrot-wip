"""STOP-NOD step 2: parallel BSE sweep driver (run inside the WSL openpilot venv).

Shards the BSE grid round-robin across N worker processes, evaluates every
scenario per set (via test_butter_nod_plant.bse_sweep), merges the shard results
and writes the final JSON table plus a determinism check (two sets evaluated
serially vs in the pool must match exactly).

Usage (inside WSL, cwd = openpilot checkout):
  BUTTER_BSE_OUT=/mnt/c/.../butter_bse.json python bse_sweep_driver.py [nproc]
"""
import json
import os
import sys
import time
import multiprocessing as mp

from openpilot.common.prefix import OpenpilotPrefix
import openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_nod_plant as H


def _serialize(result):
  def clean(row):
    return {k: v for k, v in row.items() if not k.startswith("_")}
  return {
    "params": result["params"],
    "failures": result["failures"],
    "score": result["score"],
    "pairs": [{"name": n, "b1": clean(b1), "bse": clean(bse)}
              for n, b1, bse in result["analysis"]["pairs"]],
  }


def _gate_histogram(results):
  """Count, across all sets, how many failed each gate and each scenario."""
  import re
  by_gate = {}
  by_scen = {}
  min_fail = None
  for r in results:
    gates = set()
    for f in r["failures"]:
      g = f["reason"].split(" ")[0]
      gates.add(g)
      by_scen[f["name"]] = by_scen.get(f["name"], 0) + 1
    for g in gates:
      by_gate[g] = by_gate.get(g, 0) + 1
    min_fail = len(r["failures"]) if min_fail is None else min(min_fail, len(r["failures"]))
  return {"by_gate": by_gate, "by_scenario": by_scen, "min_nfail": min_fail,
          "n_sets": len(results)}


def worker(args):
  shard, nshards, plant_mode = args
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    res = H.bse_sweep(plant_mode=plant_mode, shard=shard, nshards=nshards)
  return [_serialize(r) for r in res]


def _eval_sets(sets, plant_mode="real_sg"):
  out = []
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    for p in sets:
      analysis, failures = H._eval_params(p, plant_mode=plant_mode)
      out.append(_serialize({"params": dict(p), "failures": failures,
                             "score": analysis["score"], "analysis": analysis}))
  return out


def main():
  nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
  out_path = os.environ.get("BUTTER_BSE_OUT", "/tmp/butter_bse.json")
  plant_mode = os.environ.get("BUTTER_BSE_PLANT", "real_sg")
  t0 = time.time()

  sanity = H._replica_sanity(plant_mode=plant_mode)

  # Determinism: 2 fixed sets serial vs pool.
  det_sets = [H.bse_grid()[0], H.bse_grid()[len(H.bse_grid()) // 2]]
  serial = _eval_sets(det_sets, plant_mode)
  with mp.Pool(min(2, nproc)) as pool:
    parallel = pool.map(worker, [(0, 2, plant_mode), (1, 2, plant_mode)])
  par_sets = {json.dumps(d["params"], sort_keys=True): d for d in (parallel[0] + parallel[1])}
  det_ok = True
  for d in serial:
    key = json.dumps(d["params"], sort_keys=True)
    other = par_sets.get(key)
    if other is None or other["score"] != d["score"] or other["failures"] != d["failures"]:
      det_ok = False
      print(f"NOD-BSE-DET MISMATCH {key}")
  print(f"NOD-BSE-DET ok={det_ok} sets={len(serial)}")

  grid = H.bse_grid()
  with mp.Pool(nproc) as pool:
    shards = pool.map(worker, [(i, nproc, plant_mode) for i in range(nproc)])
  results = [r for shard in shards for r in shard]
  results.sort(key=lambda r: (len(r["failures"]) > 0, r["score"]))

  passing = [r for r in results if not r["failures"]]
  chosen = passing[0] if passing else None
  table = {
    "plant_real": H.PLANT_REAL,
    "grid": {"K_E": H.BSE_K_E, "L_E": H.BSE_L_E, "MARGIN_E": H.BSE_MARGIN_E,
             "EXTRA_MAX": H.BSE_EXTRA_MAX, "V_E": H.BSE_V_E, "n_combo": len(results)},
    "replica_sanity": sanity,
    "determinism_ok": det_ok,
    "npassing": len(passing),
    "top5": [{"params": r["params"], "score": round(r["score"], 6),
              "failures": r["failures"][:5]} for r in results[:5]],
    "all_results": [{"params": r["params"], "score": round(r["score"], 6),
                     "nfail": len(r["failures"]),
                     "failures": [{"name": f["name"], "tag": f["tag"], "reason": f["reason"]}
                                  for f in r["failures"]]} for r in results],
    "gate_histogram": _gate_histogram(results),
    "chosen": None if chosen is None else {"params": chosen["params"],
                                           "score": round(chosen["score"], 6)},
    "chosen_pairs": None if chosen is None else chosen["pairs"],
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  print(f"NOD-BSE-SANITY {json.dumps(sanity, sort_keys=True)}")
  print(f"NOD-BSE-SWEEP n_combo={len(results)} n_passing={len(passing)} nproc={nproc} "
        f"elapsed_s={time.time() - t0:.1f}")
  for i, r in enumerate(results[:5], start=1):
    print(f"NOD-BSE-TOP{i} score={r['score']:.5f} params={r['params']} "
          f"fails={len(r['failures'])} {r['failures'][:3]}")
  if chosen is None:
    print("NOD-BSE-RESULT no param set passed all gates")
  else:
    print(f"NOD-BSE-CHOSEN {chosen['params']} score={chosen['score']:.5f}")
    for pair in chosen["pairs"]:
      n, b1, bse = pair["name"], pair["b1"], pair["bse"]
      print(f"NOD-BSE-PAIR {n} | a_stop b1={b1['a_stop']} bse={bse['a_stop']} | "
            f"gap b1={b1['final_gap_m']} bse={bse['final_gap_m']} | "
            f"decel b1={b1['peak_decel']} bse={bse['peak_decel']} | bse_steps={bse['bse_steps']}")
    for pair in chosen["pairs"]:
      n, b1, bse = pair["name"], pair["b1"], pair["bse"]
      if bse["group"] == "replica" and any(k in n for k in H.NOD_REPLICA_KEYS):
        print(f"NOD-BSE-NOD {n} a_stop b1={b1['a_stop']} bse={bse['a_stop']}")


if __name__ == "__main__":
  main()
