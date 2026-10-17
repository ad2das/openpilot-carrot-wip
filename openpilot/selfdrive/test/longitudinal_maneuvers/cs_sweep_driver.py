"""STOP-NOD chauffeur-stop (CS) parallel sweep driver (WSL openpilot venv).

Shards the CS grid round-robin across N worker processes, evaluates every
scenario per set (via test_butter_nod_plant.cs_sweep), merges the shard results,
and writes the JSON table plus a determinism check (two sets serial vs pool).

Usage (inside WSL, cwd = openpilot checkout):
  BUTTER_CS_OUT=/mnt/c/.../butter_cs.json python cs_sweep_driver.py [nproc]
"""
import json
import os
import sys
import time
import multiprocessing as mp

from openpilot.common.prefix import OpenpilotPrefix
import openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_nod_plant as H


def _clean(row):
  return {k: v for k, v in row.items() if k[0] != "_"}


def _serialize(result):
  return {
    "params": result["params"],
    "hard": result["hard"],
    "comfort": result["comfort"],
    "nod_mean": result["nod_mean"],
    "nod_worst": result["nod_worst"],
    "s1_ok": result["s1_ok"],
    "passes": result["passes"],
    "pairs": [{"name": n, "tag": tag, "b1": _clean(b1), "cs": _clean(cs)}
              for n, b1, cs, tag in result["analysis"]["pairs"]],
  }


def _histogram(results):
  by_gate = {}
  by_scen = {}
  min_fail = None
  for r in results:
    gates = set()
    for f in r["hard"] + r["comfort"]:
      g = f["reason"].split(" ")[0]
      gates.add(g)
      by_scen[f["name"]] = by_scen.get(f["name"], 0) + 1
    for g in gates:
      by_gate[g] = by_gate.get(g, 0) + 1
    nf = len(r["hard"]) + len(r["comfort"]) + (0 if r["s1_ok"] else 1)
    min_fail = nf if min_fail is None else min(min_fail, nf)
  return {"by_gate": by_gate, "by_scenario": by_scen, "min_nfail": min_fail,
          "n_sets": len(results)}


def worker(args):
  shard, nshards, plant_mode = args
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    res = H.cs_sweep(plant_mode=plant_mode, shard=shard, nshards=nshards)
  return [_serialize(r) for r in res]


def _eval_sets(sets, plant_mode):
  out = []
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    for p in sets:
      analysis, hard, comfort = H._cs_eval_params(p, plant_mode=plant_mode)
      out.append(_serialize({"params": dict(p), "hard": hard, "comfort": comfort,
                             "nod_mean": analysis["nod_mean"], "nod_worst": analysis["nod_worst"],
                             "s1_ok": analysis["s1_ok"],
                             "passes": not hard and not comfort and analysis["s1_ok"],
                             "analysis": analysis}))
  return out


def main():
  nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
  out_path = os.environ.get("BUTTER_CS_OUT", "/tmp/butter_cs.json")
  plant_mode = os.environ.get("BUTTER_CS_PLANT", "real_sg")
  t0 = time.time()

  sanity = H._cs_replica_sanity(plant_mode=plant_mode)

  grid = H.cs_grid()
  det_sets = [grid[0], grid[len(grid) // 2]]
  serial = _eval_sets(det_sets, plant_mode)
  with mp.Pool(min(2, nproc)) as pool:
    parallel = pool.map(worker, [(0, 2, plant_mode), (1, 2, plant_mode)])
  par = {json.dumps(d["params"], sort_keys=True): d for d in (parallel[0] + parallel[1])}
  det_ok = True
  for d in serial:
    key = json.dumps(d["params"], sort_keys=True)
    other = par.get(key)
    if other is None or other["nod_mean"] != d["nod_mean"] or other["hard"] != d["hard"] \
       or other["comfort"] != d["comfort"]:
      det_ok = False
      print(f"NOD-CS-DET MISMATCH {key}")
  print(f"NOD-CS-DET ok={det_ok} sets={len(serial)}")

  with mp.Pool(nproc) as pool:
    shards = pool.map(worker, [(i, nproc, plant_mode) for i in range(nproc)])
  results = [r for shard in shards for r in shard]
  results.sort(key=lambda r: (not r["passes"], -r["nod_mean"]))

  passing = [r for r in results if r["passes"]]
  chosen = passing[0] if passing else None
  table = {
    "plant_real": H.PLANT_REAL,
    "grid": {**H._CS_GRID, "n_combo": len(results)},
    "replica_sanity": sanity,
    "determinism_ok": det_ok,
    "npassing": len(passing),
    "top5": [{"params": r["params"], "nod_mean": round(r["nod_mean"], 6),
              "hard": r["hard"][:5], "comfort": r["comfort"][:5], "s1_ok": r["s1_ok"]}
             for r in results[:5]],
    "all_results": [{"params": r["params"], "nod_mean": round(r["nod_mean"], 6),
                     "nod_worst": round(r["nod_worst"], 6), "s1_ok": r["s1_ok"],
                     "nfail": len(r["hard"]) + len(r["comfort"]) + (0 if r["s1_ok"] else 1),
                     "hard": r["hard"], "comfort": r["comfort"]} for r in results],
    "histogram": _histogram(results),
    "chosen": None if chosen is None else {"params": chosen["params"],
                                           "nod_mean": round(chosen["nod_mean"], 6)},
    "chosen_pairs": None if chosen is None else chosen["pairs"],
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  print(f"NOD-CS-SANITY {json.dumps(sanity, sort_keys=True)}")
  print(f"NOD-CS-SWEEP n_combo={len(results)} n_passing={len(passing)} nproc={nproc} "
        f"elapsed_s={time.time() - t0:.1f}")
  for i, r in enumerate(results[:5], start=1):
    print(f"NOD-CS-TOP{i} nod_mean={r['nod_mean']:.5f} worst={r['nod_worst']:.5f} "
          f"params={r['params']} hard={len(r['hard'])} comfort={len(r['comfort'])} s1={r['s1_ok']}")
  if chosen is None:
    print("NOD-CS-RESULT no param set passed all gates")
  else:
    print(f"NOD-CS-CHOSEN {chosen['params']} nod_mean={chosen['nod_mean']:.5f}")
    rep = [p for p in chosen["pairs"] if p["tag"] == "replica" and "[" not in p["name"]]
    for pair in rep:
      b1, cs = pair["b1"], pair["cs"]
      print(f"NOD-CS-REPLICA {pair['name']} | gap b1={b1['final_gap_m']} cs={cs['final_gap_m']} "
            f"| bind_steps cs={cs['floor_bind_steps']} v[{cs['bind_v_lo']},{cs['bind_v_hi']}] "
            f"| b1_bind={b1['floor_bind_steps']} | a_stop b1={b1['a_stop']} cs={cs['a_stop']} "
            f"| nod b1={b1['nod_swing']} cs={cs['nod_swing']}")


if __name__ == "__main__":
  main()
