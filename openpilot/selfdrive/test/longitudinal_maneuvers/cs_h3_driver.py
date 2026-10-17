"""STOP-NOD chauffeur-stop (CS) CP-H3 parallel sweep driver (WSL openpilot venv).

Runs the 432-set grid sweep FIRST with 16 worker processes, then does the
determinism check against the already-computed pool results (no extra pool).
Each worker prints+flushes a progress line per finished set.

Every set is simulated once; both H1 definitions (strict and 5 cm) are derived
from the same pair rows so the sweep is not run twice.

Usage (inside WSL, cwd = openpilot checkout):
  BUTTER_CS_OUT=/mnt/c/.../cs_h3.json python cs_h3_driver.py [nproc]
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


def _relaxed_hard(pairs):
  """H1-5cm hard failures recomputed from the same pair rows (no re-sim)."""
  out = []
  for name, b1, cs, tag in pairs:
    for reason in H._cs_gate_failures(b1, cs, tag, name, h1_relax=H.CS_H1_RELAX):
      out.append({"name": name, "tag": tag, "reason": reason})
  return out


def _verdicts(pairs, hard_strict, comfort):
  hard_relax = _relaxed_hard(pairs)
  s1 = H._cs_nod_reductions(pairs)
  s1_ok = (s1[1] >= H.CS_NOD_REDUCE and s1[3] <= H.CS_NOD_WORSE_ABS + 1e-9
           and s1[4] <= H.CS_NOD_WORSE_ABS + 1e-9)
  return {
    "hard_strict": hard_strict, "comfort": comfort,
    "hard_relax": hard_relax, "s1_ok": bool(s1_ok),
    "pass_strict": (not hard_strict) and (not comfort) and bool(s1_ok),
    "pass_relax": (not hard_relax) and (not comfort) and bool(s1_ok),
  }


def _min_gap_margin(pairs):
  m = float("inf")
  for _n, b1, cs, _t in pairs:
    if b1["final_gap_m"] is not None and cs["final_gap_m"] is not None:
      m = min(m, cs["final_gap_m"] - b1["final_gap_m"])
    if b1["min_gap_m"] is not None and cs["min_gap_m"] is not None:
      m = min(m, cs["min_gap_m"] - b1["min_gap_m"])
  return None if not (m < float("inf")) else m


def _serialize_set(p, pairs, vd, a_mean, a_worst, stop_worse, rep_worse):
  return {
    "params": dict(p),
    "nod_mean": a_mean, "nod_worst": a_worst,
    "stop_worse": stop_worse, "replica_worse": rep_worse,
    "min_gap_margin": _min_gap_margin(pairs),
    "s1_ok": vd["s1_ok"],
    "pass_strict": vd["pass_strict"], "pass_relax": vd["pass_relax"],
    "hard_strict": vd["hard_strict"], "hard_relax": vd["hard_relax"],
    "comfort": vd["comfort"],
    "pairs": [{"name": n, "tag": tag, "b1": _clean(b1), "cs": _clean(cs)}
              for n, b1, cs, tag in pairs],
  }


def worker(args):
  shard, nshards, plant_mode = args
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    grid = [g for i, g in enumerate(H.cs_grid()) if i % nshards == shard]
    out = []
    t0 = time.time()
    for k, p in enumerate(grid):
      analysis, hard, comfort = H._cs_eval_params(p, plant_mode=plant_mode, h1_relax=H.CS_EPS)
      pairs = analysis["pairs"]
      vd = _verdicts(pairs, hard, comfort)
      out.append(_serialize_set(p, pairs, vd, analysis["nod_mean"], analysis["nod_worst"],
                                analysis["stop_worse"], analysis["replica_worse"]))
      print(f"NOD-CS2-PROGRESS shard={shard} done={k + 1}/{len(grid)} "
            f"elapsed_s={time.time() - t0:.1f}", flush=True)
  return out


def _eval_sets(sets, plant_mode):
  out = []
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    for p in sets:
      analysis, hard, comfort = H._cs_eval_params(p, plant_mode=plant_mode, h1_relax=H.CS_EPS)
      pairs = analysis["pairs"]
      vd = _verdicts(pairs, hard, comfort)
      out.append(_serialize_set(p, pairs, vd, analysis["nod_mean"], analysis["nod_worst"],
                                analysis["stop_worse"], analysis["replica_worse"]))
  return out


def _histogram(results):
  by_gate = {}
  by_scen = {}
  for r in results:
    gates = set()
    for f in r["hard_strict"] + r["comfort"]:
      g = f["reason"].split(" ")[0]
      gates.add(g)
      by_scen[f["name"]] = by_scen.get(f["name"], 0) + 1
    for g in gates:
      by_gate[g] = by_gate.get(g, 0) + 1
  return {"by_gate": by_gate, "by_scenario": by_scen, "n_sets": len(results)}


def main():
  nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
  out_path = os.environ.get("BUTTER_CS_OUT", "/tmp/cs_h3.json")
  plant_mode = os.environ.get("BUTTER_CS_PLANT", "real_sg")
  t0 = time.time()

  sanity = H._cs_replica_sanity(plant_mode=plant_mode)

  grid = H.cs_grid()
  with mp.Pool(nproc) as pool:
    shards = pool.map(worker, [(i, nproc, plant_mode) for i in range(nproc)])
  results = [r for shard in shards for r in shard]
  n_strict = sum(1 for r in results if r["pass_strict"])
  n_relax = sum(1 for r in results if r["pass_relax"])

  def rank(r):
    best_margin = r["min_gap_margin"] if r["min_gap_margin"] is not None else 1e9
    return (-r["nod_mean"], best_margin)

  strict_pass = sorted([r for r in results if r["pass_strict"]], key=rank)
  relax_pass = sorted([r for r in results if r["pass_relax"]], key=rank)
  print(f"NOD-CS2-SWEEP-DONE n_combo={len(results)} strict={n_strict} relax={n_relax} "
        f"elapsed_s={time.time() - t0:.1f}", flush=True)

  # Determinism check AFTER the sweep: two fixed sets serial vs the pool results.
  det_sets = [grid[0], grid[len(grid) // 2]]
  serial = _eval_sets(det_sets, plant_mode)
  pool_by_params = {json.dumps(r["params"], sort_keys=True): r for r in results}
  det_ok = True
  for d in serial:
    key = json.dumps(d["params"], sort_keys=True)
    other = pool_by_params.get(key)
    if (other is None or other["nod_mean"] != d["nod_mean"]
        or other["hard_strict"] != d["hard_strict"] or other["comfort"] != d["comfort"]):
      det_ok = False
      print(f"NOD-CS2-DET MISMATCH {key}", flush=True)
  print(f"NOD-CS2-DET ok={det_ok} sets={len(serial)}", flush=True)

  def summarize(chosen, label):
    if chosen is None:
      return None
    return {"params": chosen["params"], "nod_mean": round(chosen["nod_mean"], 6),
            "min_gap_margin": chosen["min_gap_margin"], "label": label,
            "pairs": chosen["pairs"]}

  table = {
    "plant_real": H.PLANT_REAL,
    "long_actuator_delay": H.CS_LONG_ACTUATOR_DELAY,
    "grid": {**H._CS_GRID, "n_combo": len(results)},
    "replica_sanity": sanity,
    "determinism_ok": det_ok,
    "npassing_strict": n_strict,
    "npassing_relax": n_relax,
    "chosen_strict": summarize(strict_pass[0] if strict_pass else None, "H1-strict"),
    "chosen_relax": summarize(relax_pass[0] if relax_pass else None, "H1-5cm"),
    "histogram_strict": _histogram(results),
    "top5_strict": [{"params": r["params"], "nod_mean": round(r["nod_mean"], 6),
                     "min_gap_margin": r["min_gap_margin"],
                     "hard": r["hard_strict"][:5], "comfort": r["comfort"][:5], "s1_ok": r["s1_ok"]}
                    for r in sorted(results, key=lambda r: (not r["pass_strict"], -r["nod_mean"]))[:5]],
    "all_results": [{"params": r["params"], "nod_mean": round(r["nod_mean"], 6),
                     "nod_worst": round(r["nod_worst"], 6),
                     "min_gap_margin": r["min_gap_margin"], "s1_ok": r["s1_ok"],
                     "pass_strict": r["pass_strict"], "pass_relax": r["pass_relax"],
                     "nfail_strict": len(r["hard_strict"]) + len(r["comfort"]) + (0 if r["s1_ok"] else 1),
                     "hard_strict": r["hard_strict"], "comfort": r["comfort"]}
                    for r in results],
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  print(f"NOD-CS2-SANITY {json.dumps(sanity, sort_keys=True)}")
  for i, r in enumerate(sorted(results, key=lambda r: (not r["pass_strict"], -r["nod_mean"]))[:5], 1):
    print(f"NOD-CS2-TOP{i} nod_mean={r['nod_mean']:.5f} margin={r['min_gap_margin']} "
          f"params={r['params']} hard={len(r['hard_strict'])} comfort={len(r['comfort'])} s1={r['s1_ok']}")
  for label, chosen in (("STRICT", strict_pass[0] if strict_pass else None),
                        ("RELAX", relax_pass[0] if relax_pass else None)):
    if chosen is None:
      print(f"NOD-CS2-CHOSEN-{label} none")
      continue
    print(f"NOD-CS2-CHOSEN-{label} {chosen['params']} nod_mean={chosen['nod_mean']:.5f} "
          f"margin={chosen['min_gap_margin']}")
    for pair in chosen["pairs"]:
      if "[" in pair["name"]:
        continue
      b1, cs = pair["b1"], pair["cs"]
      print(f"NOD-CS2-{label}-ROW {pair['name']} | gap b1={b1['final_gap_m']} cs={cs['final_gap_m']} "
            f"| a_stop b1={b1['a_stop']} cs={cs['a_stop']} | nod b1={b1['nod_swing']} cs={cs['nod_swing']} "
            f"| decel b1={b1['peak_decel']} cs={cs['peak_decel']}")
  print("NOD-CS2-FINISHED", flush=True)


if __name__ == "__main__":
  main()
