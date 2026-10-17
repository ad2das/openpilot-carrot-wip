"""LongActuatorDelay sweep driver (16 workers, WSL openpilot venv).

Delays are independent, so each delay is one work unit; the 16-process pool maps
delays across processes (each delay reuses its own B1 cache inside the worker).
Writes the JSON table and prints NOD-AD-PROGRESS per finished delay.

Usage: BUTTER_AD_OUT=... python ad_sweep_driver.py [nproc]
"""
import json
import os
import sys
import time
import multiprocessing as mp

from openpilot.common.prefix import OpenpilotPrefix
import openpilot.selfdrive.test.longitudinal_maneuvers.test_butter_nod_plant as H


def _clean(m):
  return dict(m)


def _serialize(delay, pairs, hard, comfort, nod_reds, nod_mean):
  return {
    "delay": delay,
    "hard": hard,
    "comfort": comfort,
    "nod_mean": nod_mean,
    "nod_reductions": [[n, r] for n, r in nod_reds],
    "passes": (not hard) and (not comfort),
    "pairs": [{"name": n, "group": b1.get("group"), "class": cls,
               "b1": _clean(b1), "ad": _clean(ad)} for n, b1, ad, cls in pairs],
  }


def worker(args):
  delay, specs_blob, nshards = args
  with OpenpilotPrefix(create_dirs_on_enter=True, clean_dirs_on_exit=True):
    specs = H._ad_build_specs()
    cache = H.ad_b1_cache(specs)
    pairs, hard, comfort = H._ad_eval_delay(specs, cache, delay)
    reds, mean = H._ad_nod_reductions(pairs)
    print(f"NOD-AD-PROGRESS delay={delay} done=1/1 n_pairs={len(pairs)} "
          f"hard={len(hard)} comfort={len(comfort)}", flush=True)
    return [_serialize(delay, pairs, hard, comfort, reds, mean)]


def main():
  nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 16
  out_path = os.environ.get("BUTTER_AD_OUT", "/tmp/butter_ad.json")
  t0 = time.monotonic()

  delays = H.AD_DELAYS
  with mp.Pool(min(nproc, len(delays))) as pool:
    shards = pool.map(worker, [(d, None, 1) for d in delays])
  results = [r for s in shards for r in s]
  results.sort(key=lambda r: r["delay"])

  passing_all = [r for r in results if r["passes"]]
  # Largest delay that passes hard gates + S2/S3/S5/S6 (S1 informational only).
  largest = passing_all[-1]["delay"] if passing_all else None

  table = {
    "plant_real": H.PLANT_REAL,
    "b1_delay": H.AD_B1_DELAY,
    "delays": delays,
    "user_params": H.AD_USER_PARAMS,
    "results": [
      {"delay": r["delay"], "nod_mean": round(r["nod_mean"], 5), "passes": r["passes"],
       "hard": r["hard"], "comfort": r["comfort"],
       "nod_reductions": [[n, round(x, 5)] for n, x in r["nod_reductions"]],
       "pairs": r["pairs"]} for r in results],
    "largest_passing": largest,
    "note": "desktop closed-loop simulation only; no vehicle validation",
  }
  out_dir = os.path.dirname(out_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as handle:
    handle.write(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=False) + "\n")

  print(f"NOD-AD-SWEEP n_delays={len(results)} n_passing={len(passing_all)} nproc={nproc} "
        f"elapsed_s={time.monotonic() - t0:.1f}")
  for r in results:
    print(f"NOD-AD-RESULT delay={r['delay']} nod_mean={r['nod_mean']:.4f} "
          f"passes={r['passes']} hard={len(r['hard'])} comfort={len(r['comfort'])}")
  print(f"NOD-AD-LARGEST {largest}")
  print("NOD-AD-FINISHED", flush=True)


if __name__ == "__main__":
  main()
