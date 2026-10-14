#!/usr/bin/env python3
"""Spawn-safe launcher for tlvd (manager process "tlvd").

Why this module exists: enabling tlvd mid-drive used to fork the running,
multi-threaded manager inside PythonProcess.start(). The forked child could
block while manager threads held locks (Python at-fork handlers), which froze
the manager itself - no managerState, every later process start skipped
(field: route 0000043e segment 8, reproduced twice). process_config starts
this launcher with spawn=True instead: a fresh interpreter (fork+exec inside
_posixsubprocess, no Python at-fork handlers, no inherited manager state).

Module level MUST stay stdlib-only: no numpy, no openpilot.selfdrive.carrot.tlv
imports, no os.environ writes. The manager preimports every process module
(process.py:181-187, manager.py:194-195) and anything imported here would leak
into the manager and every process it starts later.

main() order is load-bearing: env -> CPU policy -> OOM score -> import tlvd ->
tlvd.main(). tlvd pulls numpy (through backends) and must find both the thread
caps and DEV=CPU already in the environment.
"""
from __future__ import annotations

import os

# Thread caps + CPU device, before anything imports numpy/OpenBLAS/MKL or
# tinygrad (tlvd's import chain). Same values the old tlvd module-level block
# set (tlvd.py W99); DEV=CPU overrides any inherited QCOM/CUDA value so
# tinygrad's Device.DEFAULT probe (tinygrad_repo/tinygrad/device.py:14,45-51)
# cannot open the QCOM GPU next to modeld.
TLVD_ENV = {
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "MKL_NUM_THREADS": "1",
  "NUMEXPR_NUM_THREADS": "1",
  "VECLIB_MAXIMUM_THREADS": "1",
  "DEV": "CPU",
}


def set_process_env() -> None:
  """Thread caps + DEV=CPU for the tlvd child, before numpy/tinygrad load."""
  for key, value in TLVD_ENV.items():
    os.environ[key] = value


def apply_cpu_policy() -> list[str]:
  """Cores 1-3 + SCHED_IDLE + nice19 on POSIX; silently skipped on Windows.

  Duplicated from tlvd.apply_cpu_policy: tlvd must not be imported before the
  policy is applied (its import chain is what the policy is meant to cover).
  tlvd.main() re-applies it idempotently for direct runs.
  """
  applied: list[str] = []
  try:
    os.sched_setaffinity(0, {1, 2, 3})
    applied.append("affinity=1,2,3")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    applied.append("sched=SCHED_IDLE")
  except (AttributeError, OSError, ValueError):
    pass
  try:
    os.nice(19)
    applied.append("nice=19")
  except (AttributeError, OSError, ValueError):
    pass
  return applied


def write_oom_score_adj(value: int = 1000, path: str = "/proc/self/oom_score_adj") -> bool:
  """Ask the kernel to OOM-kill tlvd before anything else. Best effort.

  Stdlib-only mirror of resources.write_oom_score_adj (resources.py:47-54,
  checked: resources.py:15-20 imports only os/time/collections/dataclasses, so
  it would be import-safe too - duplicated anyway to keep the launcher free of
  openpilot imports until step (d)).
  """
  try:
    with open(path, "w") as f:
      f.write(str(int(value)))
    return True
  except OSError:
    return False


def _load_tlvd():
  """Import tlvd only AFTER the env/policy/OOM steps above (numpy, tinygrad)."""
  from openpilot.selfdrive.carrot.tlv import tlvd
  return tlvd


def main() -> None:
  set_process_env()
  print(f"tlvd_launch: cpu-policy={apply_cpu_policy()}", flush=True)
  if not write_oom_score_adj(1000):
    print("tlvd_launch: oom_score_adj not writable (non-Linux or missing /proc)", flush=True)
  _load_tlvd().main()


if __name__ == "__main__":
  main()
