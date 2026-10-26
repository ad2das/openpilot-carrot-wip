"""Compile and run actual Panda hooks, without Python/CFFI or vehicle access."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


def run_compiled_probe(trace=None, *, allow_debug=True):
  """Run real hooks, optionally replaying runtime frames through stdin.

  Each trace line: time_us controls R|T address_hex bus flags data_hex expected.
  Include CGW1 and ECU replies as R events; runtime commands are T events.
  flags: fd=1, extended=2, returned=4, rejected=8. A rejected expectation raises.
  Synthetic responses test software compatibility, not ECU or lamp operation.
  """
  safety = Path(__file__).resolve().parents[1]
  with tempfile.TemporaryDirectory(prefix="hyundai_atc_c_") as work:
    output = Path(work) / "hyundai_atc_probe"
    windows_wsl = os.name == "nt" and shutil.which("gcc") is None and shutil.which("clang") is None

    def host_path(path):
      path = path.resolve()
      return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}" if windows_wsl else str(path)

    prefix = ["wsl", "-d", "Ubuntu-24.04", "--"] if windows_wsl else []
    compiler = "gcc" if windows_wsl else shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
      raise RuntimeError("A C compiler is required; this safety test must not silently skip")
    command = prefix + [compiler, "-std=gnu11", "-O1", "-Wall", "-Werror", "-Wno-pointer-to-int-cast",
                        "-fsanitize=undefined", "-fno-sanitize-recover=undefined",
                        "-I" + host_path(safety), "-I" + host_path(safety / "board"),
                        host_path(safety / "tests/hyundai_atc_probe.c"), "-o", host_path(output)]
    # fake_stm.h defines ALLOW_DEBUG, so the release probe explicitly removes it.
    compiled = subprocess.run(command + ([] if allow_debug else ["-DHYUNDAI_ATC_PROBE_RELEASE"]),
                              capture_output=True, text=True, timeout=120)
    if compiled.returncode:
      raise RuntimeError(compiled.stdout + compiled.stderr)
    executed = subprocess.run(prefix + [host_path(output)] + (["--replay"] if trace is not None else []),
                              input=trace, capture_output=True, text=True, timeout=30)
    if executed.returncode:
      raise RuntimeError(executed.stdout + executed.stderr)
    return executed.stdout


class TestHyundaiAtcCompiled(unittest.TestCase):
  def test_actual_c_hooks(self):
    for debug in (False, True):
      with self.subTest(allow_debug=debug):
        result = run_compiled_probe(allow_debug=debug)
        self.assertIn("HYUNDAI_ATC_ACTUAL_HOOK_CHECKS=", result)
        print(f"ALLOW_DEBUG={debug}: {result.splitlines()[-1]}")

  def test_replay_interface(self):
    # Positive sequence plus a wrong-side duplicate ON rejection.
    trace = "\n".join((
      "1000000 1 R 541 0 0 0000000000000000 1",
      "1000000 1 T 770 0 0 0210030000000000 1",
      "1010000 1 R 778 0 0 025003aaaaaaaaaa 1",
      "1010000 1 T 770 0 0 042fbc1500000000 1",
      "1020000 1 R 778 0 0 046fbc1500aaaaaa 1",
      "1020000 1 T 770 0 0 042fbc1600000000 1",
      "1030000 1 R 778 0 0 046fbc1600aaaaaa 1",
      "1030000 1 T 770 0 0 042fbc1503000000 1",
      "1030000 1 T 770 0 0 042fbc1603000000 0",
      "1040000 0 T 770 0 0 042fbc1500000000 1",
      "1040000 0 T 770 0 0 042fbc1600000000 1",
    ))
    self.assertIn("HYUNDAI_ATC_REPLAY_EVENTS=11", run_compiled_probe(trace))


if __name__ == "__main__":
  unittest.main()
