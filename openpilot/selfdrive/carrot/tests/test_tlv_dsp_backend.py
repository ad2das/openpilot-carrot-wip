"""WSNPE13: the Hexagon DSP detector backend (persistent SNPE server process).

The fake server below is a small python script that speaks the real protocol
(READY / OK / ERR over stdout, RUN over stdin) and exchanges bytes through the
same /dev/shm files the real C++ server uses, so these tests exercise the
backend's protocol, timeout, fallback, budget and cleanup paths without SNPE,
qemu or a device.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import textwrap
import time

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import backends

IN_BYTES = 320 * 640 * 3
OUT_FLOATS = 273000


def _heads_spec() -> dict:
  with open(backends.DSP_HEADS, encoding="utf-8") as f:
    return json.load(f)


def _ramp(seq: int) -> np.ndarray:
  """The fake output the backend must hand back through the head split."""
  return np.linspace(0.0, 1.0, OUT_FLOATS, dtype=np.float32) + np.float32(seq)


def _write_server(tmp_path, body: str) -> str:
  path = tmp_path / "fake_snpe_server.py"
  path.write_text(textwrap.dedent(body), encoding="utf-8")
  return str(path)


_HEADER = """\
import os
import sys
import time
import numpy as np

args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
in_path, out_path = args["--in"], args["--out"]
out = np.memmap(out_path, dtype=np.float32, mode="r+", shape=(273000,))
inp = np.memmap(in_path, dtype=np.uint8, mode="r", shape=(614400,))
"""


def _fake_server(tmp_path, *, ready=(IN_BYTES, OUT_FLOATS * 4), pre_ready="",
                 sleep_on=(), err_on=(), exit_after_ready=False) -> str:
  body = _HEADER + f"""
{pre_ready}
sys.stdout.write("READY {ready[0]} {ready[1]}\\n")
sys.stdout.flush()
if {exit_after_ready!r}:
  sys.exit(0)
for line in sys.stdin:
  line = line.strip()
  if not line.startswith("RUN "):
    continue
  seq = int(line.split()[1])
  if seq in {list(sleep_on)!r}:
    time.sleep(1.0)
  if seq in {list(err_on)!r}:
    sys.stdout.write(f"ERR {{seq}} boom\\n")
    sys.stdout.flush()
    continue
  assert inp.shape == (614400,)
  with np.errstate(invalid="ignore"):
    out[:] = np.linspace(0.0, 1.0, 273000, dtype=np.float32) + np.float32(seq)
  sys.stdout.write(f"OK {{seq}} 1234\\n")
  sys.stdout.flush()
"""
  return _write_server(tmp_path, body)


class FakeFallback:
  def __init__(self) -> None:
    self.calls = 0
    self.warmups = 0
    self.closed = 0

  def warmup(self) -> None:
    self.warmups += 1

  def infer(self, model_input):
    self.calls += 1
    return ["fallback"]

  def close(self) -> None:
    self.closed += 1


def _backend(tmp_path, server: str, monkeypatch, *, fallback=None):
  fallback = fallback or FakeFallback()
  monkeypatch.setattr(backends, "DSP_CACHE_ROOT", str(tmp_path / "cache"))
  monkeypatch.setattr(backends, "_dsp_shm_dir", lambda: str(tmp_path))
  backend = backends.DspBackend(backends.BUNDLED_DETECTOR, server_cmd=server,
                                command_prefix=[sys.executable],
                                fallback_factory=lambda: fallback)
  return backend, fallback


def _zero_frame() -> np.ndarray:
  return np.zeros((1, 3, 320, 640), dtype=np.uint8)


def _assert_on_fallback(backend, fallback, reason="") -> None:
  assert backend.active == "cpu_fallback"
  assert fallback.warmups == 1
  if reason:
    assert reason in backend.fallback_reason
  assert backend._proc is None


# ---------------------------------------------------------------------------
# D1-D8: the backend protocol
# ---------------------------------------------------------------------------

def test_d1_happy_path_returns_six_heads_in_canon_order(tmp_path, monkeypatch):
  server = _fake_server(tmp_path)
  backend, fallback = _backend(tmp_path, server, monkeypatch)
  try:
    backend.warmup()
    assert backend.active == "dsp"
    assert fallback.calls == 0
    spec = _heads_spec()["heads"]
    outs = backend.infer(_zero_frame())
    assert isinstance(outs, list) and len(outs) == 6
    for name, arr in zip(backends.DSP_HEAD_ORDER, outs):
      assert arr.shape == tuple(spec[name]["dims"])
      assert arr.dtype == np.float32
    concat = np.concatenate([a.reshape(-1) for a in outs])
    assert np.array_equal(concat, _ramp(2))   # warmup already consumed seq 1
    for seq in (3, 4):
      outs = backend.infer(_zero_frame())
      assert np.array_equal(np.concatenate([a.reshape(-1) for a in outs]), _ramp(seq))
    assert backend.active == "dsp"
  finally:
    backend.close()


def test_d2_err_reply_switches_to_the_fallback_for_good(tmp_path, monkeypatch):
  server = _fake_server(tmp_path, err_on=(3,))
  backend, fallback = _backend(tmp_path, server, monkeypatch)
  try:
    backend.warmup()                      # seq 1
    assert backend.active == "dsp"
    backend.infer(_zero_frame())          # seq 2: OK
    got = backend.infer(_zero_frame())    # seq 3: ERR -> fallback for this frame
    assert got == ["fallback"]
    _assert_on_fallback(backend, fallback, "ERR")
    assert fallback.calls == 1
    assert backend.infer(_zero_frame()) == ["fallback"]
    assert fallback.calls == 2
    assert backend._proc is None
  finally:
    backend.close()


def test_d3_slow_reply_falls_back_within_the_frame_timeout(tmp_path, monkeypatch):
  # Seq 1 (the warmup frame) answers immediately; seq 2 stalls.
  server = _fake_server(tmp_path, sleep_on=(2,))
  backend, fallback = _backend(tmp_path, server, monkeypatch)
  try:
    backend.warmup()
    assert backend.active == "dsp"
    # Shorten the frame timeout so the 1 s stall is decisive; the shipped value
    # (0.5 s) is what the device runs with, and the qemu rehearsal raises it.
    monkeypatch.setattr(backends, "DSP_FRAME_TIMEOUT_S", 0.2)
    started = time.monotonic()
    got = backend.infer(_zero_frame())
    elapsed = time.monotonic() - started
    assert got == ["fallback"]
    assert elapsed <= 0.7, elapsed
    _assert_on_fallback(backend, fallback, "no reply")
    assert fallback.calls == 1
  finally:
    backend.close()


def test_d4_init_error_ends_warmup_on_the_fallback(tmp_path, monkeypatch):
  path = _write_server(tmp_path, """
import sys
sys.stdout.write("ERR init runtime_unavailable\\n")
sys.stdout.flush()
sys.exit(2)
""")
  backend, fallback = _backend(tmp_path, path, monkeypatch)
  try:
    backend.warmup()
    _assert_on_fallback(backend, fallback, "runtime_unavailable")
  finally:
    backend.close()


def test_d5_eof_after_ready_falls_back_on_the_first_infer(tmp_path, monkeypatch):
  server = _fake_server(tmp_path, exit_after_ready=True)
  backend, fallback = _backend(tmp_path, server, monkeypatch)
  try:
    backend.warmup()   # READY, then the warmup infer hits EOF
    _assert_on_fallback(backend, fallback, "no reply")
  finally:
    backend.close()


def test_d6_wrong_ready_sizes_fall_back_at_warmup(tmp_path, monkeypatch):
  server = _fake_server(tmp_path, ready=(IN_BYTES, 4096))
  backend, fallback = _backend(tmp_path, server, monkeypatch)
  try:
    backend.warmup()
    _assert_on_fallback(backend, fallback, "READY sizes")
    assert str(4096) in backend.fallback_reason
  finally:
    backend.close()


def test_d7_bad_input_shape_is_a_caller_error_not_a_fallback(tmp_path, monkeypatch):
  server = _fake_server(tmp_path)
  backend, fallback = _backend(tmp_path, server, monkeypatch)
  try:
    backend.warmup()
    with pytest.raises(ValueError, match="uint8"):
      backend.infer(np.zeros((1, 3, 320, 640), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
      backend.infer(np.zeros((1, 3, 320, 641), dtype=np.uint8))
    assert backend.active == "dsp"
    assert fallback.calls == 0
    assert backend.infer(_zero_frame())[0].shape == (1, 64, 40, 80)
  finally:
    backend.close()


def test_d8_close_unlinks_the_shm_files_and_leaves_no_child(tmp_path, monkeypatch):
  server = _fake_server(tmp_path)
  backend, fallback = _backend(tmp_path, server, monkeypatch)
  backend.warmup()
  paths = (backend._in_path, backend._out_path)
  assert all(os.path.isfile(p) for p in paths)
  proc = backend._proc
  assert proc is not None and proc.poll() is None
  backend.close()
  assert not any(os.path.exists(p) for p in paths)
  assert backend._proc is None
  assert proc.poll() is not None          # the server is gone, not orphaned
  backend.close()  # idempotent


# ---------------------------------------------------------------------------
# D9: the build cache
# ---------------------------------------------------------------------------

def test_d9_ensure_dsp_server_builds_once_and_validates(tmp_path, monkeypatch):
  monkeypatch.setattr(backends, "DSP_CACHE_ROOT", str(tmp_path / "cache"))
  calls = []
  expected = backends.dsp_build_argv("/usr/bin/clang++", backends.DSP_SERVER_SRC,
                                     os.path.join(backends.SNPE_ROOT, "include"),
                                     os.path.join(backends.SNPE_ROOT, "larch64"),
                                     backends.dsp_server_path() + ".tmp")

  class Proc:
    returncode = 0
    stderr = b""

  def fake_run(argv, **kwargs):
    calls.append(list(argv))
    assert kwargs.get("timeout") == backends.DSP_BUILD_TIMEOUT_S
    out = argv[argv.index("-o") + 1]
    with open(out, "wb") as f:
      f.write(b"\x7fELF fake")
    os.chmod(out, 0o755)
    return Proc()

  first = backends.ensure_dsp_server(run=fake_run, which=lambda name: "/usr/bin/" + name)
  assert calls == [expected]
  assert os.path.isfile(first) and os.access(first, os.X_OK)
  assert first == backends.dsp_server_path()
  # The hash covers the source bytes and the flags, never a path.
  assert hashlib.sha256(b"tlv_snpe_server.cc").hexdigest()[:12] not in first
  assert os.path.basename(os.path.dirname(first)) == backends._dsp_server_hash()

  second = backends.ensure_dsp_server(run=fake_run, which=lambda name: "/usr/bin/" + name)
  assert second == first
  assert len(calls) == 1  # cached: no second compile

  real_src = backends.DSP_SERVER_SRC
  before = backends.dsp_server_path()
  src = tmp_path / "tlv_snpe_server.cc"
  src.write_text("// changed source\n", encoding="utf-8")
  monkeypatch.setattr(backends, "DSP_SERVER_SRC", str(src))
  assert backends.dsp_server_path() != before

  monkeypatch.setattr(backends, "DSP_SERVER_SRC", real_src)

  class Fail:
    returncode = 1
    stderr = b"error: something in the tail\n"

  fresh = tmp_path / "cache2"
  monkeypatch.setattr(backends, "DSP_CACHE_ROOT", str(fresh))
  with pytest.raises(backends.BackendError, match="error: something in the tail"):
    backends.ensure_dsp_server(run=lambda argv, **kw: Fail(), which=lambda name: "/usr/bin/clang++")

  with pytest.raises(backends.BackendError, match="clang"):
    backends.ensure_dsp_server(run=fake_run, which=lambda name: None)


# ---------------------------------------------------------------------------
# D10/D11: factories
# ---------------------------------------------------------------------------

def test_d10_dsp_backend_refuses_an_off_bundle_model(tmp_path, monkeypatch):
  monkeypatch.setattr(backends, "_dsp_shm_dir", lambda: str(tmp_path))
  other = tmp_path / "custom.onnx"
  other.write_bytes(b"x")
  with pytest.raises(backends.BackendError, match="bundled detector"):
    backends.DspBackend(str(other))
  with pytest.raises(backends.BackendError, match="bundled detector"):
    backends.make_backend("dsp", str(other))


def test_d11_dsp_classifier_is_the_onnx_cpu_lamp(tmp_path):
  pytest.importorskip("onnxruntime")
  classifier = backends.make_classifier("dsp", backends.BUNDLED_LAMP)
  assert isinstance(classifier, backends.OnnxCpuClassifier)
  assert classifier.name == "onnx"
  classifier.close()


# ---------------------------------------------------------------------------
# D12-D14: tlvd wiring
# ---------------------------------------------------------------------------

def test_d12_resolve_backend_name_defaults_dsp_on_tici():
  from openpilot.selfdrive.carrot.tlv import tlvd
  assert tlvd.resolve_backend_name("", True) == "dsp"
  assert tlvd.resolve_backend_name("", False) == "onnx"
  assert tlvd.resolve_backend_name("onnx", True) == "onnx"
  assert tlvd.resolve_backend_name("onnx", False) == "onnx"
  assert tlvd.resolve_backend_name("dsp", False) == "dsp"
  assert tlvd.resolve_backend_name(" null ", True) == "null"
  assert tlvd.resolve_backend_name("tinygrad", True) == "tinygrad"


def test_d13_budget_charges_the_server_cpu_delta(monkeypatch):
  from openpilot.selfdrive.carrot.tlv import resources, tlvd

  child = {"cpu": 5.0}

  class ChildBackend:
    """The one shape the budget hook needs: child_cpu_seconds()."""

    def child_cpu_seconds(self):
      return child["cpu"]

    def warmup(self):
      pass

    def close(self):
      pass

  captured = {}
  real_budget = resources.CpuCreditBudget

  class RecordingBudget(real_budget):
    def __init__(self, **kwargs):
      captured.update(kwargs)
      super().__init__(**kwargs)

  monkeypatch.setattr(resources, "os_cpu_seconds", lambda: 0.0)  # tlvd CPU only
  monkeypatch.setattr(resources, "CpuCreditBudget", RecordingBudget)
  monkeypatch.setattr(backends, "make_backend", lambda name, path, decoder=None: (ChildBackend(), None))
  monkeypatch.setattr(backends, "make_classifier", lambda name, path: backends.NullClassifier())
  monkeypatch.setattr(tlvd, "_host_is_tici", lambda: True)
  try:
    runner = tlvd.TlvRunner(_Params({"CarrotTlvBackend": "dsp"}))
  finally:
    monkeypatch.setattr(resources, "CpuCreditBudget", real_budget)
  assert runner.runtime_name == "dsp"
  assert "cpu_clock" in captured  # tlvd wires the child-CPU hook into the budget

  def _credit_for(child_jump: float) -> float:
    child["cpu"] = 5.0
    wall = {"t": 0.0}
    budget = real_budget(cpu_clock=captured["cpu_clock"], clock=lambda: wall["t"])
    wall["t"] = 1.0              # 1 s of wall: 0.4 s of refill at CPU_BUDGET
    child["cpu"] = 5.0 + child_jump
    budget.tick(1.0)
    return budget.credit

  assert _credit_for(0.0) - _credit_for(2.0) == pytest.approx(2.0, abs=1e-6)


def test_d14_dsp_build_failure_retries_onnx_then_tinygrad(monkeypatch):
  from openpilot.selfdrive.carrot.tlv import tlvd

  def _run(dsp_ok, onnx_ok, tinygrad_ok):
    calls = []

    def fake_make_backend(name, model_path="", decoder=None):
      calls.append(name)
      if name == "dsp" and not dsp_ok:
        raise backends.BackendError("no clang++")
      if name == "onnx" and not onnx_ok:
        raise backends.BackendError("onnxruntime is not available")
      if name == "tinygrad" and not tinygrad_ok:
        raise backends.BackendError("tinygrad boom")
      return _Nop(name), None

    def fake_make_classifier(name, model_path=""):
      return _Nop("cls")

    monkeypatch.setattr(backends, "make_backend", fake_make_backend)
    monkeypatch.setattr(backends, "make_classifier", fake_make_classifier)
    monkeypatch.setattr(tlvd, "_host_is_tici", lambda: True)
    runner = tlvd.TlvRunner(_Params({"CarrotTlvBackend": "dsp"}))
    return runner, calls

  runner, calls = _run(dsp_ok=False, onnx_ok=True, tinygrad_ok=True)
  assert runner.runtime_name == "onnx"
  assert runner.backend_error == ""
  assert calls[:2] == ["dsp", "onnx"]
  assert "tinygrad" not in calls

  runner, calls = _run(dsp_ok=False, onnx_ok=False, tinygrad_ok=True)
  assert runner.runtime_name == "tinygrad"
  assert runner.backend_error == ""
  assert calls[:3] == ["dsp", "onnx", "tinygrad"]


class _Nop:
  def __init__(self, name):
    self.name = name
    self.output_names = ()
    self.enabled = True

  def warmup(self):
    pass

  def close(self):
    pass

  def classify(self, crops_input):
    return None


class _Params:
  def __init__(self, values):
    self._values = dict(values)

  def get(self, key, block=False, return_default=False):
    return self._values.get(key)

  def get_int(self, key, block=False):
    value = self._values.get(key, 0)
    return 0 if value is None else int(value)

  def get_bool(self, key, block=False):
    return bool(self._values.get(key, False))

  def put(self, key, value, block=True):
    self._values[key] = value


def test_d15_stale_shm_files_of_dead_tlvds_are_removed(tmp_path, monkeypatch):
  import subprocess
  dead = subprocess.Popen([sys.executable, "-c", "pass"])
  dead.wait()
  live = os.getppid()
  for pid in (dead.pid, live):
    for side in ("in", "out"):
      (tmp_path / f"tlv_snpe_{pid}_{side}").write_bytes(b"x")
  (tmp_path / "tlv_snpe_notapid_in").write_bytes(b"x")
  (tmp_path / "other_file").write_bytes(b"x")

  removed = backends._remove_stale_dsp_shm(str(tmp_path), os.getpid())

  assert sorted(removed) == [f"tlv_snpe_{dead.pid}_in", f"tlv_snpe_{dead.pid}_out"]
  assert (tmp_path / f"tlv_snpe_{live}_in").exists()
  assert (tmp_path / f"tlv_snpe_{live}_out").exists()
  assert (tmp_path / "tlv_snpe_notapid_in").exists()
  assert (tmp_path / "other_file").exists()

  # the backend runs the sweep before creating its own files
  (tmp_path / f"tlv_snpe_{dead.pid}_in").write_bytes(b"x")
  backend, _fallback = _backend(tmp_path, _fake_server(tmp_path), monkeypatch)
  try:
    assert not (tmp_path / f"tlv_snpe_{dead.pid}_in").exists()
    assert (tmp_path / f"tlv_snpe_{os.getpid()}_in").stat().st_size == IN_BYTES
  finally:
    backend.close()


def test_d16_a_backend_that_fails_its_build_step_is_closed(monkeypatch):
  from openpilot.selfdrive.carrot.tlv import tlvd

  built = []

  class _Closing(_Nop):
    def __init__(self, name):
      super().__init__(name)
      self.closed = 0
      built.append(self)

    def close(self):
      self.closed += 1

  def fake_make_backend(name, model_path="", decoder=None):
    return _Closing(name), None

  def fake_make_classifier(name, model_path=""):
    if name == "dsp":
      raise backends.BackendError("lamp boom")
    return _Nop("cls")

  monkeypatch.setattr(backends, "make_backend", fake_make_backend)
  monkeypatch.setattr(backends, "make_classifier", fake_make_classifier)
  monkeypatch.setattr(tlvd, "_host_is_tici", lambda: True)
  runner = tlvd.TlvRunner(_Params({"CarrotTlvBackend": "dsp"}))

  assert [b.name for b in built] == ["dsp", "onnx"]
  assert built[0].closed == 1  # the dsp backend released its server and shm
  assert built[1].closed == 0  # the onnx backend is the one in use
  assert runner.runtime_name == "onnx"
