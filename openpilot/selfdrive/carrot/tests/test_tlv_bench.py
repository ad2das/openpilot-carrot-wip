"""Tests for the pure gate evaluation + /proc parsing in tools/carrot_tlv_bench.py."""
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "carrot_tlv_bench.py"
_spec = importlib.util.spec_from_file_location("carrot_tlv_bench", _MODULE_PATH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _proc(cpu_avg=10.0, max2=20.0, rss_max=100.0):
  return {"cpu_pct_avg": cpu_avg, "cpu_pct_max_2s": max2, "rss_max_mb": rss_max, "rss_end_mb": rss_max,
          "rss_series_mb": [], "samples": 60}


def _summary(**overrides):
  summary = {
    "seconds": 60.0,
    "counts": {"frame_drop": 100, "model_exec": 100, "controls": 100, "camera": 100,
               "ui_draw": 100, "thermal": 100, "tlv_msgs": 100, "proc_samples": 60},
    "frame_drop_mean": 0.2, "frame_drop_p95": 0.3,
    "model_exec_p50_ms": 30.0, "model_exec_p95_ms": 35.0,
    "controls_p99_ms": 10.2, "controls_max_ms": 12.0,
    "camera_gap_rate_s": 0.5, "camera_gap_max_skip": 1, "camera_gap_max_ms": 60.0,
    "ui_draw_p95_ms": 20.0,
    "procs": {"tlvd": _proc(cpu_avg=30.0, max2=38.0, rss_max=180.0), "modeld": _proc(), "controlsd": _proc(),
              "ui": _proc(), "camerad": _proc()},
    "tlvd_cpu_duty60_pct": 30.0, "tlvd_cpu_max10_pct": 38.0,
    "cores13_busy_pct_avg": 40.0,
    "mem_available_min_mb": 2500.0,
    "thermal_status_max": 0.0, "max_temp_c_max": 60.0, "gpu_temp_c_max": 45.0,
    "tlv_infer_p95_ms": 30.0, "tlv_proc_p95_ms": 40.0,
    "tlv_status_seen": ["running"], "tlv_dropped_final": 0,
    "tlv_rss_growth_mb": 1.0,
    "pid_changes": {},
  }
  summary.update(overrides)
  return summary


GATE_NAMES = ("modeld_exec_frame_drop", "controlsd_interval_p99_max", "camera_frame_gaps", "ui_frame_time",
              "tlvd_cpu_budget", "tlvd_rss", "mem_available", "thermal")


def test_identical_phases_pass_all_eight_gates():
  summary = bench.evaluate_gates(_summary(), _summary(), _summary())
  assert summary["verdict"] == "pass"
  assert set(summary["gates"]) == set(GATE_NAMES)
  assert all(gate["pass"] for gate in summary["gates"].values())


def test_modeld_gate_fails_on_p50_p99_and_frame_drop():
  assert not bench.evaluate_gates(_summary(), _summary(model_exec_p50_ms=40.0), _summary())["gates"]["modeld_exec_frame_drop"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(model_exec_p95_ms=45.0), _summary())["gates"]["modeld_exec_frame_drop"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(frame_drop_mean=1.5), _summary())["gates"]["modeld_exec_frame_drop"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(frame_drop_p95=1.5), _summary())["gates"]["modeld_exec_frame_drop"]["pass"]


def test_modeld_gate_fails_without_data():
  empty = _summary(counts={**_summary()["counts"], "model_exec": 0, "frame_drop": 0})
  assert not bench.evaluate_gates(empty, empty, empty)["gates"]["modeld_exec_frame_drop"]["pass"]


def test_controlsd_gate_fails_on_p99_and_max():
  assert not bench.evaluate_gates(_summary(), _summary(controls_p99_ms=13.0), _summary())["gates"]["controlsd_interval_p99_max"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(controls_max_ms=18.5), _summary())["gates"]["controlsd_interval_p99_max"]["pass"]
  # small increases inside the slack are fine
  assert bench.evaluate_gates(_summary(), _summary(controls_p99_ms=12.0, controls_max_ms=16.5), _summary())["gates"]["controlsd_interval_p99_max"]["pass"]


def test_camera_gate_fails_on_gap_rate_and_worst_skip():
  assert not bench.evaluate_gates(_summary(), _summary(camera_gap_rate_s=1.5), _summary())["gates"]["camera_frame_gaps"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(camera_gap_max_skip=3), _summary())["gates"]["camera_frame_gaps"]["pass"]
  assert bench.evaluate_gates(_summary(), _summary(camera_gap_rate_s=0.9, camera_gap_max_skip=2), _summary())["gates"]["camera_frame_gaps"]["pass"]


def test_ui_gate_fails_on_draw_and_cpu():
  assert not bench.evaluate_gates(_summary(), _summary(ui_draw_p95_ms=30.0), _summary())["gates"]["ui_frame_time"]["pass"]
  bad_ui = _summary(procs={**_summary()["procs"], "ui": _proc(cpu_avg=16.0)})
  assert not bench.evaluate_gates(_summary(), bad_ui, _summary())["gates"]["ui_frame_time"]["pass"]


def test_ui_gate_unavailable_when_nothing_published():
  no_ui = _summary(counts={**_summary()["counts"], "ui_draw": 0},
                   procs={**_summary()["procs"], "ui": _proc(cpu_avg=None)})
  gate = bench.evaluate_gates(no_ui, no_ui, _summary())["gates"]["ui_frame_time"]
  assert gate["pass"] is True and gate["raw"]["available"] is False


def test_tlvd_cpu_gate_fails_over_budget_and_on_cores_delta():
  assert not bench.evaluate_gates(_summary(), _summary(tlvd_cpu_duty60_pct=45.0), _summary())["gates"]["tlvd_cpu_budget"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(tlvd_cpu_max10_pct=65.0), _summary())["gates"]["tlvd_cpu_budget"]["pass"]
  # Missing duty data in BOTH gated phases cannot pass the gate.
  no_duty = _summary(tlvd_cpu_duty60_pct=None, tlvd_cpu_max10_pct=None)
  assert not bench.evaluate_gates(_summary(), no_duty, no_duty)["gates"]["tlvd_cpu_budget"]["pass"]
  idle_tlvd = _summary(procs={**_summary()["procs"], "tlvd": _proc(cpu_avg=5.0)}, cores13_busy_pct_avg=60.0)
  assert not bench.evaluate_gates(_summary(), idle_tlvd, _summary())["gates"]["tlvd_cpu_budget"]["pass"]


def test_cpu_pct_max_window():
  points = [(0.0, 0.0), (5.0, 1.0), (10.0, 2.0), (15.0, 2.2), (20.0, 4.2)]
  # 10 s spans: 0->10 = 20%, 5->15 = 12%, 10->20 = 22% (peak window).
  assert bench.cpu_pct_max_window(points, 10.0) == pytest.approx(22.0)
  # 5 s window: 0->5 = 20%, 15->20 = 40%.
  assert bench.cpu_pct_max_window(points, 5.0) == pytest.approx(40.0)
  # No pair spans the window -> None (caller falls back).
  assert bench.cpu_pct_max_window(points, 60.0) is None
  assert bench.cpu_pct_max_window([(0.0, 0.0)], 10.0) is None
  # A counter that went backwards is skipped, not reported as negative.
  assert bench.cpu_pct_max_window([(0.0, 5.0), (5.0, 1.0), (10.0, 1.5)], 5.0) == pytest.approx(10.0)


def test_tlvd_rss_gate_fails_on_max_and_growth():
  assert not bench.evaluate_gates(_summary(), _summary(procs={**_summary()["procs"], "tlvd": _proc(rss_max=230.0)}),
                                  _summary())["gates"]["tlvd_rss"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(), _summary(tlv_rss_growth_mb=6.0))["gates"]["tlvd_rss"]["pass"]
  assert bench.evaluate_gates(_summary(), _summary(), None)["gates"]["tlvd_rss"]["pass"]  # falls back to tlv growth


def test_mem_gate_fails_below_floor_or_big_drop():
  assert not bench.evaluate_gates(_summary(), _summary(mem_available_min_mb=550.0), _summary(mem_available_min_mb=550.0))["gates"]["mem_available"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(mem_available_min_mb=2100.0), _summary())["gates"]["mem_available"]["pass"]
  assert bench.evaluate_gates(_summary(), _summary(mem_available_min_mb=2200.0), _summary())["gates"]["mem_available"]["pass"]


def test_thermal_gate_fails_on_status_temp_and_gpu():
  assert not bench.evaluate_gates(_summary(), _summary(thermal_status_max=1.0), _summary())["gates"]["thermal"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(max_temp_c_max=64.0), _summary())["gates"]["thermal"]["pass"]
  assert not bench.evaluate_gates(_summary(), _summary(gpu_temp_c_max=50.0), _summary())["gates"]["thermal"]["pass"]
  assert bench.evaluate_gates(_summary(), _summary(max_temp_c_max=62.0, gpu_temp_c_max=47.0), _summary())["gates"]["thermal"]["pass"]


def test_camera_gap_stats():
  assert bench.camera_gap_stats([]) == (None, 0, None)
  assert bench.camera_gap_stats([(0.0, 100)]) == (None, 0, None)
  frames = [(0.0, 100), (0.05, 102), (0.10, 103), (0.15, 105)]
  rate, max_skip, max_gap_ms = bench.camera_gap_stats(frames)
  assert rate == pytest.approx(2 / 0.15)
  assert max_skip == 1
  assert max_gap_ms == pytest.approx(50.0)
  rate, max_skip, _ = bench.camera_gap_stats([(0.0, 10), (0.1, 15), (0.2, 16)])
  assert max_skip == 4
  assert rate == pytest.approx(4 / 0.2)


def test_parse_pid_stat_and_statm_recorded_lines():
  line = "1234 (python) S 1 1234 1234 0 -1 4194560 0 0 0 0 10 5 0 0 20 0 1 0 0 100 5000 123 18446744073709551615"
  assert bench.parse_pid_stat(line) == (10.0, 5.0)
  weird = "999 (we ird)(name) R 1 999 999 0 -1 0 0 0 0 0 7 3 0 0 20 0 1 0 0 100 5000 0 0"
  assert bench.parse_pid_stat(weird) == (7.0, 3.0)
  assert bench.parse_pid_statm_pages("12345 6789 100 1 0 200 0\n") == 6789
  with pytest.raises(ValueError):
    bench.parse_pid_stat("no parens here")


def test_summarize_phase_from_synthetic_report():
  report = {
    "seconds": 10.0,
    "frame_drop": [0.2, 0.4],
    "model_exec_ms": [30.0, 40.0, 50.0],
    "controls_mono_ns": [0, 10_000_000, 21_000_000],
    "cam_frames": [(0.0, 100), (0.05, 102), (0.10, 103), (0.15, 105)],
    "ui_draw_ms": [],
    "thermal_status": [0.0, 0.0],
    "max_temp_c": [58.0],
    "gpu_temp_c": [41.0, 43.0],
    "tlv_infer_ms": [20.0, 30.0],
    "tlv_proc_ms": [30.0, 45.0],
    "tlv_status_seen": ["running"],
    "tlv_dropped_final": 7,
    "carrot_tlv_seen": 2,
    "proc_samples": [
      {"t": 0.0, "procs": {"tlvd": {"cpu_s": 1.0, "rss_mb": 100.0}}, "mem_available_mb": 2000.0, "cores13": (300.0, 1000.0)},
      {"t": 1.0, "procs": {"tlvd": {"cpu_s": 1.5, "rss_mb": 110.0}}, "mem_available_mb": 1900.0, "cores13": (350.0, 1400.0)},
      {"t": 2.0, "procs": {"tlvd": {"cpu_s": 2.0, "rss_mb": 105.0}}, "mem_available_mb": 2100.0, "cores13": (420.0, 1800.0)},
    ],
    "pid_changes": {"tlvd": 0},
  }
  summary = bench.summarize_phase(report)
  assert summary["counts"] == {"frame_drop": 2, "model_exec": 3, "controls": 2, "camera": 4,
                               "ui_draw": 0, "thermal": 2, "tlv_msgs": 2, "proc_samples": 3}
  assert summary["controls_max_ms"] == pytest.approx(11.0)
  assert summary["controls_p99_ms"] == pytest.approx(10.99, abs=0.05)
  assert summary["camera_gap_rate_s"] == pytest.approx(2 / 0.15)
  assert summary["camera_gap_max_skip"] == 1
  assert summary["procs"]["tlvd"]["cpu_pct_avg"] == pytest.approx(50.0)
  assert summary["procs"]["tlvd"]["cpu_pct_max_2s"] == pytest.approx(50.0)
  assert summary["procs"]["tlvd"]["rss_max_mb"] == pytest.approx(110.0)
  assert summary["cores13_busy_pct_avg"] == pytest.approx(85.0)
  assert summary["mem_available_min_mb"] == pytest.approx(1900.0)
  assert summary["thermal_status_max"] == pytest.approx(0.0)
  assert summary["gpu_temp_c_max"] == pytest.approx(43.0)
  assert summary["tlv_rss_growth_mb"] == pytest.approx(5.0)


def _stat_text():
  lines = ["cpu 0 0 0 0 0 0 0 0 0 0"]
  for i in range(8):
    total, idle = (1000 + 100 * i, 900)
    lines.append(f"cpu{i} {total - idle} 0 0 {idle} 0 0 0 0 0 0")
  return "\n".join(lines) + "\n"


def test_proc_sampler_with_fake_proc_root(tmp_path):
  root = tmp_path / "proc"
  pid_dir = root / "1234"
  pid_dir.mkdir(parents=True)
  (pid_dir / "cmdline").write_bytes(b"python\x00-m\x00openpilot.selfdrive.carrot.tlv.tlvd\x00")
  (pid_dir / "stat").write_text("1234 (python) S 1 1234 1234 0 -1 4194560 0 0 0 0 10 5 0 0 20 0 1 0 0 100 5000 123 0\n")
  (pid_dir / "statm").write_text("2000 300 100 1 0 200 0\n")
  (root / "meminfo").write_text("MemTotal: 7908000 kB\nMemAvailable: 1000000 kB\n")
  (root / "stat").write_text(_stat_text())
  sampler = bench.ProcSampler(processes={"tlvd": "openpilot.selfdrive.carrot.tlv.tlvd"},
                              proc_root=str(root), ticks_per_s=100.0, pid_refresh_s=0.0)
  sample = sampler.sample(5.0)
  assert set(sample["procs"]) == {"tlvd"}
  assert sample["procs"]["tlvd"]["cpu_s"] == pytest.approx(0.15)
  assert sample["procs"]["tlvd"]["rss_mb"] == pytest.approx(300 * 4096 / 1024 / 1024)
  assert sample["mem_available_mb"] == pytest.approx(1000000 / 1024)
  assert sample["cores13"] is not None
  # process disappears -> a new pid later counts as a change, not as a reset
  (pid_dir / "cmdline").write_bytes(b"other\x00process\x00")
  sampler._last_scan = -1e9
  assert sampler.sample(6.0)["procs"] == {}
  pid2 = root / "4321"
  pid2.mkdir()
  (pid2 / "cmdline").write_bytes(b"python\x00-m\x00openpilot.selfdrive.carrot.tlv.tlvd\x00")
  (pid2 / "stat").write_text("4321 (python) S 1 4321 4321 0 -1 0 0 0 0 0 1 1 0 0 20 0 1 0 0 100 5000 0 0\n")
  (pid2 / "statm").write_text("2000 100 100 1 0 200 0\n")
  sampler._last_scan = -1e9
  assert sampler.sample(7.0)["procs"]["tlvd"]["cpu_s"] == pytest.approx(0.02)
  assert sampler.pid_changes["tlvd"] == 1


def test_classify_episodes_rate_per_track():
  states = [
    (0.0, "none", -1),
    (0.1, "redLikely", 7), (0.2, "redLikely", 7), (0.3, "red", 7),
    (0.5, "red", 9),
    (0.6, "none", -1),
  ]
  episodes, rates = bench.classify_episodes(states)
  assert episodes == [(7, 0.1, 0.3, 3), (9, 0.5, 0.5, 1)]
  # 2 intervals / 0.2 s = 10 Hz; the single-sample episode contributes no rate.
  assert rates == [pytest.approx(10.0)]
  assert bench.classify_episodes([]) == ([], [])
  assert bench.classify_episodes([(0.0, "none", -1)]) == ([], [])


def test_time_to_confirm_measures_redlikely_to_red():
  states = [
    (0.0, "redLikely", 5), (0.2, "redLikely", 5), (0.4, "red", 5),
    (1.0, "none", -1),
    (2.0, "redLikely", 5), (2.5, "green", 5),  # ended without a red
    (3.0, "redLikely", 5), (3.1, "red", 5),
    (4.0, "redLikely", 5), (4.1, "red", 5),  # two runs
  ]
  deltas = bench.time_to_confirm(states)
  assert deltas == [pytest.approx(0.4), pytest.approx(0.1), pytest.approx(0.1)]
  assert bench.time_to_confirm([]) == []


def test_summarize_phase_includes_classifier_stats():
  report = {
    "seconds": 10.0,
    "tlv_states": [
      (0.0, "none", -1),
      (0.2, "redLikely", 3), (0.4, "redLikely", 3), (0.6, "red", 3),
      (0.8, "red", 3), (1.0, "none", -1),
    ],
    "proc_samples": [],
  }
  summary = bench.summarize_phase(report)
  assert summary["tlv_classify_episodes"] == 1
  assert summary["tlv_classify_rate_hz_median"] == pytest.approx(5.0)
  assert summary["tlv_classify_rate_hz_max"] == pytest.approx(5.0)
  assert summary["tlv_redlikely_episodes"] == 1
  assert summary["tlv_time_to_confirm_s_median"] == pytest.approx(0.4)
  assert summary["tlv_time_to_confirm_s_max"] == pytest.approx(0.4)
  # No tlv states at all -> explicit None instead of an exception.
  empty = bench.summarize_phase({"seconds": 1.0, "proc_samples": []})
  assert empty["tlv_classify_rate_hz_median"] is None
  assert empty["tlv_time_to_confirm_s_median"] is None


class _MainFakeParams:
  def __init__(self, values=None):
    self.values = dict(values or {})
    self.writes = []

  def get(self, key, block=False, return_default=False):
    return self.values.get(key)

  def get_int(self, key, block=False):
    value = self.values.get(key, 0)
    return 0 if value is None else int(value)

  def put(self, key, value, block=True):
    self.writes.append((key, value))
    self.values[key] = value


def _install_main_stubs(monkeypatch, params, sm=None):
  params_module = types.ModuleType("openpilot.common.params")
  params_module.Params = lambda: params
  monkeypatch.setitem(sys.modules, "openpilot.common.params", params_module)
  session = sm if sm is not None else types.SimpleNamespace(seen={}, update=lambda *a, **k: None)
  messaging_module = types.ModuleType("openpilot.cereal.messaging")
  messaging_module.SubMaster = lambda *a, **k: session
  monkeypatch.setitem(sys.modules, "openpilot.cereal.messaging", messaging_module)
  # `import openpilot.cereal.messaging as messaging` follows the parent package
  # attribute when the real module was already imported by another test.
  parent = importlib.import_module("openpilot.cereal")
  monkeypatch.setattr(parent, "messaging", messaging_module, raising=False)


def _patch_bench_time(monkeypatch):
  monkeypatch.setattr(bench, "time", types.SimpleNamespace(sleep=lambda *_: None,
                                                           monotonic=bench.time.monotonic))


def test_main_aborts_tlvd_phases_when_traffic_light_detect_off(monkeypatch, capsys):
  params = _MainFakeParams({"CarrotTlvDisabled": True, "TrafficLightDetectMode": 0, "CarrotTlvBackend": "null"})
  _install_main_stubs(monkeypatch, params)
  _patch_bench_time(monkeypatch)
  calls = []
  monkeypatch.setattr(bench, "_collect", lambda sm, seconds, **k: calls.append(seconds) or {"seconds": seconds})
  rc = bench.main(["--baseline-s", "0", "--tlv-s", "0", "--soak-min", "0"])
  assert rc == 2
  assert calls == [0.0]  # the baseline ran, the tlvd and soak phases were aborted
  err = capsys.readouterr().err
  assert "TrafficLightDetectMode=0" in err
  assert "tlvd will not start" in err
  assert ("CarrotTlvDisabled", False) not in params.writes  # never unlatched past the abort
  assert all(key != "CarrotTlvMode" for key, _ in params.writes)


def test_main_runs_phases_when_traffic_light_detect_enabled(monkeypatch, capsys):
  params = _MainFakeParams({"CarrotTlvDisabled": True, "TrafficLightDetectMode": 2, "CarrotTlvBackend": "null"})
  sm = types.SimpleNamespace(seen={"carrotTlv": True}, update=lambda *a, **k: None)
  _install_main_stubs(monkeypatch, params, sm=sm)
  _patch_bench_time(monkeypatch)
  calls = []
  monkeypatch.setattr(bench, "_collect", lambda sm, seconds, **k: calls.append(seconds) or {"seconds": seconds})
  monkeypatch.setattr(bench, "summarize_phase", lambda report: {"counts": {"tlv_msgs": 0}})
  monkeypatch.setattr(bench, "evaluate_gates", lambda baseline, tlv, soak: {"verdict": "pass"})
  rc = bench.main(["--baseline-s", "5", "--tlv-s", "6", "--soak-min", "0"])
  assert rc == 0
  assert calls == [5.0, 6.0]
  payload = json.loads(capsys.readouterr().out)
  assert payload["verdict"] == "pass"
  assert payload["parameters"]["original_disabled"] == "True"
  assert ("CarrotTlvDisabled", True) in params.writes  # baseline latched tilvd off
  assert ("CarrotTlvDisabled", False) in params.writes  # tlvd phase released it
  assert all(key != "CarrotTlvMode" for key, _ in params.writes)
  assert params.values["CarrotTlvDisabled"] is True  # restore wrote the original back


# -- --quick-infer (W132) -----------------------------------------------------

class _FakeQuickDetector:
  def __init__(self):
    self.calls = 0
    self.warmed = False
    self._session = types.SimpleNamespace(get_inputs=lambda: [types.SimpleNamespace(shape=[1, 3, 320, 640])])

  def warmup(self):
    self.warmed = True

  def infer(self, x):
    self.calls += 1
    return [np.zeros((1, 1), dtype=np.float32)]

  def close(self):
    pass


class _FakeQuickClassifier:
  def __init__(self):
    self.calls = 0
    self.warmed = False

  def warmup(self):
    self.warmed = True

  def classify(self, x):
    self.calls += 1
    return np.zeros((4, 8), dtype=np.float32)

  def close(self):
    pass


def test_quick_infer_never_touches_params_and_prints_runtime(monkeypatch, capsys):
  import numpy as np
  det = _FakeQuickDetector()
  cls = _FakeQuickClassifier()
  builds = []

  def fake_build(build):
    builds.append(build)
    return "onnx", det, cls, ""

  import openpilot.selfdrive.carrot.tlv.tlvd as tlvd
  monkeypatch.setattr(bench, "_quick_infer_runtime", fake_build)
  monkeypatch.setattr(tlvd, "apply_cpu_policy", lambda: ["affinity=1,2,3"])
  monkeypatch.setattr(tlvd, "resolve_backend_name", lambda value, is_tici: "onnx")
  monkeypatch.setattr(tlvd, "_host_is_tici", lambda: False)

  rc = bench.quick_infer()
  assert rc == 0
  assert det.calls == 60 and cls.calls == 60
  assert builds and builds[0][0] == "onnx"
  out = capsys.readouterr().out
  assert "quick-infer" in out
  assert "runtime=onnx" in out
  assert "detector cpu_mean=" in out
  assert "lamp cpu_mean=" in out
  assert "peak_rss=" in out


def test_quick_infer_onnx_detector_shape_uses_session_when_onnx(monkeypatch, capsys):
  """The onnx path reads the real input shape from the session; keep that path
  exercised with a fake session so no model file is needed."""
  import numpy as np
  det = _FakeQuickDetector()
  det._session = types.SimpleNamespace(get_inputs=lambda: [types.SimpleNamespace(shape=[1, 3, 320, 640])])
  cls = _FakeQuickClassifier()

  monkeypatch.setattr(bench, "_quick_infer_runtime", lambda build: ("onnx", det, cls, ""))
  import openpilot.selfdrive.carrot.tlv.tlvd as tlvd
  monkeypatch.setattr(tlvd, "apply_cpu_policy", lambda: [])
  monkeypatch.setattr(tlvd, "resolve_backend_name", lambda value, is_tici: "onnx")
  monkeypatch.setattr(tlvd, "_host_is_tici", lambda: False)

  assert bench.quick_infer() == 0
  assert det.calls == 60
