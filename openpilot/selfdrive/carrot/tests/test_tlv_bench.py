"""Tests for the pure gate evaluation in tools/carrot_tlv_bench.py."""
import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "carrot_tlv_bench.py"
_spec = importlib.util.spec_from_file_location("carrot_tlv_bench", _MODULE_PATH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def report(*, fd=(0.2, 0.2, 0.2), exec_ms=(30.0, 31.0, 32.0), ui=(20.0, 21.0), cam_pt=(10.0, 11.0),
           sof=(50.0, 51.0), thermal=(0.0, 0.0), temp=(60.0, 61.0), infer=(20.0, 25.0), proc=(40.0, 45.0)):
  return {
    "frame_drop": list(fd), "model_exec_ms": list(exec_ms), "ui_draw_ms": list(ui),
    "cam_processing_ms": list(cam_pt), "cam_sof_gap_ms": list(sof),
    "thermal_status": list(thermal), "max_temp_c": list(temp),
    "tlv_infer_ms": list(infer), "tlv_proc_ms": list(proc),
  }


def test_identical_runs_pass():
  summary = bench.summarize(report(), report())
  assert summary["verdict"] == "pass"
  assert all(gate["pass"] for gate in summary["gates"].values())
  assert summary["baseline_clean"] is True


def test_frame_drop_regression_fails():
  summary = bench.summarize(report(), report(fd=(1.5, 1.5, 1.5)))
  assert summary["verdict"] == "fail"
  assert not summary["gates"]["model_frame_drop_unchanged"]["pass"]


def test_model_exec_regression_fails():
  summary = bench.summarize(report(), report(exec_ms=(40.0, 42.0)))
  assert not summary["gates"]["model_exec_p95_unchanged"]["pass"]


def test_ui_gate_skips_when_not_published_but_fails_on_regression():
  skipped = bench.summarize(report(ui=()), report(ui=()))
  assert skipped["gates"]["ui_frame_time_unchanged"] == {"pass": True, "available": False}
  failed = bench.summarize(report(), report(ui=(30.0, 31.0)))
  assert not failed["gates"]["ui_frame_time_unchanged"]["pass"]


def test_camerad_regression_fails():
  summary = bench.summarize(report(), report(cam_pt=(20.0, 22.0)))
  assert not summary["gates"]["camerad_timing_unchanged"]["pass"]
  summary = bench.summarize(report(), report(sof=(50.0, 80.0)))
  assert not summary["gates"]["camerad_timing_unchanged"]["pass"]


def test_thermal_regression_fails():
  assert bench.summarize(report(), report(thermal=(0.0, 1.0)))["gates"]["thermal_no_regression"]["pass"] is False
  assert bench.summarize(report(), report(temp=(60.0, 70.0)))["gates"]["thermal_no_regression"]["pass"] is False


def test_tlvd_budget_gate():
  ok = bench.summarize(report(), report(infer=(90.0,), proc=(120.0,)))
  assert ok["gates"]["tlvd_within_budget"] == {"pass": True, "available": True}
  over = bench.summarize(report(), report(infer=(150.0,)))
  assert not over["gates"]["tlvd_within_budget"]["pass"]
  missing = bench.summarize(report(), report(infer=(), proc=()))
  assert missing["gates"]["tlvd_within_budget"] == {"pass": False, "available": False}


def test_baseline_not_clean_flag():
  baseline = report()
  baseline["carrot_tlv_seen"] = 5
  assert bench.summarize(baseline, report())["baseline_clean"] is False
