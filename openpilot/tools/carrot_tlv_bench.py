#!/usr/bin/env python3
"""carrotTlv bench, per RESOURCE_BUDGET.md.

Runs on the C3X (Carrot web terminal):

  python3 openpilot/tools/carrot_tlv_bench.py --backend null
  python3 openpilot/tools/carrot_tlv_bench.py --backend tinygrad --model-path /data/models/tlv/model.onnx

Phases: baseline (CarrotTlvDisabled=True, tlvd off), tlvd (CarrotTlvDisabled=False),
then a soak (default 20 min, `--soak-min`). Every phase samples modelV2 / controlsState /
roadCameraState / uiDebug / deviceState messages plus, from /proc: per-process CPU and
RSS for tlvd, modeld, controlsd, ui and camerad, MemAvailable, and the cores 1-3
/proc/stat totals.

It prints ONE JSON document with pass/fail for each of the eight budget gates
and the raw numbers, restores CarrotTlvDisabled (and backend/model path) on exit
including Ctrl+C, and exits 0 only when every gate passes. A difference larger
than the noise is a FAIL.

Gate math (`summarize_phase`, `evaluate_gates`, the /proc parsers) is pure and
unit-tested on a PC; openpilot messaging imports stay lazy so the module is
importable anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np

from openpilot.selfdrive.carrot.tlv import backends, resources

# Gate tolerances ("beyond noise").
FRAME_DROP_DELTA = 1.0            # allowed frameDropPerc mean/p95 increase
EXEC_P95_RATIO = 1.10             # allowed model execution p95 increase
EXEC_P50_RATIO = 1.10             # allowed model execution p50 increase
CONTROL_P99_SLACK_MS = 2.0        # controlsState interval p99 increase
CONTROL_MAX_SLACK_MS = 5.0        # controlsState interval max increase
CAMERA_GAP_RATE_SLACK_S = 0.5     # extra skipped roadCameraState frameIds per second
CAMERA_GAP_MAX_SKIP_SLACK = 1     # extra frames in the worst single gap
UI_DRAW_P95_RATIO = 1.10
UI_DRAW_P95_SLACK_MS = 1.0
UI_CPU_SLACK_PCT = 5.0            # ui process CPU% increase
# W99: the strict credit budget replaces the 2 s duty / 1 Hz floor policy.
TLVD_CPU_DUTY_PCT = 42.0          # 60 s process CPU duty (all tlvd threads)
TLVD_CPU_WINDOW_PCT = 60.0        # any 10 s process CPU window
TLVD_CPU_DUTY_WINDOW_S = 60.0
TLVD_CPU_MAX_WINDOW_S = 10.0
CORES13_CPU_SLACK_PCT = 5.0       # cores 1-3 total increase beyond tlvd's own CPU
TLVD_RSS_MAX_MB = 220.0
TLVD_RSS_GROWTH_MAX_MB = 5.0      # during the soak
MEM_AVAILABLE_FLOOR_MB = 600.0
MEM_AVAILABLE_DROP_MAX_MB = 300.0
TEMP_SLACK_C = 3.0
THERMAL_YELLOW = 1

PROC_SAMPLE_S = 1.0
PID_REFRESH_S = 5.0
DEFAULT_PROCESSES = {
  "tlvd": "openpilot.selfdrive.carrot.tlv.tlvd",
  "modeld": "openpilot.selfdrive.modeld.modeld",
  "controlsd": "openpilot.selfdrive.controls.controlsd",
  "ui": "openpilot.selfdrive.ui.ui",
  "camerad": "openpilot.system.camerad.camerad",
}


# ---------------------------------------------------------------------------
# /proc parsing (pure)
# ---------------------------------------------------------------------------

def parse_pid_stat(text: str) -> tuple[float, float]:
  """(utime, stime) in clock ticks from /proc/<pid>/stat fields 14/15."""
  close = text.rfind(")")
  if close < 0:
    raise ValueError(f"bad /proc/<pid>/stat: {text[:80]!r}")
  fields = text[close + 2:].split()
  if len(fields) < 13:
    raise ValueError(f"short /proc/<pid>/stat: {len(fields)} fields")
  return float(fields[11]), float(fields[12])


def parse_pid_statm_pages(text: str) -> int:
  fields = text.split()
  if len(fields) < 2:
    raise ValueError(f"short /proc/<pid>/statm: {text!r}")
  return int(fields[1])


def clk_tck() -> float:
  try:
    return float(os.sysconf("SC_CLK_TCK"))
  except (AttributeError, ValueError, OSError):
    return 100.0


class ProcSampler:
  """Per-process CPU/RSS plus MemAvailable and cores 1-3 from /proc."""

  def __init__(self, processes: dict[str, str] | None = None, proc_root: str = "/proc",
               ticks_per_s: float | None = None, pid_refresh_s: float = PID_REFRESH_S) -> None:
    self.processes = dict(processes or DEFAULT_PROCESSES)
    self.proc_root = proc_root
    self.ticks_per_s = float(ticks_per_s if ticks_per_s is not None else clk_tck())
    self.pid_refresh_s = float(pid_refresh_s)
    self.page_size = resources.default_page_size()
    self.pids: dict[str, int | None] = dict.fromkeys(self.processes)
    self.pid_changes: dict[str, int] = dict.fromkeys(self.processes, 0)
    self._last_pid_seen: dict[str, int | None] = dict.fromkeys(self.processes)
    self._last_scan = -1e9

  def _scan(self, now: float) -> None:
    if now - self._last_scan < self.pid_refresh_s:
      return
    self._last_scan = now
    found: dict[str, int | None] = dict.fromkeys(self.processes)
    try:
      entries = os.listdir(self.proc_root)
    except OSError:
      entries = []
    for entry in entries:
      if not entry.isdigit():
        continue
      try:
        with open(os.path.join(self.proc_root, entry, "cmdline"), "rb") as f:
          cmdline = f.read().decode("utf-8", "replace").replace("\x00", " ")
      except OSError:
        continue
      for name, token in self.processes.items():
        if found[name] is None and token in cmdline:
          found[name] = int(entry)
    for name, pid in found.items():
      self.pids[name] = pid
      if pid is not None:
        last = self._last_pid_seen[name]
        if last is not None and pid != last:
          self.pid_changes[name] += 1
        self._last_pid_seen[name] = pid

  def sample(self, now: float) -> dict:
    self._scan(now)
    procs: dict[str, dict] = {}
    for name, pid in self.pids.items():
      if pid is None:
        continue
      try:
        with open(os.path.join(self.proc_root, str(pid), "stat")) as f:
          stat_text = f.read()
        with open(os.path.join(self.proc_root, str(pid), "statm")) as f:
          statm_text = f.read()
      except OSError:
        continue
      try:
        utime, stime = parse_pid_stat(stat_text)
        rss_bytes = parse_pid_statm_pages(statm_text) * self.page_size
      except ValueError:
        continue
      procs[name] = {"cpu_s": (utime + stime) / self.ticks_per_s, "rss_mb": rss_bytes / resources.MB}
    meminfo = resources.read_text(os.path.join(self.proc_root, "meminfo"))
    stat = resources.read_text(os.path.join(self.proc_root, "stat"))
    mem_available = resources.parse_meminfo_available_bytes(meminfo) if meminfo else None
    return {
      "t": float(now),
      "procs": procs,
      "mem_available_mb": None if mem_available is None else mem_available / resources.MB,
      "cores13": resources.parse_proc_stat_totals(stat, (1, 2, 3)) if stat else None,
    }


# ---------------------------------------------------------------------------
# phase statistics (pure)
# ---------------------------------------------------------------------------

def _mean(values) -> float:
  return float(np.mean(values)) if values else 0.0


def _p50(values) -> float:
  return float(np.percentile(values, 50.0)) if values else 0.0


def _p95(values) -> float:
  return float(np.percentile(values, 95.0)) if values else 0.0


def _p99(values) -> float:
  return float(np.percentile(values, 99.0)) if values else 0.0


def _max_or_none(values):
  return float(np.max(values)) if values else None


def _min_or_none(values):
  present = [v for v in values if v is not None]
  return float(np.min(present)) if present else None


def _growth_mb(series) -> float | None:
  """Robust RSS growth: median of the last 5 samples minus median of the first 5."""
  values = [v for _, v in series if v is not None]
  if len(values) < 2:
    return None
  k = min(5, max(1, len(values) // 2))
  return float(np.median(values[-k:]) - np.median(values[:k]))


def cpu_pct_max_window(points: list[tuple[float, float]], window_s: float, *,
                       tolerance: float = 0.25) -> float | None:
  """Max process CPU% over ~window_s spans of (t, cpu_s) points; None when absent.

  For each start sample the farthest sample within window_s*(1+tolerance) is
  used, requiring a span of at least window_s*(1-tolerance). The result is
  CPU seconds per wall second as a percentage of one core.
  """
  if len(points) < 2:
    return None
  lo = float(window_s) * (1.0 - tolerance)
  hi = float(window_s) * (1.0 + tolerance)
  best: float | None = None
  j = 1
  for i, (t0, c0) in enumerate(points):
    j = max(j, i + 1)
    while j + 1 < len(points) and points[j + 1][0] - t0 <= hi:
      j += 1
    if j >= len(points):
      break
    span = points[j][0] - t0
    if span < lo or points[j][1] < c0:
      continue
    value = 100.0 * (points[j][1] - c0) / span
    best = value if best is None else max(best, value)
  return best


def camera_gap_stats(frames: list) -> tuple[float | None, int, float | None]:
  """(skipped frameIds per second, worst single skip, worst gap ms)."""
  if len(frames) < 2:
    return None, 0, None
  span = frames[-1][0] - frames[0][0]
  skipped = 0
  max_skip = 0
  max_gap_ms = None
  for i in range(1, len(frames)):
    dfid = frames[i][1] - frames[i - 1][1]
    if dfid > 1:
      skipped += dfid - 1
      max_skip = max(max_skip, dfid - 1)
      gap_ms = (frames[i][0] - frames[i - 1][0]) * 1000.0
      max_gap_ms = gap_ms if max_gap_ms is None else max(max_gap_ms, gap_ms)
  rate = skipped / span if span > 0.0 else None
  return rate, max_skip, max_gap_ms


def classify_episodes(states: list) -> tuple[list, list]:
  """Per-track classification episodes from (t, state, relevant_track_id).

  An episode is a run of consecutive carrotTlv samples carrying the SAME
  relevant track id (>= 0): the lamp classifier's effective sampling rate for
  that track. Returns (episodes [(track_id, t0, t1, votes)], rates_hz) where
  rates_hz = (votes-1)/(t1-t0): intervals per second, not votes per second,
  so a two-sample 0.2 s episode reports its 5 Hz cadence. Single-sample
  episodes contribute no rate.
  """
  episodes: list[tuple[int, float, float, int]] = []
  current: list | None = None
  for t, _state, track_id in states:
    tid = int(track_id)
    if tid < 0:
      if current is not None:
        episodes.append((current[0], current[1], current[2], current[3]))
        current = None
      continue
    if current is not None and current[0] == tid:
      current[2] = float(t)
      current[3] += 1
    else:
      if current is not None:
        episodes.append((current[0], current[1], current[2], current[3]))
      current = [tid, float(t), float(t), 1]
  if current is not None:
    episodes.append((current[0], current[1], current[2], current[3]))
  rates = [(votes - 1) / (t1 - t0) for _, t0, t1, votes in episodes if t1 > t0]
  return episodes, rates


def time_to_confirm(states: list) -> list:
  """Seconds from a first redLikely to the following confirmed red, per run.

  Any other state between the two (green/yellow/none/...) ends the run without
  a measurement.
  """
  deltas: list[float] = []
  red_likely_at: float | None = None
  for t, state, _track_id in states:
    name = str(state)
    if name == "redLikely":
      if red_likely_at is None:
        red_likely_at = float(t)
    elif name == "red":
      if red_likely_at is not None:
        deltas.append(float(t) - red_likely_at)
        red_likely_at = None
    else:
      red_likely_at = None
  return deltas


def _proc_summaries(samples: list) -> dict:
  names: set[str] = set()
  for sample in samples:
    names.update(sample.get("procs", {}).keys())
  out: dict[str, dict] = {}
  for name in sorted(names):
    points = [(s["t"], s["procs"][name]["cpu_s"], s["procs"][name].get("rss_mb"))
              for s in samples if name in s.get("procs", {})]
    if not points:
      continue
    cpu_pct_avg = None
    span = points[-1][0] - points[0][0]
    if len(points) >= 2 and span > 0.0 and points[-1][1] >= points[0][1]:
      cpu_pct_avg = 100.0 * (points[-1][1] - points[0][1]) / span
    windows = []
    for i in range(1, len(points) - 1):
      window = points[i + 1][0] - points[i - 1][0]
      if window < 1.0 or window > 3.0:
        continue
      delta = points[i + 1][1] - points[i - 1][1]
      if delta >= 0.0:
        windows.append(100.0 * delta / window)
    rss = [r for _, _, r in points if r is not None]
    out[name] = {
      "cpu_pct_avg": cpu_pct_avg,
      "cpu_pct_max_2s": max(windows) if windows else None,
      "rss_max_mb": max(rss) if rss else None,
      "rss_end_mb": rss[-1] if rss else None,
      "rss_series_mb": [[round(t, 3), round(r, 2)] for t, _, r in points if r is not None],
      "samples": len(points),
    }
  return out


def _cores13_busy_pct(samples: list) -> float | None:
  pairs = [(s["t"], s["cores13"]) for s in samples if s.get("cores13") is not None]
  if len(pairs) < 2:
    return None
  t0, (idle0, total0) = pairs[0]
  t1, (idle1, total1) = pairs[-1]
  del t0, t1
  d_total = total1 - total0
  if d_total <= 0.0:
    return None
  return 100.0 * (1.0 - (idle1 - idle0) / d_total)


def summarize_phase(report: dict) -> dict:
  """Derived statistics for one collection phase."""
  controls_ns = report.get("controls_mono_ns", [])
  controls_ms = [(controls_ns[i] - controls_ns[i - 1]) / 1e6 for i in range(1, len(controls_ns)) if controls_ns[i] > controls_ns[i - 1]]
  gap_rate, max_skip, max_gap_ms = camera_gap_stats(report.get("cam_frames", []))
  samples = report.get("proc_samples", [])
  procs = _proc_summaries(samples)
  tlvd_rss_series = procs.get("tlvd", {}).get("rss_series_mb", [])
  tlvd_points = [(s["t"], s["procs"]["tlvd"]["cpu_s"]) for s in samples if "tlvd" in s.get("procs", {})]
  tlvd_cpu_avg = procs.get("tlvd", {}).get("cpu_pct_avg")
  tlvd_duty60 = cpu_pct_max_window(tlvd_points, TLVD_CPU_DUTY_WINDOW_S)
  if tlvd_duty60 is None:
    tlvd_duty60 = tlvd_cpu_avg  # phase shorter than the duty window
  tlvd_max10 = cpu_pct_max_window(tlvd_points, TLVD_CPU_MAX_WINDOW_S)
  if tlvd_max10 is None:
    tlvd_max10 = tlvd_cpu_avg
  tlv_states = report.get("tlv_states", [])
  _, classify_rates = classify_episodes(tlv_states)
  confirm_deltas = time_to_confirm(tlv_states)

  def _last(series):
    return series[-1] if series else None

  return {
    "seconds": report.get("seconds"),
    "counts": {
      "frame_drop": len(report.get("frame_drop", [])),
      "model_exec": len(report.get("model_exec_ms", [])),
      "controls": len(controls_ms),
      "camera": len(report.get("cam_frames", [])),
      "ui_draw": len(report.get("ui_draw_ms", [])),
      "thermal": len(report.get("thermal_status", [])),
      "tlv_msgs": report.get("carrot_tlv_seen", 0),
      "proc_samples": len(samples),
    },
    "frame_drop_mean": _mean(report.get("frame_drop", [])),
    "frame_drop_p95": _p95(report.get("frame_drop", [])),
    "model_exec_p50_ms": _p50(report.get("model_exec_ms", [])),
    "model_exec_p95_ms": _p95(report.get("model_exec_ms", [])),
    "controls_p99_ms": _p99(controls_ms),
    "controls_max_ms": _max_or_none(controls_ms) or 0.0,
    "camera_gap_rate_s": gap_rate,
    "camera_gap_max_skip": max_skip,
    "camera_gap_max_ms": max_gap_ms,
    "ui_draw_p95_ms": _p95(report.get("ui_draw_ms", [])),
    "procs": procs,
    "tlvd_cpu_duty60_pct": tlvd_duty60,
    "tlvd_cpu_max10_pct": tlvd_max10,
    "cores13_busy_pct_avg": _cores13_busy_pct(samples),
    "mem_available_min_mb": _min_or_none([s.get("mem_available_mb") for s in samples]),
    "thermal_status_max": _max_or_none(report.get("thermal_status", [])),
    "max_temp_c_max": _max_or_none(report.get("max_temp_c", [])),
    "gpu_temp_c_max": _max_or_none(report.get("gpu_temp_c", [])),
    "tlv_infer_p95_ms": _p95(report.get("tlv_infer_ms", [])),
    "tlv_proc_p95_ms": _p95(report.get("tlv_proc_ms", [])),
    "tlv_status_seen": report.get("tlv_status_seen", []),
    "tlv_dropped_final": report.get("tlv_dropped_final"),
    "tlv_rss_growth_mb": _growth_mb(tlvd_rss_series),
    "tlv_classify_episodes": len(classify_episodes(tlv_states)[0]),
    "tlv_classify_rate_hz_median": float(np.median(classify_rates)) if classify_rates else None,
    "tlv_classify_rate_hz_max": max(classify_rates) if classify_rates else None,
    "tlv_redlikely_episodes": len(time_to_confirm(tlv_states)),
    "tlv_time_to_confirm_s_median": float(np.median(confirm_deltas)) if confirm_deltas else None,
    "tlv_time_to_confirm_s_max": max(confirm_deltas) if confirm_deltas else None,
    "tlv_cpu_credit": _last(report.get("tlv_credit", [])),
    "tlv_cpu_duty": _last(report.get("tlv_duty", [])),
    "tlv_det_hz": _last(report.get("tlv_det_hz", [])),
    "tlv_cls_hz": _last(report.get("tlv_cls_hz", [])),
    "tlv_det_cost_ms": _last(report.get("tlv_det_cost_ms", [])),
    "tlv_cls_cost_ms": _last(report.get("tlv_cls_cost_ms", [])),
    "tlv_skipped_budget_final": _last(report.get("tlv_skipped_budget", [])),
    "tlv_tail_gated_final": _last(report.get("tlv_tail_gated", [])),
    "pid_changes": report.get("pid_changes", {}),
  }


# ---------------------------------------------------------------------------
# gate evaluation (pure)
# ---------------------------------------------------------------------------

def _join(*parts: str) -> str:
  return " ".join(parts)


def evaluate_gates(baseline: dict, tlv: dict, soak: dict | None) -> dict:
  """Evaluate all eight RESOURCE_BUDGET.md bench gates."""
  soak = soak or {}
  gates: dict[str, dict] = {}

  def gate(name: str, ok: bool, detail: str, **raw) -> None:
    gates[name] = {"pass": bool(ok), "detail": detail, "raw": raw}

  # 1. modeld executionTime p50/p99 and frameDropPerc.
  have_model = baseline["counts"]["model_exec"] > 0 and tlv["counts"]["model_exec"] > 0
  p50_ok = have_model and tlv["model_exec_p50_ms"] <= baseline["model_exec_p50_ms"] * EXEC_P50_RATIO
  p99_ok = have_model and tlv["model_exec_p95_ms"] <= baseline["model_exec_p95_ms"] * EXEC_P95_RATIO
  fd_ok = (baseline["counts"]["frame_drop"] > 0 and tlv["counts"]["frame_drop"] > 0
           and tlv["frame_drop_mean"] <= baseline["frame_drop_mean"] + FRAME_DROP_DELTA
           and tlv["frame_drop_p95"] <= baseline["frame_drop_p95"] + FRAME_DROP_DELTA)
  gate("modeld_exec_frame_drop", have_model and p50_ok and p99_ok and fd_ok,
       _join(
         f"exec p50 {tlv['model_exec_p50_ms']:.1f} vs {baseline['model_exec_p50_ms']:.1f} ms,",
         f"p99 {tlv['model_exec_p95_ms']:.1f} vs {baseline['model_exec_p95_ms']:.1f} ms,",
         f"frame_drop mean {tlv['frame_drop_mean']:.2f} vs {baseline['frame_drop_mean']:.2f}",
       ),
       p50_ok=p50_ok, p99_ok=p99_ok, frame_drop_ok=fd_ok)

  # 2. controlsState interval p99 and max.
  have_ctl = baseline["counts"]["controls"] > 0 and tlv["counts"]["controls"] > 0
  ctl_p99_ok = have_ctl and tlv["controls_p99_ms"] <= baseline["controls_p99_ms"] + CONTROL_P99_SLACK_MS
  ctl_max_ok = have_ctl and tlv["controls_max_ms"] <= baseline["controls_max_ms"] + CONTROL_MAX_SLACK_MS
  gate("controlsd_interval_p99_max", ctl_p99_ok and ctl_max_ok,
       _join(
         f"interval p99 {tlv['controls_p99_ms']:.2f} vs {baseline['controls_p99_ms']:.2f} ms,",
         f"max {tlv['controls_max_ms']:.2f} vs {baseline['controls_max_ms']:.2f} ms",
       ),
       p99_ok=ctl_p99_ok, max_ok=ctl_max_ok)

  # 3. roadCameraState frame gaps.
  have_cam = baseline["counts"]["camera"] > 1 and tlv["counts"]["camera"] > 1
  have_rates = baseline["camera_gap_rate_s"] is not None and tlv["camera_gap_rate_s"] is not None
  cam_rate_ok = have_cam and have_rates and tlv["camera_gap_rate_s"] <= baseline["camera_gap_rate_s"] + CAMERA_GAP_RATE_SLACK_S
  cam_skip_ok = have_cam and tlv["camera_gap_max_skip"] <= baseline["camera_gap_max_skip"] + CAMERA_GAP_MAX_SKIP_SLACK
  gate("camera_frame_gaps", cam_rate_ok and cam_skip_ok,
       _join(
         f"skip rate {tlv['camera_gap_rate_s']} vs {baseline['camera_gap_rate_s']} per s,",
         f"worst skip {tlv['camera_gap_max_skip']} vs {baseline['camera_gap_max_skip']} frames",
       ),
       rate_ok=cam_rate_ok, worst_skip_ok=cam_skip_ok)

  # 4. UI frame time p99 if published, else ui process CPU%.
  ui_base = baseline["procs"].get("ui", {})
  ui_tlv = tlv["procs"].get("ui", {})
  have_draw = baseline["counts"]["ui_draw"] > 0 and tlv["counts"]["ui_draw"] > 0
  draw_ok = None
  if have_draw:
    draw_ok = tlv["ui_draw_p95_ms"] <= baseline["ui_draw_p95_ms"] * UI_DRAW_P95_RATIO + UI_DRAW_P95_SLACK_MS
  cpu_ok = None
  if ui_base.get("cpu_pct_avg") is not None and ui_tlv.get("cpu_pct_avg") is not None:
    cpu_ok = ui_tlv["cpu_pct_avg"] <= ui_base["cpu_pct_avg"] + UI_CPU_SLACK_PCT
  checks = [v for v in (draw_ok, cpu_ok) if v is not None]
  draw_text = "n/a" if draw_ok is None else f"{tlv['ui_draw_p95_ms']:.2f} vs {baseline['ui_draw_p95_ms']:.2f} ms"
  cpu_text = "n/a" if cpu_ok is None else f"{ui_tlv.get('cpu_pct_avg', 0.0):.1f}% vs {ui_base.get('cpu_pct_avg', 0.0):.1f}%"
  gate("ui_frame_time", all(checks) if checks else True,
       f"draw p95 {draw_text}, ui CPU {cpu_text}",
       draw_ok=draw_ok, cpu_ok=cpu_ok, available=bool(checks))

  # 5. tlvd CPU: 60 s process duty <= 42% and every 10 s window <= 60%, plus
  # the cores 1-3 delta check (W99 credit budget replaces the 2 s/1 Hz policy).
  tlvd_t = tlv["procs"].get("tlvd", {})
  tlvd_s = soak.get("procs", {}).get("tlvd", {})
  duty_vals = [v for v in (tlv.get("tlvd_cpu_duty60_pct"), soak.get("tlvd_cpu_duty60_pct")) if v is not None]
  window_vals = [v for v in (tlv.get("tlvd_cpu_max10_pct"), soak.get("tlvd_cpu_max10_pct")) if v is not None]
  duty_ok = bool(duty_vals) and max(duty_vals) <= TLVD_CPU_DUTY_PCT
  window_ok = bool(window_vals) and max(window_vals) <= TLVD_CPU_WINDOW_PCT
  base_cores = baseline["cores13_busy_pct_avg"]
  deltas = []
  for phase in (tlv, soak):
    if base_cores is not None and phase.get("cores13_busy_pct_avg") is not None:
      deltas.append(phase["cores13_busy_pct_avg"] - base_cores)
  tlvd_cpu_avg = tlvd_t.get("cpu_pct_avg") or 0.0
  cores_ok = bool(deltas) and max(deltas) <= tlvd_cpu_avg + CORES13_CPU_SLACK_PCT
  gate("tlvd_cpu_budget", duty_ok and window_ok and cores_ok,
       _join(
         f"tlvd 60 s duty {max(duty_vals) if duty_vals else 'n/a'} <= {TLVD_CPU_DUTY_PCT}%,",
         f"10 s window {max(window_vals) if window_vals else 'n/a'} <= {TLVD_CPU_WINDOW_PCT}%,",
         f"cores1-3 delta {max(deltas) if deltas else 'n/a'} vs tlvd avg {tlvd_cpu_avg:.1f}%",
       ),
       duty_ok=duty_ok, window_ok=window_ok, cores13_ok=cores_ok)

  # 6. tlvd RSS max and soak growth.
  rss_maxes = [v for v in (tlvd_t.get("rss_max_mb"), tlvd_s.get("rss_max_mb")) if v is not None]
  rss_max_ok = bool(rss_maxes) and max(rss_maxes) <= TLVD_RSS_MAX_MB
  growth = soak.get("tlv_rss_growth_mb") if soak else tlv.get("tlv_rss_growth_mb")
  growth_ok = growth is not None and growth < TLVD_RSS_GROWTH_MAX_MB
  gate("tlvd_rss", rss_max_ok and growth_ok,
       _join(
         f"RSS max {max(rss_maxes) if rss_maxes else 'n/a'} <= {TLVD_RSS_MAX_MB} MB,",
         f"growth {growth if growth is not None else 'n/a'} < {TLVD_RSS_GROWTH_MAX_MB} MB",
       ),
       rss_max_ok=rss_max_ok, growth_ok=growth_ok)

  # 7. MemAvailable min vs baseline and the 600 MB floor.
  mem_mins = [v for v in (tlv["mem_available_min_mb"], soak.get("mem_available_min_mb")) if v is not None]
  mem_floor_ok = bool(mem_mins) and min(mem_mins) >= MEM_AVAILABLE_FLOOR_MB
  mem_drop_ok = (baseline["mem_available_min_mb"] is not None and bool(mem_mins)
                 and min(mem_mins) >= baseline["mem_available_min_mb"] - MEM_AVAILABLE_DROP_MAX_MB)
  gate("mem_available", mem_floor_ok and mem_drop_ok,
       f"min {min(mem_mins) if mem_mins else 'n/a'} MB vs baseline {baseline['mem_available_min_mb']} MB",
       floor_ok=mem_floor_ok, drop_ok=mem_drop_ok)

  # 8. thermalStatus and CPU/GPU max temp; no new throttling.
  base_status = baseline["thermal_status_max"]
  status_values = [v for v in (tlv["thermal_status_max"], soak.get("thermal_status_max")) if v is not None]
  status_ok = base_status is not None and bool(status_values) and max(status_values) <= base_status
  temp_values = [v for v in (tlv["max_temp_c_max"], soak.get("max_temp_c_max")) if v is not None]
  temp_ok = (baseline["max_temp_c_max"] is not None and bool(temp_values)
             and max(temp_values) <= baseline["max_temp_c_max"] + TEMP_SLACK_C)
  gpu_values = [v for v in (tlv["gpu_temp_c_max"], soak.get("gpu_temp_c_max")) if v is not None]
  gpu_ok = True
  if baseline["gpu_temp_c_max"] is not None and gpu_values:
    gpu_ok = max(gpu_values) <= baseline["gpu_temp_c_max"] + TEMP_SLACK_C
  throttled = bool(status_values) and max(status_values) >= THERMAL_YELLOW and (base_status is None or max(status_values) > base_status)
  gate("thermal", status_ok and temp_ok and gpu_ok and not throttled,
       _join(
         f"status {max(status_values) if status_values else 'n/a'} vs baseline {base_status},",
         f"temp {max(temp_values) if temp_values else 'n/a'} vs {baseline['max_temp_c_max']} C",
       ),
       status_ok=status_ok, temp_ok=temp_ok, gpu_ok=gpu_ok, throttled=throttled)

  verdict = all(g["pass"] for g in gates.values())
  return {"verdict": "pass" if verdict else "fail", "gates": gates}


# ---------------------------------------------------------------------------
# collection (device only; messaging wired by main)
# ---------------------------------------------------------------------------

def _collect(sm, seconds: float, rate_hz: float = 20.0, proc_sample_s: float = PROC_SAMPLE_S) -> dict:
  from openpilot.common.realtime import Ratekeeper
  sampler = ProcSampler()
  report: dict = {
    "seconds": float(seconds),
    "frame_drop": [], "model_exec_ms": [], "controls_mono_ns": [], "cam_frames": [],
    "ui_draw_ms": [], "thermal_status": [], "max_temp_c": [], "gpu_temp_c": [],
    "tlv_infer_ms": [], "tlv_proc_ms": [], "tlv_status_seen": [], "tlv_dropped_final": None,
    "tlv_states": [], "carrot_tlv_seen": 0, "proc_samples": [], "pid_changes": {},
    "tlv_credit": [], "tlv_duty": [], "tlv_det_hz": [], "tlv_cls_hz": [],
    "tlv_det_cost_ms": [], "tlv_cls_cost_ms": [], "tlv_skipped_budget": [], "tlv_tail_gated": [],
  }
  rk = Ratekeeper(rate_hz, print_delay_threshold=None)
  next_proc = 0.0
  end = time.monotonic() + float(seconds)
  while time.monotonic() < end:
    now = time.monotonic()
    sm.update(0)
    if sm.updated.get("modelV2") and sm.valid["modelV2"]:
      model = sm["modelV2"]
      report["frame_drop"].append(float(model.frameDropPerc))
      if len(model.modelExecutionTime) > 0:
        report["model_exec_ms"].append(1000.0 * max(float(v) for v in model.modelExecutionTime))
    if sm.seen.get("controlsState") and sm.updated.get("controlsState") and sm.valid["controlsState"]:
      report["controls_mono_ns"].append(int(sm.logMonoTime["controlsState"]))
    if sm.seen.get("roadCameraState") and sm.updated.get("roadCameraState") and sm.valid["roadCameraState"]:
      report["cam_frames"].append((now, int(sm["roadCameraState"].frameId)))
    if sm.seen.get("uiDebug") and sm.updated.get("uiDebug") and sm.valid["uiDebug"]:
      report["ui_draw_ms"].append(float(sm["uiDebug"].drawTimeMillis))
    if sm.seen.get("deviceState") and sm.updated.get("deviceState") and sm.valid["deviceState"]:
      state = sm["deviceState"]
      report["thermal_status"].append(float(state.thermalStatus.raw))
      if state.maxTempC > 0.0:
        report["max_temp_c"].append(float(state.maxTempC))
      if len(state.gpuTempC) > 0:
        report["gpu_temp_c"].append(float(max(state.gpuTempC)))
    if sm.seen.get("carrotTlv") and sm.updated.get("carrotTlv") and sm.valid["carrotTlv"]:
      report["carrot_tlv_seen"] += 1
      tlv = sm["carrotTlv"]
      report["tlv_infer_ms"].append(float(tlv.inferMs))
      report["tlv_proc_ms"].append(float(tlv.procMs))
      report["tlv_dropped_final"] = int(tlv.droppedFrames)
      report["tlv_states"].append((now, str(tlv.state), int(tlv.relevantTrackId)))
      report["tlv_credit"].append(float(tlv.cpuCredit))
      report["tlv_duty"].append(float(tlv.cpuDuty))
      report["tlv_det_hz"].append(float(tlv.detHz))
      report["tlv_cls_hz"].append(float(tlv.clsHz))
      report["tlv_det_cost_ms"].append(float(tlv.detCostMs))
      report["tlv_cls_cost_ms"].append(float(tlv.clsCostMs))
      report["tlv_skipped_budget"].append(int(tlv.skippedBudget))
      report["tlv_tail_gated"].append(int(tlv.tailGated))
      status = str(tlv.status)
      if status not in report["tlv_status_seen"]:
        report["tlv_status_seen"].append(status)
    if now >= next_proc:
      next_proc = now + proc_sample_s
      report["proc_samples"].append(sampler.sample(now))
    rk.keep_time()
  report["pid_changes"] = dict(sampler.pid_changes)
  return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _param_bytes(params, key: str):
  try:
    return params.get(key)
  except Exception:
    return None


def _int_param(params, key: str, default: int = 0) -> int:
  try:
    value = params.get_int(key)
  except Exception:
    return default
  return default if value is None else int(value)


def _param_text(value):
  if value is None:
    return None
  if isinstance(value, bytes):
    return value.decode("utf-8", "replace")
  return str(value)


def _tlvd_phase_error(params) -> str | None:
  """Reason when TrafficLightDetectMode keeps tlvd from starting, else None."""
  mode = _int_param(params, "TrafficLightDetectMode")
  return None if mode >= 1 else f"TrafficLightDetectMode={mode}"


def _put_raw(params, key: str, value) -> None:
  try:
    params.put(key, value)
  except Exception as exc:
    print(f"carrot_tlv_bench: failed to write {key}: {exc!r}", file=sys.stderr)


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description="carrotTlv bench (baseline vs tlvd vs soak), per RESOURCE_BUDGET.md")
  parser.add_argument("--baseline-s", type=float, default=60.0)
  parser.add_argument("--tlv-s", type=float, default=60.0)
  parser.add_argument("--soak-min", type=float, default=20.0, help="soak phase minutes; 0 skips the soak")
  parser.add_argument("--backend", default="null", help="null/onnx/tinygrad (tlvd CarrotTlvBackend)")
  parser.add_argument("--model-path", default=backends.BUNDLED_DETECTOR, help="CarrotTlvModelPath override (default: bundled detector)")
  parser.add_argument("--startup-wait-s", type=float, default=20.0)
  args = parser.parse_args(argv)

  from openpilot.common.params import Params
  import openpilot.cereal.messaging as messaging

  params = Params()
  original_disabled = _param_bytes(params, "CarrotTlvDisabled")
  original_backend = _param_bytes(params, "CarrotTlvBackend")
  original_model_path = _param_bytes(params, "CarrotTlvModelPath")

  def restore():
    for key, value in (("CarrotTlvDisabled", original_disabled), ("CarrotTlvBackend", original_backend),
                       ("CarrotTlvModelPath", original_model_path)):
      if value is not None:
        _put_raw(params, key, value)
    print("carrot_tlv_bench: restored original CarrotTlv params", file=sys.stderr)

  def on_signal(signum, frame):
    raise KeyboardInterrupt

  signal.signal(signal.SIGINT, on_signal)
  if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, on_signal)

  sm = messaging.SubMaster(["modelV2", "deviceState", "controlsState", "roadCameraState", "uiDebug", "carrotTlv"],
                           ignore_avg_freq=["uiDebug", "carrotTlv"])
  baseline_report = tlv_report = soak_report = None
  try:
    print("carrot_tlv_bench: baseline phase (tlvd off) ...", file=sys.stderr)
    _put_raw(params, "CarrotTlvDisabled", True)
    time.sleep(3.0)
    baseline_report = _collect(sm, args.baseline_s)

    block = _tlvd_phase_error(params)
    if block is not None:
      print(f"carrot_tlv_bench: {block}; tlvd will not start, aborting tlvd/soak phases", file=sys.stderr)
      return 2

    print("carrot_tlv_bench: tlvd phase (CarrotTlvDisabled=False) ...", file=sys.stderr)
    _put_raw(params, "CarrotTlvBackend", args.backend)
    if args.model_path is not None:
      _put_raw(params, "CarrotTlvModelPath", args.model_path)
    _put_raw(params, "CarrotTlvDisabled", False)
    deadline = time.monotonic() + args.startup_wait_s
    while time.monotonic() < deadline and not sm.seen.get("carrotTlv"):
      sm.update(100)
    tlv_report = _collect(sm, args.tlv_s)

    if args.soak_min > 0.0:
      print(f"carrot_tlv_bench: soak phase {args.soak_min:.1f} min ...", file=sys.stderr)
      soak_report = _collect(sm, args.soak_min * 60.0)
  except KeyboardInterrupt:
    print("carrot_tlv_bench: interrupted", file=sys.stderr)
    return 130
  finally:
    restore()

  baseline = summarize_phase(baseline_report)
  tlv = summarize_phase(tlv_report)
  soak = summarize_phase(soak_report) if soak_report is not None else None
  result = evaluate_gates(baseline, tlv, soak)
  result["raw"] = {"baseline": baseline, "tlv": tlv, "soak": soak}
  result["baseline_clean"] = baseline["counts"]["tlv_msgs"] == 0
  result["parameters"] = {
    "baseline_s": args.baseline_s, "tlv_s": args.tlv_s, "soak_min": args.soak_min,
    "backend": args.backend, "model_path": args.model_path,
    "original_disabled": _param_text(original_disabled),
  }
  last = (soak or tlv)
  telemetry = " ".join([
    "carrot_tlv_bench: telemetry",
    f"credit={last.get('tlv_cpu_credit')}",
    f"duty={last.get('tlv_cpu_duty')}",
    f"detHz={last.get('tlv_det_hz')}",
    f"clsHz={last.get('tlv_cls_hz')}",
    f"detCostMs={last.get('tlv_det_cost_ms')}",
    f"clsCostMs={last.get('tlv_cls_cost_ms')}",
    f"skippedBudget={last.get('tlv_skipped_budget_final')}",
    f"tailGated={last.get('tlv_tail_gated_final')}",
  ])
  print(telemetry, file=sys.stderr)
  print(json.dumps(result, indent=2))
  return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
  sys.exit(main())
