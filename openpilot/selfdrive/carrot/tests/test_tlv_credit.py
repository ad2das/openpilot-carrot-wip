"""W99 tests: strict CPU credit budget, cost-aware scheduling, credit telemetry."""
from __future__ import annotations

import types
from collections import deque

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.carrot.tlv import resources, tlvd, tracker

DURATION = 120.0
LONG_DURATION = 600.0
CAMERA_HZ = 20.0
CLS_COST = 0.05


class SimClocks:
  def __init__(self) -> None:
    self.t = 0.0
    self.cpu = 0.0

  def clock(self) -> float:
    return self.t

  def cpu_clock(self) -> float:
    return self.cpu


def make_budget(**kwargs):
  clocks = SimClocks()
  kwargs.setdefault("clock", clocks.clock)
  kwargs.setdefault("cpu_clock", clocks.cpu_clock)
  return resources.CpuCreditBudget(**kwargs), clocks


def run_credit_sim(det_cost: float, *, cls_cost: float = CLS_COST, tracks: bool = False,
                   duration: float = DURATION, camera_hz: float = CAMERA_HZ):
  """Camera-cadence frames scheduled by tlvd.choose_frame_kind on fake clocks."""
  budget, clocks = make_budget()
  last_detect = last_classify = -float("inf")
  dt = 1.0 / camera_hz
  detects = classifies = 0
  trace: list[tuple[float, float]] = []
  while clocks.t < duration:
    now = clocks.t
    kind = tlvd.choose_frame_kind(now=now, last_detect=last_detect, last_classify=last_classify,
                                  has_tracks=tracks, classifier_enabled=True, budget=budget)
    if kind is not None:
      budget.begin(kind, now)
      clocks.cpu += det_cost if kind == "detect" else cls_cost
      budget.end(now)
      if kind == "detect":
        detects += 1
        last_detect = last_classify = now
      else:
        classifies += 1
        last_classify = now
    clocks.t += dt
    trace.append((clocks.t, clocks.cpu))
  return clocks, detects, classifies, trace


def _max_window_cpu(trace: list[tuple[float, float]], window_s: float) -> float:
  best = 0.0
  j = 0
  for i, (t0, c0) in enumerate(trace):
    j = max(j, i)
    while j + 1 < len(trace) and trace[j + 1][0] - t0 <= window_s:
      j += 1
    if j > i and trace[j][1] >= c0:
      best = max(best, trace[j][1] - c0)
  return best


# ---------------------------------------------------------------------------
# CpuCreditBudget
# ---------------------------------------------------------------------------

def test_credit_starts_at_start_credit_and_caps():
  budget, clocks = make_budget()
  budget.tick(0.0)
  assert budget.credit == pytest.approx(resources.CpuCreditBudget.CREDIT_START_S)
  clocks.t = 100.0
  budget.tick(100.0)
  assert budget.credit == pytest.approx(resources.CpuCreditBudget.CREDIT_CAP_S)


def test_credit_charges_all_process_cpu_not_only_frames():
  budget, clocks = make_budget()
  budget.tick(0.0)  # seed the baselines
  clocks.t = 1.0
  budget.tick(1.0)
  assert budget.credit == pytest.approx(0.4 + 0.4)
  # CPU spent by OTHER threads (recorder writer, SubMaster, publish) must be
  # charged by the next tick even though no frame is running.
  clocks.cpu += 0.25
  clocks.t = 1.5
  budget.tick(1.5)
  assert budget.credit == pytest.approx(0.4 + 0.4 - 0.25 + 0.4 * 0.5)


def test_estimate_uses_init_then_ema_and_last5():
  budget, clocks = make_budget()
  assert budget.estimate("detect") == pytest.approx(0.5)
  assert budget.estimate("classify") == pytest.approx(0.05)
  assert budget.estimate_ms("detect") == pytest.approx(500.0)
  budget.begin("detect", 0.0)
  clocks.cpu += 1.0
  budget.end(0.0)
  # max(EMA 0.3*1.0+0.7*0.5=0.65, 0.8*1.0=0.8)
  assert budget.estimate("detect") == pytest.approx(0.8)


def test_max_frame_hz_gate():
  budget, clocks = make_budget()
  assert budget.can_run("classify", 0.0)
  budget.begin("classify", 0.0)
  budget.end(0.0)
  assert not budget.can_run("classify", 0.05)
  assert budget.can_run("classify", 0.1)


def test_detect_from_full_bucket_runs_over_expensive_frame():
  budget, clocks = make_budget()
  # One 3.0 s detector frame drives its estimate above the 2.0 s cap.
  budget.begin("detect", 0.0)
  clocks.cpu += 3.0
  budget.end(0.0)
  assert budget.estimate("detect") > budget.cap_s
  clocks.t = 12.0
  budget.tick(12.0)
  assert budget.credit == pytest.approx(budget.cap_s)
  # Not affordable by estimate, but the full bucket runs the detector anyway...
  assert budget.can_run("detect", 12.1)
  budget.begin("detect", 12.1)
  clocks.cpu += 3.0
  budget.end(12.1)
  # ...and credit goes negative, so the long-run average still holds.
  assert budget.credit < 0.0


def test_full_bucket_rule_is_detect_only():
  budget, clocks = make_budget()
  for i in range(6):  # build a classify estimate above the cap too
    budget.begin("classify", float(i))
    clocks.cpu += 3.0
    clocks.t = float(i)
    budget.end(float(i))
  assert budget.estimate("classify") > budget.cap_s
  clocks.t = 100.0
  budget.tick(100.0)
  assert budget.credit == pytest.approx(budget.cap_s)
  assert not budget.can_run("classify", 100.1)


def test_duty_needs_two_second_span():
  budget, clocks = make_budget()
  budget.tick(0.0)
  clocks.cpu += 0.1
  clocks.t = 1.0
  budget.tick(1.0)
  assert budget.duty is None
  clocks.t = 3.0
  budget.tick(3.0)
  assert budget.duty == pytest.approx(0.1 / 3.0)


@pytest.mark.parametrize("det_cost", (0.1, 1.3, 3.0))
@pytest.mark.parametrize("tracks", (False, True))
def test_credit_sim_long_run_duty(det_cost, tracks):
  clocks, detects, _classifies, _trace = run_credit_sim(det_cost, tracks=tracks)
  duty = clocks.cpu / clocks.t
  # Documented window guarantee: CPU <= cap_s + budget*W + one frame's cost.
  assert duty <= 0.40 + (resources.CpuCreditBudget.CREDIT_START_S + det_cost) / DURATION + 1e-9
  if det_cost <= 0.4:
    assert duty <= 0.41
  else:
    # A frame costlier than the cap may leave one unpaid overdraft at the very
    # end of a 120 s window; the amortized long run shows the plain bound.
    long_clocks, _d, _c, _t = run_credit_sim(det_cost, tracks=tracks, duration=LONG_DURATION)
    assert long_clocks.cpu / long_clocks.t <= 0.41


def test_credit_leaves_no_frame_rate_floor():
  clocks, detects, _classifies, _trace = run_credit_sim(1.3)
  rate = detects / clocks.t
  # The old governor's min_hz=1.0 would have kept this >= 1 Hz (and broken the
  # budget); the credit scheduler runs at budget / cost ~= 0.31 Hz instead.
  assert 0.29 <= rate <= 0.33
  assert detects > 0


def test_credit_sim_window_bound():
  _clocks, _detects, _classifies, trace = run_credit_sim(1.3, tracks=True)
  for window_s in (10.0, 30.0):
    used = _max_window_cpu(trace, window_s)
    bound = resources.CpuCreditBudget.CREDIT_CAP_S + 0.40 * window_s + 1.3
    assert used <= bound + 1e-9


# ---------------------------------------------------------------------------
# tlvd.choose_frame_kind
# ---------------------------------------------------------------------------

def test_choose_frame_kind_detector_first_and_intervals():
  budget, _clocks = make_budget()
  budget.tick(0.0)  # seed; credit needs 0.25 s to reach the initial 0.5 s estimate
  assert tlvd.choose_frame_kind(now=1.0, last_detect=-1e9, last_classify=-1e9, has_tracks=False,
                                classifier_enabled=True, budget=budget) == "detect"
  # 0.05 s after a detect neither interval is due without tracks -> skip.
  assert tlvd.choose_frame_kind(now=1.05, last_detect=1.0, last_classify=-1e9, has_tracks=False,
                                classifier_enabled=True, budget=budget) is None
  # With tracks the classify interval (0.2 s) can fill in between detections.
  assert tlvd.choose_frame_kind(now=1.25, last_detect=1.0, last_classify=-1e9, has_tracks=True,
                                classifier_enabled=True, budget=budget) == "classify"
  # Detector interval doubles with tracks: at 0.3 s it is not due yet.
  assert tlvd.choose_frame_kind(now=1.3, last_detect=1.0, last_classify=1.25, has_tracks=True,
                                classifier_enabled=True, budget=budget) is None
  # At 0.5 s the detector takes priority over the classifier (1.4 - 1.0 is
  # 0.3999... in float, so stay off the exact interval boundary).
  assert tlvd.choose_frame_kind(now=1.5, last_detect=1.0, last_classify=1.25, has_tracks=True,
                                classifier_enabled=True, budget=budget) == "detect"


def test_choose_frame_kind_needs_classifier_and_tracks():
  budget, _clocks = make_budget()
  assert tlvd.choose_frame_kind(now=0.0, last_detect=0.0, last_classify=-1e9, has_tracks=True,
                                classifier_enabled=False, budget=budget) is None
  assert tlvd.choose_frame_kind(now=0.0, last_detect=0.0, last_classify=-1e9, has_tracks=False,
                                classifier_enabled=True, budget=budget) is None


def test_choose_frame_kind_saves_credit_for_overdue_detector():
  budget, clocks = make_budget()
  budget.begin("detect", 0.0)  # expensive detect sets estimate ~2.4 > cap
  clocks.cpu += 3.0
  budget.end(0.0)
  clocks.t = 9.0
  budget.tick(9.0)
  assert budget.credit == pytest.approx(1.0)
  # Detector overdue (>= 3 s) and not affordable: the classifier is refused so
  # the credit can rebuild for the detector.
  assert tlvd.choose_frame_kind(now=9.0, last_detect=0.0, last_classify=-1e9, has_tracks=True,
                                classifier_enabled=True, budget=budget) is None
  # With a recent detect the same credit state runs the cheap classifier.
  assert tlvd.choose_frame_kind(now=9.0, last_detect=8.0, last_classify=-1e9, has_tracks=True,
                                classifier_enabled=True, budget=budget) == "classify"


def test_choose_frame_kind_skips_without_credit():
  budget, clocks = make_budget()
  budget.begin("detect", 0.0)  # spend everything
  clocks.cpu += 1.0
  budget.end(0.0)
  assert budget.credit < 0.0
  assert tlvd.choose_frame_kind(now=0.05, last_detect=-1e9, last_classify=-1e9, has_tracks=True,
                                classifier_enabled=True, budget=budget) is None


# ---------------------------------------------------------------------------
# carrotTlv telemetry publish
# ---------------------------------------------------------------------------

def test_publish_fills_credit_telemetry(monkeypatch):
  runner = tlvd.TlvRunner.__new__(tlvd.TlvRunner)
  runner.last_view = "far"
  runner.dropped = 5
  runner.skipped_budget = 7
  runner.budget, _clocks = make_budget()
  runner.budget.tick(0.0)
  now = __import__("time").monotonic()
  runner._frame_times = {
    "detect": deque([now - 1.0, now - 0.5]),
    "classify": deque([now - 0.3, now - 0.2, now - 0.1, now]),
  }
  runner.pipeline = types.SimpleNamespace(
    infer_ms=1.5, proc_ms=2.5, last_dets=[], last_red_likely=False,
    last_state=tracker.SignalState.none, last_conf=0.0, last_track_id=-1,
    tail_gated_total=4, tracker=types.SimpleNamespace(last_matches=[]),
  )
  sent = []
  runner.pub = types.SimpleNamespace(send=lambda service, msg: sent.append((service, msg)))
  captured = []
  real_new = messaging.new_message

  def fake_new(service, *args, **kwargs):
    msg = real_new(service, *args, **kwargs)
    captured.append(msg)
    return msg

  monkeypatch.setattr(messaging, "new_message", fake_new)
  runner._publish("running", "")

  assert sent and sent[0][0] == "carrotTlv"
  out = captured[0].carrotTlv
  assert out.cpuCredit == pytest.approx(0.4)
  assert out.cpuDuty == pytest.approx(0.0)  # None until the window spans
  assert out.skippedBudget == 7
  assert out.tailGated == 4
  assert out.detHz == pytest.approx(2.0, abs=0.01)
  assert out.clsHz == pytest.approx(10.0, abs=0.01)
  assert out.detCostMs == pytest.approx(500.0)
  assert out.clsCostMs == pytest.approx(50.0)
