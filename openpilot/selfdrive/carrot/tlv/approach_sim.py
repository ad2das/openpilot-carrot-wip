"""Pure synthetic-approach simulation for the TLV high-speed case (W74C).

Drives the REAL tracker + signal state machine (and, for the `speed_aware`
policy, the speed-aware ViewScheduler) with a synthetic constant-speed approach
to a lit red lamp, and reports the distance at which red is CONFIRMED versus the
stopping need from CONTRACT.md "High-speed approach":

  stopping need = v * latency + v^2 / (2 * 2.5 m/s^2)

Box pixel size is exact pinhole geometry: focal * housing_m / distance, with
housing 0.35 m and focal 2648 px (FAR) or 1324 px (NEAR) for the detector's
model-space crop height (the hit-curve argument). The tracker always receives
full-frame boxes and the full-frame focal (2648 px), as pipeline.infer does.
The detector and lamp classifier hit probabilities are PARAMETERS (ASSUMPTIONS)
until W71's eval numbers exist: a logistic curve in the model-space box height.

No messaging/openpilot runtime imports, so this runs (and is unit-tested) on a
PC with numpy only.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from openpilot.selfdrive.carrot.tlv import crops, resources, tracker

REFERENCE_WIDTH = crops.REFERENCE_WIDTH
REFERENCE_FOCAL = crops.FOCAL_REFERENCE_PX
NEAR_FOCAL = REFERENCE_FOCAL / 2.0
HOUSING_HEIGHT_M = tracker.HOUSING_HEIGHT_M
STOP_DECEL_MS2 = 2.5

IMAGE_X0 = 0.5 * REFERENCE_WIDTH   # lamp on the lane centre: zero lateral error
BOX_CENTER_Y = 600.0               # irrelevant to selection (cx / height only)
LAMP_SCORE = 0.8
CLASSIFIER_VOTE_PROB = 0.9         # a hit classifier vote is a strong red vote
DEFAULT_D0_M = 120.0

POLICIES = ("current", "speed_aware", "credit")

# W99 credit-policy model (the source of truth is tlvd.choose_frame_kind +
# resources.CpuCreditBudget; these mirror its constants for the standalone sim).
CREDIT_DET_COSTS = (0.3, 0.7, 1.3)
CREDIT_SPEEDS = (60.0, 80.0, 100.0)
DEFAULT_CAMERA_HZ = 20.0
DEFAULT_DET_COST_S = 1.3
DEFAULT_CLS_COST_S = 0.05
CREDIT_BUDGET = resources.CpuCreditBudget.CPU_BUDGET
CREDIT_CAP_S = resources.CpuCreditBudget.CREDIT_CAP_S
CREDIT_START_S = resources.CpuCreditBudget.CREDIT_START_S
CLASSIFY_INTERVAL_S = 0.2
DETECT_INTERVAL_S = 0.2
DETECT_INTERVAL_TRACKS_S = 0.4
DETECT_MAX_GAP_S = 3.0


def stop_need_m(speed_kph: float, *, latency_s: float = 0.0,
                decel_ms2: float = STOP_DECEL_MS2) -> float:
  """CONTRACT stopping need: v*latency + v^2/(2a), v in m/s."""
  v = speed_kph / 3.6
  return v * latency_s + v * v / (2.0 * decel_ms2)


def detector_hit_probability(box_height_px: float, *, mid_px: float = 8.0, slope: float = 0.5,
                             floor: float = 0.05, ceil: float = 0.95) -> float:
  """ASSUMED detector hit curve (logistic in the model-space box height)."""
  raw = 1.0 / (1.0 + math.exp(-slope * (box_height_px - mid_px)))
  return floor + (ceil - floor) * raw


def classifier_vote_probability(box_height_px: float, *, mid_px: float = 4.0, slope: float = 0.6,
                                floor: float = 0.10, ceil: float = 0.99) -> float:
  """ASSUMED lamp-classifier vote curve (easier than the detector)."""
  raw = 1.0 / (1.0 + math.exp(-slope * (box_height_px - mid_px)))
  return floor + (ceil - floor) * raw


@dataclass(frozen=True)
class ApproachResult:
  speed_kph: float
  rate_hz: float
  policy: str
  d_confirm_m: float | None   # distance when the state machine reaches red
  confirm_s: float | None
  red_likely_m: float | None  # first graded redLikely (classifier policies)
  red_likely_s: float | None
  first_sight_m: float
  stop_need_m: float
  stop_margin_m: float | None  # d_confirm - stop_need; None when never confirmed
  det_cost_s: float = 0.0      # per-detect CPU seconds assumed (credit table)
  cls_cost_s: float = 0.0      # per-classify CPU seconds assumed


def old_governor_rate_hz(det_cost_s: float, cls_cost_s: float = DEFAULT_CLS_COST_S, *,
                         budget: float = CREDIT_BUDGET, min_hz: float = 1.0, max_hz: float = 10.0) -> float:
  """Emulated DutyCycleGovernor pacing: rate = budget / worst recent frame cost,
  clamped to [min_hz, max_hz]. The 1 Hz floor is what breaks the 40% budget
  whenever a frame costs more than 0.4 s (the W99 problem statement)."""
  worst = max(float(det_cost_s), float(cls_cost_s))
  return max(min_hz, min(max_hz, budget / worst))


def _detection(box_h_full_px: float) -> tracker.Detection:
  half = 0.5 * box_h_full_px
  return tracker.Detection(
    cls=int(tracker.LampClass.RED), score=LAMP_SCORE,
    x1=IMAGE_X0 - half, y1=BOX_CENTER_Y - half, x2=IMAGE_X0 + half, y2=BOX_CENTER_Y + half,
  )


def _select(tracks: list[tuple], frame_idx: int, odo=None):
  return tracker.select_relevant(
    tracks, frame_idx, image_width=REFERENCE_WIDTH,
    path_x=(0.0, 40.0), path_y=(0.0, 0.0), left_intent=False, odo=odo,
  )


def simulate_approach(speed_kph: float, rate_hz: float, *, policy: str = "current",
                      d0_m: float = DEFAULT_D0_M, seed: int = 0, latency_s: float = 0.0,
                      detector_hit=detector_hit_probability,
                      classifier_hit=classifier_vote_probability,
                      max_s: float = 40.0,
                      det_cost_s: float = 0.0, cls_cost_s: float = 0.0) -> ApproachResult:
  """One constant-speed approach to a red lamp that first appears at d0_m.

  `current`: legacy FAR/NEAR alternation + detector M-of-N confirmation.
  `speed_aware`: the ViewScheduler + lamp-classifier votes (time-based) OR the
  detector belt, matching the W74C pipeline (rate_hz is the governed frame rate).
  `credit`: the W99 credit scheduler; rate_hz is the CAMERA frame rate and
  det_cost_s/cls_cost_s are the per-frame CPU costs (see
  simulate_credit_approach).
  """
  if policy not in POLICIES:
    raise ValueError(f"unknown policy: {policy!r}")
  if policy == "credit":
    return simulate_credit_approach(
      speed_kph, rate_hz, det_cost_s=det_cost_s or DEFAULT_DET_COST_S,
      cls_cost_s=cls_cost_s or DEFAULT_CLS_COST_S, d0_m=d0_m, seed=seed,
      latency_s=latency_s, detector_hit=detector_hit, classifier_hit=classifier_hit, max_s=max_s,
    )
  scheduler = None
  if policy == "speed_aware":
    # Local import: the current-policy table must run before/without W74C.
    from openpilot.selfdrive.carrot.tlv.pipeline import ViewScheduler
    scheduler = ViewScheduler()
  use_classifier = policy == "speed_aware"

  rng = np.random.default_rng(seed)
  dt = 1.0 / float(rate_hz)
  v = float(speed_kph) / 3.6
  d = float(d0_m)
  t = 0.0
  frame = 0
  light = tracker.TrafficLightTracker()
  machine = tracker.SignalStateMachine()
  d_confirm: float | None = None
  confirm_s: float | None = None
  red_likely_m: float | None = None
  red_likely_s: float | None = None

  while d > 1.0 and t <= max_s:
    view = scheduler.next_view(speed_kph) if scheduler is not None else ("far" if frame % 2 == 0 else "near")
    model_focal = REFERENCE_FOCAL if view == "far" else NEAR_FOCAL
    model_box_h = model_focal * HOUSING_HEIGHT_M / d      # detector input pixels (model space)
    full_box_h = REFERENCE_FOCAL * HOUSING_HEIGHT_M / d  # full-frame pixels
    dets = [_detection(full_box_h)] if rng.random() < detector_hit(model_box_h) else []
    # The tracker gets full-frame boxes, so it needs the full-frame focal on both
    # views (the model-space focal is only for the detector hit curve).
    light.update(frame, dets, view, REFERENCE_FOCAL, now=t)

    if use_classifier and light.tracks:
      vote_prob = classifier_hit(full_box_h)
      votes = [(track.track_id, int(tracker.LampClass.RED), CLASSIFIER_VOTE_PROB)
               for track in light.tracks if rng.random() < vote_prob]
      light.record_votes(t, votes)

    confirmed = light.confirmed(frame)
    if use_classifier:
      merged = []
      for track in light.tracks:
        result = light.confirm_track(track, frame, t, use_classifier=True)
        if result is not None:
          merged.append((track, result[0], result[1]))
      confirmed = merged

    confirmed_relevant = _select(confirmed, frame)
    relevant = confirmed_relevant
    lamp_state = tracker.lamp_state_for(confirmed_relevant[1]) if confirmed_relevant is not None else None
    machine_conf = confirmed_relevant[2] if confirmed_relevant is not None else 0.0
    if relevant is None and use_classifier and light.tracks:
      # Candidate-only frame: selection for crop ordering / redLikely only
      # (the state machine stays on confirmed input, like the pipeline).
      candidates = [(track, int(track.cls), float(track.score)) for track in light.tracks]
      relevant = _select(candidates, frame)
    state, _ = machine.update(t, lamp_state, machine_conf)
    if state == tracker.SignalState.red and d_confirm is None:
      d_confirm = d
      confirm_s = t

    if use_classifier and red_likely_m is None and relevant is not None:
      confirmed_red = (relevant is confirmed_relevant and int(relevant[1]) in tracker.RED_FAMILY)
      if not confirmed_red and relevant[0].classifier_red_likely(t):
        red_likely_m = d
        red_likely_s = t

    d -= v * dt
    t += dt
    frame += 1

  need = stop_need_m(speed_kph, latency_s=latency_s)
  margin = None if d_confirm is None else d_confirm - need
  return ApproachResult(
    speed_kph=float(speed_kph), rate_hz=float(rate_hz), policy=policy,
    d_confirm_m=d_confirm, confirm_s=confirm_s,
    red_likely_m=red_likely_m, red_likely_s=red_likely_s,
    first_sight_m=float(d0_m), stop_need_m=need, stop_margin_m=margin,
  )


def simulate_credit_approach(speed_kph: float, camera_hz: float = DEFAULT_CAMERA_HZ, *,
                             det_cost_s: float = DEFAULT_DET_COST_S,
                             cls_cost_s: float = DEFAULT_CLS_COST_S,
                             budget: float = CREDIT_BUDGET, cap_s: float = CREDIT_CAP_S,
                             d0_m: float = DEFAULT_D0_M, seed: int = 0, latency_s: float = 0.0,
                             detector_hit=detector_hit_probability,
                             classifier_hit=classifier_vote_probability,
                             max_s: float = 40.0) -> ApproachResult:
  """Approach under the W99 credit scheduler with perfect ego-motion prediction.

  Camera frames arrive at camera_hz; each one runs a detect frame (interval
  0.2 s/0.4 s + credit + the 10 Hz cap), else a classify frame (0.2 s interval
  + credit, refused while the detector is overdue), else nothing. The process
  CPU charged to the credit is det_cost_s/cls_cost_s per chosen frame; the
  credit is the REAL resources.CpuCreditBudget on simulated clocks. The REAL
  tracker/state machine are driven with predicted association (synthetic
  straight approach, yaw 0); the ASSUMED hit curves supply detector/classifier
  hits. The scheduler decision mirrors tlvd.choose_frame_kind.
  """
  # Local import like the speed_aware path: this table must run standalone.
  from openpilot.selfdrive.carrot.tlv.pipeline import ViewScheduler

  sim_t = [0.0]
  sim_cpu = [0.0]
  budget_ = resources.CpuCreditBudget(budget=budget, cap_s=cap_s,
                                      clock=lambda: sim_t[0], cpu_clock=lambda: sim_cpu[0])
  rng = np.random.default_rng(seed)
  dt = 1.0 / float(camera_hz)
  v = float(speed_kph) / 3.6
  d = float(d0_m)
  t = 0.0
  frame = 0
  last_detect = -float("inf")
  last_classify = -float("inf")
  light = tracker.TrafficLightTracker()
  machine = tracker.SignalStateMachine()
  scheduler = ViewScheduler()
  odo = tracker.EgoOdometry()
  d_confirm: float | None = None
  confirm_s: float | None = None
  red_likely_m: float | None = None
  red_likely_s: float | None = None

  while d > 1.0 and t <= max_s:
    sim_t[0] = t
    odo.update(t, v, 0.0)

    kind = None
    detect_due = t - last_detect >= (DETECT_INTERVAL_TRACKS_S if light.tracks else DETECT_INTERVAL_S)
    if detect_due and budget_.can_run("detect", t):
      kind = "detect"
    elif light.tracks and t - last_classify >= CLASSIFY_INTERVAL_S and budget_.can_run("classify", t):
      if not (t - last_detect >= DETECT_MAX_GAP_S
              and budget_.credit - budget_.estimate("classify") < budget_.estimate("detect")):
        kind = "classify"

    if kind is not None:
      budget_.begin(kind, t)
      sim_cpu[0] += det_cost_s if kind == "detect" else cls_cost_s
      full_box_h = REFERENCE_FOCAL * HOUSING_HEIGHT_M / max(d, 0.1)
      if kind == "detect":
        view = scheduler.next_view(speed_kph)
        model_focal = REFERENCE_FOCAL if view == "far" else NEAR_FOCAL
        model_box_h = model_focal * HOUSING_HEIGHT_M / d
        dets = [_detection(full_box_h)] if rng.random() < detector_hit(model_box_h) else []
        ego = tracker.EgoFrame(odo=odo, focal_px=REFERENCE_FOCAL, foe_x=IMAGE_X0, foe_y=BOX_CENTER_Y,
                               image_w=REFERENCE_WIDTH, image_h=1208)
        # Full-frame boxes -> full-frame focal for the tracker, both views.
        light.update(
          frame, dets, view, REFERENCE_FOCAL, now=t,
          predict=lambda tr, ego=ego: tracker.predict_box(
            tr, ego.odo, focal_px=ego.focal_px, foe_x=ego.foe_x, foe_y=ego.foe_y,
            image_w=ego.image_w, image_h=ego.image_h,
          ),
          anchor=(odo.s, odo.psi, t),
        )
        last_detect = t  # a detect frame also classifies, like tlvd/pipeline
      if light.tracks:
        vote_prob = classifier_hit(full_box_h)
        votes = [(track.track_id, int(tracker.LampClass.RED), CLASSIFIER_VOTE_PROB)
                 for track in light.tracks if rng.random() < vote_prob]
        light.record_votes(t, votes)
      last_classify = t

      merged = []
      for track in light.tracks:
        result = light.confirm_track(track, frame, t, use_classifier=True)
        if result is not None:
          merged.append((track, result[0], result[1]))
      confirmed_relevant = _select(merged, frame, odo=odo)
      relevant = confirmed_relevant
      lamp_state = tracker.lamp_state_for(confirmed_relevant[1]) if confirmed_relevant is not None else None
      machine_conf = confirmed_relevant[2] if confirmed_relevant is not None else 0.0
      if relevant is None and light.tracks:
        candidates = [(track, int(track.cls), float(track.score)) for track in light.tracks]
        relevant = _select(candidates, frame, odo=odo)
      state, _ = machine.update(t, lamp_state, machine_conf)
      if state == tracker.SignalState.red and d_confirm is None:
        d_confirm = d
        confirm_s = t
      if red_likely_m is None and relevant is not None:
        confirmed_red = (relevant is confirmed_relevant and int(relevant[1]) in tracker.RED_FAMILY)
        if not confirmed_red and relevant[0].classifier_red_likely(t):
          red_likely_m = d
          red_likely_s = t

      budget_.end(t)
      frame += 1

    d -= v * dt
    t += dt

  need = stop_need_m(speed_kph, latency_s=latency_s)
  margin = None if d_confirm is None else d_confirm - need
  return ApproachResult(
    speed_kph=float(speed_kph), rate_hz=float(camera_hz), policy="credit",
    d_confirm_m=d_confirm, confirm_s=confirm_s,
    red_likely_m=red_likely_m, red_likely_s=red_likely_s,
    first_sight_m=float(d0_m), stop_need_m=need, stop_margin_m=margin,
    det_cost_s=float(det_cost_s), cls_cost_s=float(cls_cost_s),
  )


def credit_governor_table(speeds=CREDIT_SPEEDS, det_costs=CREDIT_DET_COSTS, *,
                          camera_hz: float = DEFAULT_CAMERA_HZ, cls_cost_s: float = DEFAULT_CLS_COST_S,
                          seed: int = 0) -> list[ApproachResult]:
  """Confirm distance vs stop need: old governor vs the W99 credit policy.

  The old policy runs at old_governor_rate_hz() - the 1 Hz floor can exceed the
  40% budget for det_cost_s > 0.4 - while the credit policy runs the camera at
  camera_hz and schedules every frame within the budget.
  """
  results: list[ApproachResult] = []
  for cost in det_costs:
    rate = old_governor_rate_hz(cost, cls_cost_s)
    for speed in speeds:
      old = simulate_approach(speed, rate, policy="speed_aware", seed=seed)
      results.append(dataclasses.replace(old, det_cost_s=float(cost), cls_cost_s=float(cls_cost_s)))
      results.append(simulate_credit_approach(
        speed, camera_hz, det_cost_s=cost, cls_cost_s=cls_cost_s, seed=seed,
      ))
  return results


def format_credit_table(results: list[ApproachResult]) -> str:
  """Compact table for the W99 report/CLI (rate is Hz for old, camera Hz for credit)."""
  lines = [
    f"{'policy':<12} {'km/h':>4} {'det_s':>5} {'rate':>6} {'d_confirm':>9} {'need':>7} {'margin':>7} {'t_confirm':>9}",
  ]
  for r in results:
    d_text = "never" if r.d_confirm_m is None else f"{r.d_confirm_m:.1f}"
    m_text = "n/a" if r.stop_margin_m is None else f"{r.stop_margin_m:+.1f}"
    t_text = "never" if r.confirm_s is None else f"{r.confirm_s:.3f}"
    lines.append(
      f"{r.policy:<12} {r.speed_kph:>4.0f} {r.det_cost_s:>5.1f} {r.rate_hz:>6.2f} {d_text:>9} {r.stop_need_m:>7.1f} {m_text:>7} {t_text:>9}",
    )
  return "\n".join(lines)


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description="TLV approach sim: old governor vs the W99 credit policy")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--camera-hz", type=float, default=DEFAULT_CAMERA_HZ)
  args = parser.parse_args(argv)
  print(format_credit_table(credit_governor_table(seed=args.seed, camera_hz=args.camera_hz)))
  return 0


def run_matrix(policies=POLICIES, speeds=(60.0, 80.0, 100.0), rates=(1.0, 2.0, 5.0, 10.0),
               **kwargs) -> list[ApproachResult]:
  """The W74C before/after table: every speed x rate for every policy."""
  results = []
  for policy in policies:
    for speed in speeds:
      for rate in rates:
        results.append(simulate_approach(speed, rate, policy=policy, **kwargs))
  return results


def format_table(results: list[ApproachResult]) -> str:
  """Compact fixed-width table for the report/tests (3 decimals on seconds)."""
  lines = [
    f"{'policy':<12} {'km/h':>4} {'Hz':>4} {'d_confirm':>9} {'need':>7} {'margin':>7} {'t_confirm':>9} {'d_redLikely':>11}",
  ]
  for r in results:
    d_text = "never" if r.d_confirm_m is None else f"{r.d_confirm_m:.1f}"
    m_text = "n/a" if r.stop_margin_m is None else f"{r.stop_margin_m:+.1f}"
    t_text = "never" if r.confirm_s is None else f"{r.confirm_s:.3f}"
    rl_text = "-" if r.red_likely_m is None else f"{r.red_likely_m:.1f}"
    lines.append(
      f"{r.policy:<12} {r.speed_kph:>4.0f} {r.rate_hz:>4.0f} {d_text:>9} {r.stop_need_m:>7.1f} {m_text:>7} {t_text:>9} {rl_text:>11}"
    )
  return "\n".join(lines)
