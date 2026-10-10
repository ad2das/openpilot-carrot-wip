"""Pure synthetic-approach simulation for the TLV high-speed case (W74C).

Drives the REAL tracker + signal state machine (and, for the `speed_aware`
policy, the speed-aware ViewScheduler) with a synthetic constant-speed approach
to a lit red lamp, and reports the distance at which red is CONFIRMED versus the
stopping need from CONTRACT.md "High-speed approach":

  stopping need = v * latency + v^2 / (2 * 2.5 m/s^2)

Box pixel size is exact pinhole geometry: focal * housing_m / distance, with
housing 0.35 m and focal 2648 px (FAR) or 1324 px (NEAR). The detector and lamp
classifier hit probabilities are PARAMETERS (ASSUMPTIONS) until W71's eval
numbers exist: a logistic curve in the model-space box height.

No messaging/openpilot runtime imports, so this runs (and is unit-tested) on a
PC with numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from openpilot.selfdrive.carrot.tlv import crops, tracker

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

POLICIES = ("current", "speed_aware")


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


def _detection(box_h_full_px: float) -> tracker.Detection:
  half = 0.5 * box_h_full_px
  return tracker.Detection(
    cls=int(tracker.LampClass.RED), score=LAMP_SCORE,
    x1=IMAGE_X0 - half, y1=BOX_CENTER_Y - half, x2=IMAGE_X0 + half, y2=BOX_CENTER_Y + half,
  )


def _select(tracks: list[tuple], frame_idx: int):
  return tracker.select_relevant(
    tracks, frame_idx, image_width=REFERENCE_WIDTH,
    path_x=(0.0, 40.0), path_y=(0.0, 0.0), left_intent=False,
  )


def simulate_approach(speed_kph: float, rate_hz: float, *, policy: str = "current",
                      d0_m: float = DEFAULT_D0_M, seed: int = 0, latency_s: float = 0.0,
                      detector_hit=detector_hit_probability,
                      classifier_hit=classifier_vote_probability,
                      max_s: float = 40.0) -> ApproachResult:
  """One constant-speed approach to a red lamp that first appears at d0_m.

  `current`: legacy FAR/NEAR alternation + detector M-of-N confirmation.
  `speed_aware`: the ViewScheduler + lamp-classifier votes (time-based) OR the
  detector belt, matching the W74C pipeline.
  """
  if policy not in POLICIES:
    raise ValueError(f"unknown policy: {policy!r}")
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
    focal = REFERENCE_FOCAL if view == "far" else NEAR_FOCAL
    model_box_h = focal * HOUSING_HEIGHT_M / d          # detector input pixels
    full_box_h = REFERENCE_FOCAL * HOUSING_HEIGHT_M / d  # full-frame pixels
    dets = [_detection(full_box_h)] if rng.random() < detector_hit(model_box_h) else []
    light.update(frame, dets, view, focal, now=t)

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
