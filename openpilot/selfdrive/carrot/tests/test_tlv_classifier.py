"""Speed-aware scheduler + detect-then-track classifier integration (PC)."""
import numpy as np

from openpilot.selfdrive.carrot.tlv import backends, decode, pipeline, tracker

W, H = 1928, 1208
RED_CLASS = int(tracker.LampClass.RED)
MODEL_BOX = (100.0, 50.0, 140.0, 90.0)


class ScriptedBackend:
  """Detector backend returning pre-scripted decoder-format outputs per call."""

  def __init__(self, frames=None):
    self.frames = list(frames or [])
    self.calls = 0

  def infer(self, model_input):
    self.calls += 1
    if self.frames:
      return self.frames.pop(0)
    return (np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, dtype=np.int64))

  def close(self):
    pass


def _red_frame(box=MODEL_BOX, score=0.9):
  return (np.asarray([box], np.float32), np.asarray([score], np.float32), np.asarray([RED_CLASS]))


def _empty_frame():
  return (np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, dtype=np.int64))


def _far_geom():
  return pipeline.FramePipeline.geometry("far", 600.0, W, H, 2648.0)


def _make_pipe(frames, classifier):
  backend = ScriptedBackend(frames)
  return pipeline.FramePipeline(backend, decode.PlaceholderDecoder(), classifier=classifier), backend


def test_view_scheduler_patterns():
  scheduler = pipeline.ViewScheduler()
  assert [scheduler.next_view(60.0) for _ in range(8)] == ["far", "far", "far", "near"] * 2
  scheduler = pipeline.ViewScheduler()
  assert [scheduler.next_view(80.0) for _ in range(4)] == ["far", "far", "far", "near"]
  scheduler = pipeline.ViewScheduler()
  assert [scheduler.next_view(10.0) for _ in range(8)] == ["near", "near", "near", "far"] * 2
  scheduler = pipeline.ViewScheduler()
  assert [scheduler.next_view(3.0) for _ in range(4)] == ["near"] * 4
  scheduler = pipeline.ViewScheduler()
  assert [scheduler.next_view(30.0) for _ in range(4)] == ["far", "near", "far", "near"]


def test_next_view_without_vego_keeps_legacy_alternation():
  pipe = pipeline.FramePipeline(backends.NullBackend(""), decode.PlaceholderDecoder())
  # Legacy no-arg view choice follows frame parity; frame_idx advances once per
  # infer (not per call), exactly as the pre-W74C pipeline behaved.
  views = []
  for frame_idx in range(4):
    pipe.frame_idx = frame_idx
    views.append(pipe.next_view())
  assert views == ["far", "near", "far", "near"]


def test_geometry_raised_only_for_near():
  far = pipeline.FramePipeline.geometry("far", 600.0, W, H, 2648.0, raised=True)
  assert far.y0 == 360  # FAR is unchanged by the standstill flag
  near = pipeline.FramePipeline.geometry("near", 600.0, W, H, 2648.0, raised=True)
  assert near.y0 == 24
  near_normal = pipeline.FramePipeline.geometry("near", 600.0, W, H, 2648.0)
  assert near_normal.y0 == 120


def test_pipeline_classifier_confirms_and_grades():
  stub = backends.StubClassifier(cls=RED_CLASS, prob=0.9)
  pipe, backend = _make_pipe([_red_frame()], stub)
  far = _far_geom()
  near = pipeline.FramePipeline.geometry("near", 600.0, W, H, 2648.0)

  out0 = pipe.infer("far", far, image_width=W, now=0.0, detect=True)
  assert out0.state == tracker.SignalState.none
  assert out0.red_likely is True  # one strong classifier vote, not confirmed

  # Detector skipped (throttled) on the other view; classification still runs.
  out1 = pipe.infer("near", near, image_width=W, now=0.2, detect=False)
  assert backend.calls == 1
  assert out1.state == tracker.SignalState.none
  assert out1.red_likely is True

  out2 = pipe.infer("near", near, image_width=W, now=0.4, detect=False)
  assert backend.calls == 1
  assert out2.state == tracker.SignalState.red  # 3 votes within 0.6 s
  assert out2.red_likely is False


def test_vote_class_green_gate():
  f32 = np.float32
  # argmax GREEN but green mass 0.45 < 0.5 -> falls back to the best non-green (RED).
  assert pipeline.vote_class(np.array([0.30, 0, 0.45, 0, 0, 0, 0, 0.25], dtype=f32)) == int(tracker.LampClass.RED)
  # green mass 0.55 >= 0.5 -> GREEN stays.
  assert pipeline.vote_class(np.array([0.20, 0, 0.40, 0, 0.15, 0, 0, 0.25], dtype=f32)) == int(tracker.LampClass.GREEN)
  # argmax GREEN_LEFT but mass 0.46 < 0.5 -> OFF.
  assert pipeline.vote_class(np.array([0.10, 0, 0.0, 0, 0.46, 0, 0, 0.44], dtype=f32)) == int(tracker.LampClass.OFF)
  # RED_LEFT is not in the green family: unchanged.
  assert pipeline.vote_class(np.array([0.05, 0, 0.05, 0.90, 0, 0, 0, 0], dtype=f32)) == int(tracker.LampClass.RED_LEFT)
  # Exact 0.5: the argmax tie takes index 2 and the boundary is inclusive -> GREEN.
  assert pipeline.vote_class(np.array([0, 0, 0.5, 0, 0, 0, 0, 0.5], dtype=f32)) == int(tracker.LampClass.GREEN)


def test_pipeline_low_green_mass_votes_non_green():
  probs = np.array([0.30, 0, 0.45, 0, 0, 0, 0, 0.25], dtype=np.float64)

  class LowGreenClassifier:
    def classify(self, batch):
      logits = np.log(np.maximum(probs, 1e-12)).astype(np.float32)
      return np.tile(logits, (len(batch), 1))

  pipe, backend = _make_pipe([_red_frame()], LowGreenClassifier())
  far = _far_geom()
  near = pipeline.FramePipeline.geometry("near", 600.0, W, H, 2648.0)

  pipe.infer("far", far, image_width=W, now=0.0, detect=True)
  pipe.infer("near", near, image_width=W, now=0.2, detect=False)
  out2 = pipe.infer("near", near, image_width=W, now=0.4, detect=False)

  votes = [vote for track in pipe.tracker.tracks for vote in track.votes]
  assert votes  # the gate still records votes
  assert all(v[1] == RED_CLASS for v in votes)  # 0.45 green mass -> RED fallback
  assert out2.state == tracker.SignalState.red  # 3 red votes in 0.6 s, no green


def test_pipeline_without_classifier_keeps_detector_belt():
  pipe, backend = _make_pipe([_red_frame(), _red_frame(), _red_frame()], backends.NullClassifier())
  far = _far_geom()
  states = [pipe.infer("far", far, image_width=W, now=0.1 * i, detect=True).state for i in range(3)]
  assert states == [tracker.SignalState.none, tracker.SignalState.none, tracker.SignalState.red]
  assert pipe.last_red_likely is False
  # NullClassifier never divides the tracking: every frame ran the detector.
  assert backend.calls == 3


def test_detect_false_keeps_last_published_dets():
  pipe, backend = _make_pipe([_red_frame()], backends.StubClassifier(cls=RED_CLASS, prob=0.9))
  far = _far_geom()
  pipe.infer("far", far, image_width=W, now=0.0, detect=True)
  assert len(pipe.last_dets) == 1
  pipe.infer("far", far, image_width=W, now=0.2, detect=False)
  assert backend.calls == 1
  assert len(pipe.last_dets) == 1  # stale dets keep the fusion latch refreshed


def test_classify_order_is_relevant_first_and_bounded():
  pipe = pipeline.FramePipeline(backends.NullBackend(""), decode.PlaceholderDecoder())
  for i in range(5):
    box = (700.0 + 60.0 * i, 400.0, 740.0 + 60.0 * i, 440.0)
    pipe.tracker.update(i, [tracker.Detection(cls=RED_CLASS, score=0.9, x1=box[0], y1=box[1], x2=box[2], y2=box[3])],
                        "far", 2648.0, now=0.0)
  pipe.last_track_id = pipe.tracker.tracks[3].track_id
  order = pipe._classify_order()
  assert order[0].track_id == pipe.last_track_id
  assert len(order) == pipeline.MAX_CLASSIFY_TRACKS
  assert order[1].last_frame >= order[-1].last_frame  # recency order for the rest
