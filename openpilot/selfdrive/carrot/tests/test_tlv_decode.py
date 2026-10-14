"""Tests for the raw-head detector decoder (carrot-tl-vision train/decode.py port).

The bundled detector exports 6 raw heads (p3/p4/p5 box+cls); before W116 the
device paired them with PlaceholderDecoder, so decode() always returned [].
These tests pin the port's math with analytic heads and prove the bundled
artifact picks RawHeadDecoder on both backends.
"""
import shutil
import sys

import numpy as np
import pytest

from openpilot.selfdrive.carrot.tlv import backends
from openpilot.selfdrive.carrot.tlv.decode import (PlaceholderDecoder, RawHeadDecoder,
                                                   is_raw_head_layout, make_decoder)

R3B_NAMES = ("p3_box", "p3_cls", "p4_box", "p4_cls", "p5_box", "p5_cls")
R3B_SHAPES = {
  "p3_box": (1, 64, 40, 80), "p3_cls": (1, 8, 40, 80),
  "p4_box": (1, 64, 20, 40), "p4_cls": (1, 8, 20, 40),
  "p5_box": (1, 64, 10, 20), "p5_cls": (1, 8, 10, 20),
}
BOX_CHANNELS = 64


def _base(nc: int) -> dict:
  """Zero DFL heads (uniform -> dist ~7.5) and cls logits at -10 everywhere."""
  heads = {}
  for name, shape in R3B_SHAPES.items():
    if name.endswith("_cls"):
      heads[name] = np.full((1, nc) + shape[2:], -10.0, np.float32)
    else:
      heads[name] = np.zeros((1, BOX_CHANNELS) + shape[2:], np.float32)
  return heads


def _set_cls(heads, level: int, y: int, x: int, cls: int, logit: float) -> None:
  heads[f"p{level}_cls"][0, cls, y, x] = logit


def _set_dfl(heads, level: int, y: int, x: int, l: int, t: int, r: int, b: int) -> None:
  """Single 30.0 logit per side at bin (l, t, r, b): softmax -> dist exactly that bin."""
  box = heads[f"p{level}_box"]
  for side, bin_idx in enumerate((l, t, r, b)):
    box[0, side * 16 + bin_idx, y, x] = 30.0


def _pack(heads, names):
  return [heads[name] for name in names]


def _ensure_cc(tmp_path, monkeypatch) -> bool:
  """tinygrad compiles CPU kernels with clang (CC env); ziglang wrapper fallback."""
  if shutil.which("clang"):
    return True
  try:
    import ziglang  # noqa: F401
  except ImportError:
    return False
  driver = tmp_path / "zigcc_driver.py"
  driver.write_text(
    "\n".join([
      "import subprocess, sys",
      "args = [a.replace('-none-unknown-elf', '-linux-gnu') for a in sys.argv[1:]]",
      "args.append('-g0')",
      "sys.exit(subprocess.run([sys.executable, '-m', 'ziglang', 'cc', *args],",
      "                        input=sys.stdin.buffer.read()).returncode)",
    ]) + "\n",
    encoding="utf-8",
  )
  wrapper = tmp_path / "zigcc.cmd"
  wrapper.write_text(f'@"{sys.executable}" "{driver}" %*\n', encoding="ascii")
  monkeypatch.setenv("CC", str(wrapper))
  return True


def _tinygrad_available() -> bool:
  try:
    import tinygrad  # noqa: F401
    return True
  except ImportError:
    repo = backends._tinygrad_repo_path()
    if repo not in sys.path:
      sys.path.append(repo)
    try:
      import tinygrad  # noqa: F401
      return True
    except ImportError:
      return False


def test_layout_detection():
  shuffled = ("p5_cls", "p3_box", "p4_cls", "p3_cls", "p5_box", "p4_box")
  assert is_raw_head_layout(R3B_NAMES) is True
  assert is_raw_head_layout(shuffled) is True
  assert is_raw_head_layout(["boxes", "scores", "classes"]) is False
  assert is_raw_head_layout(["p3_box", "p4_cls"]) is False
  assert is_raw_head_layout(()) is False
  assert is_raw_head_layout(["p3_box", "p3_box", "p3_cls"]) is False
  assert isinstance(make_decoder(R3B_NAMES), RawHeadDecoder)
  assert isinstance(make_decoder(("boxes", "scores", "classes")), PlaceholderDecoder)
  with pytest.raises(ValueError):
    RawHeadDecoder(["boxes", "scores", "classes"])


def test_analytic_single_box():
  heads = _base(8)
  _set_cls(heads, 4, 5, 7, 2, 10.0)
  _set_dfl(heads, 4, 5, 7, 2, 3, 4, 5)
  dec = RawHeadDecoder(R3B_NAMES)
  dets = dec.decode(_pack(heads, R3B_NAMES), "far")
  assert len(dets) == 1
  det = dets[0]
  assert det.cls == 2
  assert det.score == pytest.approx(float(1.0 / (1.0 + np.exp(-10.0))), abs=1e-6)
  # anchor (7.5, 5.5), p4 stride 16, dist l=2 t=3 r=4 b=5:
  # ((7.5-2)*16, (5.5-3)*16, (7.5+4)*16, (5.5+5)*16) = (88, 40, 184, 168)
  assert (det.x1, det.y1, det.x2, det.y2) == pytest.approx((88.0, 40.0, 184.0, 168.0), abs=1e-3)
  perm = [3, 0, 5, 2, 4, 1]
  names = [R3B_NAMES[i] for i in perm]
  dets2 = RawHeadDecoder(names).decode(_pack(heads, names), "far")
  assert len(dets2) == 1 and dets2[0].cls == 2
  assert (dets2[0].x1, dets2[0].y1, dets2[0].x2, dets2[0].y2) == pytest.approx((88.0, 40.0, 184.0, 168.0), abs=1e-3)


def test_analytic_single_class():
  heads = _base(1)
  _set_cls(heads, 4, 5, 7, 0, 10.0)
  _set_dfl(heads, 4, 5, 7, 2, 3, 4, 5)
  dets = RawHeadDecoder(R3B_NAMES).decode(_pack(heads, R3B_NAMES), "near")
  assert len(dets) == 1
  assert dets[0].cls == 0
  assert (dets[0].x1, dets[0].y1, dets[0].x2, dets[0].y2) == pytest.approx((88.0, 40.0, 184.0, 168.0), abs=1e-3)


def test_nms_and_class_offset():
  dec = RawHeadDecoder(R3B_NAMES)
  # p3 cells (10, 10) and (10, 11), box 32x32 at stride 8, IoU 0.6 > 0.45.
  heads = _base(8)
  _set_cls(heads, 3, 10, 10, 0, 10.0)
  _set_cls(heads, 3, 10, 11, 0, 12.0)
  _set_dfl(heads, 3, 10, 10, 2, 2, 2, 2)
  _set_dfl(heads, 3, 10, 11, 2, 2, 2, 2)
  dets = dec.decode(_pack(heads, R3B_NAMES), "far")
  assert len(dets) == 1  # same class -> suppressed, higher score kept
  assert dets[0].score == pytest.approx(float(1.0 / (1.0 + np.exp(-12.0))), abs=1e-6)

  heads = _base(8)
  _set_cls(heads, 3, 10, 10, 0, 10.0)
  _set_cls(heads, 3, 10, 11, 1, 10.0)
  _set_dfl(heads, 3, 10, 10, 2, 2, 2, 2)
  _set_dfl(heads, 3, 10, 11, 2, 2, 2, 2)
  dets = dec.decode(_pack(heads, R3B_NAMES), "far")
  assert len(dets) == 2  # different classes: the 7680 px class offset separates them
  assert sorted(d.cls for d in dets) == [0, 1]


def test_threshold_maxdet_clip():
  # exactly-at-threshold scores are dropped (the mask is strictly greater).
  dec = RawHeadDecoder(R3B_NAMES, score_threshold=0.5)
  heads = _base(8)
  _set_cls(heads, 3, 5, 5, 0, 0.0)   # sigmoid(0) == 0.5 -> dropped
  _set_cls(heads, 3, 5, 10, 0, 1.0)  # sigmoid(1) > 0.5 -> kept
  _set_dfl(heads, 3, 5, 5, 1, 1, 1, 1)
  _set_dfl(heads, 3, 5, 10, 1, 1, 1, 1)
  dets = dec.decode(_pack(heads, R3B_NAMES), "far")
  assert len(dets) == 1
  assert dets[0].score == pytest.approx(float(1.0 / (1.0 + np.exp(-1.0))), abs=1e-6)

  # 60 separated strong cells (16 px boxes on a 16 px anchor lattice) with
  # max_det 50: NMS stops at the cap.
  dec = RawHeadDecoder(R3B_NAMES, max_det=50)
  heads = _base(8)
  cells = [(1, x) for x in range(2, 78, 2)] + [(3, x) for x in range(2, 46, 2)]
  assert len(cells) == 60
  for y, x in cells:
    _set_cls(heads, 3, y, x, 0, 10.0)
    _set_dfl(heads, 3, y, x, 1, 1, 1, 1)
  dets = dec.decode(_pack(heads, R3B_NAMES), "far")
  assert len(dets) == 50

  # a box extending past x=640 is clipped to 640; a zero-width box is dropped
  # (device-only addition; in this layout zero width is built directly because
  # clipping alone cannot make x2 <= x1 - the DFL distances are non-negative).
  dec = RawHeadDecoder(R3B_NAMES)
  heads = _base(8)
  _set_cls(heads, 3, 5, 79, 0, 10.0)
  _set_dfl(heads, 3, 5, 79, 2, 2, 10, 2)  # x2 = (79.5+10)*8 = 716 -> 640
  _set_cls(heads, 3, 5, 70, 0, 10.0)
  _set_dfl(heads, 3, 5, 70, 0, 2, 0, 2)   # x1 == x2 == 564 -> dropped
  dets = dec.decode(_pack(heads, R3B_NAMES), "far")
  assert len(dets) == 1
  assert dets[0].x1 == pytest.approx(620.0, abs=1e-3)
  assert dets[0].x2 == pytest.approx(640.0, abs=1e-3)


def test_bad_outputs_never_raise():
  dec = RawHeadDecoder(R3B_NAMES)
  heads = _base(8)
  raw = _pack(heads, R3B_NAMES)
  assert dec.decode(None, "far") == []
  assert dec.decode(raw[:5], "far") == []           # wrong count
  bad = list(raw)
  bad[0] = np.zeros((64, 40, 80), np.float32)        # wrong rank (3-D box head)
  assert dec.decode(bad, "far") == []
  assert dec.decode({"boxes": []}, "far") == []      # dict is not a raw-head list


def test_bundled_detector_uses_raw_decoder():
  pytest.importorskip("onnxruntime")
  backend, dec = backends.make_backend("onnx", backends.BUNDLED_DETECTOR)
  try:
    assert isinstance(dec, RawHeadDecoder)
    assert sorted(backend.output_names) == sorted(R3B_NAMES)
    spec = backend._session.get_inputs()[0]
    shape = tuple(int(d) for d in spec.shape)
    dtype = np.uint8 if "uint8" in spec.type else np.float32
    raw = backend.infer(np.zeros(shape, dtype))
    dets = dec.decode(raw, "far")
    assert isinstance(dets, list)
  finally:
    backend.close()


def test_bundled_detector_tinygrad_matches_onnx(tmp_path, monkeypatch):
  pytest.importorskip("onnxruntime")
  if not _tinygrad_available():
    pytest.skip("tinygrad is not available")
  if not _ensure_cc(tmp_path, monkeypatch):
    pytest.skip("no clang-compatible C compiler available for tinygrad CPU kernels")

  x = np.random.default_rng(5).integers(0, 256, (1, 3, 320, 640), dtype=np.uint8)
  onnx_backend, _ = backends.make_backend("onnx", backends.BUNDLED_DETECTOR)
  tg_backend = None
  try:
    tg_backend, _ = backends.make_backend("tinygrad", backends.BUNDLED_DETECTOR)
    assert tg_backend.output_names == onnx_backend.output_names
    raw_ort = onnx_backend.infer(x)
    raw_tg = tg_backend.infer(x)
    assert len(raw_ort) == len(raw_tg) == 6
    for name, ref, got in zip(onnx_backend.output_names, raw_ort, raw_tg):
      np.testing.assert_allclose(np.asarray(ref, np.float32), np.asarray(got, np.float32),
                                 atol=1e-3, err_msg=name)
    dec_ort = RawHeadDecoder(onnx_backend.output_names, score_threshold=0.01)
    dec_tg = RawHeadDecoder(tg_backend.output_names, score_threshold=0.01)
    d_ort = sorted(dec_ort.decode(raw_ort, "far"), key=lambda d: (d.cls, d.x1))
    d_tg = sorted(dec_tg.decode(raw_tg, "far"), key=lambda d: (d.cls, d.x1))
    assert len(d_ort) == len(d_tg)
    for a, b in zip(d_ort, d_tg):
      assert a.cls == b.cls
      assert (a.x1, a.y1, a.x2, a.y2) == pytest.approx((b.x1, b.y1, b.x2, b.y2), abs=0.5)
      assert a.score == pytest.approx(b.score, abs=1e-3)
  finally:
    onnx_backend.close()
    if tg_backend is not None:
      tg_backend.close()
