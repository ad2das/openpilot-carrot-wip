"""Exercise the production distance marker without a GPU or native UI imports."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('distance', [25.1, 35.4])
def test_marker_draws_a_rule_across_the_lane_at_the_published_target(distance):
  path = Path(__file__).resolve().parents[1] / 'onroad/model_renderer.py'
  tree = ast.parse(path.read_text(encoding='utf-8'))
  method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_draw_tf_marker_carrot')
  polylines = []
  namespace = {'hs': SimpleNamespace(polyline=lambda points, width, color: polylines.append((points, width, color)),
                                     rgba=lambda *args: args)}
  exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
  renderer = SimpleNamespace(_carrot_tf_distance=distance, _carrot_tf_left=(10, 20), _carrot_tf_right=(30, 20))
  namespace[method.name](renderer)

  assert len(polylines) == 1
  points, width, color = polylines[0]
  assert points == [(pytest.approx(13.6), pytest.approx(20.0)), (pytest.approx(26.4), pytest.approx(20.0))]
  assert width == 4.0
  assert color == (255, 255, 255, 230)


def test_marker_hidden_without_a_target():
  path = Path(__file__).resolve().parents[1] / 'onroad/model_renderer.py'
  tree = ast.parse(path.read_text(encoding='utf-8'))
  method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_draw_tf_marker_carrot')
  polylines = []
  namespace = {'hs': SimpleNamespace(polyline=lambda points, width, color: polylines.append((points, width, color)),
                                     rgba=lambda *args: args)}
  exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
  renderer = SimpleNamespace(_carrot_tf_distance=0.0, _carrot_tf_left=(10, 20), _carrot_tf_right=(30, 20))
  namespace[method.name](renderer)

  assert polylines == []
