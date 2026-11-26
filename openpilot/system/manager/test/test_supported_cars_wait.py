import ast
from pathlib import Path


def _manager_thread() -> ast.FunctionDef:
  src = (Path(__file__).parents[1] / "manager.py").read_text()
  return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "manager_thread")


def test_supported_cars_drain_waits_only_once():
  # The 5 s drain must sit under a one-shot guard. Otherwise a writer that is still
  # running re-blocks every onroad manager iteration, delaying ensure_running,
  # managerState and the power watchdog by up to 5 s each time.
  fn = _manager_thread()
  drain = next(n for n in ast.walk(fn)
               if isinstance(n, ast.While) and "supported_cars_thread.is_alive()" in ast.unparse(n.test))
  guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and drain in n.body)
  assert "not supported_cars_waited" in ast.unparse(guard.test)
  set_flag = next(i for i, s in enumerate(guard.body) if ast.unparse(s) == "supported_cars_waited = True")
  assert set_flag < guard.body.index(drain)

  init = [n for n in fn.body if isinstance(n, ast.Assign) and ast.unparse(n) == "supported_cars_waited = False"]
  assert len(init) == 1
