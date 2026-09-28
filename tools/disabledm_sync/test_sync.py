import json
import runpy
from pathlib import Path

import pytest

from tools.disabledm_sync.sync import SyncError, git, prepare, publish  # noqa: TID251 -- repository maintenance, not the vehicle tools package


def commit(repo, name, contents, message):
  path = repo / name
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(contents, encoding="utf-8")
  git(repo, "add", "--", name)
  git(repo, "commit", "-m", message)
  return git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def repos(tmp_path):
  upstream = tmp_path / "upstream"
  upstream.mkdir()
  git(upstream, "init", "-b", "carrot-wip")
  commit(upstream, "config.txt", "default dm\n", "initial")
  origin = tmp_path / "fork.git"
  git(tmp_path, "clone", "--bare", str(upstream), str(origin))
  candidate = tmp_path / "candidate"
  git(tmp_path, "clone", str(origin), str(candidate))
  commit(candidate, "disabledm.txt", "hidden override\n", "restore patch")
  git(candidate, "push", "origin", "carrot-wip")
  return upstream, origin, candidate


def remote_head(origin):
  return git(origin, "rev-parse", "carrot-wip").stdout.strip()


def test_update_keeps_patch_and_publishes_only_at_final_step(repos):
  upstream, origin, repo = repos
  original = remote_head(origin)
  source = commit(upstream, "new-feature.txt", "new upstream\n", "upstream update")
  state = prepare(repo, str(upstream))
  assert state.changed and state.original == original and state.upstream == source
  assert remote_head(origin) == original
  assert (repo / "disabledm.txt").read_text() == "hidden override\n"
  assert (repo / "new-feature.txt").read_text() == "new upstream\n"
  git(repo, "merge-base", "--is-ancestor", source, state.candidate)
  assert publish(repo, state)
  assert remote_head(origin) == state.candidate
  again = prepare(repo, str(upstream))
  assert not again.changed and not publish(repo, again)


def test_conflict_aborts_without_overwriting_either_branch(repos):
  upstream, origin, repo = repos
  original = commit(repo, "config.txt", "patched dm\n", "local setting")
  git(repo, "push", "origin", "carrot-wip")
  commit(upstream, "config.txt", "different upstream dm\n", "overlapping update")
  with pytest.raises(SyncError, match="merge failed"):
    prepare(repo, str(upstream))
  assert git(repo, "rev-parse", "HEAD").stdout.strip() == original
  assert remote_head(origin) == original
  assert not git(repo, "status", "--porcelain").stdout.strip()
  assert (repo / "config.txt").read_text() == "patched dm\n"


def test_validation_failure_leaves_remote_unchanged(repos):
  upstream, origin, repo = repos
  original = remote_head(origin)
  commit(upstream, "new-feature.txt", "incompatible upstream\n", "update")
  state = prepare(repo, str(upstream))
  assert state.changed
  # A failed workflow test never reaches publish.
  assert remote_head(origin) == original


def test_concurrent_target_update_is_not_overwritten(repos, tmp_path):
  upstream, origin, repo = repos
  commit(upstream, "new-feature.txt", "new\n", "update")
  state = prepare(repo, str(upstream))
  other = tmp_path / "other"
  git(tmp_path, "clone", str(origin), str(other))
  newer = commit(other, "owner.txt", "owner edit\n", "owner update")
  git(other, "push", "origin", "carrot-wip")
  with pytest.raises(SyncError, match="changed during validation"):
    publish(repo, state)
  assert remote_head(origin) == newer


def test_dirty_checkout_is_preserved(repos):
  upstream, origin, repo = repos
  original = remote_head(origin)
  (repo / "config.txt").write_text("unsaved work\n", encoding="utf-8")
  with pytest.raises(SyncError, match="dirty"):
    prepare(repo, str(upstream))
  assert (repo / "config.txt").read_text() == "unsaved work\n"
  assert remote_head(origin) == original


def test_changed_candidate_cannot_be_published(repos):
  upstream, origin, repo = repos
  original = remote_head(origin)
  commit(upstream, "new-feature.txt", "new\n", "update")
  state = prepare(repo, str(upstream))
  commit(repo, "unexpected.txt", "not validated\n", "unexpected")
  with pytest.raises(SyncError, match="changed after preparation"):
    publish(repo, state)
  assert remote_head(origin) == original


def test_workflow_publishes_after_tests_without_forced_push():
  root = Path(__file__).resolve().parents[2]
  workflow = (root / ".github/workflows/disabledm-sync.yml").read_text(encoding="utf-8")
  assert "17,47 * * * *" in workflow
  assert "github.repository == 'ad2das/openpilot-carrot-wip'" in workflow
  assert "persist-credentials: false" in workflow
  assert workflow.index("Check DisableDM and sync regressions") < workflow.index("Publish validated merge")
  checks = workflow[workflow.index("- name: Check DisableDM and sync regressions"):workflow.index("- name: Publish validated merge")]
  assert "server/tests/test_telemetry_origin.py" in checks
  assert "server/tests/test_upload_identity.py" in checks
  assert "--force" not in workflow
  assert "if: always()" not in workflow[workflow.index("- name: Publish validated merge"):workflow.index("- name: Report result")]


def test_prepare_records_upstream_origin(repos):
  upstream, origin, repo = repos
  source = commit(upstream, "new-feature.txt", "new upstream\n", "upstream update")
  state = prepare(repo, str(upstream))
  assert state.changed

  path = repo / "openpilot/selfdrive/carrot/upstream_origin.json"
  payload = json.loads(path.read_text(encoding="utf-8"))
  assert payload["remote"] == str(upstream)
  assert payload["branch"] == "carrot-wip"
  assert payload["commit"] == source
  assert payload["short_commit"]
  assert payload["commit_date"].startswith("'") and payload["commit_date"].endswith("'")
  assert len(payload["commit_datetime"]) == 19
  assert "upstream_origin.json" in git(repo, "show", "--name-only", "--format=", "HEAD").stdout


def test_successive_updates_keep_upload_patch_and_refresh_reported_identity(repos):
  upstream, origin, repo = repos
  root = Path(__file__).resolve().parents[2]
  helper_rel = "openpilot/selfdrive/carrot/telemetry_origin.py"
  helper_source = (root / helper_rel).read_text(encoding="utf-8")
  commit(repo, helper_rel, helper_source, "report upstream upload identity")
  git(repo, "push", "origin", "carrot-wip")

  for revision in range(2):
    original = remote_head(origin)
    source = commit(upstream, "new-feature.txt", f"upstream revision {revision}\n", "upstream update")
    state = prepare(repo, str(upstream))
    reported = runpy.run_path(str(repo / helper_rel))

    assert remote_head(origin) == original
    assert (repo / helper_rel).read_text(encoding="utf-8") == helper_source
    assert (repo / "disabledm.txt").read_text() == "hidden override\n"
    assert reported["reported_git_remote"]() == str(upstream)
    assert reported["reported_git_branch"]() == "carrot-wip"
    assert reported["reported_git_commit"]() == source
    assert source.startswith(reported["reported_git_short_commit"]())
    assert reported["reported_param_value"]("DisableDM", "2") == "0"
    assert reported["reported_param_value"]("DisableDMActive", "2") == "0"
    assert publish(repo, state)
    assert remote_head(origin) == state.candidate

    again = prepare(repo, str(upstream))
    assert not again.changed and not publish(repo, again)
    assert reported["reported_git_commit"]() == source
