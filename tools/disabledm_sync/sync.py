"""Prepare a merge in a clean checkout, then publish only after validation.

No reset, forced push, conflict preference, or mutation of the upstream remote.
The Actions workflow supplies the validation boundary between the two commands.
"""
import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


IDENTITY = ["-c", "user.name=github-actions[bot]", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com"]

UPSTREAM_ORIGIN_REL = "openpilot/selfdrive/carrot/upstream_origin.json"


class SyncError(RuntimeError):
  pass


def git(repo, *args, check=True):
  result = subprocess.run(["git", *IDENTITY, "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
  if check and result.returncode:
    raise SyncError(result.stderr.strip() or result.stdout.strip() or f"git {args[0]} failed")
  return result


def upstream_origin_payload(repo, upstream, upstream_url, branch="carrot-wip"):
  return {
    "remote": str(upstream_url),
    "branch": branch,
    "commit": upstream,
    "short_commit": git(repo, "rev-parse", "--short", upstream).stdout.strip(),
    "commit_date": git(repo, "show", "--no-patch", "--format='%ct %ci'", upstream).stdout.strip(),
    "commit_datetime": git(repo, "show", "-s", "--date=format:%Y-%m-%d %H:%M:%S", "--format=%cd", upstream).stdout.strip(),
  }


def write_upstream_origin(repo, upstream, upstream_url, branch="carrot-wip"):
  path = Path(repo) / UPSTREAM_ORIGIN_REL
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(upstream_origin_payload(repo, upstream, upstream_url, branch), indent=2) + "\n", encoding="utf-8")
  git(repo, "add", "--", UPSTREAM_ORIGIN_REL)


def clean_checkout(repo):
  if git(repo, "status", "--porcelain").stdout.strip():
    raise SyncError("Checkout is dirty; refusing to overwrite local changes.")
  if git(repo, "rev-parse", "--verify", "-q", "MERGE_HEAD", check=False).returncode == 0:
    raise SyncError("A merge is already in progress.")


@dataclass(frozen=True)
class Candidate:
  original: str
  upstream: str
  candidate: str

  @property
  def changed(self):
    return self.original != self.candidate


def prepare(repo, upstream_url, branch="carrot-wip"):
  clean_checkout(repo)
  git(repo, "check-ref-format", "--branch", branch)
  original = git(repo, "rev-parse", "HEAD").stdout.strip()
  git(repo, "fetch", "--no-tags", "--", upstream_url, f"refs/heads/{branch}")
  upstream = git(repo, "rev-parse", "FETCH_HEAD^{commit}").stdout.strip()
  if git(repo, "merge-base", "--is-ancestor", upstream, original, check=False).returncode == 0:
    return Candidate(original, upstream, original)

  merge = git(repo, "merge", "--no-ff", "--no-commit", upstream, check=False)
  if merge.returncode:
    conflicts = git(repo, "diff", "--name-only", "--diff-filter=U").stdout.strip()
    if git(repo, "rev-parse", "--verify", "-q", "MERGE_HEAD", check=False).returncode == 0:
      git(repo, "merge", "--abort")
    raise SyncError(f"Upstream merge failed; nothing published.\n{conflicts or merge.stderr.strip()}")

  write_upstream_origin(repo, upstream, upstream_url, branch)
  git(repo, "commit", "-m", f"Sync upstream {branch} at {upstream[:12]} with DisableDM patch")
  candidate = git(repo, "rev-parse", "HEAD").stdout.strip()
  return Candidate(original, upstream, candidate)


def publish(repo, state, remote="origin", branch="carrot-wip"):
  clean_checkout(repo)
  git(repo, "check-ref-format", "--branch", branch)
  head = git(repo, "rev-parse", "HEAD").stdout.strip()
  if head != state.candidate:
    raise SyncError("Candidate changed after preparation; refusing to publish.")
  if not state.changed:
    return False
  git(repo, "merge-base", "--is-ancestor", state.original, state.candidate)
  remote_ref = f"refs/heads/{branch}"
  advertised = git(repo, "ls-remote", "--refs", remote, remote_ref).stdout.split()
  if not advertised or advertised[0] != state.original:
    raise SyncError("Target branch changed during validation; a later run must retry.")
  # A normal push also rejects any concurrent update after the check above.
  git(repo, "push", remote, f"HEAD:{remote_ref}")
  return True


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("prepare", "publish"))
  parser.add_argument("--repo", type=Path, default=Path.cwd())
  parser.add_argument("--state", type=Path, required=True)
  parser.add_argument("--upstream-url", default="https://github.com/ajouatom/openpilot.git")
  parser.add_argument("--branch", default="carrot-wip")
  parser.add_argument("--remote", default="origin")
  args = parser.parse_args()
  if args.command == "prepare":
    state = prepare(args.repo, args.upstream_url, args.branch)
    args.state.write_text(json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8")
    print(f"upstream={state.upstream} candidate={state.candidate} changed={state.changed}")
  else:
    state = Candidate(**json.loads(args.state.read_text(encoding="utf-8")))
    print("Published validated update." if publish(args.repo, state, args.remote, args.branch) else "Already up to date; no commit or push.")


if __name__ == "__main__":
  main()
