# Server upload identity (2026-09-28)

The user requested that server-facing uploads report the upstream identity
instead of the fork, without removing any payload keys.

## Reported values

- repository remote: `https://github.com/ajouatom/openpilot.git`
- branch: `carrot-wip`
- commit / commit date: last upstream commit merged into the fork,
  stored in `openpilot/selfdrive/carrot/upstream_origin.json`
  (fallback: the hardcoded last-known upstream commit)
- `DisableDM` and `DisableDMActive`: original default `"0"` while the keys
  stay present in the payloads

The helper is `openpilot/selfdrive/carrot/telemetry_origin.py`.

## Applied upload paths

- `openpilot/selfdrive/carrot/server/services/popular_values.py` — settings
  snapshot (`repo_remote`, `repo_id`, `app_commit`, `values`)
- `openpilot/selfdrive/carrot/carrot_man.py` — `_tmux_upload_payload`,
  `_github_repo_url`, `_tmux_discord_content`, `get_all_toggle_values`
  (the uploaded `toggle_values.json`)
- `openpilot/selfdrive/carrot/server/services/vision_diag.py`,
  `openpilot/selfdrive/carrot/server/services/support_discord.py`,
  `openpilot/selfdrive/carrot/server/features/dashcam/upload.py` —
  diagnostic metadata

Local Params, UI and updater keep the real fork values. Every manual upstream
merge refreshes `upstream_origin.json`, so the reported commit stays a real
upstream commit.

## Retention and verification

On 2026-10-07 the user removed the fork's automatic upstream sync (the
`disabledm-sync` workflow and `tools/disabledm_sync`). Upstream merges are now
manual; each manual merge keeps the DisableDM and upload identity patches,
refreshes `upstream_origin.json` and runs the identity and upload regression
tests before publication, alongside the existing monitoring/web checks and web
build. Conflicts or failed checks stop publication and leave the remote branch
intact.

The former sync integration test used two consecutive updates in temporary Git
repositories and executed the real upload identity helper after each merge; it
was removed with the tooling on 2026-10-07.

Local workflow Python checks: 162 passed, 4 skipped because native params_pyx
is unavailable. The five focused web test files also passed. Ruff introduced
no new diagnostics; 44 existing diagnostics in touched files remain.

## Limits

- Desktop tests only; no vehicle validation.
- Public GitHub fork and PRs remain visible; this only covers uploaded
  metadata.
- `loggerd` qlog `initData` and `GithubUsername` pass through unchanged; if an
  upload path is added or shared logs are published, keep them consistent
  with this policy or extend it.
