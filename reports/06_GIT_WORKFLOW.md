# 06 — Git Workflow for a Concurrent Multi-Agent Repo

**Verified:** 2026-11-14 ~10:55 UTC on `/home/kozak/mllm/MML` (`origin/main`).
**Tool under test:** `tools/report_commit.py`. Only `--dry-run` was executed; this
agent did **not** commit or push anything.

---

## 1. TL;DR — who does what

| Role | Git rights |
| :--- | :--- |
| Subagent / report writer | Write only assigned paths. **Never** run `git add`, `commit`, `push`, `reset`, `checkout`, `stash`, `rebase` in the shared tree. |
| Orchestrator | At checkpoints, run `tools/report_commit.py` (dry-run first if unsure). This is the only sanctioned commit/push path for reports and tools. |
| Scoreboard updater | `git_sync.py` only at a quiet checkpoint, orchestrator-only (see §8). It is **not** concurrency-safe. |

Rules of thumb:

1. **One committer at a time.** `report_commit.py` takes an advisory `flock` on
   `.git/report_commit.lock`; a second run exits 3 instead of interleaving.
2. **Small, frequent commits.** Commit every ~15–20 min or after each solved task.
3. **Pull immediately before commit.** The tool always fetches and rebases first.
4. **Never force-push.** If a push is rejected, re-fetch/rebase and retry.
5. **Never commit another agent's in-progress output mid-write.** Run at
   checkpoints, when writers say their file is complete.

---

## 2. Allowed staging set — the hard allowlist

`report_commit.py` stages **only**:

```
reports/
tools/
.gitignore        (only if modified)
```

Everything else is left untouched, even if it is untracked and not ignored.
Never committed, by policy and by tooling:

```
work/  .vault/  submissions/  data/  venv/  .venv/  __pycache__/
RESULT.json  .env  .env.*  *.token  gittoken  .sigmoid_token
*.pt *.pth *.safetensors *.onnx *.ckpt *.h5 *.bin *.engine
*.zip *.tar *.gz *.7z *.parquet *.npy *.npz
model weights, datasets, archives, raw task data, caches, logs
any file > 20 MB
```

The preflight also refuses symlinks, token-like filenames, and forbidden
suffixes even if they somehow appear under `reports/` or `tools/`.

---

## 3. Exact commands

```bash
cd /home/kozak/mllm/MML

# 1) Preview: fetches (read-only) and lists exactly what would be staged.
/home/kozak/mllm/py tools/report_commit.py --dry-run

# 2) Commit + push with an explicit message (orchestrator only).
/home/kozak/mllm/py tools/report_commit.py \
  -m "docs(reports): add git workflow and safe commit tool"

# 3) Commit locally, push later.
/home/kozak/mllm/py tools/report_commit.py -m "..." --no-push

# 4) Verify.
git status -sb
git log --oneline -5
```

Flags:

| Flag | Effect |
| :--- | :--- |
| `--dry-run` | Fetch + print the planned staging set, commit message and push target. Does **not** pull, stage, commit, or push. |
| `--no-push` | Commit after rebase, but skip the push. |
| `-m/--message` | Commit message. If omitted, an auto message lists every changed report/tool. |

Exit codes: `0` success/nothing to do, `1` safety refusal, `2` fetch/branch
problem, `3` pull/rebase conflict, `4` commit failure, `5` push failure.

A real run performs, in order:

1. `git fetch origin main`, prints behind/ahead.
2. `git pull --rebase --autostash origin main` if behind.
3. Safety preflight on every in-scope candidate (secrets, size, path).
4. `git add -A -- reports tools .gitignore`.
5. Refuses if any staged file is outside the allowlist or fails preflight.
6. `git commit -m "<msg>" -- <exact in-scope paths>` (a pathspec commit, so
   anything another agent had pre-staged outside the allowlist stays out).
7. `git push origin main`. If rejected as non-fast-forward, it fetches/rebases
   **once** and retries. It never uses `--force`.
8. Prints a compact summary and exits nonzero on any failure.

---

## 4. Secret / size preflight

The tool refuses the whole commit (`exit 1`) if a candidate in `reports/` or
`tools/` looks like it contains:

- GitHub tokens (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`, `github_pat_`),
- JWTs (`eyJ…` three-part tokens),
- AWS keys (`AKIA…`/`ASIA…`, plus `as-…`/`ak-…` style keys with digits),
- PEM/PGP/OpenSSH private keys,
- OpenAI-, Slack-, or Google-style keys,
- or a token-like **filename** (`.sigmoid_token`, `gittoken`, `.env`, `*.token`).

Reports must never paste a token, JWT, cookie, or credential — not even in a
fenced code block. The scanner does not print the matched value, only the file
and line number. Files over **20 MB** are refused outright.

The scanner is a safety net, not a licence to handle secrets carelessly. The
real token files stay outside the repo (`/home/kozak/mllm/.sigmoid_token`,
`/home/kozak/mllm/gittoken`).

---

## 5. Multi-agent write protocol

1. **Own your path.** One agent per file. Do not reformat or rewrite another
   agent's report/tool.
2. **Append, don't rewrite, shared logs.** For `reports/LOG.md` prefer
   append-only entries so concurrent rebases stay trivial.
3. **Keep reports small.** Put large artifacts in `work/` and reference their
   path + SHA-256 in the report. Never put data, weights, or screenshots > 20 MB
   into `reports/`.
4. **Write atomically.** Use the editor/write tool rather than shell redirection
   so a checkpoint never captures a half-written file.
5. **No git mutations from subagents.** No `git add -A`, no `git stash`, no
   `git checkout -- .` in the shared worktree. These can destroy another
   agent's uncommitted work.
6. **Commit at checkpoints.** The orchestrator waits until active writers have
   finished their current file, runs `--dry-run`, then the real run.
7. **One push at a time.** The tool's lock + fetch/rebase/retry handles the
   normal race where another participant pushes between our fetch and push.

---

## 6. Conflict handling

### 6.1 Rebase conflict (our local commits vs upstream)

`report_commit.py` aborts a conflicted rebase and restores the pre-pull branch:

```
[ERROR] pull/rebase did not complete...
[report_commit] conflict detected; aborting the rebase ...
[report_commit] rebase aborted; branch restored.
```

Exit code 3. Nothing is committed or pushed. Resolve manually:

```bash
git status
git diff --name-only --diff-filter=U     # conflicted files
# edit files, remove <<<<<<< / ======= / >>>>>>> markers
git add <file> ...
git rebase --continue
# or give up and restore:
git rebase --abort
```

### 6.2 Autostash conflict (the dangerous case)

`git pull --rebase --autostash` can complete the rebase and then fail to
re-apply the autostash. **`git pull` may still exit 0**, leaving `UU` files and
an `autostash` entry. The tool now checks `git diff --diff-filter=U` even on a
zero exit code.

Observed on 2026-11-14: both sides had edited `.gitignore` (upstream added
`.playwright-mcp/`, `.vault/`, `RESULT.json`, `submissions/`, `*.token`,
`gittoken`; we had added `work/`). The autostash pop left conflict markers.

Recovery:

```bash
git stash list                 # top entry should be the autostash
git stash show -p stash@{0}    # inspect it; no secrets in output
git stash pop                  # re-apply; resolve markers
git add <file> && git stash drop
```

If the tool detects this case it keeps the stash, clears the markers with
`git reset --hard HEAD` (so other agents can keep working), and exits 3. **Your
edits are not lost — they remain in the top autostash.** Recover with
`git stash pop` and merge both sides by hand.

Rule: never blindly choose `--ours`/`--theirs` for a shared file such as
`.gitignore` or `reports/LOG.md`; merge both rule sets/entries.

### 6.3 Push rejected because someone else pushed

The tool fetches, rebases once, and retries automatically. If that retry hits a
conflict it aborts, leaves your commit **local and safe**, and exits 3. Re-run
after resolving. If push still fails with auth/network, exit 5 — nothing was
force-pushed.

### 6.4 Dry-run and pull

`--dry-run` fetches but **does not pull**. It reports “would run pull” if behind.
This is deliberate: dry-run must not mutate the shared working tree. The preview
therefore reflects the current tree; upstream may add files before the real run.

---

## 7. Recovery recipes

```bash
# See where you were / find lost work
git reflog -15
git log --oneline --graph --all -20

# Undo a local commit but keep its changes staged/unstaged
git reset --soft HEAD~1

# Discard a local commit AND its changes (ask the orchestrator first)
git reset --hard HEAD~1

# Undo an already-pushed commit without rewriting history
git revert <sha>
git push origin main

# Restore one deleted/modified file
git restore <path>                 # from index
git checkout HEAD -- <path>        # from last commit

# Recover stashed work
git stash list
git stash pop
```

**Secret committed by mistake:** rotate/revoke the credential first, then treat
it as exposed. If the commit is local only: `git reset --soft HEAD~1`, remove the
file, recommit. If already pushed: notify the orchestrator; history rewriting
(`git filter-repo`) must be coordinated, and the leaked credential must be
rotated regardless. Never paste the secret into chat or a report while doing
this.

**Never** run `git push --force`, `git push -f`, or `git push --force-with-lease`
in this shared repo.

---

## 8. `git_sync.py --dry-run` audit — non-dry run is not concurrency-safe

> **Post-sync correction (2026-11-14 11:30 UTC):** upstream `4e0946f` rewrote
> `git_sync.py` (+171 lines). Subagent L re-tested the new version in a clone:
> `git_sync.py --dry-run` is now **read-only** (README/level hashes unchanged,
> index clean). The old observations below apply to the pre-`4e0946f` revision.
> The **non-dry** path still runs `git add -A`, commits and pushes without a
> fetch/rebase, so the concurrency warning remains for real runs.

`git_sync.py` regenerates the root `README.md` scoreboard and pushes to
`origin/main`. In the pre-`4e0946f` revision its `--dry-run` was **not
read-only**. Inspected and tested in a throwaway detached worktree (so the
shared index was untouched):

```bash
git worktree add --detach /tmp/opencode/gsync-wt HEAD
cd /tmp/opencode/gsync-wt
printf 'scratch file\n' > notes_probe.txt
/home/kozak/mllm/py git_sync.py --dry-run
# -> [git_sync] Regenerated README.md 24-level scoreboard.
# -> [git_sync] Staged 2 file(s): README.md, notes_probe.txt
# -> [git_sync] Dry-run complete. Would commit: 'sync: ... (2 file(s))'
git status --porcelain
# -> M  README.md
# -> A  notes_probe.txt     (both left staged)
```

Findings:

| Behaviour | `git_sync.py` | `report_commit.py` |
| :--- | :--- | :--- |
| `--dry-run` side effects | Rewrites `README.md`, runs `git add -A`, leaves index dirty | fetch only; status byte-identical before/after |
| Staging scope | **everything non-ignored** (`add -A`) | `reports/`, `tools/`, `.gitignore` only |
| Fetch/rebase before commit | No | Yes (`--rebase --autostash`) |
| Concurrent push race | Push simply rejected; no retry | Rebase + one retry |
| Secret scan | Only a filename check for `.env` | Token/JWT/AWS/private-key/keyword regexes |
| Size limit | 50 MB | 20 MB |
| Locking | None | `flock` |

**Verdict:** do **not** run `git_sync.py --dry-run` concurrently with other
agents. Its `git add -A` can capture any agent's in-progress untracked file, its
README rewrite races with other writers, and its real run pushes without first
rebasing.

**When to use it:** only the orchestrator, only at a quiet checkpoint, only when
the 24-level scoreboard/`README.md` actually needs regeneration, and ideally
after `report_commit.py --dry-run` is clean. Otherwise use
`report_commit.py`. If you only need to commit (no scoreboard update), pass
`--no-readme` — but it still stages everything non-ignored.

---

## 9. `.gitignore` / remote hygiene observations

- Upstream `.gitignore` now covers `work/`, `.vault/`, `RESULT.json`,
  `submissions/`, `*.token`, `gittoken`, `orchestrator.log`, `.playwright-mcp/`,
  plus the usual weights/data/venv patterns.
- **Gap:** `.sigmoid_token` does **not** match `*.token` (it ends in `_token`,
  not `.token`). It currently lives outside the repo, so it is not at risk
  today — but if anyone copies it in, `git_sync.py`'s `add -A` would stage it.
  Recommendation: add `.sigmoid_token` and `*_token` to `.gitignore`
  (orchestrator-owned change).
- `origin` currently embeds a PAT in its URL (`x-access-token:…@github.com`).
  It is not committed, but it shows up in `git remote -v` and some error output.
  Recommendation: `git remote set-url origin https://github.com/Helloguys3324/MML.git`
  plus a credential helper, and rotate the PAT if it was ever exposed. The tool
  redacts URL credentials and token-shaped strings from all git output, but it
  cannot protect commands run outside it.

---

## 10. Verification log (no push performed)

```text
$ /home/kozak/mllm/py tools/report_commit.py --dry-run
behind origin/main by 0 commit(s), ahead by 0
changed paths: 14 total, 14 in scope, 0 outside scope
ignored protected entries: .vault/, RESULT.json, __pycache__/, submissions/, work/
would stage 14 file(s): .gitignore, reports/*, tools/*
would push to origin/main
DRY RUN OK - nothing staged, committed, or pushed.        exit 0
```

- **Status unchanged:** `git status --porcelain` was byte-identical before and
  after the dry-run.
- **End-to-end secret refusal:** a synthetic `ghp_`-shaped string in
  `tools/scratch_secret_test.txt` made the dry-run exit 1, naming only the file
  and line. The file was then deleted; the secret value was never printed.
- **Unit checks:** synthetic GitHub, JWT, `as-`, `ak-`, AWS, PEM, OpenAI, Slack,
  and Google patterns were all flagged; prose like “as-described” was clean; a
  21 MB file was refused.
- **Blocked-path test:** with dummy `.sigmoid_token`, `gittoken`, `work/`,
  `submissions/`, and `data/` entries present, the dry-run reported
  `14 in scope, 2 outside scope` and listed only reports/tools/.gitignore in
  the staging set. None of the blocked paths appeared under “would stage”.
- **Earlier implementation bug found and fixed:** the first `--dry-run` version
  pulled before checking the flag, which surfaced the `.gitignore` autostash
  conflict in §6.2. Dry-run is now fetch-only, and autostash conflicts are
  detected even when `git pull` exits 0.
- **Concurrent writers confirmed:** `reports/01_SITE_API_RECON.md` and
  `tools/validate_submission.py` appeared during the test window, so the repo
  really is being edited by multiple agents right now.

**No real commit, push, or force-push was made by this agent.**

---

## 11. Next actions for the orchestrator

1. When writers are idle, run one real `report_commit.py -m "..."` to land the
   current `reports/` + `tools/` batch (it is currently up to date with origin).
2. Add `.sigmoid_token` / `*_token` to `.gitignore`, then commit it through
   `report_commit.py` (it stages `.gitignore`).
3. Replace the token-embedded `origin` URL with a credential-helper URL.
4. Keep `git_sync.py` for quiet-time scoreboard regeneration only; never run it
   while other agents are writing.
