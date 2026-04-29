#!/usr/bin/env bash
# Deploy HexSimPy to laguna.ku.lt:/srv/shiny-server/HexSimPy
#
# Usage:
#   scripts/deploy_laguna.sh dry-run     # default — show what would change
#   scripts/deploy_laguna.sh apply       # actually push + hot-reload
#
# Transport: `git archive HEAD | ssh ... | tar -x -C DEST`.
# Why git archive + tar (not rsync)?  rsync isn't installed in
# Windows Git Bash by default and we don't want a hard dependency.
# Both git archive and ssh+tar ship with the standard toolchains.
# Trade-off: every deploy re-uploads every (tracked) file — not just
# the diff.  At ≤ 200 files / few MB this is fast enough; switch to
# rsync if the repo grows >50 MB.
#
# What this does:
#   1. Pre-flight: SSH reachability + dest writability + git clean-ish.
#   2. `git archive HEAD --format=tar` — only TRACKED files at HEAD.
#   3. `ssh ... | tar -x` overlays the archive on the server-side dir;
#      NO `--delete`, so HexSim workspace dirs and `data/` survive.
#      `tests/`, `docs/superpowers/`, and bundled HexSim sample
#      workspaces are excluded as runtime-irrelevant.
#   4. `touch restart.txt` to trigger Shiny Server's hot-reload.
#
# Safety notes:
#   * Default mode is `dry-run` — no remote writes happen unless the
#     caller passes `apply` explicitly.
#   * Passwordless SSH is required (the user confirmed the key is
#     installed on the server side).  We don't store credentials.
#   * The script refuses to run if HEAD is not a tagged commit OR the
#     `--allow-untagged` flag is passed; releases should be tagged.
set -euo pipefail

HOST="razinka@laguna.ku.lt"
DEST="/srv/shiny-server/HexSimPy"
URL="https://laguna.ku.lt/HexSimPy/"

# Always operate from the repo root so `git ls-tree` and rsync paths
# line up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Argument parsing --------------------------------------------------
MODE="${1:-dry-run}"
ALLOW_UNTAGGED=0
for arg in "$@"; do
    case "$arg" in
        --allow-untagged) ALLOW_UNTAGGED=1 ;;
    esac
done
case "$MODE" in
    dry-run|apply) ;;
    *)
        echo "Usage: $0 {dry-run|apply} [--allow-untagged]" >&2
        exit 2
        ;;
esac

# --- 1. Pre-flight checks ---------------------------------------------
echo "==> Pre-flight: git state"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: not inside a git repo" >&2
    exit 1
fi
HEAD_SHA="$(git rev-parse --short HEAD)"
HEAD_TAG="$(git describe --tags --exact-match HEAD 2>/dev/null || true)"
if [[ -z "$HEAD_TAG" && "$ALLOW_UNTAGGED" -eq 0 ]]; then
    echo "ERROR: HEAD ($HEAD_SHA) is not a tagged commit." >&2
    echo "       Tag the release first (e.g. \`git tag -a vX.Y.Z -m ...\`)" >&2
    echo "       or pass --allow-untagged to override." >&2
    exit 1
fi
DIRTY="$(git status --porcelain | grep -v '^??' || true)"
if [[ -n "$DIRTY" ]]; then
    echo "WARNING: uncommitted tracked-file changes:"
    echo "$DIRTY" | head -10
    echo "(deploy will use HEAD, not your working tree.)"
fi

echo "==> Pre-flight: SSH + writability"
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new \
    "$HOST" "test -w '$DEST'" \
    || { echo "ERROR: $HOST:$DEST not reachable or not writable" >&2; exit 1; }

# Tracked files we don't need at runtime.  Patterns are matched
# against tar member names (paths relative to the repo root).
RUNTIME_EXCLUDES=(
    --exclude='tests'                 # test suite — not used by Shiny Server
    --exclude='docs/superpowers'      # plan/spec/scratch — not user-facing
    --exclude='HexSim Examples'       # bundled HexSim sample workspaces
    --exclude='HexSim Examples [old]' # ditto
    --exclude='HexSimPLE'             # ditto
)

# --- 2. File-list summary --------------------------------------------
# Count tracked files (informational only; the deploy itself uses
# git archive, which obeys the same set).
N_FILES=$(git ls-tree -r HEAD --name-only | wc -l | tr -d ' ')
echo "==> Files at $HEAD_SHA${HEAD_TAG:+ ($HEAD_TAG)}: $N_FILES tracked"

# --- 3. Push (or dry-run) --------------------------------------------
if [[ "$MODE" == "dry-run" ]]; then
    echo "==> Dry-run: listing tar members that would be extracted"
    echo "    (excluding: ${RUNTIME_EXCLUDES[*]})"
    # `tar -tv` lists archive contents, applying --exclude.  Doesn't
    # touch the server.
    git archive HEAD --format=tar | tar -tv "${RUNTIME_EXCLUDES[@]}" \
        | head -50
    echo "    ... (use \`apply\` to actually push; first 50 lines shown)"
    REMAINING=$(git archive HEAD --format=tar | tar -t "${RUNTIME_EXCLUDES[@]}" | wc -l | tr -d ' ')
    echo "==> $REMAINING entries would be extracted (after exclusions)"
    echo
    echo "==> Dry-run complete.  Re-run with \`apply\` to actually push."
    exit 0
fi

# Apply mode.  Stream the tar and extract atop the existing dir.
# Build the remote command with each exclude properly shell-quoted —
# `${RUNTIME_EXCLUDES[*]}` would space-join and lose word boundaries
# inside `--exclude='HexSim Examples'`, breaking tar.
EXCLUDES_QUOTED=""
for ex in "${RUNTIME_EXCLUDES[@]}"; do
    EXCLUDES_QUOTED+=" $(printf '%q' "$ex")"
done
echo "==> Pushing $N_FILES tracked files to $HOST:$DEST/"
git archive HEAD --format=tar | \
    ssh -o BatchMode=yes "$HOST" \
        "tar -x --wildcards -C $(printf '%q' "$DEST")$EXCLUDES_QUOTED"

# --- 4. Hot-reload + audit trail -------------------------------------
echo "==> touch $DEST/restart.txt (Shiny Server hot-reload)"
ssh -o BatchMode=yes "$HOST" \
    "touch '$DEST/restart.txt' && echo '$HEAD_SHA${HEAD_TAG:+ $HEAD_TAG} $(date -u +%Y-%m-%dT%H:%M:%SZ)' > '$DEST/.deployed_revision'"

echo
echo "==> Done.  Deployed $HEAD_SHA${HEAD_TAG:+ $HEAD_TAG}."
echo "    URL: $URL"
