#!/usr/bin/env bash
# Deploy HexSimPy Shiny app to laguna.ku.lt
# Usage: bash scripts/deploy.sh [--dry-run]
#
# Prerequisites:
#   - Passwordless SSH access as razinka@laguna.ku.lt
#   - /srv/shiny-server/HexSimPy writable by razinka
#   - micromamba 'shiny' env at /opt/micromamba/envs/shiny on server
#
# Uses tar+ssh pipe (avoids rsync DLL issues on Windows Git Bash)

set -euo pipefail

REMOTE_USER="razinka"
REMOTE_HOST="laguna.ku.lt"
REMOTE_DIR="/srv/shiny-server/HexSimPy"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "=== Deploying HexSimPy to ${REMOTE}:${REMOTE_DIR} ==="
echo "Source: ${PROJECT_ROOT}"

# Files/dirs to deploy
INCLUDE=(
    app.py
    run.py
    environment.yml
    CLAUDE.md
    README.md
    config_columbia.yaml
    config_curonian_minimal.yaml
    config_curonian_hexsim.yaml
    salmon_ibm/
    heximpy/
    ui/
    www/
    "Columbia [small]/"
    "Curonian Lagoon/"
    Curonian/
    rivers/
)

# Build tar exclusion flags
EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.git'
    --exclude='.pytest_cache'
    --exclude='heximpy/tests'
)

if $DRY_RUN; then
    echo ""
    echo "[DRY RUN] Would deploy these items:"
    for item in "${INCLUDE[@]}"; do
        if [ -e "${PROJECT_ROOT}/${item}" ]; then
            size=$(du -sh "${PROJECT_ROOT}/${item}" 2>/dev/null | cut -f1)
            echo "  ${item} (${size})"
        else
            echo "  ${item} (not found, skipped)"
        fi
    done
    echo ""
    echo "Target: ${REMOTE}:${REMOTE_DIR}"
    exit 0
fi

# Clean remote directory
echo "Cleaning remote directory..."
ssh "${REMOTE}" "rm -rf ${REMOTE_DIR}/* ${REMOTE_DIR}/.[!.]* 2>/dev/null; true"

# Build list of existing items to tar
TAR_ARGS=()
for item in "${INCLUDE[@]}"; do
    if [ -e "${PROJECT_ROOT}/${item}" ]; then
        TAR_ARGS+=("${item}")
    else
        echo "  [skip] ${item} (not found)"
    fi
done

# Tar locally, pipe through SSH, extract on remote
echo "Transferring files..."
cd "${PROJECT_ROOT}"
tar czf - "${EXCLUDES[@]}" "${TAR_ARGS[@]}" | \
    ssh "${REMOTE}" "cd ${REMOTE_DIR} && tar xzf -"

echo ""
echo "=== Deployed successfully ==="
echo "URL: https://laguna.ku.lt/HexSimPy/"
echo ""

# Verify
ssh "${REMOTE}" "test -f ${REMOTE_DIR}/app.py && echo 'app.py: OK' || echo 'ERROR: app.py not found!'"
ssh "${REMOTE}" "ls ${REMOTE_DIR}/ | head -15"
