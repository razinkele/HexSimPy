#!/usr/bin/env bash
# Deploy HexSimPy Shiny app to laguna.ku.lt
# Usage: bash scripts/deploy.sh [--dry-run]
#
# Prerequisites:
#   - Passwordless SSH access as razinka@laguna.ku.lt
#   - /srv/shiny-server/HexSimPy writable by razinka
#   - micromamba 'shiny' env at /opt/micromamba/envs/shiny on server

set -euo pipefail

REMOTE_USER="razinka"
REMOTE_HOST="laguna.ku.lt"
REMOTE_DIR="/srv/shiny-server/HexSimPy"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[DRY RUN] No files will be transferred."
fi

echo "=== Deploying HexSimPy to ${REMOTE}:${REMOTE_DIR} ==="
echo "Source: ${PROJECT_ROOT}"

# rsync app code, simulation engine, UI, static assets, configs, and data
rsync -avz --delete $DRY_RUN \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude '.gitignore' \
    --exclude '.claude' \
    --exclude '.superpowers' \
    --exclude '.pytest_cache' \
    --exclude 'tests/' \
    --exclude 'scripts/' \
    --exclude 'docs/' \
    --exclude 'refs/' \
    --exclude 'hexsimlab/' \
    --exclude 'heximpy/tests/' \
    --exclude 'HexSim 4.0.20/' \
    --exclude 'HexSim Examples/' \
    --exclude 'HexSim Examples [old]/' \
    --exclude 'HexSimPLE/' \
    --exclude 'HexSimPLE.zip' \
    --exclude 'HexSimR-master.zip' \
    --exclude 'HexSimR-manual.pdf' \
    --exclude 'HexSim.zip' \
    --exclude 'HexSim.py.txt' \
    --exclude 'CaliforniaGnatcatcher/' \
    --exclude 'NorthernSpottedOwl/' \
    --exclude 'ManedWolf/' \
    --exclude 'TestWorkspace/' \
    --exclude 'migration_corridor_simulation_model-master*' \
    --exclude 'Schumaker_2024*' \
    --exclude 'Columbia River Migration Model.zip' \
    --exclude 'river_extent.zip' \
    --exclude '*.zip' \
    --exclude '*.bmp.aux.xml' \
    --exclude '*.bpw' \
    --exclude '*.qgz' \
    --exclude '*.c' \
    --exclude 'README.pdf' \
    --exclude 'docs/parity-report*.md' \
    "${PROJECT_ROOT}/" \
    "${REMOTE}:${REMOTE_DIR}/"

echo ""
echo "=== Deployed successfully ==="
echo "URL: https://laguna.ku.lt/HexSimPy/"
echo ""

# Verify the app entry point exists on server
ssh "${REMOTE}" "test -f ${REMOTE_DIR}/app.py && echo 'app.py: OK' || echo 'ERROR: app.py not found!'"
