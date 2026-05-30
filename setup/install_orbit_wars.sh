#!/usr/bin/env bash
# Install the orbit_wars kaggle environment into the active Python's
# kaggle_environments package. Run this once after setting up a new instance.
#
# Works on any platform (AWS, GCP, local) as long as kaggle-environments is installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/orbit_wars_env"

# Find where kaggle_environments is installed
KE_PATH=$(python3 -c "
import sys, io
# Redirect stdout to suppress open_spiel INFO spam before import
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
import logging; logging.disable(logging.CRITICAL)
import kaggle_environments, os
sys.stdout = _real_stdout
print(os.path.dirname(kaggle_environments.__file__))
" 2>/dev/null)

if [ -z "$KE_PATH" ]; then
  echo "ERROR: kaggle_environments not found. Run: pip install kaggle-environments"
  exit 1
fi

DEST="$KE_PATH/envs/orbit_wars"
mkdir -p "$DEST"
cp "$SRC/orbit_wars.py" "$SRC/orbit_wars.json" "$SRC/orbit_wars.js" "$SRC/README.md" "$DEST/"

echo "Installed orbit_wars env to: $DEST"
python3 -c "
import sys, io, logging
logging.disable(logging.CRITICAL)
_out = sys.stdout; sys.stdout = io.StringIO()
from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets
sys.stdout = _out
print('Import OK')
"