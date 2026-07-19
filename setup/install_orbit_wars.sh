#!/usr/bin/env bash
# Install the orbit_wars kaggle environment into the active Python's
# kaggle_environments package. Run this once after setting up a new instance.
#
# Works on any platform (AWS, GCP, local) as long as kaggle-environments is installed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/orbit_wars_env"

# --- GPU driver / torch compatibility guard ---------------------------------
# Jarvis (and some cloud) images ship a torch built against a CUDA newer than the
# box driver (e.g. torch +cu130 on a driver-570 / CUDA-12.8 box) -> torch silently
# falls back to CPU (cuda.is_available()==False) and training crawls on the A100
# while it sits idle. Detect that case and reinstall a driver-compatible cu128 build.
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "WARNING: GPU present but torch.cuda.is_available()==False — reinstalling cu128 torch."
    TV=$(python3 -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo 2.11.0)
    pip install --force-reinstall "torch==${TV}" torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
    python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
  fi
fi
# ----------------------------------------------------------------------------

# --- wandb: training logging is default-ON (train_torch.py --wandb defaults True), so ensure the
# package is present on every fresh/migrated box. Auth persists in ~/.netrc across pause/resume;
# only the pip install is lost on a spot host migration — this line restores it so a resumed run
# logs training internals without a manual reinstall. Non-fatal: if it fails, the run still trains
# (wandb.init just warns and continues, or pass --no-wandb). ---
pip install -q wandb 2>/dev/null && echo "wandb ready" || echo "WARNING: wandb install failed — run will train without training-side W&B"

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