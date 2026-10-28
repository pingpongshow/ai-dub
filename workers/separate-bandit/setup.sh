#!/usr/bin/env bash
# Bandit v2 model code only (training requirements and Netflix-internal packages are not needed).
set -euo pipefail
ENV="$1"
SRC="$ENV/src/bandit-v2"
[ -d "$SRC/.git" ] || git clone --depth 1 https://github.com/kwatcharasupat/bandit-v2.git "$SRC"
uv pip install --python "$ENV/bin/python" "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ENV/bin/python" numpy librosa soundfile tqdm pandas httpx
