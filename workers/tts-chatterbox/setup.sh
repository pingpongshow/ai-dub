#!/usr/bin/env bash
# Chatterbox pins torch 2.6 (no RTX 50-series support); override with a cu128 build.
set -euo pipefail
ENV="$1"
printf 'torch==2.8.0+cu128\ntorchaudio==2.8.0+cu128\n' > "$ENV/overrides.txt"
uv pip install --python "$ENV/bin/python" "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox@5de7a54aa4e5" \
  soundfile --override "$ENV/overrides.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
