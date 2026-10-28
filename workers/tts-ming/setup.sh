#!/usr/bin/env bash
# Ming-3.0-tts from source; its torch 2.6 pin is overridden with a cu128 build.
set -euo pipefail
ENV="$1"
SRC="$ENV/src/Ming-3.0-tts"
[ -d "$SRC/.git" ] || git clone --depth 1 https://github.com/inclusionAI/Ming-3.0-tts.git "$SRC"
printf 'torch==2.8.0+cu128\ntorchaudio==2.8.0+cu128\n' > "$ENV/overrides.txt"
uv pip install --python "$ENV/bin/python" -r "$SRC/requirements.txt" huggingface_hub \
  --override "$ENV/overrides.txt" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
