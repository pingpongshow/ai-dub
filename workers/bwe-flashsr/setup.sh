#!/usr/bin/env bash
# Installs FlashSR (one-step diffusion audio super-resolution, KAIST) from source.
# $1 = the environment directory. Weights download on first use (prefetch).
set -euo pipefail
ENV="$1"
SRC="$ENV/src/FlashSR_Inference"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 https://github.com/jakeoneijk/FlashSR_Inference.git "$SRC"
fi
uv pip install --python "$ENV/bin/python" torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ENV/bin/python" -e "$SRC" einops librosa soundfile tqdm scipy pyyaml huggingface_hub \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
