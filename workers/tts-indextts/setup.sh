#!/usr/bin/env bash
# Installs IndexTTS-2.5 from its tagged source into the worker environment.
# $1 = the environment directory. Weights are downloaded separately (prefetch).
set -euo pipefail
ENV="$1"
SRC="$ENV/src/index-tts"
TAG="v2.5.0"
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/index-tts/index-tts.git "$SRC"
fi
uv pip install --python "$ENV/bin/python" "torch==2.8.*" "torchaudio==2.8.*" \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ENV/bin/python" -e "$SRC" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python "$ENV/bin/python" soundfile "huggingface_hub[cli]"
