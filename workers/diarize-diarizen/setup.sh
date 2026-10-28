#!/usr/bin/env bash
# DiariZen needs its own vendored pyannote 3.1.1 fork, numpy 1.26 and torch 2.8
# (its audio loading uses torchaudio functions removed in 2.9).
set -euo pipefail
ENV="$1"
SRC="$ENV/src/DiariZen"
[ -d "$SRC/.git" ] || git clone --depth 1 https://github.com/BUTSpeechFIT/DiariZen.git "$SRC"
PY="$ENV/bin/python"
uv pip install --python "$PY" "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" "numpy==1.26.4" "setuptools<82" wheel toml scipy einops soundfile huggingface_hub "lightning<2.7" psutil
uv pip install --python "$PY" --no-build-isolation -e "$SRC/pyannote-audio" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip install --python "$PY" --no-deps -e "$SRC"
