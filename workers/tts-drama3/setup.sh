#!/usr/bin/env bash
# Installs fish-speech (Drama 3) from its tested commit. $1 = the environment directory.
# pyaudio (a microphone library for its web demo) is left out: it needs system headers.
set -euo pipefail
ENV="$1"
SRC="$ENV/src/fish-speech"
COMMIT="0f8a12e8a651062fdee9c33a894bdccd3a8513f1"
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/fishaudio/fish-speech.git "$SRC"
fi
git -C "$SRC" fetch -q origin "$COMMIT" || true
git -C "$SRC" checkout -q "$COMMIT"
printf 'pyaudio ; sys_platform == "never"\n' > "$ENV/overrides.txt"
uv pip install --python "$ENV/bin/python" --override "$ENV/overrides.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match -e "$SRC[cu128]"
uv pip install --python "$ENV/bin/python" "protobuf>=6.31.1" soundfile "huggingface_hub[cli]"
