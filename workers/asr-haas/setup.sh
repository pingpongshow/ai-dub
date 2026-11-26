#!/usr/bin/env bash
# Installs Haas-ASR from its tested commit, with the torch / triton it was tested with
# (flash-linear-attention 0.5.2 breaks on triton 3.9). $1 = the environment directory.
set -euo pipefail
ENV="$1"
COMMIT="acc614f68769dcbfae9dba96b48439c4a66295af"
uv pip install --python "$ENV/bin/python" "haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git@$COMMIT" \
  "torch==2.14.0" "triton==3.8.0" soundfile "huggingface_hub[cli]"
