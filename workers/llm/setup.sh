#!/usr/bin/env bash
# Prebuilt FlashInfer kernels (cubins + JIT cache). The image has no CUDA
# compiler, so without them vLLM cannot run attention on the GPU.
set -euo pipefail
ENV="$1"
PATH="$ENV/bin:$PATH" "$ENV/bin/vllm" download-kernels
