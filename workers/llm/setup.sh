#!/usr/bin/env bash
# Optional prebuilt FlashInfer kernels (used by some vLLM paths, e.g. sampling).
# The default settings run without them, so a failed download is not fatal.
set -uo pipefail
ENV="$1"
PATH="$ENV/bin:$PATH" "$ENV/bin/vllm" download-kernels || echo "WARNING: FlashInfer kernel download failed (optional)"
