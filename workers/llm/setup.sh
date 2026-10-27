#!/usr/bin/env bash
# Prebuilt attention kernels, so the first start doesn't compile for minutes.
set -uo pipefail
ENV="$1"
"$ENV/bin/vllm" download-kernels || echo "kernel download skipped; vLLM will compile on first start"
