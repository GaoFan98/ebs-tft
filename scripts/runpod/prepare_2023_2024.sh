#!/usr/bin/env bash

set -euo pipefail

readonly REPOSITORY_DIR="${1:-/workspace/ebs-tft}"
readonly PROTOCOL="notebooks/research_2023_2024_protocol.yaml"
readonly OUTPUT_DIR="notebooks/research_2023_2024_outputs"

cd "${REPOSITORY_DIR}"

if [[ -d "${OUTPUT_DIR}/neural_benchmark" \
  || -d "${OUTPUT_DIR}/locked_evaluation" ]]; then
  echo "Refusing to replace an output tree containing neural evidence." >&2
  exit 1
fi

readonly NORMALIZATION_MANIFEST="data/raw/2023/normalization_2023_EUR_USD.json"

if [[ -f "${NORMALIZATION_MANIFEST}" ]]; then
  uv run --no-sync ebs-tft verify-normalized-ebs \
    --output-dir data/raw/2023 \
    --year 2023 \
    --instrument EUR_USD
else
  uv run --no-sync ebs-tft normalize-consolidated-ebs \
    --source-dir data/raw/2023 \
    --output-dir data/raw/2023 \
    --year 2023 \
    --instrument EUR_USD
fi

uv run --no-sync python scripts/runpod/verify_environment.py \
  --config "${PROTOCOL}"

uv run --no-sync ebs-tft research-session-audit \
  --config "${PROTOCOL}" \
  --replace-output

uv run --no-sync ebs-tft research-model-protocol \
  --config "${PROTOCOL}" \
  --replace-output

uv run --no-sync ebs-tft research-baseline-gate \
  --config "${PROTOCOL}" \
  --replace-output

echo "Longitudinal preflight complete. Inspect the baseline gate before GPU training."
