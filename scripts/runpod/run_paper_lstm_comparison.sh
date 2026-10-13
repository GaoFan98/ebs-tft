#!/usr/bin/env bash

set -euo pipefail

readonly STAGE="${1:-development}"
readonly REPOSITORY_DIR="${2:-/workspace/ebs-tft}"
readonly MAXIMUM_NEW_CELLS="${3:-4}"

if [[ "${STAGE}" != "development" && "${STAGE}" != "replication" ]]; then
  echo "stage must be development or replication" >&2
  exit 1
fi
if [[ ! "${MAXIMUM_NEW_CELLS}" =~ ^[1-4]$ ]]; then
  echo "maximum new cells must be an integer from 1 through 4" >&2
  exit 1
fi

cd "${REPOSITORY_DIR}"

uv run --no-sync ebs-tft research-paper-lstm \
  --stage "${STAGE}" \
  --config notebooks/research_2023_2024_protocol.yaml \
  --policy notebooks/research_2023_2024_neural.yaml \
  --maximum-new-cells "${MAXIMUM_NEW_CELLS}" \
  2>&1 | tee -a "notebooks/research_2023_2024_paper_lstm_${STAGE}_terminal.log"
