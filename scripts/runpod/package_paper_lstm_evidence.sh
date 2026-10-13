#!/usr/bin/env bash

set -euo pipefail

readonly REPOSITORY_DIR="${1:-/workspace/ebs-tft}"
readonly EVIDENCE_DIR="notebooks/research_2023_2024_outputs/paper_lstm_comparison"

cd "${REPOSITORY_DIR}"

for stage in development replication; do
  if [[ ! -f "${EVIDENCE_DIR}/${stage}/run_summary.json" ]]; then
    echo "paper LSTM ${stage} is incomplete" >&2
    exit 1
  fi
done

readonly IDENTITY_SHA256="$(sha256sum "${EVIDENCE_DIR}/replication/run_identity.json" | cut -d ' ' -f 1)"
readonly ARCHIVE_PATH="${2:-/workspace/ebs-tft-paper-lstm-evidence-${IDENTITY_SHA256:0:7}.tar.gz}"
readonly TEMPORARY_PATH="${ARCHIVE_PATH}.tmp"

if [[ -e "${ARCHIVE_PATH}" || -e "${TEMPORARY_PATH}" ]]; then
  echo "refusing to replace an existing evidence archive: ${ARCHIVE_PATH}" >&2
  exit 1
fi

tar --exclude='*/native_cache/*' -czf "${TEMPORARY_PATH}" \
  notebooks/research_2023_2024_protocol.yaml \
  notebooks/research_2023_2024_neural.yaml \
  notebooks/research_2023_2024_paper_lstm_development_terminal.log \
  notebooks/research_2023_2024_paper_lstm_replication_terminal.log \
  notebooks/research_2023_2024_outputs/split_manifest.yaml \
  notebooks/research_2023_2024_outputs/neural_benchmark/session_metrics.csv \
  notebooks/research_2023_2024_outputs/locked_evaluation/session_metrics.csv \
  "${EVIDENCE_DIR}"

mv "${TEMPORARY_PATH}" "${ARCHIVE_PATH}"
sha256sum "${ARCHIVE_PATH}"
ls -lh "${ARCHIVE_PATH}"
