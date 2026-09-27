#!/usr/bin/env bash

set -euo pipefail

readonly REPOSITORY_DIR="${1:-/workspace/ebs-tft}"
readonly LOCKED_PLAN_PATH="notebooks/research_2023_2024_outputs/locked_evaluation/plan.json"
readonly DEPTH_PLAN_PATH="notebooks/research_2023_2024_outputs/depth_replication/plan.json"
readonly DEPTH_SUMMARY_PATH="notebooks/research_2023_2024_outputs/depth_replication/run_summary.json"
readonly REPORT_DIR="reports/2023_2024_analysis"

cd "${REPOSITORY_DIR}"

if [[ ! -f "${REPORT_DIR}/artifact_manifest.json" ]]; then
  echo "Generate and verify the longitudinal report before packaging." >&2
  exit 1
fi
if [[ ! -f "${LOCKED_PLAN_PATH}" ]]; then
  echo "Frozen replication plan not found: ${LOCKED_PLAN_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DEPTH_PLAN_PATH}" || ! -f "${DEPTH_SUMMARY_PATH}" ]]; then
  echo "Complete the frozen depth replication before packaging." >&2
  exit 1
fi

readonly PLAN_SHA256="$(sha256sum "${DEPTH_PLAN_PATH}" | cut -d ' ' -f 1)"
readonly ARCHIVE_PATH="${2:-/workspace/ebs-tft-2023-2024-depth-evidence-${PLAN_SHA256:0:7}.tar.gz}"
readonly TEMPORARY_PATH="${ARCHIVE_PATH}.tmp"

if [[ -e "${ARCHIVE_PATH}" || -e "${TEMPORARY_PATH}" ]]; then
  echo "Refusing to replace an existing evidence archive: ${ARCHIVE_PATH}" >&2
  exit 1
fi

tar \
  --exclude='*/predictions.parquet' \
  --exclude='*/native_cache_level_10/*' \
  --exclude='*/working_corpora/*' \
  -czf "${TEMPORARY_PATH}" \
  notebooks/research_2023_2024_protocol.yaml \
  notebooks/research_2023_2024_neural.yaml \
  notebooks/research_2023_2024_preflight.log \
  notebooks/research_2023_2024_neural_terminal.log \
  notebooks/research_2023_2024_replication_terminal.log \
  notebooks/research_2023_2024_depth_terminal.log \
  notebooks/research_2023_2024_depth_replication_terminal.log \
  notebooks/research_2023_2024_outputs/audit_summary.json \
  notebooks/research_2023_2024_outputs/session_audit.csv \
  notebooks/research_2023_2024_outputs/split_manifest.yaml \
  notebooks/research_2023_2024_outputs/terminal_summary.txt \
  notebooks/research_2023_2024_outputs/baseline_gate \
  notebooks/research_2023_2024_outputs/model_protocol \
  notebooks/research_2023_2024_outputs/neural_benchmark \
  notebooks/research_2023_2024_outputs/locked_evaluation \
  notebooks/research_2023_2024_outputs/depth_extension \
  notebooks/research_2023_2024_outputs/depth_replication \
  "${REPORT_DIR}"

mv "${TEMPORARY_PATH}" "${ARCHIVE_PATH}"
sha256sum "${ARCHIVE_PATH}"
ls -lh "${ARCHIVE_PATH}"
