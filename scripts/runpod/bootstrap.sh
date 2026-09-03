#!/usr/bin/env bash

set -euo pipefail

readonly UV_VERSION="0.12.5"
readonly PYTHON_VERSION="3.13.15"
readonly REPOSITORY_DIR="${1:-/workspace/ebs-tft}"

if [[ ! -f "${REPOSITORY_DIR}/pyproject.toml" ]]; then
  echo "Repository not found at ${REPOSITORY_DIR}" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1 \
  || [[ "$(uv --version)" != "uv ${UV_VERSION}"* ]]; then
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
    | env UV_UNMANAGED_INSTALL=/usr/local/bin sh
  hash -r
fi

if [[ "$(uv --version)" != "uv ${UV_VERSION}"* ]]; then
  echo "Expected uv ${UV_VERSION}; found $(uv --version)" >&2
  exit 1
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.local/share/uv/python}"

cd "${REPOSITORY_DIR}"
mkdir -p data/raw
uv python install "${PYTHON_VERSION}"
uv sync \
  --python "${PYTHON_VERSION}" \
  --extra dev \
  --extra pilot \
  --locked

echo "Runpod environment synchronized. No audit or training was started."
echo "Next: uv run python scripts/runpod/verify_environment.py"
