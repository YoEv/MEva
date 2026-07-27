#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-torch21}"
MUSICDISCOVERY_ENV="${MUSICDISCOVERY_ENV:-torch21}"
AUDIOBOX_ENV="${AUDIOBOX_ENV:-audiobox}"
SPLITS="${SPLITS:-clean}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[launch] starting full parallel runner in env=${TORCH_ENV}, splits=${SPLITS}"
conda run --no-capture-output -n "${TORCH_ENV}" python "scripts/run/run_full_14_experiments_parallel.py" \
  --project-root "${PROJECT_ROOT}" \
  --torch-env "${TORCH_ENV}" \
  --musicdiscovery-env "${MUSICDISCOVERY_ENV}" \
  --audiobox-env "${AUDIOBOX_ENV}" \
  --splits "${SPLITS}" \
  "$@"
