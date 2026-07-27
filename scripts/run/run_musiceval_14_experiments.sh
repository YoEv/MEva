#!/usr/bin/env bash
set -euo pipefail

# Auto-detect project root from script location (repo_root = 3 levels up from this script).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-torch21}"
MUSICDISCOVERY_ENV="${MUSICDISCOVERY_ENV:-torch21}"
AUDIOBOX_ENV="${AUDIOBOX_ENV:-audiobox}"
SPLITS="${SPLITS:-clean}"
DATASET="${DATASET:-musiceval}"
SKIP_SAE="${SKIP_SAE:-0}"
SKIP_AESTHETICS="${SKIP_AESTHETICS:-0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[launch] starting musiceval14 runner in env=${TORCH_ENV}, dataset=${DATASET}, splits=${SPLITS}"
EXTRA_ARGS=()
if [[ "${SKIP_SAE}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-sae)
fi
if [[ "${SKIP_AESTHETICS}" == "1" ]]; then
  EXTRA_ARGS+=(--no-with-aesthetics)
fi
conda run --no-capture-output -n "${TORCH_ENV}" python "scripts/run/run_musiceval_14_experiments.py" \
  --project-root "${PROJECT_ROOT}" \
  --torch-env "${TORCH_ENV}" \
  --musicdiscovery-env "${MUSICDISCOVERY_ENV}" \
  --audiobox-env "${AUDIOBOX_ENV}" \
  --dataset "${DATASET}" \
  --splits "${SPLITS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
