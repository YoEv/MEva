#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-torch21}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/paths.yaml}"
MODE="${1:-musiceval}"

cd "${PROJECT_ROOT}"

if [[ "${MODE}" == "musiceval" ]]; then
  conda run -n "${TORCH_ENV}" python scripts/data/preprocess_data.py \
    --config "${CONFIG}" \
    musiceval-copy-splits
  echo "[done] MusicEval splits copied to configured data.splits targets."
  exit 0
fi

if [[ "${MODE}" == "pairwise" ]]; then
  if [[ $# -lt 4 ]]; then
    echo "Usage:" >&2
    echo "  bash scripts/run/run_data_preprocess.sh pairwise <pairwise_csv> <out_manifest> <out_split_dir> [col_a] [col_b] [col_outcome]" >&2
    exit 1
  fi
  PAIRWISE_CSV="$2"
  OUT_MANIFEST="$3"
  OUT_SPLIT_DIR="$4"
  COL_A="${5:-item_a}"
  COL_B="${6:-item_b}"
  COL_OUTCOME="${7:-outcome}"

  conda run -n "${TORCH_ENV}" python scripts/data/preprocess_data.py \
    --config "${CONFIG}" \
    pairwise-scale-and-split \
    --pairwise-csv "${PAIRWISE_CSV}" \
    --out-manifest "${OUT_MANIFEST}" \
    --out-split-dir "${OUT_SPLIT_DIR}" \
    --col-a "${COL_A}" \
    --col-b "${COL_B}" \
    --col-outcome "${COL_OUTCOME}"
  echo "[done] Pairwise scaling (BT + MSE) and splits generated."
  exit 0
fi

echo "[error] unknown MODE: ${MODE}" >&2
echo "Use: musiceval | pairwise" >&2
exit 1
