#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-torch21}"
RELEASE_CFG="${PROJECT_ROOT}/config/paths.yaml"
OUT_DIR="outputs/reports/segment_curves"
CKPT_DIR="outputs/checkpoints/segment"

SPLITS_FLAG="clean"
INPUT_MODE="loss"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-entropy)
      INPUT_MODE="loss_entropy"
      shift
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      echo "usage: $0 [--with-entropy]" >&2
      exit 1
      ;;
  esac
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ENT_ARGS=()
if [[ "${INPUT_MODE}" == "loss_entropy" ]]; then
  ENT_ARGS+=(
    --entropy-manifest-csv "outputs/features/entropy/entropy_manifest_${SPLITS_FLAG}_train.csv"
    --entropy-manifest-csv "outputs/features/entropy/entropy_manifest_${SPLITS_FLAG}_val.csv"
    --entropy-manifest-csv "outputs/features/entropy/entropy_manifest_${SPLITS_FLAG}_test.csv"
  )
fi

echo "[stage 1/2] train segment rnn"
conda run -n "${TORCH_ENV}" python "analysis/segment/train_rnn.py" \
  --config "${RELEASE_CFG}" \
  --splits "${SPLITS_FLAG}" \
  --input-mode "${INPUT_MODE}" \
  --run-name "segment_rnn" \
  "${ENT_ARGS[@]}"

echo "[stage 2/2] predict segment curves on test"
conda run -n "${TORCH_ENV}" python "analysis/segment/predict_curves.py" \
  --config "${RELEASE_CFG}" \
  --splits "${SPLITS_FLAG}" \
  --split test \
  --checkpoint "${CKPT_DIR}/segment_rnn_${SPLITS_FLAG}_${INPUT_MODE}.pth" \
  --out-dir "${OUT_DIR}" \
  "${ENT_ARGS[@]}"

echo "[done] segment rnn analysis done."
