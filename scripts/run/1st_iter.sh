#!/usr/bin/env bash
# 1st-iter end-to-end runner: applies all §6.2 steps of
# doc/plan/1st_iter_plan.md in order.
#
# Stages:
#   (0) clean stale artifacts for 4 single DBs + all_5_datasets (never touches
#       MusicEval; never touches MusicPref features because the 30s window is
#       unchanged).
#   (1) re-score (Elo) + gen_full_splits + merge_all_datasets (CPU only).
#   (2a) full parallel runner, scope=large_scale_single
#        -> extract features for AIME / MusicArena / SongEval (MusicPref is a
#           no-op) and train 7 CNNs per DB.
#   (2b) full parallel runner, scope=large_scale_merged, skip_feature_extract
#        -> train 7 CNNs on all_5_datasets, reusing per-DB features.
#   (3) self-check: verify summary tables + test score CSVs are present.
#
# Stage (2b) depends on a small patch described in §6.3.1 that makes
# ``run_full_14_experiments_parallel.py::_build_jobs`` point the merged job's
# feature roots at the **parent** ``outputs/full/features/{loss,entropy,sae}/``
# directory instead of ``.../all_5_datasets/clean/``. Without that patch the
# merged training will not find per-DB features, and this script will warn +
# exit non-zero before launching stage (2b).

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-torch21}"
MUSICDISCOVERY_ENV="${MUSICDISCOVERY_ENV:-musicdiscovery310}"
AUDIOBOX_ENV="${AUDIOBOX_ENV:-audiobox}"
SPLITS="${SPLITS:-clean}"

SKIP_CLEAN=0
SKIP_SCORING=0
SKIP_SINGLES=0
SKIP_MERGED=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: 1st_iter.sh [options]

Environment overrides:
  PROJECT_ROOT          Repo root (auto-detected if unset).
  TORCH_ENV             conda env for rescoring + entropy/loss extract + training (default: torch21).
  MUSICDISCOVERY_ENV    conda env for SAE extract (default: musicdiscovery310).
  AUDIOBOX_ENV          conda env for audiobox baseline (default: audiobox).
  SPLITS                split tag forwarded to the parallel runner (default: clean).

Options:
  --skip-clean          Skip stage (0): clean stale artifacts.
  --skip-scoring        Skip stage (1): rescore + gen_full_splits + merge.
  --skip-singles        Skip stage (2a): parallel run for 4 single DBs.
  --skip-merged         Skip stage (2b): parallel run for all_5_datasets.
  --dry-run             Print commands that would run, but do not execute.
  -h, --help            Show this help.

Examples:
  # First run after fresh pull, full pipeline:
  bash scripts/run/1st_iter.sh

  # Re-run only the single-DB parallel stage (skip everything else):
  bash scripts/run/1st_iter.sh --skip-clean --skip-scoring --skip-merged

  # Re-run only the merged stage (requires §6.3.1 patch):
  bash scripts/run/1st_iter.sh --skip-clean --skip-scoring --skip-singles
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-clean)    SKIP_CLEAN=1; shift ;;
    --skip-scoring)  SKIP_SCORING=1; shift ;;
    --skip-singles)  SKIP_SINGLES=1; shift ;;
    --skip-merged)   SKIP_MERGED=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "[error] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

STATE_DIR="outputs/run_state/1st_iter"
mkdir -p "${STATE_DIR}"
FULL_CFG="config/data/full_datasets.yaml"
SINGLE_OVERLAY="${STATE_DIR}/full_datasets_single.yaml"
MERGED_OVERLAY="${STATE_DIR}/full_datasets_merged.yaml"

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run]" "$@"
  else
    echo "[run]" "$@"
    "$@"
  fi
}

run_sh() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] bash -c: $*"
  else
    echo "[run] bash -c: $*"
    bash -c "$*"
  fi
}

# -------------------------------------------------------------------- stage 0
if [[ "${SKIP_CLEAN}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 0/4] clean stale manifests / splits / features / reports"
  echo "======================================================================"
  # 0.1 old score manifests (overwritten ones + permanently removed fidelity / alignment)
  run_sh 'rm -f data/manifests/pairwise_relu/{musicpref_musicality,aime_music_quality,musicarena}_1to5.csv'
  run_sh 'rm -f data/manifests/pairwise_relu/{musicpref_fidelity,aime_text_audio_alignment}_1to5.csv'

  # 0.2 old full_splits (MusicEval retained)
  for ds in musicpref aime music_arena songeval all_5_datasets; do
    run_sh "rm -rf data/full_splits/${ds}"
  done

  # 0.3 old features: AIME / MusicArena / SongEval / all_5 only.
  #     MusicPref keeps its 30s features (window unchanged).
  for ds in aime music_arena songeval all_5_datasets; do
    run_sh "rm -rf outputs/full/features/loss/${ds}"
    run_sh "rm -rf outputs/full/features/entropy/${ds}"
    run_sh "rm -rf outputs/full/features/sae/${ds}"
  done

  # 0.4 old training products + run_state for 5 datasets (everything gets re-trained).
  for ds in musicpref aime music_arena songeval all_5_datasets; do
    run_sh "rm -rf outputs/full/checkpoints/${ds}"
    run_sh "rm -rf outputs/full/plots/${ds}"
    run_sh "rm -rf outputs/full/reports/${ds}"
    run_sh "rm -rf outputs/full/logs/${ds}"
    run_sh "rm -f  outputs/run_state/full14_${ds}_${SPLITS}.state.json"
    run_sh "rm -f  outputs/run_state/full14_${ds}_${SPLITS}.lock.json"
  done

  # 0.5 Clean stale overlay configs and generated configs
  run_sh "rm -rf outputs/run_state/1st_iter"
  run_sh "rm -rf outputs/run_state/full_generated_configs"
  run_sh "rm -rf outputs/run_state/full_parallel_logs"
  run_sh "rm -rf outputs/run_state/runtime_configs"
  run_sh "rm -rf outputs/run_state/runtime_splits"
  run_sh "rm -f  outputs/run_state/full14_aime_clean.state.json"
  run_sh "rm -f  outputs/run_state/full14_music_arena_clean.state.json"
  run_sh "rm -f  outputs/run_state/full14_musicpref_clean.state.json"
  run_sh "rm -f  outputs/run_state/full14_songeval_clean.state.json"
else
  echo "[stage 0/4] SKIPPED (--skip-clean)"
fi

# -------------------------------------------------------------------- stage 1
if [[ "${SKIP_SCORING}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 1/4] rescore + gen_full_splits + merge_all_datasets (CPU only)"
  echo "======================================================================"
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/fit_pairwise_manifests.py \
    --dataset musicpref --head musicality \
    --out data/manifests/pairwise_relu/musicpref_musicality_1to5.csv
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/fit_pairwise_manifests.py \
    --dataset aime --head music_quality \
    --out data/manifests/pairwise_relu/aime_music_quality_1to5.csv
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/fit_pairwise_manifests.py \
    --dataset musicarena \
    --out data/manifests/pairwise_relu/musicarena_1to5.csv
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/gen_full_splits.py
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/merge_all_datasets.py
else
  echo "[stage 1/4] SKIPPED (--skip-scoring)"
fi

# Helper: write two overlay copies of full_datasets.yaml.
# We never mutate the committed config; each overlay is a full self-contained
# copy with execution.{include_scopes,skip_feature_extract} overridden.
if [[ "${SKIP_SINGLES}" == "0" || "${SKIP_MERGED}" == "0" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] generate overlay yamls: ${SINGLE_OVERLAY}, ${MERGED_OVERLAY}"
  else
    conda run --no-capture-output -n "${TORCH_ENV}" python - <<PY
import copy, os, sys
import yaml

src = os.path.join("${PROJECT_ROOT}", "${FULL_CFG}")
single_dst = os.path.join("${PROJECT_ROOT}", "${SINGLE_OVERLAY}")
merged_dst = os.path.join("${PROJECT_ROOT}", "${MERGED_OVERLAY}")

with open(src, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

single_cfg = copy.deepcopy(base)
single_cfg.setdefault("execution", {})
single_cfg["execution"]["include_scopes"] = ["large_scale_single"]
single_cfg["execution"]["skip_feature_extract"] = False

merged_cfg = copy.deepcopy(base)
merged_cfg.setdefault("execution", {})
merged_cfg["execution"]["include_scopes"] = ["large_scale_merged"]
merged_cfg["execution"]["skip_feature_extract"] = True

os.makedirs(os.path.dirname(single_dst), exist_ok=True)
for dst, cfg in [(single_dst, single_cfg), (merged_dst, merged_cfg)]:
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    print(f"[overlay] wrote {dst}")
PY
  fi
fi

# ------------------------------------------------------------------- stage 2a
if [[ "${SKIP_SINGLES}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 2a/4] parallel runner: 4 single DBs (extract + train)"
  echo "======================================================================"
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/run/run_full_14_experiments_parallel.py \
    --project-root "${PROJECT_ROOT}" \
    --torch-env "${TORCH_ENV}" \
    --musicdiscovery-env "${MUSICDISCOVERY_ENV}" \
    --audiobox-env "${AUDIOBOX_ENV}" \
    --full-config "${SINGLE_OVERLAY}" \
    --splits "${SPLITS}"
else
  echo "[stage 2a/4] SKIPPED (--skip-singles)"
fi

# ------------------------------------------------------------------- stage 2b
if [[ "${SKIP_MERGED}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 2b/4] parallel runner: all_5_datasets (reuse features, train only)"
  echo "======================================================================"

  # Sanity guard for the §6.3.1 patch (feature-reuse for all_5_datasets).
  patch_token='if name == "all_5_datasets"'
  if ! grep -q "${patch_token}" \
       "${PROJECT_ROOT}/scripts/run/run_full_14_experiments_parallel.py"; then
    echo ""
    echo "[warn] stage 2b requires the feature-reuse patch from 1st_iter_plan §6.3.1:"
    echo "       run_full_14_experiments_parallel.py::_build_jobs must override the"
    echo "       all_5_datasets feature roots to point at the shared parent"
    echo "       outputs/full/features/{loss,entropy,sae}/ directory."
    echo "       Without that patch the merged training cannot find per-DB features."
    echo ""
    echo "[warn] skipping stage 2b. Re-run with --skip-clean --skip-scoring --skip-singles"
    echo "       after applying the patch."
    exit 3
  fi

  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/run/run_full_14_experiments_parallel.py \
    --project-root "${PROJECT_ROOT}" \
    --torch-env "${TORCH_ENV}" \
    --musicdiscovery-env "${MUSICDISCOVERY_ENV}" \
    --audiobox-env "${AUDIOBOX_ENV}" \
    --full-config "${MERGED_OVERLAY}" \
    --splits "${SPLITS}"
else
  echo "[stage 2b/4] SKIPPED (--skip-merged)"
fi

# --------------------------------------------------------------------- stage 3
echo "======================================================================"
echo "[stage 3/4] self-check: summary tables + test score CSVs"
echo "======================================================================"

missing=0
for ds in musicpref aime music_arena songeval all_5_datasets; do
  summary="outputs/full/reports/${ds}/${SPLITS}/tables/aggregate_table_14_experiments_${SPLITS}.md"
  if [[ -f "${summary}" ]]; then
    echo "[ok] ${summary}"
  else
    echo "[miss] ${summary}"
    missing=$((missing + 1))
  fi
  found=$(ls "outputs/full/reports/${ds}/${SPLITS}"/f0?_*_cnn_"${SPLITS}"_test_scores.csv 2>/dev/null | wc -l || true)
  echo "       test_scores count: ${found} (expected 7)"
  if [[ "${found}" != "7" ]]; then
    missing=$((missing + 1))
  fi
done

echo "======================================================================"
if [[ "${missing}" == "0" ]]; then
  echo "[done] 1st iter finished; all 5 datasets produced summary + 7 CNN test score CSVs."
else
  echo "[partial] ${missing} check(s) failed; see logs under outputs/run_state/"
  exit 1
fi
