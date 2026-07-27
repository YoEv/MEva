#!/usr/bin/env bash
# 2nd-iter runner: AIME audio filename fix (Option B) + AIME-only retrain.
#
# Motivation:
#   The 1st iteration ran on top of ``hf_ingest_smoke.py`` that named AIME
#   WAVs by HF row index (``AIME2025_0.wav``, ``AIME2025_1.wav`` ...) while
#   ``aime_join_survey.py`` keys tracks by the HF ``id`` field (e.g.
#   ``"05331"``).  Unless the dataset happens to be sorted by ``id``, audio
#   and labels are decoupled and AIME correlation collapses to ~0.3.
#
#   Option B (applied in this repo):
#     * ``hf_ingest_smoke.py`` now uses ``row["id"]`` as the filename stem
#       so AIME emits ``AIME2025_<id>.wav``.
#     * ``gen_full_splits.py::_build_aime`` reads ``track_id`` as string,
#       and ``_aime_audio_path`` tries both 5-digit zfill and bare int forms
#       so the resolver is robust to either HF id convention.
#
#   This script performs the one-shot server-side delta that Option B
#   requires: re-ingest AIME audio, wipe only the AIME (and all_5_datasets)
#   artifacts produced by the 1st iter, rebuild AIME splits/manifests, then
#   retrain AIME single + all_5 merged.  MusicPref / MusicArena / SongEval /
#   MusicEval artifacts are **not** touched; they remain valid.
#
# Stages:
#   (0) re-ingest AIME audio (id-aware).
#   (1) clean only AIME + all_5_datasets artifacts.
#   (2) rescore AIME + gen_full_splits (all DBs; only AIME rewrite matters)
#       + merge_all_datasets.
#   (3a) parallel runner, scope=large_scale_single, only AIME enabled in the
#        overlay.
#   (3b) parallel runner, scope=large_scale_merged, skip_feature_extract
#        (all_5_datasets reuses per-DB features).
#   (4) self-check: AIME + all_5 summary tables + test score CSVs.
#
# Stage (3b) depends on the §6.3.1 feature-reuse patch (same as 1st iter).
#
# Usage:
#   bash scripts/run/2nd_iter.sh                # full delta
#   bash scripts/run/2nd_iter.sh --skip-ingest  # keep current audio
#   bash scripts/run/2nd_iter.sh --skip-merged  # AIME single only
#   bash scripts/run/2nd_iter.sh --dry-run      # print commands

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-torch21}"
MUSICDISCOVERY_ENV="${MUSICDISCOVERY_ENV:-musicdiscovery310}"
AUDIOBOX_ENV="${AUDIOBOX_ENV:-audiobox}"
SPLITS="${SPLITS:-clean}"

SKIP_INGEST=0
SKIP_CLEAN=0
SKIP_SCORING=0
SKIP_SINGLES=0
SKIP_MERGED=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: 2nd_iter.sh [options]

Environment overrides:
  PROJECT_ROOT          Repo root (auto-detected if unset).
  TORCH_ENV             conda env for ingest + rescoring + entropy/loss extract + training (default: torch21).
  MUSICDISCOVERY_ENV    conda env for SAE extract (default: musicdiscovery310).
  AUDIOBOX_ENV          conda env for audiobox baseline (default: audiobox).
  SPLITS                split tag forwarded to the parallel runner (default: clean).

Options:
  --skip-ingest         Skip stage (0): re-ingest AIME audio.
  --skip-clean          Skip stage (1): clean AIME + all_5 artifacts.
  --skip-scoring        Skip stage (2): AIME rescore + gen_full_splits + merge.
  --skip-singles        Skip stage (3a): parallel AIME-only single run.
  --skip-merged         Skip stage (3b): parallel all_5_datasets run.
  --dry-run             Print commands that would run, but do not execute.
  -h, --help            Show this help.

Examples:
  # Full 2nd-iter delta after pulling the Option B fix:
  bash scripts/run/2nd_iter.sh

  # Audio already re-ingested; only rerun AIME + all_5 training:
  bash scripts/run/2nd_iter.sh --skip-ingest

  # Just redo AIME single (skip merged):
  bash scripts/run/2nd_iter.sh --skip-merged
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-ingest)   SKIP_INGEST=1; shift ;;
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

STATE_DIR="outputs/run_state/2nd_iter"
mkdir -p "${STATE_DIR}"
FULL_CFG="config/data/full_datasets.yaml"
SINGLE_OVERLAY="${STATE_DIR}/full_datasets_single_aime_only.yaml"
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
# Re-ingest AIME audio using the id-aware hf_ingest_smoke.py.  This wipes
# ``datasets/aime/audio/`` so the old row-index-named WAVs
# cannot be picked up accidentally.
if [[ "${SKIP_INGEST}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 0/4] re-ingest AIME audio (id-aware filenames)"
  echo "======================================================================"
  run_sh 'rm -rf datasets/aime/audio'
  run_sh 'mkdir -p datasets/aime/audio'
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/hf_ingest_smoke.py \
    --repo disco-eth/AIME --source-tag AIME2025 \
    --out-audio-dir datasets/aime/audio \
    --master-csv data/manifests/master_index.csv \
    --max-samples 0
else
  echo "[stage 0/4] SKIPPED (--skip-ingest)"
fi

# -------------------------------------------------------------------- stage 1
# Clean only AIME + all_5_datasets artifacts.  MusicPref / MusicArena /
# SongEval / MusicEval products from the 1st iteration stay intact.
if [[ "${SKIP_CLEAN}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 1/4] clean AIME + all_5_datasets stale artifacts"
  echo "======================================================================"
  # 1.1 AIME score manifest (will be re-written by fit_pairwise_manifests).
  run_sh 'rm -f data/manifests/pairwise_relu/aime_music_quality_1to5.csv'

  # 1.2 full_splits for AIME + all_5 (all_5 includes AIME so must rebuild).
  for ds in aime all_5_datasets; do
    run_sh "rm -rf data/full_splits/${ds}"
  done

  # 1.3 features for AIME + all_5.
  for ds in aime all_5_datasets; do
    run_sh "rm -rf outputs/full/features/loss/${ds}"
    run_sh "rm -rf outputs/full/features/entropy/${ds}"
    run_sh "rm -rf outputs/full/features/sae/${ds}"
  done

  # 1.4 training products + run_state for AIME + all_5.
  for ds in aime all_5_datasets; do
    run_sh "rm -rf outputs/full/checkpoints/${ds}"
    run_sh "rm -rf outputs/full/plots/${ds}"
    run_sh "rm -rf outputs/full/reports/${ds}"
    run_sh "rm -rf outputs/full/logs/${ds}"
    run_sh "rm -f  outputs/run_state/full14_${ds}_${SPLITS}.state.json"
    run_sh "rm -f  outputs/run_state/full14_${ds}_${SPLITS}.lock.json"
  done
else
  echo "[stage 1/4] SKIPPED (--skip-clean)"
fi

# -------------------------------------------------------------------- stage 2
# Rescore AIME, regenerate all full_splits (only AIME's change materially;
# others are regenerated from unchanged manifests so the write is idempotent),
# and re-merge all_5_datasets.
if [[ "${SKIP_SCORING}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 2/4] rescore AIME + gen_full_splits + merge_all_datasets"
  echo "======================================================================"
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/fit_pairwise_manifests.py \
    --dataset aime --head music_quality \
    --out data/manifests/pairwise_relu/aime_music_quality_1to5.csv
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/gen_full_splits.py
  run conda run --no-capture-output -n "${TORCH_ENV}" python \
    scripts/data/merge_all_datasets.py
else
  echo "[stage 2/4] SKIPPED (--skip-scoring)"
fi

# Overlay writer: single overlay enables ONLY AIME in large_scale_single.
if [[ "${SKIP_SINGLES}" == "0" || "${SKIP_MERGED}" == "0" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] generate overlay yamls: ${SINGLE_OVERLAY}, ${MERGED_OVERLAY}"
  else
    conda run --no-capture-output -n "${TORCH_ENV}" python - <<PY
import copy, os
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
for entry in single_cfg.get("datasets", {}).get("large_scale_single", []):
    entry["enabled"] = (entry.get("name") == "aime")

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

# ------------------------------------------------------------------- stage 3a
if [[ "${SKIP_SINGLES}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 3a/4] parallel runner: AIME-only (extract + train)"
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
  echo "[stage 3a/4] SKIPPED (--skip-singles)"
fi

# ------------------------------------------------------------------- stage 3b
if [[ "${SKIP_MERGED}" == "0" ]]; then
  echo "======================================================================"
  echo "[stage 3b/4] parallel runner: all_5_datasets (reuse features, train only)"
  echo "======================================================================"

  # Same §6.3.1 feature-reuse guard as 1st_iter.sh.
  patch_token='if name == "all_5_datasets"'
  if ! grep -q "${patch_token}" \
       "${PROJECT_ROOT}/scripts/run/run_full_14_experiments_parallel.py"; then
    echo ""
    echo "[warn] stage 3b requires the feature-reuse patch from 1st_iter_plan §6.3.1:"
    echo "       run_full_14_experiments_parallel.py::_build_jobs must override the"
    echo "       all_5_datasets feature roots to point at the shared parent"
    echo "       outputs/full/features/{loss,entropy,sae}/ directory."
    echo "       Without that patch the merged training cannot find per-DB features."
    echo ""
    echo "[warn] skipping stage 3b. Re-run with --skip-ingest --skip-clean --skip-scoring --skip-singles"
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
  echo "[stage 3b/4] SKIPPED (--skip-merged)"
fi

# --------------------------------------------------------------------- stage 4
echo "======================================================================"
echo "[stage 4/4] self-check: AIME + all_5_datasets summary + test CSVs"
echo "======================================================================"

missing=0
for ds in aime all_5_datasets; do
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
  echo "[done] 2nd iter finished; AIME + all_5_datasets have fresh summaries + 7 CNN test score CSVs."
else
  echo "[partial] ${missing} check(s) failed; see logs under outputs/run_state/"
  exit 1
fi
