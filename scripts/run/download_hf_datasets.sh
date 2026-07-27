#!/usr/bin/env bash
# 下载全部 6 个 HuggingFace 数据集到 raw_hf/。
# 需要先 huggingface-cli login（或在脚本中确认已登录）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RAW_HF="${PROJECT_ROOT}/raw_hf"

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "下载目标目录: ${RAW_HF}"
echo ""

# 登录（若已登录则会提示跳过）
huggingface-cli login --add-to-git-credential 2>/dev/null || true

declare -A REPOS=(
  ["MusicPref"]="i-need-sleep/musicprefs"
  ["AIME"]="disco-eth/AIME"
  ["AIME-survey"]="disco-eth/AIME-survey"
  ["SongEval"]="ASLP-lab/SongEval"
  ["MusicArena"]="music-arena/music-arena-dataset"
  ["MusicEval"]="BAAI/MusicEval"
)

for local_name in MusicPref AIME AIME-survey SongEval MusicArena MusicEval; do
  repo="${REPOS[$local_name]}"
  dest="${RAW_HF}/${local_name}"
  echo "=== 下载 ${repo} → ${dest} ==="
  huggingface-cli download "${repo}" \
    --repo-type dataset \
    --local-dir "${dest}"
  echo ""
done

echo "全部下载完成。"
echo "下一步：按 doc/server/data_prep.md §5.3 把音频落地到 datasets/<name>/audio/。"
