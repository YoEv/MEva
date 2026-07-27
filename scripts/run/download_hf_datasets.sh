#!/usr/bin/env bash
# Download all 6 HuggingFace datasets into raw_hf/.
# Requires huggingface-cli login (the script checks as well).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
RAW_HF="${PROJECT_ROOT}/raw_hf"

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "Download target directory: ${RAW_HF}"
echo ""

# Login (skipped if already authenticated)
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
  echo "=== downloading ${repo} -> ${dest} ==="
  huggingface-cli download "${repo}" \
    --repo-type dataset \
    --local-dir "${dest}"
  echo ""
done

echo "All downloads complete."
echo "Next: place audio under datasets/<name>/audio/ (see README)."
