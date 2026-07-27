#!/usr/bin/env bash
set -euo pipefail

# Install/prepare SAE extraction dependency (musicdiscovery, MusicGen-small pipeline).
# This script is designed for server setup and can be rerun safely.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${_SCRIPT_DIR}/../../.." && pwd)}"
TORCH_ENV="${TORCH_ENV:-torch21}"
MUSICDISCOVERY_ENV="${MUSICDISCOVERY_ENV:-musicdiscovery310}"
MUSICDISCOVERY_URL="https://github.com/PapayaResearch/musicdiscovery"
MUSICDISCOVERY_DIR="${PROJECT_ROOT}/external/musicdiscovery"
RELEASE_CFG="${PROJECT_ROOT}/config/paths.yaml"

echo "[setup] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[setup] TORCH_ENV=${TORCH_ENV}"
echo "[setup] MUSICDISCOVERY_ENV=${MUSICDISCOVERY_ENV}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[error] PROJECT_ROOT not found: ${PROJECT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ -d "${MUSICDISCOVERY_DIR}/.git" ]]; then
  echo "[setup] musicdiscovery already present: ${MUSICDISCOVERY_DIR}"
elif git config -f .gitmodules --get-regexp "submodule\\.external/musicdiscovery\\.url" >/dev/null 2>&1; then
  echo "[setup] init existing musicdiscovery submodule entry"
  git submodule update --init --recursive external/musicdiscovery
else
  echo "[setup] add musicdiscovery submodule"
  git submodule add "${MUSICDISCOVERY_URL}" external/musicdiscovery
fi

if [[ ! -f "${MUSICDISCOVERY_DIR}/requirements.txt" ]]; then
  echo "[error] requirements.txt missing in ${MUSICDISCOVERY_DIR}" >&2
  exit 1
fi

if [[ ! -f "${RELEASE_CFG}" ]]; then
  echo "[error] missing release config: ${RELEASE_CFG}" >&2
  exit 1
fi

TORCH_PY_VER="$(conda run -n "${TORCH_ENV}" python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
echo "[setup] ${TORCH_ENV} python=${TORCH_PY_VER}"

NEED_SEPARATE_ENV=0
if python - "${TORCH_PY_VER}" <<'PY'
import sys
major, minor = map(int, sys.argv[1].strip().split("."))
sys.exit(0 if (major > 3 or (major == 3 and minor >= 10)) else 1)
PY
then
  NEED_SEPARATE_ENV=0
else
  NEED_SEPARATE_ENV=1
fi

if [[ "${NEED_SEPARATE_ENV}" -eq 1 ]]; then
  echo "[setup] ${TORCH_ENV} is <3.10, prepare separate env ${MUSICDISCOVERY_ENV} (python=3.10)"
  if ! conda env list | awk '{print $1}' | rg -x "${MUSICDISCOVERY_ENV}" >/dev/null; then
    conda create -y -n "${MUSICDISCOVERY_ENV}" python=3.10
  fi
  TARGET_ENV="${MUSICDISCOVERY_ENV}"
else
  TARGET_ENV="${TORCH_ENV}"
fi

echo "[setup] install musicdiscovery deps in env: ${TARGET_ENV}"
conda run -n "${TARGET_ENV}" python -m pip install --upgrade pip

# Resolve known upstream pin conflict:
# - audiocraft==1.3.0 requires av==11.0.0
# - musicdiscovery requirements may pin av==12.3.0
# We build a temporary requirements file for server install.
REQ_IN="${MUSICDISCOVERY_DIR}/requirements.txt"
REQ_TMP="$(mktemp /tmp/musicdiscovery_requirements.XXXXXX.txt)"
python - "${REQ_IN}" "${REQ_TMP}" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
lines = src.read_text(encoding="utf-8").splitlines()
out = []
has_audiocraft_13 = any(re.match(r"^\s*audiocraft\s*==\s*1\.3\.0\s*$", ln) for ln in lines)
has_auto_interp = any(re.match(r"^\s*automated-interpretability\s*==", ln) for ln in lines)
has_datasets_221 = any(re.match(r"^\s*datasets\s*==\s*2\.21\.0\s*$", ln) for ln in lines)
for ln in lines:
    if has_audiocraft_13:
        # Conflict 1: audiocraft 1.3.0 requires av==11.0.0
        if re.match(r"^\s*av\s*==\s*12\.3\.0\s*$", ln):
            out.append("av==11.0.0")
            continue

        # Conflict 2: audiocraft 1.3.0 requires torch==2.1.0; upstream file may pin torch 2.3 + CUDA stack
        if re.match(r"^\s*torch\s*==", ln):
            continue
        if re.match(r"^\s*torchaudio\s*==", ln):
            continue
        if re.match(r"^\s*torchvision\s*==", ln):
            continue
        if re.match(r"^\s*triton\s*==", ln):
            continue
        if re.match(r"^\s*xformers\s*==", ln):
            continue
        if re.match(r"^\s*nvidia-[a-z0-9\-]+\s*==", ln):
            continue
        # Conflict 3: automated-interpretability 0.0.23 requires numpy<2.0
        if re.match(r"^\s*numpy\s*==", ln):
            continue
        # Conflict 4: datasets 2.21.0 requires fsspec<=2024.6.1
        if re.match(r"^\s*fsspec\s*==", ln):
            continue

    out.append(ln)

if has_audiocraft_13:
    # Re-add a compatible torch stack for audiocraft 1.3.0.
    out.extend([
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "torchvision==0.16.0",
    ])
if has_auto_interp:
    out.append("numpy==1.26.4")
if has_datasets_221:
    out.append("fsspec==2024.6.1")

dst.write_text("\n".join(out) + "\n", encoding="utf-8")
PY

echo "[setup] installing from patched requirements: ${REQ_TMP}"
conda run -n "${TARGET_ENV}" python -m pip install -r "${REQ_TMP}"
conda run -n "${TARGET_ENV}" python -m pip install hydra-core omegaconf
rm -f "${REQ_TMP}"

# Ensure configured SAE checkpoint bundle exists (cfg.json + weights + sparsity).
CKPT_DIR="$(python - "${RELEASE_CFG}" <<'PY'
import os
import sys
import yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
root = cfg.get("project_root", "")
ckpt = cfg.get("sae", {}).get("musicdiscovery_checkpoint_dir", "")
if ckpt and not os.path.isabs(ckpt):
    ckpt = os.path.join(root, ckpt)
print(ckpt)
PY
)"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "[error] sae.musicdiscovery_checkpoint_dir is empty in ${RELEASE_CFG}" >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"
if [[ ! -f "${CKPT_DIR}/cfg.json" || ! -f "${CKPT_DIR}/sae_weights.safetensors" || ! -f "${CKPT_DIR}/sparsity.safetensors" ]]; then
  echo "[setup] SAE checkpoint files missing, attempting auto-download to ${CKPT_DIR}"
  # Parse path format: .../sae-<exp>_k_<k>_layer_<L>/facebook/<model>
  # and probe the first available remote checkpoint (fallback when a configured layer is missing).
  read -r PREFIX MODEL CHOSEN_HOOK <<<"$(python - "${CKPT_DIR}" <<'PY'
import os
import re
import sys
import urllib.request

p = os.path.normpath(sys.argv[1]).replace("\\", "/")
m = re.search(r"(sae-(\d+)_k_(\d+)_layer_(\d+))/facebook/([^/]+)$", p)
if not m:
    print("")
    raise SystemExit(0)

exp = int(m.group(2))
k = int(m.group(3))
layer_human = int(m.group(4))
requested_hook = layer_human - 1
model = m.group(5)

base = "https://music-discovery-sae-checkpoints.s3.amazonaws.com"
known_layers = [1, 5, 11, 17, 21]
candidates = [requested_hook] + sorted(known_layers, key=lambda x: abs(x - requested_hook))
seen = set()
ordered = []
for c in candidates:
    if c < 0 or c in seen:
        continue
    seen.add(c)
    ordered.append(c)

chosen = None
for hook_layer in ordered:
    url = f"{base}/sae-{exp}_k_{k}_{hook_layer}/facebook/{model}/cfg.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if int(resp.status) == 200:
                chosen = hook_layer
                break
    except Exception:
        continue

if chosen is None:
    print("")
    raise SystemExit(0)

prefix = f"sae-{exp}_k_{k}_{chosen}/facebook/{model}"
print(prefix, model, chosen)
PY
)"
  if [[ -z "${PREFIX}" ]]; then
    echo "[error] cannot infer checkpoint URL from path: ${CKPT_DIR}" >&2
    echo "[hint] expected .../sae-4_k_32_layer_22/facebook/musicgen-small" >&2
    exit 1
  fi
  echo "[setup] selected checkpoint prefix: ${PREFIX} (hook_layer=${CHOSEN_HOOK})"
  BASE_URL="https://music-discovery-sae-checkpoints.s3.amazonaws.com"
  curl -fL -o "${CKPT_DIR}/cfg.json" "${BASE_URL}/${PREFIX}/cfg.json"
  curl -fL -o "${CKPT_DIR}/sae_weights.safetensors" "${BASE_URL}/${PREFIX}/sae_weights.safetensors"
  curl -fL -o "${CKPT_DIR}/sparsity.safetensors" "${BASE_URL}/${PREFIX}/sparsity.safetensors"
  echo "[setup] downloaded SAE checkpoint bundle for ${MODEL}"
fi

echo "[setup] done."
echo "[setup] musicdiscovery deps env: ${TARGET_ENV}"
echo "[setup] exp12/exp13 training remains in env: ${TORCH_ENV}"
echo "[setup] next: bash scripts/run/run_musiceval_14_experiments.sh"
