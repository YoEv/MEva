#!/usr/bin/env bash
# 创建 torch21 和 audiobox 两个 conda 环境。
# musicdiscovery310 由 setup_sae_musicdiscovery.sh 自动处理，无需手动创建。
set -euo pipefail

_env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

# ── torch21 ─────────────────────────────────────────────────────────────────
if _env_exists torch21; then
  echo "=== torch21 已存在，跳过创建 ==="
else
  echo "=== 创建 torch21 ==="
  conda create -y -n torch21 python=3.10
fi

echo "--- 安装 FFmpeg dev 头（解决 PyAV 编译问题）---"
conda install -n torch21 -c conda-forge -y 'ffmpeg=6.*' pkg-config

echo "--- 安装 PyTorch 2.1 ---"
conda run -n torch21 pip install --upgrade pip
conda run -n torch21 pip install \
  torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu121

echo "--- 安装其余依赖 ---"
conda run -n torch21 pip install \
  "numpy==1.26.4" "pandas==2.2.*" "scipy==1.13.*" "scikit-learn==1.4.*" \
  "pyyaml==6.*" tqdm matplotlib seaborn \
  "soundfile==0.12.*" "librosa==0.10.*" \
  "transformers==4.41.*" "accelerate==0.30.*" safetensors \
  tensorboard einops \
  "datasets==2.21.0" huggingface_hub

echo "--- 验证 torch21 ---"
conda run -n torch21 python -c \
  "import torch; print('torch21 OK:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

# ── audiobox ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

echo ""
if _env_exists audiobox; then
  echo "=== audiobox 已存在，跳过创建 ==="
else
  echo "=== 创建 audiobox ==="
  conda create -y -n audiobox python=3.10
fi
conda run -n audiobox pip install --upgrade pip
conda run -n audiobox pip install \
  torch==2.1.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
conda run -n audiobox pip install \
  -e "${PROJECT_ROOT}/external/audiobox-aesthetics"
conda run -n audiobox pip install "numpy==1.26.4" soundfile librosa tqdm pandas pyyaml

echo "--- 验证 audiobox ---"
conda run -n audiobox python -c "import torch; print('audiobox OK:', torch.__version__)"

echo ""
echo "完成。运行 setup_sae_musicdiscovery.sh 以创建 musicdiscovery310 并下载 SAE checkpoint。"
