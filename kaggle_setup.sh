#!/bin/bash
# ============================================================
# Kaggle Setup Script — Run this in a Kaggle Notebook cell
# to clone the documentary pipeline repo and install deps.
# ============================================================
# Paste this into a Kaggle notebook cell:
#
# !curl -s https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/kaggle_setup.sh | bash
# ============================================================

set -e

echo "=== Documentary Pipeline — Kaggle Setup ==="
echo ""

# 1. Clone the repo (or copy from input)
if [ -d "/kaggle/working/documentary_pipeline" ]; then
    echo "[OK] Already cloned"
else
    echo "Cloning repo..."
    # Replace with your repo URL after pushing to GitHub
    git clone https://github.com/YOUR_USER/YOUR_REPO.git /kaggle/working/documentary_pipeline
fi

cd /kaggle/working/documentary_pipeline

# 2. Install Python dependencies
echo "Installing Python dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt

# 3. Install FFmpeg
echo "Installing FFmpeg..."
apt-get install -qq ffmpeg 2>/dev/null || true

# 4. Verify
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
ffmpeg -version 2>&1 | head -1

echo ""
echo "=== Setup Complete ==="
echo "Run: cd /kaggle/working/documentary_pipeline && python run_local.py --topic 'Your Topic'"
