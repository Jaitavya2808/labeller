#!/usr/bin/env bash
# setup.sh — installs everything needed to run analyze_clips.py
# Assumes only Python 3 + bash are pre-installed (per assignment rules).
set -euo pipefail

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install opencv-python-headless

# Only needed if you run the --autolabel demo:
pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
pip install transformers pillow matplotlib

echo "Setup complete. Activate with: source .venv/bin/activate"
