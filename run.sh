#!/usr/bin/env bash
# run.sh — runs the clip analysis script.
# Usage:
#   ./run.sh summary
#   ./run.sh autolabel <clip_filename>
set -euo pipefail

source .venv/bin/activate 2>/dev/null || true

MODE="${1:-summary}"

if [ "$MODE" = "summary" ]; then
    python3 analyze_clips.py --clips_dir ./clips --summary
elif [ "$MODE" = "autolabel" ]; then
    CLIP_NAME="${2:?Usage: ./run.sh autolabel <clip_filename>}"
    python3 analyze_clips.py --clips_dir ./clips --autolabel --clip "$CLIP_NAME"
else
    echo "Unknown mode: $MODE (use 'summary' or 'autolabel <file>')"
    exit 1
fi
