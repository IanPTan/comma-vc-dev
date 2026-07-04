#!/usr/bin/env bash
# Reproduce archive.zip from the original video. Not required by evaluate.sh;
# included for reproducibility.
#
# Usage:
#   bash submissions/gsplat_v0/compress.sh [VIDEO_PATH] [OUTPUT_ZIP]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

VIDEO="${1:-videos/0.mkv}"
OUT="${2:-$HERE/archive.zip}"

cd "$ROOT"
python -m gsplat_video.compress \
    --video "$VIDEO" \
    --output "$OUT" \
    --output-dir "outputs/gsplat_v0" \
    --steps 30000
