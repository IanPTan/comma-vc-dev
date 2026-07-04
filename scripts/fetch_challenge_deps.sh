#!/usr/bin/env bash
# Fetch the parts of commaai/comma_video_compression_challenge we need to train
# and evaluate locally. These are gitignored because they're large (~130 MB
# total) and re-fetching is cheap.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="$HERE/challenge_deps"
VIDEOS="$HERE/videos"

mkdir -p "$DEPS/models" "$VIDEOS"

BASE_RAW="https://raw.githubusercontent.com/commaai/comma_video_compression_challenge/master"
BASE_LFS="https://media.githubusercontent.com/media/commaai/comma_video_compression_challenge/master"

fetch() {
    local url="$1"; local out="$2"
    if [ -f "$out" ]; then echo "already have $out"; return; fi
    echo "fetching $url -> $out"
    curl -sSL "$url" -o "$out"
}

fetch "$BASE_RAW/modules.py"       "$DEPS/modules.py"
fetch "$BASE_RAW/frame_utils.py"   "$DEPS/frame_utils.py"
fetch "$BASE_LFS/models/segnet.safetensors"  "$DEPS/models/segnet.safetensors"
fetch "$BASE_LFS/models/posenet.safetensors" "$DEPS/models/posenet.safetensors"
fetch "$BASE_LFS/videos/0.mkv"     "$VIDEOS/0.mkv"

echo "done. sizes:"
ls -lh "$DEPS"/*.py "$DEPS/models/"*.safetensors "$VIDEOS/0.mkv"
