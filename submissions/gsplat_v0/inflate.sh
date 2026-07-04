#!/usr/bin/env bash
# Read Gaussian primitives + pose INR from the extracted archive, render
# every frame via gsplat, write a `.raw` file per video (flat uint8 RGB).
#
# Signature matches submissions/baseline_fast/inflate.sh:
#   inflate.sh DATA_DIR OUTPUT_DIR FILE_LIST
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  cd "$ROOT"
  printf "Rendering %s ... " "$line"
  python -m "submissions.${SUB_NAME}.inflate" "$DATA_DIR" "$BASE" "$DST"
done < "$FILE_LIST"
