#!/usr/bin/env bash
# Stage submissions/gsplat_v0/ for a PR to commaai/comma_video_compression_challenge.
#
# Copies our gsplat_video/ package inside the submission directory so the
# submission is self-contained. Comma's evaluate.sh needs to run inflate.sh
# using ONLY files in the submission dir + the challenge repo -- our
# gsplat_video/ isn't part of their tree, so we ship it inline.
#
# Usage:
#   bash scripts/prepare_submission.sh [ARCHIVE_URL]
#
# ARCHIVE_URL, if given, is written into the PR description template.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$HERE/submissions/gsplat_v0"

if [ ! -f "$SUB/archive.zip" ] && [ -z "${1:-}" ]; then
    echo "warning: no archive.zip in $SUB/ and no ARCHIVE_URL given."
    echo "         run compress.sh first, or pass a URL."
fi

# Copy our package inside the submission for portability.
DST="$SUB/_gsplat_video"
rm -rf "$DST"
mkdir -p "$DST"
cp -R "$HERE/gsplat_video/"* "$DST/"
echo "copied gsplat_video/ -> submissions/gsplat_v0/_gsplat_video/"

# Rewire the sys.path insert in inflate.py to point at the copied package.
python <<EOF
import re
from pathlib import Path
p = Path("$SUB/inflate.py")
s = p.read_text()
s = s.replace("_ROOT.parent.parent", "_HERE")  # search alongside inflate.py
s = s.replace("from gsplat_video.", "from _gsplat_video.")
p.write_text(s)
EOF
echo "patched submissions/gsplat_v0/inflate.py to import from _gsplat_video"

echo
echo "Submission directory ready at: $SUB"
ls -la "$SUB"
