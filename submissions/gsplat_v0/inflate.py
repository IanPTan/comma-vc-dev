"""Per-video inflater called by inflate.sh.

Signature:
    python -m submissions.gsplat_v0.inflate DATA_DIR BASE DST

Reads the archive contents from `DATA_DIR` (or `DATA_DIR/BASE/` if per-video
subdirs exist), reconstructs the SceneRepresentation, and writes every
rendered frame as raw uint8 RGB to `DST`.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import torch

# gsplat_video/ lives at the repo root; the submission dir is under
# comma_video_compression_challenge/submissions/<name>/, so the repo root is
# two levels up. For the challenge's evaluate.sh, this works because
# gsplat_video/ is checked in alongside submissions/ in our fork.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gsplat_video.inflate import load_from_archive, render_to_raw


def _bundle_dir_to_zip_bytes(data_dir: Path) -> bytes:
    """Re-zip a directory of loose files into an in-memory zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in data_dir.iterdir():
            if p.is_file():
                zf.writestr(p.name, p.read_bytes())
    return buf.getvalue()


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: inflate.py DATA_DIR BASE DST", file=sys.stderr)
        sys.exit(2)

    data_dir = Path(sys.argv[1])
    base = sys.argv[2]         # e.g. "0"
    dst = Path(sys.argv[3])

    # Multi-video layout: each video's blob lives in data_dir/<BASE>/. If that
    # doesn't exist, fall back to the flat layout used in single-video eval.
    src = data_dir / base
    if not src.is_dir():
        src = data_dir

    blob = _bundle_dir_to_zip_bytes(src)
    scene, pose_inr, _shader, cfg = load_from_archive(blob)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_frames = int(cfg.get("n_frames", 1200))
    n = render_to_raw(scene, pose_inr, dst, n_frames=n_frames, device=device)
    print(f"wrote {n} frames -> {dst}")


if __name__ == "__main__":
    main()
