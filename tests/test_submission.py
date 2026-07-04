"""Submission-format sanity checks.

Verifies:
  - inflate.sh has the right signature (positional args DATA_DIR OUTPUT_DIR FILE_LIST)
  - inflate.py is importable and its main signature matches
  - encode_archive stores n_frames so inflate can render the right count
  - a full encode -> extract-to-dir -> reload cycle works (no rasterization,
    but everything up to render_to_raw)
"""
from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path

import torch

from gsplat_video.compress import encode_archive, make_initial_scene
from gsplat_video.inflate import load_from_archive
from gsplat_video.pose_inr import EgoPoseINR


_REPO = Path(__file__).resolve().parents[1]
_SUB = _REPO / "submissions" / "gsplat_v0"


def test_submission_dir_has_required_files():
    for f in ["inflate.sh", "inflate.py", "compress.sh", "__init__.py"]:
        assert (_SUB / f).exists(), f"missing {f}"
    print(f"[ok] submission dir has all required files")


def test_inflate_sh_matches_baseline_signature():
    """The bash wrapper must accept 3 positional args and iterate FILE_LIST."""
    content = (_SUB / "inflate.sh").read_text()
    assert 'DATA_DIR="$1"' in content
    assert 'OUTPUT_DIR="$2"' in content
    assert 'FILE_LIST="$3"' in content
    assert re.search(r'while .* read.* line.*; do', content)
    print("[ok] inflate.sh matches challenge signature")


def test_inflate_py_importable():
    """Submission-level inflate.py should import without errors."""
    import importlib
    m = importlib.import_module("submissions.gsplat_v0.inflate")
    assert hasattr(m, "main")
    print("[ok] submissions.gsplat_v0.inflate imports")


def test_archive_stores_frame_count():
    scene = make_initial_scene(n_road=10, n_sky=5, n_roadside=10, n_actors=5)
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)
    blob = encode_archive(scene, pose_inr, n_frames=1234)
    _, _, _, cfg = load_from_archive(blob)
    assert cfg.get("n_frames") == 1234, f"expected 1234 in cfg, got {cfg.get('n_frames')}"
    print(f"[ok] archive stores n_frames = {cfg['n_frames']}")


def test_extract_to_dir_then_reload():
    """Simulate: archive extracted to a dir, then re-bundle -> reload."""
    scene = make_initial_scene(n_road=10, n_sky=5, n_roadside=10, n_actors=5)
    pose_inr = EgoPoseINR(hidden=16, n_freqs=4)
    blob = encode_archive(scene, pose_inr, n_frames=100)

    with tempfile.TemporaryDirectory() as d:
        extracted = Path(d) / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(extracted)

        # Now re-bundle and reload (simulates inflate.py behavior).
        from submissions.gsplat_v0.inflate import _bundle_dir_to_zip_bytes
        rebundled = _bundle_dir_to_zip_bytes(extracted)
        scene_r, pose_r, _, cfg_r = load_from_archive(rebundled)
        assert cfg_r["n_frames"] == 100
        assert scene_r.total_count() == scene.total_count()
    print("[ok] extract-to-dir -> re-bundle -> reload roundtrips cleanly")


def main() -> None:
    tests = [
        test_submission_dir_has_required_files,
        test_inflate_sh_matches_baseline_signature,
        test_inflate_py_importable,
        test_archive_stores_frame_count,
        test_extract_to_dir_then_reload,
    ]
    for fn in tests:
        fn()
    print(f"\n{len(tests)}/{len(tests)} submission tests passed.")


if __name__ == "__main__":
    main()
