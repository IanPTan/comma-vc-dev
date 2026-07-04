"""Verify the video dataset loads real frames from videos/0.mkv."""
from __future__ import annotations

from pathlib import Path

import torch

from gsplat_video.constants import CAMERA_H, CAMERA_W


_HERE = Path(__file__).resolve().parents[1]
_VIDEO = _HERE / "videos/0.mkv"
_VIDEO_AVAILABLE = _VIDEO.exists()


def test_load_real_video_bounded():
    """Load just 5 frames — full 1200-frame decode is minutes on CPU."""
    if not _VIDEO_AVAILABLE:
        print("[skip] videos/0.mkv missing")
        return
    from gsplat_video.dataset import VideoFrameDataset

    ds = VideoFrameDataset(_VIDEO, max_frames=5)
    print(f"[ok] loaded {ds.num_frames} frames of shape "
          f"{tuple(ds.frames.shape[1:])} ({ds.frames.dtype})")
    assert ds.frames.shape[1:] == (CAMERA_H, CAMERA_W, 3)
    assert ds.frames.dtype == torch.uint8
    assert ds.num_frames == 5

    item = ds[0]
    assert set(item.keys()) == {"t", "frame_pair", "index"}
    assert item["frame_pair"].shape == (2, CAMERA_H, CAMERA_W, 3)
    assert 0.0 <= float(item["t"]) <= 1.0
    print(f"[ok] first item: t={float(item['t']):.4f}, "
          f"pair range [{int(item['frame_pair'].min())}, {int(item['frame_pair'].max())}]")


def test_dataloader_iteration():
    """Verify DataLoader can consume the dataset without weird pytorch issues."""
    if not _VIDEO_AVAILABLE:
        print("[skip] videos/0.mkv missing")
        return
    from torch.utils.data import DataLoader

    from gsplat_video.dataset import VideoFrameDataset

    ds = VideoFrameDataset(_VIDEO, max_frames=6)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    seen = 0
    for batch in loader:
        assert batch["frame_pair"].shape == (1, 2, CAMERA_H, CAMERA_W, 3)
        seen += 1
    assert seen == len(ds), f"iterated {seen}, expected {len(ds)}"
    print(f"[ok] DataLoader yielded {seen} pairs (dataset length = {len(ds)})")


def main() -> None:
    tests = [test_load_real_video_bounded, test_dataloader_iteration]
    for fn in tests:
        fn()
    if _VIDEO_AVAILABLE:
        print(f"\n{len(tests)}/{len(tests)} dataset tests passed.")
    else:
        print(f"\n{len(tests)} tests skipped (video missing).")


if __name__ == "__main__":
    main()
