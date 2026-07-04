"""Video dataset for training: reads frames from a video file, exposes them
as (t_norm, frame_pair) pairs for the training loop.

We overfit a single video, so we load all frames into host RAM at init.
For videos/0.mkv (1200 frames * 1164 * 874 * 3 bytes) that's ~3.7 GB uint8 —
fits fine on Colab's ~26 GB system RAM but too big for T4 VRAM.
Individual pairs get moved to the training device per iteration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .constants import CAMERA_H, CAMERA_W

_CHALLENGE_DEPS = Path(__file__).resolve().parent.parent / "challenge_deps"
if str(_CHALLENGE_DEPS) not in sys.path:
    sys.path.insert(0, str(_CHALLENGE_DEPS))


def _load_all_frames(path: Path, max_frames: int | None = None) -> Tensor:
    """Decode frames from `path` into a single uint8 tensor.

    Args:
        max_frames: if set, stop after this many frames (useful for local
            sanity tests — full decode of 1200 HEVC frames takes minutes on CPU).

    Returns:
        frames: (N, H, W, 3) uint8, matching the challenge's yuv420_to_rgb output.
    """
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(path))
    stream = container.streams.video[0]

    frames: list[Tensor] = []
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))
        if max_frames is not None and len(frames) >= max_frames:
            break
    container.close()

    return torch.stack(frames)


class VideoFrameDataset(Dataset):
    """Every item is a pair (frame_i, frame_i+1) plus the normalized time
    of frame_i within the video.

    __getitem__ returns:
        t:         scalar float in [0, 1]
        frame_pair: (2, H, W, 3) uint8
        index:     original frame index i
    """

    def __init__(self, video_path: Path, max_frames: int | None = None):
        self.frames = _load_all_frames(Path(video_path), max_frames=max_frames)
        self.n = int(self.frames.shape[0])
        assert self.frames.shape[1:] == (CAMERA_H, CAMERA_W, 3), \
            f"unexpected frame shape {self.frames.shape[1:]}, expected ({CAMERA_H},{CAMERA_W},3)"

    @property
    def num_frames(self) -> int:
        return self.n

    def __len__(self) -> int:
        return self.n - 1                             # number of pairs

    def __getitem__(self, i: int) -> dict:
        return {
            "t": torch.tensor(i / max(self.n - 1, 1), dtype=torch.float32),
            "frame_pair": self.frames[i:i + 2],       # (2, H, W, 3) uint8
            "index": i,
        }


def to_float255(frame_pair_uint8: Tensor) -> Tensor:
    """Convert (2, H, W, 3) or (B, 2, H, W, 3) uint8 -> float in [0, 255].

    Matches the challenge loss preprocessing convention.
    """
    return frame_pair_uint8.float()
