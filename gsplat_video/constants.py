"""Challenge-side constants, mirrored from challenge_deps/frame_utils.py so
the rest of gsplat_video/ doesn't need to import from challenge_deps."""
from __future__ import annotations

# Frame geometry
CAMERA_W = 1164
CAMERA_H = 874
CAMERA_FL = 910.0             # focal length in pixels, fx == fy
CAMERA_CX = CAMERA_W / 2
CAMERA_CY = CAMERA_H / 2

# Video timing
FPS = 20                      # 60 second clip -> 1200 frames total
N_FRAMES = 1200

# Segmentation / pose class metadata
SEGNET_NUM_CLASSES = 5
SEGNET_INPUT_HW = (384, 512)  # H, W after bilinear downscale
POSENET_OUT_DIM = 12          # only the first 6 are used for distortion


def intrinsics() -> list[list[float]]:
    """Return K as a 3x3 nested list (framework-neutral)."""
    return [[CAMERA_FL, 0.0, CAMERA_CX],
            [0.0, CAMERA_FL, CAMERA_CY],
            [0.0, 0.0, 1.0]]
