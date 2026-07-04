"""First-render sanity check: instantiate an untrained scene, render one
frame via gsplat, save PNG.

This is the Phase-0 gate: if this runs and produces a non-black image, the
architecture is wired correctly and we can move to actual training. If it
crashes, we know exactly which piece needs fixing.

Usage (from repo root, in an env with gsplat + torch + numpy + Pillow):
    python -m scripts.first_render --out first_frame.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("first_frame.png"))
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu. Auto-detected if omitted.")
    ap.add_argument("--n-road", type=int, default=500)
    ap.add_argument("--n-sky", type=int, default=200)
    ap.add_argument("--n-roadside", type=int, default=2000)
    ap.add_argument("--n-actors", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    from gsplat_video.compress import make_initial_scene
    from gsplat_video.constants import (CAMERA_H, CAMERA_W, intrinsics)
    from gsplat_video.pose_inr import EgoPoseINR
    from gsplat_video.rasterizer import render

    scene = make_initial_scene(n_road=args.n_road, n_sky=args.n_sky,
                               n_roadside=args.n_roadside, n_actors=args.n_actors)
    pose_inr = EgoPoseINR()

    scene.to(device)
    pose_inr.to(device)

    K = torch.tensor(intrinsics(), dtype=torch.float32, device=device)

    # Override the untrained pose INR with a fixed dashcam viewpoint at t=0.
    # World frame is driving (+x forward, +y left, +z up).
    # gsplat expects world-to-camera in the OpenCV convention (+x right,
    # +y down, +z forward).  For a dashcam at world (0, 0, 1.5) looking
    # forward:
    #     world +x (forward) -> camera +z
    #     world +y (left)    -> camera -x
    #     world +z (up)      -> camera -y
    R = torch.tensor([[0., -1.,  0.],
                      [0.,  0., -1.],
                      [1.,  0.,  0.]], device=device)
    camera_pos_world = torch.tensor([0., 0., 1.5], device=device)
    t_vec = -R @ camera_pos_world           # world-to-camera translation
    viewmat = torch.eye(4, device=device)
    viewmat[0:3, 0:3] = R
    viewmat[0:3, 3] = t_vec

    print(f"[first_render] device={device}, K={K.tolist()}")
    print(f"[first_render] scene has {scene.total_count()} Gaussians "
          f"({args.n_road} road + {args.n_sky} sky + {args.n_roadside} roadside "
          f"+ {args.n_actors} actors * 5)")

    with torch.no_grad():
        img = render(scene, torch.tensor(0.0, device=device), viewmat, K,
                     CAMERA_W, CAMERA_H)
    arr = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    print(f"[first_render] output shape: {arr.shape}, "
          f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")

    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow required to save PNG. `pip install Pillow`.") from e

    Image.fromarray(arr).save(args.out)
    print(f"[first_render] wrote {args.out}")


if __name__ == "__main__":
    main()
