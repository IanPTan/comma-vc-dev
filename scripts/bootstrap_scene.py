"""Phase-1 bootstrap runner. Reads videos/0.mkv, runs Depth-Anything on each
sampled frame, unprojects to world points, distributes to populations, and
writes a bootstrapped SceneRepresentation to disk (as a pickled state_dict
we can load in Phase-2 training).

Meant to run on Colab (GPU + transformers + depth-anything weights).
Locally the smoke test in tests/test_bootstrap.py already covers the geometry
and distribution logic.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from gsplat_video.bootstrap import (DepthEstimator, DistributionConfig,
                                    distribute_points_to_populations,
                                    estimate_ego_trajectory_simple,
                                    seed_scene_from_points, unproject_frame)
from gsplat_video.constants import FPS, N_FRAMES
from gsplat_video.dataset import VideoFrameDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=Path("videos/0.mkv"))
    ap.add_argument("--out", type=Path, default=Path("bootstrap_scene.pkl"))
    ap.add_argument("--frame-stride", type=int, default=20,
                    help="Sample every Nth frame for the point cloud.")
    ap.add_argument("--point-stride", type=int, default=8,
                    help="Sample every Nth pixel per frame.")
    ap.add_argument("--depth-scale", type=float, default=30.0,
                    help="Multiplier from unitless depth to meters.")
    ap.add_argument("--velocity-mps", type=float, default=13.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[bootstrap] loading {args.video}")
    ds = VideoFrameDataset(args.video)
    n_frames = ds.num_frames
    print(f"[bootstrap] {n_frames} frames loaded")

    poses = estimate_ego_trajectory_simple(
        n_frames=n_frames, fps=FPS, forward_velocity_mps=args.velocity_mps)

    print(f"[bootstrap] loading Depth-Anything-V2 (may download on first run)")
    depth_est = DepthEstimator(model_size="small", device=args.device)

    all_points: list[torch.Tensor] = []
    sampled = list(range(0, n_frames, args.frame_stride))
    print(f"[bootstrap] processing {len(sampled)} sampled frames "
          f"(every {args.frame_stride}th)")
    for i in sampled:
        frame = ds.frames[i]                              # (H, W, 3) uint8
        depth = depth_est.estimate(frame)                 # (H, W) float
        pose = poses[i].to(depth.device)
        pts = unproject_frame(depth, pose, stride=args.point_stride,
                              depth_scale=args.depth_scale)
        all_points.append(pts.cpu())
        if i % (args.frame_stride * 10) == 0:
            print(f"[bootstrap]   frame {i}: {pts.shape[0]} points")

    world_points = torch.cat(all_points, dim=0)
    print(f"[bootstrap] total world points: {world_points.shape[0]}")

    buckets = distribute_points_to_populations(world_points, cfg=DistributionConfig())
    for k, v in buckets.items():
        print(f"[bootstrap]   {k}: {v.shape[0]} points")

    scene = seed_scene_from_points(buckets)
    print(f"[bootstrap] seeded scene: {scene.total_count()} Gaussians")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"scene_state_dict": scene.state_dict()}, f)
    print(f"[bootstrap] wrote {args.out}")


if __name__ == "__main__":
    main()
