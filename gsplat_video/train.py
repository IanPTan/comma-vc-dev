"""Training loop.

Warmup phase: photometric MSE loss to get Gaussians roughly in place.
Task phase:   score-shaped loss with real segnet + posenet + rate estimate.

Requires gsplat installed for the actual render step (Phase 0 gate).
Locally this file imports fine but train() will raise on the first render.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .codec import GaussianCodec
from .constants import CAMERA_H, CAMERA_W, N_FRAMES, intrinsics
from .dataset import VideoFrameDataset, to_float255
from .losses import (PosenetLoss, SegnetLoss, photometric_loss,
                     score_shaped_loss)
from .populations import SceneRepresentation
from .pose_inr import EgoPoseINR


@dataclass
class TrainConfig:
    steps: int = 30_000
    lr_gaussians: float = 5e-3
    lr_pose: float = 1e-3
    lr_shader: float = 1e-3
    warmup_steps: int = 2_000     # pure photometric warmup
    w_photo_late: float = 0.1     # small stabilizer once task losses take over
    log_every: int = 100
    device: str = "cuda"
    # Loss weight modifiers so we can dial them if training gets unstable.
    seg_weight: float = 1.0
    pose_weight: float = 1.0


def train(scene: SceneRepresentation,
          pose_inr: EgoPoseINR,
          dataset: VideoFrameDataset,
          cfg: TrainConfig,
          codec: GaussianCodec | None = None) -> dict:
    """Overfit `scene` and `pose_inr` to `dataset` using score-shaped loss.

    Returns a dict of training-summary stats (final losses, rate, etc.).
    """
    from .rasterizer import render

    device = torch.device(cfg.device)
    scene.to(device)
    pose_inr.to(device)
    codec = codec or GaussianCodec()

    opt = torch.optim.AdamW([
        {"params": scene.parameters(), "lr": cfg.lr_gaussians},
        {"params": pose_inr.parameters(), "lr": cfg.lr_pose},
    ])

    seg_loss = SegnetLoss(device=str(device))
    pose_loss = PosenetLoss(device=str(device))

    K = torch.tensor(intrinsics(), dtype=torch.float32, device=device)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    last_stats: dict = {}
    step = 0
    while step < cfg.steps:
        for batch in loader:
            t = batch["t"].to(device).squeeze()          # scalar
            target_pair = to_float255(batch["frame_pair"].squeeze(0)).to(device)  # (2, H, W, 3)

            # Render two frames: at t and at the next-frame t.
            idx = int(batch["index"].item())
            t_next = torch.tensor((idx + 1) / max(dataset.num_frames - 1, 1),
                                  device=device, dtype=torch.float32)

            viewmat_a = pose_inr.viewmat(t)
            viewmat_b = pose_inr.viewmat(t_next)
            rendered_a = render(scene, t, viewmat_a, K, CAMERA_W, CAMERA_H)      # (H,W,3) in [0,1]
            rendered_b = render(scene, t_next, viewmat_b, K, CAMERA_W, CAMERA_H)

            rendered_pair = torch.stack([rendered_a, rendered_b], dim=0) * 255.0  # (2, H, W, 3)

            # Add batch dim: losses expect (B, T, H, W, 3).
            r_batched = rendered_pair.unsqueeze(0)
            t_batched = target_pair.unsqueeze(0)

            photo = photometric_loss(rendered_pair, target_pair / 255.0)

            if step < cfg.warmup_steps:
                loss = photo
                seg_val = pose_val = torch.tensor(0.0, device=device)
                rate_val = codec.rate(n_gaussians=scene.total_count())
            else:
                seg_val = cfg.seg_weight * seg_loss(r_batched, t_batched)
                pose_val = cfg.pose_weight * pose_loss(r_batched, t_batched)
                rate_val = codec.rate(n_gaussians=scene.total_count())
                loss = score_shaped_loss(seg_val, pose_val, rate_val,
                                         w_photo=cfg.w_photo_late, photo=photo)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0:
                last_stats = {
                    "step": step, "loss": loss.item(),
                    "photo": photo.item(),
                    "seg": float(seg_val),
                    "pose": float(pose_val),
                    "rate": float(rate_val),
                }
                print(f"[{step:6d}] loss={loss.item():.4f} "
                      f"photo={photo.item():.4f} "
                      f"seg={float(seg_val):.4f} "
                      f"pose={float(pose_val):.5f} "
                      f"rate={float(rate_val):.5f}")

            step += 1
            if step >= cfg.steps:
                break

    return last_stats
