"""Training loop skeleton.

Not runnable end-to-end yet — needs:
  - real segnet / posenet plugged into losses.py (from the challenge repo)
  - real dataset iterator (frames from videos/0.mkv + camera intrinsics)
  - gsplat installed

but the structure below is what we'll fill in during Phase 3 onwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .losses import (PosenetLossStub, SegnetLossStub, photometric_loss,
                     score_shaped_loss)
from .populations import SceneRepresentation
from .pose_inr import EgoPoseINR
from .rasterizer import render


@dataclass
class TrainConfig:
    steps: int = 30_000
    lr_gaussians: float = 5e-3
    lr_pose: float = 1e-3
    lr_shader: float = 1e-3
    warmup_steps: int = 2_000     # pure photometric warmup before task losses
    w_photo_late: float = 0.1     # small stabilizer once task losses take over
    width: int = 480
    height: int = 360
    device: str = "cuda"


class FrameDataset(Dataset):
    """Stub — will yield (t_norm, frame_target, K, viewmat_hint) tuples.

    Real implementation reads videos/0.mkv frames + intrinsics from the
    challenge dataloader.
    """

    def __init__(self, n_frames: int = 1200):
        self.n_frames = n_frames

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, i: int) -> dict:
        t_norm = torch.tensor(i / (self.n_frames - 1), dtype=torch.float32)
        frame = torch.zeros(3, 360, 480)          # placeholder
        K = torch.eye(3)                          # placeholder
        return dict(t=t_norm, frame=frame, K=K, index=i)


def train(scene: SceneRepresentation,
          pose_inr: EgoPoseINR,
          dataset: FrameDataset,
          cfg: TrainConfig) -> None:
    device = torch.device(cfg.device)
    scene.to(device)
    pose_inr.to(device)

    opt = torch.optim.AdamW([
        {"params": scene.parameters(), "lr": cfg.lr_gaussians},
        {"params": pose_inr.parameters(), "lr": cfg.lr_pose},
    ])

    seg_loss_fn = SegnetLossStub().to(device)
    pose_loss_fn = PosenetLossStub().to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    step = 0
    while step < cfg.steps:
        for batch in loader:
            t = batch["t"].to(device).squeeze()
            target = batch["frame"].to(device).permute(1, 2, 0)  # (H, W, 3)
            K = batch["K"].to(device).squeeze()

            viewmat = pose_inr.viewmat(t)
            rendered = render(scene, t, viewmat, K, cfg.width, cfg.height)

            photo = photometric_loss(rendered, target)

            # Warmup: photometric only. Later: score-shaped with task terms.
            if step < cfg.warmup_steps:
                loss = photo
            else:
                seg = seg_loss_fn(rendered, target)
                pose = pose_loss_fn((rendered, rendered), (target, target))  # placeholder pair
                rate_est = torch.tensor(0.005, device=device)  # TODO: real diff rate
                loss = score_shaped_loss(seg, pose, rate_est,
                                         w_photo=cfg.w_photo_late, photo=photo)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 100 == 0:
                print(f"[{step:6d}] loss={loss.item():.4f} photo={photo.item():.4f}")

            step += 1
            if step >= cfg.steps:
                break
