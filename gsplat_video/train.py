"""Training loop for the 4D-GS scene against the challenge score.

Three phases (softly interleaved):
  Warmup    photometric MSE only. Gets Gaussians into rough shape without the
            unstable segnet gradients dominating early.
  Task loss score-shaped combination of segnet CE + posenet MSE + rate +
            tiny photometric term. Optimizes the actual challenge metric.
  Validate  every val_every steps, evaluate.eval_distortion on a subset of
            frame pairs -> compute the CHALLENGE score (not just training
            loss) so we know how close to submission-quality we are.

Checkpoints every checkpoint_every steps: scene + pose_inr + optimizer +
step, as a torch.save pickle. compress.py loads the last one and encodes
it into archive.zip.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .codec import GaussianCodec
from .constants import CAMERA_H, CAMERA_W, intrinsics
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
    warmup_steps: int = 2_000
    w_photo_late: float = 0.1

    log_every: int = 100
    val_every: int = 1_000
    val_n_pairs: int = 32
    checkpoint_every: int = 2_000

    output_dir: Path = Path("outputs/gsplat_train")
    device: str = "cuda"

    # Score-shaped loss knobs.
    seg_weight: float = 1.0
    pose_weight: float = 1.0


# ---------------------------------------------------------------------------
# Checkpointing.
# ---------------------------------------------------------------------------
def save_checkpoint(scene: SceneRepresentation,
                    pose_inr: EgoPoseINR,
                    opt: torch.optim.Optimizer,
                    step: int,
                    cfg: TrainConfig,
                    path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "scene_state_dict": scene.state_dict(),
        "pose_inr_state_dict": pose_inr.state_dict(),
        "pose_inr_config": {"hidden": pose_inr.net[0].out_features,
                            "n_freqs": pose_inr.ff.n_freqs},
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "config": asdict(cfg),
    }, path)


def load_checkpoint(scene: SceneRepresentation,
                    pose_inr: EgoPoseINR,
                    opt: Optional[torch.optim.Optimizer],
                    path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu")
    scene.load_state_dict(ckpt["scene_state_dict"])
    pose_inr.load_state_dict(ckpt["pose_inr_state_dict"])
    if opt is not None and "optimizer_state_dict" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("step", 0))


def load_bootstrap(scene: SceneRepresentation, path: Path) -> None:
    """Load a scene state_dict produced by scripts/bootstrap_scene.py."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    scene.load_state_dict(d["scene_state_dict"])


# ---------------------------------------------------------------------------
# Validation: the exact challenge score on a subset of frame pairs.
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(scene: SceneRepresentation,
             pose_inr: EgoPoseINR,
             dataset: VideoFrameDataset,
             seg_loss: SegnetLoss,
             pose_loss: PosenetLoss,
             K: Tensor,
             codec: GaussianCodec,
             n_pairs: int,
             device: torch.device) -> dict:
    from .rasterizer import render

    scene.eval(); pose_inr.eval()

    seg_dist = 0.0
    pose_dist = 0.0
    n = 0
    indices = np.linspace(0, len(dataset) - 1, n_pairs).astype(int).tolist()
    for i in indices:
        item = dataset[i]
        t = item["t"].to(device)
        target_pair = to_float255(item["frame_pair"]).to(device)
        t_next = torch.tensor((item["index"] + 1) / max(dataset.num_frames - 1, 1),
                              device=device, dtype=torch.float32)

        viewmat_a = pose_inr.viewmat(t)
        viewmat_b = pose_inr.viewmat(t_next)
        r_a = render(scene, t, viewmat_a, K, CAMERA_W, CAMERA_H)
        r_b = render(scene, t_next, viewmat_b, K, CAMERA_W, CAMERA_H)
        rendered = torch.stack([r_a, r_b], dim=0) * 255.0

        r_batched = rendered.unsqueeze(0)
        t_batched = target_pair.unsqueeze(0)

        seg_dist += float(seg_loss.eval_distortion(r_batched, t_batched).mean())
        pose_dist += float(pose_loss.eval_distortion(r_batched, t_batched).mean())
        n += 1

    scene.train(); pose_inr.train()

    seg_dist /= max(n, 1)
    pose_dist /= max(n, 1)
    rate = codec.rate(n_gaussians=scene.total_count())
    score = 100 * seg_dist + 25 * rate + math.sqrt(10 * pose_dist)
    return {
        "seg_dist": seg_dist,
        "pose_dist": pose_dist,
        "rate": rate,
        "score": score,
        "n_pairs": n,
    }


# ---------------------------------------------------------------------------
# LR schedule: constant during warmup, cosine decay after.
# ---------------------------------------------------------------------------
def _lr_at(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(progress, 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
def train(scene: SceneRepresentation,
          pose_inr: EgoPoseINR,
          dataset: VideoFrameDataset,
          cfg: TrainConfig,
          codec: GaussianCodec | None = None,
          resume_from: Path | None = None,
          bootstrap_from: Path | None = None) -> dict:
    from .rasterizer import render

    device = torch.device(cfg.device)
    codec = codec or GaussianCodec()

    if bootstrap_from is not None:
        print(f"[train] loading bootstrap from {bootstrap_from}")
        load_bootstrap(scene, bootstrap_from)

    scene.to(device); pose_inr.to(device)
    opt = torch.optim.AdamW([
        {"params": scene.parameters(), "lr": cfg.lr_gaussians,
         "name": "gaussians"},
        {"params": pose_inr.parameters(), "lr": cfg.lr_pose,
         "name": "pose"},
    ])

    step = 0
    if resume_from is not None:
        print(f"[train] resuming from {resume_from}")
        step = load_checkpoint(scene, pose_inr, opt, resume_from)

    seg_loss = SegnetLoss(device=str(device))
    pose_loss = PosenetLoss(device=str(device))

    K = torch.tensor(intrinsics(), dtype=torch.float32, device=device)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    while step < cfg.steps:
        for batch in loader:
            t = batch["t"].to(device).squeeze()
            target_pair = to_float255(batch["frame_pair"].squeeze(0)).to(device)
            idx = int(batch["index"].item())
            t_next = torch.tensor((idx + 1) / max(dataset.num_frames - 1, 1),
                                  device=device, dtype=torch.float32)

            # LR schedule.
            for pg in opt.param_groups:
                base = cfg.lr_gaussians if pg["name"] == "gaussians" else cfg.lr_pose
                pg["lr"] = _lr_at(step, cfg.warmup_steps, cfg.steps, base)

            viewmat_a = pose_inr.viewmat(t)
            viewmat_b = pose_inr.viewmat(t_next)
            r_a = render(scene, t, viewmat_a, K, CAMERA_W, CAMERA_H)
            r_b = render(scene, t_next, viewmat_b, K, CAMERA_W, CAMERA_H)
            rendered_pair = torch.stack([r_a, r_b], dim=0) * 255.0
            r_batched = rendered_pair.unsqueeze(0)
            t_batched = target_pair.unsqueeze(0)

            photo = photometric_loss(rendered_pair, target_pair)

            if step < cfg.warmup_steps:
                loss = photo
                seg_val = torch.tensor(0.0, device=device)
                pose_val = torch.tensor(0.0, device=device)
                rate_val = codec.rate(n_gaussians=scene.total_count())
            else:
                seg_val = cfg.seg_weight * seg_loss(r_batched, t_batched)
                pose_val = cfg.pose_weight * pose_loss(r_batched, t_batched)
                rate_val = codec.rate(n_gaussians=scene.total_count())
                loss = score_shaped_loss(seg_val, pose_val, rate_val,
                                         w_photo=cfg.w_photo_late,
                                         photo=photo / 255.0**2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % cfg.log_every == 0:
                print(f"[{step:6d}] loss={loss.item():.4f} "
                      f"photo={photo.item():.2f} "
                      f"seg={float(seg_val):.4f} "
                      f"pose={float(pose_val):.5f} "
                      f"rate={float(rate_val):.5f} "
                      f"lr={opt.param_groups[0]['lr']:.4g}")

            if step > 0 and step % cfg.val_every == 0:
                v = validate(scene, pose_inr, dataset, seg_loss, pose_loss,
                             K, codec, cfg.val_n_pairs, device)
                stats["val"] = v
                print(f"[VAL @{step}] score={v['score']:.4f}  "
                      f"seg={v['seg_dist']:.5f}  pose={v['pose_dist']:.6f}  "
                      f"rate={v['rate']:.5f}")

            if step > 0 and step % cfg.checkpoint_every == 0:
                cp = cfg.output_dir / f"ckpt_{step:06d}.pt"
                save_checkpoint(scene, pose_inr, opt, step, cfg, cp)
                # Also always overwrite `latest.pt` for easy resume.
                save_checkpoint(scene, pose_inr, opt, step, cfg,
                                cfg.output_dir / "latest.pt")
                print(f"[ckpt] wrote {cp.name}")

            step += 1
            if step >= cfg.steps:
                break

    # Final checkpoint + final validation.
    save_checkpoint(scene, pose_inr, opt, step, cfg,
                    cfg.output_dir / "latest.pt")
    v = validate(scene, pose_inr, dataset, seg_loss, pose_loss, K, codec,
                 cfg.val_n_pairs, device)
    stats["final_val"] = v
    print(f"[FINAL] score={v['score']:.4f}  seg={v['seg_dist']:.5f}  "
          f"pose={v['pose_dist']:.6f}  rate={v['rate']:.5f}")
    return stats
