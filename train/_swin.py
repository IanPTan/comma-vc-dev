"""
Trainer for the Video Swin autoencoder (arxiv 2212.13805 backbone, no masking).

The model is `SwinVideoAutoencoder`: Swin encoder + symmetric Swin decoder,
trained with pixel MSE reconstruction loss. This is a standalone trainable
model — no masking, no codebook, no VQ. (When MAE is added later, just mask
patches in the loss / encoder input; the loop here doesn't need to change.)
"""

import os
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _resize_clip(clip: torch.Tensor, size: int) -> torch.Tensor:
    """(B, C, T, H, W) uint8/float -> (B, C, T, size, size) float in [0, 1]."""
    B, C, T, H, W = clip.shape
    x = clip.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).float()
    if x.max() > 1.5:
        x = x / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x.reshape(B, T, C, size, size).permute(0, 2, 1, 3, 4).contiguous()


def train_swin(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    frame_size: int = 256,
    save_every: int = 5,
    grad_clip: float = 1.0,
    max_batches_per_epoch: Optional[int] = None,
) -> Dict[str, list]:
    """Train `SwinVideoAutoencoder` end-to-end with pixel MSE.

    Args:
        model:        SwinVideoAutoencoder. Its `forward(x)` must return
                      `(recon, loss)`.
        train_loader: yields (B, C, T, H, W) tensors (uint8 or float).
        optimizer:    e.g. torch.optim.AdamW.
        device:       'cuda' / 'cpu'.
        num_epochs:   number of epochs.
        save_dir:     where to write checkpoints.
        frame_size:   resize each frame to this HxW before the encoder.
        save_every:   checkpoint cadence (epochs).
        grad_clip:    global L2 grad clip; pass 0 to disable.
    """
    os.makedirs(save_dir, exist_ok=True)
    history: Dict[str, list] = {
        "loss": [], "clips_per_sec": [], "step_ms": [],
        "recon_mean": [], "recon_std": [],
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_time = 0.0
        epoch_step_ms = 0.0
        epoch_clips = 0
        n_batches = 0
        recon_mean_sum = 0.0
        recon_std_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                pbar.close()
                break
            clip = _resize_clip(batch.to(device), frame_size)

            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            recon, loss = model(clip)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            epoch_loss += loss.item()
            epoch_time += dt
            epoch_step_ms += dt * 1000
            epoch_clips += clip.shape[0]
            n_batches += 1
            recon_mean_sum += recon.mean().item()
            recon_std_sum += recon.std().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ms": f"{dt*1000:.0f}",
            })

        avg_loss = epoch_loss / max(n_batches, 1)
        cps = epoch_clips / max(epoch_time, 1e-9)
        avg_step_ms = epoch_step_ms / max(n_batches, 1)
        history["loss"].append(avg_loss)
        history["clips_per_sec"].append(cps)
        history["step_ms"].append(avg_step_ms)
        history["recon_mean"].append(recon_mean_sum / max(n_batches, 1))
        history["recon_std"].append(recon_std_sum / max(n_batches, 1))

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} | "
              f"{cps:.1f} clips/s | step {avg_step_ms:.1f} ms")

        if (epoch + 1) % save_every == 0:
            ckpt = os.path.join(save_dir, f"swin_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    return history


def save_final_swin(model, optimizer, history, num_epochs, save_dir):
    final_path = os.path.join(save_dir, "swin_final.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, final_path)
    print(f"\nFinal Swin autoencoder saved to {final_path}")
    return final_path
