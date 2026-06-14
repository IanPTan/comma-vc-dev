"""
Trainer for the Video Swin autoencoder (arxiv 2212.13805 backbone, no masking).

The model is `SwinVideoAutoencoder`: Swin encoder + symmetric Swin decoder,
trained with pixel MSE reconstruction loss.
"""

import os
import time
from typing import Dict, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _raw_model(m: torch.nn.Module) -> torch.nn.Module:
    """Return the un-compiled module so state_dict keys don't carry the
    `_orig_mod.` prefix that torch.compile adds."""
    return getattr(m, "_orig_mod", m)


def _normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float32 [0, 1]."""
    x = batch.float()
    if x.max() > 1.5:
        x = x / 255.0
    return x


def train(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    val_loader=None,
    frame_size: int = 256,
    save_every: int = 5,
    grad_clip: float = 1.0,
    max_batches_per_epoch: Optional[int] = None,
    resume_epoch: int = 0,
):
    """Train `SwinVideoAutoencoder` end-to-end with pixel MSE.

    Args:
        model:        SwinVideoAutoencoder. Its `forward(x)` must return
                      `(recon, loss)`.
        train_loader: yields (B, C, T, H, W) tensors (uint8 or float).
        optimizer:    e.g. torch.optim.AdamW.
        device:       'cuda' / 'cpu'.
        num_epochs:   number of epochs.
        save_dir:     where to write checkpoints.
        val_loader:   optional validation loader.
        frame_size:   (Unused in loop, now handled by DALI).
        save_every:   checkpoint cadence (epochs).
        grad_clip:    global L2 grad clip; pass 0 to disable.
        resume_epoch: epoch to start from (0 if new training).
    """
    os.makedirs(save_dir, exist_ok=True)
    stats_path = os.path.join(save_dir, "stats.h5")
    
    # Initialize HDF5 file for logging
    if resume_epoch == 0 or not os.path.exists(stats_path):
        with h5py.File(stats_path, 'w') as f:
            g = f.create_group("loss")
            g.create_dataset("train", (num_epochs,), dtype='f', fillvalue=np.nan)
            g.create_dataset("val", (num_epochs,), dtype='f', fillvalue=np.nan)
    
    best_val_loss = float('inf')
    
    # Try to load best_val_loss if resuming
    if resume_epoch > 0:
        with h5py.File(stats_path, 'r') as f:
            val_losses = f["loss/val"][:resume_epoch]
            valid_val = val_losses[~np.isnan(val_losses)]
            if len(valid_val) > 0:
                best_val_loss = np.min(valid_val)

    for epoch in range(resume_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_time = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                pbar.close()
                break
            
            # DALI now provides the correctly resized [B, C, T, H, W] tensor.
            # We just need to normalize to [0, 1].
            clip = _normalize_batch(batch.to(device))

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
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ms": f"{dt*1000:.0f}",
            })

        avg_train_loss = epoch_loss / max(n_batches, 1)
        
        # Validation
        avg_val_loss = np.nan
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation", leave=False)
                for batch in val_pbar:
                    clip = _normalize_batch(batch.to(device))
                    
                    t0 = time.perf_counter()
                    _, loss = model(clip)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    
                    val_loss_sum += loss.item()
                    val_batches += 1
                    
                    val_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "ms": f"{dt*1000:.0f}",
                    })
            avg_val_loss = val_loss_sum / max(val_batches, 1)

        # Log to HDF5
        with h5py.File(stats_path, 'a') as f:
            f["loss/train"][epoch] = avg_train_loss
            f["loss/val"][epoch] = avg_val_loss

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

        # Checkpoints
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": _raw_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }
        
        # Save latest
        latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Save best
        if val_loader is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint["best_val_loss"] = best_val_loss
            best_path = os.path.join(save_dir, "best_val_model.pt")
            torch.save(checkpoint, best_path)
            print(f"  *** New best validation loss: {best_val_loss:.4f} (saved to {best_path})")

        # Optional periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_p = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save(checkpoint, ckpt_p)

    return stats_path


def save_final(model, optimizer, history, num_epochs, save_dir):
    # 'history' here is the stats_path returned by train()
    final_path = os.path.join(save_dir, "swin_final.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": _raw_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    print(f"\nFinal Swin autoencoder saved to {final_path}")
    return final_path
