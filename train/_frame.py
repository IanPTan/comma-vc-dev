"""
Trainer utilities and training loop for Hiera Masked Autoencoder (MAE) frame-level training.
"""

import os
import time
import math
import h5py
import numpy as np
from PIL import Image
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from typing import Dict, Optional, Tuple

def _raw_model(m: torch.nn.Module) -> torch.nn.Module:
    """Return the un-compiled module so state_dict keys don't carry the
    `_orig_mod.` prefix that torch.compile adds."""
    return getattr(m, "_orig_mod", m)

def get_parameter_groups(model: torch.nn.Module, weight_decay: float = 0.05) -> list:
    """
    Configure parameter groups. Disable weight decay on:
    - 1D tensors (biases, normalization scale/bias)
    - Positional embeddings (pos_embed)
    - Mask tokens (mask_token)
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name or "mask_token" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """Cosine learning rate decay with linear warmup multiplier."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a learning rate scheduler with linear warmup and cosine decay."""
    lr_lambda = lambda step: get_lr_multiplier(step, total_steps, warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def visualize_reconstruction(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    epoch: int,
    save_dir: str,
    mean_norm: list = [0.485, 0.456, 0.406],
    std_norm: list = [0.229, 0.224, 0.225]
):
    """
    Saves a comparison image of original, masked, and reconstructed patches.
    """
    model.eval()
    with torch.no_grad():
        # Keep up to 4 images
        images = images[:4].to(device)
        B, C, H, W = images.shape
        raw_m = _raw_model(model)
        P = raw_m.pred_stride
        NH = H // P
        NW = W // P
        
        # Forward pass through encoder and decoder
        latent, mask = raw_m.forward_encoder(images, mask_ratio=0.6)
        pred, pred_mask = raw_m.forward_decoder(latent, mask)
        
        # Unfold original images to match predicted patch shape: [B, NH * NW, P * P * C]
        images_hwc = images.permute(0, 2, 3, 1)
        patches_orig = images_hwc.reshape(B, NH, P, NW, P, C)
        patches_orig = patches_orig.permute(0, 1, 3, 2, 4, 5)
        patches_orig = patches_orig.reshape(B, NH * NW, P * P * C)
        
        # Extract original patch stats for denormalizing predictions
        mean = patches_orig.mean(dim=-1, keepdim=True)
        var = patches_orig.var(dim=-1, keepdim=True)
        std = (var + 1e-6) ** 0.5
        
        # Denormalize predictions
        pred_denorm = pred * std + mean
        
        # Expand pred_mask to match patch dimensions
        pred_mask_expanded = pred_mask.unsqueeze(-1)
        
        # Combine original patches (where visible) and reconstructed patches (where masked)
        # Note: pred_mask is True for visible, False for masked.
        patches_recon = torch.where(pred_mask_expanded, patches_orig, pred_denorm)
        
        # Masked image representation (masked patches zeroed out)
        patches_masked = torch.where(pred_mask_expanded, patches_orig, torch.zeros_like(patches_orig))
        
        # Helper to rebuild image tensor from patch tensor
        def patches_to_img(p):
            p = p.reshape(B, NH, NW, P, P, C)
            p = p.permute(0, 1, 3, 2, 4, 5)
            p = p.reshape(B, H, W, C)
            return p.permute(0, 3, 1, 2)
            
        img_recon = patches_to_img(patches_recon)
        img_masked = patches_to_img(patches_masked)
        
        # Denormalize from ImageNet stats back to [0, 1] for visual saving
        mean_t = torch.tensor(mean_norm, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(std_norm, device=device).view(1, 3, 1, 1)
        
        img_orig_vis = (images * std_t + mean_t).clamp(0, 1)
        img_masked_vis = (img_masked * std_t + mean_t).clamp(0, 1)
        img_recon_vis = (img_recon * std_t + mean_t).clamp(0, 1)
        
        # Stack comparisons side-by-side
        grid_list = []
        for i in range(B):
            grid_list.append(img_orig_vis[i])
            grid_list.append(img_masked_vis[i])
            grid_list.append(img_recon_vis[i])
            
        grid = vutils.make_grid(grid_list, nrow=3, padding=4, normalize=False)
        
        # Save as PNG
        grid_cpu = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(grid_cpu)
        os.makedirs(save_dir, exist_ok=True)
        im.save(os.path.join(save_dir, f"epoch_{epoch+1:03d}_recon.png"))

def train_frame(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    vis_dir: str,
    val_loader=None,
    save_every: int = 5,
    grad_clip: float = 1.0,
    max_batches_per_epoch: Optional[int] = None,
    resume_epoch: int = 0,
    mask_ratio: float = 0.6,
):
    """
    Train Hiera Masked Autoencoder with reconstruction loss.
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
    
    # Load best_val_loss if resuming
    if resume_epoch > 0:
        with h5py.File(stats_path, 'r') as f:
            val_losses = f["loss/val"][:resume_epoch]
            valid_val = val_losses[~np.isnan(val_losses)]
            if len(valid_val) > 0:
                best_val_loss = np.min(valid_val)

    # We will pick a fixed batch from validation loader for consistent epoch-wise visualization
    vis_batch = None
    if val_loader is not None:
        # Get first batch
        for b in val_loader:
            vis_batch = b
            break

    for epoch in range(resume_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if max_batches_per_epoch is not None and n_batches >= max_batches_per_epoch:
                pbar.close()
                break
            
            # Batch size is [B, C, H, W]
            images = batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            
            # MAE forward pass computes reconstruction MSE loss automatically
            loss, pred, label, mask = model(images, mask_ratio=mask_ratio)
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
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
                    images = batch.to(device)
                    loss, _, _, _ = model(images, mask_ratio=mask_ratio)
                    val_loss_sum += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(val_batches, 1)

        # Log stats to HDF5
        with h5py.File(stats_path, 'a') as f:
            f["loss/train"][epoch] = avg_train_loss
            f["loss/val"][epoch] = avg_val_loss

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}")

        # Save visualization of reconstructions
        if vis_batch is not None:
            visualize_reconstruction(model, vis_batch, device, epoch, vis_dir)

        # Checkpoints setup
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": _raw_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
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

def save_final_frame(model: torch.nn.Module, optimizer: torch.optim.Optimizer, stats_path: str, num_epochs: int, save_dir: str) -> str:
    final_path = os.path.join(save_dir, "hiera_mae_final.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": _raw_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, final_path)
    print(f"\nFinal Hiera MAE model saved to {final_path}")
    return final_path
