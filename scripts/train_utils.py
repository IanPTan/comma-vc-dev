import os
from math import log10

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import VideoDataset


# ---------------------------------------------------------------------------
# Shape contract
#
# The VQ-VAE itself is frame-based: it operates on tensors of shape (B, C, H, W).
# This script expects the upstream DataLoader to yield video segments of shape
# (B, C, T, H, W), where:
#     B = batch size
#     C = channels
#     T = number of frames per segment
#     H, W = spatial dims
# Each function below converts (B, C, T, H, W) -> (B*T, C, H, W) before the
# forward pass, so time is treated as extra batch.
# ---------------------------------------------------------------------------


def _flatten_time(x):
    """(B, C, T, H, W) -> (B*T, C, H, W). Returns reshaped tensor plus (B, T)."""
    B, C, T, H, W = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
    return x, B, T


def _unflatten_time(x, B, T):
    """(B*T, ...) -> (B, T, ...)"""
    return x.view(B, T, *x.shape[1:])


def train(model, train_loader, optimizer, device, num_epochs, save_dir):
    """Train the VQ-VAE on video segments. Returns a history dict of per-epoch losses.

    Each batch from `train_loader` must have shape (B, C, T, H, W).
    """
    os.makedirs(save_dir, exist_ok=True)
    history = {"recon_loss": [], "vq_loss": [], "total_loss": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_recon = 0
        epoch_vq = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            segments = batch.to(device)  # [B, C, T, H, W]
            frames, B, T = _flatten_time(segments)  # [B*T, C, H, W]

            # Forward pass
            recon, vq_loss, tokens = model(frames)

            # Reconstruction loss — how well does the output match the input?
            recon_loss = F.mse_loss(recon, frames)

            # Total loss = reconstruction + VQ commitment
            total_loss = recon_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track
            epoch_recon += recon_loss.item()
            epoch_vq += vq_loss.item()
            epoch_total += total_loss.item()

            pbar.set_postfix({
                "recon": f"{recon_loss.item():.4f}",
                "vq": f"{vq_loss.item():.4f}",
            })

        # Average losses for this epoch
        n_batches = len(train_loader)
        avg_recon = epoch_recon / n_batches
        avg_vq = epoch_vq / n_batches
        avg_total = epoch_total / n_batches

        history["recon_loss"].append(avg_recon)
        history["vq_loss"].append(avg_vq)
        history["total_loss"].append(avg_total)

        # Check codebook usage
        used, total_codes = model.quantizer.codebook_usage()

        #Reset dead entries ONCE per epoch
        with torch.no_grad():
            sample_batch = next(iter(train_loader)).to(device)
            sample_frames, _, _ = _flatten_time(sample_batch)
            z = model.encoder(sample_frames)
            flat_z = z.permute(0, 2, 3, 1).reshape(-1, model.quantizer.embed_dim)
            num_reset = model.quantizer.reset_dead_entries(flat_z)

        print(f"Epoch {epoch+1}: recon={avg_recon:.4f} | vq={avg_vq:.4f} | "
              f"codebook usage: {used}/{total_codes} ({100*used/total_codes:.0f}%)")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(save_dir, f"vqvae_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    return history


def save_final(model, optimizer, history, num_epochs, save_dir):
    final_path = os.path.join(save_dir, "vqvae_final.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")
    return final_path


def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["recon_loss"], label="Reconstruction Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Reconstruction Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["vq_loss"], label="VQ Loss", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("VQ Commitment Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.show()
    print("Training curves saved.")


def visualize_reconstructions(model, loader, title, device, save_dir, segment_idx=0):
    """Take one segment from the next batch, plot original / recon / tokens for each frame."""
    model.eval()
    batch = next(iter(loader)).to(device)  # [B, C, T, H, W]
    segment = batch[segment_idx:segment_idx + 1]  # [1, C, T, H, W]
    frames, _, T = _flatten_time(segment)  # [T, C, H, W]

    with torch.no_grad():
        recon, _, tokens = model(frames)
    recon = recon.clamp(0, 1)

    fig, axes = plt.subplots(3, T, figsize=(T * 2, 6))
    fig.suptitle(title, fontsize=14)

    # When T == 1 matplotlib returns a 1-D array; normalize to 2-D indexing.
    if T == 1:
        axes = axes.reshape(3, 1)

    for i in range(T):
        axes[0, i].imshow(frames[i].cpu().permute(1, 2, 0))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)

        axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Reconstructed", fontsize=10)

        axes[2, i].imshow(tokens[i].cpu(), cmap="viridis")
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Tokens", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"recon_{title.lower().replace(' ', '_')}.png"), dpi=150)
    plt.show()


def compute_psnr(original, reconstructed):
    """Peak Signal-to-Noise Ratio — higher is better."""
    mse = F.mse_loss(reconstructed, original).item()
    if mse == 0:
        return float('inf')
    return 10 * log10(1.0 / mse)  # max pixel value is 1.0


def evaluate(model, loader, name, device, max_batches=None):
    """Compute average PSNR and MSE over the loader's segments."""
    model.eval()
    psnr_values = []
    recon_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            segments = batch.to(device)
            frames, _, _ = _flatten_time(segments)
            recon, _, _ = model(frames)
            recon = recon.clamp(0, 1)

            for i in range(frames.shape[0]):
                psnr_values.append(compute_psnr(frames[i], recon[i]))
                recon_losses.append(F.mse_loss(recon[i], frames[i]).item())

    avg_psnr = float(np.mean(psnr_values))
    avg_mse = float(np.mean(recon_losses))
    print(f"{name:30s} | PSNR: {avg_psnr:.2f} dB | MSE: {avg_mse:.6f}")
    return avg_psnr, avg_mse


def tokenize(model, loader, device):
    """Encode every segment in `loader` to discrete tokens.

    Returns a tensor of shape (N, T, h, w) where (h, w) is the encoder's
    spatial output and N is the total number of segments seen.
    """
    model.eval()
    all_tokens = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="tokenizing"):
            segments = batch.to(device)        # [B, C, T, H, W]
            frames, B, T = _flatten_time(segments)
            tokens = model.encode(frames)      # [B*T, h, w]
            tokens = _unflatten_time(tokens, B, T)  # [B, T, h, w]
            all_tokens.append(tokens.cpu())

    return torch.cat(all_tokens, dim=0)
