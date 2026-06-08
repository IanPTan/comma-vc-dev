import os
import torch
import matplotlib.pyplot as plt
from ._utils import _flatten_time

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
