import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from ._utils import _flatten_time

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
