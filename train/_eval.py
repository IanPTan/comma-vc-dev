import torch
import torch.nn.functional as F
import numpy as np
from math import log10
from tqdm import tqdm
from ._utils import _flatten_time, _unflatten_time

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
