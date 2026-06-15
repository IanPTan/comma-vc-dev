"""Smoke test for lfq.py on comma2k19 driving video.

Pulls a handful of .mkv segments via ffmpeg, builds T-frame clips at 64x64,
and trains the LFQ video VAE long enough to see loss drop and the codebook
populate. Saves an original/recon comparison strip at the end.

Run:
    python3 scripts/test_lfq.py
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lfq import LFQVAE


DATA_ROOT = Path(
    "/Users/devchaudhari/Documents/GitHub/comma-vc-dev/data/comma2k19/Dataset_Chunk_1"
)
OUT_IMG = Path(__file__).parent / "lfq_smoke_test.png"

T = 8                 # frames per clip
FRAME_SIZE = 64       # 64x64 to match model
FILES = 6             # how many .mkv segments to pull from
FRAMES_PER_FILE = 96  # 12 clips of T=8 per file -> 72 clips total
BATCH_SIZE = 4
NUM_EPOCHS = 25
LR = 3e-4


def load_frames(mkv_path: Path, num_frames: int, size: int) -> np.ndarray:
    """Center-crop to square then scale to size x size. Returns (N, H, W, 3) uint8."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(mkv_path),
        # source is 1164x874; crop the centered 874x874 square then scale.
        "-vf", f"crop=874:874,scale={size}:{size}",
        "-pix_fmt", "rgb24",
        "-frames:v", str(num_frames),
        "-f", "rawvideo", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size, 3)


def build_clips() -> torch.Tensor:
    routes = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    mkvs = []
    for r in routes:
        for m in sorted(r.glob("*.mkv")):
            mkvs.append(m)
            if len(mkvs) >= FILES:
                break
        if len(mkvs) >= FILES:
            break

    print(f"loading {len(mkvs)} .mkv segments, {FRAMES_PER_FILE} frames each")
    chunks = []
    for p in mkvs:
        chunks.append(load_frames(p, FRAMES_PER_FILE, FRAME_SIZE))
        print(f"  {p.parent.name}/{p.name:>8s}  -> {chunks[-1].shape}")
    frames = np.concatenate(chunks, axis=0)  # (N, H, W, 3)

    x = torch.from_numpy(frames).float() / 255.0      # (N, H, W, 3) in [0,1]
    x = x.permute(0, 3, 1, 2)                         # (N, C, H, W)

    n_clips = x.shape[0] // T
    x = x[:n_clips * T]
    clips = x.view(n_clips, T, 3, FRAME_SIZE, FRAME_SIZE)
    clips = clips.permute(0, 2, 1, 3, 4).contiguous()  # (N, C, T, H, W)
    return clips


def main():
    torch.manual_seed(0)
    clips = build_clips()
    n_val = max(1, clips.shape[0] // 10)
    val_clips = clips[:n_val]
    train_clips = clips[n_val:]
    print(f"\ntrain: {train_clips.shape[0]} clips of shape {tuple(train_clips.shape[1:])}")
    print(f"val:   {val_clips.shape[0]} clips")

    # entropy_loss_weight bumped from default 0.1 → 1.0. At 0.1 the model settled
    # into a 3-code collapse on the first run: the loss difference between collapse
    # and full-codebook usage was only ~0.86, not enough to escape in ~3 epochs.
    model = LFQVAE(codebook_dim=14, entropy_loss_weight=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}\n")

    print(f"training {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        perm = torch.randperm(train_clips.shape[0])
        ep_recon, ep_q, n_batches = 0.0, 0.0, 0
        for i in range(0, train_clips.shape[0], BATCH_SIZE):
            batch = train_clips[perm[i:i + BATCH_SIZE]]
            recon, q_loss, _ = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            loss = recon_loss + q_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_recon += recon_loss.item()
            ep_q += q_loss.item()
            n_batches += 1

        used, total = model.quantizer.codebook_usage()
        print(
            f"epoch {epoch + 1:2d}/{NUM_EPOCHS} "
            f"| recon {ep_recon / n_batches:.4f} "
            f"| q {ep_q / n_batches:.4f} "
            f"| codebook {used:>4d}/{total} "
            f"({100 * used / total:.1f}%)"
        )
    print(f"\ntrained in {time.time() - t0:.1f}s")

    # validation
    model.eval()
    with torch.no_grad():
        v_recon, _, v_tokens = model(val_clips)
        v_mse = F.mse_loss(v_recon, val_clips).item()
        v_psnr = 10 * np.log10(1.0 / v_mse) if v_mse > 0 else float("inf")
    print(f"val  | mse {v_mse:.4f} | psnr {v_psnr:.2f} dB")
    print(f"val tokens shape {tuple(v_tokens.shape)} "
          f"range [{v_tokens.min().item()}, {v_tokens.max().item()}]")

    # original / recon strip
    orig = val_clips[0].clamp(0, 1)               # (C, T, H, W)
    rec = v_recon[0].clamp(0, 1)
    fig, axes = plt.subplots(2, T, figsize=(T * 1.4, 3))
    for t in range(T):
        axes[0, t].imshow(orig[:, t].permute(1, 2, 0))
        axes[0, t].axis("off")
        axes[1, t].imshow(rec[:, t].permute(1, 2, 0))
        axes[1, t].axis("off")
    axes[0, 0].set_ylabel("orig", rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("recon", rotation=0, ha="right", va="center")
    plt.suptitle(f"val clip — mse {v_mse:.4f}, psnr {v_psnr:.2f} dB")
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=110, bbox_inches="tight")
    print(f"\nsaved comparison: {OUT_IMG}")


if __name__ == "__main__":
    main()
