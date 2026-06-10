"""
Run the Swin video encoder on comma2k19 clips loaded via DALI.

This is the "encoder-only" half of arxiv 2212.13805 (Swin MAE) — no masking,
no MAE decoder. It loads a batch from the DaliDataLoader, resizes each frame
to the target spatial size, and runs a single forward pass through the Swin
encoder so we can verify shapes / param counts end-to-end on real data.

Example:
    python model/swin/train_swin.py \\
        --data-path data/comma2k19 \\
        --batch-size 2 --clip-frames 16 --frame-size 256
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset import DaliDataLoader
from model.swin.swin_video import SwinVideoEncoder


def parse_args():
    p = argparse.ArgumentParser(description="Forward pass: Swin video encoder on comma2k19.")
    p.add_argument("--data-path", type=str, required=True,
                   help="Path to dataset root (e.g. data/comma2k19).")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("-w", "--workers", type=int, default=4)
    p.add_argument("--clip-frames", type=int, default=16,
                   help="Frames per clip. Must be divisible by patch_t * window_t.")
    p.add_argument("--frame-size", type=int, default=256,
                   help="Resized HxW per frame (square).")

    # Encoder
    p.add_argument("--patch-t", type=int, default=2)
    p.add_argument("--patch-s", type=int, default=16, help="Spatial patch size (cell = 16x16).")
    p.add_argument("--window-t", type=int, default=8)
    p.add_argument("--window-s", type=int, default=4, help="4x4 cells per spatial window.")
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 2, 6, 2])
    p.add_argument("--num-heads", type=int, nargs="+", default=[3, 6, 12, 24])

    p.add_argument("--num-batches", type=int, default=1,
                   help="How many batches to run through the encoder.")
    return p.parse_args()


def resize_clip(video: torch.Tensor, size: int) -> torch.Tensor:
    """(B, C, T, H, W) uint8 -> (B, C, T, size, size) float in [0, 1]."""
    B, C, T, H, W = video.shape
    v = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).float() / 255.0
    v = F.interpolate(v, size=(size, size), mode="bilinear", align_corners=False)
    v = v.reshape(B, T, C, size, size).permute(0, 2, 1, 3, 4).contiguous()
    return v


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = device.index if device.index is not None else 0
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"gpu:    {torch.cuda.get_device_name()}")

    loader = DaliDataLoader(
        args.data_path,
        clip_frames=args.clip_frames,
        batch_size=args.batch_size,
        num_threads=args.workers,
        device_id=device_id,
    )
    print(f"batches available: {len(loader)}")

    model = SwinVideoEncoder(
        input_size=(args.clip_frames, args.frame_size, args.frame_size),
        in_channels=3,
        patch_size=(args.patch_t, args.patch_s, args.patch_s),
        window_size=(args.window_t, args.window_s, args.window_s),
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M  |  out dim: {model.out_dim}")

    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        # batch: (B, C, T, H, W) uint8 on GPU
        clip = resize_clip(batch.to(device), args.frame_size)
        with torch.no_grad():
            feats = model(clip)
        print(f"[batch {i}] in {tuple(clip.shape)} -> out {tuple(feats.shape)}")


if __name__ == "__main__":
    main()
