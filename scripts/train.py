"""
Train the Swin Video Autoencoder on comma2k19 clips loaded via DALI.

Model: `SwinVideoAutoencoder` = Swin encoder + symmetric Swin decoder, trained
end-to-end with pixel MSE. No masking, no MAE, no codebook — a standalone
trainable model.

Example:
    python scripts/train.py \\
        --data-path data/comma2k19 \\
        --batch-size 4 --clip-frames 16 --frame-size 256 --num-epochs 30
"""

import argparse
import sys
import os
from pathlib import Path

import torch

# Ensure the repo root is in the python path so we can import our packages
# when running the script from the root like: python scripts/train.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset import DaliDataLoader
from model.swin.swin_video import SwinVideoAutoencoder
from train import train, save_final, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train Swin video autoencoder on comma2k19.")
    # Data
    p.add_argument("--data-path", type=str, required=True, help="Path to the dataset root.")
    p.add_argument("--val-path", type=str, default=None, help="Optional validation data path.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("-w", "--workers", type=int, default=4, help="Number of DALI threads.")
    p.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], help="DALI device backend.")
    p.add_argument("--clip-frames", type=int, default=16,
                   help="Frames per clip. Must be divisible by patch_t * window_t.")
    p.add_argument("--frame-size", type=int, default=256)
    p.add_argument("--end-safety-margin", type=int, default=200,
                   help="Frames trimmed off the end of each .mkv when building "
                        "windows, to absorb DALI's stricter frame counting.")

    # Model
    p.add_argument("--patch-t", type=int, default=2)
    p.add_argument("--patch-s", type=int, default=16, help="16x16 per cell.")
    p.add_argument("--window-t", type=int, default=8)
    p.add_argument("--window-s", type=int, default=4, help="4x4 cells per window.")
    p.add_argument("--embed-dim", type=int, default=96)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 2, 6, 2])
    p.add_argument("--num-heads", type=int, nargs="+", default=[3, 6, 12, 24])

    # Optim / training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--save-dir", type=str, default="checkpoints/swin")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None,
                   help="Optional checkpoint path to resume from.")
    p.add_argument("--max-batches-per-epoch", type=int, default=None,
                   help="Cap batches per epoch (smoke test mode).")
    p.add_argument("--compile", action="store_true",
                   help="JIT-compile the model with torch.compile (first step "
                        "is slow, subsequent steps are 15-35%% faster).")
    p.add_argument("--compile-mode", type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode. 'reduce-overhead' is usually a "
                        "good middle ground; 'max-autotune' compiles slower "
                        "but can be a bit faster.")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_id = device.index if device.index is not None else 0
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"gpu:    {torch.cuda.get_device_name()}")

    # Data
    train_loader = DaliDataLoader(
        args.data_path,
        clip_frames=args.clip_frames,
        batch_size=args.batch_size,
        num_threads=args.workers,
        device=args.device,
        device_id=device_id,
        end_safety_margin=args.end_safety_margin,
    )
    print(f"train batches/epoch: {len(train_loader)}")

    val_loader = None
    if args.val_path is not None:
        val_loader = DaliDataLoader(
            args.val_path,
            clip_frames=args.clip_frames,
            batch_size=args.batch_size,
            num_threads=args.workers,
            device=args.device,
            device_id=device_id,
            end_safety_margin=args.end_safety_margin,
        )
        print(f"val batches/epoch:   {len(val_loader)}")

    model = SwinVideoAutoencoder(
        input_size=(args.clip_frames, args.frame_size, args.frame_size),
        in_channels=3,
        patch_size=(args.patch_t, args.patch_s, args.patch_s),
        window_size=(args.window_t, args.window_s, args.window_s),
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"model: {n_params/1e6:.2f}M params  (enc {n_enc/1e6:.2f}M, dec {n_dec/1e6:.2f}M)")

    if args.compile:
        if device.type != "cuda":
            print("warn: --compile requested but device is CPU; skipping.")
        else:
            print(f"compiling model (mode={args.compile_mode}) — first step will be slow...")
            model = torch.compile(model, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt["model_state_dict"]
        # Tolerate checkpoints saved with the compile prefix.
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        # Load into the un-compiled module so the wrapper sees fresh weights.
        getattr(model, "_orig_mod", model).load_state_dict(sd)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    history = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        frame_size=args.frame_size,
        save_every=args.save_every,
        grad_clip=args.grad_clip,
        max_batches_per_epoch=args.max_batches_per_epoch,
    )
    save_final(model, optimizer, history, args.num_epochs, args.save_dir)

    # Optional final eval
    if val_loader is not None:
        print("\n=== Validation ===")
        evaluate(model, val_loader, "val", device)


if __name__ == "__main__":
    main()
