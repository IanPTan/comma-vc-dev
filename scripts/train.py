import argparse

import torch
from torch.utils.data import DataLoader

from dataset import VideoDataset
from vqvae import VQVAE
from train_utils import train, save_final, plot_training_curves, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train a VQ-VAE on video segments.")

    # Data
    p.add_argument("--data-path", type=str, required=True,
                   help="Path passed to VideoDataset (file or directory).")
    p.add_argument("--val-path", type=str, default=None,
                   help="Optional validation data path for periodic eval.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)

    # Model
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-embeddings", type=int, default=512)

    # Optim / training
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint .pt to resume from.")

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Data — VideoDataset items are (C, T, H, W); DataLoader batches to (B, C, T, H, W).
    train_set = VideoDataset(args.data_path)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Train segments: {len(train_set):,} | batches/epoch: {len(train_loader)}")

    val_loader = None
    if args.val_path is not None:
        val_set = VideoDataset(args.val_path)
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"Val segments:   {len(val_set):,}")

    # Model + optimizer
    model = VQVAE(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_embeddings=args.num_embeddings,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Resume
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    # Train
    history = train(model, train_loader, optimizer, device, args.num_epochs, args.save_dir)
    save_final(model, optimizer, history, args.num_epochs, args.save_dir)
    plot_training_curves(history, args.save_dir)

    # Optional eval
    if val_loader is not None:
        print("\n=== Validation ===")
        evaluate(model, val_loader, "val", device)


if __name__ == "__main__":
    main()
