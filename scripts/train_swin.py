"""
Train the Swin Video Autoencoder on comma2k19 clips loaded via DALI.

This script manages experiments in `experiments/model_<num>/`, handles
reproducible seeds, and saves all configuration and stats for later use.
"""

import argparse
import sys
import os
import random
import yaml
import pathlib
from pathlib import Path

import numpy as np
import torch

# Ensure the repo root is in the python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset import DaliDataLoader
from model.swin.swin_video import SwinVideoAutoencoder
from train import train, save_final


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For perfect reproducibility, though it slows things down:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_next_exp_dir(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("model_")]
    if not existing:
        return os.path.join(base_dir, "model_0")
    
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[1]))
        except:
            pass
    
    if not nums:
        return os.path.join(base_dir, "model_0")
    
    return os.path.join(base_dir, f"model_{max(nums) + 1}")


def load_defaults():
    default_path = REPO_ROOT / "experiments" / "default.yaml"
    if default_path.exists():
        with open(default_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def parse_args(defaults):
    p = argparse.ArgumentParser(description="Train Swin video autoencoder on comma2k19.")
    
    # Experiment Management
    p.add_argument("--exp-dir", type=str, default=None, 
                   help="Specific experiment directory to use. If None, creates next model_<num>.")
    p.add_argument("--seed", type=int, default=defaults.get("seed", 42), help="Random seed for reproducibility.")

    # Data
    p.add_argument("--data-path", type=str, default=defaults.get("data_path", "data/comma2k19"), 
                   help="Path to the dataset root.")
    p.add_argument("--val-path", type=str, default=None, help="Optional validation data path.")
    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4))
    p.add_argument("-w", "--workers", type=int, default=defaults.get("workers", 4), help="Number of DALI threads.")
    p.add_argument("--device", type=str, default=defaults.get("device", "gpu"), choices=["gpu", "cpu"], help="DALI device backend.")
    p.add_argument("--clip-frames", type=int, default=defaults.get("clip_frames", 16),
                   help="Frames per clip. Must be divisible by patch_t * window_t.")
    p.add_argument("--frame-size", type=int, default=defaults.get("frame_size", 256))
    p.add_argument("--end-safety-margin", type=int, default=defaults.get("end_safety_margin", 200),
                   help="Frames trimmed off the end of each .mkv.")

    # Model
    p.add_argument("--patch-t", type=int, default=defaults.get("patch_t", 2))
    p.add_argument("--patch-s", type=int, default=defaults.get("patch_s", 16), help="16x16 per cell.")
    p.add_argument("--window-t", type=int, default=defaults.get("window_t", 8))
    p.add_argument("--window-s", type=int, default=defaults.get("window_s", 4), help="4x4 cells per window.")
    p.add_argument("--embed-dim", type=int, default=defaults.get("embed_dim", 96))
    p.add_argument("--depths", type=int, nargs="+", default=defaults.get("depths", [2, 2, 6, 2]))
    p.add_argument("--num-heads", type=int, nargs="+", default=defaults.get("num_heads", [3, 6, 12, 24]))

    # Optim / training
    p.add_argument("--lr", type=float, default=defaults.get("lr", 3e-4))
    p.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 0.05))
    p.add_argument("--grad-clip", type=float, default=defaults.get("grad_clip", 1.0))
    p.add_argument("--num-epochs", type=int, default=defaults.get("num_epochs", 30))
    p.add_argument("--save-every", type=int, default=defaults.get("save_every", 5))
    p.add_argument("--max-batches-per-epoch", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", type=str, default=defaults.get("compile_mode", "default"),
                   choices=["default", "reduce-overhead", "max-autotune"])
    
    return p.parse_args()


def main():
    defaults = load_defaults()
    args = parse_args(defaults)

    # 1. Setup Experiment Directory
    if args.exp_dir is None:
        exp_dir = get_next_exp_dir()
    else:
        exp_dir = args.exp_dir
    
    data_dir = os.path.join(exp_dir, "data")
    vis_dir = os.path.join(exp_dir, "vis")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    config_path = os.path.join(exp_dir, "config.yaml")
    
    # 2. Configuration Management
    current_config = vars(args)
    
    resume_path = os.path.join(data_dir, "checkpoint_latest.pt")
    resume_epoch = 0
    
    if os.path.exists(resume_path):
        print(f"Found existing checkpoint at {resume_path}. Resuming...")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            # Override current args with saved config to ensure reproducibility
            # but keep data_path/device etc if user wants to change environment?
            # Actually, user said config should be pushed to remote, so it defines the model.
            for k, v in saved_config.items():
                if k not in ["data_path", "val_path", "workers", "device", "exp_dir"]:
                    setattr(args, k, v)
        
        checkpoint = torch.load(resume_path, map_location="cpu")
        resume_epoch = checkpoint["epoch"]
    else:
        # New experiment: Save the config
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
        print(f"Created new experiment at {exp_dir}")

    # 3. Reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    device_id = device.index if device.index is not None else 0
    print(f"Device: {device} | Experiment: {exp_dir}")
# Data
train_loader = DaliDataLoader(
    args.data_path,
    mode="train",
    clip_frames=args.clip_frames,
    frame_size=args.frame_size,
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
        mode="val",
        clip_frames=args.clip_frames,
        frame_size=args.frame_size,
        batch_size=args.batch_size,
        num_threads=args.workers,
        device=args.device,
        device_id=device_id,
        end_safety_margin=args.end_safety_margin,
    )


    # 5. Model + optimizer
    model = SwinVideoAutoencoder(
        input_size=(args.clip_frames, args.frame_size, args.frame_size),
        in_channels=3,
        patch_size=(args.patch_t, args.patch_s, args.patch_s),
        window_size=(args.window_t, args.window_s, args.window_s),
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
    ).to(device)

    if args.compile and device.type == "cuda":
        model = torch.compile(model, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6. Resume Weights
    if resume_epoch > 0:
        checkpoint = torch.load(resume_path, map_location=device)
        sd = checkpoint["model_state_dict"]
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        getattr(model, "_orig_mod", model).load_state_dict(sd)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed from epoch {resume_epoch}")

    # 7. Train
    # train() now returns the path to stats.h5
    stats_path = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        save_dir=data_dir, # Pass the data/ subdir for pt/h5 files
        val_loader=val_loader,
        frame_size=args.frame_size,
        save_every=args.save_every,
        grad_clip=args.grad_clip,
        max_batches_per_epoch=args.max_batches_per_epoch,
        resume_epoch=resume_epoch
    )
    
    save_final(model, optimizer, stats_path, args.num_epochs, data_dir)


if __name__ == "__main__":
    main()
