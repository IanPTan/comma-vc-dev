#!/usr/bin/env python3
"""
Temporary script to overfit both the Hiera MAE encoder and decoder end-to-end
from scratch on 4 frames from the dataset using standard masking.
Scores reconstruction loss only on the masked patches.
"""

import argparse
import sys
import os
import random
import math
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

# Ensure the repo root is in the python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset import FrameDataset
from model.hiera.hiera_mae import MaskedAutoencoderHiera


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    if config_path and Path(config_path).exists():
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Try frame_model_2 config as default, fallback to default yaml
    fallback_paths = [
        REPO_ROOT / "experiments" / "frame_model_2" / "config.yaml",
        REPO_ROOT / "experiments" / "frame_default.yaml"
    ]
    for p in fallback_paths:
        if p.exists():
            print(f"Loading config from fallback path: {p}")
            with open(p, 'r') as f:
                return yaml.safe_load(f)
                
    return {}


def reconstruct_and_save(model, original_image, mask_ratio, epoch, save_path):
    model.eval()
    with torch.no_grad():
        # Get shape parameters
        B, C, H, W = original_image.shape
        P = model.pred_stride
        NH = H // P
        NW = W // P
        
        # Forward pass through encoder and decoder (with standard random masking)
        latent, mask = model.forward_encoder(original_image, mask_ratio=mask_ratio)
        pred, pred_mask = model.forward_decoder(latent, mask)
        
        # Unfold original image to match predicted patch shape: [B, NH * NW, P * P * C]
        images_hwc = original_image.permute(0, 2, 3, 1)
        patches_orig = images_hwc.reshape(B, NH, P, NW, P, C)
        patches_orig = patches_orig.permute(0, 1, 3, 2, 4, 5)
        patches_orig = patches_orig.reshape(B, NH * NW, P * P * C)
        
        # Extract original patch stats for denormalizing predictions
        mean = patches_orig.mean(dim=-1, keepdim=True)
        var = patches_orig.var(dim=-1, keepdim=True)
        std = (var + 1e-6) ** 0.5
        
        # Denormalize predictions
        pred_denorm = pred * std + mean
        # Correct layout from (C, P, P) to (P, P, C) to match patches_orig
        pred_denorm = pred_denorm.reshape(B, NH * NW, C, P, P).permute(0, 1, 3, 4, 2).reshape(B, NH * NW, P * P * C)
        
        # Expand pred_mask to match patch dimensions (True = keep/visible, False = remove/masked)
        pred_mask_expanded = pred_mask.unsqueeze(-1)
        
        # Combine original patches (where visible) and reconstructed patches (where masked)
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
        mean_norm = torch.tensor([0.485, 0.456, 0.406], device=original_image.device).view(1, 3, 1, 1)
        std_norm = torch.tensor([0.229, 0.224, 0.225], device=original_image.device).view(1, 3, 1, 1)
        
        img_orig_vis = (original_image * std_norm + mean_norm).clamp(0, 1)
        img_masked_vis = (img_masked * std_norm + mean_norm).clamp(0, 1)
        img_recon_vis = (img_recon * std_norm + mean_norm).clamp(0, 1)
        
        # Stack comparison: [Original, Masked, Reconstructed] side-by-side for each frame, then concatenate vertically
        row_list = []
        for i in range(B):
            row = torch.cat([img_orig_vis[i], img_masked_vis[i], img_recon_vis[i]], dim=2)
            row_list.append(row)
        comparison = torch.cat(row_list, dim=1)
        
        # Convert to PIL and save
        comparison_cpu = comparison.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(comparison_cpu)
        im.save(save_path)
        print(f"Saved masked reconstruction comparison (B={B}) to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Overfit Hiera encoder & decoder end-to-end to a batch of images with masking.")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay regularization")
    parser.add_argument("--save-every", type=int, default=200, help="Save reconstruction interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick check on dummy data")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Setup output directory
    output_dir = REPO_ROOT / "experiments" / "overfit_mask"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Warning: Could not load any valid configuration. Using hardcoded defaults.")
        config = {
            "embed_dim": 96,
            "num_heads": 1,
            "stages": [1, 2, 7, 2],
            "q_pool": 2,
            "q_stride": [2, 2],
            "mask_unit_size": [4, 4],
            "patch_stride": [16, 16],
            "mlp_ratio": 4.0,
            "decoder_embed_dim": 256,
            "decoder_depth": 4,
            "decoder_num_heads": 8,
            "data_path": "data/comma2k19_frames",
            "mask_ratio": 0.6,
            "amp": True,
            "amp_dtype": "float16",
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up image resolution
    # Standard dimensions for comma2k19 are 1152x832
    img_h, img_w = 832, 1152
    
    # 1. Load image batch
    if args.smoke_test:
        print("Running in SMOKE-TEST mode: Using a random tensor as image batch (B=4).")
        image = torch.randn(4, 3, img_h, img_w, device=device)
    else:
        # Load dataset to get targeted frames
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        split_path = output_dir / "dataset_split.json"
        
        try:
            dataset = FrameDataset(
                dataset_dir=config.get("data_path", "data/comma2k19_frames"),
                split_path=str(split_path),
                mode="train",
                val_split=config.get("val_split", 0.1),
                seed=args.seed,
                transform=transform
            )
            img_h, img_w = dataset.height, dataset.width
            
            frame_indices = [0, 10000, 20000, 30000]
            frame_indices = [idx for idx in frame_indices if idx < len(dataset)]
            if len(frame_indices) < 4:
                print(f"Warning: Dataset length is {len(dataset)}, only loaded indices {frame_indices}")
                
            frames = [dataset[idx] for idx in frame_indices]
            image = torch.stack(frames).to(device)
            print(f"Successfully loaded {len(frame_indices)} frames from dataset at indices {frame_indices}. Resolution: {img_w}x{img_h}")
        except Exception as e:
            print(f"Error initializing dataset or loading image: {e}")
            print("Falling back to a randomly generated image batch (B=4) for training.")
            image = torch.randn(4, 3, img_h, img_w, device=device)

    # 2. Initialize Model
    model = MaskedAutoencoderHiera(
        input_size=(img_h, img_w),
        in_chans=3,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        stages=tuple(config["stages"]),
        q_pool=config["q_pool"],
        q_stride=tuple(config["q_stride"]),
        mask_unit_size=tuple(config["mask_unit_size"]),
        patch_stride=tuple(config["patch_stride"]),
        mlp_ratio=config["mlp_ratio"],
        decoder_embed_dim=config["decoder_embed_dim"],
        decoder_depth=config["decoder_depth"],
        decoder_num_heads=config["decoder_num_heads"],
    ).to(device)

    # Filter out parameters for optimization
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name or "mask_token" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
            
    # Override weight_decay if explicitly passed via CLI
    weight_decay = args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.05)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    
    # Cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Mixed precision configuration
    amp_enabled = config.get("amp", True)
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda" and amp_dtype == torch.float16)

    pad_width = len(str(args.epochs))
    mask_ratio = config.get("mask_ratio", 0.6)
    
    print(f"Starting masked end-to-end overfitting training (mask_ratio={mask_ratio})...")
    loss_history = []
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype):
            # Run forward pass of Masked Autoencoder (calculates loss strictly on masked patches)
            loss, pred, label, mask = model(image, mask_ratio=mask_ratio)
            
        scaler.scale(loss).backward()
        
        if config.get("grad_clip", 1.0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
            
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:0{pad_width}d}/{args.epochs:0{pad_width}d} | Loss: {loss_val:.6f} | LR: {current_lr:.6f}")
            
        # Periodic visualization
        if (epoch + 1) % args.save_every == 0 or epoch == 0:
            recon_path = output_dir / f"epoch_{epoch+1:0{pad_width}d}_recon.png"
            reconstruct_and_save(model, image, mask_ratio, epoch, recon_path)

    # 4. Save final outputs
    print("\nTraining completed.")
    
    # Save loss graph
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Masked End-to-End Overfitting Loss Curve")
    plt.grid(True)
    loss_graph_path = output_dir / "loss_graph.png"
    plt.savefig(loss_graph_path)
    plt.close()
    print(f"Saved loss graph to {loss_graph_path}")
    
    # Save final comparison
    final_recon_path = output_dir / "final_recon.png"
    reconstruct_and_save(model, image, mask_ratio, args.epochs, final_recon_path)
    
    # Save trained checkpoint (full model)
    checkpoint_path = output_dir / "overfit_mask_checkpoint.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "loss_history": loss_history
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
