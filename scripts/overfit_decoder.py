#!/usr/bin/env python3
"""
Temporary script to overfit a learnable latent parameter (representing the encoder output)
and the Hiera MAE decoder to a single frame from the dataset.
Scores reconstruction of the entire image (ignoring masking).
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


def reconstruct_and_save(model, latent_param, original_image, epoch, save_path):
    model.eval()
    with torch.no_grad():
        # Get shape parameters
        B, C, H, W = original_image.shape
        P = model.pred_stride
        NH = H // P
        NW = W // P
        
        # Forward pass through decoder
        # Pass an all-True mask representing that all tokens are visible/decoded
        all_patches_mask = torch.ones((B, NH * NW), dtype=torch.bool, device=original_image.device)
        pred, _ = model.forward_decoder(latent_param, all_patches_mask)
        
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
        
        # Helper to rebuild image tensor from patch tensor
        def patches_to_img(p):
            p = p.reshape(B, NH, NW, P, P, C)
            p = p.permute(0, 1, 3, 2, 4, 5)
            p = p.reshape(B, H, W, C)
            return p.permute(0, 3, 1, 2)
            
        img_recon = patches_to_img(pred_denorm)
        
        # Denormalize from ImageNet stats back to [0, 1] for visual saving
        mean_norm = torch.tensor([0.485, 0.456, 0.406], device=original_image.device).view(1, 3, 1, 1)
        std_norm = torch.tensor([0.229, 0.224, 0.225], device=original_image.device).view(1, 3, 1, 1)
        
        img_orig_vis = (original_image * std_norm + mean_norm).clamp(0, 1)
        img_recon_vis = (img_recon * std_norm + mean_norm).clamp(0, 1)
        
        # Stack comparison: [Original, Reconstructed] side-by-side
        comparison = torch.cat([img_orig_vis[0], img_recon_vis[0]], dim=2)
        
        # Convert to PIL and save
        comparison_cpu = comparison.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(comparison_cpu)
        im.save(save_path)
        print(f"Saved reconstruction comparison to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Overfit learnable latent embeddings to a single image.")
    parser.add_argument("--config", type=str, help="Path to config.yaml")
    parser.add_argument("--lr", type=float, default=0.0024, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay regularization")
    parser.add_argument("--save-every", type=int, default=25, help="Save reconstruction interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick check on dummy data")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Setup output directory
    output_dir = REPO_ROOT / "experiments" / "overfit_decoder"
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
            "amp": True,
            "amp_dtype": "float16",
        }

    # Override weight_decay if explicitly passed via CLI
    weight_decay = args.weight_decay if args.weight_decay is not None else config.get("weight_decay", 0.05)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up image resolution
    # Standard dimensions for comma2k19 are 1152x832, fallback to 1152x832 if not in dataset
    img_h, img_w = 832, 1152
    
    # 1. Load image
    if args.smoke_test:
        print("Running in SMOKE-TEST mode: Using a random tensor as image.")
        image = torch.randn(1, 3, img_h, img_w, device=device)
    else:
        # Load dataset to get the first frame
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
            image = dataset[0].unsqueeze(0).to(device)
            print(f"Successfully loaded first image from dataset. Resolution: {img_w}x{img_h}")
        except Exception as e:
            print(f"Error initializing dataset or loading image: {e}")
            print("Falling back to a randomly generated image tensor for training.")
            image = torch.randn(1, 3, img_h, img_w, device=device)

    # 2. Initialize Decoder model
    # Note: Using the MaskedAutoencoderHiera class but we bypass the encoder completely.
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

    # Compute shape of learnable latent parameters
    num_mask_units = math.prod(model.tokens_spatial_shape_final)
    mask_unit_spatial_shape_final = model.mask_unit_spatial_shape_final
    encoder_dim_out = model.decoder_embed.in_features
    
    latent_shape = (1, num_mask_units, *mask_unit_spatial_shape_final, encoder_dim_out)
    print(f"Initializing learnable latent parameter of shape {latent_shape}...")
    
    # Initialize learnable latent embeddings (similar to Truncated Normal distribution used in Meta's code)
    latent_param = nn.Parameter(torch.empty(*latent_shape, device=device))
    nn.init.trunc_normal_(latent_param, std=0.02)
    
    # 3. Setup optimizer
    # Select decoder-only trainable parameters (decoder_embed, decoder_pos_embed, decoder_blocks, decoder_norm, decoder_pred)
    decoder_params = []
    for name, param in model.named_parameters():
        if name.startswith("decoder_") or name.startswith("mask_token"):
            decoder_params.append(param)
            
    # Group parameters for weight decay configuration (ignore decay for 1D biases/norms and latent embeddings)
    param_groups = [
        {
            "params": [p for p in decoder_params if p.dim() > 1],
            "weight_decay": weight_decay
        },
        {
            "params": [p for p in decoder_params if p.dim() <= 1] + [latent_param],
            "weight_decay": 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    
    # Cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Mixed precision configuration
    amp_enabled = config.get("amp", True)
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda" and amp_dtype == torch.float16)

    # Cache target labels for whole-image reconstruction loss
    # Pass a mask of all True to get labels for all patches
    all_patches_mask = torch.ones((1, num_mask_units), dtype=torch.bool, device=device)
    target_labels = model.get_pixel_label_2d(image, all_patches_mask, norm=True)
    
    pad_width = len(str(args.epochs))
    print("Starting overfitting training...")
    loss_history = []
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype):
            # Forward pass through decoder
            pred, _ = model.forward_decoder(latent_param, all_patches_mask)
            
            # Reconstruction loss on all patches
            pred_selected = pred[all_patches_mask]
            loss = ((pred_selected - target_labels) ** 2).mean()
            
        scaler.scale(loss).backward()
        
        if config.get("grad_clip", 1.0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([latent_param] + decoder_params, config.get("grad_clip", 1.0))
            
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
            reconstruct_and_save(model, latent_param, image, epoch, recon_path)

    # 4. Save final outputs
    print("\nTraining completed.")
    
    # Save loss graph
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Decoder Overfitting Loss Curve")
    plt.grid(True)
    loss_graph_path = output_dir / "loss_graph.png"
    plt.savefig(loss_graph_path)
    plt.close()
    print(f"Saved loss graph to {loss_graph_path}")
    
    # Save final comparison
    final_recon_path = output_dir / "final_recon.png"
    reconstruct_and_save(model, latent_param, image, args.epochs, final_recon_path)
    
    # Save trained checkpoint (decoder weights + learnable latent parameters)
    checkpoint_path = output_dir / "overfit_checkpoint.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": {k: v for k, v in model.state_dict().items() if k.startswith("decoder_") or k.startswith("mask_token")},
        "latent_param": latent_param.detach().cpu(),
        "loss_history": loss_history
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
