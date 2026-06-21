"""
Generate reconstruction comparison images from a trained Hiera MAE checkpoint.
"""

import argparse
import sys
import os
import yaml
import pathlib
from pathlib import Path

import torch
import torchvision.transforms as T

# Ensure the repo root is in the python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset.frame_dataset import FrameDataset
from model.hiera.hiera_mae import MaskedAutoencoderHiera

def main():
    parser = argparse.ArgumentParser(description="Generate reconstruction visuals from a checkpoint.")
    parser.add_argument("exp_dir", type=str, help="Experiment directory (e.g., experiments/frame_model_1)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_latest.pt", 
                        help="Checkpoint filename to load (default: checkpoint_latest.pt)")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images to reconstruct (max 4)")
    parser.add_argument("--mode", type=str, default="val", choices=["train", "val"], 
                        help="Dataset split to visualize from (default: val)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Specific dataset frame indices to reconstruct")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths
    data_dir = exp_dir / "data"
    vis_dir = exp_dir / "vis"
    checkpoint_path = data_dir / args.checkpoint

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint was saved at epoch {epoch}")

    # Initialize Dataset
    split_path = exp_dir / "dataset_split.json"
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = FrameDataset(
        dataset_dir=config["data_path"],
        split_path=str(split_path),
        mode=args.mode,
        train_split=config.get("train_split", None),
        val_split=config["val_split"],
        seed=config["seed"],
        transform=transform
    )
    
    if args.indices:
        print(f"Loading specific indices: {args.indices} from '{args.mode}' dataset split...")
        frames = []
        for idx in args.indices:
            if idx < len(val_dataset):
                frames.append(val_dataset[idx])
            else:
                print(f"Warning: Index {idx} is out of bounds for the '{args.mode}' dataset split (len={len(val_dataset)}).")
        if not frames:
            print("Error: No valid indices loaded.")
            sys.exit(1)
        vis_batch = torch.stack(frames)
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.num_images,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        # Load first batch
        vis_batch = None
        for batch in val_loader:
            vis_batch = batch
            break
        if vis_batch is None:
            print("Error: Could not retrieve a batch from validation dataset.")
            sys.exit(1)

    # Setup Model
    img_w = val_dataset.width
    img_h = val_dataset.height
    print(f"Image resolution: {img_w}x{img_h}")

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

    # Load weights
    sd = checkpoint["model_state_dict"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    getattr(model, "_orig_mod", model).load_state_dict(sd)
    model.eval()

    # Generate reconstructions
    print("Generating reconstructions...")
    
    # Custom save logic to avoid standard epoch name collision
    model.eval()
    with torch.no_grad():
        num_to_vis = len(vis_batch) if args.indices else min(args.num_images, len(vis_batch))
        images = vis_batch[:num_to_vis].to(device)
        B, C, H, W = images.shape
        raw_m = getattr(model, "_orig_mod", model)
        P = raw_m.pred_stride
        NH = H // P
        NW = W // P
        
        latent, mask = raw_m.forward_encoder(images, mask_ratio=config["mask_ratio"])
        pred, pred_mask = raw_m.forward_decoder(latent, mask)
        
        # Unfold original images
        images_hwc = images.permute(0, 2, 3, 1)
        patches_orig = images_hwc.reshape(B, NH, P, NW, P, C)
        patches_orig = patches_orig.permute(0, 1, 3, 2, 4, 5)
        patches_orig = patches_orig.reshape(B, NH * NW, P * P * C)
        
        mean = patches_orig.mean(dim=-1, keepdim=True)
        var = patches_orig.var(dim=-1, keepdim=True)
        std = (var + 1e-6) ** 0.5
        
        pred_denorm = pred * std + mean
        # Correct layout from (C, P, P) to (P, P, C) to match patches_orig
        pred_denorm = pred_denorm.reshape(B, NH * NW, C, P, P).permute(0, 1, 3, 4, 2).reshape(B, NH * NW, P * P * C)
        pred_mask_expanded = pred_mask.unsqueeze(-1)
        
        patches_recon = torch.where(pred_mask_expanded, patches_orig, pred_denorm)
        patches_masked = torch.where(pred_mask_expanded, patches_orig, torch.zeros_like(patches_orig))
        
        def patches_to_img(p):
            p = p.reshape(B, NH, NW, P, P, C)
            p = p.permute(0, 1, 3, 2, 4, 5)
            p = p.reshape(B, H, W, C)
            return p.permute(0, 3, 1, 2)
            
        img_recon = patches_to_img(patches_recon)
        img_masked = patches_to_img(patches_masked)
        
        mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        
        img_orig_vis = (images * std_t + mean_t).clamp(0, 1)
        img_masked_vis = (img_masked * std_t + mean_t).clamp(0, 1)
        img_recon_vis = (img_recon * std_t + mean_t).clamp(0, 1)
        
        grid_list = []
        for i in range(B):
            grid_list.append(img_orig_vis[i])
            grid_list.append(img_masked_vis[i])
            grid_list.append(img_recon_vis[i])
            
        import torchvision.utils as vutils
        from PIL import Image
        grid = vutils.make_grid(grid_list, nrow=3, padding=4, normalize=False)
        grid_cpu = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        im = Image.fromarray(grid_cpu)
        
        vis_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = Path(args.checkpoint).stem
        save_path = vis_dir / f"reconstruction_{ckpt_name}.png"
        im.save(save_path)
        print(f"Reconstruction image saved successfully to {save_path}")

if __name__ == "__main__":
    main()
