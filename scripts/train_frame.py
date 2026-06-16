"""
Train the Hiera Masked Autoencoder (MAE) on frame-level WebP images.
Manages experiments, loads FrameDataset, configures the AdamW optimizer with weight decay 
exemptions, sets up a linear warmup + cosine decay scheduler, and runs the training loop.
"""

import argparse
import sys
import os
import random
import yaml
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

# Ensure the repo root is in the python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset import FrameDataset
from model.hiera.hiera_mae import MaskedAutoencoderHiera
from train import (
    train_frame,
    save_final_frame,
    get_parameter_groups,
    get_cosine_schedule_with_warmup,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_latest_exp_dir(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("frame_model_")]
    if not existing:
        return os.path.join(base_dir, "frame_model_0")
    
    nums = []
    for d in existing:
        try:
            nums.append(int(d.split("_")[2]))
        except:
            pass
    
    if not nums:
        return os.path.join(base_dir, "frame_model_0")
    
    return os.path.join(base_dir, f"frame_model_{max(nums)}")

def load_defaults():
    default_path = REPO_ROOT / "experiments" / "frame_default.yaml"
    if default_path.exists():
        with open(default_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_explicit_cli_args():
    """Parse only the command-line arguments explicitly passed by the user, ignoring defaults."""
    p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    p.add_argument("--new", action="store_true")
    p.add_argument("--exp-dir", type=str)
    p.add_argument("--seed", type=int)
    p.add_argument("--data-path", type=str)
    p.add_argument("--val-split", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("-w", "--workers", dest="workers", type=int)
    p.add_argument("--mask-ratio", type=float)
    p.add_argument("--embed-dim", type=int)
    p.add_argument("--num-heads", type=int)
    p.add_argument("--stages", type=int, nargs="+")
    p.add_argument("--q-pool", type=int)
    p.add_argument("--q-stride", type=int, nargs="+")
    p.add_argument("--mask-unit-size", type=int, nargs="+")
    p.add_argument("--patch-stride", type=int, nargs="+")
    p.add_argument("--mlp-ratio", type=float)
    p.add_argument("--decoder-embed-dim", type=int)
    p.add_argument("--decoder-depth", type=int)
    p.add_argument("--decoder-num-heads", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--warmup-epochs", type=int)
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--grad-clip", type=float)
    p.add_argument("--save-every", type=int)
    p.add_argument("--max-batches-per-epoch", type=int)
    
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
    p.add_argument("--amp", type=str2bool, nargs='?')
    p.add_argument("--amp-dtype", type=str, choices=["float16", "bfloat16"])
    p.add_argument("--compile", type=str2bool, nargs='?')
    p.add_argument("--compile-mode", type=str)
    
    parsed, _ = p.parse_known_args()
    return vars(parsed)

def main():
    import shutil
    
    # 1. Parse explicit CLI overrides
    cli_overrides = get_explicit_cli_args()
    
    # 2. Check if "--new" flag is provided (or if the user entered "new" as the exp-dir)
    is_new = cli_overrides.get("new", False) or cli_overrides.get("exp_dir") == "new"
    
    if is_new:
        base_dir = "experiments"
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if d.startswith("frame_model_")]
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_")[2]))
            except:
                pass
        next_num = max(nums) + 1 if nums else 0
        new_exp_dir = os.path.join(base_dir, f"frame_model_{next_num}")
        
        # Create directories
        os.makedirs(os.path.join(new_exp_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(new_exp_dir, "vis"), exist_ok=True)
        
        # Copy frame_default.yaml to config.yaml in the new directory
        default_yaml_path = REPO_ROOT / "experiments" / "frame_default.yaml"
        new_config_path = os.path.join(new_exp_dir, "config.yaml")
        
        if default_yaml_path.exists():
            shutil.copy(default_yaml_path, new_config_path)
            print(f"Created new experiment directory: {new_exp_dir}")
            print(f"Cloned default configuration to: {new_config_path}")
        else:
            print(f"Error: Default configuration file not found at {default_yaml_path}")
            sys.exit(1)
            
        print("Please edit config.yaml to your liking and rerun the script without --new.")
        return
        
    # 3. Determine the experiment directory
    exp_dir = cli_overrides.get("exp_dir", None)
    if exp_dir is None:
        # Default to the largest existing frame_model_<num>
        exp_dir = get_latest_exp_dir()
        
    data_dir = os.path.join(exp_dir, "data")
    vis_dir = os.path.join(exp_dir, "vis")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    config_path = os.path.join(exp_dir, "config.yaml")
    
    # 4. Build the merged configuration
    # Start with default hyperparameters from frame_default.yaml
    config = load_defaults()
    
    # Update with config.yaml in the experiment directory if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        if saved_config:
            print(f"Loaded config.yaml from {exp_dir}")
            config.update(saved_config)
    else:
        # If it doesn't exist, this must be frame_model_0 or a clean dir. We clone the defaults.
        print(f"No config.yaml found in {exp_dir}. Creating from defaults...")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    # Finally, apply explicit CLI overrides
    if cli_overrides:
        overrides_to_apply = {k: v for k, v in cli_overrides.items() if k not in ["new", "exp_dir"]}
        if overrides_to_apply:
            print(f"Applying command-line overrides: {overrides_to_apply}")
            config.update(overrides_to_apply)
            
    # Save the updated configuration back to config.yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # Check if a checkpoint exists to resume
    resume_path = os.path.join(data_dir, "checkpoint_latest.pt")
    resume_epoch = 0
    if os.path.exists(resume_path):
        print(f"Found existing checkpoint at {resume_path}. Resuming...")
        checkpoint = torch.load(resume_path, map_location="cpu")
        resume_epoch = checkpoint["epoch"]

    # 5. Reproducibility Seed
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Experiment: {exp_dir}")

    # 6. FrameDataset Initialization
    # Split JSON is saved inside the experiment directory to keep splits constant
    split_path = Path(exp_dir) / "dataset_split.json"

    # Image normalization (standard ImageNet specs)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FrameDataset(
        dataset_dir=config["data_path"],
        split_path=str(split_path),
        mode="train",
        val_split=config["val_split"],
        seed=config["seed"],
        transform=transform
    )
    val_dataset = FrameDataset(
        dataset_dir=config["data_path"],
        split_path=str(split_path),
        mode="val",
        val_split=config["val_split"],
        seed=config["seed"],
        transform=transform
    )

    img_w = train_dataset.width
    img_h = train_dataset.height
    print(f"Dataset Frame Resolution: {img_w}x{img_h}")


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        pin_memory=True,
        drop_last=False
    )

    # 7. Model Setup
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

    if config["compile"] and device.type == "cuda":
        model = torch.compile(model, mode=config["compile_mode"])

    # 8. Scaled Learning Rate & Parameter Weight Decay Setup
    scaled_lr = config["lr"] * (config["batch_size"] / 256.0)
    print(f"Base Learning Rate: {config['lr']:.6f} | Scaled Learning Rate: {scaled_lr:.6f} (scaled by batch_size/256)")

    param_groups = get_parameter_groups(model, weight_decay=config["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, lr=scaled_lr)

    # Cosine learning rate schedule with step-level linear warmup
    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = len(train_loader) * config["warmup_epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 9. Resume from Checkpoint
    if resume_epoch > 0:
        checkpoint = torch.load(resume_path, map_location=device)
        sd = checkpoint["model_state_dict"]
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        getattr(model, "_orig_mod", model).load_state_dict(sd)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Successfully resumed from epoch {resume_epoch}")

    # 10. Execute Training Loop
    stats_path = train_frame(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config["num_epochs"],
        save_dir=data_dir,
        vis_dir=vis_dir,
        val_loader=val_loader,
        save_every=config["save_every"],
        grad_clip=config["grad_clip"],
        max_batches_per_epoch=config.get("max_batches_per_epoch", None),
        resume_epoch=resume_epoch,
        mask_ratio=config["mask_ratio"],
        amp=config.get("amp", False),
        amp_dtype=config.get("amp_dtype", "float16"),
    )
    
    save_final_frame(model, optimizer, stats_path, config["num_epochs"], data_dir)


if __name__ == "__main__":
    main()
