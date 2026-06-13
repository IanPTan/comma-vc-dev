import argparse
import sys
import os
import yaml
import pathlib
from pathlib import Path

import torch
from tqdm import tqdm

# Ensure the repo root is in the python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset import DaliDataLoader


def get_latest_exp_dir(base_dir="experiments"):
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
    
    return os.path.join(base_dir, f"model_{max(nums)}")


def load_defaults():
    default_path = REPO_ROOT / "experiments" / "default.yaml"
    if default_path.exists():
        with open(default_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def parse_args(defaults):
    p = argparse.ArgumentParser(description="Validate DALI output tensors for NaNs and zero variance.")
    
    p.add_argument("--exp-dir", type=str, default=None, 
                   help="Specific experiment directory to read config from. Defaults to latest.")
    p.add_argument("--data-path", type=str, default=defaults.get("data_path", "data/comma2k19"), 
                   help="Path to the dataset root.")
    p.add_argument("--val-path", type=str, default=None, help="Optional validation data path.")
    p.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4))
    p.add_argument("-w", "--workers", type=int, default=defaults.get("workers", 4), help="Number of DALI threads.")
    p.add_argument("--device", type=str, default=defaults.get("device", "gpu"), choices=["gpu", "cpu"], help="DALI device backend.")
    p.add_argument("--clip-frames", type=int, default=defaults.get("clip_frames", 16))
    p.add_argument("--frame-size", type=int, default=defaults.get("frame_size", 256))
    p.add_argument("--end-safety-margin", type=int, default=defaults.get("end_safety_margin", 200))
    
    return p.parse_args()


def validate_loader(loader, name, device):
    """Scans every batch in the loader for anomalies."""
    print(f"\nScanning {name} split ({len(loader)} batches)...")
    
    nan_batches = 0
    inf_batches = 0
    zero_var_batches = 0
    total_batches = 0
    
    pbar = tqdm(loader, desc=f"Validating {name}")
    for batch in pbar:
        # Move to GPU for fast tensor operations
        x = batch.to(device).float()
        
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        
        # Check for solid color (variance near zero)
        # Calculate std across spatial and temporal dims, keep batch dim
        std = x.view(x.shape[0], -1).std(dim=1)
        has_zero_var = (std < 1e-3).any().item()
        
        if has_nan: nan_batches += 1
        if has_inf: inf_batches += 1
        if has_zero_var: zero_var_batches += 1
        total_batches += 1
        
        pbar.set_postfix({
            "NaN": nan_batches,
            "Inf": inf_batches,
            "ZeroVar": zero_var_batches
        })
        
    print(f"\n--- {name.upper()} RESULTS ---")
    print(f"Total Batches: {total_batches}")
    print(f"NaN Batches:   {nan_batches}")
    print(f"Inf Batches:   {inf_batches}")
    print(f"Solid Color:   {zero_var_batches}")
    
    return nan_batches + inf_batches + zero_var_batches


def main():
    defaults = load_defaults()
    args = parse_args(defaults)

    if args.exp_dir is None:
        exp_dir = get_latest_exp_dir()
    else:
        exp_dir = args.exp_dir
        
    config_path = os.path.join(exp_dir, "config.yaml")
    
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        for k, v in saved_config.items():
            if k not in ["exp_dir"]: # allow override of paths if needed, but stick to config otherwise
                setattr(args, k, v)
    else:
        print(f"Warning: No config.yaml found in {exp_dir}. Using defaults.")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    device_id = device.index if device.index is not None else 0
    print(f"Testing on device: {device}")

    # Initialize Train Loader
    try:
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
            shuffle=False # Disable shuffle for sequential validation
        )
        train_issues = validate_loader(train_loader, "train", device)
    except Exception as e:
        print(f"Failed to initialize or run train loader: {e}")
        train_issues = -1

    # Initialize Val Loader
    try:
        val_path = args.val_path if args.val_path else args.data_path
        val_loader = DaliDataLoader(
            val_path,
            mode="val",
            clip_frames=args.clip_frames,
            frame_size=args.frame_size,
            batch_size=args.batch_size,
            num_threads=args.workers,
            device=args.device,
            device_id=device_id,
            end_safety_margin=args.end_safety_margin,
            shuffle=False
        )
        val_issues = validate_loader(val_loader, "val", device)
    except Exception as e:
        print(f"Failed to initialize or run val loader: {e}")
        val_issues = -1

    print("\n=========================================")
    if train_issues == 0 and val_issues == 0:
        print("SUCCESS: No anomalies detected in DALI output.")
    else:
        print("WARNING: Anomalies found in the data stream.")
    print("=========================================\n")


if __name__ == "__main__":
    main()
