#!/usr/bin/env python3
"""
Script to visualize autoencoder reconstructions from an experiment directory.
Saves the comparison plot as a PNG in the experiment's vis/ subdirectory.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to python path to allow importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FrameDataset
from model import Autoencoder
from scripts.train_ae import get_largest_experiment_dir, load_config

def main():
    parser = argparse.ArgumentParser(description="Visualize Autoencoder reconstructions from an experiment.")
    parser.add_argument('experiment_dir', nargs='?', default=None,
                        help='Path to the experiment directory. Defaults to the one with the largest number.')
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                        help='Which dataset split to visualize (default: train).')
    parser.add_argument('-i', '--indices', type=int, nargs='+', default=None,
                        help='List of indices to visualize. If not specified, 4 random samples are chosen.')
    args = parser.parse_args()

    # Resolve experiment directory
    if args.experiment_dir is None:
        experiment_dir = get_largest_experiment_dir()
    else:
        experiment_dir = args.experiment_dir

    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory '{experiment_dir}' does not exist.")
        return

    print(f"Using experiment directory: {experiment_dir}")

    # Load config
    config = load_config(experiment_dir)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and recreate the split
    dataset_path = config['dataset']
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' does not exist.")
        return

    dataset = FrameDataset(dataset_path)
    val_split = config.get('val_split', 0.0)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    if val_size > 0:
        generator = torch.Generator().manual_seed(config['seed'])
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = dataset
        val_dataset = []

    # Select split
    if args.split == 'val':
        if len(val_dataset) == 0:
            print("Error: The validation split is empty (val_split is 0.0).")
            return
        active_dataset = val_dataset
    else:
        active_dataset = train_dataset

    print(f"Split selected: {args.split} (Size: {len(active_dataset)})")

    # Load checkpoint
    data_dir = os.path.join(experiment_dir, "data")
    best_path = os.path.join(data_dir, "best.pt")
    latest_path = os.path.join(data_dir, "latest.pt")

    checkpoint_path = None
    if os.path.exists(best_path):
        checkpoint_path = best_path
    elif os.path.exists(latest_path):
        checkpoint_path = latest_path
    else:
        print(f"Error: No checkpoints found in '{data_dir}'. Run training first.")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model
    qat_enabled = config.get('qat', True)
    model = Autoencoder(
        base_channels=config['base_channels'],
        num_layers=config['num_layers'],
        bottleneck_channels=config['bottleneck_channels'],
        qat=qat_enabled
    ).to(device)

    if qat_enabled:
        import torch.ao.quantization as quantization
        model.decoder.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        quantization.prepare_qat(model.decoder, inplace=True)

    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()

    # Select indices to visualize
    if args.indices:
        indices = [idx for idx in args.indices if 0 <= idx < len(active_dataset)]
        if not indices:
            print("Error: None of the provided indices are valid for the selected split.")
            return
    else:
        random.seed(config['seed'])
        num_samples = min(4, len(active_dataset))
        indices = random.sample(range(len(active_dataset)), num_samples)

    print(f"Visualizing indices: {indices}")

    # Generate reconstructions and plot
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 3 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Fetch sample
            x = active_dataset[idx]
            x_in = x.unsqueeze(0).to(device)

            # Reconstruct
            x_out = model(x_in).squeeze(0).cpu()

            # Convert tensors to numpy images [0, 1]
            orig_np = x.permute(1, 2, 0).numpy()
            recon_np = torch.clamp(x_out, 0, 1).permute(1, 2, 0).numpy()

            # Plot original
            axes[i, 0].imshow(orig_np)
            axes[i, 0].set_title(f"Original (Index {idx})")
            axes[i, 0].axis('off')

            # Plot reconstruction
            axes[i, 1].imshow(recon_np)
            axes[i, 1].set_title("Reconstruction")
            axes[i, 1].axis('off')

    plt.tight_layout()

    # Save PNG
    vis_dir = os.path.join(experiment_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, "reconstructions.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Visualization saved to: {out_path}")

if __name__ == '__main__':
    main()
