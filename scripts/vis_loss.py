import argparse
import os
import pathlib
import h5py
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Visualize training loss from HDF5 stats.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory.")
    args = parser.parse_args()

    exp_dir = pathlib.Path(args.exp_dir)
    stats_path = exp_dir / "data" / "stats.h5"
    vis_dir = exp_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not stats_path.exists():
        print(f"Error: Stats file not found at {stats_path}")
        return

    with h5py.File(stats_path, 'r') as f:
        train_loss = f["loss/train"][:]
        val_loss = f["loss/val"][:]

    # Filter out NaNs (epochs that haven't run yet)
    mask = ~np.isnan(train_loss)
    train_loss = train_loss[mask]
    val_loss = val_loss[mask]
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o', markersize=4)
    if not np.all(np.isnan(val_loss)):
        plt.plot(epochs, val_loss, label="Val Loss", marker='s', markersize=4)
    
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss: {exp_dir.name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = vis_dir / "loss.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Loss plot saved to {save_path}")

if __name__ == "__main__":
    main()
