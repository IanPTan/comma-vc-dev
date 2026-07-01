import argparse
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path to allow importing model
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model import WIRE

# Define a color map for the 5 SegNet classes
COLOR_MAP = np.array([
    [0, 0, 0],        # Class 0: Black
    [255, 0, 0],      # Class 1: Red
    [0, 255, 0],      # Class 2: Green
    [0, 0, 255],      # Class 3: Blue
    [255, 255, 0]     # Class 4: Yellow
], dtype=np.uint8)

def colorize_mask(mask_2d):
    """Converts a 2D class integer mask to an RGB image."""
    return COLOR_MAP[mask_2d]

def main():
    parser = argparse.ArgumentParser(description="Overfit WIRE model to SegNet mask of frame 0.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=196608, help="Batch size for training (default 196608 for full-batch)")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--omega0", type=float, default=20.0, help="Gabor frequency omega0")
    parser.add_argument("--s0", type=float, default=10.0, help="Gabor scaling s0")
    parser.add_argument("--hidden_features", type=int, default=256, help="Width of hidden layers")
    parser.add_argument("--hidden_layers", type=int, default=3, help="Number of layers (depth)")
    parser.add_argument("--complex_weights", action="store_true", help="Use complex weights instead of real weights")
    parser.add_argument("--use_scheduler", action="store_true", help="Use CosineAnnealingLR scheduler to stabilize training")
    args = parser.parse_args()

    # Logger setup
    log_lines = []
    def log_print(msg):
        print(msg)
        log_lines.append(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Training on device: {device}")
    log_print(f"Command run: python {' '.join(sys.argv)}")
    
    # 1. Load both Frame 0 and SegNet mask 0 from H5 file
    h5_path = Path("data/frames.h5")
    if not h5_path.exists():
        h5_path = Path("data/frames_test.h5")
    if not h5_path.exists():
        log_print("Error: Could not find data/frames.h5 or data/frames_test.h5")
        sys.exit(1)
        
    log_print(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        frame = f['frames'][0]  # shape (H, W, 3), uint8
        seg_mask = f['seg'][0]  # shape (H, W), uint8
        
    H, W, C = frame.shape
    log_print(f"Dimensions: {W}x{H} with {C} channels")
    
    # Move target mask to device
    target_mask = torch.from_numpy(seg_mask).to(device)
    target_flat = target_mask.view(-1).long()  # (H*W,) long integers for cross-entropy
    
    # 2. Generate normalized 2D coordinate grid [-1, 1]
    y_coords = torch.linspace(-1, 1, steps=H, device=device)
    x_coords = torch.linspace(-1, 1, steps=W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)  # (H*W, 2)
    
    # Cast coordinates to bfloat16
    coords = coords.bfloat16()
    
    # 3. Setup experiment directories
    experiments_dir = Path("experiments/frame0_test")
    recon_dir = experiments_dir / "recon"
    
    if experiments_dir.exists():
        import shutil
        shutil.rmtree(experiments_dir)
        
    experiments_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare standard reference images
    orig_rgb_pil = Image.fromarray(frame)
    orig_seg_pil = Image.fromarray(colorize_mask(seg_mask))
    
    # Save target reference side-by-side as 0000.png
    # Layout: [Original RGB] | [Original SegNet] | [Original SegNet] | [Black Image]
    black_img = Image.new('RGB', (W, H))
    combined_ref = Image.new('RGB', (4 * W, H))
    combined_ref.paste(orig_rgb_pil, (0, 0))
    combined_ref.paste(orig_seg_pil, (W, 0))
    combined_ref.paste(orig_seg_pil, (2 * W, 0))
    combined_ref.paste(black_img, (3 * W, 0))
    combined_ref.save(recon_dir / "0000.png")
    
    # 4. Instantiate WIRE model
    model = WIRE(
        in_features=2,
        out_features=5,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        omega0=args.omega0,
        s0=args.s0,
        complex_weights=args.complex_weights,
        init_type='siren'
    ).to(device)
    
    model = model.bfloat16()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    loss_history = []
    acc_history = []
    
    log_print("=== Experiment Configuration ===")
    log_print(f"Model Architecture:")
    log_print(f"  - Model Type: WIRE")
    log_print(f"  - in_features: 2 (2D coordinates)")
    log_print(f"  - out_features: 5 (5-class SegNet logits)")
    log_print(f"  - hidden_features: {args.hidden_features}")
    log_print(f"  - hidden_layers: {args.hidden_layers}")
    log_print(f"  - complex_weights: {args.complex_weights}")
    log_print(f"  - init_type: siren")
    log_print(f"Gabor Parameters:")
    log_print(f"  - omega0: {args.omega0}")
    log_print(f"  - s0: {args.s0}")
    log_print(f"Training Settings:")
    log_print(f"  - epochs: {args.epochs}")
    log_print(f"  - batch_size: {args.batch_size}")
    log_print(f"  - learning_rate: {args.lr}")
    log_print(f"  - optimizer: Adam")
    log_print(f"  - criterion: CrossEntropyLoss")
    log_print(f"  - use_scheduler: {args.use_scheduler}")
    if args.use_scheduler:
        log_print(f"  - scheduler: CosineAnnealingLR (T_max={args.epochs})")
    log_print("=================================\n")
    
    pbar = tqdm(range(1, args.epochs + 1), desc="Fitting SegNet Mask")
    for epoch in pbar:
        permutation = torch.randperm(coords.size(0))
        epoch_loss = 0.0
        epoch_correct = 0
        
        for i in range(0, coords.size(0), args.batch_size):
            indices = permutation[i:i+args.batch_size]
            batch_coords = coords[indices]
            batch_targets = target_flat[indices]
            
            optimizer.zero_grad()
            pred = model(batch_coords)  # (batch_size, 5)
            loss = criterion(pred, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(indices)
            epoch_correct += (pred.argmax(dim=-1) == batch_targets).sum().item()
            
        epoch_loss /= coords.size(0)
        epoch_acc = epoch_correct / coords.size(0)
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        
        if args.use_scheduler:
            scheduler.step()
            
        # Update progress stats
        pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc*100:.2f}%")
        
        # Capture and save side-by-side reconstruction at exponential intervals (powers of 2) or final epoch
        if (epoch & (epoch - 1)) == 0 or epoch == args.epochs:
            with torch.no_grad():
                preds = []
                for i in range(0, coords.size(0), args.batch_size):
                    preds.append(model(coords[i:i+args.batch_size]))
                pred_all = torch.cat(preds, dim=0).view(H, W, 5)
                
                # A. Argmax Predicted Mask
                pred_mask = pred_all.argmax(dim=-1).cpu().numpy().astype(np.uint8)
                pred_seg_color = colorize_mask(pred_mask)
                pred_seg_pil = Image.fromarray(pred_seg_color)
                
                # B. Raw logits RGB representation (sigmoid of the first 3 logits)
                logits_rgb = torch.sigmoid(pred_all[..., :3]).float().cpu().numpy()
                logits_rgb_np = (logits_rgb * 255.0).astype(np.uint8)
                logits_pil = Image.fromarray(logits_rgb_np)
                
            # Layout: [Original RGB] | [Original SegNet] | [Predicted SegNet] | [Raw Logits RGB]
            combined_img = Image.new('RGB', (4 * W, H))
            combined_img.paste(orig_rgb_pil, (0, 0))
            combined_img.paste(orig_seg_pil, (W, 0))
            combined_img.paste(pred_seg_pil, (2 * W, 0))
            combined_img.paste(logits_pil, (3 * W, 0))
            combined_img.save(recon_dir / f"{epoch:04d}.png")
            
    # 5. Save the trained model weights
    model_path = experiments_dir / "frame0.pt"
    torch.save(model.state_dict(), model_path)
    
    # Plot and save the loss graph with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross Entropy Loss', color=color)
    ax1.plot(range(1, args.epochs + 1), loss_history, color=color, label='Cross Entropy Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  # second axes sharing the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('SegNet Loss Term (100 * distortion)', color=color)
    seg_loss_history = [100.0 * (1.0 - acc) for acc in acc_history]
    ax2.plot(range(1, args.epochs + 1), seg_loss_history, color=color, label='SegNet Loss Term')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('SegNet Mask Fitting Loss and Distortion')
    fig.tight_layout()
    plt.savefig(experiments_dir / "loss.png")
    plt.close()
    
    log_print(f"\nDone! SegNet fitting completed.")
    log_print(f"  Model weights saved to:     {model_path}")
    log_print(f"  Final training accuracy:    {epoch_acc*100:.2f}%")
    log_print(f"  Resulting SegNet loss term (100 * distortion): {100.0 * (1.0 - epoch_acc):.4f}")
    log_print(f"  Loss plot saved to:         {experiments_dir}/loss.png")
    log_print(f"  Reconstructed masks saved:  {recon_dir}/")

    # Save all accumulated log prints to report.txt
    report_path = experiments_dir / "report.txt"
    with open(report_path, "w") as rf:
        rf.write("\n".join(log_lines) + "\n")

if __name__ == '__main__':
    main()
