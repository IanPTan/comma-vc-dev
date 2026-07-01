import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to sys.path to allow importing model
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model import WIRE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Load frame 0 from H5 file
    h5_path = Path("data/frames.h5")
    if not h5_path.exists():
        h5_path = Path("data/frames_test.h5")
    if not h5_path.exists():
        print("Error: Could not find data/frames.h5 or data/frames_test.h5")
        sys.exit(1)
        
    print(f"Loading frame 0 from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        frame = f['frames'][0]  # shape (H, W, 3), uint8
        
    H, W, C = frame.shape
    print(f"Frame 0 dimensions: {W}x{H} with {C} channels")
    
    # Normalize image to [0, 1] and move to device
    img_target = torch.from_numpy(frame).float().to(device) / 255.0
    img_target_flat = img_target.view(-1, C)  # (H*W, 3)
    
    # 2. Generate normalized 2D coordinate grid [-1, 1]
    y_coords = torch.linspace(-1, 1, steps=H, device=device)
    x_coords = torch.linspace(-1, 1, steps=W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)  # (H*W, 2)
    
    # Cast input coordinates and targets to bfloat16
    print("Converting inputs to bfloat16...")
    coords = coords.bfloat16()
    img_target_flat = img_target_flat.bfloat16()
    
    # 3. Setup experiment directories
    experiments_dir = Path("experiments/frame0_test")
    recon_dir = experiments_dir / "recon"
    
    if experiments_dir.exists():
        import shutil
        shutil.rmtree(experiments_dir)
        
    experiments_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original image
    Image.fromarray(frame).save(recon_dir / "original.png")
    
    # 4. Run parameter-sweeping experiments
    widths = [8, 16, 32, 64, 128, 256, 512]
    num_trials = 10
    epochs = 100
    batch_size = 131072
    
    loss_h5_path = experiments_dir / "loss.h5"
    print(f"Starting sweeps: widths={widths}, trials={num_trials}, epochs={epochs}")
    
    with h5py.File(loss_h5_path, 'w') as h5_file:
        for w in widths:
            group = h5_file.create_group(f"width_{w}")
            print(f"\n--- Width {w} ---")
            
            for t in range(num_trials):
                # Instantiate real-valued WIRE model
                model = WIRE(
                    in_features=2,
                    out_features=3,
                    hidden_features=w,
                    hidden_layers=3,
                    omega0=20.0,
                    s0=10.0,
                    complex_weights=False,
                    init_type='siren'
                ).to(device)
                
                model = model.bfloat16()
                
                optimizer = optim.Adam(model.parameters(), lr=5e-3)
                criterion = nn.MSELoss()
                
                trial_losses = []
                
                # Wrap epoch range in tqdm progress bar
                pbar = tqdm(range(1, epochs + 1), desc=f"Width {w:3d} | Trial {t:2d}", leave=True)
                for epoch in pbar:
                    permutation = torch.randperm(coords.size(0))
                    epoch_loss = 0.0
                    
                    for i in range(0, coords.size(0), batch_size):
                        indices = permutation[i:i+batch_size]
                        batch_coords = coords[indices]
                        batch_targets = img_target_flat[indices]
                        
                        optimizer.zero_grad()
                        pred = model(batch_coords)
                        loss = criterion(pred, batch_targets)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item() * len(indices)
                        
                    epoch_loss /= coords.size(0)
                    trial_losses.append(epoch_loss)
                    
                    # Update progress bar stats
                    psnr = -10.0 * np.log10(epoch_loss) if epoch_loss > 0 else float('inf')
                    pbar.set_postfix(loss=f"{epoch_loss:.6f}", psnr=f"{psnr:.2f}dB")
                    
                # Save loss history
                group.create_dataset(f"trial_{t}", data=np.array(trial_losses, dtype=np.float32))
                
                # Generate final reconstruction
                with torch.no_grad():
                    preds = []
                    for i in range(0, coords.size(0), batch_size):
                        preds.append(model(coords[i:i+batch_size]))
                    pred_img = torch.cat(preds, dim=0).view(H, W, C)
                    pred_img = pred_img.float().clamp(0.0, 1.0).cpu().numpy()
                    
                recon_img_np = (pred_img * 255.0).astype(np.uint8)
                pil_img = Image.fromarray(recon_img_np)
                pil_img.save(recon_dir / f"width_{w}_trial_{t}.png")
                
    print(f"\nDone! All experiments completed.")
    print(f"  Loss histories saved in:  {loss_h5_path}")
    print(f"  Reconstructions saved in: {recon_dir}/")

if __name__ == '__main__':
    main()
