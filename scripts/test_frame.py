import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
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
    
    # 3. Instantiate WIRE model
    print("Instantiating WIRE model...")
    model = WIRE(
        in_features=2,
        out_features=3,
        hidden_features=256,
        hidden_layers=3,
        omega0=20.0,
        s0=10.0,
        complex_weights=False,
        init_type='siren'
    ).to(device)
    
    # Create experiments directory structure
    experiments_dir = Path("experiments/frame0_test")
    recon_dir = experiments_dir / "recon"
    
    # Wipe the directory if it exists to avoid mixing files from old runs
    if experiments_dir.exists():
        import shutil
        shutil.rmtree(experiments_dir)
        
    experiments_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the original image as 0000.png in the recon directory
    orig_pil = Image.fromarray(frame)
    orig_pil.save(recon_dir / "0000.png")
    
    # Cast model and inputs to bfloat16
    print("Converting model and inputs to bfloat16...")
    model = model.bfloat16()
    coords = coords.bfloat16()
    img_target_flat = img_target_flat.bfloat16()
    
    # 4. Train the model to overfit using minibatches to prevent VRAM overflow
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()
    
    epochs = 100
    batch_size = 131072
    report_interval = 10
    print(f"Overfitting to frame 0 for {epochs} epochs (batch size: {batch_size}, report interval: {report_interval})...")
    
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        # Shuffle coordinates each epoch
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
        loss_history.append(epoch_loss)
        
        # Merge printing and reconstruction capture condition
        if epoch == 1 or epoch % report_interval == 0:
            psnr = -10.0 * np.log10(epoch_loss) if epoch_loss > 0 else float('inf')
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {epoch_loss:.6f} | PSNR: {psnr:.2f} dB")
            
            # Capture the reconstruction
            with torch.no_grad():
                preds = []
                for i in range(0, coords.size(0), batch_size):
                    preds.append(model(coords[i:i+batch_size]))
                pred_img = torch.cat(preds, dim=0).view(H, W, C)
                pred_img = pred_img.float().clamp(0.0, 1.0).cpu().numpy()
            recon_img_np = (pred_img * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(recon_img_np)
            
            # Save the reconstruction frame as f"{epoch:04d}.png" in the recon directory
            pil_img.save(recon_dir / f"{epoch:04d}.png")
            
    # 5. Save the trained model weights
    model_path = experiments_dir / "frame0.pt"
    torch.save(model.state_dict(), model_path)
    
    # Plot and save the loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(experiments_dir / "loss.png")
    plt.close()
    
    print(f"\nDone!")
    print(f"  Model weights saved to:     {model_path}")
    print(f"  Loss plot saved to:         {experiments_dir}/loss.png")
    print(f"  Reconstructed frames saved to: {recon_dir}/")

if __name__ == '__main__':
    main()
