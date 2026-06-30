import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
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
        complex_weights=True,
        init_type='siren'
    ).to(device)
    
    # 4. Train the model to overfit using minibatches to prevent VRAM overflow
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()
    
    epochs = 1000
    batch_size = 131072
    print(f"Overfitting to frame 0 for {epochs} epochs (batch size: {batch_size})...")
    
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
        
        if epoch % 100 == 0 or epoch == 1:
            psnr = -10.0 * np.log10(epoch_loss) if epoch_loss > 0 else float('inf')
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {epoch_loss:.6f} | PSNR: {psnr:.2f} dB")
            
    # 5. Save the trained model and reconstructed image
    with torch.no_grad():
        preds = []
        for i in range(0, coords.size(0), batch_size):
            preds.append(model(coords[i:i+batch_size]))
        final_pred = torch.cat(preds, dim=0).view(H, W, C)
        final_pred = final_pred.clamp(0.0, 1.0).cpu().numpy()
        
    recon_img_np = (final_pred * 255.0).astype(np.uint8)
    recon_pil = Image.fromarray(recon_img_np)
    
    # Load original frame as PIL image
    orig_pil = Image.fromarray(frame)
    
    # Concatenate original and reconstructed images side-by-side
    combined_img = Image.new('RGB', (2 * W, H))
    combined_img.paste(orig_pil, (0, 0))
    combined_img.paste(recon_pil, (W, 0))
    
    # Save directly in data/
    recon_path = Path("data/frame0_recon.png")
    model_path = Path("data/frame0.pt")
    
    combined_img.save(recon_path)
    torch.save(model.state_dict(), model_path)
    
    print(f"\nDone!")
    print(f"  Reconstructed frame saved to: {recon_path}")
    print(f"  Model weights saved to:       {model_path}")

if __name__ == '__main__':
    main()
