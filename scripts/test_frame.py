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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Load both Frame 0 and SegNet mask 0 from H5 file
    h5_path = Path("data/frames.h5")
    if not h5_path.exists():
        h5_path = Path("data/frames_test.h5")
    if not h5_path.exists():
        print("Error: Could not find data/frames.h5 or data/frames_test.h5")
        sys.exit(1)
        
    print(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        frame = f['frames'][0]  # shape (H, W, 3), uint8
        seg_mask = f['seg'][0]  # shape (H, W), uint8
        
    H, W, C = frame.shape
    print(f"Dimensions: {W}x{H} with {C} channels")
    
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
    
    # 4. Instantiate WIRE model (5 output channels for the 5 classes)
    model = WIRE(
        in_features=2,
        out_features=5,
        hidden_features=256,
        hidden_layers=3,
        omega0=20.0,
        s0=10.0,
        complex_weights=False,
        init_type='siren'
    ).to(device)
    
    model = model.bfloat16()
    
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    batch_size = 131072
    report_interval = 10
    
    loss_history = []
    
    print(f"Starting SegNet fitting: epochs={epochs}, batch_size={batch_size}, report_interval={report_interval}")
    
    pbar = tqdm(range(1, epochs + 1), desc="Fitting SegNet Mask")
    for epoch in pbar:
        permutation = torch.randperm(coords.size(0))
        epoch_loss = 0.0
        epoch_correct = 0
        
        for i in range(0, coords.size(0), batch_size):
            indices = permutation[i:i+batch_size]
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
        
        # Update progress stats
        pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc*100:.2f}%")
        
        # Capture and save side-by-side reconstruction at reported intervals
        if epoch == 1 or epoch % report_interval == 0:
            with torch.no_grad():
                preds = []
                for i in range(0, coords.size(0), batch_size):
                    preds.append(model(coords[i:i+batch_size]))
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
    
    # Plot and save the loss graph
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_history, label='Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SegNet Mask Fitting Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(experiments_dir / "loss.png")
    plt.close()
    
    print(f"\nDone! SegNet fitting completed.")
    print(f"  Model weights saved to:     {model_path}")
    print(f"  Final training accuracy:    {epoch_acc*100:.2f}%")
    print(f"  Loss plot saved to:         {experiments_dir}/loss.png")
    print(f"  Reconstructed masks saved:  {recon_dir}/")

if __name__ == '__main__':
    main()
