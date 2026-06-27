#!/usr/bin/env python3
"""
Script to train the autoencoder on the extracted frame dataset with experiment directories.
Saves checkpoints and results.h5 in the experiment's data subdirectory.
Supports resuming training from latest.pt.
"""

import argparse
import os
import sys
import random
import yaml
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to python path to allow importing local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FrameDataset
from model import Autoencoder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_largest_experiment_dir(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)
    largest_num = -1
    for name in os.listdir(base_dir):
        if name.startswith("autoencoder_") and os.path.isdir(os.path.join(base_dir, name)):
            try:
                num = int(name.split("_")[1])
                if num > largest_num:
                    largest_num = num
            except ValueError:
                pass
    if largest_num == -1:
        return os.path.join(base_dir, "autoencoder_0")
    else:
        return os.path.join(base_dir, f"autoencoder_{largest_num}")

def get_next_experiment_dir(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)
    largest_num = -1
    for name in os.listdir(base_dir):
        if name.startswith("autoencoder_") and os.path.isdir(os.path.join(base_dir, name)):
            try:
                num = int(name.split("_")[1])
                if num > largest_num:
                    largest_num = num
            except ValueError:
                pass
    next_num = largest_num + 1
    return os.path.join(base_dir, f"autoencoder_{next_num}")

def load_config(experiment_dir, defaults_path="experiments/autoencoder_defaults.yaml"):
    if not os.path.exists(defaults_path):
        raise FileNotFoundError(f"Defaults configuration file not found at '{defaults_path}'.")
        
    with open(defaults_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config_path = os.path.join(experiment_dir, "config.yaml")
    if os.path.exists(config_path):
        print(f"Loading experiment overrides from: {config_path}")
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config.update(user_config)
    else:
        # Clone defaults to config.yaml in the experiment directory
        os.makedirs(experiment_dir, exist_ok=True)
        import shutil
        shutil.copy(defaults_path, config_path)
        print(f"Cloned defaults to: {config_path}")
            
    return config

def main():
    parser = argparse.ArgumentParser(description="Train the Autoencoder model with experiment tracking.")
    parser.add_argument('experiment_dir', nargs='?', default=None,
                        help='Path to the experiment directory (e.g. experiments/autoencoder_0). '
                             'Use "new" to create a new experiment directory by incrementing the last number. '
                             'If not specified, defaults to the autoencoder directory with the largest number.')
    args = parser.parse_args()

    # Determine experiment directory
    if args.experiment_dir == "new":
        experiment_dir = get_next_experiment_dir()
        load_config(experiment_dir)
        print(f"Created new experiment directory: {experiment_dir}")
        return
    elif args.experiment_dir is None:
        experiment_dir = get_largest_experiment_dir()
    else:
        experiment_dir = args.experiment_dir

    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")

    # Load configuration
    config = load_config(experiment_dir)
    print("Resolved configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Set seed
    set_seed(config['seed'])

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset_path = config['dataset']
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' does not exist.")
        return

    dataset = FrameDataset(dataset_path)
    val_split = config.get('val_split', 0.0)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    if val_size > 0:
        # Random split with seed-controlled generator
        generator = torch.Generator().manual_seed(config['seed'])
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = dataset
        val_dataset = []
        
    print(f"Dataset loaded. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=(device.type == 'cuda')
    )
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=(device.type == 'cuda')
        )
    else:
        val_loader = None

    # Setup directories
    data_dir = os.path.join(experiment_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    latest_path = os.path.join(data_dir, "latest.pt")
    best_path = os.path.join(data_dir, "best.pt")
    results_path = os.path.join(data_dir, "results.h5")

    # Resumption logic (checking checkpoint file existence and start epoch)
    start_epoch = 1
    best_loss = float('inf')
    best_set = config.get('best_set', 'train')
    checkpoint = None

    if os.path.exists(latest_path):
        print(f"Found existing checkpoint at {latest_path}. Checking resumption state...")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
    elif os.path.exists(best_path):
        # Fallback to load best.pt just in case best_loss tracking is needed
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        if best_set == 'train':
            best_loss = best_checkpoint.get('train_loss', best_checkpoint.get('val_loss', float('inf')))
        else:
            best_loss = best_checkpoint.get('val_loss', float('inf'))

    # Initialize model
    qat_enabled = config.get('qat', True)
    qat_start_epoch = config.get('qat_start_epoch', 1)
    
    # Check if we should initialize with QAT already prepared
    use_qat_at_init = qat_enabled and start_epoch >= qat_start_epoch
    
    model = Autoencoder(
        base_channels=config['base_channels'],
        num_layers=config['num_layers'],
        bottleneck_channels=config['bottleneck_channels'],
        qat=use_qat_at_init
    ).to(device)

    model_is_qat = False
    if use_qat_at_init:
        import torch.ao.quantization as quantization
        model.decoder.qconfig = quantization.get_default_qat_qconfig('fbgemm')
        quantization.prepare_qat(model.decoder, inplace=True)
        model_is_qat = True
        print("Initialized model with QAT prepared (resuming QAT phase).")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler_patience = config.get('scheduler_patience', 5)
    scheduler_factor = config.get('scheduler_factor', 0.5)
    scheduler_min_lr = config.get('scheduler_min_lr', 1e-6)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr
    )

    # Load state dicts if resuming
    if checkpoint is not None:
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']} with best loss so far ({best_set}): {best_loss:.6f}")

    # Training loop
    epochs = config['epochs']
    save_frequency = config.get('save_frequency', 10)

    for epoch in range(start_epoch, epochs + 1):
        # Print current learning rate at the start of the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: learning rate = {current_lr:.6f}")
        
        # Transition decoder dynamically to QAT at the start epoch
        if qat_enabled and epoch >= qat_start_epoch and not model_is_qat:
            import torch.ao.quantization as quantization
            model.decoder.qat = True
            model.decoder.qconfig = quantization.get_default_qat_qconfig('fbgemm')
            quantization.prepare_qat(model.decoder, inplace=True)
            model_is_qat = True
            
            # Re-initialize optimizer and scheduler with new QAT model parameters, keeping current learning rate
            optimizer = optim.Adam(model.parameters(), lr=current_lr)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_factor,
                patience=scheduler_patience,
                min_lr=scheduler_min_lr
            )
            print(f"Epoch {epoch}: Transitioned decoder to QAT and re-initialized optimizer/scheduler.")
            
        if qat_enabled and model_is_qat:
            print(f"Epoch {epoch}: QAT fake quantization is ACTIVE.")
        elif qat_enabled:
            print(f"Epoch {epoch}: Running in pure FP32 calibration/pretraining phase.")
        
        # 1. Train epoch
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
        
        train_loss /= len(train_dataset)

        # 2. Val epoch
        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    outputs = model(batch)
                    loss = criterion(outputs, batch)
                    val_loss += loss.item() * batch.size(0)
            val_loss /= len(val_dataset)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")

        # 3. Save results in results.h5
        with h5py.File(results_path, 'a') as f:
            if 'train_loss' not in f:
                f.create_dataset('train_loss', shape=(0,), maxshape=(None,), dtype=np.float32)
            if 'val_loss' not in f:
                f.create_dataset('val_loss', shape=(0,), maxshape=(None,), dtype=np.float32)
            
            # Ensure datasets are sized correctly for the current epoch (helps with resuming/overwriting)
            if f['train_loss'].shape[0] < epoch:
                f['train_loss'].resize(epoch, axis=0)
            if f['val_loss'].shape[0] < epoch:
                f['val_loss'].resize(epoch, axis=0)
                
            f['train_loss'][epoch - 1] = train_loss
            f['val_loss'][epoch - 1] = val_loss

        # Step scheduler based on best_set loss
        monitor_loss = train_loss if best_set == 'train' else val_loss
        scheduler.step(monitor_loss)

        # 4. Save best model if tracked loss improved
        best_set = config.get('best_set', 'train')
        current_metric = train_loss if best_set == 'train' else val_loss
        if current_metric < best_loss:
            best_loss = current_metric
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, best_path)
            print(f"Saved new best model checkpoint to {best_path} (Best Set: {best_set}, Loss: {best_loss:.6f})")

        # 5. Save latest model based on save frequency or if it's the last epoch
        if epoch % save_frequency == 0 or epoch == epochs:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, latest_path)
            print(f"Saved latest checkpoint to {latest_path} at epoch {epoch}")

    print("Training complete.")

if __name__ == '__main__':
    main()
