import os
import json
import pathlib
import random
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class FrameDataset(Dataset):
    """
    PyTorch dataset for loading individual frames saved as WebP images.
    Splits are done at the video level (using metadata.json) to prevent temporal data leakage.
    """
    def __init__(
        self,
        dataset_dir: str,
        split_path: str = "dataset_split.json",
        mode: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
        transform: Optional[T.Compose] = None,
    ):
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.split_path = pathlib.Path(split_path)
        self.mode = mode
        self.val_split = val_split
        self.seed = seed
        
        # Load metadata.json
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.dataset_dir}. "
                "Please run generate_frame_dataset.py first."
            )
            
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.width = self.metadata["new_width"]
        self.height = self.metadata["new_height"]
        
        # Video list
        videos = self.metadata["videos"]
        
        # Consistent file-level train/val split
        if self.split_path.exists():
            print(f"Loading split from {self.split_path}...")
            with open(self.split_path, 'r') as f:
                split_data = json.load(f)
            train_vids = split_data['train']
            val_vids = split_data['val']
        else:
            print(f"Generating new video-level split at {self.split_path}...")
            # Shuffle video paths deterministically
            shuffled_vids = [v["rel_path"] for v in videos]
            random.seed(self.seed)
            random.shuffle(shuffled_vids)
            
            split_idx = int(len(shuffled_vids) * (1.0 - self.val_split))
            train_vids = shuffled_vids[:split_idx]
            val_vids = shuffled_vids[split_idx:]
            
            # Save split to file
            with open(self.split_path, 'w') as f:
                json.dump({'train': train_vids, 'val': val_vids}, f, indent=2)
                
        # Filter videos according to active mode
        active_vid_paths = set(train_vids if mode == "train" else val_vids)
        self.active_videos = [v for v in videos if v["rel_path"] in active_vid_paths]
        
        # Gather all frame indices for the active videos
        self.frame_indices = []
        for v in self.active_videos:
            start = v["start_idx"]
            num = v["num_frames"]
            self.frame_indices.extend(range(start, start + num))
            
        print(f"Initialized FrameDataset ({mode}): {len(self.active_videos)} videos, {len(self.frame_indices)} total frames.")
        
        # Default transform if none provided
        if transform is not None:
            self.transform = transform
        else:
            # Standard MAE training transforms: convert to tensor, ImageNet normalization.
            transforms_list = [T.ToTensor()]
            if mode == "train":
                transforms_list.append(T.RandomHorizontalFlip(p=0.5))
            transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            self.transform = T.Compose(transforms_list)
            
    def __len__(self) -> int:
        return len(self.frame_indices)
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        frame_idx = self.frame_indices[idx]
        img_path = self.dataset_dir / f"{frame_idx:06d}.webp"
        
        # Load image (ensuring RGB)
        img = Image.open(img_path).convert("RGB")
        
        # Apply transform
        x = self.transform(img)
        return x
