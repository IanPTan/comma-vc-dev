import torch
import numpy as np
import os
import tempfile
import pathlib
import json
import subprocess
import random
from typing import List, Tuple, Optional
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

class DaliClipper:
    """
    Low-level DALI pipeline wrapper for loading specific video clips.
    """
    def __init__(
        self,
        clip_list: List[Tuple[str, int, int]],  # (path, start_frame, end_frame)
        batch_size: int = 1,
        num_threads: int = 2,
        device_id: int = 0,
        shuffle: bool = False,
    ):
        self.clip_list = clip_list
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.shuffle = shuffle
        
        if not clip_list:
            raise ValueError("clip_list cannot be empty")

        # Create temporary file list for DALI
        self._file_list_tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        for path, start, end in self.clip_list:
            self._file_list_tmp.write(f"{os.path.abspath(path)} 0 {start} {end}\n")
        self._file_list_tmp.close()

        self.pipeline = self._build_pipeline()
        self.iterator = DALIGenericIterator(
            [self.pipeline],
            output_map=["video"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True
        )

    def _build_pipeline(self):
        first_start, first_end = self.clip_list[0][1], self.clip_list[0][2]
        seq_len = first_end - first_start

        @pipeline_def(batch_size=self.batch_size, num_threads=self.num_threads, device_id=self.device_id)
        def video_pipe():
            # When file_list rows have a label column, fn.readers.video returns
            # (video, labels). We only care about the video here.
            out = fn.readers.video(
                device="gpu",
                file_list=self._file_list_tmp.name,
                sequence_length=seq_len,
                shard_id=0,
                num_shards=1,
                random_shuffle=self.shuffle,
                initial_fill=1024 if self.shuffle else 1,
                image_type=types.RGB,
                dtype=types.UINT8,
                normalized=False,
                # New DALI default; silences the deprecation warning and
                # gives a more accurate per-file frame count.
                file_list_include_preceding_frame=True,
            )
            video = out[0] if isinstance(out, (tuple, list)) else out
            # (F, H, W, C) -> (C, F, H, W)
            video = fn.transpose(video, perm=[3, 0, 1, 2])
            return video
        
        pipe = video_pipe()
        pipe.build()
        return pipe

    def __iter__(self):
        for data in self.iterator:
            yield data[0]["video"]

    def __del__(self):
        if hasattr(self, '_file_list_tmp') and os.path.exists(self._file_list_tmp.name):
            os.unlink(self._file_list_tmp.name)

    def __len__(self):
        return (len(self.clip_list) + self.batch_size - 1) // self.batch_size

class DaliDataLoader:
    """
    High-level loader that manages reproducible splits and sliding-window indexing.
    """
    def __init__(
        self,
        dataset_path: str,
        split_path: str = "dataset_split.json",
        mode: str = "train",  # "train" or "val"
        val_split: float = 0.1,
        clip_frames: int = 200,
        stride: Optional[int] = None, # If None, stride = clip_frames (no overlap)
        batch_size: int = 64,
        num_threads: int = 4,
        device_id: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        # frame_counts.json sometimes overcounts what DALI's GPU video decoder
        # actually sees (keyframe-aware seek vs naive count). Subtract this
        # many frames from each file's reported length before windowing.
        # comma2k19 HEVC needs ~200 to be safe with the legacy DALI reader.
        end_safety_margin: int = 200,
    ):
        self.dataset_path = pathlib.Path(dataset_path)
        self.split_path = pathlib.Path(split_path)
        self.clip_frames = clip_frames
        self.stride = stride if stride is not None else clip_frames
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed
        self.shuffle = (mode == "train") and shuffle
        self.end_safety_margin = end_safety_margin

        # 1. Discover MKVs and their frame counts via JSON metadata
        frame_counts = self._load_metadata()
        mkv_rel_paths = sorted(list(frame_counts.keys()))

        # 2. Build the global window list OR load from split
        if self.split_path.exists():
            print(f"Loading split from {self.split_path}...")
            with open(self.split_path, 'r') as f:
                split_data = json.load(f)
            train_clips = split_data['train']
            val_clips = split_data['val']
        else:
            print(f"Generating new split at {self.split_path} "
                  f"(safety margin {end_safety_margin} frames)...")
            all_clips = []
            for rel_path in mkv_rel_paths:
                n_frames = frame_counts.get(rel_path, 0)
                usable = n_frames - end_safety_margin
                if usable < self.clip_frames:
                    continue
                abs_path = str(self.dataset_path / rel_path)
                # Window starts: last start must satisfy start + clip_frames <= usable.
                for start in range(0, usable - self.clip_frames + 1, self.stride):
                    all_clips.append((abs_path, start, start + self.clip_frames))
            
            # Shuffle and split
            random.seed(self.seed)
            random.shuffle(all_clips)
            split_idx = int(len(all_clips) * (1.0 - val_split))
            train_clips = all_clips[:split_idx]
            val_clips = all_clips[split_idx:]
            
            # Save for later
            with open(self.split_path, 'w') as f:
                json.dump({'train': train_clips, 'val': val_clips}, f)

        # 3. Select the subset based on mode
        selected_clips = train_clips if mode == "train" else val_clips
        print(f"Initialized DaliDataLoader ({mode}): {len(selected_clips)} clips found.")

        self.clipper = DaliClipper(
            clip_list=selected_clips,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            shuffle=self.shuffle
        )

    def _load_metadata(self) -> dict:
        metadata_path = self.dataset_path / "frame_counts.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"frame_counts.json not found in {self.dataset_path}. "
                "Please run the download/processing script to generate it."
            )
            
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def __iter__(self):
        return iter(self.clipper)

    def __len__(self):
        return len(self.clipper)
