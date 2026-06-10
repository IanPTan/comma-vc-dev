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
    Low-level DALI pipeline wrapper. Uses `fn.experimental.readers.video`,
    which is frame-accurate (unlike the legacy reader's index-based count
    that miscounts HEVC files in comma2k19).
    """
    def __init__(
        self,
        file_paths: List[str],          # one entry per .mkv to draw clips from
        clip_frames: int,               # frames per sequence
        batch_size: int = 1,
        num_threads: int = 2,
        device: str = "gpu",
        device_id: int = 0,
        shuffle: bool = False,
        step: Optional[int] = None,     # stride between consecutive clip starts
                                        # in a file; defaults to clip_frames
                                        # (= non-overlapping windows).
    ):
        if not file_paths:
            raise ValueError("file_paths cannot be empty")
        self.file_paths = [os.path.abspath(p) for p in file_paths]
        self.clip_frames = clip_frames
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device = device
        self.device_id = device_id
        self.shuffle = shuffle
        self.step = step if step is not None else clip_frames

        self.pipeline = self._build_pipeline()
        self.iterator = DALIGenericIterator(
            [self.pipeline],
            output_map=["video"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        # `epoch_size` is reported by the reader after build.
        try:
            self._epoch_clips = int(self.pipeline.epoch_size("reader"))
        except Exception:
            self._epoch_clips = None

    def _build_pipeline(self):
        @pipeline_def(batch_size=self.batch_size, num_threads=self.num_threads, device_id=self.device_id)
        def video_pipe():
            out = fn.experimental.readers.video(
                device=self.device,
                filenames=self.file_paths,
                sequence_length=self.clip_frames,
                step=self.step,
                stride=1,
                random_shuffle=self.shuffle,
                shard_id=0,
                num_shards=1,
                name="reader",
            )
            video = out[0] if isinstance(out, (tuple, list)) else out
            # (F, H, W, C) -> (C, F, H, W)
            return fn.transpose(video, perm=[3, 0, 1, 2])

        pipe = video_pipe()
        pipe.build()
        return pipe

    def __iter__(self):
        for data in self.iterator:
            yield data[0]["video"]

    def __len__(self):
        # Number of batches per epoch.
        if self._epoch_clips is None:
            return 0
        return (self._epoch_clips + self.batch_size - 1) // self.batch_size

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
        device: str = "gpu",
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
        self.device = device
        self.device_id = device_id
        self.seed = seed
        self.shuffle = (mode == "train") and shuffle
        self.end_safety_margin = end_safety_margin

        # 1. Discover MKVs and their frame counts via JSON metadata
        frame_counts = self._load_metadata()
        mkv_rel_paths = sorted(list(frame_counts.keys()))

        # 2. File-level split (the experimental DALI reader does clip sampling
        # internally — we only need to hand it the list of files).
        # Filter out any file too short to yield a single clip.
        min_frames = clip_frames + end_safety_margin
        usable_rel = [p for p in mkv_rel_paths if frame_counts.get(p, 0) >= min_frames]
        dropped = len(mkv_rel_paths) - len(usable_rel)
        if dropped:
            print(f"  Skipping {dropped} files shorter than {min_frames} frames.")

        if self.split_path.exists():
            print(f"Loading split from {self.split_path}...")
            with open(self.split_path, 'r') as f:
                split_data = json.load(f)
            train_files = split_data['train']
            val_files = split_data['val']
        else:
            print(f"Generating new file-level split at {self.split_path} "
                  f"({len(usable_rel)} usable files)...")
            shuffled = list(usable_rel)
            random.seed(self.seed)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * (1.0 - val_split))
            train_files = shuffled[:split_idx]
            val_files = shuffled[split_idx:]
            with open(self.split_path, 'w') as f:
                json.dump({'train': train_files, 'val': val_files}, f)

        selected_rel = train_files if mode == "train" else val_files
        selected_abs = [str(self.dataset_path / p) for p in selected_rel]
        print(f"Initialized DaliDataLoader ({mode}): {len(selected_abs)} files "
              f"(~{sum(frame_counts.get(p, 0) // clip_frames for p in selected_rel)} clips).")

        self.clipper = DaliClipper(
            file_paths=selected_abs,
            clip_frames=clip_frames,
            batch_size=batch_size,
            num_threads=num_threads,
            device=self.device,
            device_id=device_id,
            shuffle=self.shuffle,
            step=self.stride,
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
