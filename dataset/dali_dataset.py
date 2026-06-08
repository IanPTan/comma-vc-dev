import torch
import numpy as np
import os
import tempfile
import pathlib
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
    ):
        self.clip_list = clip_list
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        
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
            video = fn.readers.video(
                device="gpu",
                file_list=self._file_list_tmp.name,
                sequence_length=seq_len,
                shard_id=0,
                num_shards=1,
                random_shuffle=False,
                image_type=types.RGB,
                dtype=types.UINT8,
                normalized=False
            )
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
    High-level loader that discovers videos in the specified structure:
    root/Dataset_chunk_n/route_id/segment_num.mkv
    """
    def __init__(
        self,
        dataset_path: str,
        clip_frames: int = 200,  # e.g., 10 seconds at 20fps
        batch_size: int = 64,
        num_threads: int = 4,
        device_id: int = 0,
        limit_clips: Optional[int] = None,
        frames_per_segment: int = 1200, # Default for comma segments (60s @ 20fps)
    ):
        self.dataset_path = pathlib.Path(dataset_path)
        self.clip_frames = clip_frames
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        
        # Discover all .mkv files
        # Pattern: Dataset_chunk_*/route_id/segment.mkv
        self.mkv_paths = sorted(list(self.dataset_path.glob("Dataset_chunk_*/*/*.mkv")))
        
        if not self.mkv_paths:
            print(f"Warning: No .mkv files found in {dataset_path}")
            self.clipper = None
            return

        # Create clips (for now, let's take one random 10s clip per video)
        # In a real scenario, you might want to sample multiple windows per video
        clip_list = []
        for path in self.mkv_paths:
            if frames_per_segment > clip_frames:
                # Random start frame within the segment
                start = np.random.randint(0, frames_per_segment - clip_frames)
                end = start + clip_frames
                clip_list.append((str(path), start, end))
            else:
                # If segment is too short, just take what we can (DALI will pad)
                clip_list.append((str(path), 0, frames_per_segment))

        if limit_clips:
            clip_list = clip_list[:limit_clips]

        self.clipper = DaliClipper(
            clip_list=clip_list,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id
        )

    def __iter__(self):
        if self.clipper:
            return iter(self.clipper)
        return iter([])

    def __len__(self):
        return len(self.clipper) if self.clipper else 0
