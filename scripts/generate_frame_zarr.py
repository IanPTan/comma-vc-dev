import argparse
import json
import math
import pathlib
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import zarr
from numcodecs import Blosc
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict

# NVIDIA DALI imports
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

def get_video_info(mkv_path: pathlib.Path) -> Tuple[int, int, int]:
    """Get original width, height, and total frame count of video using ffprobe."""
    # 1. Get width and height
    cmd_dim = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(mkv_path)
    ]
    # 2. Get frame count
    cmd_count = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", str(mkv_path)
    ]
    
    try:
        out_dim = subprocess.check_output(cmd_dim).decode().strip()
        w, h = map(int, out_dim.split(","))
        
        out_count = subprocess.check_output(cmd_count).decode().strip()
        count = int(out_count) if out_count else 0
        return w, h, count
    except Exception as e:
        # Fallback to standard
        return 1164, 874, 1197

def round_to_multiple_of_64(val: int) -> int:
    return int(round(val / 64.0) * 64)

@pipeline_def
def video_extraction_pipe(mkv_path: str, interval: int, target_w: int, target_h: int):
    # DALI's video reader
    video = fn.experimental.readers.video(
        device="gpu",
        filenames=[mkv_path],
        sequence_length=1,
        step=interval,
        stride=1,
        random_shuffle=False,
        name="reader"
    )
    
    # Resize directly on the GPU
    video = fn.resize(video, resize_x=target_w, resize_y=target_h)
    
    # Transpose layout from (F, H, W, C) to (C, F, H, W)
    return fn.transpose(video, perm=[3, 0, 1, 2])

def process_single_video(
    info: dict,
    interval: int,
    gpu_id: int,
    target_w: int,
    target_h: int,
    z_arr: zarr.Array,
    pbar: tqdm,
    pbar_lock: threading.Lock
):
    mkv_path = info['path']
    start_idx = info['start_idx']
    expected_frames = info['num_frames']
    
    if expected_frames == 0:
        return

    # Capped at 64 for parallel GPU decoding efficiency
    batch_size = min(64, expected_frames)

    try:
        # Build DALI pipeline
        pipe = video_extraction_pipe(
            batch_size=batch_size,
            num_threads=2,
            device_id=gpu_id,
            mkv_path=str(mkv_path),
            interval=interval,
            target_w=target_w,
            target_h=target_h
        )
        pipe.build()

        iterator = DALIGenericIterator(
            [pipe],
            output_map=["video"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=False
        )

        frames_written = 0
        for data in iterator:
            vid = data[0]["video"]  # shape [B, C, F, H, W] where F=1
            # Squeeze the sequence length (F=1) dimension
            frames_gpu = vid.squeeze(2)
            
            # Copy directly to CPU numpy array
            frames_cpu = frames_gpu.cpu().numpy()
            
            # Cap the batch size to avoid writing DALI-padded frames
            num_in_batch = min(frames_cpu.shape[0], expected_frames - frames_written)
            if num_in_batch <= 0:
                break

            # Write directly to Zarr slice
            w_start = start_idx + frames_written
            w_end = w_start + num_in_batch
            z_arr[w_start:w_end] = frames_cpu[:num_in_batch]
            
            frames_written += num_in_batch
            with pbar_lock:
                pbar.update(num_in_batch)

            if frames_written >= expected_frames:
                break

        # Clean up current pipeline & iterator
        del iterator
        del pipe
    except Exception as e:
        print(f"\nError processing {mkv_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate a frame dataset in Zarr format using GPU DALI.")
    parser.add_argument("--data-dir", type=str, default="data/comma2k19", help="Path to the processed dataset root.")
    parser.add_argument("--output", type=str, default="data/comma2k19_frames.zarr", help="Path to the output Zarr directory.")
    parser.add_argument("--interval", type=int, default=20, help="Interval between extracted frames.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel extraction workers.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use for decoding.")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_dir)
    output_path = pathlib.Path(args.output)
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. NVIDIA DALI requires a GPU for video decoding.")
    
    torch.cuda.set_device(args.gpu_id)

    # Use frame_counts.json for discovery if it exists, matching DaliDataLoader behavior
    metadata_path = data_root / "frame_counts.json"
    frame_counts = {}
    if metadata_path.exists():
        print(f"Loading file list from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            counts = json.load(f)
        # Filter files that actually exist and have frames
        mkv_files = [data_root / p for p, c in counts.items() if c > 0]
        mkv_files = sorted([p for p in mkv_files if p.exists()])
        
        # Save relative paths and counts
        for p in mkv_files:
            rel = str(p.relative_to(data_root))
            frame_counts[rel] = counts[rel]
    else:
        print(f"No frame_counts.json found in {data_root}. Falling back to glob...")
        mkv_files = sorted(list(data_root.glob("**/*.mkv")))
        
        # We need to probe files to get their frame counts
        print("Probing video files to get frame counts...")
        for p in tqdm(mkv_files, desc="Probing videos"):
            _, _, count = get_video_info(p)
            rel = str(p.relative_to(data_root))
            frame_counts[rel] = count

    if not mkv_files:
        print(f"No MKV files found in {data_root}. Check your data path.")
        return

    # Detect resolution from first file
    print(f"Detecting target resolution from first file...")
    orig_w, orig_h, _ = get_video_info(mkv_files[0])
    target_w = round_to_multiple_of_64(orig_w)
    target_h = round_to_multiple_of_64(orig_h)
    print(f"Original Resolution: {orig_w}x{orig_h}")
    print(f"Target Resolution (nearest multiple of 64): {target_w}x{target_h}")

    # Calculate exact frames to extract per file and cumulative offsets
    video_write_info = []
    total_extracted_frames = 0
    for p in mkv_files:
        rel = str(p.relative_to(data_root))
        fc = frame_counts[rel]
        expected_frames = math.ceil(fc / args.interval)
        video_write_info.append({
            'path': p,
            'rel_path': rel,
            'start_idx': total_extracted_frames,
            'num_frames': expected_frames
        })
        total_extracted_frames += expected_frames

    print(f"Total files: {len(mkv_files)}")
    print(f"Total frames to extract: {total_extracted_frames}")
    print(f"Initializing Zarr dataset at {output_path}...")

    # Fast Blosc LZ4 compressor
    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    
    # Initialize Zarr array in Zarr v2 format for maximum compatibility and compressor support
    z_arr = zarr.open(
        store=str(output_path),
        mode='w',
        shape=(total_extracted_frames, 3, target_h, target_w),
        chunks=(1, 3, target_h, target_w),
        dtype='uint8',
        compressor=compressor,
        zarr_format=2
    )

    # Save original and new dimensions as attributes on the dataset
    z_arr.attrs['original_width'] = orig_w
    z_arr.attrs['original_height'] = orig_h
    z_arr.attrs['new_width'] = target_w
    z_arr.attrs['new_height'] = target_h
    z_arr.attrs['interval'] = args.interval

    print(f"Starting DALI-based frame extraction with {args.workers} threads...")
    
    pbar_lock = threading.Lock()
    with tqdm(total=total_extracted_frames, desc="Extracting frames") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_single_video,
                    info,
                    args.interval,
                    args.gpu_id,
                    target_w,
                    target_h,
                    z_arr,
                    pbar,
                    pbar_lock
                )
                for info in video_write_info
            ]
            
            for future in as_completed(futures):
                # Propagate any thread exceptions if they occurred
                future.result()

    print(f"\nSuccess! Extracted {total_extracted_frames} frames to {output_path}")

if __name__ == "__main__":
    main()
