import argparse
import json
import math
import pathlib
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict

# NVIDIA DALI imports
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

def get_video_info(mkv_path: pathlib.Path) -> Tuple[int, int, int]:
    """Get original width, height, and total frame count of video using ffprobe."""
    cmd_dim = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(mkv_path)
    ]
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
        return 1164, 874, 1197

def get_multiple_of_64_less_than(val: int) -> int:
    return (val // 64) * 64

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
    output_dir: pathlib.Path,
    quality: int,
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

            # Save each frame in the batch as a WebP image
            for i in range(num_in_batch):
                img_idx = start_idx + frames_written + i
                img_path = output_dir / f"{img_idx:06d}.webp"
                
                # Convert shape [C, H, W] -> [H, W, C]
                img_hwc = frames_cpu[i].transpose(1, 2, 0)
                
                # Save as WEBP image
                img = Image.fromarray(img_hwc)
                img.save(img_path, "WEBP", quality=quality)
                
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
    parser = argparse.ArgumentParser(description="Generate a frame dataset as WebP images using GPU DALI.")
    parser.add_argument("--data-dir", type=str, default="data/comma2k19", help="Path to the processed dataset root.")
    parser.add_argument("--output", type=str, default="data/comma2k19_frames", help="Path to the output frames directory.")
    parser.add_argument("--interval", type=int, default=20, help="Interval between extracted frames.")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of parallel extraction workers.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use for decoding.")
    parser.add_argument("--quality", type=int, default=90, help="WebP compression quality (0-100).")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_dir)
    output_path = pathlib.Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    target_w = get_multiple_of_64_less_than(orig_w)
    target_h = get_multiple_of_64_less_than(orig_h)
    print(f"Original Resolution: {orig_w}x{orig_h}")
    print(f"Target Resolution (closest multiple of 64 less than actual): {target_w}x{target_h}")

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

    # Create metadata dictionary to save as JSON
    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "new_width": target_w,
        "new_height": target_h,
        "interval": args.interval,
        "total_frames": total_extracted_frames,
        "videos": [
            {
                "rel_path": info["rel_path"],
                "start_idx": info["start_idx"],
                "num_frames": info["num_frames"]
            }
            for info in video_write_info
        ]
    }

    # Save metadata JSON file
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

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
                    output_path,
                    args.quality,
                    pbar,
                    pbar_lock
                )
                for info in video_write_info
            ]
            
            for future in as_completed(futures):
                future.result()

    print(f"\nSuccess! Extracted {total_extracted_frames} frames as WebP to {output_path}")

if __name__ == "__main__":
    main()
