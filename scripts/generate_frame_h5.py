import argparse
import json
import pathlib
import subprocess
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

def get_video_dimensions(mkv_path: pathlib.Path) -> Tuple[int, int]:
    """Get original width and height of video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(mkv_path)
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip()
        w, h = map(int, out.split(","))
        return w, h
    except Exception as e:
        return 1164, 874

def round_to_multiple_of_64(val: int) -> int:
    return int(round(val / 64.0) * 64)

def extract_frames(mkv_path: pathlib.Path, width: int, height: int, interval: int) -> List[np.ndarray]:
    """
    Extract frames from an MKV file at regular intervals using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(mkv_path),
        "-vf", f"select='not(mod(n,{interval}))',scale={width}:{height}",
        "-fps_mode", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]
    
    frame_bytes = width * height * 3
    frames = []
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        
        while True:
            raw_frame = process.stdout.read(frame_bytes)
            if len(raw_frame) < frame_bytes:
                break
            
            # (H, W, C) -> (C, H, W) for torch-ready H5
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
            
        process.stdout.close()
        process.wait()
    except Exception as e:
        print(f"Error processing {mkv_path.name}: {e}")
        
    return frames

def main():
    parser = argparse.ArgumentParser(description="Generate a frame dataset in HDF5 format.")
    parser.add_argument("--data-dir", type=str, default="data/comma2k19", help="Path to the processed dataset root.")
    parser.add_argument("--output", type=str, default="data/frames.h5", help="Path to the output H5 file.")
    parser.add_argument("--interval", type=int, default=20, help="Interval between extracted frames.")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Number of parallel FFmpeg processes.")
    parser.add_argument("--compression", type=str, default="lzf", choices=["lzf", "gzip", None], help="H5 compression type.")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_dir)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use frame_counts.json for discovery if it exists, matching DaliDataLoader behavior
    metadata_path = data_root / "frame_counts.json"
    if metadata_path.exists():
        print(f"Loading file list from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            counts = json.load(f)
        # Filter files that actually exist and have frames
        mkv_files = [data_root / p for p, c in counts.items() if c > 0]
        mkv_files = sorted([p for p in mkv_files if p.exists()])
    else:
        print(f"No frame_counts.json found in {data_root}. Falling back to glob...")
        mkv_files = sorted(list(data_root.glob("**/*.mkv")))
    
    if not mkv_files:
        print(f"No MKV files found in {data_root}. Check your data path.")
        return

    print(f"Found {len(mkv_files)} files. Detecting target resolution from first file...")
    orig_w, orig_h = get_video_dimensions(mkv_files[0])
    target_w = round_to_multiple_of_64(orig_w)
    target_h = round_to_multiple_of_64(orig_h)
    print(f"Original Resolution: {orig_w}x{orig_h}")
    print(f"Target Resolution (nearest multiple of 64): {target_w}x{target_h}")

    print(f"Starting extraction into {output_path}...")

    with h5py.File(output_path, 'w') as f:
        ds = f.create_dataset(
            "frames", 
            shape=(0, 3, target_h, target_w),
            maxshape=(None, 3, target_h, target_w),
            dtype='uint8',
            chunks=(1, 3, target_h, target_w), # Optimized for per-frame random access
            compression=args.compression
        )
        
        # Save original and new dimensions as attributes on the dataset
        ds.attrs['original_width'] = orig_w
        ds.attrs['original_height'] = orig_h
        ds.attrs['new_width'] = target_w
        ds.attrs['new_height'] = target_h
        ds.attrs['interval'] = args.interval

        total_frames = 0
        
        # Use with to guarantee progress bar is closed cleanly
        with tqdm(total=len(mkv_files), desc="Extracting frames") as pbar:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Map absolute paths to extract_frames tasks
                futures = {
                    executor.submit(extract_frames, f_path, target_w, target_h, args.interval): f_path 
                    for f_path in mkv_files
                }
                
                for future in as_completed(futures):
                    extracted_frames = future.result()
                    
                    if extracted_frames:
                        num_new = len(extracted_frames)
                        ds.resize((total_frames + num_new, 3, target_h, target_w))
                        ds[total_frames:] = np.stack(extracted_frames)
                        total_frames += num_new
                    
                    pbar.update(1)
                    pbar.set_postfix({"total_frames": total_frames})

    print(f"\nSuccess! Extracted {total_frames} frames to {output_path}")


if __name__ == "__main__":
    main()
