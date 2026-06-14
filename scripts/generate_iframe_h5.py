import argparse
import json
import pathlib
import subprocess
import numpy as np
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

def extract_iframes(mkv_path: pathlib.Path, size: int = 256) -> List[np.ndarray]:
    """
    Extract all I-frames from an MKV file using ffmpeg.
    -skip_frame nokey avoids decoding non-keyframes.
    """
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-skip_frame", "nokey",
        "-i", str(mkv_path),
        "-vf", f"scale={size}:{size}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]
    
    frame_bytes = size * size * 3
    frames = []
    
    try:
        # bufsize 10^8 is ~100MB, plenty for a few I-frames
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        
        while True:
            raw_frame = process.stdout.read(frame_bytes)
            if len(raw_frame) < frame_bytes:
                break
            
            # (H, W, C) -> (C, H, W) for torch-ready H5
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((size, size, 3))
            frame = frame.transpose(2, 0, 1)
            frames.append(frame)
            
        process.stdout.close()
        process.wait()
    except Exception as e:
        print(f"Error processing {mkv_path.name}: {e}")
        
    return frames

def main():
    parser = argparse.ArgumentParser(description="Generate an I-frame dataset in HDF5 format.")
    # Defaults aligned with repo structure
    parser.add_argument("--data-dir", type=str, default="data/comma2k19", help="Path to the processed dataset root.")
    parser.add_argument("--output", type=str, default="data/iframes.h5", help="Path to the output H5 file.")
    parser.add_argument("--size", type=int, default=256, help="Target H/W for frames.")
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

    print(f"Found {len(mkv_files)} files. Starting extraction into {output_path}...")

    with h5py.File(output_path, 'w') as f:
        ds = f.create_dataset(
            "frames", 
            shape=(0, 3, args.size, args.size),
            maxshape=(None, 3, args.size, args.size),
            dtype='uint8',
            chunks=(1, 3, args.size, args.size), # Optimized for per-frame random access
            compression=args.compression
        )
        
        total_frames = 0
        pbar = tqdm(total=len(mkv_files), desc="Extracting I-frames")
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Map absolute paths to extract_iframes tasks
            futures = {executor.submit(extract_iframes, f_path, args.size): f_path for f_path in mkv_files}
            
            for future in as_completed(futures):
                extracted_frames = future.result()
                
                if extracted_frames:
                    num_new = len(extracted_frames)
                    ds.resize((total_frames + num_new, 3, args.size, args.size))
                    ds[total_frames:] = np.stack(extracted_frames)
                    total_frames += num_new
                
                pbar.update(1)
                pbar.set_postfix({"total_frames": total_frames})

    print(f"\nSuccess! Extracted {total_frames} frames to {output_path}")

if __name__ == "__main__":
    main()
