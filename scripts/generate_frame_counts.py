import argparse
import json
import pathlib
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

from tqdm import tqdm

def get_frame_count(mkv_path: pathlib.Path) -> Tuple[str, int]:
    """Parallelizable task to count frames in an MKV file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", 
        "-count_packets", "-show_entries", "stream=nb_read_packets", 
        "-of", "csv=p=0", str(mkv_path)
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip()
        return (str(mkv_path), int(out))
    except Exception as e:
        # Fallback for broken files
        return (str(mkv_path), 0)

def main():
    parser = argparse.ArgumentParser(description="Generate frame_counts.json for an existing dataset.")
    parser.add_argument("--data-dir", type=str, default="data/comma2k19", help="Path to the processed dataset root.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()

    processed_root = pathlib.Path(args.data_dir)
    if not processed_root.exists():
        print(f"Error: Directory {processed_root} does not exist.")
        return

    print(f"Scanning for MKV files in {processed_root}...")
    mkv_files = sorted(list(processed_root.glob("**/*.mkv")))
    
    if not mkv_files:
        print("No MKV files found.")
        return

    print(f"Found {len(mkv_files)} files. Starting frame count with {args.workers} workers...")
    
    frame_counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # We process absolute paths but will store relative ones
        results = list(tqdm(executor.map(get_frame_count, mkv_files), total=len(mkv_files), desc="Counting Frames"))
        
        for abs_path_str, count in results:
            # Convert to relative path for portability
            abs_path = pathlib.Path(abs_path_str)
            try:
                rel_path = abs_path.relative_to(processed_root)
                frame_counts[str(rel_path)] = count
            except ValueError:
                # If path isn't relative to root (shouldn't happen with glob), use name
                frame_counts[abs_path.name] = count

    output_path = processed_root / "frame_counts.json"
    with open(output_path, 'w') as f:
        json.dump(frame_counts, f, indent=2)

    print(f"\nSuccess! Saved frame counts for {len(frame_counts)} files to {output_path}")

if __name__ == "__main__":
    main()
