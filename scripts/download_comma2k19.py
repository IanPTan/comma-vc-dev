import argparse
import json
import os
import pathlib
import shutil
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

from huggingface_hub import snapshot_download
from tqdm import tqdm

def remux_file(task: Tuple[pathlib.Path, pathlib.Path, bool]) -> bool:
    """
    Parallelizable task to remux a single HEVC file to MKV and optionally delete raw.
    task: (hevc_path, out_mkv_path, destructive)
    """
    hevc_path, out_mkv_path, destructive = task
    
    # Skip if already exists
    if out_mkv_path.exists():
        if destructive:
            # If processed version exists, we don't need the raw segment at all
            shutil.rmtree(hevc_path.parent, ignore_errors=True)
        return True

    out_mkv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command: force 20fps for raw HEVC bitstream
    cmd = [
        "ffmpeg", "-nostdin", "-y", 
        "-f", "hevc", "-r", "20", 
        "-i", str(hevc_path), 
        "-vcodec", "copy", 
        str(out_mkv_path), 
        "-loglevel", "error"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            if destructive:
                # Delete the entire segment folder (containing video.hevc, logs, etc)
                shutil.rmtree(hevc_path.parent, ignore_errors=True)
            return True
    except Exception as e:
        print(f"Error remuxing {hevc_path}: {e}")
    
    return False

def get_frame_count(mkv_path: pathlib.Path) -> Tuple[str, int]:
    """Parallelizable task to count frames in an MKV file."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", 
        "-count_packets", "-show_entries", "stream=nb_read_packets", 
        "-of", "csv=p=0", str(mkv_path)
    ]
    try:
        out = subprocess.check_output(cmd).decode().strip()
        return (str(mkv_path), int(out))
    except:
        return (str(mkv_path), 0)

def main():
    parser = argparse.ArgumentParser(description="Download and process comma2k19 dataset.")
    parser.add_argument("--destructive", "-d", action="store_true", help="Delete raw data after processing to save space.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for remuxing.")
    args = parser.parse_args()

    # 1. Setup Paths
    base_dir = pathlib.Path("data")
    raw_root = base_dir / "comma2k19_raw"
    processed_root = base_dir / "comma2k19"
    
    # 2. Download from HuggingFace
    print("--- Phase 1: Downloading Dataset ---")
    snapshot_download(
        repo_id="commaai/comma2k19",
        repo_type="dataset",
        allow_patterns="raw_data/Chunk_*.zip",
        local_dir=raw_root
    )

    # 3. Extraction (Sequential to save space)
    print("\n--- Phase 2: Sequential Extraction ---")
    raw_data_dir = raw_root / "raw_data"
    zip_files = sorted(list(raw_data_dir.glob("Chunk_*.zip")))
    
    for zip_path in zip_files:
        filename = zip_path.stem # e.g. Chunk_1
        target_dir = raw_data_dir / f"Dataset_{filename}"
        
        if target_dir.exists():
            print(f"Skipping extraction for {filename} (already exists)")
            os.remove(zip_path) # Always delete zip if target exists
            continue
            
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
            
        # Move "Chunk_1" to "Dataset_Chunk_1"
        extracted_folder = raw_data_dir / filename
        if extracted_folder.exists():
            extracted_folder.rename(target_dir)
        elif (raw_data_dir / filename.lower()).exists(): # handle lowercase
            (raw_data_dir / filename.lower()).rename(target_dir)
            
        # Delete zip immediately
        os.remove(zip_path)

    # 4. Remuxing (Parallel with Delete-as-you-go)
    print("\n--- Phase 3: Parallel Remuxing (HEVC -> MKV) ---")
    print("    Note: Raw segments are deleted immediately after successful remux if --destructive is set.")
    remux_tasks = []
    # Path pattern: raw_data_dir/Dataset_Chunk_n/dongle|timestamp/segment/video.hevc
    for hevc_path in raw_data_dir.glob("Dataset_Chunk_*/*/*/video.hevc"):
        # Parse path
        segment_dir = hevc_path.parent
        segment_num = segment_dir.name
        route_dir = segment_dir.parent
        route_full_name = route_dir.name
        chunk_dir = route_dir.parent
        chunk_name = chunk_dir.name
        
        # Extract timestamp from "dongle|timestamp"
        timestamp = route_full_name.split('|')[-1]
        
        # Target: processed_root/Dataset_Chunk_n/timestamp/segment.mkv
        out_mkv = processed_root / chunk_name / timestamp / f"{segment_num}.mkv"
        remux_tasks.append((hevc_path, out_mkv, args.destructive))

    if remux_tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(tqdm(executor.map(remux_file, remux_tasks), total=len(remux_tasks), desc="Remuxing"))
    else:
        print("No HEVC files found for remuxing.")

    # 5. Final Parent Folder Cleanup
    if args.destructive and raw_root.exists():
        print("\n--- Phase 4: Cleaning up empty raw parent folders ---")
        shutil.rmtree(raw_root, ignore_errors=True)

    # 6. Metadata Generation (Parallel)
    print("\n--- Phase 5: Metadata Generation ---")
    mkv_files = sorted(list(processed_root.glob("**/*.mkv")))
    frame_counts = {}
    
    if mkv_files:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # We still need absolute paths for ffprobe, but we'll store relative ones
            results = list(tqdm(executor.map(get_frame_count, mkv_files), total=len(mkv_files), desc="Counting Frames"))
            for abs_path_str, count in results:
                # Convert to relative path for portability
                rel_path = pathlib.Path(abs_path_str).relative_to(processed_root)
                frame_counts[str(rel_path)] = count
                
        with open(processed_root / "frame_counts.json", 'w') as f:
            json.dump(frame_counts, f, indent=2)
        print(f"Saved relative frame counts for {len(frame_counts)} files.")

    print("\nProcessing Complete!")

if __name__ == "__main__":
    main()
