import subprocess
import pathlib
import numpy as np
import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def check_single_mkv(path: pathlib.Path) -> tuple[str, bool, str, int]:
    """
    Checks the integrity of a single MKV file using ffmpeg and returns frame count via ffprobe.
    Returns (path, is_valid, error_message, frame_count)
    """
    # 1. Get Frame Count via ffprobe (most reliable, same as generate_frame_counts.py)
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", 
        "-count_packets", "-show_entries", "stream=nb_read_packets", 
        "-of", "csv=p=0", str(path)
    ]
    count = 0
    try:
        out = subprocess.check_output(probe_cmd, timeout=30).decode().strip()
        count = int(out) if out else 0
    except:
        pass

    # 2. Check Integrity via ffmpeg
    check_cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), 
        "-f", "null", "-", "-threads", "1"
    ]
    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr.strip()
        
        # Filter false positive timestamp warnings
        actual_errors = []
        if stderr:
            for line in stderr.split('\n'):
                if "monotonically increasing dts" in line: continue
                if any(k in line.lower() for k in ["error", "invalid", "corrupt", "missing"]):
                    if not line.startswith("video:") and not line.startswith("[out#0"):
                        actual_errors.append(line)
        
        is_valid = (result.returncode == 0 and len(actual_errors) == 0)
        return (str(path), is_valid, "\n".join(actual_errors), count)
        
    except subprocess.TimeoutExpired:
        return (str(path), False, "Timeout during integrity check", count)
    except Exception as e:
        return (str(path), False, str(e), count)

def main():
    parser = argparse.ArgumentParser(description="Check MKV integrity and collect frame count stats.")
    parser.add_argument("--data-dir", type=str, default="data/comma2k19", help="Directory containing MKV files.")
    parser.add_argument("-p", "--proportion", type=float, default=1.0, help="Proportion of files to sample (0.0 to 1.0).")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    mkv_files = list(data_dir.glob("**/*.mkv"))
    
    if not mkv_files:
        print(f"No MKV files found in {data_dir}")
        return

    # Randomly sample based on proportion
    if 0.0 < args.proportion < 1.0:
        num_to_sample = max(1, int(len(mkv_files) * args.proportion))
        print(f"Randomly sampling {num_to_sample} files ({args.proportion*100:.1f}%) from {len(mkv_files)} total files...")
        mkv_files = random.sample(mkv_files, num_to_sample)
    elif args.proportion >= 1.0:
        print(f"Checking all {len(mkv_files)} MKV files...")
    else:
        print("Error: Proportion must be greater than 0.0")
        return

    corrupted = []
    frame_counts = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(check_single_mkv, mkv_files), total=len(mkv_files)))

    for path, is_valid, err, count in results:
        if count > 0:
            frame_counts.append(count)
        if not is_valid:
            corrupted.append((path, err))

    print("\n" + "="*50)
    if not corrupted:
        print("SUCCESS: All checked MKV files passed integrity checks!")
    else:
        print(f"FAILED: Found {len(corrupted)} corrupted files in the sample.")
        # Only print first 10 corrupted to avoid spam
        for path, err in corrupted[:10]:
            print(f"  - {path}")
            print(f"    Error: {err[:150]}...")
    
    if frame_counts:
        counts = np.array(frame_counts)
        print("\nFrame Count Statistics (from sample):")
        print(f"  Files Sampled: {len(counts)}")
        print(f"  Min:           {np.min(counts)}")
        print(f"  Max:           {np.max(counts)}")
        print(f"  Median:        {np.median(counts)}")
        print(f"  Mean:          {np.mean(counts):.1f}")
        print(f"  25th Pctl:     {np.percentile(counts, 25)}")
        print(f"  75th Pctl:     {np.percentile(counts, 75)}")
        
        if np.all(counts == counts[0]):
            print(f"\nUniform Length: All sampled files are exactly {counts[0]} frames.")
        else:
            print(f"\nVariable Lengths: Sampled files range from {np.min(counts)} to {np.max(counts)} frames.")
    print("="*50)

if __name__ == "__main__":
    main()
