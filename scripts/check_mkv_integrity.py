import subprocess
import pathlib
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def check_single_mkv(path: pathlib.Path) -> tuple[str, bool, str]:
    """
    Checks the integrity of a single MKV file using ffmpeg.
    Returns (path, is_valid, error_message)
    """
    # -v error: only show errors
    # -i: input file
    # -f null -: decode and discard output (tests the whole file)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path), 
        "-f", "null", "-", "-threads", "1"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and not result.stderr:
            return (str(path), True, "")
        else:
            return (str(path), False, result.stderr.strip())
    except subprocess.TimeoutExpired:
        return (str(path), False, "Timeout during integrity check")
    except Exception as e:
        return (str(path), False, str(e))

def main():
    data_dir = pathlib.Path("data/comma2k19")
    mkv_files = list(data_dir.glob("**/*.mkv"))
    
    if not mkv_files:
        print(f"No MKV files found in {data_dir}")
        return

    print(f"Checking integrity of {len(mkv_files)} MKV files using ffmpeg...")
    
    corrupted = []
    # Use multiple processes to speed up decoding
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(check_single_mkv, mkv_files), total=len(mkv_files)))

    for path, is_valid, err in results:
        if not is_valid:
            corrupted.append((path, err))

    print("\n" + "="*50)
    if not corrupted:
        print("SUCCESS: All MKV files passed integrity checks!")
    else:
        print(f"FAILED: Found {len(corrupted)} corrupted files:")
        for path, err in corrupted:
            print(f"  - {path}")
            print(f"    Error: {err[:200]}...") # Truncate long errors
    print("="*50)

if __name__ == "__main__":
    main()
