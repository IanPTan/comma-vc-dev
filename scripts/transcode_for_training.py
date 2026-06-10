"""
Pre-transcode comma2k19 .mkv (HEVC, native 1164x874) into small H.264 .mp4
files at training resolution. Run once; training reads from the transcoded
copy and becomes ~10x faster per batch.

The DALI experimental video reader pays a per-clip seek + keyframe-decode
cost. For HEVC with native resolution that's seconds. For small H.264 it's
milliseconds. This script moves that cost off the training-time critical path.

Usage:
    python scripts/transcode_for_training.py \\
        --src /content/comma2k19_local \\
        --dst /content/comma2k19_256 \\
        --size 256 --workers 4
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


def detect_nvenc() -> bool:
    """Return True if ffmpeg has h264_nvenc available."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        ).stdout
        return "h264_nvenc" in out
    except Exception:
        return False


def transcode_one(args: Tuple[Path, Path, int, bool]) -> Tuple[str, Optional[int], Optional[str]]:
    """Transcode one file. Returns (rel_path_str, frame_count, error_msg)."""
    src, dst, size, use_nvenc = args
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 1024:
        # Already done; just probe the frame count.
        try:
            n = _count_frames(dst)
            return (str(dst), n, None)
        except Exception as e:
            return (str(dst), None, f"probe failed: {e}")

    # Build ffmpeg command.
    if use_nvenc:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", str(src),
            "-vf", f"scale_cuda={size}:{size}",
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
            "-an",  # drop audio
            str(dst),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", f"scale={size}:{size}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-an",
            str(dst),
        ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", "replace")[:300]
        # Drop partial output so the next run retries.
        if dst.exists():
            dst.unlink(missing_ok=True)
        return (str(dst), None, f"ffmpeg failed: {msg}")

    try:
        n = _count_frames(dst)
    except Exception as e:
        return (str(dst), None, f"probe failed: {e}")
    return (str(dst), n, None)


def _count_frames(path: Path) -> int:
    """Accurate frame count via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out)


def main():
    p = argparse.ArgumentParser(description="Pre-transcode comma2k19 to small H.264 for fast training.")
    p.add_argument("--src", type=str, required=True, help="Source root (with Dataset_Chunk_*/...).")
    p.add_argument("--dst", type=str, required=True, help="Destination root for transcoded mp4s.")
    p.add_argument("--size", type=int, default=256, help="Output frame edge (square).")
    p.add_argument("--workers", type=int, default=4, help="Parallel ffmpeg processes.")
    p.add_argument("--no-nvenc", action="store_true", help="Force software encode (libx264).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on files (for testing).")
    args = p.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()
    if not src_root.exists():
        print(f"source does not exist: {src_root}")
        sys.exit(1)

    use_nvenc = (not args.no_nvenc) and detect_nvenc()
    print(f"encoder: {'h264_nvenc (GPU)' if use_nvenc else 'libx264 (CPU)'}")
    print(f"src:     {src_root}")
    print(f"dst:     {dst_root}")
    print(f"size:    {args.size}x{args.size}, workers={args.workers}")

    mkvs = sorted(src_root.rglob("*.mkv"))
    if args.limit:
        mkvs = mkvs[:args.limit]
    print(f"found {len(mkvs)} .mkv files")
    if not mkvs:
        sys.exit(0)

    # Build (src, dst, size, use_nvenc) triplets, preserving relative dir layout
    # but swapping the .mkv extension for .mp4.
    jobs = []
    for src in mkvs:
        rel = src.relative_to(src_root)
        dst = (dst_root / rel).with_suffix(".mp4")
        jobs.append((src, dst, args.size, use_nvenc))

    t0 = time.perf_counter()
    frame_counts = {}
    n_ok = n_skip = n_fail = 0

    with mp.Pool(args.workers) as pool:
        for i, (path, n, err) in enumerate(pool.imap_unordered(transcode_one, jobs), start=1):
            rel = str(Path(path).relative_to(dst_root))
            if err is None and n is not None:
                frame_counts[rel] = n
                n_ok += 1
            else:
                n_fail += 1
                print(f"  FAIL {rel}: {err}")
            if i % 25 == 0 or i == len(jobs):
                elapsed = time.perf_counter() - t0
                rate = i / max(elapsed, 1e-9)
                eta = (len(jobs) - i) / max(rate, 1e-9)
                print(f"  [{i}/{len(jobs)}] ok={n_ok} fail={n_fail} "
                      f"| {rate:.2f} files/s | ETA {eta/60:.1f} min")

    fc_path = dst_root / "frame_counts.json"
    with open(fc_path, "w") as f:
        json.dump(frame_counts, f)
    print(f"\nwrote {len(frame_counts)} entries to {fc_path}")
    print(f"total time: {(time.perf_counter()-t0)/60:.1f} min")
    print(f"\nNext step: re-run training with --data-path {dst_root}")
    print("(remember to delete dataset_split.json so a fresh file-level split gets built)")


if __name__ == "__main__":
    main()
