#!/usr/bin/env python3
"""
Script to extract frames from a video and save them to an HDF5 file.
Extracts all frames resized to 384x512 by default.
Runs SegNet on even frames (storing output in 'seg') and PoseNet on the first 3 pairs (storing output in 'pos').
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import h5py
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add project root to sys.path to allow importing evaluator
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluator.loss import get_manager

def get_video_info(video_path):
    """Get the width, height, and total frames of the video using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
        '-of', 'csv=p=0',
        video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e.stderr}")
        raise
    
    lines = result.stdout.strip().split('\n')
    if len(lines) == 0 or not lines[0]:
        raise ValueError(f"Could not parse ffprobe output: {result.stdout}")

    stream_parts = lines[0].split(',')
    width = int(stream_parts[0])
    height = int(stream_parts[1])
    
    total_frames = None
    if len(lines) >= 2 and stream_parts[2] != 'N/A' and lines[1].strip() != 'N/A':
        try:
            r_frame_rate = stream_parts[2]
            fps_parts = r_frame_rate.split('/')
            if len(fps_parts) == 2:
                fps = float(fps_parts[0]) / float(fps_parts[1])
            else:
                fps = float(r_frame_rate)
                
            duration = float(lines[1].strip())
            total_frames = int(duration * fps)
        except Exception:
            pass
            
    return width, height, total_frames

def main():
    parser = argparse.ArgumentParser(description="Extract all frames and run evaluation models into an HDF5 file.")
    parser.add_argument('-i', '--input', default='data/0.mkv', help='Path to the input video file (default: data/0.mkv)')
    parser.add_argument('-o', '--output', default='data/frames.h5', help='Path to the output HDF5 file (default: data/frames.h5)')
    parser.add_argument('-r', action='store_true', help='Disable resizing (keep original dimensions)')
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit the number of frames to process')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    disable_resize = args.r

    if not os.path.exists(input_path):
        print(f"Error: Input video file '{input_path}' does not exist.")
        return

    # Create output directory if it does not exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Reading video: {input_path}")
    try:
        orig_width, orig_height, total_frames = get_video_info(input_path)
        print(f"Original video dimensions: {orig_width}x{orig_height}")
        if total_frames:
            print(f"Estimated total frames: {total_frames}")
        else:
            print("Total frames count not available in container metadata.")
    except Exception as e:
        print(f"Failed to get video info: {e}")
        return

    if disable_resize:
        width, height = orig_width, orig_height
        print("Resizing is disabled.")
    else:
        width = 512
        height = 384
        print(f"Resizing enabled. Target dimensions: {width}x{height}")

    # Set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing models on device: {device}")
    manager = get_manager(device=device)

    orig_frame_size = orig_width * orig_height * 3

    # Command to decode video frames to raw RGB24 at original resolution
    ffmpeg_cmd = [
        'ffmpeg', '-i', input_path,
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ]

    print(f"Extracting and saving frames to {output_path}...")
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    saved_count = 0
    total_processed = 0
    pose_buffer = []

    try:
        with h5py.File(output_path, 'w') as h5_file:
            # frames: (num_frames, height, width, 3)
            dset_frames = h5_file.create_dataset(
                'frames',
                shape=(0, height, width, 3),
                maxshape=(None, height, width, 3),
                dtype=np.uint8,
                chunks=(1, height, width, 3)
            )
            # seg: (num_even_frames, height, width)
            dset_seg = h5_file.create_dataset(
                'seg',
                shape=(0, height, width),
                maxshape=(None, height, width),
                dtype=np.uint8,
                chunks=(1, height, width)
            )
            # pos: (3, 12)
            dset_pos = h5_file.create_dataset(
                'pos',
                shape=(0, 12),
                maxshape=(None, 12),
                dtype=np.float32,
                chunks=(1, 12)
            )

            with tqdm(total=total_frames, desc="Processing frames", unit="f") as pbar:
                while True:
                    if args.limit is not None and total_processed >= args.limit:
                        break
                    raw_frame = process.stdout.read(orig_frame_size)
                    if len(raw_frame) != orig_frame_size:
                        break

                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((orig_height, orig_width, 3))
                    
                    if not disable_resize:
                        img = Image.fromarray(frame)
                        img_resized = img.resize((width, height), resample=Image.Resampling.BILINEAR)
                        frame = np.array(img_resized)

                    # Store all frames in H5
                    dset_frames.resize(dset_frames.shape[0] + 1, axis=0)
                    dset_frames[-1] = frame
                    saved_count += 1

                    # Run SegNet on even frames (0, 2, 4, ...)
                    if total_processed % 2 == 0:
                        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
                        with torch.inference_mode():
                            seg_out = manager.segnet(frame_t)
                            seg_argmax = seg_out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                        dset_seg.resize(dset_seg.shape[0] + 1, axis=0)
                        dset_seg[-1] = seg_argmax

                    # Run PoseNet on the first 3 pairs (frames 0 to 5)
                    if total_processed < 6:
                        pose_buffer.append(frame)
                        if len(pose_buffer) == 2:
                            pair_t = torch.stack([
                                torch.from_numpy(pose_buffer[0]).permute(2, 0, 1),
                                torch.from_numpy(pose_buffer[1]).permute(2, 0, 1)
                            ]).unsqueeze(0).float().to(device)
                            
                            with torch.inference_mode():
                                posenet_in = manager.posenet.preprocess_input(pair_t)
                                posenet_out = manager.posenet(posenet_in)
                                pos_val = posenet_out['pose'].squeeze(0).cpu().numpy().astype(np.float32)
                            
                            dset_pos.resize(dset_pos.shape[0] + 1, axis=0)
                            dset_pos[-1] = pos_val
                            pose_buffer = []

                    total_processed += 1
                    pbar.update(1)
                    pbar.set_postfix(saved=saved_count)
            
            frames_shape = dset_frames.shape
            seg_shape = dset_seg.shape
            pos_shape = dset_pos.shape

    except Exception as e:
        print(f"An error occurred during frame extraction: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        process.terminate()
        process.wait()

    print(f"Finished. Total frames processed: {total_processed}")
    print(f"  Saved in 'frames' (all): {frames_shape}")
    print(f"  Saved in 'seg' (even):   {seg_shape}")
    print(f"  Saved in 'pos' (first 3 pairs): {pos_shape}")

if __name__ == '__main__':
    main()
