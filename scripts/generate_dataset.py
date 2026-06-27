#!/usr/bin/env python3
"""
Script to extract frames from a video and save them to an HDF5 file.
Extracts the first 6 frames and then the remaining even frames.
Resizes frames to the closest multiple of 64 less than the dimensions by default.
"""

import argparse
import os
import subprocess
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Extract selected frames from a video into an HDF5 file.")
    parser.add_argument('-i', '--input', default='data/0.mkv', help='Path to the input video file (default: data/0.mkv)')
    parser.add_argument('-o', '--output', default='data/frames.h5', help='Path to the output HDF5 file (default: data/frames.h5)')
    parser.add_argument('-r', action='store_true', help='Disable resizing (default is to resize to closest multiple of 64)')
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
        width = (orig_width // 64) * 64
        height = (orig_height // 64) * 64
        print(f"Resizing enabled. Target dimensions (closest multiple of 64): {width}x{height}")

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

    try:
        with h5py.File(output_path, 'w') as h5_file:
            # We create a resizable dataset for the frames.
            # Shape is (num_frames, height, width, 3)
            dset = h5_file.create_dataset(
                'frames',
                shape=(0, height, width, 3),
                maxshape=(None, height, width, 3),
                dtype=np.uint8,
                chunks=(1, height, width, 3)
            )

            with tqdm(total=total_frames, desc="Processing frames", unit="f") as pbar:
                while True:
                    raw_frame = process.stdout.read(orig_frame_size)
                    if len(raw_frame) != orig_frame_size:
                        break

                    # Frame selection logic: first 6 frames (0-5) and remaining even frames (6, 8, ...)
                    if total_processed < 6 or total_processed % 2 == 0:
                        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((orig_height, orig_width, 3))
                        
                        if not disable_resize:
                            img = Image.fromarray(frame)
                            img_resized = img.resize((width, height), resample=Image.Resampling.BILINEAR)
                            frame = np.array(img_resized)

                        # Append to HDF5 dataset
                        dset.resize(dset.shape[0] + 1, axis=0)
                        dset[-1] = frame
                        saved_count += 1

                    total_processed += 1
                    pbar.update(1)
                    pbar.set_postfix(saved=saved_count)

    except Exception as e:
        print(f"An error occurred during frame extraction: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return
    finally:
        process.terminate()
        process.wait()

    print(f"Finished. Total frames processed: {total_processed}, frames saved to H5: {saved_count}")

if __name__ == '__main__':
    main()
