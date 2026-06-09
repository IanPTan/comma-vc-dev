#!/bin/bash

# This script fixes MKVs that were remuxed with incorrect framerates.
# It extracts the raw HEVC bitstream and re-remuxes it with -r 20.

PROCESSED_DIR="data/comma2k19"

if [ ! -d "$PROCESSED_DIR" ]; then
    echo "Error: $PROCESSED_DIR does not exist."
    exit 1
fi

echo "Starting framerate fix for MKVs in $PROCESSED_DIR..."

# Find all .mkv files in the processed data
find "$PROCESSED_DIR" -name "*.mkv" | while read -r mkv_path; do
    echo "Fixing $mkv_path..."
    
    # Create a temporary raw hevc file
    temp_hevc="${mkv_path}.temp.hevc"
    fixed_mkv="${mkv_path}.fixed.mkv"
    
    # 1. Extract raw bitstream (ignoring timestamps)
    ffmpeg -nostdin -y -i "$mkv_path" -vcodec copy -f hevc "$temp_hevc" -loglevel error
    
    if [ $? -eq 0 ]; then
        # 2. Re-remux with correct 20fps metadata
        ffmpeg -nostdin -y -f hevc -r 20 -i "$temp_hevc" -vcodec copy "$fixed_mkv" -loglevel error
        
        if [ $? -eq 0 ]; then
            # Replace original with fixed version
            mv "$fixed_mkv" "$mkv_path"
            echo "Successfully fixed $mkv_path"
        else
            echo "ERROR: Failed to re-remux $mkv_path"
        fi
    else
        echo "ERROR: Failed to extract bitstream from $mkv_path"
    fi
    
    # Cleanup temp file
    rm -f "$temp_hevc"
done

echo "Framerate fix complete."
