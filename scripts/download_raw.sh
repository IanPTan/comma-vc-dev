# run from repo root to download the raw comma2k19 dataset

: <<'COMPLETED'
hf download commaai/comma2k19 --repo-type dataset --include "raw_data/*" --local-dir data/comma2k19_raw

RAW_DIR="data/comma2k19_raw/raw_data"

echo "Starting sequential extraction to save space..."

for zip_file in "$RAW_DIR"/Chunk_*.zip; do
    if [ -f "$zip_file" ]; then
        filename=$(basename "$zip_file" .zip)
        echo "Extracting $filename..."
        
        # Unzip into the same directory
        unzip -q "$zip_file" -d "$RAW_DIR"
        
        # Rename "Chunk_n" to "Dataset_chunk_n" for our DaliDataLoader
        if [ -d "$RAW_DIR/$filename" ]; then
            mv "$RAW_DIR/$filename" "$RAW_DIR/Dataset_$filename"
        elif [ -d "$RAW_DIR/${filename,,}" ]; then # handle potential lowercase
            mv "$RAW_DIR/${filename,,}" "$RAW_DIR/Dataset_$filename"
        fi
        
        # Delete the zip immediately to free up space
        rm "$zip_file"
        echo "Done with $filename."
    fi
done

echo "All chunks extracted and cleaned up."
COMPLETED


# --- Remuxing and Cleanup Phase ---
RAW_DIR="data/comma2k19_raw/raw_data"
PROCESSED_DIR="data/comma2k19"

echo "Starting remuxing to MKV and clearing raw segments..."

# Find all video.hevc files in the raw data
find "$RAW_DIR" -name "video.hevc" | while read -r hevc_path; do
    # hevc_path looks like: data/comma2k19_raw/raw_data/Dataset_Chunk_1/dongle|timestamp/segment/video.hevc
    
    # Extract path components
    SEGMENT_DIR=$(dirname "$hevc_path")            # .../dongle|timestamp/segment
    SEGMENT_NUM=$(basename "$SEGMENT_DIR")         # segment (e.g., 10)
    ROUTE_DIR=$(dirname "$SEGMENT_DIR")            # .../dongle|timestamp
    ROUTE_FULL_NAME=$(basename "$ROUTE_DIR")       # dongle|timestamp
    CHUNK_DIR=$(dirname "$ROUTE_DIR")              # .../Dataset_Chunk_1
    CHUNK_NAME=$(basename "$CHUNK_DIR")            # Dataset_Chunk_1
    
    # Extract just the timestamp from dongle|timestamp
    TIMESTAMP=$(echo "$ROUTE_FULL_NAME" | cut -d'|' -f2)
    
    # Define output path: data/comma2k19/Dataset_Chunk_1/timestamp/segment.mkv
    OUT_PATH="$PROCESSED_DIR/$CHUNK_NAME/$TIMESTAMP"
    mkdir -p "$OUT_PATH"
    
    echo "Remuxing $CHUNK_NAME | $TIMESTAMP | Segment $SEGMENT_NUM..."
    
    # Remux using ffmpeg (copying codec, no re-encoding)
    ffmpeg -nostdin -y -i "$hevc_path" -vcodec copy "$OUT_PATH/$SEGMENT_NUM.mkv" -loglevel error
    
    if [ $? -eq 0 ]; then
        # If successful, delete the entire raw segment directory
        rm -rf "$SEGMENT_DIR"
        echo "Successfully remuxed and deleted raw segment $SEGMENT_NUM."
    else
        echo "ERROR: Failed to remux $hevc_path. Keeping raw file."
    fi
done

echo "Processing complete. Final data is in $PROCESSED_DIR"
