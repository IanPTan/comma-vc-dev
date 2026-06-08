# run from repo root to download the raw comma2k19 dataset
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
