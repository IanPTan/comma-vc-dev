#!/bin/bash

# Base directory for extracted chunks
BASE_DIR="data/comma2k19_raw/raw_data"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist."
    exit 1
fi

echo "Checking dongle_id consistency within chunks..."
echo "----------------------------------------------"

for chunk in "$BASE_DIR"/Dataset_Chunk_*; do
    if [ -d "$chunk" ]; then
        chunk_name=$(basename "$chunk")
        
        # Get all directories containing '|', extract the part before '|', and find unique ones
        unique_ids=$(ls -1 "$chunk" | grep '|' | cut -d'|' -f1 | sort -u)
        id_count=$(echo "$unique_ids" | wc -l)
        
        if [ "$id_count" -eq 0 ]; then
            echo "$chunk_name: No routes found."
        elif [ "$id_count" -eq 1 ]; then
            echo "$chunk_name: All routes belong to a single dongle_id: $unique_ids"
        else
            echo "$chunk_name: Multiple dongle_ids found ($id_count):"
            echo "$unique_ids" | sed 's/^/  - /'
        fi
    fi
done
