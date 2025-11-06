#!/bin/bash
# Batch convert Syndrone scenes to unified format

set -e  # Exit on error

# Configuration
INPUT_BASE="Town01_Opt_120/Town01_Opt_120/ClearNoon"
OUTPUT_BASE="output_unified"
FOV=90.0

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Syndrone Batch Conversion Script"
echo "========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# List of scenes to convert
SCENES=(
    "height20m"
    "height50m"
    "height80m"
)

# Convert each scene
for scene in "${SCENES[@]}"; do
    echo -e "${YELLOW}Processing: $scene${NC}"
    
    INPUT_DIR="$INPUT_BASE/$scene"
    OUTPUT_DIR="$OUTPUT_BASE/$scene"
    
    # Check if input directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "  [WARN] Directory not found: $INPUT_DIR, skipping..."
        continue
    fi
    
    # Check if already converted
    if [ -f "$OUTPUT_DIR/complete_log.txt" ]; then
        echo "  [INFO] Already converted, skipping..."
        continue
    fi
    
    # Run conversion
    echo "  [INFO] Converting $INPUT_DIR -> $OUTPUT_DIR"
    python convert_syndrone.py "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --fov "$FOV" \
        --dataset_name "syndrone" \
        --version "0.1"
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}[SUCCESS] Converted $scene${NC}"
    else
        echo "  [ERROR] Failed to convert $scene"
    fi
    
    echo ""
done

echo "========================================="
echo "Batch conversion complete!"
echo "========================================="
echo ""
echo "Output directory: $OUTPUT_BASE/"
echo ""
echo "To visualize results, run:"
echo "  python syndrone_multi_frame_vis.py --data_dir $INPUT_BASE/height20m --limit 50"

