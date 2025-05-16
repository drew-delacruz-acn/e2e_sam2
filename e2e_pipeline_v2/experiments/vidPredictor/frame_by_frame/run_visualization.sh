#!/bin/bash
# Run the box visualization script with default arguments

# Default paths (modify these as needed)
RESULTS_JSON="frame_by_frame_rezy_logs/segmentation_results.json"
FRAMES_DIR="$1"  # Use first command line argument as frames directory
OUTPUT_DIR="frame_by_frame_rezy_logs/box_visualizations"

# Check if frames directory is provided
if [ -z "$FRAMES_DIR" ]; then
    echo "Error: Please provide the frames directory as the first argument"
    echo "Usage: ./run_visualization.sh /path/to/frames/directory"
    echo "Example: ./run_visualization.sh /home/ubuntu/code/drew/e2e_sam2/data/frames/Scenes\ 061-080__265H-2-_20230815215828529"
    exit 1
fi

# Check if results JSON exists
if [ ! -f "$RESULTS_JSON" ]; then
    echo "Error: Results JSON file not found at $RESULTS_JSON"
    echo "Please provide the correct path to the segmentation results JSON"
    exit 1
fi

# Make script executable if it's not already
chmod +x visualize_boxes.py

# Run the visualization script
echo "Running visualization with:"
echo "  Results JSON: $RESULTS_JSON"
echo "  Frames directory: $FRAMES_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""

python visualize_boxes.py \
    --results_json "$RESULTS_JSON" \
    --frames_dir "$FRAMES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --color_mode "consistent"

echo ""
echo "Visualization complete!"
echo "Results saved to: $OUTPUT_DIR" 