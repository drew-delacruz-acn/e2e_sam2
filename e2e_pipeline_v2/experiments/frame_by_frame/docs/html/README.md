# SAM2 Video Segmentation Pipeline

This pipeline takes object detection data and video frames, then uses SAM2 to create consistent object segmentation across the entire video.

## What It Does

1. **Input**: Object detection JSON files + video frame images
2. **Process**: Uses SAM2 to segment and track objects across frames
3. **Output**: JSON summary of object tracking + visualization images

### Example Use Case
You have a video of someone holding a "Time Stick" object. The pipeline will:
- Find the Time Stick in detection files 
- Use SAM2 to segment it precisely in each frame
- Track it consistently across the video
- Output where it appears and how well it was detected

## Quick Start

### Installation
See [INSTALLATION.md](../INSTALLATION.md) for setup instructions.

### Basic Usage
```bash
python main.py \
  --detections_dir /path/to/detection/files \
  --frames_dir /path/to/video/frames \
  --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --results_dir results
```

## Input Format

### Detection Files
JSON files named by frame number (`0.json`, `1.json`, etc.):
```json
[
  {
    "label": "Time Stick",
    "confidence": 0.95,
    "coordinates": [100, 150, 200, 250]
  }
]
```

### Video Frames
Image files named by frame number (`0.jpg`, `1.jpg`, etc.)

## Output

### Results Directory
```
results/
├── segmentation_results.json  # Main output - object tracking data
└── visualizations/            # Frame images with segmentation overlays
    ├── frame_0000.png
    └── frame_0001.png
```

### Output JSON Format
```json
[
  {
    "samObjectId": 1,
    "samDictatedAppearances": [
      {"frameNum": 0, "boundingBox": [100, 150, 200, 250]},
      {"frameNum": 1, "boundingBox": [102, 153, 202, 253]}
    ],
    "cdfsodPredictions": [
      {"frameNumber": 0, "class": "Time Stick", "confidence": 0.95},
      {"frameNumber": 1, "class": "Time Stick", "confidence": 0.94}
    ]
  }
]
```

This shows:
- **samDictatedAppearances**: Where SAM2 found the object in each frame
- **cdfsodPredictions**: Where/when the original detector found it

## Common Options

```bash
# Use lower confidence threshold to include more detections
python main.py --confidence_threshold 0.7 [other args...]

# Process every 5th frame only (faster)
python main.py --vis_stride 5 [other args...]

# Skip generating visualization images
python main.py --no_vis [other args...]

# Save binary mask files for each object
python main.py --save_masks [other args...]

# Enable debug output
python main.py --debug [other args...]
```

## How It Works

1. **Load detections** from JSON files
2. **Filter by confidence** (default: 0.9)
3. **Group into tracking objects** (same object across frames)
4. **Initialize SAM2** with object locations
5. **Propagate segmentation** through all video frames
6. **Match segments to detections** and create summary
7. **Generate visualizations** and save results

## Visual Documentation

- [Pipeline Flow Diagrams](pipeline_diagram.html) - See how data flows through the system
- [Data Structure Diagrams](data_structures_diagram.html) - Understand input/output formats

## Requirements

- Python 3.8+
- 8GB+ RAM (16GB recommended)
- GPU with 4GB+ memory (or CPU)
- ~10GB disk space for models

## Example Command

```bash
python main.py \
  --detections_dir "/Users/andrewdelacruz/e2e_sam2/gitignore_exception/data/detections/Scenes 061-080__265H-2-_20230815215828529" \
  --frames_dir "/Users/andrewdelacruz/e2e_sam2/gitignore_exception/data/frames/Scenes 061-080__265H-2-_20230815215828529" \
  --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt \
  --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --results_dir results
```

## Need Help?

1. Run `python main.py --help` to see all options
2. Check that your detection JSON files match the expected format
3. Ensure frame images are named numerically (0.jpg, 1.jpg, etc.)
4. Try with `--debug` flag to see detailed processing information
