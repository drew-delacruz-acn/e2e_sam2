# CD-FSOD Detector with Continuity-Based Tracking

This implementation provides a custom detector for CD-FSOD detections with continuity-based tracking. It identifies both first appearances and reappearances of objects after significant gaps, which is useful for testing SAM2 propagation capabilities.

## Features

- Tracks objects across frames using IoU and class matching
- Detects first appearances of objects
- Detects reappearances after a configurable gap
- Supports confidence thresholding
- Allows class/label mapping
- Includes visualization tools

## Directory Structure

The JSON files should be named by frame index (0.json, 1.json, etc.) and contain detection data in this format:

```json
[
  {
    "coordinates": [x1, y1, x2, y2],
    "label": "class_name",
    "confidence": 0.9
  },
  ...
]
```

## Running with Custom Input

1. Prepare your input data:
   - Place your CD-FSOD JSON files in a directory (e.g., `data/json_files/`)
   - Place your video frames in a directory (e.g., `data/frames/`)

2. Run the detector script:

```bash
python run_cd_fsod.py --json_dir data/json_files --frames_dir data/frames --visualize
```

## Command-line Arguments

- `--json_dir`: Directory containing CD-FSOD JSON files (required)
- `--frames_dir`: Directory containing video frames (required)
- `--output_dir`: Directory to save output visualizations (default: 'output')
- `--confidence`: Confidence threshold for detections (default: 0.2)
- `--iou`: IoU threshold for object matching (default: 0.5)
- `--min_gap`: Minimum frame gap to consider as a reappearance (default: 10)
- `--queries`: Comma-separated list of object classes to detect (use "all" for all classes)
- `--start_frame`: First frame to process (default: 0)
- `--end_frame`: Last frame to process (-1 for all frames)
- `--visualize`: Generate visualization of detections

## Example Usage

Detect all classes with default parameters and visualize results:

```bash
python run_cd_fsod.py --json_dir data/json_files --frames_dir data/frames --visualize
```

Detect specific classes with custom thresholds:

```bash
python run_cd_fsod.py --json_dir data/json_files --frames_dir data/frames --queries "monitor,uniform" --confidence 0.3 --min_gap 15 --visualize
```

Process only a subset of frames:

```bash
python run_cd_fsod.py --json_dir data/json_files --frames_dir data/frames --start_frame 50 --end_frame 100 --visualize
```

## Integration with SAM2

To integrate with SAM2 for propagation testing:

1. Run the CD-FSOD detector to get first appearances and reappearances
2. Pass these detections to SAM2 for mask generation
3. Propagate masks through consecutive frames
4. Evaluate propagation quality by comparing with ground truth or through visual inspection

This approach allows testing how well SAM2 can propagate masks through continuous sequences and handle reappearances of objects. 