# CD-FSOD Detector Integration

This document explains how to use the CD-FSOD detector integration with the object tracking pipeline.

## Overview

The CD-FSOD detector allows you to use pre-computed object detections from the CD-FSOD model instead of using the OWLv2 detector. This integration supports both first appearance and reappearance detection - a key feature for tracking objects that disappear and reappear in videos.

## Key Features

- **Continuity-based tracking**: Tracks objects across frames and identifies first appearances and reappearances
- **Appearance gap control**: Configure how many frames an object must be absent before counting as a reappearance
- **Confidence threshold**: Filter out low-confidence detections
- **Compatible with SAM2**: Works with the existing SAM2 propagation pipeline

## How to Use

### Command Line Interface

The pipeline supports the CD-FSOD detector through the following options:

```bash
# Use the main pipeline with CD-FSOD detector
python src/object_tracking_pipeline.py \
  --frames-dir path/to/frames \
  --detector cd_fsod \
  --cd-fsod-path path/to/json_detections \
  --min-gap-frames 10 \
  --confidence 0.5 \
  --text-queries "all" \
  --sam2-checkpoint path/to/sam2_checkpoint \
  --sam2-config path/to/sam2_config \
  --output-dir ./tracking_results

# Or use the dedicated test script for CD-FSOD integration
python test_cd_fsod_integration.py \
  --frames-dir path/to/frames \
  --cd-fsod-path path/to/json_detections \
  --min-gap-frames 10 \
  --confidence 0.5 \
  --sam2-checkpoint path/to/sam2_checkpoint \
  --sam2-config path/to/sam2_config \
  --output-dir ./cd_fsod_results
```

### Parameters

- `--detector cd_fsod`: Specify to use the CD-FSOD detector (for main pipeline)
- `--cd-fsod-path`: Directory containing the CD-FSOD JSON detection files (numbered 0.json, 1.json, etc.)
- `--min-gap-frames`: Minimum number of frames an object must be absent to count as a reappearance
- `--confidence`: Confidence threshold for filtering detections
- `--text-queries`: Object classes to detect. Use "all" to include all detected classes.
- `--separate-objects`: Process each object separately for better SAM2 stability

## CD-FSOD Detector Format

The CD-FSOD detector expects a directory containing numbered JSON files (0.json, 1.json, etc.) with detections in the following format:

```json
[
  {
    "coordinates": [x1, y1, x2, y2],
    "label": "Object Class",
    "confidence": 0.95
  },
  ...
]
```

Where:
- `coordinates`: Bounding box in [x1, y1, x2, y2] format
- `label`: Object class (e.g., "TVA Monitor", "Time Stick")
- `confidence`: Detection confidence score between 0 and 1

## Integration Details

The integration includes:

1. **CD-FSOD Detector Class**: Loads and processes JSON detection files
2. **Continuity Tracking**: Identifies objects across frames
3. **Pipeline Integration**: Seamlessly works with the existing pipeline
4. **Frame-to-detection mapping**: Maps frames to significant detection events

## Tips for Best Results

- **Confidence threshold**: Start with a value of 0.5 and adjust as needed
- **Min gap frames**: A value of 10-30 works well depending on video frame rate
- **Separate objects**: Use `--separate-objects` for more reliable SAM2 propagation
- **Class filtering**: Use specific class names with `--text-queries` to focus on objects of interest 