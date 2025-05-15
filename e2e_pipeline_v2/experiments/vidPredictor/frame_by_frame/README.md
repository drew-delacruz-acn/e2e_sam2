# SAM2 Video Segmentation Pipeline

This package provides a modular implementation of a video object segmentation pipeline using SAM2 (Segment Anything Model 2). The pipeline takes object detection data and performs segmentation across video frames.

## Features

- Load and process object detection data from JSON files
- Filter detections by confidence score
- Convert detections to a tracking format
- Use SAM2 to segment objects across video frames
- Visualize and save segmentation results

## Directory Structure

```
frame_by_frame/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── detection_processor.py   # Detection loading and processing
│   ├── sam2_segmenter.py        # SAM2 video segmentation
│   ├── visualization.py         # Result visualization
│   └── utils.py                 # Helper utilities
├── main.py                      # Main script to run the pipeline
└── README.md                    # This file
```

## Requirements

- Python 3.6+
- PyTorch
- Matplotlib
- NumPy
- PIL (Pillow)
- SAM2 (Segment Anything Model 2)

## Usage

Run the pipeline using the main script:

```bash
python main.py \
  --detections_dir /path/to/detections \
  --frames_dir /path/to/frames \
  --sam2_checkpoint /path/to/sam2_checkpoint.pt \
  --model_cfg /path/to/sam2_config.yaml \
  --confidence_threshold 0.9 \
  --vis_stride 1 \
  --save_path /path/to/output_visualizations
```

### Arguments

- `--detections_dir`: Directory containing object detection JSON files
- `--frames_dir`: Directory containing video frame images
- `--sam2_checkpoint`: Path to the SAM2 model checkpoint file
- `--model_cfg`: Path to the SAM2 model configuration file
- `--confidence_threshold`: Minimum confidence threshold for filtering detections (default: 0.9)
- `--vis_stride`: Visualization stride - display every nth frame (default: 1)
- `--save_path`: Optional path to save visualization results

## Detection File Format

Each detection file should be a JSON file with the following structure:

```json
[
  {
    "label": "object_class",
    "confidence": 0.95,
    "coordinates": [x1, y1, x2, y2]
  },
  ...
]
```

The filename should be the frame number (e.g., `0.json`, `1.json`, etc.).

## Extending the Pipeline

The modular structure makes it easy to extend or modify components:

- To add support for a new detection format, modify `detection_processor.py`
- To change visualization options, modify `visualization.py`
- To add new segmentation methods, extend `sam2_segmenter.py`

## License

This project is available under the MIT License. 