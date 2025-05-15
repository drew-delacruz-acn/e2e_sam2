# SAM2 Video Segmentation Pipeline

This package provides a modular implementation of a video object segmentation pipeline using SAM2 (Segment Anything Model 2). The pipeline takes object detection data and performs segmentation across video frames.

## Features

- Load and process object detection data from JSON files
- Filter detections by confidence score
- Convert detections to a tracking format
- Use SAM2 to segment objects across video frames
- Match SAM2 segmentations with CDFSOD detections
- Generate object-centric tracking and detection summary in JSON format
- Visualize and save segmentation results
- Save all results to a single organized directory

## Directory Structure

```
frame_by_frame/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── detection_processor.py   # Detection loading and processing
│   ├── sam2_segmenter.py        # SAM2 video segmentation
│   ├── visualization.py         # Result visualization
│   ├── result_processor.py      # Process and format results
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
  --results_dir /path/to/results \
  --iou_threshold 0.5
```

### Arguments

- `--detections_dir`: Directory containing object detection JSON files
- `--frames_dir`: Directory containing video frame images
- `--sam2_checkpoint`: Path to the SAM2 model checkpoint file
- `--model_cfg`: Path to the SAM2 model configuration file
- `--confidence_threshold`: Minimum confidence threshold for filtering detections (default: 0.9)
- `--vis_stride`: Visualization stride - display every nth frame (default: 1)
- `--results_dir`: Directory to save all results (JSON and visualization images) (default: 'results')
- `--iou_threshold`: IoU threshold for matching detections to segments (default: 0.3)
- `--debug`: Enable debug mode with additional logging
- `--no_vis`: Skip visualization generation

## Results Directory Structure

When you run the pipeline, it creates a results directory with this structure:

```
results_dir/
├── segmentation_results.json   # Object-centric tracking and detection data
└── visualizations/             # Directory containing visualization images
    ├── frame_0000.png
    ├── frame_0001.png
    ├── ...
```

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

## Output JSON Format

The output JSON file contains an object-centric view of the video segmentation results:

```json
[
  {
    "samObjectId": 1,
    "samDictatedAppearances": [
      {"frameNum": 1, "boundingBox": [100, 150, 200, 250]},
      {"frameNum": 2, "boundingBox": [105, 152, 205, 252]},
      ...
    ],
    "cdfsodPredictions": [
      {"frameNumber": 1, "class": "Time Stick", "confidence": 0.92},
      {"frameNumber": 3, "class": "Time Stick", "confidence": 0.88},
      ...
    ]
  },
  ...
]
```

This format allows for:
- Tracking objects consistently across frames
- Seeing where CDFSOD succeeded or failed to detect objects
- Identifying changes in classification over time
- Understanding an object's trajectory through the video

## Extending the Pipeline

The modular structure makes it easy to extend or modify components:

- To add support for a new detection format, modify `detection_processor.py`
- To change visualization options, modify `visualization.py`
- To add new segmentation methods, extend `sam2_segmenter.py`
- To change how results are processed, modify `result_processor.py`

## License

This project is available under the MIT License.

<!-- 
python e2e_pipeline_v2/experiments/vidPredictor/frame_by_frame/main.py --detections_dir "/Users/andrewdelacruz/e2e_sam2/gitignore_exception/data/detections/Scenes 061-080__265H-2-_20230815215828529" --frames_dir "/Users/andrewdelacruz/e2e_sam2/gitignore_exception/data/frames/Scenes 061-080__265H-2-_20230815215828529" --sam2_checkpoint checkpoints/sam2.1_hiera_large.pt --model_cfg configs/sam2.1/sam2.1_hiera_l.yaml --results_dir results
-->
