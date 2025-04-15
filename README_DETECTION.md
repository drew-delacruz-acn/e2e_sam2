# Object Detection Pipeline (Owlv2)

This is a simplified version of the full segmentation pipeline, focusing only on the Object Detection component using Owlv2. This version has no dependencies on SAM2, making it easier to set up and run.

## Quick Start

```bash
# Run with default parameters
python run_detection.py

# Run with custom image and queries
python run_detection.py --image path/to/image.jpg --queries "person" "car" "dog"
```

## Features

- **Text-prompted detection**: Find objects in images by simply describing them
- **Multi-object detection**: Detect multiple object types in a single pass
- **Automatic filtering**: Removes overlapping bounding boxes
- **Visualization**: Display and save detection results with bounding boxes
- **Cropping**: Extract detected objects into separate image files

## Repository Structure

```
project_root/
├── pipeline/
│   ├── __init__.py              # Pipeline package
│   ├── config.py                # Configuration settings
│   ├── pipeline_detection.py    # Detection pipeline class
│   ├── modules/
│       ├── __init__.py
│       ├── detection.py         # Object detection (Owlv2)
│       ├── visualization.py     # Visualization utilities
│       └── utils.py             # Common utilities
├── data/                        # Input data
├── results/                     # Output results
├── detect_only.py               # Simple detection script
├── run_detection.py             # Script to run the detection pipeline
└── requirements.txt             # Dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Internet connection (for downloading model weights)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Run the detection script:
```bash
python run_detection.py
```

This automatically creates a virtual environment and installs the required dependencies.

## Usage

### Command Line Options

```bash
python run_detection.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--image` | Path to input image | `data/thor_hammer.jpeg` |
| `--queries` | Text queries for detection | `["hammer"]` |
| `--visualize` | Show visualizations | `True` |
| `--save` | Save results to disk | `True` |
| `--results-dir` | Directory to save results | `results` |
| `--detection-threshold` | Detection confidence threshold | `0.1` |

### Example Commands

```bash
# Detect people and dogs in an image
python run_detection.py --image photos/park.jpg --queries "person" "dog"

# Use a higher confidence threshold
python run_detection.py --detection-threshold 0.3

# Save results to a custom directory
python run_detection.py --results-dir output/detection_results
```

### Programmatic Usage

```python
from pipeline.pipeline_detection import DetectionPipeline

# Initialize the pipeline
pipeline = DetectionPipeline()

# Run detection
results = pipeline.run(
    image_path="data/my_image.jpg",
    text_queries=["cat", "dog", "bird"],
    visualize=True,
    save_results=True
)

# Access the results
detection_results = results["detection"]
boxes = detection_results["boxes"]
scores = detection_results["scores"]
labels = detection_results["labels"]
```

## How It Works

1. **Configuration**: The pipeline is initialized with default or custom configuration.
2. **Detection**: The image is processed using Owlv2's text-prompted object detection.
3. **Filtering**: Overlapping bounding boxes are filtered based on IoU (Intersection over Union).
4. **Visualization**: Results are displayed with bounding boxes and saved if requested.
5. **Output**: Detected objects are cropped and saved as separate images.

## Output

Results are saved to the specified directory with this structure:

```
results/
├── crops/                     # Cropped objects
│   ├── box_0.png
│   └── box_1.png
└── visualizations/
    └── detection_results.png  # Detection visualization
```

## Dependencies

- torch
- torchvision
- transformers
- Pillow
- matplotlib
- numpy

## Extending the Pipeline

The modular design makes it easy to extend:

1. Add new detection models by creating classes that follow the `ObjectDetector` interface
2. Customize visualization by extending the `Visualizer` class
3. Add post-processing steps by modifying the `DetectionPipeline.run()` method 