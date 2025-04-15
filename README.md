2# Object Detection and Segmentation Pipeline

A modular pipeline for object detection (using Owlv2) and segmentation (using SAM2).

## Project Structure

```
project_root/
├── pipeline/
│   ├── __init__.py           # Pipeline package
│   ├── config.py             # Configuration settings
│   ├── pipeline.py           # Main pipeline class
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── detection.py      # Object detection (Owlv2)
│   │   ├── segmentation.py   # Segmentation (SAM2)
│   │   ├── visualization.py  # Visualization utilities
│   │   └── utils.py          # Common utilities
├── data/                     # Input data
├── results/                  # Output results
├── run_pipeline.py           # Script to run the pipeline
└── requirements.txt          # Dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU recommended

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Run the pipeline script (will create a virtual environment if needed):
```bash
python run_pipeline.py
```

## Usage

### Basic Usage

```bash
python run_pipeline.py --image path/to/image.jpg --queries object1 object2
```

### Command Line Arguments

- `--image`: Path to input image (default: "../data/thor_hammer.jpeg")
- `--queries`: Text queries for detection (default: ["hammer"])
- `--visualize`: Show visualizations (flag, default: True)
- `--save`: Save results to disk (flag, default: True)
- `--results-dir`: Directory to save results (default: "../results")
- `--detection-threshold`: Detection confidence threshold (default: 0.1)
- `--segmentation-threshold`: Segmentation confidence threshold (default: 0.5)

### Programmatic Usage

```python
from pipeline.pipeline import Pipeline

# Initialize with custom configuration
custom_config = {
    "detection": {
        "threshold": 0.2  # Override detection threshold
    },
    "segmentation": {
        "score_threshold": 0.7  # Override segmentation score threshold
    }
}

# Create pipeline
pipeline = Pipeline(config_override=custom_config)

# Run pipeline
results = pipeline.run(
    image_path="path/to/image.jpg",
    text_queries=["chair", "table"],
    visualize=True,
    save_results=True
)
```

## Results

Results are saved to the specified output directory (default: "../results") with the following structure:

```
results/
├── crops/                      # Cropped objects
│   ├── box_0.png
│   └── box_1.png
└── visualizations/
    ├── detection_results.png   # Detection visualization
    ├── box_segmentation/       # Box-based segmentation
    │   ├── seg_result_0_0.png
    │   └── seg_result_1_0.png
    └── point_segmentation/     # Point-based segmentation
        ├── seg_result_0_0.png
        └── seg_result_1_0.png
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 