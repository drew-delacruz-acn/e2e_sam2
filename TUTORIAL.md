# Object Detection and Segmentation Pipeline Tutorial

This tutorial explains how to use the modular object detection and segmentation pipeline we've created, as well as the organization of the repository.

## Repository Structure

```
project_root/
├── pipeline/                  # Main pipeline package
│   ├── __init__.py           # Pipeline package initialization
│   ├── config.py             # Configuration settings
│   ├── pipeline.py           # Main pipeline class
│   ├── modules/              # Pipeline modules
│       ├── __init__.py       # Modules package initialization
│       ├── detection.py      # Object detection (Owlv2)
│       ├── segmentation.py   # Segmentation (SAM2)
│       ├── visualization.py  # Visualization utilities
│       └── utils.py          # Common utilities
├── data/                     # Input data directory
│   └── thor_hammer.jpeg      # Sample image
├── results/                  # Output results
├── sam2/                     # SAM2 model repository
├── run_pipeline.py           # Script to run the pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # Project overview
```

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Virtual environment (automatically created by `run_pipeline.py`)
- Dependencies: PyTorch, Torchvision, Transformers, OpenCV, etc.

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Run the pipeline script:
```bash
python run_pipeline.py
```

This will:
- Create a virtual environment (if it doesn't exist)
- Install all required dependencies
- Set up the SAM2 package from the local directory

## Using the Pipeline

### Command Line Usage

The `run_pipeline.py` script provides a convenient command-line interface:

```bash
python run_pipeline.py --image path/to/image.jpg --queries object1 object2 --results-dir custom_results
```

#### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--image` | Path to input image | `data/thor_hammer.jpeg` |
| `--queries` | Text queries for detection | `["hammer"]` |
| `--visualize` | Show visualizations | `True` |
| `--save` | Save results to disk | `True` |
| `--results-dir` | Directory to save results | `results` |
| `--detection-threshold` | Detection confidence threshold | `0.1` |
| `--segmentation-threshold` | Segmentation confidence threshold | `0.5` |

### Programmatic Usage

You can also use the pipeline in your own Python code:

```python
from pipeline.pipeline import Pipeline

# Optional custom configuration
custom_config = {
    "detection": {
        "threshold": 0.2,  # Override detection threshold
        "iou_threshold": 0.6  # Override IoU threshold for box filtering
    },
    "segmentation": {
        "score_threshold": 0.7  # Override segmentation score threshold
    },
    "paths": {
        "results_dir": "custom_results"  # Custom results directory
    }
}

# Initialize the pipeline
pipeline = Pipeline(config_override=custom_config)  # Or use Pipeline() for defaults

# Run the pipeline
results = pipeline.run(
    image_path="path/to/image.jpg",
    text_queries=["chair", "table"],
    visualize=True,  # Show visualizations
    save_results=True  # Save results to disk
)

# Work with the results
detection_results = results["detection"]
point_segmentation = results["point_segmentation"]
box_segmentation = results["box_segmentation"]
```

## Pipeline Components

### 1. Configuration (config.py)

The configuration module provides default settings and a method to override them:

```python
# Default configuration in config.py
DEFAULT_CONFIG = {
    "detection": {
        "model_name": "microsoft/Owlv2-uncropped-finetuned-coco",
        "threshold": 0.1,
        "iou_threshold": 0.5
    },
    "segmentation": {
        "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "checkpoints/sam2.1_hiera_large.pt",
        "score_threshold": 0.5
    },
    "paths": {
        "results_dir": "results"
    }
}
```

### 2. Object Detection (detection.py)

The `ObjectDetector` class uses Owlv2 from Hugging Face Transformers to detect objects based on text queries:

```python
from pipeline.modules.detection import ObjectDetector

detector = ObjectDetector(config)
results = detector.detect(
    image_path="path/to/image.jpg", 
    text_queries=["hammer", "chair"]
)
```

### 3. Segmentation (segmentation.py)

The `Segmenter` class uses SAM2 (Segment Anything Model 2) to create precise segmentation masks:

```python
from pipeline.modules.segmentation import Segmenter

segmenter = Segmenter(config)
results = segmenter.segment_with_boxes(image_np, boxes)
# or
results = segmenter.segment_with_points(image_np, boxes)
```

### 4. Visualization (visualization.py)

The `Visualizer` class provides utilities to visualize detection and segmentation results:

```python
from pipeline.modules.visualization import Visualizer

visualizer = Visualizer(config)
visualizer.show_detections(image, boxes, scores, labels, text_queries)
visualizer.show_segmentation(image, segmentation_results)
```

### 5. Utilities (utils.py)

Common utility functions:

```python
from pipeline.modules.utils import calculate_iou, ensure_dir, save_crops

# Calculate IoU between two bounding boxes
iou = calculate_iou(box1, box2)

# Ensure a directory exists
ensure_dir("path/to/directory")

# Save cropped images from bounding boxes
crop_paths = save_crops(image, boxes, "path/to/save")
```

## Handling SAM2 Dependencies

The pipeline is configured to work with the SAM2 repository structure. If you encounter issues with SAM2 imports, make sure:

1. The SAM2 repository is cloned and installed in development mode:
```bash
cd sam2
pip install -e .
```

2. The paths in `segmentation.py` point to the correct SAM2 directories:
```python
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sam2_dir = os.path.join(root_dir, "sam2")
```

## Output Results

Results are saved to the specified output directory with this structure:

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

## Troubleshooting

### Common Issues

1. **SAM2 Import Errors**: 
   - Solution: Make sure SAM2 is installed in development mode (`pip install -e .` from the SAM2 directory)
   - Alternative: Run from a different directory to avoid Python package namespace conflicts

2. **Missing Dependencies**:
   - Solution: Run `pip install -r requirements.txt` to install all dependencies

3. **GPU Issues**:
   - Solution: The code automatically selects the best available device (CUDA, MPS, or CPU)

4. **File Not Found Errors**:
   - Solution: Make sure paths are correct relative to the current working directory

## Example Workflow

1. **Detect Objects**:
   ```bash
   python run_pipeline.py --image data/my_image.jpg --queries "person" "dog" --detection-threshold 0.2
   ```

2. **Review Results**:
   - Check detection visualizations in the terminal
   - Examine the segmentation masks for each detected object
   - Results are saved to the `results` directory

3. **Custom Configuration**:
   - Adjust thresholds for detection and segmentation
   - Change model parameters as needed
   - Modify visualization options

## Extending the Pipeline

The modular design makes it easy to extend the pipeline:

1. **Add New Detection Models**:
   - Create a new detection module in `pipeline/modules/`
   - Implement the same interface as `ObjectDetector`

2. **Add New Segmentation Models**:
   - Create a new segmentation module in `pipeline/modules/`
   - Implement the same interface as `Segmenter`

3. **Add Post-Processing Steps**:
   - Modify the `Pipeline.run()` method to include additional processing

4. **Customize Visualizations**:
   - Extend the `Visualizer` class with new visualization methods 