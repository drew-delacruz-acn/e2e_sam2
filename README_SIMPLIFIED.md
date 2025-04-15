# Object Detection and Segmentation Pipeline

A modular pipeline for object detection and image segmentation.

## Quick Start

### Detection Only (Recommended)

```bash
# Run with default parameters
python run_detection.py

# Run with custom image and queries
python run_detection.py --image path/to/image.jpg --queries "person" "car"
```

### Full Pipeline (Requires SAM2)

```bash
# Run with default parameters
python run_pipeline.py

# Run with custom image and queries
python run_pipeline.py --image path/to/image.jpg --queries "person" "car"
```

## Repository Organization

```
project_root/
├── pipeline/                       # Main pipeline package
│   ├── modules/                    # Pipeline components
│   │   ├── detection.py            # Object detection (Owlv2) 
│   │   ├── segmentation.py         # Image segmentation (SAM2)
│   │   ├── visualization.py        # Visualization utilities
│   │   └── utils.py                # Common utilities
│   ├── pipeline.py                 # Full pipeline with detection + segmentation
│   └── pipeline_detection.py       # Detection-only pipeline
├── detect_only.py                  # Simple detection script
├── run_detection.py                # Script to run detection pipeline
├── run_pipeline.py                 # Script to run full pipeline
├── TUTORIAL.md                     # Detailed tutorial
├── DETECTION_ONLY.md               # Guide for detection-only usage
└── requirements.txt                # Python dependencies
```

## Command Line Options

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

## Troubleshooting

1. **SAM2 Import Errors**: Use the detection-only version (`run_detection.py`)
2. **MPS/CUDA Issues**: The detection module uses CPU by default for better compatibility
3. **Model Download Problems**: Check your internet connection

## Documentation

- `TUTORIAL.md`: Complete tutorial on using and extending the pipeline
- `DETECTION_ONLY.md`: Guide focused on the detection-only version
- `README_DETECTION.md`: Detailed documentation for the detection pipeline 