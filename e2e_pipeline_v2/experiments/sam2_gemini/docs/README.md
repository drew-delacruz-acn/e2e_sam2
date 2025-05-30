# SAM2 Gemini: End-to-End Object Detection, Segmentation & Embedding Pipeline

## Overview

The SAM2 Gemini system is an end-to-end computer vision pipeline that combines:
- **Text-based object detection** using OWLv2 (Open World Localization v2)
- **High-precision segmentation** using SAM2 (Segment Anything Model 2)
- **Multi-model embedding generation** using CLIP, ViT, and ResNet50
- **Comprehensive result organization** with visualizations and metadata

## System Architecture

```
Input: Video frames or images
    ↓
[OWLv2 Detection] → Bounding boxes for objects matching text queries
    ↓
[SAM2 Segmentation] → Precise masks for detected objects
    ↓
[Embedding Generation] → Feature vectors using multiple models
    ↓
Output: Organized results with crops, masks, embeddings, and visualizations
```

## Files Overview

### Core Scripts
- **`sam2_vid_predict.py`** - Standalone OWLv2 detection for video frames
- **`segment_gemini.py`** - Main pipeline script for segmentation and embedding generation

### Supporting Modules
- **`modules/embedding/`** - Multi-model embedding generation system

## Prerequisites

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic dependencies
pip install torch torchvision transformers opencv-python numpy matplotlib Pillow PyYAML tqdm
```

### 2. SAM2 Installation

```bash
# Clone and install SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

### 3. Download Model Checkpoints

```bash
# Create checkpoints directory in project root
mkdir -p checkpoints

# Download SAM2 models (choose based on your needs)
# Large model (best performance)
wget -O checkpoints/sam2.1_hiera_large.pt [SAM2_LARGE_CHECKPOINT_URL]

# Base+ model (good balance)
wget -O checkpoints/sam2.1_hiera_base_plus.pt [SAM2_BASE_PLUS_CHECKPOINT_URL]
```

### 4. Download Config Files

```bash
# Ensure you have the SAM2 config files
mkdir -p configs/sam2.1
# Copy config files from the SAM2 repository
cp sam2/configs/sam2.1/* configs/sam2.1/
```

## Usage

### Method 1: Object Detection Only (sam2_vid_predict.py)

Use this for finding objects in video frames without segmentation.

```bash
cd e2e_pipeline_v2/experiments/sam2_gemini

# Basic usage - detect objects in a single frame
python sam2_vid_predict.py \
  --frames-dir /path/to/video/frames \
  --frame-idx 0 \
  --text-prompt "hammer, tool, object"

# List available frames first
python sam2_vid_predict.py \
  --frames-dir /path/to/video/frames \
  --list-frames

# Advanced usage with custom threshold
python sam2_vid_predict.py \
  --frames-dir /path/to/video/frames \
  --frame-idx 50 \
  --text-prompt "magical cube, glowing object, tesseract" \
  --threshold 0.1 \
  --device cuda
```

**Key Arguments:**
- `--frames-dir`: Directory containing video frames
- `--frame-idx`: Which frame to process (0-based)
- `--text-prompt`: Comma-separated list of objects to detect
- `--threshold`: Detection confidence threshold (default: 0.1)
- `--device`: Device to use (cuda/mps/cpu)

### Method 2: Full Pipeline (segment_gemini.py)

Use this for complete processing: detection → segmentation → embeddings.

#### A. Single Image Processing

```bash
# Process a single image with automatic segmentation
python segment_gemini.py \
  --image /path/to/image.jpg \
  --output_dir results \
  --models clip vit resnet50

# Process with detection-based segmentation
python segment_gemini.py \
  --image /path/to/image.jpg \
  --detections_dir /path/to/detection/jsons \
  --output_dir results \
  --models clip vit
```

#### B. Directory Processing

```bash
# Process all images in a directory
python segment_gemini.py \
  --input_dir /path/to/images \
  --output_dir results \
  --models clip vit resnet50

# Process nested video directories recursively
python segment_gemini.py \
  --input_dir /path/to/video/folders \
  --recursive \
  --detections_dir /path/to/detections \
  --output_dir results
```

#### C. Using Pre-existing Bounding Boxes

```bash
# Use bounding boxes from JSON file
python segment_gemini.py \
  --image /path/to/image.jpg \
  --use_bbox_json /path/to/bboxes.json \
  --output_dir results

# Use full image bounds (segment entire image)
python segment_gemini.py \
  --image /path/to/image.jpg \
  --use_full_image_predictor \
  --output_dir results
```

### Key Arguments for segment_gemini.py

**Input Options (choose one):**
- `--image`: Process a single image
- `--input_dir`: Process all images in directory
- `--recursive`: Process nested directories

**Segmentation Modes (choose one):**
- Default: Automatic mask generator
- `--use_full_image_predictor`: Use full image bounds
- `--use_bbox_json`: Use bounding boxes from JSON
- `--detections_dir`: Use detection-based segmentation

**Configuration:**
- `--output_dir`: Where to save results (default: output_results)
- `--models`: Embedding models to use (clip, vit, resnet50)
- `--min_area`: Minimum segment area in pixels (default: 1000)
- `--extensions`: File extensions to process (default: .jpg, .jpeg, .png)
- `--use_points` / `--no_points`: Use/don't use foreground points with bboxes

## Output Structure

The pipeline creates organized results:

```
output_results/
└── image_name/
    ├── segments/                    # Individual object segments
    │   ├── image_name_0.png
    │   └── image_name_1.png
    ├── padded_segments/             # Centered 224x224 versions
    │   ├── padded_image_name_0.png
    │   └── padded_image_name_1.png
    ├── embeddings/                  # Feature vectors
    │   └── image_name_embeddings.json
    └── image_name_segmentation_overlay.png  # Visualization
```

### Detection Results (sam2_vid_predict.py)
```
results/
└── image_name-timestamp/
    ├── visualizations/
    │   └── owlvit_detections.png    # Bounding boxes visualization
    ├── detections/
    │   └── model_name/
    │       ├── object1_0.95_abc123.png  # Crops with score and ID
    │       └── object2_0.87_def456.png
    └── metadata.json                # Detection metadata
```

## Input Data Formats

### Detection JSON Format
If using `--detections_dir`, each image should have a corresponding JSON:

```json
[
  {
    "coordinates": [x1, y1, x2, y2],
    "label": "hammer"
  },
  {
    "coordinates": [x1, y1, x2, y2],
    "label": "tool"
  }
]
```

### Bounding Box JSON Format
If using `--use_bbox_json`:

```json
{
  "image_name1": [x1, y1, x2, y2],
  "image_name2": [[x1, y1, x2, y2], [x1, y1, x2, y2]]
}
```

## Common Workflows

### 1. Video Analysis Pipeline

```bash
# Step 1: Extract frames from video (using your preferred tool)
ffmpeg -i video.mp4 -vf fps=1 frames/frame_%04d.jpg

# Step 2: Detect objects in key frames
python sam2_vid_predict.py \
  --frames-dir frames \
  --frame-idx 0 \
  --text-prompt "target objects"

# Step 3: Run full segmentation and embedding pipeline
python segment_gemini.py \
  --input_dir frames \
  --detections_dir detection_results \
  --output_dir final_results \
  --models clip vit resnet50
```

### 2. Object Tracking Preparation

```bash
# Process all frames in a video directory
python segment_gemini.py \
  --input_dir /path/to/video/frames \
  --detections_dir /path/to/detections \
  --output_dir tracking_ready \
  --models clip vit \
  --recursive
```

### 3. Ground Truth Comparison

```bash
# Generate embeddings for ground truth objects
python segment_gemini.py \
  --input_dir ground_truth_images \
  --output_dir ground_truth_embeddings \
  --models clip vit resnet50

# Process test images
python segment_gemini.py \
  --input_dir test_images \
  --detections_dir detections \
  --output_dir test_results \
  --models clip vit resnet50
```

## Troubleshooting

### Common Issues

1. **SAM2 Import Errors**
   ```bash
   # Make sure SAM2 is installed correctly
   cd sam2 && pip install -e .
   ```

2. **Missing Model Checkpoints**
   ```bash
   # Download checkpoints to correct location
   ls checkpoints/  # Should show .pt files
   ls configs/      # Should show .yaml files
   ```

3. **CUDA/MPS Issues**
   ```bash
   # Force CPU if GPU issues occur
   python segment_gemini.py --force-cpu ...
   ```

4. **Memory Issues**
   ```bash
   # Use smaller batch sizes or process fewer images at once
   # Or use a smaller SAM2 model (base instead of large)
   ```

5. **Import Path Issues**
   ```bash
   # Make sure you're running from the project root
   cd /path/to/e2e_sam2
   python e2e_pipeline_v2/experiments/sam2_gemini/segment_gemini.py ...
   ```

### Debug Mode

Enable detailed logging:

```bash
python segment_gemini.py --debug --image test.jpg --output_dir debug_results
```

## Configuration Tips

### For Better Detection:
- Lower `--threshold` (e.g., 0.05) for more sensitive detection
- Use more descriptive text prompts
- Try multiple related terms: "hammer, tool, mallet, hitting tool"

### For Better Segmentation:
- Increase `--min_area` to filter small segments
- Use `--use_points` for more precise segmentation with bounding boxes
- Try different SAM2 model sizes based on your needs

### For Performance:
- Use `--force_cpu` if GPU memory is limited
- Process smaller batches of images
- Use fewer embedding models if speed is critical

## Model Information

### Embedding Models Used:
- **CLIP**: `openai/clip-vit-base-patch32` - Good for general object understanding
- **ViT**: `google/vit-base-patch16-224` - Vision transformer for detailed features
- **ResNet50**: Pre-trained on ImageNet - Classical CNN features

### Detection Models:
- **OWLv2**: `google/owlv2-base-patch16` - Open-vocabulary object detection

### Segmentation Models:
- **SAM2**: Various sizes available (tiny, small, base+, large)

## Notes for Team Handoff

1. **Dependencies**: The system requires specific versions of PyTorch, transformers, and SAM2
2. **Model Downloads**: Models are downloaded automatically on first use but require internet
3. **Memory Usage**: Large models require significant GPU memory; use CPU for testing
4. **File Paths**: Always run from project root to avoid import issues
5. **Results**: The system generates comprehensive outputs - check the JSON files for programmatic access

## Future Improvements

- Add support for video input directly (currently requires pre-extracted frames)
- Implement batch processing for better GPU utilization
- Add configuration files instead of command-line arguments
- Integrate with object tracking systems
- Add support for custom embedding models

## Contact

For questions about this system, refer to the original implementation in the `sam2_pipeline` directory which has additional modular components and test scripts.
