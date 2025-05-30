# End-to-End Object Detection and Segmentation Pipeline

This pipeline combines OwlViT (or OwlViT-2) for text-based object detection with SAM2 (Segment Anything Model 2) for high-quality image segmentation.

## Features

- **Text-Based Object Detection**: Detect objects in images using natural language queries
- **High-Quality Segmentation**: Generate precise segmentation masks for detected objects
- **Two Segmentation Methods**:
  - Box-based segmentation: uses bounding boxes directly
  - Point-based segmentation: uses the center point of each box
- **Comprehensive Outputs**:
  - Detection visualizations
  - Individual object crops
  - Segmentation masks with confidence scores
  - Segmentation overlays on the original image
- **Configurable**: Easily customize thresholds, models, and save options

## Installation

1. Ensure you have the SAM2 repository installed:
   ```bash
   # Clone SAM2 if needed
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .
   cd ..
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision transformers opencv-python numpy matplotlib Pillow PyYAML
   ```

## Usage

### Command-Line Interface

The easiest way to use the pipeline is through the `run_e2e_pipeline.py` script:

```bash
python run_e2e_pipeline.py --image path/to/image.jpg --queries "chair" "table" "lamp"
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to config YAML file | `e2e_pipeline_v2/config.yaml` |
| `--image` | Path to input image | Required |
| `--queries` | Text queries for detection | Uses config if not specified |
| `--detection-threshold` | Detection confidence threshold | Uses config value |
| `--segmentation-threshold` | Segmentation confidence threshold | Uses config value |
| `--results-dir` | Directory to save results | Uses config value |
| `--model-type` | Type of detector model (`owlvit` or `owlvit2`) | Uses config value |
| `--force-cpu` | Force using CPU even if GPU is available | False |
| `--no-viz` | Disable visualization | False |
| `--no-save` | Disable saving results | False |

### Python API

You can also use the pipeline in your own code:

```python
from e2e_pipeline_v2.pipeline import DetectionSegmentationPipeline

# Initialize the pipeline
pipeline = DetectionSegmentationPipeline("e2e_pipeline_v2/config.yaml")

# Run the pipeline
results = pipeline.run(
    image_path="path/to/image.jpg",
    text_queries=["chair", "table", "lamp"],
    visualize=True,
    save_results=True
)

# Access results
detection_results = results["detection"]
point_segmentation = results["point_segmentation"]
box_segmentation = results["box_segmentation"]
output_dir = results["output_dir"]
```

## Configuration

The pipeline uses a YAML configuration file. Here's an example:

```yaml
# Detection configuration
detection:
  model_type: "owlvit2"  # Options: owlvit, owlvit2
  model_name: "google/owlv2-base-patch16"  # Model name from Hugging Face
  force_cpu: false  # Force CPU usage

# Segmentation configuration
segmentation:
  queries: ["hammer", "person", "chair"]  # Default text queries
  detection_threshold: 0.1  # Confidence threshold for detection
  segmentation_threshold: 0.5  # Threshold for segmentation
  results_dir: "results"  # Directory to save results
  model_config: "configs/sam2.1/sam2.1_hiera_l.yaml"  # SAM2 model config
  checkpoint: "checkpoints/sam2.1_hiera_large.pt"  # SAM2 checkpoint
```

## Output Structure

Results are saved with the following structure:

```
results/
└── {image_name}/
    ├── visualizations/
    │   └── detection_results.png  # Detection visualization
    ├── crops/
    │   ├── box_0.png  # Cropped object 1
    │   └── box_1.png  # Cropped object 2
    ├── segmentation/
    │   ├── point_segmentation/
    │   │   ├── point_seg_0_0_score_0.95.png  # Visualization
    │   │   └── point_seg_0_0_mask.png  # Binary mask
    │   └── box_segmentation/
    │       ├── box_seg_0_0_score_0.98.png  # Visualization
    │       └── box_seg_0_0_mask.png  # Binary mask
    └── detections/
        └── {model_name}/
            ├── object1_0.95_abcd1234.png  # Named by label and score
            └── object2_0.87_efgh5678.png
```

## Example Results

The pipeline generates several visualizations:

1. **Detection Results**: Shows bounding boxes with labels and confidence scores
2. **Segmentation Masks**: Binary masks for each detected object
3. **Segmentation Overlays**: Original image with colored mask overlay and contours

## Troubleshooting

- **SAM2 Import Errors**: Make sure SAM2 is properly installed with `pip install -e .` from the SAM2 directory
- **Memory Issues**: Try using a smaller SAM2 model (tiny or small) by changing the config
- **No Objects Detected**: Try lowering the detection threshold or using different text queries 

## Ground Truth and Embedding Comparison

### Ground Truth Generation

To generate ground truth embeddings and mapping for your reference objects:

1. Place your ground truth images in a directory (e.g., `ground_truth/ground_truth_images/`)
2. Run the ground truth processing script:

```bash
python e2e_pipeline_v2/scripts/process_ground_truth.py \
  --input_dir ground_truth/ground_truth_images \
  --output_dir ground_truth/ground_truth_embeddings \
  --create_mapping \
  --models vit resnet50
```

This will:
- Generate embeddings for each ground truth image using ViT and ResNet50 models
- Create a mapping file (`ground_truth_mapping.json`) that defines:
  - Object classes (derived from image filenames)
  - Similarity thresholds (default: 0.75)
  - Paths to images and their embeddings

### Embedding Comparison

To compare detected objects against ground truth:

```bash
python e2e_pipeline_v2/scripts/compare_embeddings.py \
  --ground_truth_dir ground_truth \
  --results_dir results \
  --object object_name \
  --output comparison_results.json
```

For example, to compare all detections of Loki's crown:
```bash
python e2e_pipeline_v2/scripts/compare_embeddings.py \
  --ground_truth_dir ground_truth \
  --results_dir results \
  --object loki_crown \
  --output loki_crown_comparison.json
```

```bash
python e2e_pipeline_v2/scripts/compare_embeddings.py \
 -- ground_truth_dir ground_truth \
 -- results_dir results \
  --output embedding_comparison_results.json
```

### Testing Methodology

The pipeline uses a multi-model approach for robust object comparison:

1. **Ground Truth Setup**:
   - Collect high-quality reference images of target objects
   - Name images descriptively (e.g., `loki_crown.jpg`, `time_stick.png`)
   - Generate embeddings using multiple models (ViT and ResNet50)
   - Create class mappings for consistent object categorization

2. **Embedding Generation**:
   - ViT (Vision Transformer):
     - Uses CLS token for image representation
     - Good at capturing global features and relationships
   - ResNet50:
     - Uses deep convolutional features
     - Strong at capturing local patterns and textures

3. **Similarity Computation**:
   - Calculates cosine similarity between detected objects and ground truth
   - Computes similarities for both models
   - Takes average similarity across models
   - Compares against threshold (default: 0.75)

4. **Results Analysis**:
   - Per-object similarity scores
   - Model-specific similarities
   - Average similarity across models
   - Threshold-based matching decisions
   - Detailed comparison reports in JSON format

5. **Best Practices**:
   - Use consistent lighting and angles for ground truth images
   - Include multiple variants of objects if appearance varies
   - Adjust similarity thresholds based on application needs
   - Consider model-specific weights if one performs better

### Output Format

The comparison results JSON includes:
```json
{
  "object_name": {
    "detection_id": "unique_id",
    "label": "detected_label",
    "score": detection_confidence,
    "similarities": {
      "vit": vit_similarity_score,
      "resnet50": resnet_similarity_score
    },
    "average_similarity": average_score,
    "exceeds_threshold": boolean,
    "crop_path": "path/to/detection/crop"
  }
}
```

This comprehensive testing approach ensures reliable object matching by:
- Using multiple model perspectives
- Considering both global and local features
- Providing detailed similarity metrics
- Supporting threshold-based decision making

python -m e2e_pipeline_v2.process_embeddings --image data/thor_hammer.jpeg --queries "hammer" --models clip vit resnet50 --output_dir results/embeddings_full --force_cpu