# End-to-End Modular Pipeline

This pipeline provides an end-to-end solution for:
1. Sampling frames from videos (or processing individual images)
2. Detecting and segmenting objects using SAM2
3. Generating embeddings for the detected objects
4. Saving all results to a structured JSON format

## Directory Structure

```
e2e_pipeline/
│
├── main.py               # Main pipeline orchestration
├── config.yaml           # Configuration file
│
├── modules/              # Core modules
│   ├── frame_sampling.py # Video frame sampling
│   ├── segmentation.py   # Object detection and segmentation
│   ├── embedding.py      # Embedding generation
│   └── serialization.py  # JSON serialization
│
└── test_scripts/         # Test scripts for each module
    ├── test_frame_sampling.py
    ├── test_segmentation.py
    ├── test_embedding.py
    ├── test_serialization.py
    └── test_full_pipeline.py
```

## Usage

### Configuration

Edit the `config.yaml` file to set up your pipeline:

```yaml
mode: "image"  # or "video"

video:
  video_path: "path/to/video.mp4"
  frame_sampling:
    method: "scene"  # Options: "scene", "uniform", "random", "sequential"
    params:
      threshold: 30.0

image:
  image_path: "path/to/image.jpg"

segmentation:
  queries: ["object1", "object2"]  # Text queries for object detection
  detection_threshold: 0.1
  segmentation_threshold: 0.5
  results_dir: "results"

embedding:
  model_types: ["clip"]  # Options: "clip", "vit", "resnet50"
  device: "cuda"  # or "cpu"

serialization:
  output_json: "results/crop_embeddings.json"
```

### Running the Pipeline

Run the full pipeline:

```bash
python -m e2e_pipeline.main --config e2e_pipeline/config.yaml
```

### Testing Individual Modules

Test frame sampling:
```bash
python -m e2e_pipeline.test_scripts.test_frame_sampling --video_path path/to/video.mp4
```

Test segmentation:
```bash
python -m e2e_pipeline.test_scripts.test_segmentation --image_path path/to/image.jpg --queries object1 object2
```

Test embedding generation:
```bash
python -m e2e_pipeline.test_scripts.test_embedding --crops_json path/to/metadata.json
```

Test serialization:
```bash
python -m e2e_pipeline.test_scripts.test_serialization --embeddings_json path/to/embeddings.json
```

Test the full pipeline:
```bash
python -m e2e_pipeline.test_scripts.test_full_pipeline
```

## Output

The pipeline generates several outputs:

1. Cropped images of detected objects in `results/crops/`
2. Metadata for each crop in `results/{image_name}_metadata.json`
3. Final embeddings with all metadata in `results/crop_embeddings.json`

The final JSON contains entries with:
- Crop path
- Original image path
- Bounding box coordinates
- Text query that detected the object
- Detection confidence score
- Embeddings for each requested model 