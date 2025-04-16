# End-to-End Modular Pipeline Plan

## 1. Directory Structure (Suggested)

```
e2e_pipeline/
│
├── main.py
├── config.yaml
│
├── modules/
│   ├── frame_sampling.py
│   ├── segmentation.py
│   ├── embedding.py
│   └── serialization.py
│
├── FrameSampling/         # (your existing)
├── embedding_gen/         # (your existing)
├── ... (other code)
```

## 2. Config Example (`config.yaml`)

```yaml
mode: "video"  # or "image"

video:
  video_path: "data/my_video.mp4"
  frame_sampling:
    method: "scene"
    params:
      threshold: 30.0

image:
  image_path: "data/my_image.jpg"

segmentation:
  queries: ["hammer"]
  detection_threshold: 0.1
  segmentation_threshold: 0.5
  results_dir: "results"

embedding:
  model_types: ["clip", "vit"]
  device: "cuda"

serialization:
  output_json: "results/crop_embeddings.json"
```

## 3. Main Pipeline (`main.py`)

```python
import yaml
from modules.frame_sampling import sample_frames_from_video
from modules.segmentation import segment_image
from modules.embedding import generate_embeddings_for_crops
from modules.serialization import save_embeddings_json

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config['mode'] == 'video':
        frame_paths = sample_frames_from_video(config['video'])
    elif config['mode'] == 'image':
        frame_paths = [config['image']['image_path']]
    else:
        raise ValueError("Invalid mode in config.")

    all_crop_infos = []
    for frame_path in frame_paths:
        crop_infos = segment_image(frame_path, config['segmentation'])
        all_crop_infos.extend(crop_infos)

    embeddings = generate_embeddings_for_crops(all_crop_infos, config['embedding'])
    save_embeddings_json(embeddings, config['serialization'])

if __name__ == "__main__":
    main("config.yaml")
```

## 4. Module Outlines

### A. Frame Sampling (`modules/frame_sampling.py`)

```python
def sample_frames_from_video(video_config):
    # Use VideoFrameSampler from FrameSampling
    # Save frames as images, return list of image paths
    # Use video_config['frame_sampling'] for method/params
    pass
```

### B. Segmentation (`modules/segmentation.py`)

```python
def segment_image(image_path, segmentation_config):
    # Use your pipeline to segment/crop the image
    # Return list of dicts: [{crop_path, original_image, ...}, ...]
    pass
```

### C. Embedding Generation (`modules/embedding.py`)

```python
def generate_embeddings_for_crops(crop_infos, embedding_config):
    # For each crop, generate embeddings using selected models
    # Return list of dicts: [{crop_path, model, embedding, ...}, ...]
    pass
```

### D. Serialization (`modules/serialization.py`)

```python
def save_embeddings_json(embeddings, serialization_config):
    # Save the embeddings and metadata to a JSON file
    pass
```

## 5. Notes

- Each module should be independently testable.
- You can expand each function to handle logging, error handling, etc.
- The config can be extended for more options as needed.
