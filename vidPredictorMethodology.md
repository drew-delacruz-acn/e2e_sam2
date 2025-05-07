# Video Prediction Pipeline Methodology

## Overview

The video prediction pipeline is a comprehensive system for detecting, tracking, and segmenting objects in video sequences. It combines state-of-the-art models to achieve robust performance:

- **OWLv2** for text-prompted object detection
- **SAM2** (Segment Anything Model 2) for high-quality segmentation
- **Custom object tracker** using appearance embeddings and spatial consistency
- **ResNet50-based embedding extractor** for appearance modeling

## Architecture Components

The pipeline consists of four main components:

1. **Object Detection** - OWLv2 for text-prompted object detection
2. **Object Tracking** - Custom tracker with embedding-based identity preservation
3. **Instance Segmentation** - SAM2 for precise object masks
4. **Mask Propagation** - Temporal consistency of segmentation across frames

## Core Process Flow

### Initialization

```python
# Initialize components
self.detector = OWLv2Detector(device=device)
self.sam_wrapper = SAM2VideoWrapper(sam2_checkpoint, sam2_config, device=device)
self.tracker = ObjectTracker()
self.embedding_extractor = EmbeddingExtractor(device=device)
```

### First Frame Processing

For the first frame, the pipeline detects and initializes object tracking:

```python
# Detect objects in first frame
detections = self.detector.detect(
    image=first_frame,
    text_queries=text_queries,
    threshold=self.confidence_threshold
)

# Initialize object tracking
for i, (box, label, conf) in enumerate(zip(detections["boxes"], detections["labels"], detections["scores"])):
    # Extract embedding for this object
    embedding = self.embedding_extractor.extract(box_area)
    
    # Create a unique object ID
    object_id = self.next_id
    self.next_id += 1
    
    # Add box to SAM2 to get mask
    mask_logits = self.sam_wrapper.add_box(frame_idx=0, obj_id=object_id, box=box_coords)
    
    # Store this object with the ID
    self.tracked_objects[object_id] = {
        "id": object_id,
        "class": label,
        "first_detected": 0,
        "boxes": [box_coords],
        "embeddings": [embedding],
        "masks": [mask_logits],
        "last_seen": 0,
        "confidence": [conf]
    }
```

### Object Tracking Across Frames

The object tracker matches new detections with existing objects using both spatial and appearance cues:

```python
# Calculate IoU between two boxes
def calculate_iou(self, box1, box2):
    # ... IoU calculation ...
    return intersection / union

# Create matching scores based on IoU and embedding similarity
for obj_id, obj_data in self.tracked_objects.items():
    # Calculate IoU with last known position
    last_box = obj_data["trajectory"][-1][1]
    iou = self.calculate_iou(box, last_box)
    
    # Calculate embedding similarity
    emb_sim = cosine_similarity([det_embedding], [obj_data["embedding"]])[0][0]
    
    # Combined score
    combined_score = (self.iou_weight * iou) + (self.emb_weight * emb_sim)
    
    match_scores.append((obj_id, i, combined_score, iou, emb_sim))
```

### Segmentation with SAM2

SAM2 provides precise segmentation masks for each detected object:

```python
# Add a box prompt for an object
def add_box(self, frame_idx, obj_id, box):
    box_coords = np.array(box, dtype=np.float32)
    
    _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
        inference_state=self.inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=box_coords
    )
    
    # Return processed mask
    if len(out_mask_logits) > 0:
        return self.process_mask_logits(out_mask_logits[0])
    else:
        return None
```

### Mask Propagation

SAM2 propagates segmentation masks across video frames:

```python
# Propagate masks to all frames in the video
def propagate_masks(self, objects_to_track=None):
    video_segments = {}
    
    # Iterate through the propagation results
    for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
        # Filter objects if needed
        # ...
        
        # Store masks by object ID
        video_segments[out_frame_idx] = {
            obj_id: self.process_mask_logits(mask_logit)
            for obj_id, mask_logit in zip(filtered_obj_ids, filtered_mask_logits)
        }
        
    return video_segments
```

## How Multiple Objects of the Same Class are Differentiated

When multiple objects of the same class are present in a scene, the pipeline differentiates them through several mechanisms:

1. **Unique Object IDs**: Each detected object receives a unique ID regardless of class

2. **Appearance Embeddings**: ResNet50 features capture the unique visual appearance of each object instance:

```python
class EmbeddingExtractor:
    def __init__(self, device=None):
        # Load model and remove classification head
        model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        
    def extract(self, image_crop):
        # Extract features
        with torch.no_grad():
            features = self.model(tensor)
        return features.cpu().numpy().flatten()
```

3. **Hybrid Tracking**: Objects are matched across frames using both spatial overlap (IoU) and appearance similarity:

```python
# Combined score for matching
combined_score = (self.iou_weight * iou) + (self.emb_weight * emb_sim)
```

4. **Progressive Refinement**: Object embeddings are updated using a moving average to adapt to appearance changes:

```python
# Update with moving average
self.tracked_objects[obj_id]["embedding"] = (
    0.7 * self.tracked_objects[obj_id]["embedding"] + 
    0.3 * det_embedding
)
```

5. **Instance-Level Segmentation**: SAM2 provides individual segmentation masks for each object instance

## Usage Example

```python
# Initialize the pipeline
pipeline = ObjectTrackingPipeline(
    owlv2_checkpoint=args.owlv2_checkpoint,
    sam2_checkpoint=args.sam2_checkpoint,
    sam2_config=args.sam2_config,
    output_dir=args.output_dir,
    confidence_threshold=args.confidence
)

# Process video with specified object classes to detect
pipeline.process_video(
    frames_dir=args.frames_dir,
    text_queries=args.text_queries  # e.g., ["goat", "person", "car"]
)
```

The pipeline saves tracking results, segmentation masks, and visualizations for each processed frame, maintaining object identities throughout the video. 