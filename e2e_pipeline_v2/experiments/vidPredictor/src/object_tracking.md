# Object Tracking System

This document explains the object tracking system implemented in `object_tracker.py`. The system uses a hybrid approach that combines spatial positioning (IoU) and visual appearance similarity (embeddings) to track objects across video frames and distinguish between multiple instances of the same object class.

## Core Tracking Concepts

The tracking system integrates two key metrics:

1. **Intersection over Union (IoU)** - A spatial metric that measures the overlap between bounding boxes
2. **Embedding Similarity** - A visual appearance metric that uses deep learning features to determine if objects look similar

By combining these metrics, the system can solve challenging tracking scenarios:

- Track objects that move (position changes but appearance stays similar)
- Distinguish multiple instances of the same object class (similar appearance but different positions)
- Re-identify objects after occlusion or when they leave and re-enter the frame

## How the Tracking Works

### Initialization and Parameters

The tracking system uses several important parameters:

```python
def __init__(self, iou_weight=0.5, emb_weight=0.5, match_threshold=0.4, 
             min_iou_threshold=0.1, min_emb_threshold=0.5,
             new_instance_max_iou=0.1):
```

- `iou_weight` (0.5): Weight given to IoU when calculating matching scores
- `emb_weight` (0.5): Weight given to embedding similarity when calculating matching scores
- `match_threshold` (0.4): Minimum combined score required to match a detection to an existing object
- `min_iou_threshold` (0.1): Minimum IoU required for a potential match
- `min_emb_threshold` (0.5): Minimum embedding similarity required for a potential match
- `new_instance_max_iou` (0.1): Maximum IoU to consider a new instance of the same class

### Tracking Process

For each new frame:

1. **Extract Embeddings for New Detections**
   - For each detected object, crop the region of interest
   - Pass through a pre-trained CNN (ResNet50) to extract a feature vector (embedding)
   - This embedding captures the visual appearance of the object

2. **Match Detections to Existing Objects**
   - Group existing tracked objects by class
   - For each detection, compare with existing objects of the same class
   - Calculate two similarity metrics:
     - IoU with the last known position
     - Cosine similarity between feature embeddings
   - Combine these metrics with a weighted average

3. **Apply Thresholds**
   - Apply minimum thresholds for both IoU and embedding similarity
   - Only consider matches where both metrics meet minimum requirements
   - This prevents matches based on only position or only appearance

4. **Assign Optimal Matches**
   - Sort potential matches by combined score (descending)
   - Assign each detection to its best matching object, ensuring one-to-one matching
   - Update matched objects with new position and embedding (using a moving average)

5. **Handle New Instances**
   - For unmatched detections, determine if they're new objects or new instances of existing classes
   - Calculate maximum IoU and embedding similarity with existing objects of the same class
   - Consider a detection as a new instance if:
     - It has low spatial overlap with existing instances (IoU ≤ new_instance_max_iou)
     - Or it has sufficiently different appearance despite the same class label

### Cosine Similarity and Embeddings

The system uses cosine similarity to compare embeddings:

```python
emb_sim = cosine_similarity([det_embedding], [obj_data["embedding"]])[0][0]
```

Cosine similarity measures the cosine of the angle between two vectors, ranging from -1 (opposite) to 1 (identical). For object tracking:

- Values close to 1 indicate very similar visual appearance
- The embeddings are extracted using a deep CNN model (ResNet50 with classification head removed)
- These embeddings capture high-level visual features that are robust to small variations in lighting, pose, etc.

When an object is matched, its embedding is updated using a moving average to adapt to appearance changes:

```python
self.tracked_objects[obj_id]["embedding"] = (
    0.7 * self.tracked_objects[obj_id]["embedding"] + 
    0.3 * det_embedding
)
```

### Intersection over Union (IoU)

IoU is calculated between boxes to measure spatial overlap:

```python
def calculate_iou(self, box1, box2):
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes overlap
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate areas
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    union = area1 + area2 - intersection
    
    return intersection / union
```

IoU ranges from 0 (no overlap) to 1 (perfect overlap):
- High IoU (>0.5) suggests the same object in similar position
- Low IoU (<0.1) suggests different objects or significant movement

## Handling Multiple Instances of the Same Class

The key challenge addressed by this system is distinguishing between multiple instances of the same object class (e.g., two people wearing identical "Loki's Armor").

When a new detection doesn't match any existing object, the system explicitly checks if it should be a new instance:

```python
# Check if this is a new instance of an existing class
is_new_instance = True
max_emb_sim = 0.0
max_iou = 0.0

# Look at all existing objects of the same class
for obj_id, obj_data in self.tracked_objects.items():
    if obj_data["class"] == label:
        # Calculate IoU and embedding similarity
        # ...
        
        # If significant overlap or very high similarity with any existing instance,
        # this might not be a new instance
        if iou > self.new_instance_max_iou and emb_sim > self.min_emb_threshold:
            is_new_instance = False
            break
```

This allows the system to:
1. Maintain the same ID for objects that move/change pose (moderate IoU, high embedding similarity)
2. Create new IDs for new instances of the same class (low IoU, potentially high embedding similarity)

## Practical Examples

### Scenario 1: Same Object Moving

For an object that moves between frames:
- IoU might decrease as the object moves (e.g., IoU = 0.3)
- Embedding similarity remains high (e.g., emb_sim = 0.9)
- Combined score = 0.5 * 0.3 + 0.5 * 0.9 = 0.6
- This exceeds the match_threshold of 0.4, so the object maintains the same ID

### Scenario 2: Different Instances of Same Class

For two "Loki's Armor" instances worn by different people:
- IoU would be very low or zero (objects in different positions)
- Embedding similarity might be high (similar appearance)
- If IoU < min_iou_threshold, no match is considered
- System creates a new object ID for the second instance
- Logs indicate: "Created new instance X of (Loki's Armor) - distinct from existing instances"

### Scenario 3: Similar Objects in Similar Positions

For similar objects appearing in similar positions:
- Both IoU and embedding similarity would be high
- The object with the highest combined score gets matched
- Other objects might be considered new instances
- The system uses `new_instance_max_iou` to determine when spatial separation is sufficient

## Performance Considerations

This hybrid approach balances several trade-offs:

- **Computational Efficiency**: Computing embeddings for each detection adds computation but provides crucial appearance information
- **Accuracy vs. Speed**: More sophisticated matching logic improves tracking accuracy but increases complexity
- **Parameter Tuning**: The system is configurable through multiple parameters that can be tuned for specific scenarios

## Summary

The object tracking system uses a sophisticated approach that combines spatial positioning (IoU) and visual appearance (embeddings) to:

1. Track objects across frames even when they move
2. Distinguish between multiple instances of the same object class
3. Adapt to appearance changes through embedding updates
4. Maintain object identity through occlusions and reappearances

By balancing these spatial and visual cues, the system can handle challenging tracking scenarios more effectively than approaches that rely on just one type of information. 