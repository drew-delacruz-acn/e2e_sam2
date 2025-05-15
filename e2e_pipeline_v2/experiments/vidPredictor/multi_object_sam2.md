# Multi-Object SAM2 Segmentation Pipeline Refactor Plan

## Overview
This document outlines a plan to adapt the current video segmentation pipeline to robustly support multi-object segmentation in a single pass (non-`--separate-objects` mode), using explicit, per-frame, per-object annotation logic inspired by the working notebook approach.

---

## 1. How the Code Works Now

### a. Detection Loading
- For each scene, all frames and detection JSONs are loaded.
- Each detection JSON contains bounding boxes and class labels for objects detected in that frame.

### b. Object Track Initialization
- The pipeline identifies the first frame where each unique object appears (and possibly reappears after a gap).

### c. Object Tracking: How Embeddings and IoU Are Used

The current pipeline uses a custom `ObjectTracker` that combines **bounding box overlap (IoU)** and **appearance embeddings** to track objects across frames and decide whether a detection is a new instance or matches an existing one.

#### Key Steps:

1. **Initialization (First Frame):**
   - Each detected object is assigned a unique ID.
   - The object's bounding box and an appearance embedding (from a feature extractor) are stored.

2. **Matching in Subsequent Frames:**
   - For each new detection, the tracker compares it to all existing tracked objects of the same class.
   - Two main metrics are computed:
     - **IoU (Intersection over Union):** Measures how much the new box overlaps with the last known box of a tracked object.
     - **Embedding Similarity:** Cosine similarity between the new detection's embedding and the tracked object's embedding.
   - A **combined score** is calculated as a weighted sum of IoU and embedding similarity.
   - The detection is matched to an existing object if:
     - The combined score exceeds a threshold.
     - Both IoU and embedding similarity are above their respective minimum thresholds.
   - If matched, the object's trajectory and embedding are updated (using a moving average for embeddings).

3. **Determining New Instances:**
   - If a detection cannot be matched to any existing object (i.e., combined score or thresholds not met), it is considered a **new instance**.
   - Additionally, even for the same class, if the IoU with all existing objects is below a certain maximum (e.g., `new_instance_max_iou`) and embedding similarity is low, a new object ID is assigned.
   - This prevents merging distinct objects of the same class that are spatially or visually separate.

#### Parameters Used:
- `iou_weight` and `emb_weight`: Control the influence of IoU vs. embedding similarity in the combined score.
- `min_iou_threshold` and `min_emb_threshold`: Minimum values for a match to be considered.
- `new_instance_max_iou`: Maximum IoU allowed to consider a detection as a new instance (prevents merging overlapping objects).

#### Summary Table:

| Metric         | Purpose                                      | How Used                                                      |
|----------------|----------------------------------------------|---------------------------------------------------------------|
| IoU            | Spatial overlap                              | High IoU = likely same object; low IoU = likely new instance  |
| Embedding Sim. | Visual/appearance similarity                 | High similarity = likely same object                          |
| Combined Score | Weighted sum of IoU and embedding similarity | Must exceed threshold for a match                             |

**This logic ensures that objects are only merged/tracked as the same instance if they are both spatially close and visually similar, and new IDs are assigned when a detection is sufficiently different in space or appearance.**

### d. Segmentation Initialization
- For each object, segmentation is initialized (e.g., by passing a bounding box to SAM2) **only at the first frame where the object is detected**.
- This is the only frame where an explicit annotation is provided to the segmentation model for that object.

### e. Mask Propagation
- After initialization, SAM2 propagates the mask for each object across subsequent frames.
- The model attempts to track and segment the object as it moves, appears, or disappears, using only the initial annotation as a reference.

### f. No Explicit Handling of Absent Objects
- If an object is not detected in a frame, the pipeline does not provide any negative or dummy annotation.
- The segmentation model may continue to propagate the mask, even if the object is no longer present, unless it is robust enough to stop on its own.

### g. Multi-Object Handling
- In default mode (without `--separate-objects`), the pipeline attempts to process all objects together, but this is unreliable for complex cases.
- With `--separate-objects`, each object is processed independently, which is more robust but less efficient.

---

## 2. How the Code Needs to Change (Non-Separate-Objects Mode)

### **Goal:**
Enable robust, explicit, per-frame, per-object annotation and segmentation, as in the notebook, for multi-object scenarios in a single pass.

### **Step-by-Step Plan:**

#### 1. Annotation Data Structure
- **Introduce or accept** a data structure (e.g., a list or dict) that specifies, for each frame and each object, the annotation (bounding box, points, or dummy/negative) to use.
- This should mirror the `objects` and `frameOccurences` structure from the notebook. **See the "Example: Per-Frame, Per-Object Annotation Logic (Notebook Style)" section below for a concrete implementation pattern.**

#### 2. Annotation Loop Logic
- **Modify the main processing loop** so that for each frame, for each object:
    - If an annotation is present, use the real annotation (box/points).
    - If not, provide a dummy/negative annotation (e.g., a box/point outside the image or a negative label).
- This loop should run for all frames and all objects, not just at first appearance. **See the example section below for a concrete implementation.**

#### 3. Segmentation Input Handling
- **Update the code that calls the segmentation model** so it can accept and process multiple annotations per frame (one for each object), including negatives/dummies.
- Ensure the model's state is updated for each object in each frame as per the annotation.

#### 4. Propagation Logic
- **Adjust propagation logic** so that explicit annotations guide segmentation in every frame.
- Propagation can still be used, but should be corrected or overridden by explicit annotations when provided.

#### 5. Handling Absent Objects
- **Explicitly send a negative/dummy annotation** for absent objects in each frame, to prevent the model from hallucinating masks.

#### 6. Batch Processing of Multiple Objects
- **Ensure the segmentation model and visualization code** can handle and display multiple objects' masks in the same frame.

#### 7. Input/Output Interface
- **Allow the pipeline to accept explicit per-frame, per-object annotations** (boxes/points/negatives) as input, or provide a way to generate them from detections.

#### 8. Testing and Validation
- **Test the new logic** with multiple objects appearing/disappearing in different frames, ensuring masks are only generated where and when they should be.

#### 9. Documentation and User Interface
- **Update documentation and CLI help** to explain the new annotation logic and how to use it.

---

## Example: Per-Frame, Per-Object Annotation Logic (Notebook Style)

To clarify the intended annotation and segmentation logic, here is a simplified example (adapted from the working notebook):

```python
objects = [
    {
        'objectName': 'scepter',
        'objectID': 1,
        'frameOccurences': [
            {'frameNum': 0, 'framePoints': [[500, 300], [800, 250]], 'pointTypes': [1, 1]}
        ]
    }
]

allFramesClasses = []
for objectClass in objects:
    for occurence in objectClass['frameOccurences']:
        if occurence['frameNum'] not in allFramesClasses:
            allFramesClasses.append(occurence['frameNum'])

def send_sam2_api(ann_frame_idx_VAR, ann_obj_id_VAR, pointsVar, labelsVar):
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx_VAR,
        obj_id=ann_obj_id_VAR,
        points=np.array(pointsVar, dtype=np.float32),
        labels=np.array(labelsVar, np.int32),
    )

for frameNum in allFramesClasses:
    for objectClass in objects:
        validOccurences = [w for w in objectClass['frameOccurences'] if w['frameNum'] == frameNum]
        if len(validOccurences):
            occ = validOccurences[0]
            send_sam2_api(frameNum, objectClass['objectID'], occ['framePoints'], occ['pointTypes'])
        else:
            # Send a dummy/negative annotation if object is absent
            send_sam2_api(frameNum, objectClass['objectID'], [[-1, -1]], [0])
```

**Key points illustrated:**
- The `objects` data structure holds all per-object, per-frame annotations.
- For each frame and each object, the code checks if an annotation exists:
    - If yes, it sends the real annotation to the segmentation model.
    - If not, it sends a dummy/negative annotation (e.g., a point at `[-1, -1]` with label `0`).
- This ensures every object is explicitly handled in every frame, preventing mask hallucination and supporting robust multi-object segmentation.

---

## 3. Key Files Likely to Require Changes
- `video_segmentation_pipeline.py` (main pipeline logic)
- `src/object_tracking_pipeline.py` (object tracking and segmentation logic)
- `src/sam2_wrapper.py` (SAM2 interface)
- (Possibly) detection modules if annotation generation is to be automated

---

## 4. Risks and Considerations
- **Performance:** Explicit per-frame annotation may be slower, but is more robust for multi-object scenarios.
- **Backward Compatibility:** Consider how to maintain support for the original propagation-based workflow.
- **User Input:** Decide whether to require explicit annotation input or to generate it from detections.

---

## 5. Next Steps
- Review and finalize the data structure for annotations.
- Map out the main processing loop changes.
- Prototype the new annotation logic in a test script before full integration.
- Incrementally refactor and test the pipeline, starting with the annotation and segmentation input logic.

---

*End of plan.* 