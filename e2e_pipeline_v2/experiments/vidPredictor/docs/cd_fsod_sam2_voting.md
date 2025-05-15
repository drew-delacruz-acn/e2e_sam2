# CDFSOD_SAM2_VotingPipeline

## Overview
A video object detection and segmentation pipeline that fuses CD-FSOD detections and SAM2 mask propagation using an object-centric voting strategy. The goal is to improve temporal consistency and correct short-term misclassifications by assigning the most frequent label to each object across all frames where it is present (as indicated by SAM2 mask propagation).

---

## Pipeline Steps

### 1. **Input Preparation**
- **Frames:** Directory of video frames for each scene.
- **CD-FSOD Detections:** JSON files per frame with bounding boxes, labels, and scores.
- **SAM2 Masks:** Segmentation masks for each object, propagated across frames.

### 2. **Object Tracking and Mask Propagation**
- For each scene:
  - Track objects using CD-FSOD detections (associate detections across frames).
  - For each detected object, use SAM2 to propagate its mask across frames, producing a span of frames where the object is present.

### 3. **Object-Centric Voting**
- For each object instance:
  1. **Collect all frames** where the SAM2 mask exists (inclusion over exclusion).
  2. **Gather all predicted labels** (from CD-FSOD and/or mask propagation) for these frames.
  3. **Assign the most frequent label** (majority vote) as the object's label for all frames in the span.
  4. **Update per-frame predictions** to reflect this voted label.

### 4. **Edge Case Handling**
- **Ties:** If two or more labels are equally frequent, use a tie-breaker (e.g., label from the first frame, or highest average confidence).
- **Unusable Masks:** Only exclude masks that are completely unusable (e.g., empty or corrupted). Otherwise, include all masks in the process.
- **Drift:** If mask propagation drifts to a new object, consider splitting the span (advanced, optional).

### 5. **Output**
- Save the temporally-smoothed, corrected predictions and masks for each object and frame.
- Log corrections and any tie-breaks for review.

---

## Pseudocode

```python
for scene in scenes:
    # 1. Load frames, detections, and masks
    frames = load_frames(scene)
    detections = load_cdfsod_detections(scene)
    masks = run_sam2_and_propagate(scene, detections)

    # 2. Track objects (associate detections across frames)
    tracked_objects = track_objects(detections)

    # 3. For each object, perform voting
    for obj_id, obj_track in tracked_objects.items():
        # Get all frames where SAM2 mask exists (inclusion over exclusion)
        mask_frames = [f for f in obj_track.frames if mask_exists(masks, obj_id, f)]
        if not mask_frames:
            continue
        # Gather all labels for these frames
        labels = [get_label(detections, obj_id, f) for f in mask_frames]
        # Majority vote
        voted_label = most_frequent_label(labels)
        # Tie-breaker if needed
        if is_tie(labels):
            voted_label = tie_breaker(labels, detections, mask_frames)
        # Update all frames in span
        for f in mask_frames:
            update_prediction(obj_id, f, voted_label, masks[obj_id][f])

    # 4. Save results
    save_corrected_predictions(scene, tracked_objects, masks)
```

---

## Notes
- This approach assumes reliable mask propagation from SAM2. If mask drift is a concern, consider additional checks (e.g., mask overlap with previous frames).
- The pipeline can be extended to weight votes by confidence or mask quality if needed.
- All corrections and tie-breaks should be logged for transparency and debugging. 