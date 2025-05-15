# CDFSOD_SAM2_VotingPipeline: Implementation Roadmap

## Overview
A video object detection and segmentation pipeline that fuses CD-FSOD detections and SAM2 mask propagation using an object-centric voting strategy. The goal is to improve temporal consistency and correct short-term misclassifications by assigning the most frequent label to each object across all frames where it is present (as indicated by SAM2 mask propagation).

## Implementation Roadmap (Iterative Approach)

### Phase 1: Setup & Code Reuse (Acceptance Criteria: New Pipeline Script Established)
1. **Create new script file** at the same level as `video_segmentation_pipeline.py`:
   ```
   cd_fsod_sam2_voting_pipeline.py
   ```

2. **Import necessary modules** from original pipeline:
   ```python
   import argparse
   import os
   import sys
   import logging
   import time
   import json
   import re
   from pathlib import Path
   from datetime import datetime
   import torch
   import gc
   
   # Add the src directory to the Python path
   sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
   
   # Import core components from the original pipeline
   from video_segmentation_pipeline import (
       setup_logger, natural_sort_key, log_pipeline_progress, apply_detector_patches
   )
   from object_tracking_pipeline import ObjectTrackingPipeline
   from cd_fsod_detector import CDFSODDetector
   ```

3. **Create `main()` function** with argument parsing from original pipeline:
   - Reuse parser setup and add new voting-specific arguments
   - Use same input/output structure for directories
   - Include the `--separate-objects` flag which is critical for our use case

4. **Define a basic scene processing function**:
   - Reuse the scene directory scanning logic
   - Maintain the same logging patterns

### Phase 2: IoU-Based Tracking Implementation (Acceptance Criteria: Objects Tracked Solely with IoU)
1. **Modify object tracking** to use only IoU:
   ```python
   def create_tracking_pipeline(args):
       """Create tracking pipeline with IoU-only tracking"""
       return ObjectTrackingPipeline(
           owlv2_checkpoint=None,
           sam2_checkpoint=args.sam2_checkpoint,
           sam2_config=args.sam2_config,
           output_dir=args.output_dir,
           confidence_threshold=args.confidence,
           detector_type="cd_fsod",
           cd_fsod_path=args.detections_path,
           min_gap_frames=args.min_gap_frames,
           mask_quality_threshold=args.mask_quality_threshold,
           tracking_method="iou_only",  # New parameter
           iou_threshold=args.iou_threshold,  # New parameter
       )
   ```

2. **Add IoU threshold argument** to script:
   ```python
   parser.add_argument("--iou-threshold", type=float, default=0.5,
                      help="IoU threshold for object tracking")
   ```

3. **Test tracking** with sample scenes to confirm objects are tracked correctly.

### Phase 3: Voting Mechanism Implementation (Acceptance Criteria: Labels Consistent Across Object Instances)
1. **Implement voting function** to process each object after detection:
   ```python
   def apply_temporal_voting(pipeline, logger):
       """Apply majority voting to object labels across frames"""
       logger.info("Applying majority voting for consistent object labeling...")
       
       # Iterate through tracked objects
       for obj_id, obj_data in pipeline.tracked_objects.items():
           # Get all frames where object has a mask
           object_frames = get_frames_with_masks(obj_data)
           
           # Gather all labels
           labels = get_all_labels_for_object(obj_data, object_frames)
           
           # Apply majority voting
           voted_label = get_majority_label(labels)
           
           # Apply voted label to all frames
           apply_voted_label(obj_data, voted_label, object_frames)
   ```

2. **Add voting strategy selection**:
   ```python
   parser.add_argument("--voting-strategy", choices=["majority", "weighted"], 
                      default="majority", help="Voting strategy to use")
   ```

3. **Implement tie-breaking logic** for cases with equal votes.

### Phase 4: Filling in Missed Detections (Acceptance Criteria: Complete Object Trajectories)
1. **Implement detection filling function**:
   ```python
   def fill_missed_detections(pipeline, logger):
       """Fill in missed detections using propagated masks"""
       logger.info("Filling in missed detections from mask propagation...")
       
       for obj_id, obj_data in pipeline.tracked_objects.items():
           # Get all frames with masks but missing detections
           missed_frames = get_frames_missing_detections(obj_data)
           
           # Create detections for missed frames
           create_detections_from_masks(obj_data, missed_frames)
   ```

2. **Add parameter to control filling behavior**:
   ```python
   parser.add_argument("--fill-missed-detections", action="store_true",
                      help="Fill in missed detections using mask propagation")
   ```

### Phase 5: Output & Results (Acceptance Criteria: Proper Result Storage)
1. **Implement result saving** that mirrors the original pipeline:
   - Save detection JSONs with updated/voted labels
   - Save masks for filled-in detections
   - Record metadata about corrections made

2. **Add visualization options** to show before/after comparisons:
   ```python
   parser.add_argument("--visualize-corrections", action="store_true",
                      help="Generate visualizations of corrections made")
   ```

## Reused Code Sections (No Rewriting Needed)

### From `video_segmentation_pipeline.py`:
1. **Logger Setup**: Reuse `setup_logger()` function
2. **Directory Handling**: Reuse directory scanning and creation logic
3. **Progress Tracking**: Reuse `log_pipeline_progress()` function
4. **Memory Management**: Reuse memory clearing code for GPU
5. **Argument Parsing**: Follow similar structure, adding new arguments
6. **Natural Sorting**: Reuse `natural_sort_key()` function

## Acceptance Criteria for Final Implementation

1. **Tracking Accuracy**: Objects must be correctly tracked across frames using only IoU
2. **Label Consistency**: Each object should have a single, consistent label across its lifetime
3. **Detection Completeness**: Missed detections should be filled in where masks exist
4. **Output Structure**: Results should follow the same directory structure as the original pipeline
5. **Performance**: Processing time should be within 20% of the original pipeline
6. **Documentation**: Code should be well-documented with clear explanations for the voting mechanism

## Command-Line Usage Example

```bash
python cd_fsod_sam2_voting_pipeline.py \
  --frames-root "data/frames/" \
  --detections-root "data/detections_cdfsod/" \
  --output-root "results/voting_output" \
  --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --confidence 0.9 \
  --min-gap-frames 20 \
  --separate-objects \
  --text-queries "all" \
  --iou-threshold 0.5 \
  --voting-strategy "majority" \
  --fill-missed-detections
```
```