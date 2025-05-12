# CD-FSOD Detections Integration Plan

## 1. Understanding the Data

### 1.1 CD-FSOD Detection Format
- The CD-FSOD predictions are stored in numbered JSON files (0.json, 1.json, etc.)
- Each file contains an array of object detections with:
  - `coordinates`: Bounding box in [x1, y1, x2, y2] format
  - `label`: Object class (e.g., "TVA Monitor", "TVA Uniform", "Time Stick")
  - `confidence`: Detection confidence score

### 1.2 Current Pipeline Format
- The existing pipeline uses OWLv2 detector which outputs:
  - `boxes`: Bounding boxes (similar to coordinates in CD-FSOD)
  - `labels`: Object classes
  - `scores`: Confidence scores

## 2. Implementation Strategy

### 2.1 Create a CD-FSOD Adapter Module
1. **Create a new file**: `src/cd_fsod_detector.py`
2. **Implement a detector class**: `CDFSODDetector` with an interface similar to `OWLv2Detector`
3. **Core functionality**:
   - Load all JSON files from a specified directory during initialization
   - Process detections across all frames using a continuity-based approach
   - Identify the first frame where each unique object appears AND where it reappears after a significant gap
   - Implement the `detect()` method to return objects based on first appearances and reappearances
   - Apply confidence thresholds to filter low-confidence detections

### 2.2 Continuity-Based Object Identification Strategy

1. **Identify object continuity across frames**:
   - Group detections based on similarity of position, size, and class
   - Use IoU (Intersection over Union) to measure spatial similarity between detections
   - Track each unique object's presence through consecutive frames

2. **Detect First Appearances and Reappearances**:
   - **First Appearance**: Record when an object is first detected
   - **Reappearance**: Record when an object returns after being absent for more than `min_gap_frames`
   - Maintain a "last seen" frame counter for each unique object
   - When (current_frame - last_seen_frame) > min_gap_frames, treat as a reappearance

3. **Frame Mapping**:
   - Create a mapping between frame indices and all objects that either:
     - First appear in that frame
     - Reappear in that frame after a significant gap
   - This mapping will be used by the `detect()` method to return frame-specific results

### 2.3 Integration with Existing Pipeline

1. **Modify the object tracking pipeline**:
   - Add option to use CD-FSOD detections instead of OWLv2
   - Create a factory method to instantiate the appropriate detector
   - Ensure the pipeline can handle both detector types transparently

2. **Configuration Options**:
   - Add configuration parameters for:
     - JSON directory path
     - Confidence threshold specific to CD-FSOD detections
     - Class mapping (if CD-FSOD uses different class names)
     - IoU threshold for considering detections as the same object
     - **Min gap frames**: Minimum number of frames an object must be absent to count as a reappearance

## 3. Implementation Plan

### 3.1 Phase 1: CD-FSOD Adapter Implementation

1. **Create the CD-FSOD detector class**:
   ```python
   class CDFSODDetector:
       def __init__(self, json_dir, confidence_threshold=0.2, iou_threshold=0.5, 
                   min_gap_frames=10, label_mapping=None):
           # Load all JSON files and process them
           # Track object continuity across frames
           # Record first appearances and reappearances after gaps
           # Create mapping between frames and significant detection events
           
       def detect(self, image, text_queries, threshold=None):
           # Extract frame_idx from metadata or filename
           # Return objects that first appear or reappear in this frame
           # Format to match OWLv2Detector output (boxes, labels, scores)
   ```

2. **Implement continuity-based object tracking**:
   - Load all detections from JSON files
   - For each frame, compare detections with previous frame:
     - Match objects using IoU and class
     - Update "last seen" frame for each tracked object
     - Record new objects and reappearing objects
   - Handle edge cases like brief occlusions or detection flickering

3. **Create frame-to-detection mapping**:
   - For each frame, maintain a list of objects that:
     - First appear in that frame
     - Reappear after being absent for at least `min_gap_frames`
   - This enables the `detect()` method to return only significant detection events

### 3.2 Phase 2: Pipeline Integration

1. **Modify pipeline initialization**:
   - Add detector type as a configuration option
   - Create factory method to instantiate the appropriate detector
   - Update documentation to reflect new options

2. **Update the pipeline processing logic**:
   - Ensure the pipeline works with both detector types
   - Pass frame information to the detector for CD-FSOD to identify the correct frame
   - Maintain backward compatibility

3. **Configure label mapping**:
   - Create a configuration option for mapping CD-FSOD labels to existing labels
   - Handle cases where CD-FSOD detects classes not in the original set

### 3.3 Phase 3: Testing and Validation

1. **Create test cases**:
   - Unit tests for the CD-FSOD adapter and continuity tracking
   - Integration tests with the full pipeline
   - Performance comparison between OWLv2 and CD-FSOD
   - Tests for various gap sizes and reappearance scenarios

2. **Validation metrics**:
   - Compare object tracking performance
   - Compare segmentation quality with SAM2 propagation
   - Measure how well SAM2 handles reappearances
   - Measure processing time differences

3. **Documentation**:
   - Update user documentation with new options
   - Create examples for using CD-FSOD detections
   - Document any known limitations or differences

## 4. Advanced Features (Optional)

### 4.1 Adaptive Gap Threshold

- Implement dynamic adjustment of the minimum gap threshold based on:
  - Video frame rate
  - Object class (some objects might reappear more frequently)
  - Object motion characteristics

### 4.2 Reappearance Verification

- Implement additional verification for reappearances:
  - Check for visual similarity beyond just position and class
  - Consider trajectory prediction to distinguish similar objects
  - Use confidence scores to prioritize high-confidence reappearances

### 4.3 Visualization Tools

- Add visualization options to show:
  - First appearances with one color
  - Reappearances with another color
  - Object continuity through frames
  - Gaps where objects disappeared

### 4.4 Performance Optimization

- Implement efficient JSON loading and processing
- Add caching mechanisms to improve repeated runs
- Optimize memory usage for large video processing

## 5. Implementation Timeline

1. **Week 1**: Create CD-FSOD adapter with continuity-based object tracking
2. **Week 2**: Integrate with pipeline and implement configuration options
3. **Week 3**: Testing, validation, and performance optimization
4. **Week 4**: Documentation and advanced features implementation

## 6. Example Usage

```python
# Example configuration for using CD-FSOD detections with continuity tracking
config = {
    "detector_type": "cd_fsod",
    "cd_fsod_path": "/path/to/json/files",
    "confidence_threshold": 0.2,
    "iou_threshold": 0.5,  # For identifying same object across frames
    "min_gap_frames": 10,  # Minimum frames absent to count as reappearance
    "label_mapping": {
        "TVA Monitor": "monitor",
        "TVA Uniform": "uniform",
        "Time Stick": "time_stick"
    }
}

# Initialize pipeline with CD-FSOD detector
pipeline = ObjectTrackingPipeline(config)

# Process video as usual
pipeline.process_video(frames_dir="video_frames", text_queries=["monitor", "uniform"])
```

## 7. Implementation Notes

### 7.1 Handling Different Confidence Scales

CD-FSOD detections may use a different confidence scale than OWLv2. The implementation should:
- Analyze the typical confidence range in CD-FSOD detections
- Implement normalization if needed
- Allow configurable thresholds specific to CD-FSOD

### 7.2 Continuity Tracking Challenges

When tracking object continuity and reappearances:
- Handle brief detection failures without treating them as reappearances
- Distinguish between multiple similar objects of the same class
- Account for object appearance changes over time
- Balance between too-sensitive and too-insensitive gap thresholds

### 7.3 SAM2 Integration Considerations

- Test how well SAM2 can propagate masks through continuous sequences
- Evaluate SAM2's ability to handle reappearances of the same object
- Consider providing SAM2 with additional context during reappearances
- Compare performance when using all detections vs. first-appearance-only approach

### 7.4 Performance Considerations

- Preprocess CD-FSOD detections during initialization to avoid repeated work
- Create efficient data structures for quick frame-based lookups
- Consider memory usage when loading all detections for long videos with many objects
