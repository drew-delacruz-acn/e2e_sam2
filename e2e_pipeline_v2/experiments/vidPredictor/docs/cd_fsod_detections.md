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
   - Load JSON files from a specified directory
   - Match frame indices to JSON file names
   - Convert the JSON format to the expected detector output format
   - Apply confidence thresholds to filter low-confidence detections

### 2.2 Integration with Existing Pipeline

1. **Modify the object tracking pipeline**:
   - Add option to use CD-FSOD detections instead of OWLv2
   - Create a factory method to instantiate the appropriate detector
   - Ensure the pipeline can handle both detector types transparently

2. **Frame Mapping**:
   - Implement a mechanism to match video frames to the correct JSON file
   - Support different naming conventions or explicit mapping

3. **Configuration Options**:
   - Add configuration parameters for:
     - JSON directory path
     - Confidence threshold specific to CD-FSOD detections
     - Class mapping (if CD-FSOD uses different class names)

## 3. Implementation Plan

### 3.1 Phase 1: CD-FSOD Adapter Implementation

1. **Create the CD-FSOD detector class**:
   ```python
   class CDFSODDetector:
       # Constructor to accept path to JSON directory and configuration
       # Methods to load and cache JSON files
       # detect() method with same interface as OWLv2Detector
       # Utilities for filtering and format conversion
   ```

2. **Implement JSON loading and parsing**:
   - Efficient loading to avoid repeatedly reading the same files
   - Robust error handling for missing or malformed files
   - Support for different JSON naming patterns

3. **Create data conversion functions**:
   - Convert CD-FSOD format to pipeline format
   - Handle edge cases (empty detections, invalid boxes)
   - Normalize confidence scores if needed

### 3.2 Phase 2: Pipeline Integration

1. **Modify pipeline initialization**:
   - Add detector type as a configuration option
   - Create factory method to instantiate the appropriate detector
   - Update documentation to reflect new options

2. **Update the pipeline processing logic**:
   - Ensure the pipeline works with both detector types
   - Handle any detector-specific processing requirements
   - Maintain backward compatibility

3. **Configure label mapping**:
   - Create a configuration option for mapping CD-FSOD labels to existing labels
   - Handle cases where CD-FSOD detects classes not in the original set

### 3.3 Phase 3: Testing and Validation

1. **Create test cases**:
   - Unit tests for the CD-FSOD adapter
   - Integration tests with the full pipeline
   - Performance comparison between OWLv2 and CD-FSOD

2. **Validation metrics**:
   - Compare object tracking performance
   - Compare segmentation quality
   - Measure processing time differences

3. **Documentation**:
   - Update user documentation with new options
   - Create examples for using CD-FSOD detections
   - Document any known limitations or differences

## 4. Advanced Features (Optional)

### 4.1 Hybrid Detection Mode

- Implement an option to combine detections from both OWLv2 and CD-FSOD
- Create merging strategies (union, intersection, confidence-weighted)
- Add configuration to control merging behavior

### 4.2 Visualization Tools

- Add visualization options specific to CD-FSOD detections
- Create comparison visualizations between different detectors
- Implement confidence visualization for better analysis

### 4.3 Performance Optimization

- Implement batch processing for CD-FSOD files
- Add caching mechanisms to improve repeated runs
- Optimize memory usage for large video processing

## 5. Implementation Timeline

1. **Week 1**: Create CD-FSOD adapter and basic JSON loading
2. **Week 2**: Integrate with pipeline and implement configuration options
3. **Week 3**: Testing, validation, and performance optimization
4. **Week 4**: Documentation and advanced features implementation

## 6. Example Usage

```python
# Example configuration for using CD-FSOD detections
config = {
    "detector_type": "cd_fsod",
    "cd_fsod_path": "/path/to/json/files",
    "confidence_threshold": 0.2,
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

### 7.2 Handling Missing Frames

If CD-FSOD detections don't cover all frames:
- Implement interpolation options for missing frames
- Provide configuration to control interpolation behavior
- Add logging for frames without detections

### 7.3 Performance Considerations

- CD-FSOD files can be preloaded to avoid I/O during processing
- Consider memory usage for large videos
- Implement efficient filtering to handle the large number of low-confidence detections
