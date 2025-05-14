# CD-FSOD Frame-by-Frame Processing Implementation Plan

## Overview

This document outlines a plan to modify the existing pipeline to support frame-by-frame processing of CD-FSOD detections, similar to how OWLv2 detector works. This will enable the pipeline to process all detections in each frame rather than only first appearances and reappearances.

## Implementation Steps

### ✅ Step 1: Create Extended CD-FSOD Detector Class

**Description:**
Instead of creating a new class, we modified the existing `CDFSODDetector` class to directly support frame-by-frame processing.

**Completed Actions:**
- [x] Modified the existing `CDFSODDetector.detect()` method to return all detections from the frame's JSON file
- [x] Maintained the detection format matching OWLv2 output format (boxes, scores, labels)
- [x] Ensured text query filtering is implemented
- [x] Added confidence threshold filtering in detect method
- [x] Maintained proper frame index extraction from various input types

**Implementation Details:**
- Instead of creating a new class, the existing `CDFSODDetector` was modified
- The `detect` method now returns all detections for a frame directly from `self.detections_by_frame`
- The original first appearance and reappearance tracking remains internally intact for reference
- Custom confidence threshold support was added to the `detect` method

### ✅ Step 2: Update Detector Factory Method

**Description:**
Update the `_create_detector` method in `ObjectTrackingPipeline` class to support the modified detector behavior.

**Completed Actions:**
- [x] Updated documentation in the factory method to explain the new detection behavior
- [x] Added explanatory comments for the CD-FSOD detector creation
- [x] Maintained support for the original `cd_fsod` detector type with the new behavior
- [x] Ensured proper parameter passing to constructor
- [x] Kept backward compatibility in the overall pipeline architecture

**Implementation Details:**
- Documentation updated to indicate that CD-FSOD now behaves like OWLv2, processing all detections in a frame
- Explanatory comments added to clarify that first appearances and reappearances are still tracked internally
- No actual code changes were needed since we modified the existing class rather than creating a new one

### ✅ Step 3: Update Command-Line Interface

**Description:**
Since we modified the existing detector class directly, this step was simplified.

**Completed Actions:**
- [x] Updated the documentation in `run_cd_fsod.py` to reflect the new behavior
- [x] Removed the now-redundant `--show_all_detections` flag
- [x] Modified the detection type display to indicate "ALL DETECTIONS" with appropriate modifiers
- [x] Updated the user-facing messages to make the new behavior clear

**Implementation Details:**
- Removed `--show_all_detections` flag since all detections are now returned by default
- Updated visual indicators in output images to show "ALL DETECTIONS" with additional context
- Added explanatory print statement: "NOTE: Detector now processes ALL detections in each frame"

### Step 4: Verify Object Tracker Compatibility

**Description:**
Review and potentially enhance the `ObjectTracker` class to ensure it effectively handles frame-by-frame tracking using IoU and embeddings.

**Acceptance Criteria:**
- [ ] `update_tracks` method effectively matches detections to existing tracks
- [ ] IoU calculation properly implemented
- [ ] Embedding similarity comparison properly implemented
- [ ] Track creation and updating logic correctly maintains object identity
- [ ] Deactivation of tracks after absence works correctly
- [ ] Returns dictionary mapping object IDs to current boxes

### Step 5: Add Documentation

**Description:**
Update documentation and comments to explain the new processing mode and its differences from the original mode.

**Acceptance Criteria:**
- [x] Code comments added to explain the updated detector behavior
- [x] Comments in relevant pipeline methods explaining behavior with different detector types
- [ ] README updated to document the new behavior
- [ ] Example command added to documentation

### Step 6: Testing Plan

**Description:**
Define tests to verify the correct operation of the new processing mode.

**Acceptance Criteria:**
- [ ] Test script created to compare outputs between the old and new behavior
- [ ] Verification that all detections in JSON files are processed
- [ ] Tracking consistency check to ensure objects maintain consistent IDs
- [ ] Performance measurement to assess any impact on processing speed

## Usage Examples

### Standalone CD-FSOD Processing

The `run_cd_fsod.py` script provides a way to run the CD-FSOD detector independently:

```bash
# Basic usage - processes all detections in JSON files
python run_cd_fsod.py --json_dir "/path/to/detections" --confidence 0.2

# With visualization - processes and visualizes all detections
python run_cd_fsod.py --json_dir "/path/to/detections" --frames_dir "/path/to/frames" --visualize --confidence 0.2

# Filter by specific object classes
python run_cd_fsod.py --json_dir "/path/to/detections" --frames_dir "/path/to/frames" --visualize --queries "person,car" --confidence 0.2

# Show detailed tracking information
python run_cd_fsod.py --json_dir "/path/to/detections" --frames_dir "/path/to/frames" --visualize --show_track_info --confidence 0.2
```

### Full Pipeline with SAM2 Integration

To run the full object tracking pipeline with SAM2 segmentation:

```bash
# Basic usage with CD-FSOD detector
python -m src.object_tracking_pipeline \
  --frames-dir "/path/to/frames" \
  --text-queries "person" "car" "dog" \
  --detector cd_fsod \
  --cd-fsod-path "/path/to/detections" \
  --sam2-checkpoint "/path/to/sam2.pth" \
  --sam2-config "/path/to/config.yaml" \
  --output-dir "./results"

# With separate object processing (helps with memory issues on complex videos)
python -m src.object_tracking_pipeline \
  --frames-dir "/path/to/frames" \
  --text-queries "person" "car" "dog" \
  --detector cd_fsod \
  --cd-fsod-path "/path/to/detections" \
  --sam2-checkpoint "/path/to/sam2.pth" \
  --sam2-config "/path/to/config.yaml" \
  --output-dir "./results" \
  --separate-objects

# With custom confidence threshold
python -m src.object_tracking_pipeline \
  --frames-dir "/path/to/frames" \
  --text-queries "person" "car" "dog" \
  --detector cd_fsod \
  --cd-fsod-path "/path/to/detections" \
  --sam2-checkpoint "/path/to/sam2.pth" \
  --sam2-config "/path/to/config.yaml" \
  --output-dir "./results" \
  --confidence 0.25
```

## Current Status

- **Steps Completed**: Steps 1, 2, and 3 have been completed
- **Next Steps**: Verify object tracker compatibility, complete documentation, and implement testing plan
- **Major Changes**: 
  - Instead of creating a new detector class, the existing `CDFSODDetector` class was modified
  - The detector now processes all detections in each frame by default
  - First appearances and reappearances are still tracked internally for reference

## Expected Benefits

1. **Processing Flexibility**: All detections are processed in every frame, providing more comprehensive detection results
2. **Consistent Interface**: Same pipeline works for both OWLv2 and CD-FSOD with similar behavior
3. **Enhanced Tracking**: Better object continuity with frame-by-frame processing
4. **Minimal Code Changes**: Reuses existing pipeline architecture with targeted modifications

## Remaining Considerations

1. **Testing**: Comprehensive testing is needed to ensure tracking quality is maintained with the new behavior
2. **Documentation**: Update README and other documentation to reflect the changes
3. **Performance**: Verify there are no significant performance impacts when processing all detections 