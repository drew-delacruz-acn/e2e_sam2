# CD-FSOD Frame-by-Frame Processing Implementation Plan

## Overview

This document outlines a plan to modify the existing pipeline to support frame-by-frame processing of CD-FSOD detections, similar to how OWLv2 detector works. This will enable the pipeline to process all detections in each frame rather than only first appearances and reappearances.

## Implementation Steps

### Step 1: Create Extended CD-FSOD Detector Class

**Description:**
Create a new detector class that extends the existing `CDFSODDetector` class but overrides the `detect` method to return all detections in a frame rather than just first appearances and reappearances.

**Acceptance Criteria:**
- [ ] New `CDFSODFrameDetector` class defined in `src/cd_fsod_detector.py`
- [ ] Class inherits from existing `CDFSODDetector`
- [ ] `detect` method overridden to return all detections from the frame's JSON file
- [ ] Detection format matches OWLv2 output format (boxes, scores, labels)
- [ ] Text query filtering implemented
- [ ] Confidence threshold filtering implemented
- [ ] Properly handles frame index extraction from various input types

### Step 2: Update Detector Factory Method

**Description:**
Modify the `_create_detector` method in `ObjectTrackingPipeline` class to support the new detector type.

**Acceptance Criteria:**
- [ ] Factory method updated to recognize `"cd_fsod_frame"` detector type
- [ ] New detector type creates an instance of `CDFSODFrameDetector`
- [ ] Required parameters properly passed to constructor
- [ ] Appropriate error message if required parameters are missing
- [ ] Original detector types (`"owlv2"` and `"cd_fsod"`) still work correctly

### Step 3: Update Command-Line Interface

**Description:**
Add a new command-line parameter to allow easy switching between the original and frame-by-frame CD-FSOD processing modes.

**Acceptance Criteria:**
- [ ] `--frame_by_frame` flag added to `run_cd_fsod.py` argument parser
- [ ] Flag documented in help text
- [ ] Main function updated to select correct detector type based on flag
- [ ] Default behavior (without flag) maintains backward compatibility

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
- [ ] Code comments added to explain the new detector class
- [ ] Comments in relevant pipeline methods explaining behavior with different detector types
- [ ] README updated to document the new `--frame_by_frame` option
- [ ] Example command added to documentation

### Step 6: Testing Plan

**Description:**
Define tests to verify the correct operation of the new processing mode.

**Acceptance Criteria:**
- [ ] Test script created to compare outputs with and without `--frame_by_frame`
- [ ] Verification that with `--frame_by_frame`, all detections in JSON files are processed
- [ ] Verification that without `--frame_by_frame`, only first appearances/reappearances are processed
- [ ] Tracking consistency check to ensure objects maintain consistent IDs
- [ ] Performance measurement to assess any impact on processing speed

## Expected Benefits

1. **Processing Flexibility**: Choose between processing only first appearances/reappearances or all detections in every frame
2. **Consistent Interface**: Same pipeline works for both OWLv2 and CD-FSOD with similar behavior
3. **Enhanced Tracking**: Better object continuity with frame-by-frame processing
4. **Minimal Code Changes**: Reuses existing pipeline architecture with targeted extensions

## Potential Risks and Mitigation

1. **Processing Overhead**: 
   - Risk: Processing all detections may increase memory and CPU usage
   - Mitigation: Add optional filtering by confidence threshold to reduce processing load

2. **Object ID Consistency**: 
   - Risk: Object IDs might not be consistent between original and frame-by-frame modes
   - Mitigation: Ensure robust tracking with both IoU and embeddings

3. **SAM2 Integration**: 
   - Risk: Increased number of objects may impact SAM2 performance
   - Mitigation: Add option to limit number of objects processed by SAM2 