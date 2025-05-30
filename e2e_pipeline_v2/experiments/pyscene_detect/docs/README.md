# PySceneDetect Toolkit

A comprehensive toolkit for automatic scene detection in videos and image sequences using PySceneDetect with multiple algorithms and visualization capabilities.

## Overview

This toolkit provides multiple approaches for scene boundary detection:

1. **Video Processing**: Extract frames from videos and detect scene changes
2. **Frame Processing**: Analyze existing image sequences for scene boundaries  
3. **Batch Processing**: Process multiple videos or frame folders simultaneously
4. **Interactive Visualization**: Streamlit app for parameter tuning and result visualization
5. **Programmatic API**: Python module for integration into other projects

## Directory Structure

```
pyscene_detect/
├── scripts/                     # All Python scripts
│   ├── batch_scene_detect.py    # Video processing pipeline
│   ├── batch_frame_folders.py   # Batch frame processing
│   ├── scene_detect_from_frames.py # Core detection engine
│   ├── scene_detector.py        # Python API module
│   ├── scene_visualizer.py      # Interactive web app
│   └── pyscene_detect.py        # Alternative implementation
├── docs/                        # Documentation
│   └── README.md               # This file
└── batch_scene_detect/          # Example processed results
    └── [scene_folders]/         # Organized scene outputs
```

## Features

- **Multiple Detection Algorithms**: Adaptive, Content-based, and Threshold detection
- **Batch Processing**: Handle multiple videos or frame directories at once
- **Flexible Output**: CSV reports, organized frame directories, and visualizations
- **Parameter Tuning**: Interactive web interface for experimenting with detection settings
- **High Quality**: Uses PySceneDetect library for robust scene detection

## Installation

### Requirements

Ensure you have Python 3.7+ and the following dependencies:

```bash
pip install opencv-python>=4.8.0
pip install scenedetect>=0.6.0
pip install streamlit>=1.32.0
pip install matplotlib>=3.3.0
pip install tqdm>=4.50.0
pip install pandas
pip install numpy
```

Or install from the project requirements:
```bash
pip install -r ../../../requirements.txt
```

### Additional Dependencies

For video processing, you'll also need FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Quick Start

### 1. Process a Single Video

```bash
cd scripts
python batch_scene_detect.py /path/to/video_folder --output ./results --fps 24 --detector adaptive
```

### 2. Process Existing Frame Sequences

```bash
cd scripts
python batch_frame_folders.py /path/to/parent_folder --output ./results --fps 24 --detector adaptive
```

### 3. Interactive Visualization

```bash
cd scripts
streamlit run scene_visualizer.py
```

### 4. Single Frame Directory

```bash
cd scripts
python scene_detect_from_frames.py /path/to/frames --fps 24 --output scenes.csv
```

## Scripts Documentation

### 1. `scripts/batch_scene_detect.py`

**Purpose**: Process multiple videos, extract frames, and detect scene boundaries.

**Usage**:
```bash
cd scripts
python batch_scene_detect.py <videos_folder> [options]
```

**Key Features**:
- Extracts frames from videos using FFmpeg
- Runs scene detection on extracted frames
- Organizes frames by detected scenes
- Generates CSV reports with timing information

**Example**:
```bash
cd scripts
python batch_scene_detect.py ./videos \
    --output ./output \
    --fps 24 \
    --detector adaptive \
    --threshold 27 \
    --sigma 0.33 \
    --min-scene-len 15 \
    --frame-format jpg \
    --skip-existing
```

**Output Structure**:
```
output/
├── video1/
│   ├── frames/          # All extracted frames
│   ├── scenes.csv       # Scene boundaries and timing
│   └── scenes/          # Frames organized by scene
│       ├── scene_001/
│       ├── scene_002/
│       └── ...
└── video2/
    └── ...
```

### 2. `scripts/batch_frame_folders.py`

**Purpose**: Process multiple folders containing pre-extracted frames.

**Usage**:
```bash
cd scripts
python batch_frame_folders.py <parent_folder> [options]
```

**Key Features**:
- Processes existing frame sequences
- No video extraction needed
- Batch processing of multiple frame directories
- Optional frame copying to scene folders

**Example**:
```bash
cd scripts
python batch_frame_folders.py ./frame_folders \
    --output ./output \
    --fps 24 \
    --frame-format jpg \
    --detector adaptive \
    --sigma 0.33 \
    --copy-frames
```

### 3. `scripts/scene_detect_from_frames.py`

**Purpose**: Core scene detection script for single frame directory.

**Usage**:
```bash
cd scripts
python scene_detect_from_frames.py <folder> --fps <fps> [options]
```

**Key Features**:
- Works with numbered frame sequences (0.jpg, 1.jpg, etc.)
- Three detection algorithms available
- Outputs CSV with scene boundaries and timing

**Detection Algorithms**:

- **Adaptive** (default): Adapts to content changes over time
  ```bash
  --detector adaptive --sigma 0.33
  ```

- **Content**: Based on HSV histogram differences
  ```bash
  --detector content --threshold 27.0
  ```

- **Threshold**: Simple threshold-based detection
  ```bash
  --detector threshold --threshold 12.0
  ```

**Example**:
```bash
cd scripts
python scene_detect_from_frames.py ./frames \
    --fps 24 \
    --ext .jpg \
    --detector adaptive \
    --sigma 0.33 \
    --min-scene-len 15 \
    --output scenes.csv
```

### 4. `scripts/scene_detector.py`

**Purpose**: Python module providing a simple API for programmatic use.

**Usage**:
```python
import sys
sys.path.append('./scripts')
from scene_detector import detect_scenes

# Detect scene boundaries
scene_cuts = detect_scenes(
    frames_dir="/path/to/frames",
    fps=24,
    detector="adaptive",
    adaptive_threshold=2.0,
    min_scene_len=15
)

print(f"Scene boundaries at frames: {scene_cuts}")
```

**API Parameters**:
- `frames_dir`: Path to directory with numbered frames
- `fps`: Frame rate for timing calculations
- `detector`: "adaptive", "content", or "threshold"
- `threshold`: Detection sensitivity (lower = more sensitive)
- `adaptive_threshold`: Adaptive detector parameter
- `min_scene_len`: Minimum scene length in frames
- `ext`: File extension (.jpg, .png)

### 5. `scripts/scene_visualizer.py`

**Purpose**: Interactive Streamlit web application for parameter tuning and visualization.

**Features**:
- Real-time parameter adjustment
- Visual timeline of detected scenes
- Frame-by-frame scene visualization
- Export capabilities for results
- Scene boundary highlighting

**Usage**:
```bash
cd scripts
streamlit run scene_visualizer.py
```

Then open http://localhost:8501 in your browser.

**Interface Components**:
- Scene directory selector
- Detection parameter controls
- Timeline visualization
- Scene summary table
- Frame grid display with boundary highlighting
- Export functionality

### 6. `scripts/pyscene_detect.py`

**Purpose**: Alternative scene detection implementation using custom histogram analysis.

**Features**:
- Custom HSV histogram-based detection
- Frame organization by scenes
- Independent of PySceneDetect library

**Usage**:
```bash
cd scripts
python pyscene_detect.py --input ./frames --output ./scenes --threshold 30.0
```

## Parameter Guidelines

### Detection Sensitivity

**Adaptive Detector**:
- `sigma` (0.0-1.0): Lower values = more sensitive to changes
- Recommended: 0.1-0.5 for sensitive detection, 0.5-1.0 for conservative

**Content Detector**:
- `threshold` (1-100): Lower values = more sensitive
- Recommended: 15-30 for most content, 30-50 for stable scenes

**Threshold Detector**:
- `threshold` (1-50): Lower values = more sensitive
- Recommended: 8-15 for most content

### Scene Length

- `min-scene-len`: Minimum frames per scene
- Recommended: 15-30 frames (0.5-1 second at 24fps)
- Lower values: More scene breaks, higher noise
- Higher values: Fewer scene breaks, may miss quick cuts

## Output Formats

### CSV Reports

Generated `scenes.csv` contains:
```csv
scene_id,start_frame,end_frame,start_time_sec,end_time_sec
1,0,43,0.000,1.792
2,43,90,1.792,3.750
3,90,150,3.750,6.250
```

### Scene Summary

Generated `scene_summary.csv` contains:
```csv
scene_name,start_frame,end_frame,frame_count
scene_001,0,43,44
scene_002,43,90,48
scene_003,90,150,61
```

## Example Workflows

### Workflow 1: Video Processing Pipeline

```bash
cd scripts

# 1. Process all videos in a folder
python batch_scene_detect.py ./input_videos \
    --output ./results \
    --fps 24 \
    --detector adaptive \
    --sigma 0.3

# 2. Results will be organized as:
# results/video1/scenes.csv
# results/video1/scenes/scene_001/
# results/video2/scenes.csv
# results/video2/scenes/scene_002/
```

### Workflow 2: Frame Analysis with Visualization

```bash
cd scripts

# 1. Process frame folders
python batch_frame_folders.py ./frame_folders \
    --output ./analysis \
    --copy-frames

# 2. Visualize results interactively
streamlit run scene_visualizer.py
# Navigate to generated scenes in web interface
```

### Workflow 3: Parameter Tuning

```bash
cd scripts

# 1. Start with visualization tool
streamlit run scene_visualizer.py

# 2. Experiment with parameters in web interface
# 3. Note optimal parameters for your content

# 4. Run batch processing with tuned parameters
python batch_scene_detect.py ./videos \
    --detector adaptive \
    --sigma 0.2 \
    --min-scene-len 20
```

### Workflow 4: Programmatic Integration

```python
import sys
sys.path.append('./scripts')
from scene_detector import detect_scenes
import csv

# Process multiple frame directories
frame_dirs = ["./frames1", "./frames2", "./frames3"]

for frames_dir in frame_dirs:
    # Detect scenes
    boundaries = detect_scenes(
        frames_dir=frames_dir,
        fps=24,
        detector="adaptive",
        adaptive_threshold=0.3
    )
    
    # Save results
    output_file = f"{frames_dir}_scenes.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scene_start_frame"])
        for boundary in boundaries:
            writer.writerow([boundary])
```

## Performance Considerations

### Processing Speed

- **Adaptive detector**: Fastest, good for real-time applications
- **Content detector**: Medium speed, most accurate for general content
- **Threshold detector**: Fastest, but less sophisticated

### Memory Usage

- Large frame sequences may require significant memory
- Consider processing in batches for very large datasets
- Use `--skip-existing` to resume interrupted batch jobs

### Accuracy vs Speed

- Lower `min-scene-len`: More accurate but potentially noisy
- Higher thresholds: Faster but may miss subtle scene changes
- Adaptive detector: Best balance for most content types

## Troubleshooting

### Common Issues

**No scenes detected**:
- Lower the detection threshold
- Check frame numbering (should start from 0 or 1)
- Verify frame format matches `--ext` parameter

**Too many scenes detected**:
- Increase detection threshold
- Increase `min-scene-len` parameter
- Try content detector instead of adaptive

**FFmpeg errors**:
- Ensure FFmpeg is installed and in PATH
- Check video file format compatibility
- Verify sufficient disk space for frame extraction

**Memory errors**:
- Process smaller batches
- Use lower resolution frames
- Increase system memory or use swap

### Frame Naming Requirements

Frames must be numbered sequentially:
- ✅ Good: `0.jpg, 1.jpg, 2.jpg, ...`
- ✅ Good: `0001.png, 0002.png, 0003.png, ...`
- ❌ Bad: `frame_a.jpg, frame_b.jpg, ...`
- ❌ Bad: `img001.jpg, img003.jpg, img005.jpg, ...` (gaps)

## Advanced Usage

### Custom Detection Parameters

For fine-tuned control, modify parameters in the scripts or use the programmatic API:

```python
import sys
sys.path.append('./scripts')
from scene_detector import detect_scenes

# Advanced adaptive detection
scenes = detect_scenes(
    frames_dir="./frames",
    fps=24,
    detector="adaptive",
    adaptive_threshold=1.5,      # How quickly to adapt
    min_content_val=12.0,        # Minimum change threshold
    window_width=3,              # Smoothing window
    min_scene_len=20             # Minimum scene duration
)
```

### Integration with Other Tools

The CSV outputs can be easily integrated with video editing tools, analysis pipelines, or other computer vision systems:

```python
import pandas as pd

# Load scene detection results
scenes_df = pd.read_csv("scenes.csv")

# Convert to video editing format
for _, row in scenes_df.iterrows():
    start_time = row['start_time_sec']
    end_time = row['end_time_sec']
    duration = end_time - start_time
    print(f"Scene {row['scene_id']}: {start_time:.3f}s - {end_time:.3f}s ({duration:.3f}s)")
```

## Recommended Development Workflow

### Phase 1: Ground Truth Collection

**Critical Step**: Before deploying this toolkit at scale, you should create a ground truth dataset to optimize parameters for your specific content.

1. **Manual Annotation**: 
   - Select 10-20 representative videos/frame sequences from your dataset
   - Manually identify and mark scene boundaries at frame-level precision
   - Document the rationale for each scene boundary decision
   - Create a standardized annotation format (CSV with start/end frames)

2. **Annotation Guidelines**:
   - Define what constitutes a "scene change" for your use case
   - Consider factors: camera cuts, location changes, significant action transitions
   - Be consistent across different annotators if using multiple people
   - Include edge cases and ambiguous boundaries in your ground truth

**Example Ground Truth Format**:
```csv
video_name,scene_id,start_frame,end_frame,boundary_type,confidence
video1,1,0,45,camera_cut,high
video1,2,45,120,location_change,high
video1,3,120,180,action_transition,medium
```

### Phase 2: Parameter Optimization

3. **Systematic Testing**:
   ```bash
   cd scripts
   
   # Test different detector combinations on ground truth
   for detector in adaptive content threshold; do
     for threshold in 15 20 25 30; do
       python scene_detect_from_frames.py ./ground_truth_frames \
         --fps 24 \
         --detector $detector \
         --threshold $threshold \
         --output results_${detector}_${threshold}.csv
     done
   done
   ```

