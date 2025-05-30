# Video Blur Detection Experiments

This directory contains experimental code for video analysis and processing, with particular focus on blur detection and frame quality assessment.

## 📹 Blur Detection (`quantify_blurriness.py`)

A comprehensive tool for detecting and analyzing blur in videos using multiple detection methods. This script implements state-of-the-art blur detection techniques and provides visual tools to help calibrate and understand blur thresholds.

### Features

- **Multiple Blur Detection Methods**
  - **Laplacian Variance**: Detects edges and measures their intensity
  - **Tenengrad (Sobel Gradient)**: Measures gradient magnitude
  - **FFT High-Frequency Analysis**: Examines frequency domain components
  - **Temporal Difference**: Compares each frame's sharpness to neighboring frames

- **Foreground Extraction (Optional)**
  - Uses edge density to identify and focus analysis on foreground objects, ignoring smooth backgrounds (`--bg_remove` flag).

- **Dynamic Thresholding**
  - **Percentile-Based**: Automatically sets thresholds at the 25th percentile of values
  - **Standard Deviation**: Sets thresholds at mean - 1.0*std_dev
  - **Fixed**: Uses pre-defined constant thresholds

- **Visualization Tools**
  - Time-series plots of all metrics with threshold lines
  - Histograms showing the distribution of blur scores
  - Sample frame visualization with metrics
  - Method comparison grids showing how different thresholds classify the same frames

- **Performance Optimizations**
  - Downsampling for large resolution videos
  - Frame-skipping to analyze only a subset of frames
  - Detailed performance metrics

### Usage

Basic usage (single video):
```bash
python quantify_blurriness.py --input path/to/your/video.mp4
```

Basic usage (directory of videos):
```bash
python quantify_blurriness.py --input path/to/your/directory/
```

With performance optimizations for large videos:
```bash
python quantify_blurriness.py --input path/to/your/video.mp4 --resize 0.25 --skip 5
```

Enable foreground extraction:
```bash
python quantify_blurriness.py --input path/to/your/video.mp4 --bg_remove
```

All options:
```
--input       Path to video file OR directory containing videos (Required)
--resize      Resize factor (e.g., 0.5 for half size) [Default: 1.0]
--skip        Process every N frames [Default: 1]
--save_path   Custom path to save results plot (only used for single video input)
--method      Threshold method (percentile, stddev, fixed) [Default: percentile]
--bg_remove   Enable foreground extraction based on edge density
--stakeholder Generate stakeholder-friendly visualizations (single video only)
--confidence  Include confidence scores in output
--visualize_mask Save a visualization of the foreground mask creation steps for one frame (requires --bg_remove)
```

### Output

Results are saved in a timestamped directory under `blur_results/analysis_{timestamp}/`.

**When processing a single video:**

- `blur_plot_{video_name}_{timestamp}.png`: Main metrics plot with thresholds
- `blur_plot_{video_name}_{timestamp}_histograms.png`: Distribution of blur scores
- `blur_data_{video_name}_{timestamp}.csv`: Raw data with all metrics and classifications
- `thresholds_{video_name}_{timestamp}.txt`: Detailed threshold values and statistics
- `sample_frames.png`: Visualization of sample frames with metrics
- `method_comparison/`: Comparative analysis of different threshold methods
  - `method_comparison_grid.png`: Grid comparing how each method classifies frames
  - `method_analysis.txt`: Analysis of method disagreements
- Individual frame categorizations in `method_{name}/{sharp|blurry}/` directories
- Optional: `confidence_scores_{video_name}_{timestamp}.csv` if `--confidence` or `--stakeholder` used
- Optional: Stakeholder summary files in `summary/` if `--stakeholder` used

**When processing a directory:**

- `aggregated_results.json`: A JSON file containing results for all processed videos. The structure is:
  ```json
  {
    "video1.mp4": {
      "scores": [ { "frame": 0, "laplacian": ..., "tenengrad": ..., "fft": ... }, ... ],
      "thresholds": { "percentile": { ... }, "stddev": { ... }, "fixed": { ... } },
      "blur_flags": { "percentile": [ true, false, ... ], "stddev": [ ... ], "fixed": [ ... ] },
      "confidence": [ { "frame": 0, "confidence": ..., "leading_metric": ..., "is_blurry": ... }, ... ] // Optional
    },
    "video2.avi": { ... }
  }
  ```
- Individual plots, visualizations, and CSV files are generally *not* generated for each video when processing a directory to avoid clutter.
- Optional: `foreground_mask_visualization_frame_{frame_idx}.png` if `--bg_remove` and `--visualize_mask` are used (saved once per run for the first frame with a valid mask).

### Interpreting Results

- **Higher values** for all metrics (Laplacian, Tenengrad, FFT) indicate **sharper** frames
- Frames are classified as blurry when all three metrics fall below their respective thresholds
- The temporal check flags frames that show a significant drop in sharpness compared to neighbors
- Method comparison helps determine which thresholding approach works best for your content

### Example Results

The method comparison grid shows how different thresholding approaches classify the same frames:
- Green borders/titles indicate frames classified as sharp
- Red borders/titles indicate frames classified as blurry
- Frames where methods disagree are particularly useful for threshold calibration

## Other Experiments

This directory also contains other experimental code:
- `segment_gemini.py`: Segmentation experiments with Gemini
- Other utility scripts for video analysis and processing

## Prerequisites

Required Python packages:
```
opencv-python
numpy
matplotlib
```

---

## 🔧 Function Overview & Workflow

This section provides a detailed overview of all the tools in the blurriness filter system, how they work together, and how to run them in sequence.

### Core Analysis Tools

#### 1. **`scripts/quantify_blurriness_mp4.py` - Video Blur Analysis**
**Purpose**: Analyzes blur in video files using multiple detection algorithms.

**Key Functions**:
- `laplacian_blur()` - Calculates edge intensity using Laplacian variance
- `tenengrad_blur()` - Measures gradient magnitude with Sobel operators  
- `fft_blur()` - Analyzes high-frequency components in frequency domain
- `extract_foreground_mask()` - Identifies foreground objects for focused analysis
- `analyze_video()` - Main analysis function that processes entire videos

**Usage**:
```bash
# Single video analysis
python scripts/quantify_blurriness_mp4.py --input path/to/video.mp4

# Batch processing directory of videos  
python scripts/quantify_blurriness_mp4.py --input path/to/video/directory/

# With performance optimizations
python scripts/quantify_blurriness_mp4.py --input video.mp4 --resize 0.5 --skip 2 --bg_remove
```

**Outputs**: 
- `blur_data_*.csv` - Per-frame metrics
- `blur_plot_*.png` - Visualization plots
- `aggregated_results.json` - Batch processing results

#### 2. **`scripts/quantify_bluriness_frames.py` - Image Sequence Analysis**
**Purpose**: Analyzes blur in sequences of individual image files (frames extracted from videos).

**Key Functions**:
- Same blur detection functions as `quantify_blurriness_mp4.py`
- `analyze_image_sequence()` - Processes folders of image files instead of video files
- `natural_sort_key()` - Ensures correct chronological ordering of frames

**Usage**:
```bash
# Analyze extracted frames from a scene
python scripts/quantify_bluriness_frames.py --input path/to/frames/directory/

# With background removal and mask visualization
python scripts/quantify_bluriness_frames.py --input frames/ --bg_remove --visualize_mask
```

**When to use**: When you have pre-extracted frames from videos rather than video files.

#### 3. **`scripts/blurriness_filter.py` - Detection Filtering**
**Purpose**: Filters object detection results based on blur analysis, removing detections from low-quality frames.

**Key Functions**:
- `blurriness_filter()` - Main filtering function with scene-specific thresholds
- Supports multiple filtering methods (percentile, standard deviation, fixed thresholds)
- Handles different blur metrics (laplacian, tenengrad, fft, boolean flags)

**Usage**:
```bash
# Filter detections using laplacian metric, keeping top 75% sharpest frames
python scripts/blurriness_filter.py \
  --json_path detections.json \
  --csv_path agg_results.csv \
  --metric laplacian \
  --method percentile \
  --threshold 75

# Filter using boolean blur flags
python scripts/blurriness_filter.py \
  --json_path detections.json \
  --csv_path agg_results.csv \
  --metric is_blurry_fixed \
  --method threshold \
  --boolean_threshold  # Keep blurry frames (default is keep non-blurry)
```

**Inputs**: 
- JSON file with object detections
- CSV file with blur metrics (from previous analysis steps)

**Outputs**: 
- Filtered JSON with only high-quality detections

### Analysis & Visualization Tools

#### 4. **`scripts/analysis_results.py` - Data Aggregation**
**Purpose**: Combines multiple blur analysis CSV files into a single aggregated dataset.

**Key Functions**:
- `combine_blur_analysis_csvs()` - Merges CSV files from different scenes/videos
- Automatically extracts scene names from file paths
- Handles error cases gracefully

**Usage**:
```bash
# Combine all CSV files in analysis directory
python scripts/analysis_results.py /path/to/analysis_directory/ --output combined_results.csv
```

**When to use**: After running blur analysis on multiple videos/scenes, to create a unified dataset.

#### 5. **`scripts/blur_analysis_dashboard.py` - Interactive Visualization**
**Purpose**: Streamlit-based interactive dashboard for exploring blur analysis results.

**Key Functions**:
- `find_scene_folder()` - Locates image folders for visualization
- `find_image_file()` - Handles different image file naming conventions  
- `main()` - Creates interactive Streamlit interface

**Features**:
- Interactive threshold adjustment with real-time feedback
- Side-by-side comparison of blurry vs. clear frames
- Distribution plots and statistics
- Scene-by-scene analysis

**Usage**:
```bash
# Launch interactive dashboard
streamlit run scripts/blur_analysis_dashboard.py
```

**Requirements**: Upload a combined CSV file and specify the base directory containing image folders.

#### 6. **`test_blurriness_filter.py` - Testing Suite**
**Purpose**: Unit tests for the filtering functionality.

**Usage**:
```bash
# Run tests
python -m pytest test_blurriness_filter.py -v
```

---

## 🔄 Complete Workflow

### **Option A: Video-based Analysis**

1. **Analyze videos**:
   ```bash
   python scripts/quantify_blurriness_mp4.py --input /path/to/videos/ --bg_remove
   ```

2. **Combine results** (if processing multiple videos):
   ```bash
   python scripts/analysis_results.py blur_results/analysis_timestamp/ --output combined_blur_data.csv
   ```

3. **Filter object detections**:
   ```bash
   python scripts/blurriness_filter.py \
     --json_path object_detections.json \
     --csv_path combined_blur_data.csv \
     --metric laplacian \
     --method percentile \
     --threshold 75
   ```

4. **Interactive analysis**:
   ```bash
   streamlit run scripts/blur_analysis_dashboard.py
   # Upload combined_blur_data.csv and set image directory
   ```

### **Option B: Image Sequence Analysis**

1. **Analyze extracted frames**:
   ```bash
   python scripts/quantify_bluriness_frames.py --input /path/to/scene_frames/
   ```

2. **Continue with steps 2-4 from Option A**

### **Quality Control Pipeline Integration**

The typical integration into a computer vision pipeline:

```
Raw Videos → Frame Extraction → Blur Analysis → Detection Filtering → Clean Results
     ↓              ↓                ↓               ↓               ↓
Video Files    Image Frames    CSV Metrics    Filtered JSON    High-Quality 
                                                              Detections Only
```

**Key Integration Points**:
- **Before object detection**: Use blur analysis to skip processing low-quality frames
- **After object detection**: Filter results to remove detections from blurry frames  
- **Quality assessment**: Use dashboard to validate thresholds and understand data quality

---

## 🎯 Choosing the Right Tool

- **`scripts/quantify_blurriness_mp4.py`**: Start here for video files
- **`scripts/quantify_bluriness_frames.py`**: Use when you have pre-extracted frames
- **`scripts/blurriness_filter.py`**: Essential for filtering object detection results
- **`scripts/analysis_results.py`**: Required when processing multiple videos/scenes
- **`scripts/blur_analysis_dashboard.py`**: Use for threshold calibration and result validation

## 📁 Directory Structure

```
blurriness_filter/
├── scripts/                           # Main analysis scripts
│   ├── quantify_blurriness_mp4.py    # Video blur analysis
│   ├── quantify_bluriness_frames.py  # Image sequence analysis  
│   ├── blurriness_filter.py          # Detection filtering
│   ├── analysis_results.py           # Data aggregation
│   └── blur_analysis_dashboard.py    # Interactive dashboard
├── test_blurriness_filter.py         # Unit tests
├── blur_results/                     # Output directory for results
├── blur_results_images/              # Output directory for image analysis
└── README.md                         # This documentation
```

## 📊 Understanding Output Metrics

- **Laplacian**: Edge detection strength (higher = sharper)
- **Tenengrad**: Gradient magnitude (higher = sharper) 
- **FFT**: High-frequency energy (higher = sharper)
- **is_blurry**: Boolean classification based on all metrics
- **Confidence**: How certain the classification is

Higher values indicate sharper, clearer images. The system classifies frames as blurry when **all three metrics** fall below their respective thresholds. 