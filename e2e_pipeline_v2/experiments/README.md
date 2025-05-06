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
python quantify_blurriness.py --input path/to/your/video.mp4 --input_type video
```

Basic usage (directory of videos):
```bash
python quantify_blurriness.py --input path/to/your/directory/ --input_type video
```

Basic usage (directory of image frames):
```bash
python quantify_blurriness.py --input path/to/your/frames_directory/ --input_type frames
```

With performance optimizations (works for video and frames):
```bash
python quantify_blurriness.py --input path/to/your/video.mp4 --input_type video --resize 0.25 --skip 5
python quantify_blurriness.py --input path/to/your/frames_directory/ --input_type frames --resize 0.5 --skip 2
```

Enable foreground extraction (works for video and frames):
```bash
python quantify_blurriness.py --input path/to/your/video.mp4 --input_type video --bg_remove
python quantify_blurriness.py --input path/to/your/frames_directory/ --input_type frames --bg_remove
```

Visualize foreground mask generation (first valid frame/image only):
```bash
python quantify_blurriness.py --input path/to/your/frames_directory/ --input_type frames --bg_remove --visualize_mask
```

All options:
```
--input       Path to video file, directory of videos, OR directory of image frames (Required)
--input_type  Specify if input is 'video' or 'frames' (Required)
--resize      Resize factor (e.g., 0.5 for half size) [Default: 1.0]
--skip        Process every N frames/images [Default: 1]
--save_path   Custom path to save results plot (only used for single video input)
--method      Threshold method (percentile, stddev, fixed) [Default: percentile]
--bg_remove   Enable foreground extraction based on edge density
--stakeholder Generate stakeholder-friendly visualizations (single video only)
--confidence  Include confidence scores in output
--visualize_mask Save a visualization of the foreground mask creation steps for one frame/image (requires --bg_remove)
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

**When processing a directory of videos (`--input_type video`):**

- `aggregated_results.json`: A JSON file containing results for all processed videos. Keys are video filenames.
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

**When processing a directory of frames (`--input_type frames`):**

- `aggregated_results.json`: A JSON file containing results for all processed image frames. Key is the input directory name.
  ```json
  {
    "input_directory_name": {
      "type": "frames",
      "settings": { "resize_factor": ..., "skip": ..., "bg_remove": ..., "method": ... },
      "thresholds": { "percentile": { ... }, "stddev": { ... }, "fixed": { ... } },
      "results": [
        { 
          "index": 0, // Index in the sorted file list
          "filename": "frame_0001.png",
          "laplacian": ..., 
          "tenengrad": ..., 
          "fft": ..., 
          "is_blurry": { "percentile": false, "stddev": false, "fixed": false }
        },
        ...
      ],
      "confidence": [
        { "index": 0, "filename": "frame_0001.png", "confidence": 85, "leading_metric": "laplacian", "is_blurry": false }, // Optional
        ...
      ]
    }
  }
  ```
- Score distribution/sequence plots (`blur_plot_frames_{dir_name}_{timestamp}.png` and associated histograms) *are* generated. Note: the sequence plot's x-axis represents the image index in the sorted list, not time.
- **Other individual plots, CSVs, stakeholder reports, and sample image visualizations are not generated** in frames mode.
- Optional: `foreground_mask_visualization_img_{image_index}.png` if `--bg_remove` and `--visualize_mask` are used (saved once per run for the first image with a valid mask).

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