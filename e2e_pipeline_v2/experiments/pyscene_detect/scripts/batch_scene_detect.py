#!/usr/bin/env python3
"""batch_scene_detect.py
Process all videos in a folder, extract frames, and detect scene boundaries.

Example:
    python batch_scene_detect.py /path/to/videos --output /path/to/output --fps 30 \
           --detector adaptive --sigma 0.3 --min-scene-len 15

Positional arguments
--------------------
videos_folder       The directory containing video files (.mp4, .mov, etc.)

Optional arguments
------------------
--output PATH       Output directory for frames and scene information (default: ./output)
--fps FLOAT         Frames-per-second to use for extraction (default: 30)
--ext STR           Video file extensions to process (default: mp4,mov,avi)
--frame-format STR  Format for extracted frames (jpg or png, default: jpg)
--detector NAME     adaptive | content | threshold (default: adaptive)
--threshold FLOAT   Content/threshold detector cut-score (default: 27)
--sigma FLOAT       Adaptive detector rolling-avg factor (default: 0.33)
--min-scene-len N   Minimum scene length in frames (default: 15)
--skip-existing     Skip videos that already have a scenes.csv file
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil
import csv

def extract_frames(video_path, frames_dir, fps, frame_format='jpg'):
    """Extract frames from a video using ffmpeg."""
    os.makedirs(frames_dir, exist_ok=True)
    
    # Build the ffmpeg command
    frame_pattern = os.path.join(frames_dir, f"%d.{frame_format}")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality
        "-start_number", "0",
        frame_pattern
    ]
    
    print(f"[INFO] Extracting frames from {video_path}")
    subprocess.run(cmd, check=True)
    
    # Count how many frames were extracted
    frames = list(frames_dir.glob(f"*.{frame_format}"))
    print(f"[INFO] Extracted {len(frames)} frames to {frames_dir}")
    return len(frames)

def run_scene_detection(frames_dir, output_csv, fps, detector="adaptive", 
                       threshold=27, sigma=0.33, min_scene_len=15, frame_format='jpg'):
    """Run scene detection on the extracted frames."""
    # Get the path to the scene_detect_from_frames.py script
    script_dir = Path(__file__).parent
    scene_detect_script = script_dir / "scene_detect_from_frames.py"
    
    # Build command to run the scene detection script
    cmd = [
        sys.executable, str(scene_detect_script),
        str(frames_dir),
        "--fps", str(fps),
        "--ext", f".{frame_format}",
        "--detector", detector,
        "--threshold", str(threshold),
        "--sigma", str(sigma),
        "--min-scene-len", str(min_scene_len),
        "--output", str(output_csv)
    ]
    
    print(f"[INFO] Running scene detection using {detector} detector")
    subprocess.run(cmd, check=True)
    return output_csv

def organize_by_scenes(video_name, frames_dir, scenes_csv, output_dir, frame_format='jpg'):
    """Organize frames into scene folders based on the CSV output."""
    scenes_dir = output_dir / video_name / "scenes"
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Read the scenes CSV
    scenes = []
    with open(scenes_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            scene_id, start_frame, end_frame = int(row[0]), int(row[1]), int(row[2])
            scenes.append((scene_id, start_frame, end_frame))
    
    # Create scene directories and copy frames
    for scene_id, start_frame, end_frame in scenes:
        scene_dir = scenes_dir / f"scene_{scene_id:03d}"
        os.makedirs(scene_dir, exist_ok=True)
        
        # Copy frames to scene directory
        for frame_num in range(start_frame, end_frame + 1):
            src_file = frames_dir / f"{frame_num}.{frame_format}"
            dst_file = scene_dir / f"{frame_num}.{frame_format}"
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
        
        print(f"[INFO] Copied frames {start_frame}-{end_frame} to {scene_dir}")
    
    return scenes_dir

def process_video(video_path, output_dir, fps, detector, threshold, 
                 sigma, min_scene_len, frame_format='jpg', skip_existing=False):
    """Process a single video: extract frames, detect scenes, organize by scenes."""
    video_name = video_path.stem
    video_output_dir = output_dir / video_name
    frames_dir = video_output_dir / "frames"
    scenes_csv = video_output_dir / "scenes.csv"
    
    # Create output directory
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Skip if scenes.csv already exists and skip_existing is True
    if skip_existing and scenes_csv.exists():
        print(f"[INFO] Skipping {video_path} - scenes.csv already exists")
        return
    
    # Extract frames
    num_frames = extract_frames(video_path, frames_dir, fps, frame_format)
    
    if num_frames == 0:
        print(f"[WARNING] No frames extracted from {video_path}")
        return
    
    # Run scene detection
    run_scene_detection(
        frames_dir, scenes_csv, fps, detector, 
        threshold, sigma, min_scene_len, frame_format
    )
    
    # Organize frames by scenes
    if scenes_csv.exists():
        organize_by_scenes(video_name, frames_dir, scenes_csv, output_dir, frame_format)
    else:
        print(f"[WARNING] No scenes detected in {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch process videos for scene detection")
    parser.add_argument("videos_folder", type=Path, help="Directory containing video files")
    parser.add_argument("--output", type=Path, default="./output", 
                        help="Output directory for frames and scene data")
    parser.add_argument("--fps", type=float, default=30, 
                        help="Frames per second to extract")
    parser.add_argument("--ext", type=str, default="mp4,mov,avi", 
                        help="Comma-separated list of video extensions to process")
    parser.add_argument("--frame-format", type=str, default="jpg", choices=["jpg", "png"],
                        help="Format for extracted frames")
    parser.add_argument("--detector", default="adaptive", 
                        choices=["adaptive", "content", "threshold"],
                        help="Scene detection algorithm")
    parser.add_argument("--threshold", type=float, default=27, 
                        help="Threshold for content/threshold detectors")
    parser.add_argument("--sigma", type=float, default=0.33, 
                        help="Sigma factor for adaptive detector (0.0-1.0)")
    parser.add_argument("--min-scene-len", type=int, default=15, 
                        help="Minimum scene length in frames")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos that already have a scenes.csv file")
    
    args = parser.parse_args()
    
    # Validate videos folder
    if not args.videos_folder.is_dir():
        print(f"Error: {args.videos_folder} is not a directory")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get list of video files
    extensions = args.ext.split(",")
    video_files = []
    for ext in extensions:
        video_files.extend(args.videos_folder.glob(f"*.{ext}"))
    
    if not video_files:
        print(f"No video files with extensions {args.ext} found in {args.videos_folder}")
        return 1
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video
    for video_path in video_files:
        print(f"\n{'='*80}\nProcessing: {video_path}\n{'='*80}")
        try:
            process_video(
                video_path, args.output, args.fps, args.detector,
                args.threshold, args.sigma, args.min_scene_len,
                args.frame_format, args.skip_existing
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {e}")
    
    print("\nBatch processing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 