#!/usr/bin/env python3
"""batch_frame_folders.py
Process all subfolders containing frame images and detect scene boundaries.

Example:
    python batch_frame_folders.py /path/to/parent_folder --output /path/to/output --fps 30 \
           --detector adaptive --sigma 0.3 --min-scene-len 15

Positional arguments
--------------------
parent_folder       The parent directory containing subdirectories with frame images

Optional arguments
------------------
--output PATH       Output directory for scene information (default: ./output)
--fps FLOAT         Frames-per-second to use for timecode calculations (default: 30)
--frame-format STR  Format of the frame images to look for (jpg or png, default: jpg)
--pattern STR       Filename pattern to match frames (default: "%d" - numbered frames)
--detector NAME     adaptive | content | threshold (default: adaptive)
--threshold FLOAT   Content/threshold detector cut-score (default: 27)
--sigma FLOAT       Adaptive detector rolling-avg factor (default: 0.33)
--min-scene-len N   Minimum scene length in frames (default: 15)
--skip-existing     Skip folders that already have a scenes.csv file
--copy-frames       Copy frames to scene folders (can use a lot of disk space)
"""
import argparse
import os
import subprocess
import sys
import re
from pathlib import Path
import shutil
import csv

def natural_sort_key(s):
    """Sort strings containing numbers naturally."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def find_frame_files(frames_dir, frame_format='jpg', pattern='%d'):
    """Find all frame files in the directory that match the pattern."""
    all_files = list(frames_dir.glob(f"*.{frame_format}"))
    
    # If using numbered frames like 1.jpg, 2.jpg, etc.
    if pattern == '%d':
        # Sort files naturally so 1.jpg comes before 10.jpg
        frame_files = sorted(all_files, key=lambda x: natural_sort_key(x.name))
    else:
        # For more complex patterns, we'd need to implement custom matching
        # This is a simplified implementation
        frame_files = sorted(all_files)
    
    return frame_files

def run_scene_detection(frames_dir, output_csv, fps, detector="adaptive", 
                       threshold=27, sigma=0.33, min_scene_len=15, frame_format='jpg'):
    """Run scene detection on the frames directory."""
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

def organize_by_scenes(folder_name, frames_dir, scenes_csv, output_dir, frame_format='jpg', copy_frames=False):
    """Organize frames into scene folders based on the CSV output."""
    scenes_dir = output_dir / folder_name / "scenes"
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Read the scenes CSV
    scenes = []
    with open(scenes_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            scene_id, start_frame, end_frame = int(row[0]), int(row[1]), int(row[2])
            scenes.append((scene_id, start_frame, end_frame))
    
    # Get all frame files
    frame_files = find_frame_files(frames_dir, frame_format)
    if not frame_files:
        print(f"[WARNING] No frame files found in {frames_dir}")
        return
    
    # Map frame numbers to files
    # This assumes frames are numbered sequentially from 0 or 1
    frame_map = {}
    start_idx = 0  # Assume frames start at 0
    if frame_files and re.search(r'(\d+)', frame_files[0].stem):
        first_num = int(re.search(r'(\d+)', frame_files[0].stem).group(1))
        start_idx = first_num
    
    for i, frame_file in enumerate(frame_files):
        frame_map[start_idx + i] = frame_file
    
    # Create scene directories and copy/link frames if requested
    scene_info = {}
    for scene_id, start_frame, end_frame in scenes:
        scene_dir = scenes_dir / f"scene_{scene_id:03d}"
        os.makedirs(scene_dir, exist_ok=True)
        
        # Save scene info
        scene_info[f"scene_{scene_id:03d}"] = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": end_frame - start_frame + 1
        }
        
        # Copy frames to scene directory if requested
        if copy_frames:
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num in frame_map:
                    src_file = frame_map[frame_num]
                    dst_file = scene_dir / src_file.name
                    shutil.copy2(src_file, dst_file)
            
            print(f"[INFO] Copied frames {start_frame}-{end_frame} to {scene_dir}")
    
    # Write scene info summary
    summary_file = output_dir / folder_name / "scene_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["scene_name", "start_frame", "end_frame", "frame_count"])
        for scene_name, info in scene_info.items():
            writer.writerow([
                scene_name,
                info["start_frame"],
                info["end_frame"],
                info["frame_count"]
            ])
    
    return scenes_dir

def process_frame_folder(folder_path, output_dir, fps, detector, threshold, 
                        sigma, min_scene_len, frame_format='jpg', 
                        skip_existing=False, copy_frames=False):
    """Process a single folder of frames: detect scenes, organize by scenes."""
    folder_name = folder_path.name
    folder_output_dir = output_dir / folder_name
    scenes_csv = folder_output_dir / "scenes.csv"
    
    # Create output directory
    os.makedirs(folder_output_dir, exist_ok=True)
    
    # Skip if scenes.csv already exists and skip_existing is True
    if skip_existing and scenes_csv.exists():
        print(f"[INFO] Skipping {folder_path} - scenes.csv already exists")
        return
    
    # Check if folder contains frame files
    frame_files = find_frame_files(folder_path, frame_format)
    if not frame_files:
        print(f"[WARNING] No {frame_format} files found in {folder_path}")
        return
    
    print(f"[INFO] Found {len(frame_files)} frame files in {folder_path}")
    
    # Run scene detection
    run_scene_detection(
        folder_path, scenes_csv, fps, detector, 
        threshold, sigma, min_scene_len, frame_format
    )
    
    # Organize frames by scenes
    if scenes_csv.exists():
        organize_by_scenes(folder_name, folder_path, scenes_csv, output_dir, 
                          frame_format, copy_frames)
    else:
        print(f"[WARNING] No scenes detected in {folder_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch process frame folders for scene detection")
    parser.add_argument("parent_folder", type=Path, help="Parent directory containing frame folders")
    parser.add_argument("--output", type=Path, default="./output", 
                        help="Output directory for scene data")
    parser.add_argument("--fps", type=float, default=30, 
                        help="Frames per second (for timecode calculations)")
    parser.add_argument("--frame-format", type=str, default="jpg", choices=["jpg", "png"],
                        help="Format of frame images to process")
    parser.add_argument("--pattern", type=str, default="%d",
                        help="Filename pattern to match frames (default: '%d' for numbered frames)")
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
                        help="Skip folders that already have a scenes.csv file")
    parser.add_argument("--copy-frames", action="store_true",
                        help="Copy frames to scene folders (uses more disk space)")
    
    args = parser.parse_args()
    
    # Validate parent folder
    if not args.parent_folder.is_dir():
        print(f"Error: {args.parent_folder} is not a directory")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get all subdirectories in the parent folder
    subdirs = [d for d in args.parent_folder.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {args.parent_folder}")
        return 1
    
    print(f"Found {len(subdirs)} subdirectories to process")
    
    # Process each folder
    for folder_path in subdirs:
        print(f"\n{'='*80}\nProcessing: {folder_path}\n{'='*80}")
        try:
            process_frame_folder(
                folder_path, args.output, args.fps, args.detector,
                args.threshold, args.sigma, args.min_scene_len,
                args.frame_format, args.skip_existing, args.copy_frames
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {folder_path}: {e}")
    
    print("\nBatch processing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 