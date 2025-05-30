#!/usr/bin/env python3
"""
PySceneDetect script for processing individual frame images.
This script takes a directory of frames and identifies unique frames/scenes.
"""

import os
import cv2
import numpy as np
import argparse
import re
from tqdm import tqdm
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode

# Natural sorting function for filenames with numbers
def natural_sort_key(s):
    """
    Sort strings that contain numbers in human order.
    e.g. ["img1.jpg", "img10.jpg", "img2.jpg"] -> ["img1.jpg", "img2.jpg", "img10.jpg"]
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def detect_scenes_from_frames(frames_dir, threshold=30.0, min_scene_len=15):
    """
    Detect scenes from a directory of frame images using histogram comparison.
    
    Args:
        frames_dir: Directory containing frame images (named sequentially)
        threshold: Content detection threshold (higher = less sensitive)
        min_scene_len: Minimum scene length in frames
        
    Returns:
        List of unique frame indices that represent scene changes
    """
    # Get all frame files and sort them using natural sorting
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))],
                         key=natural_sort_key)
    
    if not frame_files:
        print(f"No image files found in {frames_dir}")
        return []
    
    print(f"Processing {len(frame_files)} frames...")
    
    # We don't need PySceneDetect objects anymore since we're using our own implementation
    
    # Let's take a completely different approach
    # Instead of using the PySceneDetect API, let's implement our own scene detection
    
    # Using Content-Based Scene Detection manually
    content_val_threshold = threshold
    min_delta_hsv = 15.0  # Minimum change in HSV color space to consider a new scene
    
    unique_frames = []
    prev_frame_hsv = None
    
    for i, frame_file in enumerate(tqdm(frame_files, desc="Analyzing frames")):
        # Read the frame
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        # First frame is always a scene change
        if i == 0:
            unique_frames.append(i)
            # Convert to HSV for better color comparison
            prev_frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            continue
        
        # Convert current frame to HSV
        curr_frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate the average histogram difference
        hist_diff = 0
        
        # Calculate histogram for each channel (H, S, V)
        for channel in range(3):
            hist1 = cv2.calcHist([prev_frame_hsv], [channel], None, [64], [0, 256])
            hist2 = cv2.calcHist([curr_frame_hsv], [channel], None, [64], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms
            diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            # Convert correlation to a difference (1 - correlation)
            diff = 1.0 - diff
            hist_diff += diff
        
        # Average difference across channels
        hist_diff /= 3.0
        
        # Scale the difference to be more in line with PySceneDetect thresholds
        content_val = hist_diff * 100.0
        
        # Detect scene changes
        if content_val >= content_val_threshold:
            # Ensure minimum scene length
            if len(unique_frames) == 0 or (i - unique_frames[-1]) >= min_scene_len:
                unique_frames.append(i)
        
        # Update previous frame
        prev_frame_hsv = curr_frame_hsv
    
    return unique_frames

def extract_unique_frames(frames_dir, output_dir, threshold=30.0, min_scene_len=15):
    """
    Extract frames organized by scenes and save to separate directories.
    
    Args:
        frames_dir: Input directory containing all frames
        output_dir: Output directory to save scene directories
        threshold: Detection threshold
        min_scene_len: Minimum scene length in frames
    
    Returns:
        Dictionary with scene directories and frame counts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get scene boundary indices
    scene_boundaries = detect_scenes_from_frames(frames_dir, threshold, min_scene_len)
    
    if not scene_boundaries:
        print("No scene changes detected with current threshold.")
        return {}
    
    # Get all frame files and sort them using natural sorting
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))],
                         key=natural_sort_key)
    
    # Create scene ranges with no overlap
    scene_ranges = []
    for i in range(len(scene_boundaries)):
        start_idx = scene_boundaries[i]
        # If this is the last scene, the end is the last frame
        if i == len(scene_boundaries) - 1:
            end_idx = len(frame_files)
        else:
            end_idx = scene_boundaries[i+1]
        
        scene_ranges.append((start_idx, end_idx))
    
    # Copy frames to scene directories keeping original filenames 
    scene_info = {}
    
    for scene_idx, (start_frame, end_frame) in enumerate(scene_ranges):
        # Create scene directory
        scene_dir = os.path.join(output_dir, f"scene_{scene_idx}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Copy all frames for this scene with original filenames
        frame_count = 0
        for frame_idx in range(start_frame, end_frame):
            if frame_idx < len(frame_files):
                src_path = os.path.join(frames_dir, frame_files[frame_idx])
                
                # Keep original filename
                dst_path = os.path.join(scene_dir, frame_files[frame_idx])
                
                # Copy using OpenCV to potentially resize or modify if needed
                img = cv2.imread(src_path)
                cv2.imwrite(dst_path, img)
                frame_count += 1
        
        scene_info[scene_dir] = frame_count
        print(f"Scene {scene_idx}: {frame_count} frames (original frames {start_frame}-{end_frame-1})")
    
    return scene_info

def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect unique frames from a directory of images')
    parser.add_argument('--input', '-i', required=True, help='Directory containing input frames')
    parser.add_argument('--output', '-o', required=True, help='Directory to save unique frames')
    parser.add_argument('--threshold', '-t', type=float, default=30.0, 
                        help='Content detection threshold (lower is more sensitive)')
    parser.add_argument('--min-scene-len', '-m', type=int, default=15,
                        help='Minimum scene length in frames')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        exit(1)
    
    # Extract unique frames
    unique_frames = extract_unique_frames(
        args.input, 
        args.output, 
        threshold=args.threshold,
        min_scene_len=args.min_scene_len
    )
    
    # Print results
    total_scenes = len(unique_frames)
    total_frames = sum(unique_frames.values())
    
    print(f"\nFound {total_scenes} scenes with {total_frames} total frames")
    print(f"Scene directories saved to: {args.output}") 