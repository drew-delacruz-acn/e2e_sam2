#!/usr/bin/env python3
"""
PySceneDetect script for processing individual frame images.
This script takes a directory of frames and identifies unique frames/scenes.
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode

def detect_scenes_from_frames(frames_dir, threshold=30.0, min_scene_len=15):
    """
    Detect scenes from a directory of frame images using PySceneDetect.
    
    Args:
        frames_dir: Directory containing frame images (named sequentially)
        threshold: Content detection threshold (higher = less sensitive)
        min_scene_len: Minimum scene length in frames
        
    Returns:
        List of unique frame indices that represent scene changes
    """
    # Get all frame files and sort them
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not frame_files:
        print(f"No image files found in {frames_dir}")
        return []
    
    print(f"Processing {len(frame_files)} frames...")
    
    # Create detector and scene manager
    detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    scene_manager = SceneManager()
    scene_manager.add_detector(detector)
    
    # Process each frame
    for i, frame_file in enumerate(tqdm(frame_files, desc="Analyzing frames")):
        # Read the frame
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        # Create a timecode object for the current frame
        timecode = FrameTimecode(i, fps=30)  # Assuming 30fps
        
        # Process the frame
        scene_manager.process_frame(timecode, frame)
    
    # Get the list of scenes
    scene_list = scene_manager.get_scene_list()
    
    # Extract the starting frames of each scene
    unique_frame_indices = [scene[0].get_frames() for scene in scene_list]
    
    return unique_frame_indices

def extract_unique_frames(frames_dir, output_dir, threshold=30.0, min_scene_len=15):
    """
    Extract unique frames from a directory and save them to output directory.
    
    Args:
        frames_dir: Input directory containing all frames
        output_dir: Output directory to save unique frames
        threshold: Detection threshold
        min_scene_len: Minimum scene length in frames
    
    Returns:
        List of paths to unique frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique frame indices
    unique_indices = detect_scenes_from_frames(frames_dir, threshold, min_scene_len)
    
    if not unique_indices:
        print("No scene changes detected with current threshold.")
        return []
    
    # Get all frame files and sort them
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Copy unique frames to output directory
    unique_frame_paths = []
    for idx in unique_indices:
        if idx < len(frame_files):
            src_path = os.path.join(frames_dir, frame_files[idx])
            dst_path = os.path.join(output_dir, f"scene_{idx}_{frame_files[idx]}")
            
            # Copy using OpenCV to potentially resize or modify if needed
            img = cv2.imread(src_path)
            cv2.imwrite(dst_path, img)
            unique_frame_paths.append(dst_path)
    
    return unique_frame_paths

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
    print(f"\nFound {len(unique_frames)} unique frames/scenes")
    print(f"Unique frames saved to: {args.output}") 