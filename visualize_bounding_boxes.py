#!/usr/bin/env python3
"""
Visualization Script for SAM2 Segmentation Results
--------------------------------------------------
This script reads segmentation results from a JSON file and draws bounding boxes 
on frames, saving them organized by object ID.
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
import re

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize SAM2 segmentation results')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to segmentation results JSON file')
    parser.add_argument('--frames_dir', type=str, required=True,
                       help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, default='bbox_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--thickness', type=int, default=2,
                       help='Thickness of bounding box lines')
    parser.add_argument('--color_seed', type=int, default=42,
                       help='Random seed for color generation')
    parser.add_argument('--draw_all_objects', action='store_true',
                       help='Draw all objects on each frame')
    return parser.parse_args()

def load_json_results(json_path):
    """Load segmentation results from JSON file"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results

def get_frame_list(frames_dir):
    """Get list of frame filenames and their corresponding frame numbers"""
    frames = {}
    frame_pattern = re.compile(r'frame_(\d+)\.png')
    
    for file in os.listdir(frames_dir):
        match = frame_pattern.match(file)
        if match:
            frame_num = int(match.group(1))
            frames[frame_num] = os.path.join(frames_dir, file)
    
    return frames

def get_object_appearances(results):
    """Organize appearances by object ID"""
    object_appearances = {}
    
    for obj in results:
        obj_id = obj['samObjectId']
        appearances = []
        
        for app in obj['samDictatedAppearances']:
            frame_num = app['frameNum']
            bbox = app['boundingBox']
            appearances.append({
                'frame_num': frame_num,
                'bbox': bbox
            })
        
        object_appearances[obj_id] = appearances
    
    return object_appearances

def generate_colors(num_colors, seed=42):
    """Generate distinct colors for visualization"""
    np.random.seed(seed)
    colors = {}
    for i in range(num_colors):
        color = (
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255)),
            int(np.random.randint(0, 255))
        )
        colors[i+1] = color  # Object IDs typically start from 1
    return colors

def draw_bbox(image, bbox, color, thickness=2):
    """Draw a bounding box on an image"""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Skip if bbox is empty or invalid
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return image
    
    # Ensure coordinates are valid for cv2.rectangle
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1]-1, x2), min(image.shape[0]-1, y2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def create_visualization(frames, object_appearances, colors, output_dir, thickness=2, draw_all_objects=False):
    """Create visualization images with bounding boxes"""
    # Create directories for each object
    for obj_id in object_appearances.keys():
        os.makedirs(os.path.join(output_dir, f'object_{obj_id}'), exist_ok=True)
    
    # Create a directory for all objects if requested
    if draw_all_objects:
        os.makedirs(os.path.join(output_dir, 'all_objects'), exist_ok=True)
    
    # Track which frames we've processed for each object
    processed_frames = {obj_id: set() for obj_id in object_appearances.keys()}
    
    # Process each object's appearances
    for obj_id, appearances in object_appearances.items():
        for app in appearances:
            frame_num = app['frame_num']
            bbox = app['bbox']
            
            # Skip processing if bbox is empty
            if all(coord == 0 for coord in bbox):
                continue
            
            if frame_num in frames:
                # Read frame image
                frame_path = frames[frame_num]
                image = cv2.imread(frame_path)
                
                if image is None:
                    print(f"Warning: Could not read frame {frame_path}")
                    continue
                
                # Draw bounding box for individual object
                obj_img = image.copy()
                obj_img = draw_bbox(obj_img, bbox, colors[obj_id], thickness)
                
                # Add text label
                x1, y1, _, _ = [int(coord) for coord in bbox]
                cv2.putText(obj_img, f"Object {obj_id}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[obj_id], 2)
                
                # Save individual object visualization
                obj_output_path = os.path.join(output_dir, f'object_{obj_id}', f'frame_{frame_num:04d}.png')
                cv2.imwrite(obj_output_path, obj_img)
                
                # Mark this frame as processed for this object
                processed_frames[obj_id].add(frame_num)
    
    # If drawing all objects on each frame
    if draw_all_objects:
        # Get all unique frame numbers that have any object
        all_frames = set()
        for obj_id, frames_set in processed_frames.items():
            all_frames.update(frames_set)
        
        # Process each frame that has at least one object
        for frame_num in all_frames:
            if frame_num in frames:
                # Read frame image
                frame_path = frames[frame_num]
                image = cv2.imread(frame_path)
                
                if image is None:
                    print(f"Warning: Could not read frame {frame_path}")
                    continue
                
                # Create a copy for drawing all objects
                all_obj_img = image.copy()
                
                # Draw bounding boxes for all objects in this frame
                for obj_id, appearances in object_appearances.items():
                    for app in appearances:
                        if app['frame_num'] == frame_num:
                            bbox = app['bbox']
                            # Skip if bbox is empty
                            if all(coord == 0 for coord in bbox):
                                continue
                            
                            all_obj_img = draw_bbox(all_obj_img, bbox, colors[obj_id], thickness)
                            
                            # Add text label
                            x1, y1, _, _ = [int(coord) for coord in bbox]
                            cv2.putText(all_obj_img, f"Object {obj_id}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[obj_id], 2)
                
                # Save visualization with all objects
                all_output_path = os.path.join(output_dir, 'all_objects', f'frame_{frame_num:04d}.png')
                cv2.imwrite(all_output_path, all_obj_img)

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load segmentation results
    print(f"Loading segmentation results from {args.json_path}")
    results = load_json_results(args.json_path)
    
    # Get frame list
    print(f"Loading frames from {args.frames_dir}")
    frames = get_frame_list(args.frames_dir)
    if not frames:
        print("Error: No frames found in the specified directory")
        return
    
    print(f"Found {len(frames)} frames")
    
    # Get object appearances
    object_appearances = get_object_appearances(results)
    print(f"Found {len(object_appearances)} objects")
    
    # Generate colors for visualization
    colors = generate_colors(len(object_appearances), args.color_seed)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualization(
        frames, 
        object_appearances, 
        colors, 
        args.output_dir, 
        args.thickness,
        args.draw_all_objects
    )
    
    print(f"Visualizations saved to {args.output_dir}")
    print("Done!")

if __name__ == "__main__":
    main() 