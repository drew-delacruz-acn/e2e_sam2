#!/usr/bin/env python3
"""
Visualize Bounding Boxes from Segmentation Results

This script reads a segmentation results JSON file, draws bounding boxes on the original frames,
and saves them to object-specific folders.
"""

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize bounding boxes from segmentation results')
    parser.add_argument('--results_json', type=str, required=True,
                        help='Path to segmentation results JSON file')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, default='box_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--color_mode', type=str, default='unique',
                        choices=['unique', 'consistent'],
                        help='Color mode for bounding boxes: unique (different color per frame) or consistent (same color per object)')
    parser.add_argument('--frame_format', type=str, default='{:06d}.jpg',
                        help='Format string for frame filenames')
    return parser.parse_args()

def load_segmentation_results(json_path):
    """Load segmentation results from JSON file"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Validate the results
    if not results or not isinstance(results, list):
        raise ValueError(f"Invalid segmentation results format in {json_path}")
    
    print(f"Loaded results for {len(results)} objects")
    return results

def get_frame_path(frames_dir, frame_num, frame_format):
    """Get the path to a specific frame"""
    # Try different naming conventions
    format_options = [
        frame_format,                  # e.g., 000001.jpg
        f"{frame_num}.jpg",            # e.g., 1.jpg
        f"{frame_num}.png",            # e.g., 1.png
        f"frame_{frame_num}.jpg",      # e.g., frame_1.jpg
        f"frame_{frame_num:06d}.jpg",  # e.g., frame_000001.jpg
    ]
    
    for fmt in format_options:
        if isinstance(fmt, str) and '{' in fmt:
            # Apply format string
            try:
                frame_name = fmt.format(frame_num)
            except:
                frame_name = str(frame_num) + ".jpg"
        else:
            frame_name = fmt
            
        frame_path = os.path.join(frames_dir, frame_name)
        if os.path.exists(frame_path):
            return frame_path
    
    # If we tried all options and none worked, raise an error
    raise FileNotFoundError(f"Could not find frame {frame_num} in {frames_dir}")

def generate_color_for_object(obj_id, consistent=True):
    """Generate a color for an object"""
    if consistent:
        # Use deterministic color based on object ID
        np.random.seed(obj_id * 100)
        color = (
            int(np.random.randint(50, 230)),
            int(np.random.randint(50, 230)),
            int(np.random.randint(50, 230))
        )
        np.random.seed(None)  # Reset the seed
        return color
    else:
        # Generate a random color
        return (
            int(np.random.randint(50, 230)),
            int(np.random.randint(50, 230)),
            int(np.random.randint(50, 230))
        )

def draw_bounding_box(image, box, obj_id, obj_name=None, color=None):
    """Draw a bounding box on an image"""
    # Default color if not provided
    if color is None:
        color = generate_color_for_object(obj_id, consistent=True)
    
    # Extract box coordinates
    x1, y1, x2, y2 = box
    
    # Convert coordinates to integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label text
    label = f"ID: {obj_id}"
    if obj_name:
        label += f" - {obj_name}"
    
    # Draw text background
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(
        image,
        (x1, y1 - text_size[1] - 10),
        (x1 + text_size[0] + 10, y1),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        label,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    
    return image

def main():
    """Main function to visualize bounding boxes"""
    args = parse_args()
    
    # Load segmentation results
    results = load_segmentation_results(args.results_json)
    
    # Create output directories
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a directory for each object
    for obj in results:
        obj_id = obj['samObjectId']
        obj_dir = base_output_dir / f"object_{obj_id}"
        obj_dir.mkdir(exist_ok=True, parents=True)
    
    # Add a directory for all objects combined
    all_objects_dir = base_output_dir / "all_objects"
    all_objects_dir.mkdir(exist_ok=True, parents=True)
    
    # Track all frames that need visualization for "all objects" view
    all_frames = {}
    for obj in results:
        obj_id = obj['samObjectId']
        for appearance in obj['samDictatedAppearances']:
            frame_num = appearance['frameNum']
            if frame_num not in all_frames:
                all_frames[frame_num] = []
            all_frames[frame_num].append((obj_id, appearance['boundingBox']))
    
    # Process each object
    print("Generating visualizations...")
    for obj in tqdm(results, desc="Objects"):
        obj_id = obj['samObjectId']
        obj_dir = base_output_dir / f"object_{obj_id}"
        
        # Get object name from predictions if available
        obj_name = None
        if obj['cdfsodPredictions'] and len(obj['cdfsodPredictions']) > 0:
            obj_name = obj['cdfsodPredictions'][0]['class']
        
        # Generate a consistent color for this object
        obj_color = generate_color_for_object(obj_id, consistent=True)
        
        # Process each appearance
        for appearance in obj['samDictatedAppearances']:
            frame_num = appearance['frameNum']
            box = appearance['boundingBox']
            
            # Skip empty or invalid boxes (all zeros or negative coordinates)
            if (box[0] <= 0 and box[1] <= 0 and box[2] <= 0 and box[3] <= 0) or \
               (box[0] >= box[2] or box[1] >= box[3]):
                continue
            
            try:
                # Get frame path and load image
                frame_path = get_frame_path(args.frames_dir, frame_num, args.frame_format)
                image = cv2.imread(frame_path)
                
                if image is None:
                    print(f"Warning: Could not read frame {frame_path}")
                    continue
                
                # Draw bounding box
                image_with_box = draw_bounding_box(
                    image.copy(), 
                    box, 
                    obj_id, 
                    obj_name=obj_name,
                    color=obj_color if args.color_mode == 'consistent' else None
                )
                
                # Save the image
                output_path = obj_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(output_path), image_with_box)
                
            except FileNotFoundError as e:
                print(f"Warning: {e}")
    
    # Generate combined visualizations for all objects
    print("Generating combined visualizations...")
    for frame_num, objects in tqdm(all_frames.items(), desc="Combined frames"):
        try:
            # Get frame path and load image
            frame_path = get_frame_path(args.frames_dir, frame_num, args.frame_format)
            image = cv2.imread(frame_path)
            
            if image is None:
                print(f"Warning: Could not read frame {frame_path}")
                continue
            
            # Draw all bounding boxes
            image_with_boxes = image.copy()
            for obj_id, box in objects:
                # Skip empty or invalid boxes
                if (box[0] <= 0 and box[1] <= 0 and box[2] <= 0 and box[3] <= 0) or \
                   (box[0] >= box[2] or box[1] >= box[3]):
                    continue
                
                # Find object name
                obj_name = None
                for obj in results:
                    if obj['samObjectId'] == obj_id and obj['cdfsodPredictions']:
                        obj_name = obj['cdfsodPredictions'][0]['class']
                        break
                
                # Draw the box
                obj_color = generate_color_for_object(obj_id, consistent=True)
                image_with_boxes = draw_bounding_box(
                    image_with_boxes, 
                    box, 
                    obj_id, 
                    obj_name=obj_name,
                    color=obj_color
                )
            
            # Save the image
            output_path = all_objects_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(output_path), image_with_boxes)
            
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    
    print(f"Visualizations saved to {args.output_dir}")
    print(f"- Individual objects: {len(results)} folders")
    print(f"- Combined visualizations in: {str(all_objects_dir)}")

if __name__ == "__main__":
    main() 