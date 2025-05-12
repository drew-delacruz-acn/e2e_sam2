#!/usr/bin/env python3
import os
import numpy as np
import argparse
import cv2
from pathlib import Path
import re
from src.cd_fsod_detector import CDFSODDetector

# Add natural sorting function
def natural_sort_key(s):
    """Key function for natural sorting"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

# Add the MockImage class for frame_idx handling
class MockImage(np.ndarray):
    def __new__(cls, input_array, frame_idx):
        # Create a new numpy array
        obj = np.asarray(input_array).view(cls)
        # Add frame_idx as a separate attribute
        obj._frame_idx = frame_idx
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self._frame_idx = getattr(obj, '_frame_idx', None)
        
    @property
    def frame_idx(self):
        return self._frame_idx

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run CD-FSOD Detector with custom JSON files')
    parser.add_argument('--json_dir', type=str, required=True, 
                        help='Directory containing CD-FSOD JSON files (0.json, 1.json, etc.)')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save output visualizations')
    parser.add_argument('--confidence', type=float, default=0.2,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold for object matching')
    parser.add_argument('--min_gap', type=int, default=10,
                        help='Minimum frame gap to consider as a reappearance')
    parser.add_argument('--queries', type=str, default='all',
                        help='Comma-separated list of object classes to detect (use "all" for all classes)')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='First frame to process')
    parser.add_argument('--end_frame', type=int, default=-1,
                        help='Last frame to process (-1 for all frames)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of detections')
    parser.add_argument('--debug', action='store_true',
                        help='Print additional debug information')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the detector
    detector = CDFSODDetector(
        json_dir=args.json_dir,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        min_gap_frames=args.min_gap
    )
    
    # Get all available classes from detections
    all_classes = set()
    for frame_idx, detections in detector.detections_by_frame.items():
        for detection in detections:
            all_classes.add(detection['label'])
    
    # Determine which classes to detect
    if args.queries.lower() == 'all':
        text_queries = list(all_classes)
    else:
        text_queries = [q.strip() for q in args.queries.split(',')]
    
    print(f"Loaded detector with {len(detector.detections_by_frame)} frames")
    print(f"Available classes: {sorted(list(all_classes))}")
    print(f"Detecting classes: {text_queries}")
    
    # Print object tracking information if debugging
    if args.debug:
        print("\nObject tracking information:")
        print(f"First appearances:")
        for frame_idx in sorted(detector.first_appearances.keys()):
            if detector.first_appearances[frame_idx]:
                print(f"  Frame {frame_idx}: {len(detector.first_appearances[frame_idx])} objects")
                for detection in detector.first_appearances[frame_idx]:
                    print(f"    {detection['label']}: {detection['confidence']:.2f} at {detection['coordinates']}")
        
        print(f"\nReappearances:")
        for frame_idx in sorted(detector.reappearances.keys()):
            if detector.reappearances[frame_idx]:
                print(f"  Frame {frame_idx}: {len(detector.reappearances[frame_idx])} objects")
                for detection in detector.reappearances[frame_idx]:
                    print(f"    {detection['label']}: {detection['confidence']:.2f} at {detection['coordinates']}")
        
        print()
    
    # Get list of frame files and sort them naturally
    frame_files = [f for f in os.listdir(args.frames_dir) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Use natural sorting instead of alphabetical
    frame_files.sort(key=natural_sort_key)
    
    if args.end_frame == -1:
        args.end_frame = len(frame_files) - 1
    
    print(f"Processing frames {args.start_frame} to {args.end_frame}")
    
    # Process each frame
    for i, frame_file in enumerate(frame_files[args.start_frame:args.end_frame+1]):
        frame_idx = args.start_frame + i
        frame_path = os.path.join(args.frames_dir, frame_file)
        
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error loading frame: {frame_path}")
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use MockImage to properly handle frame_idx
        frame_rgb = MockImage(frame_rgb, frame_idx)
        
        # Detect objects
        results = detector.detect(frame_rgb, text_queries)
        
        # Print detection results
        print(f"Frame {frame_idx} ({frame_file}):")
        if len(results['boxes']) == 0:
            print("  No detections")
        else:
            for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
                print(f"  {label}: {score:.2f} at {box}")
        
        # Visualize if requested
        if args.visualize:
            # Create a copy for visualization
            vis_frame = frame.copy()
            
            # Draw detection status text
            if frame_idx in detector.first_appearances and detector.first_appearances[frame_idx]:
                cv2.putText(vis_frame, "FIRST APPEARANCE", (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
                cv2.putText(vis_frame, "REAPPEARANCE", (20, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw each detection
            for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                text = f"{label}: {score:.2f}"
                cv2.putText(vis_frame, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save visualization
            output_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(output_path, vis_frame)
    
    print(f"Processing complete. Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 

# python run_cd_fsod.py --json_dir "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529" --frames_dir "data/frames//Scenes 061-080__265H-2-_20230815215828529" --visualize --min_gap 10 --confidence 0.9
# Loaded detector with 54 frames
# Available classes: ["Sylvie's horned headpiece", 'TVA Uniform', 'Time Stick']
# Detecting classes: ["Sylvie's horned headpiece", 'TVA Uniform', 'Time Stick']
# Processing frames 0 to 53
# Frame 0 (0.jpg):
#   No detections
# Frame 1 (1.jpg):
#   No detections
# Frame 2 (2.jpg):
#   No detections
# Frame 3 (3.jpg):
#   Time Stick: 0.99 at [453 224 643 473]
# Frame 4 (4.jpg):
#   Time Stick: 1.00 at [385 252 587 476]
# Frame 5 (5.jpg):
#   Time Stick: 0.97 at [313 262 524 477]
# Frame 6 (6.jpg):
#   No detections
# Frame 7 (7.jpg):
#   No detections
# Frame 8 (8.jpg):
#   No detections
# Frame 9 (9.jpg):
#   No detections
# Frame 10 (10.jpg):
#   No detections
# Frame 11 (11.jpg):
#   No detections
# Frame 12 (12.jpg):
#   No detections
# Frame 13 (13.jpg):
#   TVA Uniform: 0.97 at [374 115 604 490]
#   Time Stick: 0.92 at [591 346 655 481]
# Frame 14 (14.jpg):
#   No detections
# Frame 15 (15.jpg):
#   No detections
# Frame 16 (16.jpg):
#   No detections
# Frame 17 (17.jpg):
#   No detections
# Frame 18 (18.jpg):
#   Sylvie's horned headpiece: 0.99 at [322 128 481 202]
# Frame 19 (19.jpg):
#   Time Stick: 0.94 at [240 156 417 412]
# Frame 20 (20.jpg):
#   Sylvie's horned headpiece: 0.98 at [536  82 721 177]
# Frame 21 (21.jpg):
#   No detections
# Frame 22 (22.jpg):
#   No detections
# Frame 23 (23.jpg):
#   No detections
# Frame 24 (24.jpg):
#   No detections
# Frame 25 (25.jpg):
#   No detections
# Frame 26 (26.jpg):
#   No detections
# Frame 27 (27.jpg):
#   No detections
# Frame 28 (28.jpg):
#   No detections
# Frame 29 (29.jpg):
#   No detections
# Frame 30 (30.jpg):
#   No detections
# Frame 31 (31.jpg):
#   No detections
# Frame 32 (32.jpg):
#   No detections
# Frame 33 (33.jpg):
#   Sylvie's horned headpiece: 0.96 at [582 104 671 201]
# Frame 34 (34.jpg):
#   No detections
# Frame 35 (35.jpg):
#   No detections
# Frame 36 (36.jpg):
#   No detections
# Frame 37 (37.jpg):
#   No detections
# Frame 38 (38.jpg):
#   No detections
# Frame 39 (39.jpg):
#   No detections
# Frame 40 (40.jpg):
#   No detections
# Frame 41 (41.jpg):
#   No detections
# Frame 42 (42.jpg):
#   No detections
# Frame 43 (43.jpg):
#   Sylvie's horned headpiece: 0.91 at [442 160 541 219]
# Frame 44 (44.jpg):
#   No detections
# Frame 45 (45.jpg):
#   No detections
# Frame 46 (46.jpg):
#   No detections
# Frame 47 (47.jpg):
#   No detections
# Frame 48 (48.jpg):
#   No detections
# Frame 49 (49.jpg):
#   No detections
# Frame 50 (50.jpg):
#   No detections
# Frame 51 (51.jpg):
#   No detections
# Frame 52 (52.jpg):
#   No detections
# Frame 53 (53.jpg):
#   No detections
# Processing complete. Visualizations saved to output