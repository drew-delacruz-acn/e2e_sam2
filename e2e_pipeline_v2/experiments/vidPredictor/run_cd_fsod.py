#!/usr/bin/env python3
import os
import numpy as np
import argparse
import cv2
from pathlib import Path
from src.cd_fsod_detector import CDFSODDetector

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
    
    # Get list of frame files
    frame_files = sorted([f for f in os.listdir(args.frames_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    
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