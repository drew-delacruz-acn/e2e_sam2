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
    parser.add_argument('--frames_dir', type=str, required=False,
                        help='Directory containing video frames (required if --visualize is used)')
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
    parser.add_argument('--show_all_detections', action='store_true',
                        help='Show all detections instead of just first appearances and reappearances')
    parser.add_argument('--show_track_info', action='store_true',
                        help='Show detailed object tracking information')
    
    args = parser.parse_args()
    
    # Check if frames_dir is provided when visualize is True
    if args.visualize and not args.frames_dir:
        parser.error("--frames_dir is required when --visualize is set")
    
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
    print(f"Minimum gap frames: {args.min_gap}")
    
    # Print object tracking information if debugging
    if args.debug:
        print("\nObject tracking information:")
        print(f"First appearances:")
        for frame_idx in sorted(detector.first_appearances.keys()):
            if detector.first_appearances[frame_idx]:
                print(f"  Frame {frame_idx}: {len(detector.first_appearances[frame_idx])} objects")
                for detection in detector.first_appearances[frame_idx]:
                    print(f"    {detection['label']} (ID: {detection.get('object_id', 'unknown')}): {detection['confidence']:.2f} at {detection['coordinates']}")
        
        print(f"\nReappearances:")
        for frame_idx in sorted(detector.reappearances.keys()):
            if detector.reappearances[frame_idx]:
                print(f"  Frame {frame_idx}: {len(detector.reappearances[frame_idx])} objects")
                for detection in detector.reappearances[frame_idx]:
                    print(f"    {detection['label']} (ID: {detection.get('object_id', 'unknown')}): {detection['confidence']:.2f} at {detection['coordinates']}")
        
        print()
    
    # Get frame indices from detector data
    frame_indices = sorted(list(detector.detections_by_frame.keys()))
    
    if args.end_frame == -1:
        args.end_frame = max(frame_indices) if frame_indices else 0
    
    start_frame = args.start_frame
    end_frame = args.end_frame
    
    print(f"Processing frames {start_frame} to {end_frame}")
    print("-" * 50)
    
    # Display tracking summary information
    print("\nTracking summary:")
    print("First Appearances:")
    for frame_idx in sorted(detector.first_appearances.keys()):
        if detector.first_appearances[frame_idx]:
            for detection in detector.first_appearances[frame_idx]:
                obj_id = detection.get('object_id', 'unknown')
                print(f"  Frame {frame_idx}: {detection['label']} ({detection['confidence']:.2f}) ID: {obj_id}")
                
    print(f"\nReappearances (after gap of > {args.min_gap} frames):")
    for frame_idx in sorted(detector.reappearances.keys()):
        if detector.reappearances[frame_idx]:
            for detection in detector.reappearances[frame_idx]:
                obj_id = detection.get('object_id', 'unknown')
                print(f"  Frame {frame_idx}: {detection['label']} ({detection['confidence']:.2f}) ID: {obj_id}")
    
    # Display object track information if requested
    if args.show_track_info and hasattr(detector, 'object_tracks'):
        print("\nObject Tracks:")
        for obj_id, track in detector.object_tracks.items():
            frame_indices = [frame_idx for frame_idx, _ in track]
            print(f"  {obj_id}: {len(track)} appearances in frames {frame_indices}")
    
    print("-" * 50)
    
    # Determine frame processing approach based on visualize flag
    if args.visualize:
        # Get list of frame files and sort them naturally
        frame_files = [f for f in os.listdir(args.frames_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Use natural sorting instead of alphabetical
        frame_files.sort(key=natural_sort_key)
        
        # Process each frame with visualization
        for i, frame_file in enumerate(frame_files[start_frame:end_frame+1]):
            frame_idx = start_frame + i
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
            
            # Get all detections if requested
            if args.show_all_detections and frame_idx in detector.detections_by_frame:
                all_detections = []
                for detection in detector.detections_by_frame[frame_idx]:
                    if detection['label'] in text_queries:
                        all_detections.append({
                            'coordinates': detection['coordinates'],
                            'label': detection['label'],
                            'confidence': detection['confidence']
                        })
                
                results = {
                    'boxes': np.array([d['coordinates'] for d in all_detections]) if all_detections else np.zeros((0, 4)),
                    'labels': [d['label'] for d in all_detections],
                    'scores': np.array([d['confidence'] for d in all_detections]) if all_detections else np.zeros(0)
                }
                
                detection_type = "ALL DETECTIONS"
            else:
                # Detect objects (first appearances and reappearances only)
                results = detector.detect(frame_rgb, text_queries)
                
                # Determine detection type for this frame
                if frame_idx in detector.first_appearances and detector.first_appearances[frame_idx]:
                    if frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
                        detection_type = "FIRST APPEARANCE + REAPPEARANCE"
                    else:
                        detection_type = "FIRST APPEARANCE"
                elif frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
                    detection_type = "REAPPEARANCE"
                else:
                    detection_type = "NONE"
            
            # Print detection results
            print(f"Frame {frame_idx} ({frame_file}) - {detection_type}:")
            if len(results['boxes']) == 0:
                print("  No detections")
            else:
                for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
                    print(f"  {label}: {score:.2f} at {box}")
            
            # Create a copy for visualization
            vis_frame = frame.copy()
            
            # Draw detection status text
            if args.show_all_detections:
                cv2.putText(vis_frame, "ALL DETECTIONS", (20, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                if frame_idx in detector.first_appearances and detector.first_appearances[frame_idx]:
                    cv2.putText(vis_frame, "FIRST APPEARANCE", (20, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
                    cv2.putText(vis_frame, "REAPPEARANCE", (20, 60), 
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
    else:
        # Process frames without visualization (JSON-only mode)
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in detector.detections_by_frame:
                # Create a mock frame with just the index (no image data needed)
                mock_frame = MockImage(np.zeros((1, 1, 3), dtype=np.uint8), frame_idx)
                
                # Get all detections if requested
                if args.show_all_detections:
                    all_detections = []
                    for detection in detector.detections_by_frame[frame_idx]:
                        if detection['label'] in text_queries:
                            all_detections.append({
                                'coordinates': detection['coordinates'],
                                'label': detection['label'],
                                'confidence': detection['confidence']
                            })
                    
                    results = {
                        'boxes': np.array([d['coordinates'] for d in all_detections]) if all_detections else np.zeros((0, 4)),
                        'labels': [d['label'] for d in all_detections],
                        'scores': np.array([d['confidence'] for d in all_detections]) if all_detections else np.zeros(0)
                    }
                    
                    detection_type = "ALL DETECTIONS"
                else:
                    # Detect objects (first appearances and reappearances only)
                    results = detector.detect(mock_frame, text_queries)
                    
                    # Determine detection type for this frame
                    if frame_idx in detector.first_appearances and detector.first_appearances[frame_idx]:
                        if frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
                            detection_type = "FIRST APPEARANCE + REAPPEARANCE"
                        else:
                            detection_type = "FIRST APPEARANCE"
                    elif frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
                        detection_type = "REAPPEARANCE"
                    else:
                        detection_type = "NONE"
                
                # Print detection results
                print(f"Frame {frame_idx} - {detection_type}:")
                if len(results['boxes']) == 0:
                    print("  No detections")
                else:
                    for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
                        print(f"  {label}: {score:.2f} at {box}")
    
    if args.visualize:
        print(f"Processing complete. Visualizations saved to {args.output_dir}")
    else:
        print(f"Processing complete.")

if __name__ == "__main__":
    main() 

# Example usage with visualization:
# python run_cd_fsod.py --json_dir "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529" --frames_dir "data/frames//Scenes 061-080__265H-2-_20230815215828529" --visualize --min_gap 10 --confidence 0.9

# Example usage with JSON only (no visualization):
# python run_cd_fsod.py --json_dir "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529" --min_gap 10 --confidence 0.9

# Example usage to show all detections (including continuous tracks):
# python run_cd_fsod.py --json_dir "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529" --min_gap 10 --confidence 0.9 --show_all_detections

# Example usage with detailed track info:
# python run_cd_fsod.py --json_dir "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/data/detections/Scenes 061-080__265H-2-_20230815215828529" --min_gap 10 --confidence 0.9 --show_track_info