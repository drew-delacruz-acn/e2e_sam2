#!/usr/bin/env python3
import os
import numpy as np
import argparse
import json
import cv2
import tempfile
import matplotlib.pyplot as plt
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

def create_test_data():
    """Create artificial test data with an object that appears, disappears, and reappears."""
    # Create a temporary directory for JSON files
    temp_dir = tempfile.mkdtemp()
    
    # Create 40 frames of test data (extended to allow for larger gaps)
    test_data = {}
    
    # Define a common ID for the same object to ensure tracking
    monitor_id = "monitor_1"
    
    # Initial position
    base_x, base_y = 100, 100
    
    # Object appears in frames 0-4
    for i in range(5):
        x = base_x + i * 5
        y = base_y + i * 2
        test_data[f"{i}.json"] = [
            {"coordinates": [x, y, x + 50, y + 50], "label": "monitor", "confidence": 0.9, "id": monitor_id}
        ]
    
    # Last position at frame 4
    last_x, last_y = base_x + 4 * 5, base_y + 4 * 2
    
    # Object disappears for frames 5-14 (10 frame gap)
    for i in range(5, 15):
        test_data[f"{i}.json"] = []
    
    # Object reappears in frames 15-19 with positions that maintain reasonable IoU
    for i in range(15, 20):
        offset = i - 15
        x = last_x + 10 + offset * 5  
        y = last_y + 5 + offset * 2
        test_data[f"{i}.json"] = [
            {"coordinates": [x, y, x + 50, y + 50], "label": "monitor", "confidence": 0.9, "id": monitor_id}
        ]
    
    # Update last position
    last_x, last_y = x, y
    
    # Object disappears for frames 20-34 (15 frame gap - larger than min_gap)
    for i in range(20, 35):
        test_data[f"{i}.json"] = []
    
    # Object reappears in frames 35-39
    for i in range(35, 40):
        offset = i - 35
        x = last_x + 15 + offset * 5
        y = last_y + 8 + offset * 2
        test_data[f"{i}.json"] = [
            {"coordinates": [x, y, x + 50, y + 50], "label": "monitor", "confidence": 0.9, "id": monitor_id}
        ]
    
    # Create JSON files
    for filename, detections in test_data.items():
        with open(os.path.join(temp_dir, filename), 'w') as f:
            json.dump(detections, f)
    
    return temp_dir

def create_blank_frames(num_frames):
    """Create blank frames for visualization."""
    frames = []
    for i in range(num_frames):
        # Create a white image
        frame = np.ones((400, 600, 3), dtype=np.uint8) * 255
        # Add frame number
        cv2.putText(frame, f"Frame {i}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        frames.append(frame)
    return frames

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Test CD-FSOD Detector continuity tracking')
    parser.add_argument('--output_dir', type=str, default='fsod_test_output',
                        help='Directory to save output visualizations')
    parser.add_argument('--min_gap', type=int, default=10,
                        help='Minimum frame gap to consider as a reappearance')
    parser.add_argument('--use_temp_data', action='store_true',
                        help='Use automatically generated test data')
    parser.add_argument('--json_dir', type=str,
                        help='Directory containing CD-FSOD JSON files (if not using temp data)')
    parser.add_argument('--debug', action='store_true',
                        help='Print additional debug information')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Either use user-provided JSON directory or create test data
    if args.use_temp_data:
        json_dir = create_test_data()
        print(f"Created test data in temporary directory: {json_dir}")
        
        # If debugging, print out the test data
        if args.debug:
            print("\nTest data:")
            for i in range(40):  # Updated to 40 frames
                filename = f"{i}.json"
                filepath = os.path.join(json_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        print(f"Frame {i}: {data}")
            print()
    else:
        if not args.json_dir:
            print("Error: Must provide --json_dir or use --use_temp_data")
            return
        json_dir = args.json_dir
    
    # Initialize the detector
    detector = CDFSODDetector(
        json_dir=json_dir,
        confidence_threshold=0.2,
        iou_threshold=0.3,  # Lower IoU threshold to better track objects across gaps
        min_gap_frames=args.min_gap
    )
    
    print(f"Initialized detector with {len(detector.detections_by_frame)} frames")
    print(f"Min gap frames: {args.min_gap}")
    
    # Print object tracking information if debugging
    if args.debug:
        print("\nObject tracking:")
        frames = sorted(detector.detections_by_frame.keys())
        for frame_idx in frames:
            if frame_idx in detector.first_appearances and detector.first_appearances[frame_idx]:
                print(f"First appearances in frame {frame_idx}: {len(detector.first_appearances[frame_idx])}")
            if frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
                print(f"Reappearances in frame {frame_idx}: {len(detector.reappearances[frame_idx])}")
        print()
    
    # Create blank frames for visualization
    if args.use_temp_data:
        frames = create_blank_frames(40)  # Updated to 40 frames
    else:
        # For real data, create as many frames as we have detection files
        frames = create_blank_frames(max(detector.detections_by_frame.keys()) + 1)
    
    # Track where detections occur
    first_appearances = []
    reappearances = []
    
    # Process each frame
    for frame_idx in range(len(frames)):
        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        
        # Use MockImage to properly handle frame_idx
        frame_rgb = MockImage(frame_rgb, frame_idx)
        
        # Detect objects with the "monitor" query
        results = detector.detect(frame_rgb, ["monitor"])
        
        # Create a visualization frame
        vis_frame = frames[frame_idx].copy()
        
        # Draw detection status text
        if frame_idx in detector.first_appearances and detector.first_appearances[frame_idx]:
            first_appearances.append(frame_idx)
            cv2.putText(vis_frame, "FIRST APPEARANCE", (200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if frame_idx in detector.reappearances and detector.reappearances[frame_idx]:
            reappearances.append(frame_idx)
            cv2.putText(vis_frame, "REAPPEARANCE", (350, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw each detection returned by the detect method
        for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            text = f"{label}: {score:.2f}"
            cv2.putText(vis_frame, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw all CD-FSOD detections (for reference) in gray
        if frame_idx in detector.detections_by_frame:
            for detection in detector.detections_by_frame[frame_idx]:
                x1, y1, x2, y2 = [int(coord) for coord in detection["coordinates"]]
                
                # Draw bounding box in gray (thinner line)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
        
        # Save visualization
        output_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(output_path, vis_frame)
        
        # Print detection results
        print(f"Frame {frame_idx}:")
        if len(results['boxes']) == 0:
            print("  No detections")
        else:
            for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
                print(f"  {label}: {score:.2f} at {box}")
    
    # Create a summary visualization
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(frames)), [0] * len(frames), 'k-', alpha=0.3)  # Timeline
    
    # Mark frames with detections
    for idx in range(len(frames)):
        if idx in detector.detections_by_frame and detector.detections_by_frame[idx]:
            plt.plot(idx, 0, 'ko', alpha=0.3)  # Gray dot for all detections
    
    # Mark first appearances and reappearances
    for idx in first_appearances:
        plt.plot(idx, 0, 'ro', markersize=10, label="First Appearance" if idx == first_appearances[0] else "")
    
    for idx in reappearances:
        plt.plot(idx, 0, 'bo', markersize=10, label="Reappearance" if idx == reappearances[0] else "")
    
    # Add legend and labels
    plt.legend()
    plt.title("CD-FSOD Detector: First Appearances and Reappearances")
    plt.xlabel("Frame")
    plt.yticks([])  # Hide y-axis
    plt.tight_layout()
    
    # Save timeline visualization
    timeline_path = os.path.join(args.output_dir, "timeline.png")
    plt.savefig(timeline_path)
    
    print(f"\nTest completed. Visualizations saved to {args.output_dir}")
    print(f"First appearances in frames: {first_appearances}")
    print(f"Reappearances in frames: {reappearances}")
    print(f"Timeline visualization saved to {timeline_path}")

if __name__ == "__main__":
    main() 