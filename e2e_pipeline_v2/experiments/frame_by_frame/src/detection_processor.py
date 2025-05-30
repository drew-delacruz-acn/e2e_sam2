import os
import json
from pathlib import Path
from typing import Dict, List, Any
import re

def natural_sort_key(s):
    """
    Sort strings containing numbers naturally (e.g. frame_1, frame_2, frame_10).
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s))]

def load_detection_files(detections_dir: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load all detection JSON files from the directory
    
    Args:
        detections_dir: Path to directory containing detection JSON files
        
    Returns:
        Dictionary mapping frame indices to lists of detections
    """
    detections_path = Path(detections_dir)
    detection_files = sorted([f for f in detections_path.glob("*.json")], key=natural_sort_key)
    
    detections_by_frame = {}
    for file_path in detection_files:
        try:
            # Extract frame index from filename
            frame_idx = int(file_path.stem)
            
            # Load detections
            with open(file_path, 'r') as f:
                detections = json.load(f)
            
            # Store by frame
            detections_by_frame[frame_idx] = detections
            
        except ValueError:
            print(f"Skipping file {file_path} - could not parse frame index")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(detections_by_frame)} detection files")
    return detections_by_frame

def filter_detections_by_confidence(detections_by_frame, confidence_threshold=0.5):
    """
    Filter detections dictionary to keep only those with confidence above threshold
    
    Args:
        detections_by_frame: Dictionary mapping frame numbers to lists of detections
        confidence_threshold: Minimum confidence score to keep a detection (default: 0.5)
        
    Returns:
        Filtered dictionary with only high-confidence detections
    """
    filtered_detections = {}
    
    total_detections = 0
    kept_detections = 0
    
    for frame_num, detections in detections_by_frame.items():
        # Filter detections for this frame
        high_confidence_detections = []
        
        for detection in detections:
            total_detections += 1
            confidence = detection.get('confidence', 0.0)
            
            if confidence >= confidence_threshold:
                high_confidence_detections.append(detection)
                kept_detections += 1
        
        # Only add frame if it has any detections after filtering
        if high_confidence_detections:
            filtered_detections[frame_num] = high_confidence_detections
        else:
            # Keep the frame with empty list to maintain frame sequence
            filtered_detections[frame_num] = []
    
    # Print statistics
    if total_detections > 0:
        percentage_kept = (kept_detections / total_detections) * 100
        print(f"Filtered detections: kept {kept_detections}/{total_detections} ({percentage_kept:.1f}%)")
        print(f"Confidence threshold: {confidence_threshold}")
    else:
        print("No detections found to filter")
    
    return filtered_detections

def convert_detections_to_tracking_format(detections_by_frame):
    """
    Convert detection dictionary to tracking format with bounding boxes
    
    Args:
        detections_by_frame: Dictionary mapping frame numbers to lists of detections
        
    Returns:
        List of objects in tracking format
    """
    # Track objects by label to assign consistent IDs
    object_ids = {}
    next_id = 1
    
    objects = []
    
    # Process each frame's detections
    for frame_num, detections in sorted(detections_by_frame.items()):
        for detection in detections:
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)
            coords = detection.get('coordinates', [])
            
            # Skip detections without proper coordinates
            if len(coords) != 4:
                continue
                
            # Get or assign object ID
            if label not in object_ids:
                object_ids[label] = next_id
                next_id += 1
            
            object_id = object_ids[label]
            
            # Find or create object entry
            obj = next((o for o in objects if o['objectName'] == label and o['objectID'] == object_id), None)
            
            if obj is None:
                obj = {
                    'objectName': label,
                    'objectID': object_id,
                    'frameOccurences': []
                }
                objects.append(obj)
            
            # Add frame occurrence with bounding box
            obj['frameOccurences'].append({
                'frameNum': frame_num,
                'box': coords,  # [x1, y1, x2, y2]
                'confidence': confidence
            })
    
    return objects 