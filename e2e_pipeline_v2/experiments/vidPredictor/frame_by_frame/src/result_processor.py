import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Return 0 if there's no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Compute the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area
    
    # Return the IoU value
    return intersection_area / union_area if union_area > 0 else 0.0

def get_bounding_box_from_mask(mask):
    """
    Calculate bounding box coordinates from a binary mask
    
    Args:
        mask: Binary mask array
        
    Returns:
        Bounding box coordinates [x1, y1, x2, y2]
    """
    # Find non-zero elements (the mask)
    mask_positions = np.where(mask)
    
    # No mask points, return empty box
    if len(mask_positions[0]) == 0:
        return [0, 0, 0, 0]
    
    # Get the boundary coordinates
    y_min, y_max = np.min(mask_positions[0]), np.max(mask_positions[0])
    x_min, x_max = np.min(mask_positions[1]), np.max(mask_positions[1])
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def match_detections_to_segments(video_segments, detections_by_frame, iou_threshold=0.5):
    """
    Match CDFSOD detections to SAM2 segmented objects based on IoU
    
    Args:
        video_segments: Dictionary mapping frame indices to segmentation results
        detections_by_frame: Dictionary mapping frame indices to CDFSOD detections
        iou_threshold: Minimum IoU value to consider a match
        
    Returns:
        Dictionary mapping object IDs to their detection matches
    """
    object_detections = {}
    
    # Process each frame
    for frame_idx in sorted(video_segments.keys()):
        # Get segmentation results for this frame
        frame_segments = video_segments[frame_idx]
        
        # Check if we have detections for this frame
        if frame_idx not in detections_by_frame:
            continue
        
        # Get detections for this frame
        frame_detections = detections_by_frame[frame_idx]
        
        # Find matches between segments and detections
        for obj_id, mask in frame_segments.items():
            # If this is a new object, initialize its detection list
            if obj_id not in object_detections:
                object_detections[obj_id] = []
            
            # Get bounding box for the mask
            sam_box = get_bounding_box_from_mask(mask)
            
            # Find matching detections
            matches = []
            for detection in frame_detections:
                detection_box = detection['coordinates']
                iou = calculate_iou(sam_box, detection_box)
                
                if iou >= iou_threshold:
                    matches.append((iou, detection))
            
            # If we have matches, add the best one to this object's detections
            if matches:
                # Sort by IoU (highest first)
                matches.sort(reverse=True, key=lambda x: x[0])
                best_match = matches[0][1]
                
                # Add to object's detections
                object_detections[obj_id].append({
                    'frameNumber': frame_idx,
                    'class': best_match['label'],
                    'confidence': best_match['confidence']
                })
    
    return object_detections

def create_segmentation_summary(video_segments, tracking_objects, object_detections):
    """
    Create a summary of segmentation results in the desired format
    
    Args:
        video_segments: Dictionary mapping frame indices to segmentation results
        tracking_objects: List of tracking objects with frame occurrences
        object_detections: Dictionary mapping object IDs to their detection matches
        
    Returns:
        List of objects with their appearance and detection information
    """
    result = []
    
    # Get all unique object IDs
    object_ids = set()
    for frame_idx, segments in video_segments.items():
        object_ids.update(segments.keys())
    
    # Process each object
    for obj_id in sorted(object_ids):
        # Get all frames where this object appears
        appearances = []
        for frame_idx in sorted(video_segments.keys()):
            if obj_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][obj_id]
                box = get_bounding_box_from_mask(mask)
                appearances.append({
                    'frameNum': frame_idx,
                    'boundingBox': box
                })
        
        # Get all detections for this object
        cdfsod_predictions = object_detections.get(obj_id, [])
        
        # Create object summary
        obj_summary = {
            'samObjectId': int(obj_id),
            'samDictatedAppearances': appearances,
            'cdfsodPredictions': cdfsod_predictions
        }
        
        result.append(obj_summary)
    
    return result

def save_results_to_json(results, output_path):
    """
    Save results to a JSON file
    
    Args:
        results: Results data to save
        output_path: Path to the output JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}") 