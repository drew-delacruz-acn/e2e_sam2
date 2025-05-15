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

def get_tracking_box_for_frame(tracking_objects, obj_id, frame_idx):
    """
    Get the bounding box for an object in a specific frame from tracking data
    
    Args:
        tracking_objects: List of tracking objects
        obj_id: Object ID to find
        frame_idx: Frame number to find
        
    Returns:
        Bounding box [x1, y1, x2, y2] or None if not found
    """
    try:
        # Convert obj_id to int to ensure consistent comparison
        obj_id_int = int(obj_id)
        
        # Debug when object ID is not found
        found = False
        
        for obj in tracking_objects:
            if obj['objectID'] == obj_id_int:
                found = True
                for occurrence in obj['frameOccurences']:
                    if occurrence['frameNum'] == frame_idx:
                        return occurrence['box']
                # Debug missing frame
                print(f"  [DEBUG] Object {obj_id} exists but frame {frame_idx} not found in its occurrences")
                return None
                
        if not found:
            print(f"  [DEBUG] Object ID {obj_id} not found in tracking objects (type: {type(obj_id)})")
            print(f"  [DEBUG] Available object IDs: {[obj['objectID'] for obj in tracking_objects]}")
            
        return None
    except Exception as e:
        print(f"  [ERROR] Exception in get_tracking_box_for_frame: {e} for obj_id={obj_id} (type: {type(obj_id)})")
        return None

def match_detections_to_segments(video_segments, detections_by_frame, tracking_objects, iou_threshold=0.3):
    """
    Match CDFSOD detections to SAM2 segmented objects based on IoU and tracking information
    
    Args:
        video_segments: Dictionary mapping frame indices to segmentation results
        detections_by_frame: Dictionary mapping frame indices to CDFSOD detections
        tracking_objects: List of tracking objects with frame occurrences
        iou_threshold: Minimum IoU value to consider a match (default: 0.3)
        
    Returns:
        Dictionary mapping object IDs to their detection matches
    """
    print("\n=== DEBUG: Matching detections to segments ===")
    object_detections = {}
    
    # Validate that video_segments is not empty
    if not video_segments:
        print("WARNING: video_segments dictionary is empty!")
        print("No segments to match with detections.")
        return {}
    
    # First, establish object ID to class mapping from tracking_objects
    object_class_map = {}
    for obj in tracking_objects:
        obj_id = obj['objectID']
        obj_class = obj['objectName']
        object_class_map[obj_id] = obj_class
    
    print(f"Found {len(object_class_map)} objects in tracking data")
    print(f"Object classes: {object_class_map}")
    
    # Debug info about video segments
    print(f"\nVideo segments info:")
    print(f"Total frames with segments: {len(video_segments)}")
    
    # Count objects per frame
    all_objects = set()
    for frame_idx, segments in video_segments.items():
        objects_in_frame = list(segments.keys())
        all_objects.update([int(obj_id) for obj_id in objects_in_frame])
        print(f"  Frame {frame_idx}: {len(objects_in_frame)} objects: {objects_in_frame}")
    
    print(f"Total unique objects in video segments: {len(all_objects)}")
    print(f"Object IDs in video segments: {sorted(all_objects)}")
    print(f"Object IDs in tracking data: {sorted(object_class_map.keys())}")
    
    # Check for mismatches
    missing_in_tracking = [obj_id for obj_id in all_objects if obj_id not in object_class_map]
    missing_in_segments = [obj_id for obj_id in object_class_map if obj_id not in all_objects]
    
    if missing_in_tracking:
        print(f"WARNING: {len(missing_in_tracking)} objects in segments but not in tracking: {missing_in_tracking}")
    
    if missing_in_segments:
        print(f"WARNING: {len(missing_in_segments)} objects in tracking but not in segments: {missing_in_segments}")
    
    # Next, build initial detections directly from tracking information
    for obj in tracking_objects:
        obj_id = obj['objectID']
        if obj_id not in object_detections:
            object_detections[obj_id] = []
        
        # Add each frame occurrence as a detection
        for occurrence in obj['frameOccurences']:
            frame_num = occurrence['frameNum']
            confidence = occurrence.get('confidence', 0.0)
            
            # Check if there's a detection in this frame
            if frame_num in detections_by_frame and len(detections_by_frame[frame_num]) > 0:
                # Find the matching detection based on IoU
                tracking_box = occurrence['box']
                
                # Find all matching detections based on IoU
                matches = []
                for detection in detections_by_frame[frame_num]:
                    detection_box = detection['coordinates']
                    iou = calculate_iou(tracking_box, detection_box)
                    
                    if iou >= iou_threshold:
                        matches.append((iou, detection))
                
                # If we have matches, use the best one
                if matches:
                    matches.sort(reverse=True, key=lambda x: x[0])
                    best_match = matches[0][1]
                    
                    # Add to object's detections
                    object_detections[obj_id].append({
                        'frameNumber': frame_num,
                        'class': best_match['label'],
                        'confidence': best_match['confidence']
                    })
                    print(f"Added detection for object {obj_id} ({best_match['label']}) at frame {frame_num}")
                else:
                    # If no match found, just use the object's class
                    object_detections[obj_id].append({
                        'frameNumber': frame_num,
                        'class': object_class_map[obj_id],
                        'confidence': confidence
                    })
                    print(f"No matching detection found, using object class: {object_class_map[obj_id]} at frame {frame_num}")
    
    # Print out summary of detected objects
    for obj_id, detections in object_detections.items():
        print(f"Object {obj_id} ({object_class_map.get(obj_id, 'Unknown')}) has {len(detections)} detections")
        if len(detections) > 0:
            frame_numbers = [d['frameNumber'] for d in detections]
            print(f"  Frame numbers: {frame_numbers}")
    
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
    print("\n=== DEBUG: Creating segmentation summary ===")
    result = []
    
    # Validate inputs
    if not video_segments:
        print("WARNING: video_segments is empty, no segmentation results to summarize")
        return []
    
    # Get all unique object IDs
    object_ids = set()
    for frame_idx, segments in video_segments.items():
        for obj_id in segments.keys():
            object_ids.add(int(obj_id))
    
    print(f"Found {len(object_ids)} unique object IDs in video segments")
    print(f"Object IDs: {sorted(object_ids)}")
    
    # Check which objects have CDFSOD predictions
    cdfsod_objects = set(object_detections.keys())
    print(f"Found {len(cdfsod_objects)} objects with CDFSOD predictions")
    print(f"CDFSOD object IDs: {sorted(cdfsod_objects)}")
    
    # Objects missing CDFSOD predictions
    missing_cdfsod = object_ids - cdfsod_objects
    if missing_cdfsod:
        print(f"WARNING: {len(missing_cdfsod)} objects without CDFSOD predictions: {sorted(missing_cdfsod)}")
    
    # Objects with CDFSOD predictions but no segmentation
    missing_segments = cdfsod_objects - object_ids
    if missing_segments:
        print(f"WARNING: {len(missing_segments)} objects with CDFSOD predictions but no segmentation: {sorted(missing_segments)}")
    
    # Process each object
    for obj_id in sorted(object_ids):
        # Convert obj_id to int for consistent comparison
        obj_id_int = int(obj_id)
        
        print(f"\nProcessing object {obj_id_int}:")
        
        # Get all frames where this object appears
        appearances = []
        frame_appearances = []
        
        for frame_idx in sorted(video_segments.keys()):
            # Make sure we're comparing correctly (string vs int)
            str_obj_id = str(obj_id_int)
            
            # Debug the object types
            if frame_idx not in video_segments:
                print(f"  Frame {frame_idx} not in video_segments, skipping")
                continue
                
            # Debug segment keys
            segment_keys = list(video_segments[frame_idx].keys())
            print(f"  Frame {frame_idx} has objects: {segment_keys}")
            
            if str_obj_id in video_segments[frame_idx]:
                print(f"  Object {obj_id_int} found in frame {frame_idx}")
                
                # Try to get box from tracking data first (more accurate)
                box = get_tracking_box_for_frame(tracking_objects, obj_id_int, frame_idx)
                
                # If not found, compute from mask
                if box is None:
                    print(f"  No tracking box found, computing from mask")
                    mask = video_segments[frame_idx][str_obj_id]
                    box = get_bounding_box_from_mask(mask)
                    print(f"  Computed box from mask: {box}")
                else:
                    print(f"  Found tracking box: {box}")
                
                appearances.append({
                    'frameNum': frame_idx,
                    'boundingBox': box
                })
                
                frame_appearances.append(frame_idx)
        
        # Get all detections for this object
        cdfsod_predictions = object_detections.get(obj_id_int, [])
        
        # Debug the appearance vs predictions
        print(f"Object {obj_id_int}: {len(appearances)} appearances, {len(cdfsod_predictions)} CDFSOD predictions")
        print(f"  Appearances in frames: {frame_appearances}")
        
        if cdfsod_predictions:
            pred_frames = [p['frameNumber'] for p in cdfsod_predictions]
            print(f"  CDFSOD predictions in frames: {pred_frames}")
            
            # Debug frames with predictions but no appearances
            pred_no_app = [f for f in pred_frames if f not in frame_appearances]
            if pred_no_app:
                print(f"  WARNING: {len(pred_no_app)} frames with predictions but no appearances: {pred_no_app}")
                
            # Debug frames with appearances but no predictions
            app_no_pred = [f for f in frame_appearances if f not in pred_frames]
            if app_no_pred:
                print(f"  WARNING: {len(app_no_pred)} frames with appearances but no predictions: {app_no_pred}")
        
        # Create object summary
        obj_summary = {
            'samObjectId': int(obj_id_int),
            'samDictatedAppearances': appearances,
            'cdfsodPredictions': cdfsod_predictions
        }
        
        result.append(obj_summary)
    
    # Final summary
    print(f"\nCreated summary for {len(result)} objects")
    for obj_summary in result:
        obj_id = obj_summary['samObjectId']
        n_appearances = len(obj_summary['samDictatedAppearances'])
        n_predictions = len(obj_summary['cdfsodPredictions'])
        print(f"Object {obj_id}: {n_appearances} appearances, {n_predictions} predictions")
    
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