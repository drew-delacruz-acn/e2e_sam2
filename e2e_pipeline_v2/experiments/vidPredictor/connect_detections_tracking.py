#!/usr/bin/env python3
"""
Detection-Tracking Connector

This script connects the original JSON detections with the output of the video segmentation pipeline,
creating a comprehensive mapping between input detections and all frames where the objects appear.

Usage:
    python connect_detections_tracking.py --detections-dir <path> --results-dir <path> --output-file <path>
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import re


def natural_sort_key(s):
    """Key function for natural sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', str(s))]


def load_json_file(file_path: str) -> Any:
    """Load JSON data from a file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json_file(data: Any, file_path: str) -> None:
    """Save data as JSON to a file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_detection_files(detections_dir: str) -> Dict[int, List[Dict]]:
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
            detections = load_json_file(str(file_path))
            
            # Store by frame
            detections_by_frame[frame_idx] = detections
            
        except ValueError:
            print(f"Skipping file {file_path} - could not parse frame index")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(detections_by_frame)} detection files")
    return detections_by_frame


def load_tracking_results(results_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load tracking results from the pipeline output directory
    
    Args:
        results_dir: Path to directory containing tracking results
        
    Returns:
        Tuple of (tracking_results, object_frame_mapping, tracking_summary)
    """
    results_path = Path(results_dir)
    
    # Load main tracking results
    tracking_results_path = results_path / "tracking_results.json"
    tracking_results = load_json_file(str(tracking_results_path))
    
    # Load object frame mapping
    mapping_path = results_path / "object_frame_mapping.json"
    object_frame_mapping = load_json_file(str(mapping_path))
    
    # Load tracking summary (if available)
    summary_path = results_path / "tracking_summary.json"
    tracking_summary = load_json_file(str(summary_path)) if summary_path.exists() else {}
    
    print(f"Loaded tracking results with {len(tracking_results.get('objects', {}))} objects")
    return tracking_results, object_frame_mapping, tracking_summary


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: First box coordinates [x1, y1, x2, y2]
        box2: Second box coordinates [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes overlap
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate box areas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou


def match_detections_to_tracked_objects(
    detections_by_frame: Dict[int, List[Dict]], 
    tracking_results: Dict,
    object_frame_mapping: Dict
) -> Dict[int, Dict[int, Dict]]:
    """
    Match original detections to tracked objects using class and box IoU
    
    Args:
        detections_by_frame: Dictionary mapping frame indices to lists of detections
        tracking_results: Tracking results from the pipeline
        object_frame_mapping: Mapping of object IDs to frames
        
    Returns:
        Dictionary mapping frame indices to dictionaries mapping detection indices to object info
    """
    # Prepare result structure
    detection_to_object = {}
    
    # Get tracked objects
    tracked_objects = tracking_results.get("objects", {})
    
    # Process each frame
    for frame_idx, detections in detections_by_frame.items():
        detection_to_object[frame_idx] = {}
        
        # For each detection in this frame
        for det_idx, detection in enumerate(detections):
            det_label = detection.get("label", "")
            det_box = detection.get("coordinates", [])
            
            # Skip if missing key information
            if not det_label or not det_box:
                continue
            
            # Find matching tracked object
            best_match = None
            best_iou = 0.0
            
            for obj_id, obj_data in tracked_objects.items():
                obj_class = obj_data.get("class", "")
                
                # First filter by class
                if obj_class != det_label:
                    continue
                
                # Check if this object was in this frame
                all_frames = []
                for quality in ["high_quality_frames", "low_quality_frames"]:
                    if quality in object_frame_mapping.get(obj_id, {}):
                        all_frames.extend(object_frame_mapping[obj_id][quality])
                
                # Skip if object not in this frame
                if frame_idx not in all_frames:
                    continue
                
                # Find the corresponding box
                for box_idx, box in enumerate(obj_data.get("boxes", [])):
                    # Calculate IoU
                    iou = calculate_iou(det_box, box)
                    
                    # Update best match
                    if iou > best_iou:
                        best_iou = iou
                        best_match = {
                            "object_id": obj_id,
                            "iou": iou,
                            "box_idx": box_idx
                        }
            
            # If we found a good match (IoU > 0.5)
            if best_match and best_match["iou"] > 0.5:
                detection_to_object[frame_idx][det_idx] = best_match
    
    return detection_to_object


def create_detection_tracking_links(
    detections_by_frame: Dict[int, List[Dict]],
    tracking_results: Dict,
    object_frame_mapping: Dict,
    detection_to_object: Dict[int, Dict[int, Dict]]
) -> Dict:
    """
    Create comprehensive mapping between detections and tracked objects
    
    Args:
        detections_by_frame: Dictionary mapping frame indices to lists of detections
        tracking_results: Tracking results from the pipeline
        object_frame_mapping: Mapping of object IDs to frames
        detection_to_object: Mapping from detections to tracked objects
        
    Returns:
        Dictionary with the mapping structure
    """
    # Get tracked objects
    tracked_objects = tracking_results.get("objects", {})
    
    # Create result structure
    links = {
        "metadata": tracking_results.get("metadata", {}),
        "frames": {}
    }
    
    # For each frame with detections
    for frame_idx, detections in detections_by_frame.items():
        frame_links = {}
        
        # For each detection in this frame
        for det_idx, detection in enumerate(detections):
            # Check if this detection was matched to a tracked object
            if frame_idx in detection_to_object and det_idx in detection_to_object[frame_idx]:
                match = detection_to_object[frame_idx][det_idx]
                obj_id = match["object_id"]
                
                # Get object data
                obj_data = tracked_objects.get(obj_id, {})
                
                # Get all frames this object appears in
                all_appearances = []
                
                # Add high quality frames
                for frame in object_frame_mapping.get(obj_id, {}).get("high_quality_frames", []):
                    # Find the box for this frame (approximate by index order)
                    box = None
                    if frame < len(obj_data.get("boxes", [])):
                        box = obj_data["boxes"][frame]
                    
                    all_appearances.append({
                        "frame": frame,
                        "box": box,
                        "quality": "high"
                    })
                
                # Add low quality frames
                for frame in object_frame_mapping.get(obj_id, {}).get("low_quality_frames", []):
                    # Find the box for this frame (approximate by index order)
                    box = None
                    if frame < len(obj_data.get("boxes", [])):
                        box = obj_data["boxes"][frame]
                    
                    all_appearances.append({
                        "frame": frame,
                        "box": box,
                        "quality": "low"
                    })
                
                # Sort by frame number
                all_appearances.sort(key=lambda x: x["frame"])
                
                # Add to links
                frame_links[det_idx] = {
                    "detection": detection,
                    "object_id": obj_id,
                    "object_class": obj_data.get("class", ""),
                    "match_iou": match["iou"],
                    "all_appearances": all_appearances
                }
        
        # Add frame links to result
        if frame_links:
            links["frames"][str(frame_idx)] = frame_links
    
    return links


def create_updated_detections(
    detections_by_frame: Dict[int, List[Dict]],
    detection_tracking_links: Dict
) -> Dict[int, List[Dict]]:
    """
    Create updated detection files with tracking information
    
    Args:
        detections_by_frame: Original detections by frame
        detection_tracking_links: Links between detections and tracking
        
    Returns:
        Updated detections by frame
    """
    updated_detections = {}
    
    # Copy original detections
    for frame_idx, detections in detections_by_frame.items():
        updated_detections[frame_idx] = []
        
        # For each detection
        for det_idx, detection in enumerate(detections):
            # Create a copy of the detection
            updated_detection = detection.copy()
            
            # Add tracking information if available
            if str(frame_idx) in detection_tracking_links["frames"] and det_idx in detection_tracking_links["frames"][str(frame_idx)]:
                link_info = detection_tracking_links["frames"][str(frame_idx)][det_idx]
                
                # Add tracking field
                updated_detection["tracking"] = {
                    "object_id": link_info["object_id"],
                    "match_iou": link_info["match_iou"],
                    "all_appearances": link_info["all_appearances"]
                }
            
            # Add to updated detections
            updated_detections[frame_idx].append(updated_detection)
    
    return updated_detections


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Connect input detections with tracking outputs")
    parser.add_argument("--detections-dir", required=True, help="Directory containing input detection JSON files")
    parser.add_argument("--results-dir", required=True, help="Directory containing tracking results")
    parser.add_argument("--output-file", required=True, help="Output JSON file for detection-tracking links")
    parser.add_argument("--save-updated-detections", action="store_true", help="Save updated detection files with tracking info")
    parser.add_argument("--updated-detections-dir", help="Directory to save updated detection files")
    
    args = parser.parse_args()
    
    # Load detections
    detections_by_frame = load_detection_files(args.detections_dir)
    
    # Load tracking results
    tracking_results, object_frame_mapping, tracking_summary = load_tracking_results(args.results_dir)
    
    # Match detections to tracked objects
    detection_to_object = match_detections_to_tracked_objects(
        detections_by_frame, tracking_results, object_frame_mapping
    )
    
    # Create detection-tracking links
    detection_tracking_links = create_detection_tracking_links(
        detections_by_frame, tracking_results, object_frame_mapping, detection_to_object
    )
    
    # Save detection-tracking links
    save_json_file(detection_tracking_links, args.output_file)
    print(f"Saved detection-tracking links to {args.output_file}")
    
    # Create and save updated detection files if requested
    if args.save_updated_detections:
        if not args.updated_detections_dir:
            print("Error: --updated-detections-dir must be specified when using --save-updated-detections")
            return
        
        # Create updated detections
        updated_detections = create_updated_detections(
            detections_by_frame, detection_tracking_links
        )
        
        # Create output directory
        updated_dir = Path(args.updated_detections_dir)
        updated_dir.mkdir(exist_ok=True, parents=True)
        
        # Save updated detection files
        for frame_idx, detections in updated_detections.items():
            output_path = updated_dir / f"{frame_idx}.json"
            save_json_file(detections, str(output_path))
        
        print(f"Saved {len(updated_detections)} updated detection files to {args.updated_detections_dir}")


if __name__ == "__main__":
    main() 