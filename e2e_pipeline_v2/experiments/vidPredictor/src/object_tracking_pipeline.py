#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple, Union, Optional
import time
import re

# Add natural sorting function
def natural_sort_key(s):
    """Key function for natural sorting"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', str(s))]

# Import our components
from owlv2_detector import OWLv2Detector
from cd_fsod_detector import CDFSODDetector  # Add import for CD-FSOD detector
from sam2_wrapper import SAM2VideoWrapper
from object_tracker import ObjectTracker
from embedding_extractor import EmbeddingExtractor

class ObjectTrackingPipeline:
    def __init__(
        self,
        owlv2_checkpoint: str,
        sam2_checkpoint: str,
        sam2_config: str,
        output_dir: str,
        confidence_threshold: float = 0.1,
        device: Optional[torch.device] = None,
        detector_type: str = "owlv2",  # New parameter for detector type
        cd_fsod_path: Optional[str] = None,  # Path to CD-FSOD JSON directory
        min_gap_frames: int = 10,  # Min gap frames for CD-FSOD detector
        mask_quality_threshold: int = 0,  # Minimum pixel count for mask quality assessment
    ):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Create detector based on type
        self.detector_type = detector_type
        self.detector = self._create_detector(
            detector_type=detector_type,
            owlv2_checkpoint=owlv2_checkpoint,
            cd_fsod_path=cd_fsod_path,
            confidence_threshold=confidence_threshold,
            min_gap_frames=min_gap_frames,
            device=self.device
        )
        
        # Initialize tracking components
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.confidence_threshold = confidence_threshold
        self.tracker = ObjectTracker()
        self.embedding_extractor = EmbeddingExtractor(device=self.device)
        
        # Create SAM2 wrapper
        self.sam_wrapper = SAM2VideoWrapper(
            checkpoint_path=sam2_checkpoint,
            config_path=sam2_config,
            device=self.device
        )
        
        # Store tracked objects and propagation results
        self.tracked_objects = {}  # Store tracked objects with masks
        self.propagation_results = {}  # Store SAM2 propagation results
        self.boxes_by_frame = {}  # Store boxes by frame, including fallbacks
        
        # Store quality threshold for mask assessment
        self.mask_quality_threshold = mask_quality_threshold
        
        # Tracking state
        self.next_id = 1
        
    def _create_detector(
        self,
        detector_type: str,
        owlv2_checkpoint: str,
        cd_fsod_path: Optional[str] = None,
        confidence_threshold: float = 0.1,
        min_gap_frames: int = 10,
        device: torch.device = None
    ):
        """
        Factory method to create the appropriate detector.
        
        Args:
            detector_type: Type of detector ('owlv2' or 'cd_fsod')
            owlv2_checkpoint: Path to OWLv2 checkpoint file
            cd_fsod_path: Path to CD-FSOD JSON directory
            confidence_threshold: Minimum confidence threshold
            min_gap_frames: Minimum gap frames for CD-FSOD detector (used for tracking)
            device: Torch device for OWLv2 detector
            
        Returns:
            Initialized detector object
            
        Notes:
            - The 'owlv2' detector processes all detections in each frame
            - The 'cd_fsod' detector also processes all detections in each frame (not just first 
              appearances and reappearances), making its behavior consistent with OWLv2
        """
        if detector_type == "owlv2":
            return OWLv2Detector(device=device)
        elif detector_type == "cd_fsod":
            if cd_fsod_path is None:
                raise ValueError("cd_fsod_path must be provided when using CD-FSOD detector")
            # The CD-FSOD detector now processes all detections in each frame like OWLv2
            # We still track first appearances and reappearances internally for reference,
            # but all detections are returned when detect() is called
            return CDFSODDetector(
                json_dir=cd_fsod_path,
                confidence_threshold=confidence_threshold,
                min_gap_frames=min_gap_frames
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. Must be 'owlv2' or 'cd_fsod'")
        
    def process_video(self, frames_dir: str, text_queries: List[str]):
        # Get all frames sorted using natural sort
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")], key=natural_sort_key)
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Processing {len(frame_files)} frames with queries: {text_queries}")
        print(f"Using detector: {self.detector_type}")
        
        # Initialize SAM2 with the video frames directory
        print(f"Setting up SAM2 with frames directory: {frames_dir}")
        self.sam_wrapper.set_video(frames_dir=frames_dir)
        
        # Results storage
        results = {
            "object_tracks": {},
            "frame_results": {},
            "metadata": {
                "queries": text_queries,
                "frame_count": len(frame_files),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detector_type": self.detector_type  # Include detector type in metadata
            }
        }
        
        # Process first frame: detect → segment → initialize tracking
        first_frame_path = frame_files[0]
        first_frame = Image.open(first_frame_path).convert("RGB")
        first_frame_np = np.array(first_frame)
        
        # Extract actual frame index from filename
        first_frame_idx = self._extract_frame_idx_from_path(first_frame_path)
        print(f"Processing first frame with extracted index: {first_frame_idx}")
        
        # Detect objects in first frame
        # For CD-FSOD, we pass the frame filename to help extract the frame index
        first_frame_data = first_frame
        if self.detector_type == "cd_fsod":
            # For CD-FSOD detector, we need to provide frame information
            # We'll use the extracted frame index to ensure proper JSON matching
            first_frame_data = {
                "image": first_frame,
                "frame_path": str(first_frame_path),
                "frame_idx": first_frame_idx
            }
            
        detections = self.detector.detect(
            image=first_frame_data,
            text_queries=text_queries,
            threshold=self.confidence_threshold
        )
        
        # Initialize object tracking
        print(f' Detections {detections}')
        for i, (box, label, conf) in enumerate(zip(detections["boxes"], detections["labels"], detections["scores"])):
            if conf < self.confidence_threshold:
                continue
                
            # Convert box to XYXY format if needed
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure box coordinates are valid
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(first_frame_np.shape[1], x2)
            y2 = min(first_frame_np.shape[0], y2)
            
            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2:
                print(f"Skipping invalid box: {[x1, y1, x2, y2]}")
                continue
                
            box_area = first_frame_np[y1:y2, x1:x2]
            
            # Extract embedding for this object
            try:
                embedding = self.embedding_extractor.extract(box_area)
            except Exception as e:
                print(f"Error extracting embedding: {e}")
                continue
            
            # Create a unique object ID first
            object_id = self.next_id
            self.next_id += 1
            
            # Add box to SAM2 to get mask - match test pattern
            box_coords = [x1, y1, x2, y2]
            print(f"Adding box for object {object_id} at frame {first_frame_idx}: {box_coords}")
            mask_logits = self.sam_wrapper.add_box(frame_idx=first_frame_idx, obj_id=object_id, box=box_coords)
            
            # Skip if mask generation failed
            if mask_logits is None:
                print(f"Failed to generate mask for object {object_id}")
                continue
            
            # Store this object with the ID
            self.tracked_objects[object_id] = {
                "id": object_id,
                "class": label,
                "first_detected": first_frame_idx,  # Use extracted frame index
                "boxes": [box_coords],
                "embeddings": [embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding],
                "masks": [mask_logits.cpu().numpy() if isinstance(mask_logits, torch.Tensor) else mask_logits],
                "last_seen": first_frame_idx,  # Use extracted frame index
                "confidence": [conf]
            }
            
            # Store in results
            results["object_tracks"][object_id] = self.tracked_objects[object_id]
        
        # Run mask propagation for all tracked objects
        if self.tracked_objects:
            print("Running mask propagation for all tracked objects...")
            object_ids = list(self.tracked_objects.keys())
            segments, boxes_by_frame = self.sam_wrapper.propagate_masks(objects_to_track=object_ids)
            self.propagation_results = segments
            results["propagated_boxes_by_frame"] = boxes_by_frame
            print(f"Propagated masks for {len(object_ids)} objects across {len(segments)} frames")
        else:
            print("No objects to propagate")
        
        # Visualize first frame results
        first_frame_vis = self._visualize_frame(
            frame=first_frame_np,
            frame_idx=first_frame_idx,  # Use extracted frame index
            objects=self.tracked_objects
        )
        
        # Save first frame results
        results["frame_results"][first_frame_idx] = {  # Use extracted frame index
            "detections": [
                {"box": box.tolist() if isinstance(box, torch.Tensor) else box, 
                 "label": label, 
                 "confidence": conf}
                for box, label, conf in zip(detections["boxes"], detections["labels"], detections["scores"])
                if conf >= self.confidence_threshold
            ],
            "tracked_objects": [obj_id for obj_id in self.tracked_objects.keys()]
        }
        
        # Process remaining frames - use propagation results where available
        for i in range(1, len(frame_files)):
            frame_path = frame_files[i]
            # Extract actual frame index from filename
            frame_idx = self._extract_frame_idx_from_path(frame_path)
            print(f"Processing frame {i}/{len(frame_files)} with extracted index: {frame_idx}")
            
            frame = Image.open(frame_path).convert("RGB")
            frame_np = np.array(frame)
            
            # Get new detections
            # Prepare frame data based on detector type
            frame_data = frame
            if self.detector_type == "cd_fsod":
                # For CD-FSOD detector, we need to provide frame information
                frame_data = {
                    "image": frame,
                    "frame_path": str(frame_path),
                    "frame_idx": frame_idx
                }
                
            detections = self.detector.detect(
                image=frame_data,
                text_queries=text_queries,
                threshold=self.confidence_threshold
            )
            
            # Track objects across frames
            # Convert detections to the format expected by the tracker
            tracker_detections = []
            for i in range(len(detections["boxes"])):
                tracker_detections.append({
                    "box": detections["boxes"][i].tolist() if isinstance(detections["boxes"][i], torch.Tensor) else detections["boxes"][i],
                    "score": detections["scores"][i].item() if isinstance(detections["scores"][i], torch.Tensor) else detections["scores"][i],
                    "text": detections["labels"][i]
                })


            # Get boxes for current frame from tracker
            current_boxes = self.tracker.update_tracks(
                frame=frame_np,
                frame_idx=frame_idx,  # Use extracted frame index
                detections=tracker_detections,
                embedding_extractor=self.embedding_extractor,
                output_dir=None
            )

            # First, identify objects that are brand new in this frame by checking the tracker's output
            # and cross-referencing with objects we're already tracking in our pipeline
            new_objects_in_this_frame = []
            for obj_id, box in current_boxes.items():
                # Check if this object ID already exists in our pipeline tracking but was just updated
                # or if it's completely new to the pipeline
                if obj_id not in self.tracked_objects:
                    # Brand new object - get info from tracker
                    tracker_obj_data = self.tracker.tracked_objects[obj_id]
                    
                    # Create a new entry in our pipeline's tracking
                    embedding = tracker_obj_data["embedding"]
                    
                    # Initialize with empty data structures that will be populated
                    self.tracked_objects[obj_id] = {
                        "id": obj_id,
                        "class": tracker_obj_data["class"],
                        "first_detected": frame_idx,  # Mark as detected in current frame
                        "boxes": [box],  # Start with current box
                        "embeddings": [embedding.copy() if isinstance(embedding, np.ndarray) else embedding],
                        "masks": [],  # Will be filled below
                        "last_seen": frame_idx,
                        "confidence": [0.0]  # We don't have the original confidence, use placeholder
                    }
                    
                    # Add to list of new objects
                    new_objects_in_this_frame.append(obj_id)
                    print(f"Created new pipeline-tracked object {obj_id} at frame {frame_idx}")
            
            # Process each tracked object - use propagated masks if available
            # First, find objects that were updated in this frame
            tracked_in_this_frame = []
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data["last_seen"] == frame_idx:
                    tracked_in_this_frame.append(obj_id)

            for obj_id in tracked_in_this_frame:
                obj_data = self.tracked_objects[obj_id]
                
                # Check if this is a new object by checking if it's in our new objects list
                # or by checking the first_detected field matches the current frame
                is_new_object = (obj_id in new_objects_in_this_frame) or (obj_data["first_detected"] == frame_idx)
                
                # If this is a new object, we need to initialize it in SAM2
                if is_new_object:
                    box = obj_data["boxes"][-1]
                    print(f"Adding new object {obj_id} at frame {frame_idx}")
                    mask_logits = self.sam_wrapper.add_box(frame_idx=frame_idx, obj_id=obj_id, box=box)
                    
                    if mask_logits is None:
                        print(f"Failed to generate mask for new object {obj_id}")
                        # Use a fallback empty mask
                        mask_logits = np.zeros((frame_np.shape[0], frame_np.shape[1]), dtype=bool)
                    
                    obj_data["masks"].append(
                        mask_logits.cpu().numpy() if isinstance(mask_logits, torch.Tensor) else mask_logits
                    )
                # For existing objects, use propagated masks if available
                else:
                    # Check if we have a propagated mask for this object at this frame
                    if frame_idx in self.propagation_results and obj_id in self.propagation_results[frame_idx]:
                        propagated_mask = self.propagation_results[frame_idx][obj_id]
                        obj_data["masks"].append(
                            propagated_mask.cpu().numpy() if isinstance(propagated_mask, torch.Tensor) else propagated_mask
                        )
                    else:
                        # Fallback: use the previous mask if propagation didn't yield a result
                        prev_mask = obj_data["masks"][-1]
                        print(f"Warning: No propagated mask for object {obj_id} at frame {frame_idx}, using previous mask")
                        obj_data["masks"].append(prev_mask)
                
                # Update results
                results["object_tracks"][obj_id] = obj_data
            
            # Run mask propagation if we found new objects in this frame
            if new_objects_in_this_frame:
                print(f"Found {len(new_objects_in_this_frame)} new objects in frame {frame_idx}, running propagation...")
                try:
                    print(f"Running propagation for object {obj_id}...")
                    result = self.sam_wrapper.propagate_masks(objects_to_track=[obj_id])
                    print(f"DEBUG: propagate_masks(objects_to_track={obj_id}) returned type: {type(result)}")
                    if isinstance(result, tuple):
                        print(f"DEBUG: propagate_masks(objects_to_track={obj_id}) tuple length: {len(result)}")
                        segments, boxes_by_frame = result
                    else:
                        print(f"DEBUG: propagate_masks(objects_to_track={obj_id}) value: {result}")
                        segments = result
                        boxes_by_frame = {}  # Initialize empty dict if no boxes returned
                    
                    # Store all propagation results
                    for frame_idx, masks in segments.items():
                        if frame_idx not in self.propagation_results:
                            self.propagation_results[frame_idx] = {}
                        
                        # Only copy if the propagation returned masks for this object
                        if obj_id in masks:
                            self.propagation_results[frame_idx][obj_id] = masks[obj_id]
                            
                            # Analyze mask quality
                            mask = masks[obj_id]
                            pixel_count = self._count_mask_pixels(mask)
                            quality_status = "HIGH QUALITY" if pixel_count >= self.mask_quality_threshold else "LOW QUALITY" if pixel_count > 0 else "EMPTY"
                        
                    # If we also have boxes_by_frame, store those too
                    if isinstance(boxes_by_frame, dict) and boxes_by_frame:
                        for frame_idx, boxes in boxes_by_frame.items():
                            if frame_idx not in self.boxes_by_frame:
                                self.boxes_by_frame[frame_idx] = {}
                            
                            # Only copy if boxes exist for this object
                            if obj_id in boxes:
                                self.boxes_by_frame[frame_idx][obj_id] = boxes[obj_id]
                                
                                # Check fallback box silently
                                if isinstance(boxes[obj_id], dict) and boxes[obj_id].get("is_fallback", False):
                                    pass  # No logging
                    
                    # NEW CODE: Update tracked_objects with all boxes from propagation
                    all_boxes = []
                    for frame_idx in sorted(boxes_by_frame.keys()):
                        if obj_id in boxes_by_frame[frame_idx]:
                            box_data = boxes_by_frame[frame_idx][obj_id]
                            if isinstance(box_data, dict):
                                box = box_data["box"]
                                all_boxes.append(box)
                                print(f"Adding box for frame {frame_idx} to object {obj_id}: {box}")
                            else:
                                all_boxes.append(box_data)
                                print(f"Adding raw box for frame {frame_idx} to object {obj_id}: {box_data}")
                    
                    # Add logging to debug the tracked_objects dictionary
                    print(f"DEBUG: tracked_objects keys before update: {list(self.tracked_objects.keys())}")
                    print(f"DEBUG: Is object {obj_id} in tracked_objects? {obj_id in self.tracked_objects}")
                    print(f"DEBUG: Number of boxes collected for object {obj_id}: {len(all_boxes)}")
                    
                    # Only update if we have boxes
                    if not all_boxes:
                        print(f"WARNING: No boxes collected for object {obj_id}, skipping tracked_objects update")
                        continue
                    
                    # Update tracked_objects with the new boxes, with proper error handling
                    if obj_id in self.tracked_objects:
                        self.tracked_objects[obj_id]["boxes"] = all_boxes
                        # Also collect and store masks
                        all_masks = []
                        for frame_idx in sorted(self.propagation_results.keys()):
                            if obj_id in self.propagation_results[frame_idx]:
                                all_masks.append(self.propagation_results[frame_idx][obj_id])
                        
                        # Store masks if we have them
                        if all_masks:
                            self.tracked_objects[obj_id]["masks"] = all_masks
                            print(f"Stored {len(all_masks)} masks for object {obj_id}")
                        
                        print(f"Successfully updated boxes for object {obj_id}")
                    else:
                        # Create the object entry if it doesn't exist
                        print(f"Object {obj_id} not found in tracked_objects. Creating entry.")
                        self.tracked_objects[obj_id] = {
                            "id": obj_id,
                            "class": obj_data["class"],  # Use the class from the original detected object
                            "first_detected": obj_data["first_detected"],  # Use the original detection frame
                            "last_seen": frame_idx,
                            "confidence": [obj_data.get("confidence", [1.0])[0] if isinstance(obj_data.get("confidence"), list) else obj_data.get("confidence", 1.0)],
                            "boxes": all_boxes,
                            "masks": obj_data.get("masks", [])  # Add masks with empty list as default
                        }
                        print(f"Created new entry for object {obj_id}")
                
                    print(f"Successfully propagated masks for object {obj_id}, available in {len(segments)} frames")
                except Exception as e:
                    print(f"Error during propagation for object {obj_id}: {e}")
                    import traceback
                    print(f"Propagation error traceback: {traceback.format_exc()}")
                    
            # Store updated object data only if it doesn't exist yet
            print(f"DEBUG: After propagation - Is object {obj_id} in tracked_objects? {obj_id in self.tracked_objects}")
            if obj_id not in self.tracked_objects:
                print(f"Adding object {obj_id} to tracked_objects after propagation")
                self.tracked_objects[obj_id] = obj_data
            else:
                print(f"Object {obj_id} already exists in tracked_objects, keeping existing entry")
                
            # Always update the results dict
            results["object_tracks"][obj_id] = obj_data
        
        # Generate visualizations for all frames
        for i, frame_path in enumerate(frame_files):
            # Extract the actual frame index
            frame_idx = self._extract_frame_idx_from_path(frame_path)
            
            # Get visible objects for this frame
            visible_objects = {
                obj_id: data for obj_id, data in self.tracked_objects.items()
                if data["first_detected"] <= frame_idx <= data["last_seen"]
            }
            
            # Load the frame
            frame = np.array(Image.open(frame_path).convert("RGB"))
            
            # Visualize without saving
            self._visualize_frame(frame=frame, frame_idx=frame_idx, objects=visible_objects)
        
        # Save per-object visualizations
        self.save_per_object_visualizations(frames_dir)
        
        # Save first detection frames
        self.save_first_detections(frames_dir)
        
        # Save mapping of objects to frames they appear in
        self.save_object_frame_mapping()
        
        # Prepare a streamlined version of tracking results without redundant data
        streamlined_results = {
            "metadata": {
                "queries": text_queries,
                "frame_count": len(frame_files),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detector_type": self.detector_type
            },
            "objects": {}
        }
        
        # Only include essential tracking data without masks
        for obj_id, obj_data in self.tracked_objects.items():
            # Ensure masks are always a list
            masks = obj_data.get("masks", [])
            if not isinstance(masks, list):
                masks = [masks]
                
            streamlined_results["objects"][str(obj_id)] = {
                "id": obj_id,
                "class": obj_data["class"],
                "boxes": obj_data["boxes"],  # Keep only the ID, class, and boxes
                # Make sure we keep masks for visualization
                "masks": masks  # Always store masks as a list
                # Removed: first_detected, last_seen, confidence
            }
        
        # Save streamlined tracking results
        with open(self.output_dir / "tracking_results.json", "w") as f:
            json_results = self._prepare_for_json(streamlined_results)
            json.dump(json_results, f, indent=2)
        
        print(f"All results saved to: {self.output_dir}")
        return streamlined_results
    
    def _visualize_frame(self, frame, frame_idx, objects):
        """Visualize objects on frame and save to output directory"""
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Get a colormap for object IDs
        colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(objects))))
        
        # Draw each object
        for i, (obj_id, obj_data) in enumerate(objects.items()):
            # Only show objects visible in this frame
            if obj_data["last_seen"] != frame_idx:
                continue
                
            # Get color for this object
            color = colors[i % len(colors)]
            color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Draw bounding box
            box = obj_data["boxes"][-1]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color_rgb, 2)
            
            # Draw mask overlay
            if "masks" in obj_data:
                if isinstance(obj_data["masks"], list) and obj_data["masks"]:
                    mask = obj_data["masks"][-1]
                else:
                    mask = obj_data["masks"]
                
                if isinstance(mask, np.ndarray):
                    # Convert logits to binary mask if needed
                    if mask.dtype == np.float32 or mask.dtype == np.float64:
                        mask = mask > 0
                    
                    # Create binary mask (ensure it's 2D)
                    if len(mask.shape) > 2:
                        mask = np.squeeze(mask)
                    
                    # Check if mask is valid
                    if mask.size == 0 or mask.ndim != 2:
                        print(f"Warning: Invalid mask for object {obj_id}, shape: {mask.shape}")
                        continue
                    
                    # Convert to bool and ensure shape is compatible
                    mask_bool = mask.astype(bool)
                    
                    try:
                        # Create a colored mask image
                        colored_mask = np.zeros_like(vis_frame)
                        colored_mask[mask_bool] = color_rgb  # Use RGB without alpha
                        
                        # Blend the mask with the original frame
                        alpha = 0.5
                        vis_frame = cv2.addWeighted(colored_mask, alpha, vis_frame, 1.0, 0)
                    except Exception as e:
                        print(f"Error applying mask for object {obj_id}: {e}")
                        continue
            
            # Draw label
            label = f"{obj_data['class']} #{obj_id}"
            if "confidence" in obj_data:
                if isinstance(obj_data["confidence"], list) and obj_data["confidence"]:
                    conf = obj_data["confidence"][-1]
                else:
                    conf = obj_data["confidence"]
                text = f"{label} ({conf:.2f})"
            else:
                text = label
            cv2.putText(vis_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
        
        # Return visualization (removed saving to disk)
        return vis_frame
    
    def _prepare_for_json(self, data):
        """Convert numpy arrays and tensors to Python lists for JSON serialization"""
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            return str(data)

    def save_first_detections(self, frames_dir):
        """Save the first detection frame for each tracked object with bounding box"""
        print("Saving first detection frames for each object...")
        
        # Create directory for first detections
        first_detections_dir = self.output_dir / "first_detections"
        first_detections_dir.mkdir(exist_ok=True, parents=True)
        
        # Get frame paths with natural sorting
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")], key=natural_sort_key)
        
        # Create mapping from frame indices to frame files
        frame_map = {}
        for frame_path in frame_files:
            idx = self._extract_frame_idx_from_path(frame_path)
            frame_map[idx] = frame_path
        
        # Get detection files if using CD-FSOD detector
        detection_files = None
        if hasattr(self, 'detector') and hasattr(self.detector, 'json_dir'):
            detector_json_dir = Path(self.detector.json_dir)
            detection_files = sorted([f for f in detector_json_dir.glob("*.json")], key=natural_sort_key)
            print(f"Using detection files from: {detector_json_dir}")
        
        # For each tracked object
        for obj_id, obj_data in self.tracked_objects.items():
            # Get frame where object was first detected
            first_frame_idx = obj_data["first_detected"]
            
            # Skip if index is not in our frame map
            if first_frame_idx not in frame_map:
                print(f"Warning: First frame index {first_frame_idx} for object {obj_id} not found in frame map")
                continue
            
            # Log the detection source JSON file
            if detection_files is not None:
                matching_json = None
                for json_file in detection_files:
                    if int(json_file.stem) == first_frame_idx:
                        matching_json = json_file
                        break
                
                if matching_json:
                    print(f"  Object #{obj_id} ({obj_data['class']}) first detected in frame {first_frame_idx}, using {matching_json.name}")
                    # Log the JSON content for the first few detections in that file
                    try:
                        with open(matching_json, 'r') as f:
                            json_data = json.load(f)
                            print(f"    JSON file contains {len(json_data)} detections")
                            # Show the first detection for each class to avoid too much output
                            classes_shown = set()
                            for detection in json_data:
                                label = detection.get('label', 'unknown')
                                if label not in classes_shown and label == obj_data['class']:
                                    confidence = detection.get('confidence', 0)
                                    coords = detection.get('coordinates', [])
                                    print(f"    Sample detection for {label}: confidence={confidence:.4f}, coordinates={coords}")
                                    classes_shown.add(label)
                                    break
                    except Exception as e:
                        print(f"    Error reading JSON file: {e}")
            
            # Load the frame
            frame_path = frame_map[first_frame_idx]
            frame = np.array(Image.open(frame_path).convert("RGB"))
            
            # Draw bounding box (get the first box from the boxes list)
            if not obj_data["boxes"]:
                print(f"Warning: No boxes available for object {obj_id}")
                continue
                
            detection_box = obj_data["boxes"][0]
            x1, y1, x2, y2 = map(int, detection_box)
            
            # Get color for this object - use plt.cm which works across matplotlib versions
            color = plt.cm.tab10(obj_id % 10)[:3]
            color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_rgb, 2)
            
            # Add label
            label = f"Object #{obj_id}: {obj_data['class']}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_rgb, 2)
            
            # Save the annotated frame
            output_path = first_detections_dir / f"object_{obj_id}_{obj_data['class']}_first_detection.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"  Saved first detection frame for object #{obj_id} ({obj_data['class']})")
        
        print(f"All first detection frames saved to {first_detections_dir}")

    def save_per_object_visualizations(self, frames_dir):
        """Save per-object visualizations with masks overlaid on original frames based on mask quality"""
        print("Saving per-object segmentation visualizations based on mask quality...")
        
        # Create base directory for object masks
        object_masks_dir = self.output_dir / "object_masks"
        object_masks_dir.mkdir(exist_ok=True, parents=True)
        
        # Get frame paths with natural sorting
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")], key=natural_sort_key)
        
        # Create mapping from frame indices to frame files
        frame_map = {}
        for frame_path in frame_files:
            idx = self._extract_frame_idx_from_path(frame_path)
            frame_map[idx] = frame_path
        
        # Initialize structure to store quality metrics for each object
        object_quality_stats = {}
        for obj_id in self.tracked_objects.keys():
            object_quality_stats[obj_id] = {
                "high_quality_count": 0,  # Frames with pixel count > threshold
                "low_quality_count": 0,   # Frames with 0 < pixel count <= threshold
                "empty_mask_count": 0,    # Frames with 0 pixels
                "frames_saved": 0,        # Total frames saved
                "frames_processed": 0,    # Total frames processed
                "high_quality_frames": [],
                "low_quality_frames": [],
                "empty_mask_frames": []
            }
            
            # Create directory for this object
            obj_dir = object_masks_dir / f"object_{obj_id}_{self.tracked_objects[obj_id]['class']}"
            obj_dir.mkdir(exist_ok=True)
            
        # Process all frames in propagation results
        for frame_idx in sorted(self.propagation_results.keys()):
            # Skip if frame is not in our frame map
            if frame_idx not in frame_map:
                continue
            
            # Get objects that have masks in this frame
            frame_objects = self.propagation_results[frame_idx]
            
            # Process each object with a mask in this frame
            for obj_id, mask in frame_objects.items():
                # Skip if object is not in tracked_objects (shouldn't happen, but just in case)
                if obj_id not in self.tracked_objects:
                    print(f"  Warning: Object {obj_id} has a mask but is not in tracked_objects")
                    continue
                
                # Get object data
                obj_data = self.tracked_objects[obj_id]
                obj_class = obj_data.get('class', 'unknown')
                
                # Skip if mask is None or dimensionally invalid
                if mask is None or (isinstance(mask, np.ndarray) and mask.size == 0):
                    print(f"  Skipping frame {frame_idx} for object {obj_id} - empty mask")
                    continue
                
                # Count frame as processed
                object_quality_stats[obj_id]["frames_processed"] += 1
                
                # Count true pixels in mask to assess quality
                pixel_count = self._count_mask_pixels(mask)
                
                # Skip frames with zero pixels (empty masks)
                if pixel_count == 0:
                    object_quality_stats[obj_id]["empty_mask_count"] += 1
                    object_quality_stats[obj_id]["empty_mask_frames"].append(frame_idx)
                    continue  # Skip saving this frame
                
                # Load the original frame
                frame_path = frame_map[frame_idx]
                frame = np.array(Image.open(frame_path).convert("RGB"))
                
                # Create visualization with just this object's mask
                vis_frame = frame.copy()
                
                # Get a color for this object (consistent with pipeline visualization)
                color = plt.cm.tab10(obj_id % 10)[:3]
                color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                
                # Categorize mask quality
                is_high_quality = pixel_count > self.mask_quality_threshold
                
                # Update statistics based on quality
                if is_high_quality:
                    object_quality_stats[obj_id]["high_quality_count"] += 1
                    object_quality_stats[obj_id]["high_quality_frames"].append(frame_idx)
                else:
                    object_quality_stats[obj_id]["low_quality_count"] += 1
                    object_quality_stats[obj_id]["low_quality_frames"].append(frame_idx)
                
                # Check if we're using a fallback box
                is_fallback = False
                if hasattr(self, 'boxes_by_frame') and frame_idx in self.boxes_by_frame and obj_id in self.boxes_by_frame[frame_idx]:
                    box_data = self.boxes_by_frame[frame_idx][obj_id]
                    if isinstance(box_data, dict) and box_data.get("is_fallback", False):
                        is_fallback = True
                
                # Apply mask overlay
                if isinstance(mask, np.ndarray):
                    # Convert to binary mask if needed
                    if mask.dtype == np.float32 or mask.dtype == np.float64:
                        mask = mask > 0
                    
                    # Ensure mask is 2D
                    if len(mask.shape) > 2:
                        mask = np.squeeze(mask)
                    
                    # Check if mask is valid
                    if mask.size == 0 or mask.ndim != 2:
                        print(f"Warning: Invalid mask for object {obj_id} on frame {frame_idx}, shape: {mask.shape}")
                        continue
                    
                    # Convert to bool and ensure shape is compatible
                    mask_bool = mask.astype(bool)
                    
                    try:
                        # Use different visualization styles based on quality
                        # Use normal overlay for non-empty masks
                        colored_mask = np.zeros_like(vis_frame)
                        colored_mask[mask_bool] = color_rgb  # Use RGB without alpha
                        
                        # Blend the mask with the original frame - higher alpha for high quality
                        alpha = 0.6 if is_high_quality else 0.3
                        vis_frame = cv2.addWeighted(colored_mask, alpha, vis_frame, 1.0, 0)
                    except Exception as e:
                        print(f"Error applying mask for object {obj_id} on frame {frame_idx}: {e}")
                        continue
                
                # Add title with object info and quality metrics
                title_text = f"Object #{obj_id}: {obj_class} - {pixel_count} pixels"
                
                # Add quality indicators to the title
                if not is_high_quality:
                    title_text += f" [LOW QUALITY ≤ {self.mask_quality_threshold}]"
                if is_fallback:
                    title_text += " [FALLBACK BOX]"
                
                cv2.putText(vis_frame, title_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_rgb, 2)
                
                # If using fallback box, draw it
                if is_fallback and hasattr(self, 'boxes_by_frame') and frame_idx in self.boxes_by_frame and obj_id in self.boxes_by_frame[frame_idx]:
                    box_data = self.boxes_by_frame[frame_idx][obj_id]
                    if isinstance(box_data, dict) and "box" in box_data:
                        box = box_data["box"]
                        x1, y1, x2, y2 = map(int, box)
                        # Draw with a different color for fallback boxes
                        fallback_color = (50, 50, 255)  # Blue for fallback
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), fallback_color, 3)
                        cv2.putText(vis_frame, "FALLBACK BOX", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fallback_color, 2)
                
                # Save visualization
                obj_dir = object_masks_dir / f"object_{obj_id}_{obj_class}"
                output_path = obj_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(output_path), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                object_quality_stats[obj_id]["frames_saved"] += 1
        
        # Log quality statistics for each object
        for obj_id, stats in object_quality_stats.items():
            obj_class = self.tracked_objects[obj_id]['class']
            print(f"  Object #{obj_id} ({obj_class}) mask quality statistics:")
            print(f"    Frames processed: {stats['frames_processed']}")
            print(f"    Frames saved: {stats['frames_saved']}")
            print(f"    High quality masks (>{self.mask_quality_threshold} pixels): {stats['high_quality_count']}")
            print(f"    Low quality masks (1-{self.mask_quality_threshold} pixels): {stats['low_quality_count']}")
            print(f"    Empty masks (0 pixels): {stats['empty_mask_count']} (not saved)")
            
            # Store quality metrics in the tracked object for later use
            self.tracked_objects[obj_id]['mask_quality_stats'] = stats
        
        print(f"All per-object mask visualizations saved to {object_masks_dir}")
    
    def _count_mask_pixels(self, mask):
        """Count the number of true pixels in a mask"""
        # Handle tensor masks
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
            
        # Convert to boolean if needed
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = mask > 0
            
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = np.squeeze(mask)
            
        # Count true pixels
        if mask.size > 0 and mask.ndim == 2:
            return np.sum(mask).astype(int)
        else:
            return 0

    def save_object_frame_mapping(self):
        """Save a mapping of object IDs to the frames they appear in with quality metrics."""
        print("Creating object-to-frame mapping with quality metrics...")
        
        mapping = {}
        
        # For each object
        for obj_id, obj_data in self.tracked_objects.items():
            # Initialize frame lists based on quality
            high_quality_frames = []
            low_quality_frames = []
            empty_mask_frames = []
            
            # Process all frames that have masks for this object
            for frame_idx in sorted(self.propagation_results.keys()):
                # Check if this frame has a mask for this object
                if obj_id in self.propagation_results[frame_idx]:
                    # Get the mask
                    mask = self.propagation_results[frame_idx][obj_id]
                    
                    # Skip if mask is None
                    if mask is None:
                        continue
                    
                    # Count pixels to determine quality
                    pixel_count = self._count_mask_pixels(mask)
                    
                    # Categorize by quality
                    if pixel_count > self.mask_quality_threshold:
                        high_quality_frames.append(frame_idx)
                    elif pixel_count > 0:
                        low_quality_frames.append(frame_idx)
                    else:
                        empty_mask_frames.append(frame_idx)
                
            # No need to get first_detected and last_seen metadata anymore
            
            # Store in the mapping - streamlined version with only essential data
            mapping[str(obj_id)] = {
                "class": obj_data["class"],
                "high_quality_frames": high_quality_frames,
                "low_quality_frames": low_quality_frames,
                "empty_mask_frames": empty_mask_frames,
                "mask_quality_threshold": self.mask_quality_threshold
            }
        
        # Write the mapping to a JSON file
        mapping_path = self.output_dir / "object_frame_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        # Log statistics - calculate on the fly from lists
        total_objects = len(mapping)
        total_high_quality = sum(len(data["high_quality_frames"]) for data in mapping.values())
        total_low_quality = sum(len(data["low_quality_frames"]) for data in mapping.values())
        total_empty = sum(len(data["empty_mask_frames"]) for data in mapping.values())
        total_saved = total_high_quality + total_low_quality
        
        print(f"Object tracking quality statistics:")
        print(f"  Total objects tracked: {total_objects}")
        print(f"  Total high quality frames (>{self.mask_quality_threshold} pixels): {total_high_quality}")
        print(f"  Total low quality frames (1-{self.mask_quality_threshold} pixels): {total_low_quality}")
        print(f"  Total empty mask frames (0 pixels): {total_empty} (not saved)")
        print(f"  Total saved frames (non-empty masks): {total_saved}")
        
        # Generate a separate summary JSON with overall statistics
        summary = {
            "total_objects": total_objects,
            "total_frames_processed": len(self.propagation_results),
            "quality_metrics": {
                "high_quality_frames": total_high_quality,
                "low_quality_frames": total_low_quality,
                "empty_mask_frames": total_empty,
                "mask_quality_threshold": self.mask_quality_threshold
            },
            "objects": {
                obj_id: {
                    "class": data["class"],
                    "quality_counts": {
                        "high_quality": len(data["high_quality_frames"]),
                        "low_quality": len(data["low_quality_frames"]),
                        "empty": len(data["empty_mask_frames"])
                    }
                } for obj_id, data in mapping.items()
            }
        }
        
        # Save the summary
        summary_path = self.output_dir / "tracking_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved tracking summary to {summary_path}")
        
        # Verify consistency with saved mask images
        try:
            print("Verifying consistency with saved mask images...")
            consistent = True
            
            for obj_id, obj_data in mapping.items():
                obj_class = obj_data["class"]
                mask_dir = self.output_dir / "object_masks" / f"object_{obj_id}_{obj_class}"
            
                if not mask_dir.exists():
                    print(f"  Warning: No mask directory for object {obj_id}")
                    consistent = False
                    continue
            
                # Check that all non-empty frames have corresponding images
                saved_frames = obj_data["high_quality_frames"] + obj_data["low_quality_frames"]
                for frame_idx in saved_frames:
                    mask_path = mask_dir / f"frame_{frame_idx:04d}.jpg"
                    if not mask_path.exists():
                        print(f"  Warning: Missing mask image for object {obj_id} frame {frame_idx}")
                        consistent = False
            
                # Check that empty mask frames do NOT have images
                for frame_idx in obj_data["empty_mask_frames"]:
                    mask_path = mask_dir / f"frame_{frame_idx:04d}.jpg"
                    if mask_path.exists():
                        print(f"  Warning: Found mask image for empty mask object {obj_id} frame {frame_idx}")
                        consistent = False
            
            if consistent:
                print("Verification successful: All objects' masks are consistent with the mapping")
            else:
                print("Warning: Inconsistencies found between object mapping and mask images")
        except Exception as e:
            print(f"Error during verification: {e}")
        
        print(f"Saved object-to-frame mapping to {mapping_path}")
        return mapping_path

    def _detect_and_track_all_objects(self, frames_dir, frame_files, text_queries, results):
        """Detect and track all objects without running SAM2"""
        print("Phase 1: Detecting and tracking objects...")
        
        # Reset tracker and tracked objects
        self.tracker = ObjectTracker()
        all_objects = {}
        
        # Process all frames - frame_files should already be naturally sorted
        for i, frame_path in enumerate(frame_files):
            # Extract the actual frame index from the filename
            frame_idx = self._extract_frame_idx_from_path(frame_path)
            
            # Load frame
            frame = Image.open(frame_path).convert("RGB")
            frame_np = np.array(frame)
            
            # Prepare frame data based on detector type
            frame_data = frame
            if self.detector_type == "cd_fsod":
                frame_data = {
                    "image": frame,
                    "frame_path": str(frame_path),
                    "frame_idx": frame_idx  # Use extracted frame index
                }
            
            # Detect objects
            detections = self.detector.detect(
                image=frame_data,
                text_queries=text_queries,
                threshold=self.confidence_threshold
            )
            
            # Convert to tracker format
            tracker_detections = []
            for i in range(len(detections["boxes"])):
                if detections["scores"][i] >= self.confidence_threshold:
                    tracker_detections.append({
                        "box": detections["boxes"][i].tolist() if isinstance(detections["boxes"][i], torch.Tensor) else detections["boxes"][i],
                        "score": detections["scores"][i].item() if isinstance(detections["scores"][i], torch.Tensor) else detections["scores"][i],
                        "text": detections["labels"][i]
                    })

            # Update tracker
            current_boxes = self.tracker.update_tracks(
                frame=frame_np,
                frame_idx=frame_idx,  # Use extracted frame index
                detections=tracker_detections,
                embedding_extractor=self.embedding_extractor,
                output_dir=None
            )

            # Store frame detections
            if frame_idx not in results["frame_results"]:
                results["frame_results"][frame_idx] = {"detections": [], "tracked_objects": []}
                
            results["frame_results"][frame_idx]["detections"] = [
                {"box": box.tolist() if isinstance(box, torch.Tensor) else box, 
                 "label": label, 
                 "confidence": conf}
                for box, label, conf in zip(detections["boxes"], detections["labels"], detections["scores"])
                if conf >= self.confidence_threshold
            ]
            
            # Update tracked objects
            for obj_id, box in current_boxes.items():
                tracker_obj = self.tracker.tracked_objects[obj_id]
                
                # Get embedding
                x1, y1, x2, y2 = [int(c) for c in box]
                crop = frame_np[y1:y2, x1:x2]
                embedding = self.embedding_extractor.extract(crop)
                
                # Check if object exists in our all_objects dict
                if obj_id not in all_objects:
                    # New object
                    all_objects[obj_id] = {
                        "id": obj_id,
                        "class": tracker_obj["class"],
                        "first_detected": frame_idx,  # Use extracted frame index
                        "boxes": [box],
                        "embeddings": [embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding],
                        "masks": [],  # Will be filled during SAM2 phase
                        "last_seen": frame_idx,  # Use extracted frame index
                        "confidence": [0.0]  # Placeholder
                    }
                    
                    print(f"Created new object {obj_id} ({tracker_obj['class']}) at frame {frame_idx}")
                else:
                    # Update existing object
                    all_objects[obj_id]["boxes"].append(box)
                    all_objects[obj_id]["embeddings"].append(
                        embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
                    )
                    all_objects[obj_id]["last_seen"] = frame_idx  # Use extracted frame index
                    all_objects[obj_id]["confidence"].append(0.0)  # Placeholder
                
                # Add to frame results
                results["frame_results"][frame_idx]["tracked_objects"].append(obj_id)
        
        print(f"Found {len(all_objects)} unique objects to process with SAM2")
        return all_objects

    def process_video_separate_objects(self, frames_dir: str, text_queries: List[str]):
        """Process video with separate SAM2 initialization for each object"""
        # Get all frames sorted with natural sorting
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")], key=natural_sort_key)
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Processing {len(frame_files)} frames with queries: {text_queries}")
        print(f"Using detector: {self.detector_type}")
        
        # Create a mapping of loop indices to actual frame indices
        frame_indices = []
        for i, frame_path in enumerate(frame_files):
            frame_idx = self._extract_frame_idx_from_path(frame_path)
            frame_indices.append(frame_idx)
        
        # Results storage
        results = {
            "object_tracks": {},
            "frame_results": {idx: {"detections": [], "tracked_objects": []} for idx in frame_indices},
            "metadata": {
                "queries": text_queries,
                "frame_count": len(frame_files),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detector_type": self.detector_type  # Include detector type in metadata
            }
        }
        
        # First detect and track all objects without SAM2
        all_objects = self._detect_and_track_all_objects(frames_dir, frame_files, text_queries, results)
        
        # Now process each object separately with SAM2
        self.tracked_objects = {}  # Clear existing tracked objects
        self.propagation_results = {}  # Will store all propagation results
        
        for obj_id, obj_data in all_objects.items():
            print(f"\n==== Processing object {obj_id} ({obj_data['class']}) separately ====")
            
            # Reset SAM2 completely for this object
            print(f"Resetting SAM2 state for object {obj_id}...")
            self.sam_wrapper.set_video(frames_dir=frames_dir)
            
            # Get the first frame this object appears in
            first_frame_idx = obj_data["first_detected"]
            first_box = obj_data["boxes"][0]
            
            # Add box to SAM2
            print(f"Adding box for object {obj_id} at frame {first_frame_idx}")
            mask_logits = self.sam_wrapper.add_box(frame_idx=first_frame_idx, obj_id=obj_id, box=first_box)
            
            if mask_logits is None:
                print(f"Failed to generate mask for object {obj_id}")
                continue
                
            # Update mask in the object data
            obj_data["masks"] = [mask_logits.cpu().numpy() if isinstance(mask_logits, torch.Tensor) else mask_logits]
            
            # Run propagation just for this object
            try:
                print(f"Running propagation for object {obj_id}...")
                result = self.sam_wrapper.propagate_masks(objects_to_track=[obj_id])
                print(f"DEBUG: propagate_masks(objects_to_track={obj_id}) returned type: {type(result)}")
                if isinstance(result, tuple):
                    print(f"DEBUG: propagate_masks(objects_to_track={obj_id}) tuple length: {len(result)}")
                    segments, boxes_by_frame = result
                else:
                    print(f"DEBUG: propagate_masks(objects_to_track={obj_id}) value: {result}")
                    segments = result
                    boxes_by_frame = {}  # Initialize empty dict if no boxes returned
                
                # Store all propagation results
                for frame_idx, masks in segments.items():
                    if frame_idx not in self.propagation_results:
                        self.propagation_results[frame_idx] = {}
                    
                    # Only copy if the propagation returned masks for this object
                    if obj_id in masks:
                        self.propagation_results[frame_idx][obj_id] = masks[obj_id]
                        
                        # Analyze mask quality
                        mask = masks[obj_id]
                        pixel_count = self._count_mask_pixels(mask)
                        quality_status = "HIGH QUALITY" if pixel_count >= self.mask_quality_threshold else "LOW QUALITY" if pixel_count > 0 else "EMPTY"
                    
                # If we also have boxes_by_frame, store those too
                if isinstance(boxes_by_frame, dict) and boxes_by_frame:
                    for frame_idx, boxes in boxes_by_frame.items():
                        if frame_idx not in self.boxes_by_frame:
                            self.boxes_by_frame[frame_idx] = {}
                        
                        # Only copy if boxes exist for this object
                        if obj_id in boxes:
                            self.boxes_by_frame[frame_idx][obj_id] = boxes[obj_id]
                            
                            # Check fallback box silently
                            if isinstance(boxes[obj_id], dict) and boxes[obj_id].get("is_fallback", False):
                                pass  # No logging
                
                # NEW CODE: Update tracked_objects with all boxes from propagation
                all_boxes = []
                for frame_idx in sorted(boxes_by_frame.keys()):
                    if obj_id in boxes_by_frame[frame_idx]:
                        box_data = boxes_by_frame[frame_idx][obj_id]
                        if isinstance(box_data, dict):
                            box = box_data["box"]
                            all_boxes.append(box)
                            print(f"Adding box for frame {frame_idx} to object {obj_id}: {box}")
                        else:
                            all_boxes.append(box_data)
                            print(f"Adding raw box for frame {frame_idx} to object {obj_id}: {box_data}")
                
                # Add logging to debug the tracked_objects dictionary
                print(f"DEBUG: tracked_objects keys before update: {list(self.tracked_objects.keys())}")
                print(f"DEBUG: Is object {obj_id} in tracked_objects? {obj_id in self.tracked_objects}")
                print(f"DEBUG: Number of boxes collected for object {obj_id}: {len(all_boxes)}")
                
                # Only update if we have boxes
                if not all_boxes:
                    print(f"WARNING: No boxes collected for object {obj_id}, skipping tracked_objects update")
                    continue
                
                # Update tracked_objects with the new boxes, with proper error handling
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id]["boxes"] = all_boxes
                    # Also collect and store masks
                    all_masks = []
                    for frame_idx in sorted(self.propagation_results.keys()):
                        if obj_id in self.propagation_results[frame_idx]:
                            all_masks.append(self.propagation_results[frame_idx][obj_id])
                    
                    # Store masks if we have them
                    if all_masks:
                        self.tracked_objects[obj_id]["masks"] = all_masks
                        print(f"Stored {len(all_masks)} masks for object {obj_id}")
                    
                    print(f"Successfully updated boxes for object {obj_id}")
                else:
                    # Create the object entry if it doesn't exist
                    print(f"Object {obj_id} not found in tracked_objects. Creating entry.")
                    self.tracked_objects[obj_id] = {
                        "id": obj_id,
                        "class": obj_data["class"],  # Use the class from the original detected object
                        "first_detected": obj_data["first_detected"],  # Use the original detection frame
                        "last_seen": frame_idx,
                        "confidence": [obj_data.get("confidence", [1.0])[0] if isinstance(obj_data.get("confidence"), list) else obj_data.get("confidence", 1.0)],
                        "boxes": all_boxes,
                        "masks": obj_data.get("masks", [])  # Add masks with empty list as default
                    }
                    print(f"Created new entry for object {obj_id}")
            
                print(f"Successfully propagated masks for object {obj_id}, available in {len(segments)} frames")
            except Exception as e:
                print(f"Error during propagation for object {obj_id}: {e}")
                import traceback
                print(f"Propagation error traceback: {traceback.format_exc()}")

            # Store updated object data only if it doesn't exist yet
            print(f"DEBUG: After propagation - Is object {obj_id} in tracked_objects? {obj_id in self.tracked_objects}")
            if obj_id not in self.tracked_objects:
                print(f"Adding object {obj_id} to tracked_objects after propagation")
                self.tracked_objects[obj_id] = obj_data
            else:
                print(f"Object {obj_id} already exists in tracked_objects, keeping existing entry")
                
            # Always update the results dict
            results["object_tracks"][obj_id] = obj_data
        
        # Generate visualizations for all frames
        for i, frame_path in enumerate(frame_files):
            # Extract the actual frame index
            frame_idx = frame_indices[i]
            
            # Get visible objects for this frame
            visible_objects = {
                obj_id: data for obj_id, data in self.tracked_objects.items()
                if data["first_detected"] <= frame_idx <= data["last_seen"]
            }
            
            # Load the frame
            frame = np.array(Image.open(frame_path).convert("RGB"))
            
            # Visualize without saving
            self._visualize_frame(frame=frame, frame_idx=frame_idx, objects=visible_objects)
        
        # Save per-object visualizations
        self.save_per_object_visualizations(frames_dir)
        
        # Save first detection frames
        self.save_first_detections(frames_dir)
        
        # Save mapping of objects to frames they appear in
        self.save_object_frame_mapping()
        
        # Prepare a streamlined version of tracking results without redundant data
        streamlined_results = {
            "metadata": {
                "queries": text_queries,
                "frame_count": len(frame_files),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detector_type": self.detector_type
            },
            "objects": {}
        }
        
        # Only include essential tracking data without masks
        for obj_id, obj_data in self.tracked_objects.items():
            # Ensure masks are always a list
            masks = obj_data.get("masks", [])
            if not isinstance(masks, list):
                masks = [masks]
                
            streamlined_results["objects"][str(obj_id)] = {
                "id": obj_id,
                "class": obj_data["class"],
                "boxes": obj_data["boxes"],  # Keep only the ID, class, and boxes
                # Make sure we keep masks for visualization
                "masks": masks  # Always store masks as a list
                # Removed: first_detected, last_seen, confidence
            }
        
        # Save streamlined tracking results
        with open(self.output_dir / "tracking_results.json", "w") as f:
            json_results = self._prepare_for_json(streamlined_results)
            json.dump(json_results, f, indent=2)
        
        print(f"All results saved to: {self.output_dir}")
        return streamlined_results

    def _extract_frame_idx_from_path(self, frame_path):
        """
        Extract frame index from filename (e.g., '10.jpg' -> 10)
        """
        # Extract frame index from filename (assuming filenames like '0.jpg', '1.jpg', '10.jpg', etc.)
        try:
            # Get just the filename without extension
            filename = Path(frame_path).stem
            # Try to convert to integer directly (handles filenames like '0', '1', '10', etc.)
            frame_idx = int(filename)
            return frame_idx
        except ValueError:
            # If we can't extract a number directly, use a fallback index
            print(f"Warning: Could not extract frame index from {frame_path}, using fallback index.")
            return 0

def main():
    parser = argparse.ArgumentParser(description="Object Tracking Pipeline with OWLv2 and SAM2")
    parser.add_argument("--frames-dir", required=True, help="Directory containing video frames")
    parser.add_argument("--text-queries", required=True, nargs="+", help="Text queries for object detection")
    parser.add_argument("--output-dir", default="./tracking_results", help="Output directory for results")
    parser.add_argument("--owlv2-checkpoint", required=False, help="Path to OWLv2 checkpoint")
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", required=True, help="Path to SAM2 config file")
    parser.add_argument("--confidence", type=float, default=0.1, help="Confidence threshold for detections")
    parser.add_argument("--separate-objects", action="store_true", help="Process each object separately to avoid dtype issues")
    
    # Add CD-FSOD detector options
    parser.add_argument("--detector", choices=["owlv2", "cd_fsod"], default="owlv2", help="Detector type to use")
    parser.add_argument("--cd-fsod-path", help="Path to CD-FSOD JSON detections directory (required if using cd_fsod detector)")
    parser.add_argument("--min-gap-frames", type=int, default=10, help="Minimum gap frames for CD-FSOD reappearances")
    
    args = parser.parse_args()
    
    # Check for required arguments based on detector type
    if args.detector == "cd_fsod" and not args.cd_fsod_path:
        parser.error("--cd-fsod-path is required when using cd_fsod detector")
    
    # Initialize pipeline
    pipeline = ObjectTrackingPipeline(
        owlv2_checkpoint=args.owlv2_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        detector_type=args.detector,
        cd_fsod_path=args.cd_fsod_path,
        min_gap_frames=args.min_gap_frames
    )
    
    # Process video using the appropriate method
    if args.separate_objects:
        pipeline.process_video_separate_objects(
            frames_dir=args.frames_dir,
            text_queries=args.text_queries
        )
    else:
        pipeline.process_video(
            frames_dir=args.frames_dir,
            text_queries=args.text_queries
        )

if __name__ == "__main__":
    main()