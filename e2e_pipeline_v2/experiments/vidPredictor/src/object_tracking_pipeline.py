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

# Import our components
from owlv2_detector import OWLv2Detector
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
    ):
        # Set device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        print(f"Using device: {device}")
        
        # Initialize components
        self.detector = OWLv2Detector(device=device)
        self.sam_wrapper = SAM2VideoWrapper(sam2_checkpoint, sam2_config, device=device)
        self.tracker = ObjectTracker()
        self.embedding_extractor = EmbeddingExtractor(device=device)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Config
        self.confidence_threshold = confidence_threshold
        
        # Tracking state
        self.tracked_objects = {}  # id -> object data
        self.next_id = 1
        self.propagation_results = {}  # Store SAM2 propagation results
        
    def process_video(self, frames_dir: str, text_queries: List[str]):
        # Get all frames sorted
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")])
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Processing {len(frame_files)} frames with queries: {text_queries}")
        
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
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Process first frame: detect → segment → initialize tracking
        first_frame_path = frame_files[0]
        first_frame = Image.open(first_frame_path).convert("RGB")
        first_frame_np = np.array(first_frame)
        
        # Detect objects in first frame
        detections = self.detector.detect(
            image=first_frame,
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
            print(f"Adding box for object {object_id} at frame 0: {box_coords}")
            mask_logits = self.sam_wrapper.add_box(frame_idx=0, obj_id=object_id, box=box_coords)
            
            # Skip if mask generation failed
            if mask_logits is None:
                print(f"Failed to generate mask for object {object_id}")
                continue
            
            # Store this object with the ID
            self.tracked_objects[object_id] = {
                "id": object_id,
                "class": label,
                "first_detected": 0,  # frame 0
                "boxes": [box_coords],
                "embeddings": [embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding],
                "masks": [mask_logits.cpu().numpy() if isinstance(mask_logits, torch.Tensor) else mask_logits],
                "last_seen": 0,  # frame 0
                "confidence": [conf]
            }
            
            # Store in results
            results["object_tracks"][object_id] = self.tracked_objects[object_id]
        
        # Run mask propagation for all tracked objects
        if self.tracked_objects:
            print("Running mask propagation for all tracked objects...")
            object_ids = list(self.tracked_objects.keys())
            segments = self.sam_wrapper.propagate_masks(objects_to_track=object_ids)
            self.propagation_results = segments
            print(f"Propagated masks for {len(object_ids)} objects across {len(segments)} frames")
        else:
            print("No objects to propagate")
        
        # Visualize first frame results
        first_frame_vis = self._visualize_frame(
            frame=first_frame_np,
            frame_idx=0,
            objects=self.tracked_objects
        )
        
        # Save first frame results
        results["frame_results"][0] = {
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
        for frame_idx in range(1, len(frame_files)):
            print(f"Processing frame {frame_idx}/{len(frame_files)}")
            frame_path = frame_files[frame_idx]
            frame = Image.open(frame_path).convert("RGB")
            frame_np = np.array(frame)
            
            # Get new detections
            detections = self.detector.detect(
                image=frame,
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

            print(f"-----Initializing new objects-----")
            print(f"detections: {tracker_detections}")
            print(f"-----Initializing {tracker_detections}-----")

            # Get boxes for current frame from tracker
            current_boxes = self.tracker.update_tracks(
                frame=frame_np,
                frame_idx=frame_idx,
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
                    # First attempt: Try to propagate only the new objects
                    segments = self.sam_wrapper.propagate_masks(objects_to_track=new_objects_in_this_frame)
                    
                    # Update the propagation results with the new segments
                    for f_idx, frame_segments in segments.items():
                        if f_idx not in self.propagation_results:
                            self.propagation_results[f_idx] = {}
                        
                        # Add new object segments to existing propagation results
                        for obj_id, mask in frame_segments.items():
                            self.propagation_results[f_idx][obj_id] = mask
                    
                    print(f"Propagated masks for {len(new_objects_in_this_frame)} new objects")
                except Exception as e:
                    print(f"Error during mask propagation for new objects: {e}")
                    
                    # Second attempt: Try resetting SAM2 state and re-adding all objects
                    print("Attempting to reset SAM2 state and re-add objects...")
                    try:
                        # Reset the SAM2 state completely
                        self.sam_wrapper.reset_state()
                        
                        # Re-add all objects in frame order
                        objects_by_frame = {}
                        for obj_id, obj_data in self.tracked_objects.items():
                            frame_first_detected = obj_data["first_detected"]
                            if frame_first_detected not in objects_by_frame:
                                objects_by_frame[frame_first_detected] = []
                            objects_by_frame[frame_first_detected].append((obj_id, obj_data["boxes"][0]))
                        
                        # Process each frame in order
                        for frame_to_process in sorted(objects_by_frame.keys()):
                            for obj_id, box in objects_by_frame[frame_to_process]:
                                print(f"Re-adding object {obj_id} at frame {frame_to_process}")
                                mask = self.sam_wrapper.add_box(frame_idx=frame_to_process, obj_id=obj_id, box=box)
                                
                                # Update the object's first mask if needed
                                if mask is not None and len(self.tracked_objects[obj_id]["masks"]) > 0:
                                    self.tracked_objects[obj_id]["masks"][0] = mask
                        
                        # Run propagation for ALL objects
                        print("Running propagation for all objects after reset")
                        all_object_ids = list(self.tracked_objects.keys())
                        segments = self.sam_wrapper.propagate_masks(objects_to_track=all_object_ids)
                        
                        # Replace all propagation results
                        self.propagation_results = segments
                        print(f"Successfully propagated all objects after reset")
                    except Exception as reset_error:
                        print(f"Error after SAM2 reset: {reset_error}")
                        print("Using fallback approach: applying initial masks to all frames for new objects")
                        
                        # Third attempt (fallback): Use initial masks for all frames
                        for obj_id in new_objects_in_this_frame:
                            if obj_id in self.tracked_objects and len(self.tracked_objects[obj_id]["masks"]) > 0:
                                # Get the initial mask for this object
                                initial_mask = self.tracked_objects[obj_id]["masks"][0]
                                
                                # Apply to all subsequent frames (from current frame to end)
                                for future_frame_idx in range(frame_idx, len(frame_files)):
                                    if future_frame_idx not in self.propagation_results:
                                        self.propagation_results[future_frame_idx] = {}
                                    self.propagation_results[future_frame_idx][obj_id] = initial_mask
                                
                                print(f"Applied initial mask for object {obj_id} to all future frames")
            
            # Visualize this frame
            frame_vis = self._visualize_frame(
                frame=frame_np,
                frame_idx=frame_idx,
                objects={obj_id: data for obj_id, data in self.tracked_objects.items()
                         if data["last_seen"] == frame_idx}
            )
            
            # Save frame results
            results["frame_results"][frame_idx] = {
                "detections": [
                    {"box": box.tolist() if isinstance(box, torch.Tensor) else box, 
                     "label": label, 
                     "confidence": conf}
                    for box, label, conf in zip(detections["boxes"], detections["labels"], detections["scores"])
                    if conf >= self.confidence_threshold
                ],
                "tracked_objects": [
                    obj_id for obj_id, data in self.tracked_objects.items()
                    if data["last_seen"] == frame_idx
                ]
            }
        
        # Save per-object visualizations (each object in its own folder)
        self.save_per_object_visualizations(frames_dir)
        
        # Save first detection frames (each object with its initial detection box)
        self.save_first_detections(frames_dir)
        
        # Save final results
        with open(self.output_dir / "tracking_results.json", "w") as f:
            # Convert numpy arrays and tensors to lists
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2)
        
        print(f"All results saved to: {self.output_dir}")
        return results
    
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
            mask = obj_data["masks"][-1]
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
                    print(f"Mask shape: {mask.shape}, Mask dtype: {mask.dtype}")
                    print(f"Vis frame shape: {vis_frame.shape}")
                    print(f"Colored mask shape: {colored_mask.shape}")
                    continue
            
            # Draw label
            label = f"{obj_data['class']} #{obj_id}"
            conf = obj_data["confidence"][-1]
            text = f"{label} ({conf:.2f})"
            cv2.putText(vis_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
        
        # Save visualization
        output_path = self.output_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
        
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
        """Save the first frame each object appears in with its detection box
        
        Creates a folder for each object containing only the first frame it was detected in,
        with the detection box drawn (without segmentation mask).
        """
        print("Saving first detection frames for each object...")
        
        # Create base directory for first detections
        first_detections_dir = self.output_dir / "first_detections"
        first_detections_dir.mkdir(exist_ok=True, parents=True)
        
        # Get frame paths
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")])
        
        # For each object
        for obj_id, obj_data in self.tracked_objects.items():
            # Create directory for this object
            obj_dir = first_detections_dir / f"object_{obj_id}_{obj_data['class']}"
            obj_dir.mkdir(exist_ok=True)
            
            # Get first frame index and box
            first_frame_idx = obj_data["first_detected"]
            first_box = obj_data["boxes"][0]
            
            # Ensure the frame exists
            if first_frame_idx < len(frame_files):
                # Load the first frame
                frame_path = frame_files[first_frame_idx]
                frame = np.array(Image.open(frame_path).convert("RGB"))
                
                # Create visualization with just the detection box
                vis_frame = frame.copy()
                
                # Get a color for this object (consistent with pipeline visualization)
                cmap = plt.cm.get_cmap("tab10")
                color = cmap(obj_id % 10)[:3]
                color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                
                # Draw bounding box
                x1, y1, x2, y2 = [int(coord) for coord in first_box]
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color_rgb, 3)  # Thicker line for visibility
                
                # Add title with object info
                title_text = f"Object #{obj_id}: {obj_data['class']} (First detected at frame {first_frame_idx})"
                cv2.putText(vis_frame, title_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_rgb, 2)
                
                # Save visualization
                output_path = obj_dir / f"first_detection_frame_{first_frame_idx:04d}.jpg"
                cv2.imwrite(str(output_path), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
                
                print(f"  Saved first detection frame for object #{obj_id} ({obj_data['class']})")
            else:
                print(f"  Error: First frame index {first_frame_idx} out of range for object #{obj_id}")
        
        print(f"All first detection frames saved to {first_detections_dir}")

    def save_per_object_visualizations(self, frames_dir):
        """Save per-object visualizations with masks overlaid on original frames"""
        print("Saving per-object segmentation visualizations...")
        
        # Create base directory for object masks
        object_masks_dir = self.output_dir / "object_masks"
        object_masks_dir.mkdir(exist_ok=True, parents=True)
        
        # Get frame paths
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")])
        
        # For each object
        for obj_id, obj_data in self.tracked_objects.items():
            # Create directory for this object
            obj_dir = object_masks_dir / f"object_{obj_id}_{obj_data['class']}"
            obj_dir.mkdir(exist_ok=True)
            
            # Process each frame for this object
            for frame_idx in range(len(frame_files)):
                # Check if this object has a mask for this frame
                if frame_idx in self.propagation_results and obj_id in self.propagation_results[frame_idx]:
                    # Load the original frame
                    frame_path = frame_files[frame_idx]
                    frame = np.array(Image.open(frame_path).convert("RGB"))
                    
                    # Get the mask for this object in this frame
                    mask = self.propagation_results[frame_idx][obj_id]
                    
                    # Create visualization with just this object's mask
                    vis_frame = frame.copy()
                    
                    # Get a color for this object (consistent with pipeline visualization)
                    cmap = plt.cm.get_cmap("tab10")
                    color = cmap(obj_id % 10)[:3]
                    color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    
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
                            print(f"Error applying mask for object {obj_id} on frame {frame_idx}: {e}")
                            continue
                    
                    # Add title with object info
                    title_text = f"Object #{obj_id}: {obj_data['class']}"
                    cv2.putText(vis_frame, title_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_rgb, 2)
                    
                    # Save visualization
                    output_path = obj_dir / f"frame_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(output_path), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
            
            print(f"  Saved visualizations for object #{obj_id} ({obj_data['class']})")
        
        print(f"All per-object mask visualizations saved to {object_masks_dir}")

    def _detect_and_track_all_objects(self, frames_dir, frame_files, text_queries, results):
        """Detect and track all objects without running SAM2"""
        print("Phase 1: Detecting and tracking objects...")
        
        # Reset tracker and tracked objects
        self.tracker = ObjectTracker()
        all_objects = {}
        
        # Process all frames
        for frame_idx, frame_path in enumerate(frame_files):
            # Load frame
            frame = Image.open(frame_path).convert("RGB")
            frame_np = np.array(frame)
            
            # Detect objects
            detections = self.detector.detect(
                image=frame,
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
                frame_idx=frame_idx,
                detections=tracker_detections,
                embedding_extractor=self.embedding_extractor,
                output_dir=None
            )
            
            # Store frame detections
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
                        "first_detected": frame_idx,
                        "boxes": [box],
                        "embeddings": [embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding],
                        "masks": [],  # Will be filled during SAM2 phase
                        "last_seen": frame_idx,
                        "confidence": [0.0]  # Placeholder
                    }
                    
                    print(f"Created new object {obj_id} ({tracker_obj['class']}) at frame {frame_idx}")
                else:
                    # Update existing object
                    all_objects[obj_id]["boxes"].append(box)
                    all_objects[obj_id]["embeddings"].append(
                        embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
                    )
                    all_objects[obj_id]["last_seen"] = frame_idx
                    all_objects[obj_id]["confidence"].append(0.0)  # Placeholder
                
                # Add to frame results
                results["frame_results"][frame_idx]["tracked_objects"].append(obj_id)
        
        print(f"Found {len(all_objects)} unique objects to process with SAM2")
        return all_objects

    def process_video_separate_objects(self, frames_dir: str, text_queries: List[str]):
        """Process video with separate SAM2 initialization for each object"""
        # Get all frames sorted
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")])
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Processing {len(frame_files)} frames with queries: {text_queries}")
        
        # Results storage
        results = {
            "object_tracks": {},
            "frame_results": {i: {"detections": [], "tracked_objects": []} for i in range(len(frame_files))},
            "metadata": {
                "queries": text_queries,
                "frame_count": len(frame_files),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
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
                segments = self.sam_wrapper.propagate_masks(objects_to_track=[obj_id])
                
                # Store the propagation results
                for f_idx, frame_segments in segments.items():
                    if f_idx not in self.propagation_results:
                        self.propagation_results[f_idx] = {}
                    
                    if obj_id in frame_segments:
                        self.propagation_results[f_idx][obj_id] = frame_segments[obj_id]
                
                print(f"Successfully propagated masks for object {obj_id}")
            except Exception as e:
                print(f"Error during propagation for object {obj_id}: {e}")
                print("Using fallback approach: copying initial mask to other frames")
                
                # Use the initial mask for all frames where this object is present
                last_frame = obj_data["last_seen"]
                for f_idx in range(first_frame_idx, last_frame + 1):
                    if f_idx not in self.propagation_results:
                        self.propagation_results[f_idx] = {}
                    self.propagation_results[f_idx][obj_id] = obj_data["masks"][0]
            
            # Store updated object data
            self.tracked_objects[obj_id] = obj_data
            results["object_tracks"][obj_id] = obj_data
        
        # Generate visualizations for all frames
        for frame_idx in range(len(frame_files)):
            # Get visible objects for this frame
            visible_objects = {
                obj_id: data for obj_id, data in self.tracked_objects.items()
                if data["first_detected"] <= frame_idx <= data["last_seen"]
            }
            
            # Load the frame
            frame_path = frame_files[frame_idx]
            frame = np.array(Image.open(frame_path).convert("RGB"))
            
            # Visualize
            self._visualize_frame(frame=frame, frame_idx=frame_idx, objects=visible_objects)
        
        # Save per-object visualizations
        self.save_per_object_visualizations(frames_dir)
        
        # Save first detection frames
        self.save_first_detections(frames_dir)
        
        # Save final results
        with open(self.output_dir / "tracking_results.json", "w") as f:
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2)
        
        print(f"All results saved to: {self.output_dir}")
        return results

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
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ObjectTrackingPipeline(
        owlv2_checkpoint=args.owlv2_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence
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