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

def main():
    parser = argparse.ArgumentParser(description="Object Tracking Pipeline with OWLv2 and SAM2")
    parser.add_argument("--frames-dir", required=True, help="Directory containing video frames")
    parser.add_argument("--text-queries", required=True, nargs="+", help="Text queries for object detection")
    parser.add_argument("--output-dir", default="./tracking_results", help="Output directory for results")
    parser.add_argument("--owlv2-checkpoint", required=False, help="Path to OWLv2 checkpoint")
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", required=True, help="Path to SAM2 config file")
    parser.add_argument("--confidence", type=float, default=0.1, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ObjectTrackingPipeline(
        owlv2_checkpoint=args.owlv2_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence
    )
    
    # Process video
    pipeline.process_video(
        frames_dir=args.frames_dir,
        text_queries=args.text_queries
    )

if __name__ == "__main__":
    main()

# goat'}]-----
# Updated object 1 (goat) with score 0.74 (IoU: 0.71, Emb: 0.77)
# Updated object 3 (goat) with score 0.63 (IoU: 0.39, Emb: 0.87)
# Updated object 4 (goat) with score 0.53 (IoU: 0.36, Emb: 0.69)
# Updated object 2 (goat) with score 0.42 (IoU: 0.12, Emb: 0.72)
# Created new object 5 (goat)
# Created new object 6 (goat)
# Created new object 7 (goat)
# Created new object 8 (goat)
# Created new object 9 (goat)
# Created new object 10 (goat)
# Created new object 11 (goat)
# Created new pipeline-tracked object 5 at frame 4
# Created new pipeline-tracked object 6 at frame 4
# Created new pipeline-tracked object 7 at frame 4
# Created new pipeline-tracked object 8 at frame 4
# Created new pipeline-tracked object 9 at frame 4
# Created new pipeline-tracked object 10 at frame 4
# Created new pipeline-tracked object 11 at frame 4
# Adding new object 5 at frame 4
# /home/ubuntu/code/drew/sam2/sam2/sam2_video_predictor.py:786: UserWarning: cannot import name '_C' from 'sam2' (/home/ubuntu/code/drew/sam2/sam2/__init__.py)

# Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
#   pred_masks_gpu = fill_holes_in_mask_scores(
# Adding new object 6 at frame 4
# Adding new object 7 at frame 4
# Adding new object 8 at frame 4
# Adding new object 9 at frame 4
# Adding new object 10 at frame 4
# Adding new object 11 at frame 4
# Found 7 new objects in frame 4, running propagation...
# propagate in video:   0%|                                                                                                                                        | 0/13 [00:00<?, ?it/s]
# Error during mask propagation: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
# Available methods on predictor:
#   T_destination
#   add_all_frames_to_correct_as_cond
#   add_module
#   add_new_mask
#   add_new_points
#   add_new_points_or_box
#   add_tpos_enc_to_obj_ptrs
#   apply
#   backbone_stride
#   bfloat16
#   binarize_mask_from_pts_for_mem_enc
#   buffers
#   call_super_init
#   children
#   clear_all_prompts_in_frame
#   clear_non_cond_mem_around_input
#   compile
#   cpu
#   cuda
#   device
#   directly_add_no_mem_embed
#   double
#   dump_patches
#   eval
#   extra_repr
#   fill_hole_area
#   fixed_no_obj_ptr
#   float
#   forward
#   forward_image
#   from_pretrained
#   get_buffer
#   get_extra_state
#   get_parameter
#   get_submodule
#   half
#   hidden_dim
#   image_encoder
#   image_size
#   init_state
#   iou_prediction_use_sigmoid
#   ipu
#   load_state_dict
#   mask_downsample
#   maskmem_tpos_enc
#   max_cond_frames_in_attn
#   max_obj_ptrs_in_encoder
#   mem_dim
#   memory_attention
#   memory_encoder
#   memory_temporal_stride_for_eval
#   modules
#   mtia
#   multimask_max_pt_num
#   multimask_min_pt_num
#   multimask_output_for_tracking
#   multimask_output_in_sam
#   named_buffers
#   named_children
#   named_modules
#   named_parameters
#   no_mem_embed
#   no_mem_pos_enc
#   no_obj_embed_spatial
#   no_obj_ptr
#   non_overlap_masks
#   non_overlap_masks_for_mem_enc
#   num_feature_levels
#   num_maskmem
#   obj_ptr_proj
#   obj_ptr_tpos_proj
#   only_obj_ptrs_in_the_past_for_eval
#   parameters
#   pred_obj_scores
#   pred_obj_scores_mlp
#   proj_tpos_enc_in_obj_ptrs
#   propagate_in_video
#   propagate_in_video_preflight
#   register_backward_hook
#   register_buffer
#   register_forward_hook
#   register_forward_pre_hook
#   register_full_backward_hook
#   register_full_backward_pre_hook
#   register_load_state_dict_post_hook
#   register_load_state_dict_pre_hook
#   register_module
#   register_parameter
#   register_state_dict_post_hook
#   register_state_dict_pre_hook
#   remove_object
#   requires_grad_
#   reset_state
#   sam_image_embedding_size
#   sam_mask_decoder
#   sam_mask_decoder_extra_args
#   sam_prompt_embed_dim
#   sam_prompt_encoder
#   set_extra_state
#   set_submodule
#   share_memory
#   sigmoid_bias_for_mem_enc
#   sigmoid_scale_for_mem_enc
#   soft_no_obj_ptr
#   state_dict
#   to
#   to_empty
#   track_step
#   train
#   training
#   type
#   use_high_res_features_in_sam
#   use_mask_input_as_output_without_sam
#   use_mlp_for_obj_ptr_proj
#   use_multimask_token_for_obj_ptr
#   use_obj_ptrs_in_encoder
#   use_signed_tpos_enc_to_obj_ptrs
#   xpu
#   zero_grad
# Traceback (most recent call last):
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 540, in <module>
#     main()
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 534, in main
#     pipeline.process_video(
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 298, in process_video
#     segments = self.sam_wrapper.propagate_masks(objects_to_track=new_objects_in_this_frame)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/sam2_wrapper.py", line 222, in propagate_masks
#     raise e
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/sam2_wrapper.py", line 194, in propagate_masks
#     for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 36, in generator_context
#     response = gen.send(None)
#                ^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/sam2_video_predictor.py", line 603, in propagate_in_video
#     current_out, pred_masks = self._run_single_frame_inference(
#                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/sam2_video_predictor.py", line 762, in _run_single_frame_inference
#     current_out = self.track_step(
#                   ^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/modeling/sam2_base.py", line 835, in track_step
#     current_out, sam_outputs, _, _ = self._track_step(
#                                      ^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/modeling/sam2_base.py", line 761, in _track_step
#     pix_feat = self._prepare_memory_conditioned_features(
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/modeling/sam2_base.py", line 667, in _prepare_memory_conditioned_features
#     pix_feat_with_mem = self.memory_attention(
#                         ^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/modeling/memory_attention.py", line 155, in forward
#     output = layer(
#              ^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/modeling/memory_attention.py", line 94, in forward
#     tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/modeling/memory_attention.py", line 74, in _forward_ca
#     tgt2 = self.cross_attn_image(
#            ^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/sam2/sam2/modeling/sam/transformer.py", line 281, in forward
#     v = self.v_proj(v)
#         ^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
#     return forward_call(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 125, in forward
#     return F.linear(input, self.weight, self.bias)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float