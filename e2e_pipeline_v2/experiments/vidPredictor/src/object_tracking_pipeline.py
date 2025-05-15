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

        # --- NEW LOGIC: Track all objects across all frames first ---
        # Use the existing _detect_and_track_all_objects to get all objects and their boxes per frame
        all_objects = self._detect_and_track_all_objects(frames_dir, frame_files, text_queries, results)

        # Build a set of all object IDs
        all_obj_ids = list(all_objects.keys())
        # Build a set of all frame indices
        all_frame_indices = [self._extract_frame_idx_from_path(f) for f in frame_files]

        # Build per-frame, per-object annotation structure
        # For each frame, for each object, provide real or dummy annotation
        annotations_by_frame = {}
        for frame_idx in all_frame_indices:
            annotations_by_frame[frame_idx] = []
            for obj_id in all_obj_ids:
                obj_data = all_objects[obj_id]
                # Find if this object has a box for this frame
                frame_box = None
                for i, box_frame_idx in enumerate(range(obj_data["first_detected"], obj_data["last_seen"]+1)):
                    if box_frame_idx == frame_idx and i < len(obj_data["boxes"]):
                        frame_box = obj_data["boxes"][i]
                        break
                if frame_box is not None:
                    # Real annotation
                    annotations_by_frame[frame_idx].append({
                        "obj_id": obj_id,
                        "box": frame_box,
                        "is_dummy": False
                    })
                else:
                    # Dummy/negative annotation (box outside image)
                    # We'll use [-1, -1, -1, -1] as a dummy box
                    annotations_by_frame[frame_idx].append({
                        "obj_id": obj_id,
                        "box": [-1, -1, -1, -1],
                        "is_dummy": True
                    })

        # --- NEW LOGIC: For each frame, send all annotations to SAM2 ---
        self.tracked_objects = {}  # Reset tracked objects for this phase
        self.propagation_results = {}  # Reset propagation results

        for i, frame_path in enumerate(frame_files):
            frame_idx = self._extract_frame_idx_from_path(frame_path)
            frame = Image.open(frame_path).convert("RGB")
            frame_np = np.array(frame)

            frame_annotations = annotations_by_frame[frame_idx]
            masks_for_frame = {}
            for ann in frame_annotations:
                obj_id = ann["obj_id"]
                box = ann["box"]
                is_dummy = ann["is_dummy"]
                # Only add real boxes to SAM2, but always call add_box (SAM2 should handle dummy boxes appropriately)
                mask_logits = self.sam_wrapper.add_box(frame_idx=frame_idx, obj_id=obj_id, box=box)
                if mask_logits is None:
                    # Fallback to empty mask
                    mask_logits = np.zeros((frame_np.shape[0], frame_np.shape[1]), dtype=bool)
                masks_for_frame[obj_id] = mask_logits
                # Update tracked_objects
                if obj_id not in self.tracked_objects:
                    self.tracked_objects[obj_id] = {
                        "id": obj_id,
                        "class": all_objects[obj_id]["class"],
                        "first_detected": all_objects[obj_id]["first_detected"],
                        "boxes": [],
                        "embeddings": [],
                        "masks": [],
                        "last_seen": frame_idx,
                        "confidence": []
                    }
                self.tracked_objects[obj_id]["boxes"].append(box)
                self.tracked_objects[obj_id]["masks"].append(mask_logits)
                self.tracked_objects[obj_id]["last_seen"] = frame_idx
            self.propagation_results[frame_idx] = masks_for_frame

            # Optionally, update results dict for this frame
            results["frame_results"][frame_idx] = {
                "annotations": frame_annotations,
                "tracked_objects": list(masks_for_frame.keys())
            }

        print(f"Processed {len(all_obj_ids)} objects across {len(all_frame_indices)} frames with explicit per-frame, per-object annotations.")
        # Optionally, return results or save to disk
        return results

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