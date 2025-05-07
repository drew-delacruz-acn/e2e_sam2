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

            # Process each tracked object - use propagated masks if available
            # First, find objects that were updated in this frame
            tracked_in_this_frame = []
            for obj_id, obj_data in self.tracked_objects.items():
                if obj_data["last_seen"] == frame_idx:
                    tracked_in_this_frame.append(obj_id)

            for obj_id in tracked_in_this_frame:
                obj_data = self.tracked_objects[obj_id]
                # If this is a new object, we need to initialize it in SAM2
                if obj_data["first_detected"] == frame_idx:
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

# python e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py   --frames-dir "/home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset"   --text-queries "goat"   --output-dir ./tracking_results   --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt   --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml  --confidence 0.3
# Using device: cuda
# OWLv2 using device: cuda
# SAM2 using device: cuda
# /home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   warnings.warn(
# /home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
# Processing 13 frames with queries: ['goat']
# Setting up SAM2 with frames directory: /home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset
# frame loading (JPEG): 100%|██████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 36.96it/s]
# Set video from directory: /home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset
# /home/ubuntu/code/drew/e2e_sam2/venv/lib/python3.12/site-packages/transformers/models/owlv2/processing_owlv2.py:213: FutureWarning: `post_process_object_detection` method is deprecated for OwlVitProcessor and will be removed in v5. Use `post_process_grounded_object_detection` instead.
#   warnings.warn(
# Detected goat with confidence 0.334 at [647.4, 88.0, 780.2, 234.6]
# Detected goat with confidence 0.327 at [741.8, 135.0, 783.2, 221.3]
# Detected goat with confidence 0.360 at [62.6, 140.2, 662.0, 482.4]
# Detected goat with confidence 0.486 at [260.3, 143.4, 670.4, 486.2]
#  Detections {'boxes': tensor([[647.4100,  88.0470, 780.1903, 234.6174],
#         [741.7838, 135.0038, 783.1549, 221.3380],
#         [ 62.5728, 140.2188, 662.0476, 482.3569],
#         [260.2548, 143.4311, 670.4420, 486.1872]], device='cuda:0'), 'scores': tensor([0.3343, 0.3275, 0.3599, 0.4860], device='cuda:0'), 'labels': ['goat', 'goat', 'goat', 'goat']}
# Adding box for object 1 at frame 0: [647, 88, 780, 234]
# /home/ubuntu/code/drew/sam2/sam2/sam2_video_predictor.py:786: UserWarning: cannot import name '_C' from 'sam2' (/home/ubuntu/code/drew/sam2/sam2/__init__.py)

# Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
#   pred_masks_gpu = fill_holes_in_mask_scores(
# Adding box for object 2 at frame 0: [741, 135, 783, 221]
# Adding box for object 3 at frame 0: [62, 140, 662, 482]
# Adding box for object 4 at frame 0: [260, 143, 670, 486]
# Running mask propagation for all tracked objects...
# propagate in video:   0%|                                                                                                     | 0/13 [00:00<?, ?it/s]Processed frame 0, found 4 objects
# Processed frame 1, found 4 objects
# propagate in video:  15%|██████████████▎                                                                              | 2/13 [00:00<00:04,  2.22it/s]Processed frame 2, found 4 objects
# propagate in video:  23%|█████████████████████▍                                                                       | 3/13 [00:01<00:06,  1.48it/s]Processed frame 3, found 4 objects
# propagate in video:  31%|████████████████████████████▌                                                                | 4/13 [00:02<00:07,  1.20it/s]Processed frame 4, found 4 objects
# propagate in video:  38%|███████████████████████████████████▊                                                         | 5/13 [00:04<00:07,  1.03it/s]Processed frame 5, found 4 objects
# propagate in video:  46%|██████████████████████████████████████████▉                                                  | 6/13 [00:05<00:07,  1.09s/it]Processed frame 6, found 4 objects
# propagate in video:  54%|██████████████████████████████████████████████████                                           | 7/13 [00:06<00:07,  1.20s/it]Processed frame 7, found 4 objects
# propagate in video:  62%|█████████████████████████████████████████████████████████▏                                   | 8/13 [00:08<00:06,  1.31s/it]Processed frame 8, found 4 objects
# propagate in video:  69%|████████████████████████████████████████████████████████████████▍                            | 9/13 [00:10<00:05,  1.39s/it]Processed frame 9, found 4 objects
# propagate in video:  77%|██████████████████████████████████████████████████████████████████████▊                     | 10/13 [00:11<00:04,  1.44s/it]Processed frame 10, found 4 objects
# propagate in video:  85%|█████████████████████████████████████████████████████████████████████████████▊              | 11/13 [00:13<00:02,  1.48s/it]Processed frame 11, found 4 objects
# propagate in video:  92%|████████████████████████████████████████████████████████████████████████████████████▉       | 12/13 [00:14<00:01,  1.51s/it]Processed frame 12, found 4 objects
# propagate in video: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:16<00:00,  1.26s/it]
# Propagated masks for 4 objects across 13 frames
# Processing frame 1/13
# Detected goat with confidence 0.551 at [292.1, 145.3, 658.3, 483.1]
# Detected goat with confidence 0.302 at [631.3, 114.5, 943.1, 476.2]
# Detected goat with confidence 0.333 at [72.5, 143.8, 638.6, 474.4]
# Detected goat with confidence 0.334 at [811.2, 169.6, 940.2, 470.8]
# -----Initializing new objects-----
# detections: [{'box': [292.1224670410156, 145.3485870361328, 658.2644653320312, 483.1181640625], 'score': 0.5508526563644409, 'text': 'goat'}, {'box': [631.3438110351562, 114.49091339111328, 943.1325073242188, 476.17657470703125], 'score': 0.30174025893211365, 'text': 'goat'}, {'box': [72.48252868652344, 143.81036376953125, 638.5888671875, 474.4100341796875], 'score': 0.33306947350502014, 'text': 'goat'}, {'box': [811.1805419921875, 169.5598907470703, 940.1514892578125, 470.7922668457031], 'score': 0.3338971734046936, 'text': 'goat'}]
# -----Initializing [{'box': [292.1224670410156, 145.3485870361328, 658.2644653320312, 483.1181640625], 'score': 0.5508526563644409, 'text': 'goat'}, {'box': [631.3438110351562, 114.49091339111328, 943.1325073242188, 476.17657470703125], 'score': 0.30174025893211365, 'text': 'goat'}, {'box': [72.48252868652344, 143.81036376953125, 638.5888671875, 474.4100341796875], 'score': 0.33306947350502014, 'text': 'goat'}, {'box': [811.1805419921875, 169.5598907470703, 940.1514892578125, 470.7922668457031], 'score': 0.3338971734046936, 'text': 'goat'}]-----
# Initialized object 1 (goat)
# -----Initializing [{'box': [292.1224670410156, 145.3485870361328, 658.2644653320312, 483.1181640625], 'score': 0.5508526563644409, 'text': 'goat'}, {'box': [631.3438110351562, 114.49091339111328, 943.1325073242188, 476.17657470703125], 'score': 0.30174025893211365, 'text': 'goat'}, {'box': [72.48252868652344, 143.81036376953125, 638.5888671875, 474.4100341796875], 'score': 0.33306947350502014, 'text': 'goat'}, {'box': [811.1805419921875, 169.5598907470703, 940.1514892578125, 470.7922668457031], 'score': 0.3338971734046936, 'text': 'goat'}]-----
# Initialized object 2 (goat)
# -----Initializing [{'box': [292.1224670410156, 145.3485870361328, 658.2644653320312, 483.1181640625], 'score': 0.5508526563644409, 'text': 'goat'}, {'box': [631.3438110351562, 114.49091339111328, 943.1325073242188, 476.17657470703125], 'score': 0.30174025893211365, 'text': 'goat'}, {'box': [72.48252868652344, 143.81036376953125, 638.5888671875, 474.4100341796875], 'score': 0.33306947350502014, 'text': 'goat'}, {'box': [811.1805419921875, 169.5598907470703, 940.1514892578125, 470.7922668457031], 'score': 0.3338971734046936, 'text': 'goat'}]-----
# Initialized object 3 (goat)
# -----Initializing [{'box': [292.1224670410156, 145.3485870361328, 658.2644653320312, 483.1181640625], 'score': 0.5508526563644409, 'text': 'goat'}, {'box': [631.3438110351562, 114.49091339111328, 943.1325073242188, 476.17657470703125], 'score': 0.30174025893211365, 'text': 'goat'}, {'box': [72.48252868652344, 143.81036376953125, 638.5888671875, 474.4100341796875], 'score': 0.33306947350502014, 'text': 'goat'}, {'box': [811.1805419921875, 169.5598907470703, 940.1514892578125, 470.7922668457031], 'score': 0.3338971734046936, 'text': 'goat'}]-----
# Initialized object 4 (goat)
# Traceback (most recent call last):
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 395, in <module>
#     main()
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 389, in main
#     pipeline.process_video(
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 218, in process_video
#     if obj_data["first_detected"] == frame_idx:
#        ~~~~~~~~^^^^^^^^^^^^^^^^^^
# TypeError: list indices must be integers or slices, not str