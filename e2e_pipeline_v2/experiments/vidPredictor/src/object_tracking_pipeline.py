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
        self.detector = OWLv2Detector( device=device)
        self.sam_wrapper = SAM2VideoWrapper(sam2_checkpoint, sam2_config, device=device)
        self.tracker = ObjectTracker()
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Config
        self.confidence_threshold = confidence_threshold
        
        # Tracking state
        self.tracked_objects = {}  # id -> object data
        self.next_id = 1
        
    def process_video(self, frames_dir: str, text_queries: List[str]):
        # Get all frames sorted
        frames_path = Path(frames_dir)
        frame_files = sorted([f for f in frames_path.glob("*.jpg") or frames_path.glob("*.png")])
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Processing {len(frame_files)} frames with queries: {text_queries}")
        
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
            box_area = first_frame_np[int(y1):int(y2), int(x1):int(x2)]
            
            # Extract embedding for this object
            embedding = self.detector.extract_embedding(first_frame, box)
            
            # Add box to SAM2 to get mask
            self.sam_wrapper.initialize_video_frames([first_frame_np])
            self.sam_wrapper.add_box(box)
            mask_logits = self.sam_wrapper.get_mask_for_frame(0)
            
            # Store this object with a new ID
            object_id = self.next_id
            self.next_id += 1
            
            self.tracked_objects[object_id] = {
                "id": object_id,
                "class": label,
                "first_detected": 0,  # frame 0
                "boxes": [box.tolist() if isinstance(box, torch.Tensor) else box],
                "embeddings": [embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding],
                "masks": [mask_logits.cpu().numpy() if isinstance(mask_logits, torch.Tensor) else mask_logits],
                "last_seen": 0,  # frame 0
                "confidence": [conf]
            }
            
            # Store in results
            results["object_tracks"][object_id] = self.tracked_objects[object_id]
        
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
        
        # Process remaining frames
        for frame_idx in range(1, len(frame_files)):
            print(f"Processing frame {frame_idx}/{len(frame_files)}")
            frame_path = frame_files[frame_idx]
            frame = Image.open(frame_path).convert("RGB")
            frame_np = np.array(frame)
            
            # Get new detections
            detections = self.detector.detect(
                image=frame,
                text_queries=text_queries,
                confidence_threshold=self.confidence_threshold
            )
            
            # Track objects across frames
            tracked_in_frame = self.tracker.track_objects(
                previous_objects=self.tracked_objects,
                new_detections={
                    "boxes": detections["boxes"],
                    "labels": detections["labels"],
                    "scores": detections["scores"],
                },
                current_frame=frame,
                detector=self.detector,
                frame_idx=frame_idx
            )
            
            # Initialize SAM2 with the current frame
            self.sam_wrapper.initialize_video_frames([frame_np])
            
            # Process each tracked object
            for obj_id, obj_data in tracked_in_frame.items():
                # If this is a new object, initialize SAM2 with its box
                if obj_data["first_detected"] == frame_idx:
                    box = obj_data["boxes"][-1]
                    self.sam_wrapper.add_box(box)
                    mask_logits = self.sam_wrapper.get_mask_for_frame(0)
                    obj_data["masks"].append(
                        mask_logits.cpu().numpy() if isinstance(mask_logits, torch.Tensor) else mask_logits
                    )
                
                # If this is a continuing object, propagate mask from previous frame
                else:
                    prev_mask = obj_data["masks"][-1]
                    prev_box = obj_data["boxes"][-2]  # Second-to-last box
                    curr_box = obj_data["boxes"][-1]  # Latest box
                    
                    # Prepare mask from previous frame
                    prev_mask_tensor = torch.tensor(prev_mask).to(self.device)
                    
                    # Process the propagation - simplified for demo
                    self.sam_wrapper.add_mask(prev_mask_tensor)
                    propagated_mask = self.sam_wrapper.get_mask_for_frame(0)
                    
                    obj_data["masks"].append(
                        propagated_mask.cpu().numpy() if isinstance(propagated_mask, torch.Tensor) else propagated_mask
                    )
                
                # Update tracked objects
                self.tracked_objects[obj_id] = obj_data
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
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Draw bounding box
            box = obj_data["boxes"][-1]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw mask overlay
            mask = obj_data["masks"][-1]
            if isinstance(mask, np.ndarray):
                # Convert logits to binary mask if needed
                if mask.dtype == np.float32 or mask.dtype == np.float64:
                    mask = mask > 0
                
                # Create mask overlay
                mask_overlay = vis_frame.copy()
                mask_color = (*color, 128)  # Add alpha
                
                # Apply mask
                mask_bool = mask.astype(bool)
                mask_overlay[mask_bool] = mask_color
                
                # Blend
                alpha = 0.5
                vis_frame = cv2.addWeighted(mask_overlay, alpha, vis_frame, 1-alpha, 0)
            
            # Draw label
            label = f"{obj_data['class']} #{obj_id}"
            conf = obj_data["confidence"][-1]
            text = f"{label} ({conf:.2f})"
            cv2.putText(vis_frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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



# python object_tracking_pipeline.py \
#   --frames-dir /path/to/video/frames \
#   --text-queries "car,person,bicycle" \
#   --output-dir ./tracking_results \
#   --owlv2-checkpoint /path/to/owlv2/model \
#   --sam2-checkpoint /path/to/sam2/model \
#   --sam2-config /path/to/sam2/config \
#   --confidence 0.2

# tected goat with confidence 0.102 at [259.6, 142.9, 405.1, 475.4]
# Detected goat with confidence 0.119 at [372.4, 85.9, 857.8, 473.5]
# Detected goat with confidence 0.137 at [648.6, 102.1, 781.8, 454.4]
# Detected goat with confidence 0.140 at [267.1, 89.3, 458.9, 478.5]
# Detected goat with confidence 0.175 at [368.9, 369.0, 471.7, 479.3]
# Detected goat with confidence 0.104 at [348.0, 356.5, 409.6, 478.5]
# Detected goat with confidence 0.155 at [70.1, 430.9, 177.0, 480.4]
#  Detections [{'box': [647.4099731445312, 88.04713439941406, 780.1903686523438, 234.6172332763672], 'score': 0.33430543541908264, 'label': 0, 'text': 'goat'}, {'box': [338.20086669921875, 91.2125015258789, 379.37939453125, 163.16427612304688], 'score': 0.14112025499343872, 'label': 0, 'text': 'goat'}, {'box': [266.2550964355469, 96.26411437988281, 368.3487548828125, 182.85906982421875], 'score': 0.11737194657325745, 'label': 0, 'text': 'goat'}, {'box': [613.8699340820312, 150.54696655273438, 662.9747314453125, 217.56640625], 'score': 0.11062438040971756, 'label': 0, 'text': 'goat'}, {'box': [741.7838745117188, 135.00384521484375, 783.1548461914062, 221.33778381347656], 'score': 0.32746225595474243, 'label': 0, 'text': 'goat'}, {'box': [646.8192749023438, 112.86333465576172, 766.5968017578125, 234.1778564453125], 'score': 0.14156973361968994, 'label': 0, 'text': 'goat'}, {'box': [731.4899291992188, 124.2536849975586, 848.8386840820312, 362.7610778808594], 'score': 0.21958966553211212, 'label': 0, 'text': 'goat'}, {'box': [62.57223129272461, 140.21884155273438, 662.048095703125, 482.3572998046875], 'score': 0.35990920662879944, 'label': 0, 'text': 'goat'}, {'box': [260.2547302246094, 143.4312286376953, 670.4420166015625, 486.187255859375], 'score': 0.485964298248291, 'label': 0, 'text': 'goat'}, {'box': [45.00772476196289, 248.51048278808594, 63.05266189575195, 286.515869140625], 'score': 0.1412220597267151, 'label': 0, 'text': 'goat'}, {'box': [42.06721496582031, 246.10081481933594, 73.11885833740234, 408.2206115722656], 'score': 0.10995682328939438, 'label': 0, 'text': 'goat'}, {'box': [656.430908203125, 93.86257934570312, 860.3806762695312, 452.3819580078125], 'score': 0.20164254307746887, 'label': 0, 'text': 'goat'}, {'box': [713.7269897460938, 103.97442626953125, 855.4326171875, 455.4469909667969], 'score': 0.13221421837806702, 'label': 0, 'text': 'goat'}, {'box': [259.5897216796875, 142.90858459472656, 405.1321105957031, 475.4350891113281], 'score': 0.10194676369428635, 'label': 0, 'text': 'goat'}, {'box': [372.4270935058594, 85.86034393310547, 857.8440551757812, 473.54547119140625], 'score': 0.11945652216672897, 'label': 0, 'text': 'goat'}, {'box': [648.5542602539062, 102.0804672241211, 781.7900390625, 454.4112548828125], 'score': 0.1372501105070114, 'label': 0, 'text': 'goat'}, {'box': [267.06689453125, 89.34765625, 458.937255859375, 478.5168762207031], 'score': 0.13998572528362274, 'label': 0, 'text': 'goat'}, {'box': [368.91650390625, 369.0050964355469, 471.6726989746094, 479.3065185546875], 'score': 0.1745559275150299, 'label': 0, 'text': 'goat'}, {'box': [348.00006103515625, 356.5211181640625, 409.606689453125, 478.4908752441406], 'score': 0.10445903986692429, 'label': 0, 'text': 'goat'}, {'box': [70.08139038085938, 430.9385986328125, 176.95315551757812, 480.38909912109375], 'score': 0.1554412841796875, 'label': 0, 'text': 'goat'}]
# Traceback (most recent call last):
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 337, in <module>
#     main()
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 331, in main
#     pipeline.process_video(
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/object_tracking_pipeline.py", line 91, in process_video
#     for i, (box, label, conf) in enumerate(zip(detections["boxes"], detections["labels"], detections["scores"])):
#                                                ~~~~~~~~~~^^^^^^^^^
# TypeError: list indices must be integers or slices, not str
