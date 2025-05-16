import os
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Generator

class SAM2VideoSegmenter:
    def __init__(self, model_cfg, checkpoint_path):
        """
        Initialize SAM2 Video Segmenter
        
        Args:
            model_cfg: Path to model configuration file
            checkpoint_path: Path to model checkpoint file
        """
        self.device = self._select_device()
        self._configure_device()
        self._load_model(model_cfg, checkpoint_path)
        
    def _select_device(self):
        """Select the best available device for computation"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device
        
    def _configure_device(self):
        """Configure device-specific settings"""
        if self.device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
    
    def _load_model(self, model_cfg, checkpoint_path):
        """Load SAM2 model"""
        from sam2.build_sam import build_sam2_video_predictor
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=self.device)
    
    def initialize_video(self, frames_dir: str):
        """
        Initialize the video for segmentation
        
        Args:
            frames_dir: Directory containing video frames
            
        Returns:
            Inference state for the video
        """
        inference_state = self.predictor.init_state(video_path=frames_dir)
        self.predictor.reset_state(inference_state)
        return inference_state
    
    def process_tracking_objects(self, tracking_objects: List[Dict], inference_state, frame_nums: List[int]):
        """
        Process tracking objects and send them to SAM2
        
        Args:
            tracking_objects: List of tracking objects with frame occurrences
            inference_state: SAM2 inference state
            frame_nums: List of frame numbers to process
            
        Returns:
            None
        """
        print("\n=== DEBUG: Processing tracking objects ===")
        print(f"Total objects: {len(tracking_objects)}")
        print(f"Total frames to process: {len(frame_nums)}")
        
        # Track which objects are sent to SAM2
        objects_sent = {obj["objectID"]: [] for obj in tracking_objects}
        
        for frame_num in frame_nums:
            print(f"\nProcessing frame {frame_num}")
            for obj_class in tracking_objects:
                obj_id = obj_class["objectID"]
                obj_name = obj_class["objectName"]
                
                valid_occurrences = [w for w in obj_class['frameOccurences'] if w['frameNum'] == frame_num]
                if valid_occurrences:
                    occ = valid_occurrences[0]  # TODO - handle multiple instances
                    print(f'  Object {obj_id} ({obj_name}) found, box: {occ["box"]}')
                    self._send_to_sam2(inference_state, frame_num, obj_id, occ['box'])
                    objects_sent[obj_id].append(frame_num)
                else:
                    print(f"  Object {obj_id} ({obj_name}) NOT in frame {frame_num}, sending default box")
                    self._send_to_sam2(inference_state, frame_num, obj_id, [-1,-1,-1,-1])
        
        print("\nSummary of objects sent to SAM2:")
        for obj_id, frames in objects_sent.items():
            print(f"  Object {obj_id}: sent in {len(frames)} frames")
            if len(frames) > 0:
                print(f"    Frame numbers: {frames[:10]}{'...' if len(frames) > 10 else ''}")
    
    def _send_to_sam2(self, inference_state, frame_idx, obj_id, box):
        """Send data to SAM2 API"""
        self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box,
        )
    
    def propagate_segmentation(self, inference_state, reverse=False, min_mask_pixels=10):
        """
        Propagate segmentation through video frames
        
        Args:
            inference_state: SAM2 inference state
            reverse: Whether to propagate in reverse direction
            min_mask_pixels: Minimum number of pixels required for a valid mask
            
        Returns:
            Dictionary mapping frame indices to segmentation results
        """
        print("\n=== DEBUG: Propagating segmentation ===")
        video_segments = {}
        segment_stats = {}  # Track objects and their appearances
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state, reverse=reverse):
            # Debug mask logits
            if len(out_obj_ids) > 0:
                print(f"Frame {out_frame_idx}: Found {len(out_obj_ids)} objects: {out_obj_ids}")
                
                # Create frame entry if needed
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = {}
                
                for i, obj_id in enumerate(out_obj_ids):
                    # Convert mask to binary numpy array
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    mask_sum = np.sum(mask)
                    print(f"  Object {obj_id}: mask sum = {mask_sum}")
                    
                    # Only include masks that have significant pixels
                    # This filters out default or almost empty masks
                    if mask_sum >= min_mask_pixels:
                        # Calculate mask dimensions and check if it's a reasonable size
                        if len(mask.shape) >= 2:
                            height, width = mask.shape[:2]
                            
                            # Calculate the bounding box
                            mask_positions = np.where(mask)
                            
                            if len(mask_positions[0]) > 0:
                                y_min, y_max = np.min(mask_positions[0]), np.max(mask_positions[0])
                                x_min, x_max = np.min(mask_positions[1]), np.max(mask_positions[1])
                                
                                # Calculate box dimensions
                                box_width = x_max - x_min
                                box_height = y_max - y_min
                                
                                # Only include masks with reasonable dimensions
                                if box_width > 0 and box_height > 0:
                                    video_segments[out_frame_idx][str(obj_id)] = mask
                                    
                                    # Update stats
                                    if obj_id not in segment_stats:
                                        segment_stats[obj_id] = []
                                    segment_stats[obj_id].append(out_frame_idx)
                                else:
                                    print(f"  WARNING: Object {obj_id} has zero-size bounding box in frame {out_frame_idx}")
                            else:
                                print(f"  WARNING: Object {obj_id} has no mask points in frame {out_frame_idx}")
                        else:
                            print(f"  WARNING: Object {obj_id} has unexpected mask shape: {mask.shape}")
                    else:
                        print(f"  WARNING: Object {obj_id} has empty or minimal mask in frame {out_frame_idx}")
            else:
                print(f"Frame {out_frame_idx}: No objects detected")
        
        # Print summary of segmentation results
        print("\nSegmentation Summary:")
        print(f"Total frames with segments: {len(video_segments)}")
        print(f"Total unique objects: {len(segment_stats)}")
        
        for obj_id, frames in segment_stats.items():
            print(f"  Object {obj_id}: appears in {len(frames)} frames")
            if len(frames) > 0:
                print(f"    Frame numbers: {frames[:10]}{'...' if len(frames) > 10 else ''}")
        
        return video_segments 