#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
from typing import Dict, List, Tuple, Union, Optional
import argparse

# SAM2 imports
from sam2.build_sam import build_sam2_video_predictor

class SAM2VideoPredictor:
    def __init__(self, checkpoint_path, model_config, device=None):
        """
        Initialize the SAM2 video predictor with checkpoint and config
        
        Args:
            checkpoint_path: Path to the SAM2 model checkpoint
            model_config: Path to the model configuration file
            device: Device to run inference on ('cuda', 'mps', or 'cpu')
        """
        # Set up device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                # Fall back to CPU for unsupported ops on MPS
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Using device: {device}")
        
        # Set up precision
        if device == "cuda":
            # Use bfloat16 for better performance
            self.dtype = torch.bfloat16
            # Turn on TensorFloat32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.dtype = torch.float32
        
        # Load the SAM2 video predictor
        self.predictor = build_sam2_video_predictor(
            model_config, 
            checkpoint_path, 
            device=device
        )
        print("SAM2 Video Predictor initialized")
    
    def set_video(self, video_frames):
        """
        Set the video frames for the predictor
        
        Args:
            video_frames: List of video frames (PIL Images or numpy arrays)
        """
        # Convert frames to proper format if needed
        processed_frames = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                # If already numpy array, ensure RGB
                if frame.shape[-1] == 4:  # RGBA
                    frame = frame[..., :3]  # Convert to RGB
                processed_frames.append(frame)
            else:
                # Convert PIL Image to numpy array
                processed_frames.append(np.array(frame.convert("RGB")))
        
        # Initialize the predictor with the video frames
        self.predictor.initialize_video_frames(processed_frames)
        self.num_frames = len(processed_frames)
        print(f"Set {len(processed_frames)} frames for processing")
    
    def add_points(self, frame_idx, obj_id, points, labels=None):
        """
        Add point prompts for an object
        
        Args:
            frame_idx: Frame index to interact with
            obj_id: Unique object ID
            points: List of (x, y) coordinates
            labels: List of point labels (1=positive, 0=negative)
        
        Returns:
            Mask on the current frame
        """
        if labels is None:
            # Default to positive points
            labels = np.ones(len(points), dtype=np.int64)
        
        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int64)
        
        mask, _ = self.predictor.add_new_points_or_box(
            frame_idx=frame_idx,
            obj_id=obj_id,
            point_coords=point_coords,
            point_labels=point_labels
        )
        
        return mask
    
    def add_box(self, frame_idx, obj_id, box):
        """
        Add a box prompt for an object
        
        Args:
            frame_idx: Frame index to interact with
            obj_id: Unique object ID
            box: Bounding box coordinates [x_min, y_min, x_max, y_max]
        
        Returns:
            Mask on the current frame
        """
        box_coords = np.array(box, dtype=np.float32)
        
        mask, _ = self.predictor.add_new_points_or_box(
            frame_idx=frame_idx,
            obj_id=obj_id,
            box_coords=box_coords
        )
        
        return mask
    
    def add_points_with_box(self, frame_idx, obj_id, points, box, labels=None):
        """
        Add both point and box prompts for an object
        
        Args:
            frame_idx: Frame index to interact with
            obj_id: Unique object ID
            points: List of (x, y) coordinates
            box: Bounding box coordinates [x_min, y_min, x_max, y_max]
            labels: List of point labels (1=positive, 0=negative)
        
        Returns:
            Mask on the current frame
        """
        if labels is None:
            # Default to positive points
            labels = np.ones(len(points), dtype=np.int64)
        
        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int64)
        box_coords = np.array(box, dtype=np.float32)
        
        mask, _ = self.predictor.add_new_points_or_box(
            frame_idx=frame_idx,
            obj_id=obj_id,
            point_coords=point_coords,
            point_labels=point_labels,
            box_coords=box_coords
        )
        
        return mask
    
    def propagate_masks(self, objects_to_track=None):
        """
        Propagate masks to all frames in the video
        
        Args:
            objects_to_track: List of object IDs to track (default: all objects)
            
        Returns:
            Dictionary mapping frame indices to segmentation results
        """
        video_segments = {}
        
        # Propagate to all frames
        for frame_idx in range(self.num_frames):
            masks, obj_ids = self.predictor.propagate_masks_to_frame(
                frame_idx=frame_idx,
                objects_to_track=objects_to_track
            )
            
            if len(masks) > 0:
                video_segments[frame_idx] = {
                    "masks": masks,
                    "obj_ids": obj_ids
                }
        
        return video_segments
    
    def visualize_frame(self, frame_idx, mask=None, obj_ids=None, points=None, box=None, figsize=(10, 10)):
        """
        Visualize a frame with segmentation mask and prompts
        
        Args:
            frame_idx: Frame index to visualize
            mask: Optional segmentation mask to overlay
            obj_ids: Object IDs for the masks
            points: Optional points to display
            box: Optional box to display
            figsize: Figure size for the plot
        """
        frame = self.predictor.get_video_frame(frame_idx)
        
        plt.figure(figsize=figsize)
        plt.imshow(frame)
        plt.title(f"frame {frame_idx}")
        
        if mask is not None:
            colors = [(1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5), (1, 1, 0, 0.5), (1, 0, 1, 0.5)]
            
            if obj_ids is None:
                # Single mask
                plt.imshow(mask, alpha=0.5, cmap="jet")
            else:
                # Multiple masks
                for i, (m, obj_id) in enumerate(zip(mask, obj_ids)):
                    color_idx = (obj_id % len(colors))
                    plt.imshow(m, alpha=0.5, cmap=plt.cm.colors.ListedColormap([colors[color_idx]]))
        
        if points is not None:
            for p, label in points:
                if label == 1:  # Positive point
                    plt.plot(p[0], p[1], 'go', markersize=8)
                else:  # Negative point
                    plt.plot(p[0], p[1], 'ro', markersize=8)
        
        if box is not None:
            x1, y1, x2, y2 = box
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b', linewidth=2)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def load_video_frames(video_path=None, frames_dir=None, pattern="*.jpg"):
    """
    Load video frames either from a video file or directory of image frames
    
    Args:
        video_path: Path to a video file
        frames_dir: Directory containing image frames
        pattern: Glob pattern for frame files
        
    Returns:
        List of frames as numpy arrays
    """
    frames = []
    
    if video_path and os.path.exists(video_path):
        # Load frames from video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
    elif frames_dir and os.path.isdir(frames_dir):
        # Load frames from directory
        frame_files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
        
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir} with pattern {pattern}")
        
        for frame_file in frame_files:
            frame = np.array(Image.open(frame_file).convert("RGB"))
            frames.append(frame)
    
    else:
        raise ValueError("Either video_path or frames_dir must be provided")
    
    return frames


def main():
    parser = argparse.ArgumentParser(description="SAM2 Video Segmentation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to SAM2 model config")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--frames-dir", type=str, help="Directory with video frames")
    parser.add_argument("--frame-pattern", type=str, default="*.jpg", help="Frame filename pattern")
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], help="Device for inference")
    parser.add_argument("--output-dir", type=str, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video and not args.frames_dir:
        parser.error("Either --video or --frames-dir must be provided")
    
    # Load video frames
    try:
        video_frames = load_video_frames(args.video, args.frames_dir, args.frame_pattern)
        print(f"Loaded {len(video_frames)} frames")
    except Exception as e:
        print(f"Error loading video frames: {e}")
        return
    
    # Initialize SAM2 Video Predictor
    predictor = SAM2VideoPredictor(args.checkpoint, args.config, args.device)
    
    # Set video frames
    predictor.set_video(video_frames)
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Example: Add a demo point and box
    print("This is a command-line tool for SAM2 video processing.")
    print("Use the Python class directly for interactive usage.")
    
    # Show the first frame
    predictor.visualize_frame(0)


if __name__ == "__main__":
    main() 