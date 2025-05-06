# sam2_wrapper.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Assuming the SAM2 package is installed and available
from sam2.build_sam import build_sam2_video_predictor

class SAM2VideoWrapper:
    def __init__(self, checkpoint_path, config_path, device=None):
        """Initialize SAM2 video predictor"""
        # Set up device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                # Fall back to CPU for unsupported ops
                import os
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            else:
                device = "cpu"
        
        self.device = device
        print(f"SAM2 using device: {device}")
        
        # Set up device-specific options
        if device == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
        # Initialize the predictor
        self.predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        self.frames = None
        self.num_frames = 0
    
    def set_video(self, frames):
        """Set video frames for processing"""
        self.frames = frames
        self.num_frames = len(frames)
        self.predictor.initialize_video_frames(frames)
        print(f"Set {len(frames)} frames for processing")
    
    def add_box(self, frame_idx, obj_id, box):
        """Add a box prompt for an object
        
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
    
    def propagate_masks(self, objects_to_track=None):
        """Propagate masks to all frames in the video
        
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
    
    def visualize_frame(self, frame_idx, mask=None, obj_ids=None, box=None):
        """Visualize a frame with segmentation mask and prompts"""
        frame = self.predictor.get_video_frame(frame_idx)
        
        plt.figure(figsize=(10, 10))
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
        
        if box is not None:
            x1, y1, x2, y2 = box
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b', linewidth=2)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()