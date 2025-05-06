# sam2_wrapper.py (corrected version)
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image

# Import SAM2 functionality
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
        self.inference_state = None
        self.frames_dir = None
    
    def set_video(self, frames=None, frames_dir=None):
        """
        Set video for processing either from frames array or directory
        
        Args:
            frames: List of frames as numpy arrays (optional)
            frames_dir: Directory containing video frames (optional)
        """
        if frames_dir is not None:
            # Using directory path approach (like in vidPredict_demo.py)
            self.frames_dir = frames_dir
            self.inference_state = self.predictor.init_state(video_path=frames_dir)
            self.predictor.reset_state(self.inference_state)
            print(f"Set video from directory: {frames_dir}")
            return
            
        elif frames is not None:
            # Using direct frames approach - save frames to temporary directory
            import tempfile
            import cv2
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            print(f"Creating temporary directory for frames: {temp_dir}")
            
            # Save frames to temp directory
            for i, frame in enumerate(frames):
                # Convert to BGR for OpenCV
                if isinstance(frame, np.ndarray):
                    # If RGB, convert to BGR
                    if frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                else:
                    # If PIL image, convert to numpy then BGR
                    frame_np = np.array(frame)
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                
                # Save frame
                cv2.imwrite(os.path.join(temp_dir, f"{i}.jpg"), frame_bgr)
            
            # Initialize with temp directory
            self.frames_dir = temp_dir
            self.inference_state = self.predictor.init_state(video_path=temp_dir)
            self.predictor.reset_state(self.inference_state)
            print(f"Set {len(frames)} frames via temporary directory")
        else:
            raise ValueError("Either frames or frames_dir must be provided")
    
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
        
        _,_,mask = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_coords
        )
        return mask
    
    def propagate_masks(self, objects_to_track=None):
        """Propagate masks to all frames in the video
        
        Args:
            objects_to_track: List of object IDs to track (default: all objects)
            
        Returns:
            Dictionary mapping frame indices to segmentation results
        """
        # Run propagation using the correct method
        try:
            # This is the correct method name in SAM2
            segments = self.predictor.propagate_in_video(objects_to_track=objects_to_track)
            
            # Convert returned segments to the desired format
            video_segments = {}
            
            # Process segments based on the structure returned
            # The exact structure might vary depending on SAM2 implementation,
            # so we'll handle the most likely cases
            
            if isinstance(segments, dict):
                # If already in frame_idx -> data format
                for frame_idx, data in segments.items():
                    if 'masks' in data and 'obj_ids' in data:
                        masks = data['masks']
                        obj_ids = data['obj_ids']
                        
                        # Convert masks to numpy if needed
                        if isinstance(masks, torch.Tensor):
                            masks = masks.cpu().detach().numpy()
                            # Handle dimensions
                            if masks.ndim > 3:
                                masks = masks.squeeze(0)
                        
                        video_segments[frame_idx] = {
                            'masks': masks,
                            'obj_ids': obj_ids
                        }
            else:
                # If returned as a list or other structure,
                # we need to adapt the processing accordingly
                print("Propagation result format may require adapting the processing code.")
                # Return the raw result for debugging
                return segments
            
            return video_segments
        
        except Exception as e:
            print(f"Error during mask propagation: {str(e)}")
            # Print available methods for debugging
            print("Available methods on predictor:")
            for method_name in dir(self.predictor):
                if not method_name.startswith('_'):
                    print(f"  {method_name}")
            raise e
    
    def visualize_frame(self, frame_idx, mask=None, obj_ids=None, box=None, output_dir=None):
        """Visualize a frame with segmentation mask and prompts"""
        # Get the frame
        if self.frames_dir:
            # Find frame file
            frame_files = [
                p for p in os.listdir(self.frames_dir)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            frame_files.sort(key=lambda p: int(os.path.splitext(p)[0]))
            
            if frame_idx < len(frame_files):
                frame_path = os.path.join(self.frames_dir, frame_files[frame_idx])
                frame = np.array(Image.open(frame_path))
            else:
                raise ValueError(f"Frame index {frame_idx} out of range")
        else:
            # Try to get from predictor
            try:
                frame = self.predictor.get_video_frame(frame_idx)
            except:
                raise ValueError("Cannot retrieve frame - no frames directory or getter method available")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(frame)
        plt.title(f"Frame {frame_idx}")
        
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
        
        # Save visualization if output directory provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, f"segment_frame_{frame_idx:04d}.jpg")
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            print(f"Saved segmentation visualization to {vis_path}")
        
        plt.show()