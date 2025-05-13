# sam2_wrapper.py (corrected version)
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision

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

            print(f'USING DIRECT FRAMES APPROACH')
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
    
    def reset_state(self, frames_dir=None):
        """Reset SAM2 state to ensure consistent data types
        
        Args:
            frames_dir: Optional new frames directory to use
        """
        # Update frames_dir if provided
        if frames_dir is not None:
            self.frames_dir = frames_dir
            
        # Ensure we have a frames directory
        if self.frames_dir is None:
            raise ValueError("Cannot reset state without frames_dir. Call set_video first or provide frames_dir.")
            
        # Reinitialize state completely from scratch
        print(f"Resetting SAM2 state with frames directory: {self.frames_dir}")
        self.inference_state = self.predictor.init_state(video_path=self.frames_dir)
        self.predictor.reset_state(self.inference_state)
    
    def tensor_to_numpy(self, tensor):
        """Safely convert tensor to numpy array, handling device and gradient issues."""
        if isinstance(tensor, torch.Tensor):
            # Move to CPU if on another device and detach from computation graph
            return tensor.detach().cpu().numpy()
        return tensor  # Already a numpy array or other format
    
    def process_mask_logits(self, mask_logits, threshold=0.0):
        """Process mask logits to binary masks with proper dimensions"""
        # Convert to numpy if needed
        if isinstance(mask_logits, torch.Tensor):
            mask_np = self.tensor_to_numpy(mask_logits)
        else:
            mask_np = mask_logits
            
        # Squeeze dimensions
        if mask_np.ndim > 2:
            mask_np = np.squeeze(mask_np)
            
        # Apply threshold to get binary mask
        return mask_np > threshold
    
    def add_box(self, frame_idx, obj_id, box):
        """Add a box prompt for an object
        
        Args:
            frame_idx: Frame index to interact with
            obj_id: Unique object ID
            box: Bounding box coordinates [x_min, y_min, x_max, y_max]
        
        Returns:
            Binary mask on the current frame
        """
        box_coords = np.array(box, dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_coords
        )
        
        # Match vidPredict_demo.py by thresholding the logits
        # Return only the first mask (there should be only one)
        if len(out_mask_logits) > 0:
            return self.process_mask_logits(out_mask_logits[0])
        else:
            return None
    
    def show_mask(self, mask, plt_axis, obj_id=None):
        """Show a mask on the given matplotlib axis, similar to vidPredict_demo.py"""
        # Ensure mask is properly processed
        if isinstance(mask, torch.Tensor):
            mask = self.process_mask_logits(mask)
        
        # Ensure mask is 2D
        print(f"Original mask shape in show_mask: {mask.shape}")
        if len(mask.shape) == 1:
            print(f"Warning: Mask is 1D with shape {mask.shape}, cannot display properly")
            # Return without trying to display this mask
            return
        elif len(mask.shape) > 2:
            # Squeeze to make it 2D
            mask = np.squeeze(mask)
            print(f"Squeezed mask to shape: {mask.shape}")
            
        # Get mask dimensions - now we're sure it's 2D
        h, w = mask.shape
            
        # Choose color based on object ID
        if obj_id is None:
            color = np.array([1, 0, 0, 0.6])  # default red with alpha
        else:
            cmap = plt.get_cmap("tab10")
            color = np.array([*cmap(obj_id % 10)[:3], 0.6])
        
        # Create colored mask
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        # Display
        plt_axis.imshow(mask_image)
    
    def propagate_masks(self, objects_to_track=None):
        """Propagate masks to all frames in the video
        
        Args:
            objects_to_track: List of object IDs to track (default: all objects)
            
        Returns:
            Tuple: (video_segments, boxes_by_frame)
                video_segments: Dictionary mapping frame indices to segmentation results
                boxes_by_frame: Dictionary mapping frame indices to bounding boxes per object
        """
        if self.inference_state is None:
            raise ValueError("Must call set_video before propagating masks")
            
        video_segments = {}
        boxes_by_frame = {}
        
        try:
            print(f"Starting propagate_masks with objects_to_track={objects_to_track}")
            print(f"Using predictor type: {type(self.predictor).__name__}")
            
            # Get propagate_in_video method to inspect
            propagate_method = self.predictor.propagate_in_video
            print(f"Propagate method: {propagate_method.__name__} from {propagate_method.__module__}")
            
            # Directly handle the 3-element tuple from propagate_in_video
            iterator = self.predictor.propagate_in_video(self.inference_state)
            print(f"Iterator type: {type(iterator).__name__}")
            
            # Debug first item from iterator without consuming it
            try:
                # Use tee to peek at the first item without consuming the iterator
                import itertools
                iterator, debug_iterator = itertools.tee(iterator)
                first_item = next(debug_iterator, None)
                print(f"First item type: {type(first_item).__name__}")
                print(f"First item value: {first_item}")
                if isinstance(first_item, tuple):
                    print(f"First item tuple length: {len(first_item)}")
                    for i, element in enumerate(first_item):
                        print(f"  Element {i}: type={type(element).__name__}, value={element}")
            except Exception as peek_error:
                print(f"Error peeking at iterator: {peek_error}")
            
            # Create a unified iterator wrapper that can handle different return values
            def robust_iterator_wrapper(iterator):
                """Wrap the propagate_in_video iterator to handle different API versions."""
                for item in iterator:
                    if not isinstance(item, tuple):
                        print(f"WARNING: Expected tuple but got {type(item).__name__}")
                        # Skip non-tuple items
                        continue
                        
                    if len(item) == 3:
                        # API version returning (frame_idx, obj_ids, mask_logits)
                        yield item
                    elif len(item) == 2:
                        # API version returning (frame_idx, mask_logits) - infer obj_ids
                        frame_idx, mask_logits = item
                        # If objects_to_track is specified, use that as obj_ids
                        # If not, use a default obj_id of 1
                        obj_ids = objects_to_track if objects_to_track is not None else [1]
                        
                        # Make mask_logits a list if it isn't already (needed for iteration)
                        if not isinstance(mask_logits, list):
                            # Ensure we have one mask_logit per obj_id
                            mask_logits = [mask_logits] * len(obj_ids)
                        
                        print(f"Converted 2-element tuple to 3-element: frame_idx={frame_idx}, obj_ids={obj_ids}, mask_logits (count)={len(mask_logits)}")
                        yield (frame_idx, obj_ids, mask_logits)
                    else:
                        print(f"WARNING: Unexpected tuple length: {len(item)}")
                        # Try to use the tuple elements we have
                        if len(item) >= 1:
                            frame_idx = item[0]
                            # Fill in other fields with defaults
                            obj_ids = objects_to_track if objects_to_track is not None else [1]
                            mask_logits = [torch.zeros((1, 1))] * len(obj_ids)  # Empty masks
                            if len(item) >= 2:
                                # Try to use the second element as mask_logits
                                mask_logits_element = item[1]
                                if isinstance(mask_logits_element, list):
                                    mask_logits = mask_logits_element
                                else:
                                    mask_logits = [mask_logits_element] * len(obj_ids)
                            
                            print(f"Constructed reasonable default 3-element tuple from unexpected tuple length {len(item)}")
                            yield (frame_idx, obj_ids, mask_logits)
            
            # Use our robust iterator wrapper
            frame_count = 0
            for out_frame_idx, out_obj_ids, out_mask_logits in robust_iterator_wrapper(iterator):
                frame_count += 1
                print(f"Processing item {frame_count}: frame={out_frame_idx}, objects={out_obj_ids}")
                
                # Filter objects if needed
                if objects_to_track is not None:
                    print(f"Filtering objects to track: {objects_to_track}")
                    indices = [i for i, obj_id in enumerate(out_obj_ids) if obj_id in objects_to_track]
                    if not indices:
                        print(f"No matching objects found, skipping frame {out_frame_idx}")
                        continue
                    filtered_obj_ids = [out_obj_ids[i] for i in indices]
                    filtered_mask_logits = [out_mask_logits[i] for i in indices]
                    print(f"After filtering: {len(filtered_obj_ids)} objects remain")
                else:
                    filtered_obj_ids = out_obj_ids
                    filtered_mask_logits = out_mask_logits
                    print(f"No filtering applied, using all {len(filtered_obj_ids)} objects")
                    
                video_segments[out_frame_idx] = {}
                boxes_by_frame[out_frame_idx] = {}
                
                # Process each object
                for i, (obj_id, mask_logit) in enumerate(zip(filtered_obj_ids, filtered_mask_logits)):
                    print(f"Processing object {obj_id} in frame {out_frame_idx}")
                    print(f"Mask logit type: {type(mask_logit).__name__}, shape: {getattr(mask_logit, 'shape', 'unknown')}")
                    
                    # Convert logits to binary mask (torch tensor)
                    if isinstance(mask_logit, torch.Tensor):
                        mask = (mask_logit.sigmoid() > 0.5)
                        print(f"Converted tensor mask, shape: {mask.shape}")
                    else:
                        print(f"Converting non-tensor mask type: {type(mask_logit).__name__}")
                        mask = (torch.from_numpy(mask_logit).sigmoid() > 0.5)
                        print(f"Converted numpy mask, shape: {mask.shape}")
                    
                    # Save to results
                    video_segments[out_frame_idx][obj_id] = mask.cpu().numpy()
                    
                    try:
                        # Use torchvision.ops.masks_to_boxes (expects (N, H, W))
                        print(f"Creating bounding box from mask shape: {mask.shape}")
                        expanded_mask = mask[None] if mask.ndim < 3 else mask
                        print(f"Expanded mask shape: {expanded_mask.shape}")
                        
                        box = torchvision.ops.masks_to_boxes(expanded_mask)[0].cpu().tolist()
                        print(f"Generated box: {box}")
                        
                        # Only save if the box is valid (non-zero area)
                        if box[0] < box[2] and box[1] < box[3]:
                            boxes_by_frame[out_frame_idx][obj_id] = {
                                "box": box,
                                "class": obj_id
                            }
                        else:
                            print(f"Warning: Empty or invalid box for object {obj_id} in frame {out_frame_idx}, box: {box}")
                    except Exception as box_error:
                        print(f"Error creating box for object {obj_id}: {box_error}")
                    
                print(f"Processed frame {out_frame_idx}, found {len(filtered_obj_ids)} objects")
            
            print(f"Finished propagation, processed {frame_count} frames, found {len(video_segments)} frames with objects")
            return video_segments, boxes_by_frame
        
        except Exception as e:
            print(f"Error during mask propagation: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # Since we don't want a fallback, just raise the exception
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
                # Ensure frame is a numpy array
                if isinstance(frame, torch.Tensor):
                    frame = self.tensor_to_numpy(frame)
                    # Squeeze if necessary
                    if frame.ndim > 3:
                        frame = np.squeeze(frame, axis=0)
            except:
                raise ValueError("Cannot retrieve frame - no frames directory or getter method available")
        
        # Create figure and axis
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # Show the frame
        ax.imshow(frame)
        ax.set_title(f"Frame {frame_idx}")
        
        # Show masks using the show_mask function
        if mask is not None:
            if obj_ids is None:
                # Single mask case
                self.show_mask(mask, ax)
            else:
                # Multiple masks case
                for i, (m, obj_id) in enumerate(zip(mask, obj_ids)):
                    self.show_mask(m, ax, obj_id=obj_id)
        
        # Show box
        if box is not None:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      edgecolor='green', facecolor=(0,0,0,0), lw=2))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save visualization if output directory provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, f"segment_frame_{frame_idx:04d}.jpg")
            plt.savefig(vis_path, bbox_inches='tight', dpi=150)
            print(f"Saved segmentation visualization to {vis_path}")
        
        plt.show()
        print('HELLO')
        #WHY

#         Adding box to frame 0...
# /home/ubuntu/code/drew/sam2/sam2/sam2_video_predictor.py:786: UserWarning: cannot import name '_C' from 'sam2' (/home/ubuntu/code/drew/sam2/sam2/__init__.py)

# Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
#   pred_masks_gpu = fill_holes_in_mask_scores(
# Mask shape: (540, 960)
# Mask dtype: bool
# Mask values: min=False, max=True
# Number of True pixels: 88393
# Traceback (most recent call last):
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/test_sam2_integration.py", line 41, in <module>
#     sam_wrapper.visualize_frame(
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/sam2_wrapper.py", line 258, in visualize_frame
#     self.show_mask(m, ax, obj_id=obj_id)
#   File "/home/ubuntu/code/drew/e2e_sam2/e2e_pipeline_v2/experiments/vidPredictor/src/sam2_wrapper.py", line 158, in show_mask
#     h, w = mask.shape
#     ^^^^
# ValueError: not enough values to unpack (expected 2, got 1)