# test_sam2_integration.py
from src.frame_loader import load_frames_from_directory
from src.sam2_wrapper import SAM2VideoWrapper
import os
import torch
import numpy as np

# Create output directory
output_dir = "./sam2_results"
os.makedirs(output_dir, exist_ok=True)

# Load frames
frames, _ = load_frames_from_directory("/home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset/")
# Initialize SAM2
sam_wrapper = SAM2VideoWrapper(
    checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
    config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
)

# Set video
sam_wrapper.set_video(frames_dir='/home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset/')

#TODO: IDEA
# write test that only calls sam2 reset/load states 

# Test segmentation with a sample box
frame_idx = 0
box = [250, 69, 773, 474]  # Example box

# Add box and get mask
print("Adding box to frame 0...")
mask = sam_wrapper.add_box(frame_idx, obj_id=1, box=box)

# Debug output to check what mask we got
if mask is not None:
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Mask values: min={mask.min()}, max={mask.max()}")
    print(f"Number of True pixels: {np.sum(mask)}")
else:
    print("No mask returned!")

# Visualize and save - pass the single mask directly, not in a list
sam_wrapper.visualize_frame(
    frame_idx, 
    mask=mask,  # Changed from [mask] to just mask 
    obj_ids=None,  # Changed from [1] to None since we're passing a single mask
    box=box,
    output_dir=output_dir
)

# Test propagation
print("Propagating masks...")
segments = sam_wrapper.propagate_masks(objects_to_track=[1])
print(f'Saving results to {output_dir}')

print('Visualizing propagated frames...')
# Visualize a few propagated frames
for idx in range(0, min(len(frames), 30), 1):
    if idx in segments:
        print(f"Visualizing frame {idx}")
        # The segments dictionary maps frame_idx -> {obj_id -> mask}
        frame_segments = segments[idx]
        
        # Debug print to check the mask shapes
        for obj_id, mask in frame_segments.items():
            print(f"  Object {obj_id} mask shape: {mask.shape}")
            print(f"  Object {obj_id} mask type: {mask.dtype}")
            print(f"  Object {obj_id} number of True pixels: {np.sum(mask)}")
        
        # For a single object, just pass the mask directly
        if len(frame_segments) == 1:
            obj_id = list(frame_segments.keys())[0]
            mask = frame_segments[obj_id]
            
            # Visualize with a single mask
            sam_wrapper.visualize_frame(
                idx,
                mask=mask,
                obj_ids=None,
                output_dir=output_dir
            )
        else:
            # For multiple objects, pass lists
            obj_ids = list(frame_segments.keys())
            masks = [frame_segments[obj_id] for obj_id in obj_ids]
            
            # Visualize the frame with all masks
            sam_wrapper.visualize_frame(
                idx,
                mask=masks,
                obj_ids=obj_ids,
                output_dir=output_dir
            )
    else:
        print(f"No segments found for frame {idx}")

