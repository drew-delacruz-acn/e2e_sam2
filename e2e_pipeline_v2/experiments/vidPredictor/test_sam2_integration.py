# test_sam2_integration.py
from src.frame_loader import load_frames_from_directory
from src.sam2_wrapper import SAM2VideoWrapper
import os

# Create output directory
output_dir = "./sam2_results"
os.makedirs(output_dir, exist_ok=True)

# Load frames
frames, _ = load_frames_from_directory("./video_frames")

# Initialize SAM2
sam_wrapper = SAM2VideoWrapper(
    checkpoint_path="../checkpoints/sam2.1_hiera_large.pt",
    config_path="configs/sam2.1/sam2.1_hiera_l.yaml"
)

# Set video
sam_wrapper.set_video(frames)

# Test segmentation with a sample box
frame_idx = 0
box = [100, 100, 300, 300]  # Example box

# Add box and get mask
mask = sam_wrapper.add_box(frame_idx, obj_id=1, box=box)

# Visualize and save
sam_wrapper.visualize_frame(
    frame_idx, 
    mask=mask, 
    obj_ids=[1], 
    box=box,
    output_dir=output_dir
)

# Test propagation
print("Propagating masks...")
segments = sam_wrapper.propagate_masks(objects_to_track=[1])
print(f"Propagated to {len(segments)} frames")

# Visualize a few propagated frames
for idx in range(0, min(len(frames), 30), 10):
    if idx in segments:
        masks = segments[idx]["masks"]
        obj_ids = segments[idx]["obj_ids"]
        sam_wrapper.visualize_frame(
            idx,
            mask=masks,
            obj_ids=obj_ids,
            output_dir=output_dir
        )