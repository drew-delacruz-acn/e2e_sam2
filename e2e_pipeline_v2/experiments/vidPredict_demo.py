import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def tensor_to_numpy(tensor):
    """Safely convert tensor to numpy array, handling device and gradient issues."""
    if isinstance(tensor, torch.Tensor):
        # Move to CPU if on another device and detach from computation graph
        return tensor.detach().cpu().numpy().squeeze(0) if tensor.ndim > 0 else tensor.detach().cpu().numpy()
    return tensor  # Already a numpy array or other format

def show_mask(mask, ax, obj_id=None, random_color=False):
    # Ensure mask is a numpy array
    if isinstance(mask, torch.Tensor):
        mask = tensor_to_numpy(mask)
        
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# Create a results directory to save visualizations
results_dir = "./sam2_results"
os.makedirs(results_dir, exist_ok=True)
print(f"Saving results to: {results_dir}")

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = '/home/ubuntu/code/drew/test_data/frames/Scenes 001-020__220D-2-_20230815190723523/subset/'

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
# Save the figure
plt.savefig(os.path.join(results_dir, f"frame_{frame_idx}_original.png"))
plt.close()

# Initialize the video inference state
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# Define the frame we want to annotate and the object ID
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max)
box = np.array([250, 69, 773, 474], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# Show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
# Use the safer conversion function
mask_np = tensor_to_numpy(out_mask_logits[0]) > 0.0
show_mask(mask_np, plt.gca(), obj_id=out_obj_ids[0])
# Save the figure
plt.savefig(os.path.join(results_dir, f"frame_{ann_frame_idx}_box_mask.png"))
print(f'Saving to {os.path.join(results_dir, f"frame_{ann_frame_idx}_box_mask.png")}')

# Save the raw mask as a separate file
binary_mask = tensor_to_numpy(out_mask_logits[0]) > 0.0
print(f"Binary mask shape: {binary_mask.shape}")
plt.figure(figsize=(9, 6))
plt.title(f"Binary mask for frame {ann_frame_idx}")
plt.imshow(binary_mask, cmap='gray')
plt.savefig(os.path.join(results_dir, f"frame_{ann_frame_idx}_mask_only.png"))
plt.close()

# ----- NOTEBOOK-STYLE PROPAGATION -----
print("Running video propagation like the notebook...")

# Run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    # Convert masks to numpy and store by object ID
    video_segments[out_frame_idx] = {
        out_obj_id: (tensor_to_numpy(out_mask_logits[i]) > 0.0)
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    print(f"Processed frame {out_frame_idx}, found {len(out_obj_ids)} objects")

# Define how many frames to process
num_frames_to_process = min(10, len(frame_names))  # Process first 10 frames or all if less
vis_frame_stride = 1  # Process every frame (set higher for skipping frames)

# Render the segmentation results
plt.close("all")
for out_frame_idx in range(0, num_frames_to_process, vis_frame_stride):
    if out_frame_idx == ann_frame_idx:
        continue  # Skip the annotated frame we already processed
        
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    
    # Show masks for this frame if available
    if out_frame_idx in video_segments:
        masks_for_frame = video_segments[out_frame_idx]
        for out_obj_id, out_mask in masks_for_frame.items():
            if out_obj_id == ann_obj_id:  # Only show our object of interest
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
                print(f"Rendering mask for object {out_obj_id} in frame {out_frame_idx}")
    else:
        print(f"No masks found for frame {out_frame_idx}")
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, f"frame_{out_frame_idx}_propagated.png"))
    plt.close()  # Close figure to avoid memory issues

print(f"All results saved to: {results_dir}")