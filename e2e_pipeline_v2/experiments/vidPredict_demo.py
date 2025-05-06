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


def show_mask(mask, ax, obj_id=None, random_color=False):
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


inference_state = predictor.init_state(video_path=video_dir)


predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([250, 69, 773, 474], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# Save the figure
plt.savefig(os.path.join(results_dir, f"frame_{ann_frame_idx}_box_mask.png"))

# Let's also save the raw mask as a separate file
binary_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
plt.figure(figsize=(9, 6))
plt.title(f"Binary mask for frame {ann_frame_idx}")
plt.imshow(binary_mask, cmap='gray')
plt.savefig(os.path.join(results_dir, f"frame_{ann_frame_idx}_mask_only.png"))

# Add propagation to other frames and save results
num_frames_to_process = min(10, len(frame_names))  # Process first 10 frames or all if less

for frame_idx in range(num_frames_to_process):
    if frame_idx == ann_frame_idx:
        continue  # Skip the initial frame we already processed
    
    # Propagate to this frame
    obj_ids, mask_logits = predictor.propagate_in_video(
        inference_state=inference_state,
        frame_idx=frame_idx,
    )
    
    # Create visualization
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {frame_idx} - propagated mask")
    img = Image.open(os.path.join(video_dir, frame_names[frame_idx]))
    plt.imshow(img)
    
    # Find the index of our object ID in the returned results
    if len(obj_ids) > 0:
        for i, obj_id in enumerate(obj_ids):
            if obj_id == ann_obj_id:
                mask = (mask_logits[i] > 0.0).cpu().numpy()
                show_mask(mask, plt.gca(), obj_id=obj_id)
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, f"frame_{frame_idx}_propagated.png"))
    plt.close()  # Close figure to avoid memory issues

print(f"All results saved to: {results_dir}")