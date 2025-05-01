# Standard library imports
import os
import json
import argparse
import logging
import traceback
from pathlib import Path
import sys
import time
import gc  # For garbage collection
from PIL import Image

# Add root directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

###########################################
# LOGGING CONFIGURATION
###########################################

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


sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)


np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


image = Image.open('data/mjolnir_gt.jpeg')
image = np.array(image.convert("RGB"))


masks = mask_generator.generate(image)

print(masks)


# Create output directory if it doesn't exist
output_dir = "mask_outputs"
os.makedirs(output_dir, exist_ok=True)


# Loop through each mask
for i, mask_data in enumerate(masks):
    # Get the binary mask
    binary_mask = mask_data['segmentation']
    
    # Make sure the mask is in the correct format
    binary_mask = binary_mask.astype(bool)
    
    # Create a copy of the original image
    segment_original = np.zeros_like(image)
    
    # Copy the original pixels where the mask is True
    segment_original[~binary_mask] = image[~binary_mask]
    
    # Save with proper color conversion
    cv2.imwrite(f"{output_dir}/segment_{i}_color.png", cv2.cvtColor(segment_original, cv2.COLOR_RGB2BGR))
     # For debugging: save the mask itself to check if it's correct
    mask_image = binary_mask.astype(np.uint8) * 255
    cv2.imwrite(f"{output_dir}/binary_mask_{i}.png", mask_image)