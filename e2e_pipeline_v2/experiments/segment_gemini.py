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


image = Image.open('../test_data/objects/Scenes 001-020__101B-1-_20230726152900590/0_ornate circular mirror_0.png')
image = np.array(image.convert("RGB"))


# /home/ubuntu/code/drew/test_data/objects/Scenes 001-020__101B-1-_20230726152900590/0_ornate circular mirror_0.png

masks = mask_generator.generate(image)



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

def generate_embeddings_for_folder(
    input_folder, 
    output_folder, 
    models=["vit", "resnet50"], 
    device=None,
    extensions=['.jpg', '.jpeg', '.png']
):
    """
    Generate embeddings for all images in a folder.
    
    Args:
        input_folder (str): Path to folder containing images
        output_folder (str): Path to save embeddings
        models (list): List of models to use ("vit", "resnet50", or "clip")
        device (str): Device to use (cuda, cpu, or None for auto-detect)
        extensions (list): List of image file extensions to process
    
    Returns:
        dict: Summary of processed files and their embeddings
    """
    from pathlib import Path
    import os
    import json
    import torch
    import numpy as np
    from tqdm import tqdm
    
    from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType
    
    # Create Path objects
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    
    # Check if input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory '{input_dir}' does not exist or is not a directory.")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"No images found in {input_dir} with extensions: {', '.join(extensions)}")
        return {}
    
    print(f"Found {len(image_files)} images to process")
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_types=models,
        device=device
    )
    
    # Process each image
    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "models": models,
        "processed_files": []
    }
    
    for image_file in tqdm(image_files, desc="Generating embeddings"):
        try:
            # Generate output filename
            output_path = output_dir / f"{image_file.stem}_embeddings.json"
            
            # Generate embeddings for each model
            results = {}
            for model_type in models:
                embedding = embedding_generator.generate_embedding(
                    image=str(image_file),
                    model_type=model_type
                )
                
                # Convert to list for JSON serialization
                results[model_type] = embedding.tolist()
            
            # Save embeddings
            with open(output_path, 'w') as f:
                json.dump({
                    "image_path": str(image_file),
                    "embeddings": results
                }, f, indent=2)
            
            # Add to summary
            summary["processed_files"].append({
                "image": str(image_file),
                "embedding_file": str(output_path)
            })
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    
    # Save summary
    summary_path = output_dir / "embedding_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Processed {len(summary['processed_files'])} images")
    print(f"Summary saved to: {summary_path}")
    
    return summary

generate_embeddings_for_folder(
    input_folder="mask_outputs",
    output_folder="mask_outputs/embeddings",
    models=["vit", "resnet50"],
    device=device
)
