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
from tqdm import tqdm

from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('segment_and_embed')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Segment images and generate embeddings")
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Directory containing images to segment and embed")
    parser.add_argument("--output_dir", type=str, default="output_results",
                      help="Directory to save results")
    parser.add_argument("--models", type=str, nargs="+", default=["vit", "resnet50"],
                      choices=["clip", "vit", "resnet50"],
                      help="Embedding models to use")
    parser.add_argument("--min_area", type=int, default=1000,
                      help="Minimum area (in pixels) for segments to consider")
    parser.add_argument("--extensions", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"],
                      help="Image file extensions to process")
    return parser.parse_args()

def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        logger.info(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS."
        )
        
    return device

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
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def process_image_and_generate_segments(
    image_path, 
    mask_generator, 
    output_dir, 
    min_area=1000
):
    """
    Process an image with SAM2, save segments, and return segment paths.
    
    Args:
        image_path: Path to the input image
        mask_generator: SAM2 mask generator
        output_dir: Directory to save segments
        min_area: Minimum area for segments to be considered
        
    Returns:
        List of paths to saved segment images
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        
        # Load and process the image
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        
        # Generate masks
        masks = mask_generator.generate(image)
        
        # Filter masks by area
        filtered_masks = [mask for mask in masks if mask['area'] > min_area]
        
        logger.info(f"Generated {len(filtered_masks)} masks for {image_basename}")
        
        # List to store segment paths
        segment_paths = []
        
        # Process and save each segment
        for i, mask_data in enumerate(filtered_masks):
            # Get the binary mask
            binary_mask = mask_data['segmentation']
            binary_mask = binary_mask.astype(bool)
            
            # Create a copy of the original image
            segment_image = np.zeros_like(image)
            
            # Copy the original pixels where the mask is True
            segment_image[binary_mask] = image[binary_mask]
            
            # Create output path with the requested naming format
            output_path = os.path.join(output_dir, f"{image_name}_{i}.png")
            
            # Save with proper color conversion
            cv2.imwrite(output_path, cv2.cvtColor(segment_image, cv2.COLOR_RGB2BGR))
            
            # Add to segment paths
            segment_paths.append(output_path)
            
            # Also save the binary mask for reference
            mask_path = os.path.join(output_dir, f"{image_name}_{i}_mask.png")
            cv2.imwrite(mask_path, binary_mask.astype(np.uint8) * 255)
        
        return segment_paths
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return []

def generate_embeddings_for_segments(
    segment_paths, 
    embedding_generator, 
    output_dir, 
    original_image
):
    """
    Generate embeddings for all segments from a single image and save to one JSON.
    
    Args:
        segment_paths: List of paths to segment images
        embedding_generator: EmbeddingGenerator instance
        output_dir: Directory to save embeddings
        original_image: Path to the original image
        
    Returns:
        Path to the generated JSON file
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the original image basename
        original_basename = os.path.basename(original_image)
        original_name = os.path.splitext(original_basename)[0]
        
        # Create embeddings dict
        embeddings_data = {
            "original_image": str(original_image),
            "segments": {}
        }
        
        # Process each segment
        for segment_path in segment_paths:
            # Get segment ID from filename
            segment_basename = os.path.basename(segment_path)
            segment_id = os.path.splitext(segment_basename)[0]  # Remove extension
            
            # Generate embeddings for each model
            segment_embeddings = {}
            for model_type in embedding_generator.model_types:
                try:
                    embedding = embedding_generator.generate_embedding(
                        image=segment_path, 
                        model_type=model_type.value
                    )
                    segment_embeddings[model_type.value] = embedding.tolist()
                except Exception as e:
                    logger.warning(f"Error generating {model_type.value} embedding for {segment_path}: {str(e)}")
            
            # Add to embeddings data
            embeddings_data["segments"][segment_id] = {
                "path": segment_path,
                "embeddings": segment_embeddings
            }
        
        # Create output path
        output_path = os.path.join(output_dir, f"{original_name}_embeddings.json")
        
        # Save embeddings
        with open(output_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return None

def main():
    args = parse_args()
    
    # Setup device
    device = get_device()
    
    # Create input/output Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    segments_dir = output_dir / "segments"
    embeddings_dir = output_dir / "embeddings"
    
    # Check if input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
        return 1
    
    # Create output directories
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in args.extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    if not image_files:
        logger.error(f"No images found in {input_dir} with extensions: {', '.join(args.extensions)}")
        return 1
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Initialize SAM2
    sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    logger.info("Loading SAM2 model...")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_types=args.models,
        device=device
    )
    
    # Process each image
    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "processed_images": []
    }
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            logger.info(f"Processing {image_file}")
            
            # Step 1: Generate segments
            segment_paths = process_image_and_generate_segments(
                image_path=str(image_file),
                mask_generator=mask_generator,
                output_dir=segments_dir,
                min_area=args.min_area
            )
            
            if not segment_paths:
                logger.warning(f"No valid segments found for {image_file}")
                continue
            
            # Step 2: Generate embeddings for all segments
            embeddings_path = generate_embeddings_for_segments(
                segment_paths=segment_paths,
                embedding_generator=embedding_generator,
                output_dir=embeddings_dir,
                original_image=str(image_file)
            )
            
            # Add to summary
            summary["processed_images"].append({
                "image": str(image_file),
                "segments_count": len(segment_paths),
                "segments": segment_paths,
                "embeddings_file": embeddings_path
            })
            
            # Clean up to avoid memory leaks
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")
    
    # Save summary
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Processed {len(summary['processed_images'])} images")
    logger.info(f"Summary saved to: {summary_path}")
    
    return 0

if __name__ == "__main__":

    #python segment_gemini.py --input_dir /home/ubuntu/code/drew/test_data/objects/Scenes\ 001-020__101B-1-_20230726152900590/ --output_dir results
    sys.exit(main())
