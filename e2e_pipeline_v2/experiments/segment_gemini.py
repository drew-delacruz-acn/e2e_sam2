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
from PIL import Image, ImageOps

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
    
    # Create mutually exclusive group for input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=str,
                      help="Directory containing images to segment and embed")
    input_group.add_argument("--image", type=str, 
                      help="Path to a single image to segment and embed")
    
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

def apply_image_tranforms(image_path, output_path):
    """
    Resize and pad image to 224x224 while maintaining aspect ratio.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the padded image
        
    Returns:
        The padded PIL Image
    """
    # STEP 1 - APPLY PADDING
    image = Image.open(image_path).convert("RGB")
    target_width = target_height = 224
    width, height = image.size
    
    if width == target_width and height == target_height:
        resized_padded_image = image
    else:
        if width >= height:
            scale = target_width / width
        else:
            scale = target_height / height

        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        pad_left = max((target_width - new_width) // 2, 0)
        pad_right = target_width - new_width - pad_left
        pad_top = max((target_height - new_height) // 2, 0)
        pad_bottom = target_height - new_height - pad_top
        
        resized_padded_image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill='grey')
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    resized_padded_image.save(output_path)
    return resized_padded_image

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

def create_segmentation_visualization(image, masks, output_path):
    """
    Create a visualization of all segments overlaid on the original image.
    
    Args:
        image: Original image as numpy array
        masks: List of mask dictionaries from SAM2
        output_path: Path to save the visualization
    """
    # Create a copy of the original image
    vis_image = image.copy()
    
    # Create an overlay image for the segments
    overlay = np.zeros_like(vis_image, dtype=np.uint8)
    
    # Generate random colors for each mask
    colors = []
    np.random.seed(42)  # For consistent colors
    for _ in range(len(masks)):
        # Generate bright, distinct colors
        color = np.random.randint(100, 255, size=3).tolist()
        colors.append(color)
    
    # Draw each mask with a different color
    for i, mask_data in enumerate(masks):
        binary_mask = mask_data['segmentation']
        color = colors[i]
        
        # Color the mask area
        colored_mask = np.zeros_like(vis_image, dtype=np.uint8)
        colored_mask[binary_mask] = color
        
        # Add to the overlay with transparency
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
        
        # Draw the contour
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color, 2)
    
    # Combine the original image and the overlay
    result = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
    
    # Add a legend showing segment numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    padding = 10
    
    # Find centroids of each mask and add segment number
    for i, mask_data in enumerate(masks):
        binary_mask = mask_data['segmentation']
        
        # Find the centroid
        moments = cv2.moments(binary_mask.astype(np.uint8))
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            
            # Draw segment number at centroid
            cv2.putText(result, str(i), (cX, cY), font, font_scale, 
                      (255, 255, 255), font_thickness + 1, cv2.LINE_AA)  # Shadow
            cv2.putText(result, str(i), (cX, cY), font, font_scale, 
                      colors[i], font_thickness, cv2.LINE_AA)
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    logger.info(f"Segmentation visualization saved to {output_path}")

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
        Tuple of (list of segment paths, path to debug visualization, image_dir)
    """
    try:
        # Get base filename without extension
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        
        # Create organized directory structure
        image_dir = os.path.join(output_dir, image_name)
        segments_dir = os.path.join(image_dir, "segments")
        
        # Create directories
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(segments_dir, exist_ok=True)
        
        # Load and process the image
        image = Image.open(image_path)
        image_np = np.array(image.convert("RGB"))

        # Generate masks
        masks = mask_generator.generate(image_np)

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
            segment_image = np.zeros_like(image_np)
            
            # Copy the original pixels where the mask is True
            segment_image[binary_mask] = image_np[binary_mask]
            
            # Create output path with the requested naming format
            output_path = os.path.join(segments_dir, f"{image_name}_{i}.png")
            
            # Save with proper color conversion
            cv2.imwrite(output_path, cv2.cvtColor(segment_image, cv2.COLOR_RGB2BGR))
            
            # Add to segment paths
            segment_paths.append(output_path)
        
        # Create and save a visualization with all segments overlaid on the original image
        debug_vis_path = os.path.join(image_dir, f"{image_name}_segmentation_overlay.png")
        create_segmentation_visualization(image_np, filtered_masks, debug_vis_path)
        
        return segment_paths, debug_vis_path, image_dir
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return [], None, None

def generate_embeddings_for_segments(
    segment_paths, 
    embedding_generator, 
    image_dir, 
    original_image
):
    """
    Generate embeddings for all segments from a single image and save to one JSON.
    
    Args:
        segment_paths: List of paths to segment images
        embedding_generator: EmbeddingGenerator instance
        image_dir: Root directory for this image's outputs
        original_image: Path to the original image
        
    Returns:
        Path to the generated JSON file
    """
    try:
        # Get the original image basename
        original_basename = os.path.basename(original_image)
        original_name = os.path.splitext(original_basename)[0]
        
        # Create organized directory structure
        padded_dir = os.path.join(image_dir, "padded_segments")
        embeddings_dir = os.path.join(image_dir, "embeddings")
        
        # Create directories
        os.makedirs(padded_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)
        
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
            
            # Apply padding to create a properly sized image for embedding
            padded_path = os.path.join(padded_dir, f"padded_{segment_basename}")
            apply_image_tranforms(segment_path, padded_path)
            
            # Generate embeddings for each model
            segment_embeddings = {}
            for model_type in embedding_generator.model_types:
                try:
                    # Use the padded image for embedding generation
                    embedding = embedding_generator.generate_embedding(
                        image=padded_path, 
                        model_type=model_type.value
                    )
                    segment_embeddings[model_type.value] = embedding.tolist()
                except Exception as e:
                    logger.warning(f"Error generating {model_type.value} embedding for {segment_path}: {str(e)}")
            
            # Add to embeddings data
            embeddings_data["segments"][segment_id] = {
                "path": segment_path,  # Keep the original path in the metadata
                "padded_path": padded_path,  # Add the padded path for reference
                "embeddings": segment_embeddings
            }
        
        # Create output path
        output_path = os.path.join(embeddings_dir, f"{original_name}_embeddings.json")
        
        # Save embeddings
        with open(output_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logger.info(f"Embeddings saved to {output_path}")
        logger.info(f"Padded images saved to {padded_dir} for inspection")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return None

def main():
    args = parse_args()
    
    # Setup device
    device = get_device()
    
    # Create output Path object
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SAM2
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    logger.info("Loading SAM2 model...")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        min_mask_region_area=10.0,
    )
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_types=args.models,
        device=device
    )
    
    # Prepare summary
    summary = {
        "output_directory": str(output_dir),
        "processed_images": []
    }
    
    # Determine processing mode - single image or directory
    if args.image:
        # Single image mode
        image_file = Path(args.image)
        if not image_file.exists():
            logger.error(f"Image file '{image_file}' does not exist.")
            return 1
            
        logger.info(f"Processing single image: {image_file}")
        summary["mode"] = "single_image"
        summary["input_image"] = str(image_file)
        
        # Process the single image
        try:
            # Generate segments
            segment_paths, debug_vis_path, image_dir = process_image_and_generate_segments(
                image_path=str(image_file),
                mask_generator=mask_generator,
                output_dir=output_dir,
                min_area=args.min_area
            )
            
            if not segment_paths or not image_dir:
                logger.warning(f"No valid segments found for {image_file}")
                return 1
            
            # Generate embeddings for all segments
            embeddings_path = generate_embeddings_for_segments(
                segment_paths=segment_paths,
                embedding_generator=embedding_generator,
                image_dir=image_dir,
                original_image=str(image_file)
            )
            
            # Add to summary
            summary["processed_images"].append({
                "image": str(image_file),
                "image_dir": image_dir,
                "segments_count": len(segment_paths),
                "segments": segment_paths,
                "segmentation_overlay": debug_vis_path,
                "embeddings_file": embeddings_path
            })
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")
            return 1
            
    else:
        # Directory mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            logger.error(f"Input directory '{input_dir}' does not exist or is not a directory.")
            return 1
        
        summary["mode"] = "directory"
        summary["input_directory"] = str(input_dir)
        
        # Find all image files
        image_files = []
        for ext in args.extensions:
            image_files.extend(list(input_dir.glob(f"*{ext}")))
            image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            logger.error(f"No images found in {input_dir} with extensions: {', '.join(args.extensions)}")
            return 1
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image in the directory
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                logger.info(f"Processing {image_file}")
                
                # Generate segments
                segment_paths, debug_vis_path, image_dir = process_image_and_generate_segments(
                    image_path=str(image_file),
                    mask_generator=mask_generator,
                    output_dir=output_dir,
                    min_area=args.min_area
                )
                
                if not segment_paths or not image_dir:
                    logger.warning(f"No valid segments found for {image_file}")
                    continue
                
                # Generate embeddings for all segments
                embeddings_path = generate_embeddings_for_segments(
                    segment_paths=segment_paths,
                    embedding_generator=embedding_generator,
                    image_dir=image_dir,
                    original_image=str(image_file)
                )
                
                # Add to summary
                summary["processed_images"].append({
                    "image": str(image_file),
                    "image_dir": image_dir,
                    "segments_count": len(segment_paths),
                    "segments": segment_paths,
                    "segmentation_overlay": debug_vis_path,
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
    # Example commands:
    # Process directory: python segment_gemini.py --input_dir /path/to/directory --output_dir results
    # Process single image: python segment_gemini.py --image /path/to/image.jpg --output_dir results
    sys.exit(main())