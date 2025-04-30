# Standard library imports
import os
import json
import argparse
import logging
from pathlib import Path
import sys
import time

# Add root directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Local imports
from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType


# Configure logging
def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("segment_embeddings.log")
        ]
    )
    return logging.getLogger("segment_embeddings")


# Initialize logger
logger = setup_logging()


def get_device():
    """Determine the appropriate device for computation"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU for computation")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon). Note that SAM2 might give different outputs on MPS.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU. This may be slower than GPU acceleration.")
    
    logger.info(f"Selected device: {device}")
    return device


def initialize_sam2():
    """Initialize SAM2 model and related components"""
    logger.info("Building SAM2 model...")
    logger.info(f'Starting path: {os.getcwd()}')
    
    start_time = time.time()
    device = get_device()
    sam2 = build_sam2(
        config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_path="checkpoints/sam2.1_hiera_large.pt",
        device=device
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"SAM2 model built successfully in {elapsed_time:.2f} seconds")
    return sam2


def create_image_predictor(model):
    """Create and configure the image predictor"""
    logger.info("Creating SAM2 image predictor...")
    return SAM2ImagePredictor(model)


def create_mask_generator(model):
    """Create and configure the automatic mask generator"""
    logger.info("Creating automatic mask generator...")
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,  # Number of points to sample along each side of the image
        pred_iou_thresh=0.86,  # Threshold for predicted IoU to keep a mask
        stability_score_thresh=0.92,  # Threshold for mask stability score
        crop_n_layers=1,  # Number of image layers to use for mask prediction
        crop_n_points_downscale_factor=2,  # Downscale factor for points in crops
        min_mask_region_area=100  # Minimum area (in pixels) for a mask to be kept
    )


def load_image(image_path):
    """Load and prepare an image for processing"""
    logger.info(f"Loading image from {image_path}")
    
    if not os.path.exists(image_path):
        error_msg = f"Image file not found: {image_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    image = cv2.imread(image_path)
    if image is None:
        error_msg = f"Could not load image from {image_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info(f"Image loaded successfully: {image.shape}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def extract_segment(image, mask):
    """Extract a segment from the image based on the mask"""
    # Create a copy of the image
    segment = np.zeros_like(image)
    # Apply mask to extract the segment
    segment[mask] = image[mask]
    
    # Get bounding box to crop the segment
    indices = np.where(mask)
    if len(indices[0]) == 0:
        logger.warning("Empty mask detected, skipping segment extraction")
        return None
    
    y_min, y_max = indices[0].min(), indices[0].max()
    x_min, x_max = indices[1].min(), indices[1].max()
    
    # Crop the segment
    cropped_segment = segment[y_min:y_max+1, x_min:x_max+1]
    logger.debug(f"Segment extracted with dimensions: {cropped_segment.shape}")
    return cropped_segment


def save_mask_image(mask, output_path):
    """Save a binary mask as an image"""
    # Convert boolean mask to 8-bit
    mask_img = mask.astype(np.uint8) * 255
    cv2.imwrite(str(output_path), mask_img)
    return output_path


def visualize_results(image, masks, embeddings, output_dir):
    """Visualize the segmentation and embeddings"""
    logger.info("Generating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Display image with segmentation masks
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title("Segmentation Masks")
    
    # Overlay masks with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
    for i, mask in enumerate(masks):
        binary_mask = mask["segmentation"]
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw contours for each mask
        color = colors[i][:3]  # RGB
        for contour in contours:
            plt.fill(
                contour[:, 0, 0], 
                contour[:, 0, 1], 
                color=color,
                alpha=0.3
            )
            plt.plot(
                contour[:, 0, 0], 
                contour[:, 0, 1], 
                color=color,
                linewidth=2
            )
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    vis_path = output_dir / "segmentation_visualization.png"
    plt.savefig(vis_path)
    plt.close()
    
    logger.info(f"Visualization saved to {vis_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Segment images and generate embeddings")
    parser.add_argument("--image", type=str, required=True, 
                        help="Path to the input image")
    parser.add_argument("--output", type=str, default="segmentation_results",
                        help="Directory to save the results")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Maximum number of segments to process (default: process all)")
    parser.add_argument("--models", type=str, choices=["vit", "resnet50", "both"], default="both",
                        help="Which embedding models to use (default: both)")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Starting segment_gemini_outputs.py")
    
    try:
        args = parse_args()
        logger.info(f"Arguments: image={args.image}, output={args.output}, max_segments={args.max_segments}, models={args.models}")
        
        # Determine which embedding models to use
        model_types = []
        if args.models == "vit" or args.models == "both":
            model_types.append(ModelType.VIT)
        if args.models == "resnet50" or args.models == "both":
            model_types.append(ModelType.RESNET50)
        
        start_time = time.time()
        
        # Modify process_image to accept max_segments and model_types
        def modified_process_image():
            # Create output directories
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Create specific folders for each type of output
            masks_dir = output_dir / "masks"
            segments_dir = output_dir / "segments"
            
            masks_dir.mkdir(exist_ok=True)
            segments_dir.mkdir(exist_ok=True)
            
            logger.info(f"Created output directories in {output_dir}")
            
            # Initialize components
            sam2 = initialize_sam2()
            mask_generator = create_mask_generator(sam2)
            
            # Initialize embedding generator
            logger.info(f"Initializing embedding generator with models: {[m.value for m in model_types]}")
            embedding_generator = EmbeddingGenerator(
                model_types=model_types,
                device=get_device().type
            )
            
            # Load image
            image = load_image(args.image)
            
            # Generate masks automatically
            logger.info("Generating masks...")
            start_time = time.time()
            masks = mask_generator.generate(image)
            elapsed_time = time.time() - start_time
            
            # Save all masks immediately for debugging
            logger.info("Saving all detected masks for debugging...")
            all_masks_dir = output_dir / "all_masks"
            all_masks_dir.mkdir(exist_ok=True)
            
            # Create a combined mask visualization
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.title(f"All Detected Masks ({len(masks)})")
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))
            for i, mask_data in enumerate(masks):
                # Save individual mask
                mask = mask_data["segmentation"]
                mask_path = all_masks_dir / f"mask_{i}.png"
                save_mask_image(mask, mask_path)
                
                # Add to visualization
                binary_mask = mask
                contours, _ = cv2.findContours(
                    binary_mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Draw contours for each mask
                color = colors[i][:3]  # RGB
                for contour in contours:
                    plt.fill(
                        contour[:, 0, 0], 
                        contour[:, 0, 1], 
                        color=color,
                        alpha=0.3
                    )
                    plt.plot(
                        contour[:, 0, 0], 
                        contour[:, 0, 1], 
                        color=color,
                        linewidth=2
                    )
            
            # Save the combined visualization
            plt.axis('off')
            plt.tight_layout()
            all_masks_vis_path = all_masks_dir / "all_masks_visualization.png"
            plt.savefig(all_masks_vis_path)
            plt.close()
            
            logger.info(f"Saved all {len(masks)} masks to {all_masks_dir}")
            logger.info(f"Saved visualization of all masks to {all_masks_vis_path}")
            
            # Limit number of masks if specified
            if args.max_segments is not None and len(masks) > args.max_segments:
                logger.info(f"Limiting processing to {args.max_segments} of {len(masks)} masks")
                masks = masks[:args.max_segments]
            else:
                logger.info(f"Generated {len(masks)} masks in {elapsed_time:.2f} seconds")
            
            # Process each mask and generate embeddings
            segments_info = []
            
            logger.info("Processing segments and generating embeddings...")
            for i, mask_data in enumerate(masks):
                logger.info(f"Processing segment {i+1}/{len(masks)} ({(i+1)/len(masks)*100:.1f}%)")
                
                # Extract mask and segment
                mask = mask_data["segmentation"]
                
                # Save the binary mask
                mask_path = masks_dir / f"mask_{i}.png"
                save_mask_image(mask, mask_path)
                logger.info(f"Saved mask to {mask_path}")
                
                # Extract the segment
                segment = extract_segment(image, mask)
                
                if segment is None:
                    logger.warning(f"Skipping empty segment {i}")
                    continue
                
                # Save segment
                segment_path = segments_dir / f"segment_{i}.png"
                cv2.imwrite(str(segment_path), cv2.cvtColor(segment, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved segment to {segment_path}")
                
                # Generate embeddings
                logger.info(f"Generating embeddings for segment {i}...")
                
                # Initialize dictionary for embeddings
                embedding_data = {
                    "segment_id": i,
                    "segment_path": str(segment_path),
                    "mask_path": str(mask_path),
                    "embeddings": {},
                    "mask_data": {
                        "area": float(mask_data["area"]),
                        "bbox": mask_data["bbox"],
                        "predicted_iou": float(mask_data["predicted_iou"]),
                        "stability_score": float(mask_data["stability_score"])
                    }
                }
                
                # Generate embeddings for selected models
                try:
                    for model_type in model_types:
                        model_name = model_type.value
                        logger.info(f"Generating {model_name} embedding...")
                        embedding = embedding_generator.generate_embedding(segment, model_type)
                        logger.info(f"Generated {model_name} embedding: {embedding.shape}")
                        embedding_data["embeddings"][model_name] = embedding.tolist()
                    
                    segments_info.append(embedding_data)
                except Exception as e:
                    logger.error(f"Error generating embeddings for segment {i}: {str(e)}")
            
            # Save all segments info
            results_path = output_dir / "segments_embeddings.json"
            with open(results_path, "w") as f:
                json.dump({
                    "image_path": args.image,
                    "num_segments": len(segments_info),
                    "segments": segments_info
                }, f, indent=2)
            
            logger.info(f"Saved results data to {results_path}")
            
            # Visualize results
            visualize_results(image, masks, segments_info, output_dir)
            
            logger.info(f"Processing complete. Results saved to {output_dir}")
            
            # Return summary
            return {
                "num_masks": len(masks),
                "num_processed_segments": len(segments_info),
                "output_directory": str(output_dir)
            }
        
        results = modified_process_image()
        elapsed_time = time.time() - start_time
        
        logger.info(f"Processed {results['num_masks']} masks, {results['num_processed_segments']} segments")
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to: {results['output_directory']}")
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)