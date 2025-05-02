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
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
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
    
    # Create mutually exclusive group for segmentation modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--use_full_image_predictor", action="store_true",
                      help="Use SAM2 image predictor with full image bounds")
    mode_group.add_argument("--use_bbox_json", type=str, metavar="JSON_PATH",
                      help="Use SAM2 image predictor with bounding boxes from JSON file")
    mode_group.add_argument("--detections_dir", type=str,
                      help="Directory containing detection JSONs corresponding to images")
    
    # Option to use foreground points
    parser.add_argument("--use_points", action="store_true", default=True,
                      help="Use foreground points at bbox midpoints (default: True)")
    parser.add_argument("--no_points", action="store_false", dest="use_points",
                      help="Don't use foreground points with bbox")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug-level logging")
    
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
    logger.debug(f"Creating segmentation visualization with {len(masks)} masks")
    
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
    
    logger.debug(f"Generated {len(colors)} colors for mask visualization")
    
    # Draw each mask with a different color
    for i, mask_data in enumerate(masks):
        binary_mask = mask_data['segmentation']
        color = colors[i]
        
        # Get metadata if available 
        label = mask_data.get('label', f"Segment {i}")
        score = mask_data.get('predicted_iou', 0.0)
        area = mask_data.get('area', np.sum(binary_mask))
        
        logger.debug(f"Processing mask {i}: label='{label}', score={score:.4f}, area={area}")
        
        # Color the mask area
        colored_mask = np.zeros_like(vis_image, dtype=np.uint8)
        mask_bool = binary_mask.astype(bool)
        colored_mask[mask_bool] = color
        
        # Add to the overlay with transparency
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
        
        # Draw the contour
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color, 2)
        logger.debug(f"Drew mask {i} with {len(contours)} contours")
    
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
        
        # Try to get label if available
        label = mask_data.get('label', f"{i}")
        
        # Find the centroid
        moments = cv2.moments(binary_mask.astype(np.uint8))
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            
            # Draw segment number at centroid
            cv2.putText(result, label, (cX, cY), font, font_scale, 
                      (255, 255, 255), font_thickness + 1, cv2.LINE_AA)  # Shadow
            cv2.putText(result, label, (cX, cY), font, font_scale, 
                      colors[i], font_thickness, cv2.LINE_AA)
            logger.debug(f"Added label '{label}' for mask {i} at centroid ({cX}, {cY})")
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    logger.info(f"Segmentation visualization saved to {output_path}")

def process_image_and_generate_segments(
    image_path, 
    mask_generator, 
    output_dir, 
    min_area=10000
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

def load_bboxes_from_json(json_path):
    """
    Load bounding boxes from a JSON file.
    
    Expected format:
    {
        "image_name1": [x1, y1, x2, y2],
        "image_name2": [x1, y1, x2, y2],
        ...
    }
    
    Or for multiple boxes per image:
    {
        "image_name1": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
        "image_name2": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
        ...
    }
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary mapping image names to bounding boxes
    """
    try:
        with open(json_path, 'r') as f:
            bbox_data = json.load(f)
        
        logger.info(f"Loaded bounding box data for {len(bbox_data)} images from {json_path}")
        return bbox_data
    except Exception as e:
        logger.error(f"Error loading bounding box JSON from {json_path}: {str(e)}")
        return {}

def process_image_with_predictor(
    image_path, 
    predictor, 
    output_dir, 
    bboxes=None,
    min_area=1000,
    use_points=True
):
    """
    Process an image with SAM2 image predictor using bounding box(es),
    save segments, and return segment paths.
    
    Args:
        image_path: Path to the input image
        predictor: SAM2 image predictor
        output_dir: Directory to save segments
        bboxes: List of bounding box coordinates [x1, y1, x2, y2] or None for full image
        min_area: Minimum area for segments to be considered
        use_points: Whether to use foreground points at bbox midpoints
        
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
        
        # Set the image for the predictor
        predictor.set_image(image_np)
        
        # Handle bounding box options
        boxes_to_process = []
        
        # If no specific boxes, use full image bounds
        if bboxes is None:
            height, width = image_np.shape[:2]
            boxes_to_process = [[0, 0, width, height]]
            logger.info(f"Using full image bounds ({width}x{height}) for {image_basename}")
        else:
            # Convert to list of lists if it's a single box
            if isinstance(bboxes[0], int):
                boxes_to_process = [bboxes]
            else:
                boxes_to_process = bboxes
            logger.info(f"Using {len(boxes_to_process)} bounding boxes from JSON for {image_basename}")
        
        # Process each bounding box
        all_masks = []
        all_scores = []
        segment_paths = []
        
        for box_idx, bbox in enumerate(boxes_to_process):
            # Calculate midpoint of the bounding box
            x1, y1, x2, y2 = bbox
            
            # Convert box to torch tensor and correct format
            box_torch = torch.tensor(bbox, device=predictor.device)[None, :]
            
            # Prepare point inputs if using points
            point_coords = None
            point_labels = None
            
            if use_points:
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                
                # Create point coordinates tensor (shape [1, 2])
                point_coords = torch.tensor([[midpoint_x, midpoint_y]], device=predictor.device)
                
                # Create point labels tensor (1 = foreground point)
                point_labels = torch.tensor([1], device=predictor.device)
                
                logger.info(f"Using foreground point at ({midpoint_x:.1f}, {midpoint_y:.1f}) for box {box_idx}")
            else:
                logger.info(f"Using only bounding box without points for box {box_idx}")
            
            # Get prediction with box and optional point
            with torch.inference_mode():
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box_torch,
                    multimask_output=False
                )
            
            # Create a format similar to automatic mask generator for visualization
            for i, mask in enumerate(masks):
                # Calculate area
                area = np.sum(mask)
                if area < min_area:
                    continue
                    
                all_masks.append({
                    'segmentation': mask,
                    'area': area,
                    'bbox': bbox,
                    'predicted_iou': scores[i],
                    'source_box_idx': box_idx,
                    'used_point': use_points
                })
                all_scores.append(scores[i])
        
        logger.info(f"Generated {len(all_masks)} valid masks for {image_basename}")
        
        # Sort masks by score for better visualization
        sorted_indices = np.argsort(all_scores)[::-1]  # Descending order
        filtered_masks = [all_masks[i] for i in sorted_indices]
        
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
            box_idx = mask_data.get('source_box_idx', 0)
            point_suffix = "_point" if mask_data.get('used_point', False) else "_nopoint"
            output_path = os.path.join(segments_dir, f"{image_name}_box{box_idx}{point_suffix}_seg{i}.png")
            
            # Save with proper color conversion
            cv2.imwrite(output_path, cv2.cvtColor(segment_image, cv2.COLOR_RGB2BGR))
            
            # Add to segment paths
            segment_paths.append(output_path)
        
        # Create and save a visualization with all segments overlaid on the original image
        debug_vis_path = os.path.join(image_dir, f"{image_name}_segmentation_overlay.png")
        create_segmentation_visualization(image_np, filtered_masks, debug_vis_path)
        
        # Also create a visualization of the bounding boxes
        bbox_vis_path = os.path.join(image_dir, f"{image_name}_bboxes.png")
        visualize_bounding_boxes(image_np, boxes_to_process, bbox_vis_path, use_points=use_points)
        
        return segment_paths, debug_vis_path, image_dir
    
    except Exception as e:
        logger.error(f"Error processing {image_path} with predictor: {str(e)}")
        traceback.print_exc()
        return [], None, None

def visualize_bounding_boxes(image, bboxes, output_path, use_points=True):
    """
    Create a visualization of the bounding boxes on the image.
    
    Args:
        image: Original image as numpy array
        bboxes: List of bounding box coordinates [x1, y1, x2, y2]
        output_path: Path to save the visualization
        use_points: Whether to visualize the midpoint used as foreground point
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw each bounding box
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        
        # Generate a random color
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        
        # Draw rectangle
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Calculate midpoint 
        midpoint_x = int((x1 + x2) / 2)
        midpoint_y = int((y1 + y2) / 2)
        
        # Draw midpoint if using points
        if use_points:
            # Draw a circle at the midpoint (point coordinate)
            cv2.circle(vis_image, (midpoint_x, midpoint_y), 5, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(vis_image, (midpoint_x, midpoint_y), 5, (0, 0, 0), 1)     # Black outline
            
            # Add label for midpoint
            cv2.putText(vis_image, "Point", (midpoint_x + 10, midpoint_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add label for box
        cv2.putText(vis_image, f"Box {i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    logger.info(f"Bounding box visualization saved to {output_path}")

def show_mask(mask, ax, random_color=False, borders = True):
    # If random_color is True, generate a random color with 60% opacity
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # Otherwise use a predefined blue color with 60% opacity
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # Get height and width from the last two dimensions of the mask
    h, w = mask.shape[-2:]
    
    # Convert mask to 8-bit unsigned integer type
    mask = mask.astype(np.uint8)
    
    # Create colored mask by multiplying binary mask with color
    # Reshape operations ensure proper broadcasting
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # If borders are requested
    if borders:
        import cv2
        # Find external contours in the binary mask
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        # Smooth each contour using Douglas-Peucker algorithm
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        
        # Draw the smoothed contours on the mask image with white color and 50% opacity
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    
    # Display the mask image on the provided matplotlib axis
    ax.imshow(mask_image)

def save_individual_mask_visualizations(image, masks, output_dir):
    """
    Save individual visualizations for each mask using show_mask function.
    
    Args:
        image: Original image as numpy array
        masks: List of mask dictionaries from SAM2
        output_dir: Directory to save the visualizations
        
    Returns:
        Path to the directory containing mask visualizations
    """
    # Ensure output directory exists
    mask_vis_dir = os.path.join(output_dir, "mask_visualizations")
    os.makedirs(mask_vis_dir, exist_ok=True)
    
    # Process each mask
    for i, mask_data in enumerate(masks):
        binary_mask = mask_data['segmentation']
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display original image
        ax.imshow(image)
        
        # Show mask using the show_mask function
        show_mask(binary_mask, ax, random_color=True, borders=True)
        
        # Clean up the visualization
        ax.axis('off')
        
        # Save the figure
        output_path = os.path.join(mask_vis_dir, f"mask_{i}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)  # Close the figure to free memory
        
    logger.info(f"Individual mask visualizations saved to {mask_vis_dir}")
    return mask_vis_dir

def load_detections_from_json(json_path):
    """
    Load detections from a JSON file.
    
    Expected format:
    [
        {
            "coordinates": [x1, y1, x2, y2],
            "label": "class_name"
        },
        ...
    ]
    
    Args:
        json_path: Path to the detection JSON file
        
    Returns:
        List of detections, each with 'coordinates' and 'label'
    """
    try:
        logger.info(f"Loading detections from: {json_path}")
        with open(json_path, 'r') as f:
            detections = json.load(f)
        
        # Log detection info
        valid_detections = []
        for i, detection in enumerate(detections):
            if "coordinates" not in detection or "label" not in detection:
                logger.warning(f"Detection {i} in {json_path} is missing required fields: {detection}")
                continue
                
            coords = detection["coordinates"]
            label = detection["label"]
            
            if len(coords) != 4:
                logger.warning(f"Detection {i} in {json_path} has invalid coordinates format: {coords}")
                continue
                
            valid_detections.append(detection)
            logger.debug(f"Detection {i}: label='{label}', bbox={coords}")
        
        logger.info(f"Loaded {len(valid_detections)} valid detections out of {len(detections)} from {json_path}")
        return valid_detections
    except Exception as e:
        logger.error(f"Error loading detections from {json_path}: {str(e)}")
        logger.exception("Detailed error:")
        return []

def process_image_with_detections(
    image_path,
    predictor,
    output_dir,
    detections_dir,
    min_area=1000,
    use_points=True
):
    """
    Process an image with SAM2 image predictor using detection boxes,
    save segments, and return segment paths.
    
    Args:
        image_path: Path to the input image
        predictor: SAM2 image predictor
        output_dir: Directory to save segments
        detections_dir: Directory containing detection JSONs
        min_area: Minimum area for segments to consider
        use_points: Whether to use foreground points at bbox midpoints
        
    Returns:
        Tuple of (list of segment paths, path to debug visualization, image_dir)
    """
    try:
        # Get base filename without extension
        image_basename = os.path.basename(image_path)
        image_name = os.path.splitext(image_basename)[0]
        
        logger.info(f"=== Processing image with detections: {image_path} ===")
        
        # Find corresponding detection JSON 
        detection_json_path = os.path.join(detections_dir, f"{image_name}.json")
        if not os.path.exists(detection_json_path):
            logger.error(f"No detection JSON found at {detection_json_path}")
            return [], None, None
        
        logger.info(f"Found detection file: {detection_json_path}")    
        
        # Load detections
        detections = load_detections_from_json(detection_json_path)
        if not detections:
            logger.error(f"No valid detections found in {detection_json_path}")
            return [], None, None
        
        # Create organized directory structure
        image_dir = os.path.join(output_dir, image_name)
        segments_dir = os.path.join(image_dir, "segments")
        
        # Create directories
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(segments_dir, exist_ok=True)
        logger.debug(f"Created output directories: {image_dir}, {segments_dir}")
        
        # Load and process the image
        logger.info(f"Loading image: {image_path}")
        image = Image.open(image_path)
        image_np = np.array(image.convert("RGB"))
        logger.debug(f"Image loaded with shape: {image_np.shape}")
        
        # Set the image for the predictor
        predictor.set_image(image_np)
        logger.debug(f"Set image for SAM2 predictor")
        
        # Process each bounding box from detections
        all_masks = []
        all_scores = []
        segment_paths = []
        boxes_to_process = []
        labels = []
        
        # Extract bounding boxes and labels from detections
        for i, detection in enumerate(detections):
            if "coordinates" not in detection or "label" not in detection:
                logger.warning(f"Skipping detection {i}: missing coordinates or label")
                continue
                
            bbox = detection["coordinates"]
            label = detection["label"]
            
            if len(bbox) != 4:
                logger.warning(f"Skipping detection {i}: invalid coordinates format")
                continue
                
            boxes_to_process.append(bbox)
            labels.append(label)
        
        logger.info(f"Processing {len(boxes_to_process)} detections for {image_basename}")
        
        start_time = time.time()
        
        # Process each detection's bounding box
        for box_idx, (bbox, label) in enumerate(zip(boxes_to_process, labels)):
            # Extract coordinates
            x1, y1, x2, y2 = bbox
            
            # Log progress
            logger.info(f"Processing detection {box_idx+1}/{len(boxes_to_process)}: label='{label}', bbox={bbox}")
            
            # Convert box to torch tensor and correct format
            box_torch = torch.tensor(bbox, device=predictor.device)[None, :]
            
            # Prepare point inputs if using points
            point_coords = None
            point_labels = None
            
            if use_points:
                midpoint_x = (x1 + x2) / 2
                midpoint_y = (y1 + y2) / 2
                
                # Create point coordinates tensor (shape [1, 2])
                point_coords = torch.tensor([[midpoint_x, midpoint_y]], device=predictor.device)
                
                # Create point labels tensor (1 = foreground point)
                point_labels = torch.tensor([1], device=predictor.device)
                
                logger.info(f"Using foreground point at ({midpoint_x:.1f}, {midpoint_y:.1f}) for detection {box_idx} ({label})")
            else:
                logger.info(f"Using only bounding box without points for detection {box_idx} ({label})")
            
            # Get prediction with box and optional point
            logger.debug(f"Running SAM2 prediction for detection {box_idx}...")
            mask_start_time = time.time()
            try:
                with torch.inference_mode():
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box_torch,
                        multimask_output=False
                    )
                mask_time = time.time() - mask_start_time
                logger.debug(f"SAM2 prediction completed in {mask_time:.2f}s, got {len(masks)} masks")
            except Exception as e:
                logger.error(f"Error during SAM2 prediction for detection {box_idx}: {str(e)}")
                logger.exception("Detailed error:")
                continue
            
            # Create a format similar to automatic mask generator for visualization
            for i, mask in enumerate(masks):
                # Calculate area
                area = np.sum(mask)
                logger.debug(f"Mask {i} for detection {box_idx} has area: {area} pixels")
                
                if area < min_area:
                    logger.debug(f"Skipping mask {i} (area {area} < min_area {min_area})")
                    continue
                    
                all_masks.append({
                    'segmentation': mask,
                    'area': area,
                    'bbox': bbox,
                    'predicted_iou': scores[i],
                    'source_box_idx': box_idx,
                    'label': label,
                    'used_point': use_points
                })
                all_scores.append(scores[i])
                logger.debug(f"Added mask {i} for detection {box_idx} with score: {scores[i]:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Generated {len(all_masks)} valid masks in {total_time:.2f}s")
        
        if len(all_masks) == 0:
            logger.warning(f"No valid masks generated for {image_basename}")
            return [], None, None
        
        # Sort masks by score for better visualization
        sorted_indices = np.argsort(all_scores)[::-1]  # Descending order
        filtered_masks = [all_masks[i] for i in sorted_indices]
        
        logger.info(f"Saving {len(filtered_masks)} segments...")
        
        # Process and save each segment
        for i, mask_data in enumerate(filtered_masks):
            # Get the binary mask
            binary_mask = mask_data['segmentation']
            binary_mask = binary_mask.astype(bool)
            
            # Create a copy of the original image
            segment_image = np.zeros_like(image_np)
            
            # Copy the original pixels where the mask is True
            segment_image[binary_mask] = image_np[binary_mask]
            
            # Get metadata for naming
            box_idx = mask_data.get('source_box_idx', 0)
            label = mask_data.get('label', 'unknown')
            point_suffix = "_point" if mask_data.get('used_point', False) else "_nopoint"
            score = mask_data.get('predicted_iou', 0.0)
            
            # Create output path with the requested naming format:
            # image_name_label_detection_index.png
            output_path = os.path.join(segments_dir, f"{image_name}_{label}_{box_idx}{point_suffix}.png")
            
            # Save with proper color conversion
            cv2.imwrite(output_path, cv2.cvtColor(segment_image, cv2.COLOR_RGB2BGR))
            logger.debug(f"Saved segment {i} to {output_path} (score: {score:.4f})")
            
            # Add to segment paths
            segment_paths.append(output_path)
        
        # Create and save a visualization with all segments overlaid on the original image
        logger.info(f"Creating visualization overlays...")
        debug_vis_path = os.path.join(image_dir, f"{image_name}_segmentation_overlay.png")
        create_segmentation_visualization(image_np, filtered_masks, debug_vis_path)
        
        # Also create a visualization of the bounding boxes with labels
        bbox_vis_path = os.path.join(image_dir, f"{image_name}_detections.png")
        visualize_detections(image_np, boxes_to_process, labels, bbox_vis_path, use_points=use_points)
        
        logger.info(f"=== Finished processing {image_basename}: {len(segment_paths)} segments created ===")
        return segment_paths, debug_vis_path, image_dir
    
    except Exception as e:
        logger.error(f"Error processing {image_path} with detections: {str(e)}")
        logger.exception("Detailed error:")
        return [], None, None

def visualize_detections(image, bboxes, labels, output_path, use_points=True):
    """
    Create a visualization of the detection bounding boxes with labels on the image.
    
    Args:
        image: Original image as numpy array
        bboxes: List of bounding box coordinates [x1, y1, x2, y2]
        labels: List of class labels for each box
        output_path: Path to save the visualization
        use_points: Whether to visualize the midpoint used as foreground point
    """
    logger.debug(f"Creating detection visualization with {len(bboxes)} boxes")
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw each bounding box
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = bbox
        
        # Generate a color based on the label for consistency
        # Hash the label string to get a consistent color for each class
        hash_val = hash(label) % 255
        color = (hash_val, 255 - hash_val, (hash_val * 2) % 255)
        
        # Draw rectangle
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        logger.debug(f"Drew detection {i}: label='{label}', bbox={bbox}")
        
        # Calculate midpoint 
        midpoint_x = int((x1 + x2) / 2)
        midpoint_y = int((y1 + y2) / 2)
        
        # Draw midpoint if using points
        if use_points:
            # Draw a circle at the midpoint (point coordinate)
            cv2.circle(vis_image, (midpoint_x, midpoint_y), 5, (0, 255, 0), -1)  # Green filled circle
            cv2.circle(vis_image, (midpoint_x, midpoint_y), 5, (0, 0, 0), 1)     # Black outline
            logger.debug(f"Drew point for detection {i} at ({midpoint_x}, {midpoint_y})")
        
        # Add label
        text = f"{label} ({i})"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(vis_image, (int(x1), int(y1)-text_height-10), (int(x1)+text_width+5, int(y1)), color, -1)
        
        # Draw text
        cv2.putText(vis_image, text, (int(x1), int(y1)-5), font, font_scale, (255, 255, 255), thickness)
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    logger.info(f"Detection visualization saved to {output_path}")

def main():
    args = parse_args()
    
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
        
    # Setup device
    device = get_device()
    
    # Create output Path object
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SAM2
    sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    logger.info("Loading SAM2 model...")
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    
    # Determine which segmentation mode to use
    bbox_data = {}
    
    if args.use_bbox_json:
        # Mode 3: Use image predictor with bounding boxes from JSON
        logger.info(f"Using SAM2 image predictor with bounding boxes from {args.use_bbox_json}")
        segmentation_mode = "bbox_predictor"
        predictor = SAM2ImagePredictor(sam2)
        bbox_data = load_bboxes_from_json(args.use_bbox_json)
        
        process_func = lambda img_path, out_dir, min_area: process_image_with_predictor(
            img_path, 
            predictor, 
            out_dir, 
            bboxes=bbox_data.get(Path(img_path).stem, None),
            min_area=min_area,
            use_points=args.use_points
        )
        
    elif args.use_full_image_predictor:
        # Mode 2: Use image predictor with full image bounds
        logger.info("Using SAM2 image predictor with full image bounds")
        segmentation_mode = "full_image_predictor"
        predictor = SAM2ImagePredictor(sam2)
        
        process_func = lambda img_path, out_dir, min_area: process_image_with_predictor(
            img_path, 
            predictor, 
            out_dir, 
            bboxes=None,  # None means use full image bounds
            min_area=min_area,
            use_points=args.use_points
        )
        
    elif args.detections_dir:
        # Mode 4: Use detection-based segmentation
        logger.info(f"Using detection-based segmentation with directory: {args.detections_dir}")
        segmentation_mode = "detection_based"
        predictor = SAM2ImagePredictor(sam2)
        process_func = lambda img_path, out_dir, min_area: process_image_with_detections(
            img_path, 
            predictor, 
            out_dir, 
            args.detections_dir,
            min_area=min_area,
            use_points=args.use_points
        )
        
    else:
        # Mode 1: Use automatic mask generator (default)
        logger.info("Using SAM2 automatic mask generator")
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=64,
            points_per_batch=128,
            pred_iou_thresh=0.7,
            min_mask_region_area=10.0,
        )
        
        process_func = lambda img_path, out_dir, min_area: process_image_and_generate_segments(
            img_path, mask_generator, out_dir, min_area
        )
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator(
        model_types=args.models,
        device=device
    )
    
    # Prepare summary
    summary = {
        "output_directory": str(output_dir),
        "processed_images": [],
        "segmentation_mode": segmentation_mode,
        "used_points": args.use_points if segmentation_mode != "automatic_mask_generator" else None
    }
    
    # Determine processing mode - single image or directory
    if args.image:
        # Single image mode
        image_file = Path(args.image)
        if not image_file.exists():
            logger.error(f"Image file '{image_file}' does not exist.")
            return 1
            
        # Check for corresponding detection file if using detection mode
        if args.detections_dir:
            detection_json_path = os.path.join(args.detections_dir, f"{image_file.stem}.json")
            if not os.path.exists(detection_json_path):
                logger.error(f"No detection JSON found for {image_file} at {detection_json_path}")
                return 1
            
        logger.info(f"Processing single image: {image_file}")
        summary["mode"] = "single_image"
        summary["input_image"] = str(image_file)
        
        # Process the single image
        try:
            # Generate segments
            segment_paths, debug_vis_path, image_dir = process_func(
                img_path=str(image_file),
                out_dir=output_dir,
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
                "mask_visualizations_dir": os.path.join(image_dir, "mask_visualizations"),
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
        
        # If using detection mode, filter to only images with corresponding detection files
        if args.detections_dir:
            valid_image_files = []
            for image_file in image_files:
                detection_json_path = os.path.join(args.detections_dir, f"{image_file.stem}.json")
                if os.path.exists(detection_json_path):
                    valid_image_files.append(image_file)
                else:
                    logger.warning(f"Skipping {image_file}: no corresponding detection JSON found")
            
            logger.info(f"Found {len(valid_image_files)} out of {len(image_files)} images with corresponding detection files")
            image_files = valid_image_files
            
            if not image_files:
                logger.error(f"No images with corresponding detection files found")
                return 1
        
        # Process each image in the directory
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                logger.info(f"Processing {image_file}")
                
                # Generate segments
                segment_paths, debug_vis_path, image_dir = process_func(
                    img_path=str(image_file),
                    out_dir=output_dir,
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
                    "mask_visualizations_dir": os.path.join(image_dir, "mask_visualizations"),
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



    
# def show_mask(mask, ax, random_color=False, borders = True):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30/255, 144/255, 255/255, 0.6])
#     h, w = mask.shape[-2:]
#     mask = mask.astype(np.uint8)
#     mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     if borders:
#         import cv2
#         contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
#         # Try to smooth contours
#         contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
#         mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
#     ax.imshow(mask_image)

# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

# def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca(), borders=borders)
#         if point_coords is not None:
#             assert input_labels is not None
#             show_points(point_coords, input_labels, plt.gca())
#         if box_coords is not None:
#             # boxes
#             show_box(box_coords, plt.gca())
#         if len(scores) > 1:
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         plt.show()

# show_masks(image, masks, scores, box_coords=input_box)


# python e2e_pipeline_v2/experiments/segment_gemini.py --image path/to/image.jpg --output_dir results --detections_dir path/to/detections --use_points