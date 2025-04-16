#!/usr/bin/env python
"""
SAM2 Model Inference Script
This script loads a fine-tuned SAM2 model and performs inference on an image.
"""
import os
import argparse
import numpy as np
import torch
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with SAM2 model")
    parser.add_argument("--image_path", type=str, default="sample_image.jpg",
                      help="Path to input image")
    parser.add_argument("--mask_path", type=str, default="sample_mask.png",
                      help="Path to mask defining region to segment")
    parser.add_argument("--sam2_checkpoint", type=str, default="sam2_hiera_large.pt",
                      help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--num_points", type=int, default=30,
                      help="Number of points to sample from the mask")
    parser.add_argument("--output_path", type=str, default="output_segmentation.png",
                      help="Path to save output segmentation")
    return parser.parse_args()

def setup_device():
    """Set up computation device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Use bfloat16 for the entire script (memory efficient)
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU. This will be significantly slower.")
    return device

def read_image(image_path, mask_path):
    """Read and resize image and mask"""
    img = cv2.imread(image_path)[...,::-1]  # Read image as RGB
    mask = cv2.imread(mask_path, 0)  # Mask of the region to segment
    
    # Resize image to maximum size of 1024
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), 
                     interpolation=cv2.INTER_NEAREST)
    return img, mask

def get_points(mask, num_points):
    """Sample points inside the input mask"""
    points = []
    if np.sum(mask > 0) == 0:
        print("Warning: Mask is empty, cannot sample points")
        return np.array(points)
        
    for i in range(num_points):
        coords = np.argwhere(mask > 0)
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)

def create_segmentation_visualization(image, seg_map):
    """Create colored visualization of segmentation map"""
    rgb_image = np.copy(image)
    overlay = np.zeros_like(rgb_image)
    
    # Assign random colors to each segment
    for id_class in range(1, seg_map.max() + 1):
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        overlay[seg_map == id_class] = color
    
    # Blend original image with segmentation
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, rgb_image, 1 - alpha, 0, rgb_image)
    
    # Draw segment boundaries
    contours = []
    for id_class in range(1, seg_map.max() + 1):
        mask = (seg_map == id_class).astype(np.uint8)
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(contour)
    
    cv2.drawContours(rgb_image, contours, -1, (255, 255, 255), 1)
    
    return rgb_image

def main():
    """Main inference function"""
    args = parse_args()
    device = setup_device()
    
    # Load image and mask
    print(f"Reading image from {args.image_path}")
    image, mask = read_image(args.image_path, args.mask_path)
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    
    # Sample points from mask
    input_points = get_points(mask, args.num_points)
    if len(input_points) == 0:
        print("Error: Could not sample points from mask")
        return
    print(f"Sampled {len(input_points)} points from mask")
    
    # Load model
    print(f"Loading model from {args.sam2_checkpoint}")
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():  # Prevent gradient calculation for efficient inference
        predictor.set_image(image)  # Image encoder
        masks, scores, logits = predictor.predict(  # Prompt encoder + mask decoder
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    
    # Process results
    masks = masks[:, 0].astype(bool)
    sorted_indices = np.argsort(scores[:, 0])[::-1]  # Sort by scores (highest first)
    sorted_masks = masks[sorted_indices]
    
    # Create segmentation map
    print("Creating segmentation map...")
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    
    # Assign each mask to a segment, avoiding overlaps
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        # Skip if this mask overlaps too much with existing segments
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        # Remove overlapping parts
        mask = mask & ~occupancy_mask
        # Add to segmentation map
        seg_map[mask] = i + 1
        # Update occupancy mask
        occupancy_mask = occupancy_mask | mask
    
    # Create visualization
    vis_image = create_segmentation_visualization(image, seg_map)
    
    # Save results
    print(f"Saving segmentation to {args.output_path}")
    cv2.imwrite(args.output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"Found {seg_map.max()} segments")
    print("Done!")

if __name__ == "__main__":
    main()