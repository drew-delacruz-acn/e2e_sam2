import os
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

# Import from our modules
from config import get_device, MODEL_CONFIG, SAM2_CHECKPOINT, SAVE_DIR, NUM_INFERENCE_POINTS
from utils.data_utils import read_image_for_inference, get_points_from_mask, visualize_segmentation
from models.model_utils import load_sam2_model, get_predictor, load_model_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with SAM2 model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask defining region to segment")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to fine-tuned checkpoint (uses original SAM2 if not provided)")
    parser.add_argument("--model_config", type=str, default=MODEL_CONFIG, help="Path to model config")
    parser.add_argument("--points", type=int, default=NUM_INFERENCE_POINTS, help="Number of points to sample")
    parser.add_argument("--output", type=str, default="segmentation_output.png", help="Output file path")
    return parser.parse_args()

def run_inference():
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load image and mask
    print(f"Loading image from {args.image}")
    image, mask = read_image_for_inference(args.image, args.mask)
    
    # Sample points from mask
    input_points = get_points_from_mask(mask, args.points)
    
    # Check if we have points
    if len(input_points) == 0:
        print("Error: No points could be generated from the mask. Is the mask empty?")
        return
    
    # Load model
    if args.checkpoint is None:
        print(f"Loading original SAM2 model from {SAM2_CHECKPOINT}")
        sam2_model = load_sam2_model(args.model_config, SAM2_CHECKPOINT, device)
    else:
        print(f"Loading fine-tuned SAM2 model from {args.checkpoint}")
        sam2_model = load_sam2_model(args.model_config, SAM2_CHECKPOINT, device)
        sam2_model = load_model_checkpoint(sam2_model, args.checkpoint)
    
    predictor = get_predictor(sam2_model, device)
    
    # Run inference
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    
    # Post-process masks
    masks = masks[:, 0].astype(bool)
    sorted_masks = masks[np.argsort(scores[:, 0])][::-1].astype(bool)
    
    # Create segmentation map
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if (mask * occupancy_mask).sum() / (mask.sum() + 1e-10) > 0.15:
            continue
        mask[occupancy_mask] = 0
        seg_map[mask] = i + 1
        occupancy_mask[mask] = 1
    
    # Visualize segmentation
    rgb_image = visualize_segmentation(seg_map)
    
    # Save output
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Input Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(rgb_image)
    plt.title("Segmentation Output")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Segmentation saved to {args.output}")
    
    # Save just the segmentation as an image
    cv2.imwrite(f"segmap_{args.output}", seg_map)
    cv2.imwrite(f"visual_{args.output}", rgb_image[..., ::-1])  # Convert RGB to BGR for cv2
    
    return seg_map, rgb_image

if __name__ == "__main__":
    run_inference() 