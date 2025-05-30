#!/usr/bin/env python
"""
SAM2 Model Inference Script
This script loads a fine-tuned SAM2 model and performs inference on an image.
Optionally compares performance with the original pre-trained model.
"""
import os
import sys
import argparse
import numpy as np
import torch
import cv2
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import setup_device, load_sam2_model
from utils.data_utils import read_image, get_points
from utils.visualization import create_segmentation_visualization, create_seg_map
from utils.metrics import compute_metrics
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with SAM2 model")
    parser.add_argument("--image_path", type=str, default="test/bear_raw.jpg",
                      help="Path to input image")
    parser.add_argument("--mask_path", type=str, default="test/bear_mask.png",
                      help="Path to mask defining region to segment")
    parser.add_argument("--gt_mask_path", type=str, default="test/bear_mask.png",
                      help="Path to ground truth mask for accuracy evaluation (optional)")
    parser.add_argument("--sam2_checkpoint", type=str, default="../outputs/models/model_best.torch",
                      help="Path to fine-tuned SAM2 checkpoint")
    parser.add_argument("--original_checkpoint", type=str, default="../checkpoints/sam2.1_hiera_large.pt",
                      help="Path to original SAM2 checkpoint for comparison (if not provided, no comparison is done)")
    parser.add_argument("--model_cfg", type=str,  default="../configs/sam2.1/sam2.1_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--num_points", type=int, default=30,
                      help="Number of points to sample from the mask")
    parser.add_argument("--output_path", type=str, default="../outputs/visualizations/output_segmentation.png",
                      help="Path to save output segmentation")
    parser.add_argument("--compare", action="store_true",
                      help="Compare finetuned and original model performance")
    return parser.parse_args()

def run_inference(model, image, input_points, device, model_name="Model"):
    """Run inference with the given model and return results"""
    predictor = SAM2ImagePredictor(model)
    
    start_time = time.time()
    with torch.no_grad():  # Prevent gradient calculation for efficient inference
        predictor.set_image(image)  # Image encoder
        masks, scores, logits = predictor.predict(  # Prompt encoder + mask decoder
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    end_time = time.time()
    
    # Create segmentation map
    seg_map, occupancy_mask = create_seg_map(masks, scores)
    
    print(f"{model_name} Inference Time: {end_time - start_time:.3f} seconds")
    print(f"{model_name} Found {seg_map.max()} segments")
    
    return seg_map, masks, scores

def main():
    """Main inference function"""
    args = parse_args()
    device = setup_device()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load image and mask
    print(f"Reading image from {args.image_path}")
    image, mask, gt_mask = read_image(args.image_path, args.mask_path, args.gt_mask_path)
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    
    # Sample points from mask
    input_points = get_points(mask, args.num_points)
    if len(input_points) == 0:
        print("Error: Could not sample point from mask")
        return
    print(f"Sampled {len(input_points)} points from mask")
    
    results = {}
    
    # Determine if the fine-tuned model is a state dict
    finetuned_is_state_dict = args.sam2_checkpoint.endswith('.torch') or args.sam2_checkpoint.endswith('.pt')
    
    # Load and run fine-tuned model
    print(f"Loading fine-tuned model from {args.sam2_checkpoint}")
    finetuned_model = load_sam2_model(args.model_cfg, args.sam2_checkpoint, device, is_state_dict=finetuned_is_state_dict)
    finetuned_seg_map, finetuned_masks, finetuned_scores = run_inference(
        finetuned_model, image, input_points, device, "Fine-tuned Model"
    )
    
    # Create visualization for fine-tuned model
    finetuned_vis = create_segmentation_visualization(image, finetuned_seg_map)
    finetuned_output_path = args.output_path
    print(f"Saving fine-tuned model segmentation to {finetuned_output_path}")
    cv2.imwrite(finetuned_output_path, cv2.cvtColor(finetuned_vis, cv2.COLOR_RGB2BGR))
    
    # Compute metrics for fine-tuned model
    if gt_mask is not None:
        finetuned_metrics = compute_metrics(finetuned_seg_map, gt_mask)
        results["Fine-tuned Model"] = finetuned_metrics
        print("\nFine-tuned Model Metrics:")
        for metric, value in finetuned_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # If comparison is enabled, load and run original model
    if args.compare and args.original_checkpoint:
        print(f"\nLoading original model from {args.original_checkpoint}")
        # Try to determine if original checkpoint is a state dict or a complete checkpoint
        original_is_state_dict = False  # Default assumption for original model
        
        try:
            # Check what type of file the original checkpoint is
            checkpoint = torch.load(args.original_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and not any(k in checkpoint for k in ["model", "state_dict"]):
                # If it's a dict but doesn't have model or state_dict keys, it's likely a direct state dict
                original_is_state_dict = True
                print("Detected original checkpoint as a state dict")
        except Exception:
            # If we can't easily determine, stick with default
            pass
            
        original_model = load_sam2_model(args.model_cfg, args.original_checkpoint, device, is_state_dict=original_is_state_dict)
        original_seg_map, original_masks, original_scores = run_inference(
            original_model, image, input_points, device, "Original Model"
        )
        
        # Create visualization for original model
        original_vis = create_segmentation_visualization(image, original_seg_map)
        original_output_path = os.path.splitext(args.output_path)[0] + "_original" + os.path.splitext(args.output_path)[1]
        print(f"Saving original model segmentation to {original_output_path}")
        cv2.imwrite(original_output_path, cv2.cvtColor(original_vis, cv2.COLOR_RGB2BGR))
        
        # Compute metrics for original model
        if gt_mask is not None:
            original_metrics = compute_metrics(original_seg_map, gt_mask)
            results["Original Model"] = original_metrics
            print("\nOriginal Model Metrics:")
            for metric, value in original_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # Show improvement
            print("\nImprovement (Fine-tuned vs Original):")
            for metric in finetuned_metrics.keys():
                diff = finetuned_metrics[metric] - original_metrics[metric]
                print(f"  {metric}: {diff:.4f} ({'+' if diff > 0 else ''}{diff/original_metrics[metric]*100:.2f}%)")
        
        # Create side-by-side comparison
        comparison_img = np.hstack((finetuned_vis, original_vis))
        comparison_output_path = os.path.splitext(args.output_path)[0] + "_comparison" + os.path.splitext(args.output_path)[1]
        print(f"Saving comparison to {comparison_output_path}")
        cv2.imwrite(comparison_output_path, cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
    
    print("Done!")

if __name__ == "__main__":
    main()