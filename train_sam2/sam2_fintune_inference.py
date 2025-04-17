#!/usr/bin/env python
"""
SAM2 Model Inference Script
This script loads a fine-tuned SAM2 model and performs inference on an image.
Optionally compares performance with the original pre-trained model.
"""
import os
import argparse
import numpy as np
import torch
import cv2
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with SAM2 model")
    parser.add_argument("--image_path", type=str, default="sample_image.jpg",
                      help="Path to input image")
    parser.add_argument("--mask_path", type=str, default="sample_mask.png",
                      help="Path to mask defining region to segment")
    parser.add_argument("--gt_mask_path", type=str, default=None,
                      help="Path to ground truth mask for accuracy evaluation (optional)")
    parser.add_argument("--sam2_checkpoint", type=str, default="sam2_hiera_large.pt",
                      help="Path to fine-tuned SAM2 checkpoint")
    parser.add_argument("--original_checkpoint", type=str, default=None,
                      help="Path to original SAM2 checkpoint for comparison (if not provided, no comparison is done)")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml",
                      help="Path to model config")
    parser.add_argument("--num_points", type=int, default=30,
                      help="Number of points to sample from the mask")
    parser.add_argument("--output_path", type=str, default="output_segmentation.png",
                      help="Path to save output segmentation")
    parser.add_argument("--compare", action="store_true",
                      help="Compare finetuned and original model performance")
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

def read_image(image_path, mask_path, gt_mask_path=None):
    """Read and resize image and mask"""
    img = cv2.imread(image_path)[...,::-1]  # Read image as RGB
    mask = cv2.imread(mask_path, 0)  # Mask of the region to segment
    
    # Resize image to maximum size of 1024
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), 
                     interpolation=cv2.INTER_NEAREST)
    
    # Load ground truth mask if provided
    gt_mask = None
    if gt_mask_path:
        gt_mask = cv2.imread(gt_mask_path, 0)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask, (int(mask.shape[1]), int(mask.shape[0])),
                             interpolation=cv2.INTER_NEAREST)
            # Convert to binary mask if needed
            if gt_mask.max() > 1:
                gt_mask = (gt_mask > 0).astype(np.uint8)
    
    return img, mask, gt_mask

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

def create_segmentation_visualization(image, seg_map, alpha=0.5):
    """Create colored visualization of segmentation map"""
    rgb_image = np.copy(image)
    overlay = np.zeros_like(rgb_image)
    
    # Assign random colors to each segment
    for id_class in range(1, seg_map.max() + 1):
        color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        overlay[seg_map == id_class] = color
    
    # Blend original image with segmentation
    cv2.addWeighted(overlay, alpha, rgb_image, 1 - alpha, 0, rgb_image)
    
    # Draw segment boundaries
    contours = []
    for id_class in range(1, seg_map.max() + 1):
        mask = (seg_map == id_class).astype(np.uint8)
        contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(contour)
    
    cv2.drawContours(rgb_image, contours, -1, (255, 255, 255), 1)
    
    return rgb_image

def compute_metrics(pred_mask, gt_mask):
    """Compute metrics between predicted mask and ground truth mask"""
    if gt_mask is None:
        return None
    
    # Convert to binary masks
    pred_binary = pred_mask > 0
    gt_binary = gt_mask > 0
    
    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    # Compute IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0
    
    # Compute Dice coefficient
    dice = (2 * intersection) / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0
    
    # Compute precision and recall
    true_positives = intersection
    false_positives = pred_binary.sum() - true_positives
    false_negatives = gt_binary.sum() - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "IoU": iou,
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    
    return metrics

def create_seg_map(masks, scores):
    """Create segmentation map from masks"""
    masks = masks[:, 0].astype(bool)
    sorted_indices = np.argsort(scores[:, 0])[::-1]  # Sort by scores (highest first)
    sorted_masks = masks[sorted_indices]
    
    # Create segmentation map
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
    
    return seg_map, occupancy_mask

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

def load_model(model_path, model_cfg, device, is_state_dict=False):
    """Load the model from checkpoint or state dict"""
    try:
        if is_state_dict:
            # Load from state dict (fine-tuned model)
            print(f"Loading model from state dict: {model_path}")
            # First load the base model with the architecture
            model = build_sam2(model_cfg, None, device=device)
            # Then load the state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            # Load regular checkpoint
            print(f"Loading model from checkpoint: {model_path}")
            # For original checkpoint, use the standard build_sam2 function
            model = build_sam2(model_cfg, model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        # Try the alternative approach - load base model and then checkpoint
        try:
            model = build_sam2(model_cfg, None, device=device)
            checkpoint = torch.load(model_path, map_location=device)
            # Try different ways the checkpoint might be structured
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                # Some SAM checkpoints store weights in a 'model' key
                model.load_state_dict(checkpoint["model"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Some checkpoints use 'state_dict' key
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Try direct loading
                model.load_state_dict(checkpoint)
            print("Alternative loading method successful!")
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            raise RuntimeError(f"Could not load model from {model_path}")
    
    return model

def main():
    """Main inference function"""
    args = parse_args()
    device = setup_device()
    
    # Load image and mask
    print(f"Reading image from {args.image_path}")
    image, mask, gt_mask = read_image(args.image_path, args.mask_path, args.gt_mask_path)
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    
    # Sample points from mask
    input_points = get_points(mask, args.num_points)
    if len(input_points) == 0:
        print("Error: Could not sample points from mask")
        return
    print(f"Sampled {len(input_points)} points from mask")
    
    results = {}
    
    # Determine if the fine-tuned model is a state dict
    finetuned_is_state_dict = args.sam2_checkpoint.endswith('.torch') or args.sam2_checkpoint.endswith('.pt')
    
    # Load and run fine-tuned model
    print(f"Loading fine-tuned model from {args.sam2_checkpoint}")
    finetuned_model = load_model(args.sam2_checkpoint, args.model_cfg, device, is_state_dict=finetuned_is_state_dict)
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
            
        original_model = load_model(args.original_checkpoint, args.model_cfg, device, is_state_dict=original_is_state_dict)
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