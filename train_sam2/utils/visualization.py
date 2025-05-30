"""
Visualization utilities for SAM2 training and inference.
"""
import os
import numpy as np
import cv2


def create_visualization(image, mask, alpha=0.5):
    """Create colored visualization of segmentation mask"""
    vis = image.copy()
    colored_mask = np.zeros_like(image)
    
    # Create a colored overlay for the mask
    colored_mask[mask > 0.5] = [255, 0, 0]  # Red for mask
    
    # Blend the original image with the colored mask
    cv2.addWeighted(colored_mask, alpha, vis, 1 - alpha, 0, vis)
    
    # Draw contours
    binary_mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    
    return vis


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


def save_comparison_visualization(img, gt_mask, pretrained_mask, finetuned_mask, save_path, sample_name):
    """Save side-by-side visualization of results"""
    # Create visualizations
    gt_vis = create_visualization(img, gt_mask)
    pretrained_vis = create_visualization(img, pretrained_mask)
    finetuned_vis = create_visualization(img, finetuned_mask)
    
    # Add titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    color = (255, 255, 255)
    
    # Add padding for titles
    padding = 40
    gt_vis = cv2.copyMakeBorder(gt_vis, padding, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    pretrained_vis = cv2.copyMakeBorder(pretrained_vis, padding, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    finetuned_vis = cv2.copyMakeBorder(finetuned_vis, padding, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Add titles
    cv2.putText(gt_vis, 'Ground Truth', (10, 30), font, font_scale, color, font_thickness)
    cv2.putText(pretrained_vis, 'Pre-trained', (10, 30), font, font_scale, color, font_thickness)
    cv2.putText(finetuned_vis, 'Fine-tuned', (10, 30), font, font_scale, color, font_thickness)
    
    # Combine images horizontally
    comparison = np.hstack((gt_vis, pretrained_vis, finetuned_vis))
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the comparison image
    output_path = os.path.join(save_path, f'{sample_name}_comparison.png')
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison visualization to: {output_path}")
    
    return output_path


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