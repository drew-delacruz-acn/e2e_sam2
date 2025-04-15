"""
Utility functions for the pipeline.
"""
import os
import numpy as np
from PIL import Image

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
    Returns:
        IoU value
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate box areas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def ensure_dir(directory):
    """
    Ensure a directory exists, create if it doesn't
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_crops(image, boxes, output_dir):
    """
    Save cropped images from bounding boxes
    Args:
        image: PIL Image
        boxes: List of bounding boxes
        output_dir: Directory to save crops
    Returns:
        List of paths to saved crops
    """
    ensure_dir(output_dir)
    
    crop_paths = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_image = image.crop((x1, y1, x2, y2))
        
        output_path = os.path.join(output_dir, f"box_{idx}.png")
        cropped_image.save(output_path)
        crop_paths.append(output_path)
    
    return crop_paths 