"""
Utility functions for the detection module.
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def save_crops(image, boxes, output_dir, base_filename="box"):
    """
    Save cropped images from bounding boxes
    
    Args:
        image: PIL Image
        boxes: List of bounding boxes
        output_dir: Directory to save crops
        base_filename: Base name for output files
        
    Returns:
        List of paths to saved crops
    """
    ensure_dir(output_dir)
    
    crop_paths = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_image = image.crop((x1, y1, x2, y2))
        
        output_path = os.path.join(output_dir, f"{base_filename}_{idx}.png")
        cropped_image.save(output_path)
        crop_paths.append(output_path)
    
    return crop_paths

def visualize_detections(image, boxes, scores, labels, text_queries, output_path=None):
    """
    Visualize detection results
    
    Args:
        image: PIL Image
        boxes: List of bounding boxes
        scores: List of confidence scores
        labels: List of label indices
        text_queries: List of text queries corresponding to label indices
        output_path: Path to save visualization (optional)
        
    Returns:
        None or saved image path
    """
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image_np)
    
    # Plot each bounding box
    for box, score, label in zip(boxes, scores, labels):
        # Convert box tensor to list
        box_list = [coord.item() if hasattr(coord, 'item') else coord for coord in box]
        x1, y1, x2, y2 = box_list
        
        # Create rectangle patch
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        text = f"{text_queries[label]}: {score:.2f}"
        plt.text(x1, y1, text, color='white', fontsize=12, 
                 bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    
    # Save or show the figure
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()
        return None 