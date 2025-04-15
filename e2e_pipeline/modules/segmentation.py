"""
Module for image segmentation and object detection.
"""
import os
from pathlib import Path
import sys
import json
import numpy as np
from PIL import Image
import torch

# Add the root directory to path to fix import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import from existing pipeline
from pipeline.modules.detection import ObjectDetector
from pipeline.modules.segmentation import Segmenter
from pipeline.config import get_config

def custom_predict_with_box(segmenter, image, box):
    """
    Custom function to predict masks using a box input, 
    working around the interface inconsistency in SAM2
    """
    # Set image for prediction
    segmenter.predictor.set_image(image)
    
    # Convert box to torch tensor and correct format
    box_torch = torch.tensor(box, device=segmenter.predictor.device)[None, :]
    
    # Get prediction - try different parameter names to find the one that works
    try:
        # First try with 'box' parameter
        with torch.inference_mode():
            masks, scores, _ = segmenter.predictor.predict(
                point_coords=None,
                point_labels=None, 
                box=box_torch,
                multimask_output=False
            )
    except TypeError:
        # If that fails, try with 'boxes' parameter
        try:
            with torch.inference_mode():
                masks, scores, _ = segmenter.predictor.predict(
                    point_coords=None,
                    point_labels=None, 
                    boxes=box_torch,
                    multimask_output=False
                )
        except TypeError:
            # If both fail, try without the parameter name
            with torch.inference_mode():
                masks, scores, _ = segmenter.predictor.predict(
                    None, None, box_torch, False
                )
    
    return masks[0], scores[0]

def segment_image(image_path, segmentation_config):
    """
    Segment an image to detect and crop objects based on text queries.
    
    Args:
        image_path (str): Path to the input image.
        segmentation_config (dict): Configuration for segmentation with keys:
            - queries: List of text queries for object detection
            - detection_threshold: Confidence threshold for detection
            - segmentation_threshold: Threshold for segmentation
            - results_dir: Directory to save results
            
    Returns:
        list: List of dictionaries containing crop information with keys:
            - crop_path: Path to the cropped image
            - original_image: Path to the original image
            - bbox: Bounding box coordinates [x1, y1, x2, y2]
            - query: The text query that detected this object
            - score: The confidence score of the detection
    """
    # Prepare results directory
    results_dir = Path(segmentation_config.get("results_dir", "results"))
    crops_dir = results_dir / "crops"
    masks_dir = results_dir / "masks"
    vis_dir = results_dir / "visualizations"
    
    for dir_path in [crops_dir, masks_dir, vis_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Override configurations
    config_override = {
        "detection": {
            "threshold": segmentation_config.get("detection_threshold", 0.1)
        },
        "segmentation": {
            "threshold": segmentation_config.get("segmentation_threshold", 0.5),
            "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "checkpoint": "checkpoints/sam2.1_hiera_large.pt"
        },
        "paths": {
            "results_dir": str(results_dir)
        }
    }
    
    # Get configuration
    config = get_config(config_override)
    
    # Initialize the detector and segmenter
    detector = ObjectDetector(config)
    segmenter = Segmenter(config)
    
    # Get the text queries
    text_queries = segmentation_config.get("queries", ["object"])
    
    # Run detection
    detection_results = detector.detect(
        image_path=image_path,
        text_queries=text_queries
    )
    
    # Extract detection results
    image = detection_results["image"]
    boxes = detection_results["boxes"]
    scores = detection_results["scores"]
    labels = detection_results["labels"]
    
    # Return early if no objects detected
    if len(boxes) == 0:
        print(f"No objects detected in {image_path}")
        return []
    
    # Convert PIL image to numpy array for segmentation
    image_np = np.array(image)
    
    # Run segmentation on detected objects
    segmentation_results = []
    for i, box in enumerate(boxes):
        box_list = [int(coord) for coord in box.tolist()]
        
        # Use segment_with_boxes for a single box
        try:
            # Use custom predict function instead of the built-in one
            masks, scores_seg = custom_predict_with_box(segmenter, image_np, box_list)
            
            # Add to results
            segmentation_results.append({
                "masks": masks,
                "scores": scores_seg,
                "box": box_list
            })
        except Exception as e:
            print(f"Error segmenting box {i}: {e}")
            # If segmentation fails, just use the box
            segmentation_results.append({
                "masks": None,
                "scores": None,
                "box": box_list
            })
    
    # Combine detection and segmentation results
    crop_infos = []
    for i, (box, score, label, seg_result) in enumerate(zip(boxes, scores, labels, segmentation_results)):
        # Convert box to list of integers
        box_list = [int(coord) for coord in box.tolist()]
        
        # Save the cropped image
        crop_path = str(crops_dir / f"{Path(image_path).stem}_crop_{i}.png")
        
        # Get the cropped image
        x1, y1, x2, y2 = box_list
        crop = image.crop((x1, y1, x2, y2))
        crop.save(crop_path)
        
        # Create crop info dictionary
        crop_info = {
            "crop_path": crop_path,
            "original_image": image_path,
            "bbox": box_list,
            "query": text_queries[label.item()],
            "score": score.item()
        }
        
        crop_infos.append(crop_info)
    
    # Save metadata to JSON
    metadata_path = results_dir / f"{Path(image_path).stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(crop_infos, f, indent=2)
    
    print(f"Processed {image_path}: detected and segmented {len(crop_infos)} objects")
    return crop_infos

# Test function for this module
def test_segmentation():
    """Test function for segmentation module"""
    # Configuration for testing
    test_image_path = "/Users/andrewdelacruz/e2e_sam2/data/thor_hammer.jpeg"  # Update with your test image
    test_config = {
        "queries": ["hammer", "person", "chair"],
        "detection_threshold": 0.1,
        "segmentation_threshold": 0.5,
        "results_dir": "results/test_segmentation"
    }
    
    try:
        crop_infos = segment_image(test_image_path, test_config)
        print(f"Test successful! Segmented {len(crop_infos)} objects.")
        for i, crop_info in enumerate(crop_infos):
            print(f"Crop {i+1}:")
            print(f"  Query: {crop_info['query']}")
            print(f"  Score: {crop_info['score']:.4f}")
            print(f"  Crop path: {crop_info['crop_path']}")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_segmentation() 