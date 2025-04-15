"""
Script to run only the object detection part of the pipeline
"""
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from pipeline.config import get_config
from pipeline.modules.detection import ObjectDetector

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run object detection using Owlv2")
    parser.add_argument("--image", type=str, default="data/thor_hammer.jpeg", 
                        help="Path to input image")
    parser.add_argument("--queries", type=str, nargs="+", default=["hammer"], 
                        help="Text queries for detection")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save results to disk")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--detection-threshold", type=float, default=0.1,
                        help="Detection confidence threshold")
    return parser.parse_args()

def visualize_detections(image, boxes, scores, labels, text_queries, save_path=None):
    """Visualize detection results with bounding boxes"""
    plt.figure(figsize=(10, 8))
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.title('Detected Objects')
    
    ax = plt.gca()
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for box, score, label in zip(boxes, scores, labels):
        box = box.detach().cpu().numpy()
        x, y, x2, y2 = box
        width, height = x2 - x, y2 - y
        
        color = colors[label % len(colors)]
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        plt.text(x, y-10, f"{text_queries[label]}: {score:.2f}", color=color, fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Detection visualization saved to {save_path}")
    
    plt.show()

def save_crops(image, boxes, output_dir):
    """Save cropped images from bounding boxes"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    crop_paths = []
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_image = image.crop((x1, y1, x2, y2))
        
        output_path = os.path.join(output_dir, f"box_{idx}.png")
        cropped_image.save(output_path)
        crop_paths.append(output_path)
    
    return crop_paths

def main():
    """Main function to run detection"""
    args = parse_args()
    
    # Custom configuration for detection
    config_override = {
        "detection": {
            "threshold": args.detection_threshold
        },
        "paths": {
            "results_dir": args.results_dir
        }
    }
    
    # Get configuration
    config = get_config(config_override)
    
    # Initialize the detector
    detector = ObjectDetector(config)
    
    # Run detection
    detection_results = detector.detect(
        image_path=args.image,
        text_queries=args.queries
    )
    
    # Extract results
    image = detection_results["image"]
    boxes = detection_results["boxes"]
    scores = detection_results["scores"]
    labels = detection_results["labels"]
    text_queries = detection_results["text_queries"]
    
    # Visualize and save results if requested
    if args.save:
        # Create directories
        vis_dir = os.path.join(args.results_dir, "visualizations")
        crops_dir = os.path.join(args.results_dir, "crops")
        os.makedirs(vis_dir, exist_ok=True)
        os.makedirs(crops_dir, exist_ok=True)
        
        # Visualize detections
        detection_vis_path = os.path.join(vis_dir, "detection_results.png")
        visualize_detections(
            image, boxes, scores, labels, text_queries,
            save_path=detection_vis_path
        )
        
        # Save cropped objects
        crop_paths = save_crops(image, boxes, crops_dir)
        print(f"Saved {len(crop_paths)} cropped images to {crops_dir}")
    else:
        # Just visualize without saving
        visualize_detections(image, boxes, scores, labels, text_queries)
    
    print(f"Detection completed successfully!")

if __name__ == "__main__":
    main() 