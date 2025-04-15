"""
End-to-end object detection pipeline (without segmentation).
"""
import os
import numpy as np
from PIL import Image

from .config import get_config
from .modules.detection import ObjectDetector
from .modules.visualization import Visualizer
from .modules.utils import save_crops, ensure_dir

class DetectionPipeline:
    """Object detection pipeline without segmentation"""
    
    def __init__(self, config_override=None):
        """
        Initialize the pipeline with configuration
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config(config_override)
        self.detector = ObjectDetector(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Ensure results directory exists
        ensure_dir(self.config["paths"]["results_dir"])
        print(f"Detection Pipeline initialized with results directory: {self.config['paths']['results_dir']}")
    
    def run(self, image_path, text_queries, visualize=True, save_results=True):
        """
        Run the detection pipeline
        Args:
            image_path: Path to input image
            text_queries: List of text queries for detection
            visualize: Whether to display visualizations
            save_results: Whether to save results
        Returns:
            Dictionary with detection results
        """
        print(f"Running detection pipeline on {image_path} with queries: {text_queries}")
        
        # Step 1: Object Detection
        detection_results = self.detector.detect(image_path, text_queries)
        
        # Create paths for saving results
        vis_dir = os.path.join(self.config["paths"]["results_dir"], "visualizations")
        crops_dir = os.path.join(self.config["paths"]["results_dir"], "crops")
        if save_results:
            ensure_dir(vis_dir)
            ensure_dir(crops_dir)
        
        # Step 2: Save cropped images if requested
        if save_results:
            crop_paths = save_crops(
                detection_results["image"], 
                detection_results["boxes"],
                crops_dir
            )
            print(f"Saved {len(crop_paths)} cropped images to {crops_dir}")
        
        # Step 3: Visualization if requested
        if visualize and save_results:
            detection_vis_path = os.path.join(vis_dir, "detection_results.png")
            self.visualizer.show_detections(
                detection_results["image"],
                detection_results["boxes"],
                detection_results["scores"],
                detection_results["labels"],
                detection_results["text_queries"],
                save_path=detection_vis_path
            )
        elif visualize:
            self.visualizer.show_detections(
                detection_results["image"],
                detection_results["boxes"],
                detection_results["scores"],
                detection_results["labels"],
                detection_results["text_queries"]
            )
        
        # Return results
        return {
            "detection": detection_results
        }

# Example usage
if __name__ == "__main__":
    # Initialize with default configuration
    pipeline = DetectionPipeline()
    
    # Run the pipeline
    results = pipeline.run(
        image_path="data/thor_hammer.jpeg",
        text_queries=["hammer"],
        visualize=True,
        save_results=True
    ) 