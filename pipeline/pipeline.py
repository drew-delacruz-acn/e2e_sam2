"""
End-to-end object detection and segmentation pipeline.
"""
import os
import numpy as np
from PIL import Image

from .config import get_config
from .modules.detection import ObjectDetector
from .modules.segmentation import Segmenter
from .modules.visualization import Visualizer
from .modules.utils import save_crops, ensure_dir

class Pipeline:
    """End-to-end object detection and segmentation pipeline"""
    
    def __init__(self, config_override=None):
        """
        Initialize the pipeline with configuration
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config(config_override)
        self.detector = ObjectDetector(self.config)
        self.segmenter = Segmenter(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Ensure results directory exists
        ensure_dir(self.config["paths"]["results_dir"])
        print(f"Pipeline initialized with results directory: {self.config['paths']['results_dir']}")
    
    def run(self, image_path, text_queries, visualize=True, save_results=True):
        """
        Run the complete pipeline
        Args:
            image_path: Path to input image
            text_queries: List of text queries for detection
            visualize: Whether to display visualizations
            save_results: Whether to save results
        Returns:
            Dictionary with all results
        """
        print(f"Running pipeline on {image_path} with queries: {text_queries}")
        
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
        
        # Step 4: Segmentation
        image_np = np.array(detection_results["image"])
        point_segmentation = self.segmenter.segment_with_points(
            image_np, detection_results["boxes"]
        )
        box_segmentation = self.segmenter.segment_with_boxes(
            image_np, detection_results["boxes"]
        )
        
        # Step 5: Visualization of segmentation if requested
        if visualize and save_results:
            point_seg_dir = os.path.join(vis_dir, "point_segmentation")
            box_seg_dir = os.path.join(vis_dir, "box_segmentation")
            ensure_dir(point_seg_dir)
            ensure_dir(box_seg_dir)
            
            self.visualizer.show_segmentation(
                image_np, point_segmentation,
                score_threshold=self.config["segmentation"]["score_threshold"],
                save_dir=point_seg_dir
            )
            
            self.visualizer.show_segmentation(
                image_np, box_segmentation,
                score_threshold=self.config["segmentation"]["score_threshold"],
                save_dir=box_seg_dir
            )
        elif visualize:
            self.visualizer.show_segmentation(
                image_np, point_segmentation,
                score_threshold=self.config["segmentation"]["score_threshold"]
            )
            
            self.visualizer.show_segmentation(
                image_np, box_segmentation,
                score_threshold=self.config["segmentation"]["score_threshold"]
            )
        
        # Return all results
        return {
            "detection": detection_results,
            "point_segmentation": point_segmentation,
            "box_segmentation": box_segmentation
        }

# Example usage
if __name__ == "__main__":
    # Initialize with default configuration
    pipeline = Pipeline()
    
    # Run the complete pipeline
    results = pipeline.run(
        image_path="../data/thor_hammer.jpeg",
        text_queries=["hammer", "mallet"],
        visualize=True,
        save_results=True
    )
