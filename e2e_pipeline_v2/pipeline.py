"""
End-to-end object detection and segmentation pipeline using OwlViT and SAM2.
"""
import os
import yaml
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union

from e2e_pipeline_v2.modules.detection.detection import ObjectDetector
from e2e_pipeline_v2.modules.segmentation.segmentation import Segmenter
from e2e_pipeline_v2.modules.detection.utils import ensure_dir, save_crops, visualize_detections
from e2e_pipeline_v2.modules.embedding import EmbeddingGenerator, ModelType

class DetectionSegmentationPipeline:
    """End-to-end object detection and segmentation pipeline"""
    
    def __init__(self, config_path):
        """
        Initialize the pipeline with configuration
        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize detector - only use model_type, model_name is hardcoded
        model_type = self.config.get("detection", {}).get("model_type", "owlvit2")
        
        # Simple model selection, using hardcoded paths in the detector class
        detection_config = {
            "model_type": model_type,
            "threshold": self.config["segmentation"]["detection_threshold"],
            "force_cpu": self.config.get("detection", {}).get("force_cpu", False)
        }
        
        print(f"Initializing detector with model_type={model_type}")
        self.detector = ObjectDetector(detection_config)
        
        # Initialize segmenter
        segmentation_config = {
            "segmentation": {
                "model_config": self.config.get("segmentation", {}).get("model_config", "configs/sam2.1/sam2.1_hiera_l.yaml"),
                "checkpoint": self.config.get("segmentation", {}).get("checkpoint", "checkpoints/sam2.1_hiera_large.pt"),
                "score_threshold": self.config["segmentation"]["segmentation_threshold"]
            }
        }
        self.segmenter = Segmenter(segmentation_config)
        
        # Ensure results directory exists
        self.results_dir = self.config["segmentation"]["results_dir"]
        ensure_dir(self.results_dir)
        print(f"Pipeline initialized with results directory: {self.results_dir}")
        
        # Embedding generator (initialized on demand)
        self._embedding_generator = None
    
    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Lazy-loaded embedding generator to avoid loading models unless needed"""
        if self._embedding_generator is None:
            # Get embedding models from config or use defaults
            model_types = self.config.get("embedding", {}).get("model_types", ["clip"])
            
            # Initialize embedding generator
            self._embedding_generator = EmbeddingGenerator(
                model_types=model_types
            )
        return self._embedding_generator
    
    def visualize_mask(self, image, mask, border_color=(0, 255, 0), border_thickness=2, alpha=0.5):
        """
        Visualize a segmentation mask on the image
        Args:
            image: RGB image as numpy array
            mask: Binary mask
            border_color: Color of the border (BGR)
            border_thickness: Thickness of the border
            alpha: Opacity of the mask overlay
        Returns:
            Image with mask overlay
        """
        # Convert mask to proper format for visualization
        mask_bool = mask.astype(bool)
        mask_viz = np.zeros_like(image)
        mask_viz[mask_bool] = (0, 0, 255)  # Red for the mask area
        
        # Create border using contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create a copy of the original image
        vis_image = image.copy()
        
        # Draw mask overlay
        mask_overlay = cv2.addWeighted(
            vis_image, 
            1.0, 
            mask_viz.astype(np.uint8), 
            alpha, 
            0
        )
        
        # Apply the overlay only where mask is True
        vis_image = np.where(
            np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2),
            mask_overlay,
            vis_image
        )
        
        # Draw contours
        cv2.drawContours(
            vis_image, 
            contours, 
            -1, 
            border_color, 
            border_thickness
        )
        
        return vis_image
    
    def save_segmentation_results(self, image, segmentation_results, output_dir, prefix="segmentation"):
        """
        Save segmentation results as images
        Args:
            image: RGB image as numpy array
            segmentation_results: List of segmentation results
            output_dir: Directory to save results
            prefix: Prefix for output filenames
        Returns:
            List of paths to saved images
        """
        ensure_dir(output_dir)
        saved_paths = []
        
        for i, result in enumerate(segmentation_results):
            masks = result["masks"]
            scores = result["scores"]
            
            # Convert scores to a format we can iterate through
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            elif isinstance(scores, np.ndarray) and scores.ndim == 0:
                scores = np.array([float(scores)])
            elif not hasattr(scores, '__iter__'):
                scores = np.array([float(scores)])
            
            # Iterate through masks and scores
            for j, (mask, score) in enumerate(zip(masks, scores)):
                # Skip masks with low confidence
                if float(score) < self.config["segmentation"]["segmentation_threshold"]:
                    continue
                
                # Convert mask to numpy array if it's a tensor
                if isinstance(mask, torch.Tensor):
                    mask_array = mask.cpu().numpy()
                else:
                    # Already a NumPy array
                    mask_array = mask
                
                # Ensure mask is 2D
                if mask_array.ndim > 2:
                    mask_array = mask_array[0]  # Take first channel if multi-channel
                
                # Create visualization
                vis_image = self.visualize_mask(image, mask_array)
                
                # Save the visualization
                output_path = os.path.join(output_dir, f"{prefix}_{i}_{j}_score_{float(score):.2f}.png")
                cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                saved_paths.append(output_path)
                
                # Also save the binary mask
                mask_path = os.path.join(output_dir, f"{prefix}_{i}_{j}_mask.png")
                cv2.imwrite(mask_path, (mask_array * 255).astype(np.uint8))
                saved_paths.append(mask_path)
        
        return saved_paths
    
    def generate_embeddings(self, 
                          image: np.ndarray, 
                          segmentation_results: List[Dict[str, Any]],
                          output_dir: str,
                          apply_masks: bool = True,
                          save_json: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings for segmented objects
        
        Args:
            image: RGB image as numpy array
            segmentation_results: List of segmentation results
            output_dir: Directory to save embeddings
            apply_masks: Whether to apply masks (True) or just use crops (False)
            save_json: Whether to save embeddings to JSON file
            
        Returns:
            List of embeddings with metadata
        """
        # Ensure output directory exists
        ensure_dir(output_dir)
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings_for_segmentation_results(
            image=image,
            segmentation_results=segmentation_results,
            apply_masks=apply_masks
        )
        
        # Save embeddings to JSON file
        if save_json and embeddings:
            output_path = os.path.join(output_dir, "embeddings.json")
            self.embedding_generator.save_embeddings_json(
                embeddings=embeddings,
                output_path=output_path,
                include_metadata=True
            )
        
        return embeddings
    
    def run(self, 
          image_path: str, 
          text_queries: Optional[List[str]] = None, 
          visualize: bool = True, 
          save_results: bool = True,
          generate_embeddings: bool = False,
          embedding_model_types: Optional[List[ModelType]] = None):
        """
        Run the complete pipeline
        Args:
            image_path: Path to input image
            text_queries: List of text queries for detection (if None, uses config)
            visualize: Whether to display visualizations
            save_results: Whether to save results
            generate_embeddings: Whether to generate embeddings for segmented objects
            embedding_model_types: List of embedding model types to use (if None, uses config or defaults to ["clip"])
            
        Returns:
            Dictionary with all results
        """
        # Use queries from config if not provided
        if text_queries is None:
            text_queries = self.config["segmentation"]["queries"]
        
        print(f"Running pipeline on {image_path} with queries: {text_queries}")
        
        # Step 1: Object Detection
        detection_results = self.detector.detect(image_path, text_queries)
        
        # Create paths for saving results
        output_base_dir = os.path.join(self.results_dir, os.path.basename(image_path).split('.')[0])
        ensure_dir(output_base_dir)
        
        vis_dir = os.path.join(output_base_dir, "visualizations")
        crops_dir = os.path.join(output_base_dir, "crops")
        segmentation_dir = os.path.join(output_base_dir, "segmentation")
        embeddings_dir = os.path.join(output_base_dir, "embeddings")
        
        if save_results:
            ensure_dir(vis_dir)
            ensure_dir(crops_dir)
            ensure_dir(segmentation_dir)
            if generate_embeddings:
                ensure_dir(embeddings_dir)
        
        # Step 2: Save detection results if requested
        if save_results and detection_results["boxes"]:
            # Save detection visualization
            detection_vis_path = os.path.join(vis_dir, "detection_results.png")
            visualize_detections(
                detection_results["image"],
                detection_results["boxes"],
                detection_results["scores"],
                detection_results["labels"],
                detection_results["text_queries"],
                output_path=detection_vis_path
            )
            
            # Save cropped images
            crop_paths = save_crops(
                detection_results["image"], 
                detection_results["boxes"],
                crops_dir
            )
            print(f"Saved {len(crop_paths)} cropped images to {crops_dir}")
            
            # Also save using the detector's built-in save method
            self.detector.save_results(detection_results, output_base_dir)
        
        # Step 3: Segmentation
        if not detection_results["boxes"]:
            print("No objects detected. Skipping segmentation.")
            return {
                "detection": detection_results,
                "point_segmentation": [],
                "box_segmentation": [],
                "output_dir": output_base_dir
            }
        
        # Convert PIL image to numpy array
        image_np = np.array(detection_results["image"])
        
        # Run both types of segmentation
        point_segmentation = self.segmenter.segment_with_points(
            image_np, detection_results["boxes"]
        )
        box_segmentation = self.segmenter.segment_with_boxes(
            image_np, detection_results["boxes"]
        )
        
        # Step 4: Save segmentation results if requested
        if save_results:
            # Create segmentation subdirectories
            point_seg_dir = os.path.join(segmentation_dir, "point_segmentation")
            box_seg_dir = os.path.join(segmentation_dir, "box_segmentation")
            ensure_dir(point_seg_dir)
            ensure_dir(box_seg_dir)
            
            # Save point-based segmentation results
            point_paths = self.save_segmentation_results(
                image_np, 
                point_segmentation, 
                point_seg_dir, 
                prefix="point_seg"
            )
            print(f"Saved {len(point_paths)} point-based segmentation results to {point_seg_dir}")
            
            # Save box-based segmentation results
            box_paths = self.save_segmentation_results(
                image_np, 
                box_segmentation, 
                box_seg_dir, 
                prefix="box_seg"
            )
            print(f"Saved {len(box_paths)} box-based segmentation results to {box_seg_dir}")
        
        # Step 5: Generate embeddings if requested
        embeddings_results = {}
        if generate_embeddings:
            # Set up embedding model types
            if embedding_model_types is not None:
                self._embedding_generator = EmbeddingGenerator(model_types=embedding_model_types)
            
            # Generate embeddings for point-based segmentation
            point_embeddings = self.generate_embeddings(
                image=image_np,
                segmentation_results=point_segmentation,
                output_dir=os.path.join(embeddings_dir, "point_embeddings"),
                apply_masks=True,
                save_json=save_results
            )
            
            # Generate embeddings for box-based segmentation
            box_embeddings = self.generate_embeddings(
                image=image_np,
                segmentation_results=box_segmentation,
                output_dir=os.path.join(embeddings_dir, "box_embeddings"),
                apply_masks=True,
                save_json=save_results
            )
            
            # Store results
            embeddings_results = {
                "point_embeddings": point_embeddings,
                "box_embeddings": box_embeddings
            }
            
            print(f"Generated {len(point_embeddings)} point-based and {len(box_embeddings)} box-based embeddings")
        
        # Return all results
        results = {
            "detection": detection_results,
            "point_segmentation": point_segmentation,
            "box_segmentation": box_segmentation,
            "output_dir": output_base_dir
        }
        
        # Add embeddings if generated
        if generate_embeddings:
            results["embeddings"] = embeddings_results
        
        return results


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run object detection and segmentation pipeline")
    parser.add_argument("--config", type=str, default="e2e_pipeline_v2/config.yaml", 
                        help="Path to config YAML file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--queries", type=str, nargs="+", 
                        help="Text queries for detection (overrides config)")
    parser.add_argument("--no-viz", action="store_true", 
                        help="Disable visualization")
    parser.add_argument("--no-save", action="store_true", 
                        help="Disable saving results")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DetectionSegmentationPipeline(args.config)
    
    # Run pipeline
    results = pipeline.run(
        image_path=args.image,
        text_queries=args.queries,
        visualize=not args.no_viz,
        save_results=not args.no_save
    )
    
    print(f"Pipeline completed successfully!")
    if not args.no_save:
        print(f"Results saved to: {results['output_dir']}") 