#!/usr/bin/env python3
"""
Test script for the detection module.
This script tests object detection on an image using different model types.
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from e2e_pipeline_v2.modules.detection import ObjectDetector

def test_detection(image_path, queries=None, model_types=None, output_dir=None, threshold=0.1, min_score=0.1):
    """
    Test object detection on an image
    
    Args:
        image_path: Path to the image
        queries: List of text queries for detection
        model_types: List of model types to test
        output_dir: Base directory to save results
        threshold: Detection confidence threshold
        min_score: Minimum score for saving crops
    """
    if queries is None:
        queries = ["hammer"]
    
    if model_types is None:
        model_types = ["owlvit", "owlvit2"]
    
    if output_dir is None:
        output_dir = "results/detectors"
    
    # Test each model type
    for model_type in model_types:
        print(f"\n=== Testing {model_type} detector ===")
        
        try:
            # Configure the detector
            detector_config = {
                "model_type": model_type,
                "threshold": threshold,
                "force_cpu": True  # Force CPU to avoid MPS float64 error
            }
            
            # Initialize the detector
            detector = ObjectDetector(detector_config)
            
            # Run detection
            results = detector.detect(image_path, queries)
            
            # Configure saving options
            save_options = {
                "save_original": True,
                "save_visualizations": True,
                "save_crops": True,
                "save_metadata": True,
                "min_score": min_score
            }
            
            # Save results with improved structure
            saved_results = detector.save_results(results, output_dir, save_options)
            
            # Print results
            print(f"\nDetection results for {model_type} ({results['model_name'].split('/')[-1]}):")
            print(f"Found {len(results['boxes'])} objects")
            
            if "metadata" in saved_results:
                print(f"Results saved to: {saved_results['base_dir']}")
                print(f"Metadata file: {os.path.basename(saved_results['metadata'])}")
            
            if "visualization" in saved_results:
                print(f"Visualization: {os.path.relpath(saved_results['visualization'], start=saved_results['base_dir'])}")
            
            if "crops" in saved_results and saved_results["crops"]:
                crop_dir = os.path.dirname(saved_results["crops"][0])
                print(f"Crops directory: {os.path.relpath(crop_dir, start=saved_results['base_dir'])}")
                print(f"Saved {len(saved_results['crops'])} crops (threshold: {min_score})")
                
                # Print top 5 crops by score
                if "detections" in saved_results:
                    detections = sorted(saved_results["detections"], key=lambda x: x["score"], reverse=True)
                    print("\nTop detections:")
                    for i, detection in enumerate(detections[:5]):
                        print(f"  {i+1}. {detection['label']} (score: {detection['score']:.3f}, id: {detection['id']})")
                    
                    if len(detections) > 5:
                        print(f"  ... and {len(detections)-5} more")
        
        except Exception as e:
            print(f"Error testing {model_type} detector: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Test object detection")
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Path to the image file")
    parser.add_argument("--queries", type=str, nargs="+", default=["hammer"],
                        help="Text queries for detection")
    parser.add_argument("--model_types", type=str, nargs="+", 
                        choices=["owlvit", "owlvit2"],
                        default=["owlvit", "owlvit2"],
                        help="Model types to test")
    parser.add_argument("--output_dir", type=str, default="results/detectors",
                        help="Base directory to save results")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Detection confidence threshold")
    parser.add_argument("--min_score", type=float, default=0.01,
                        help="Minimum score for saving crops")
    
    args = parser.parse_args()
    
    # Verify the image file exists
    if not os.path.isfile(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return 1
    
    # Run the test
    test_detection(
        args.image_path, 
        args.queries, 
        args.model_types, 
        args.output_dir,
        args.threshold,
        args.min_score
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 