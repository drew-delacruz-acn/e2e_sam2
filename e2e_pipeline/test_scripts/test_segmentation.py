"""
Test script for the segmentation module.
"""
import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from e2e_pipeline.modules.segmentation import segment_image

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test segmentation module')
    parser.add_argument('--image_path', type=str, default='/Users/andrewdelacruz/e2e_sam2/data/thor_hammer.jpeg',
                      help='Path to image file')
    parser.add_argument('--queries', type=str, nargs='+', default=['hammer'],
                      help='Text queries for object detection')
    parser.add_argument('--detection_threshold', type=float, default=0.1,
                      help='Threshold for object detection')
    parser.add_argument('--segmentation_threshold', type=float, default=0.5,
                      help='Threshold for segmentation')
    parser.add_argument('--results_dir', type=str, default='results/test_segmentation',
                      help='Directory to save results')
    args = parser.parse_args()
    
    # Configuration for testing
    test_config = {
        "queries": args.queries,
        "detection_threshold": args.detection_threshold,
        "segmentation_threshold": args.segmentation_threshold,
        "results_dir": args.results_dir
    }
    
    # Print configuration
    print(f"Testing segmentation with configuration:")
    print(f"  Image path: {args.image_path}")
    print(f"  Queries: {test_config['queries']}")
    print(f"  Detection threshold: {test_config['detection_threshold']}")
    print(f"  Segmentation threshold: {test_config['segmentation_threshold']}")
    print(f"  Results directory: {test_config['results_dir']}")
    
    # Run the segmentation
    try:
        crop_infos = segment_image(args.image_path, test_config)
        print(f"\nSuccess! Segmented {len(crop_infos)} objects.")
        
        if crop_infos:
            print("\nSegmentation results:")
            for i, crop_info in enumerate(crop_infos):
                print(f"Object {i+1}:")
                print(f"  Query: {crop_info['query']}")
                print(f"  Score: {crop_info['score']:.4f}")
                print(f"  Bounding box: {crop_info['bbox']}")
                print(f"  Crop saved to: {crop_info['crop_path']}")
                print()
            
            print(f"Results saved to: {args.results_dir}")
        else:
            print("No objects detected in the image.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 