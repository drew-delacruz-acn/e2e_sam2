#!/usr/bin/env python3
"""
SAM2 Video Segmentation Pipeline
--------------------------------
This script takes detection files and video frames to run segmentation 
with SAM2 and visualize the results.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt

# Import modules
from src.utils import set_env_variables, get_frame_names, get_unique_frame_numbers
from src.detection_processor import (
    load_detection_files, 
    filter_detections_by_confidence,
    convert_detections_to_tracking_format
)
from src.sam2_segmenter import SAM2VideoSegmenter
from src.visualization import visualize_segmentation_results
from src.result_processor import (
    match_detections_to_segments,
    create_segmentation_summary,
    save_results_to_json
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SAM2 Video Segmentation Pipeline')
    parser.add_argument('--detections_dir', type=str, required=True,
                        help='Directory containing detection JSON files')
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing video frames')
    parser.add_argument('--sam2_checkpoint', type=str, required=True,
                        help='Path to SAM2 checkpoint file')
    parser.add_argument('--model_cfg', type=str, required=True,
                        help='Path to SAM2 model configuration file')
    parser.add_argument('--confidence_threshold', type=float, default=0.9,
                        help='Confidence threshold for filtering detections')
    parser.add_argument('--vis_stride', type=int, default=1,
                        help='Stride for visualization (display every nth frame)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save all results (JSON and images)')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for matching detections to segments')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    parser.add_argument('--no_vis', action='store_true',
                        help='Skip visualization generation')
    return parser.parse_args()

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def main():
    """Main function to run the pipeline"""
    # Set environment variables
    set_env_variables()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create results directory
    results_dir = ensure_dir(args.results_dir)
    vis_dir = ensure_dir(os.path.join(results_dir, "visualizations"))
    json_path = os.path.join(results_dir, "segmentation_results.json")
    
    # Load and process detections
    print("Loading detection files...")
    detections_by_frame = load_detection_files(args.detections_dir)
    
    print(f"Filtering detections with confidence threshold {args.confidence_threshold}...")
    filtered_detections = filter_detections_by_confidence(
        detections_by_frame, confidence_threshold=args.confidence_threshold
    )
    
    print("Converting detections to tracking format...")
    tracking_objects = convert_detections_to_tracking_format(filtered_detections)
    
    if args.debug:
        print(f"Found {len(tracking_objects)} tracking objects:")
        for obj in tracking_objects:
            print(f"  ID: {obj['objectID']}, Name: {obj['objectName']}, Occurrences: {len(obj['frameOccurences'])}")
    
    # Get frame names
    frame_names = get_frame_names(args.frames_dir)
    
    # Initialize SAM2 segmenter
    print("Initializing SAM2 segmenter...")
    segmenter = SAM2VideoSegmenter(args.model_cfg, args.sam2_checkpoint)
    
    # Initialize video
    print("Setting up video inference...")
    inference_state = segmenter.initialize_video(args.frames_dir)
    
    # Get unique frame numbers from tracking objects
    frame_nums = get_unique_frame_numbers(tracking_objects)
    
    # Process tracking objects with SAM2
    print("Processing object detections with SAM2...")
    segmenter.process_tracking_objects(tracking_objects, inference_state, frame_nums)
    
    # Propagate segmentation through video
    print("Propagating segmentation through video...")
    video_segments = segmenter.propagate_segmentation(inference_state)
    
    # Match detections to segments
    print("Matching detections to segments...")
    object_detections = match_detections_to_segments(
        video_segments, 
        filtered_detections, 
        tracking_objects,
        iou_threshold=args.iou_threshold
    )
    
    # Create summary of segmentation results
    print("Creating segmentation summary...")
    results = create_segmentation_summary(video_segments, tracking_objects, object_detections)
    
    # Save results to JSON
    print(f"Saving results to {json_path}...")
    save_results_to_json(results, json_path)
    
    # Visualize results
    if not args.no_vis:
        print("Visualizing segmentation results...")
        figures = visualize_segmentation_results(
            args.frames_dir, 
            frame_names, 
            video_segments, 
            vis_frame_stride=args.vis_stride,
            save_path=vis_dir
        )
        
        print(f"Processed {len(frame_names)} frames with {len(tracking_objects)} object classes")
        print(f"Visualization saved to {vis_dir}")
    
    print(f"All results saved to {results_dir}")
    print("Done!")

if __name__ == "__main__":
    main() 