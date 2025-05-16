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
import json

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

def dump_debug_info(data, filename, output_dir):
    """Dump debug information to a JSON file"""
    debug_path = os.path.join(output_dir, filename)
    with open(debug_path, 'w') as f:
        if isinstance(data, dict):
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    serializable_data[k] = {}
                    for sub_k, sub_v in v.items():
                        if hasattr(sub_v, 'tolist'):
                            serializable_data[k][sub_k] = sub_v.tolist()
                        else:
                            serializable_data[k][sub_k] = sub_v
                else:
                    serializable_data[k] = v
            json.dump(serializable_data, f, indent=2)
        else:
            json.dump(data, f, indent=2)
    print(f"Debug info saved to {debug_path}")

def main():
    """Main function to run the pipeline"""
    # Set environment variables
    set_env_variables()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create results directory
    results_dir = ensure_dir(args.results_dir)
    vis_dir = ensure_dir(os.path.join(results_dir, "visualizations"))
    debug_dir = ensure_dir(os.path.join(results_dir, "debug")) if args.debug else None
    json_path = os.path.join(results_dir, "segmentation_results.json")
    
    print("\n=== VIDEO SEGMENTATION PIPELINE START ===")
    print(f"Detection files directory: {args.detections_dir}")
    print(f"Frames directory: {args.frames_dir}")
    print(f"Results will be saved to: {results_dir}")
    
    # Load and process detections
    print("\n=== STEP 1: Loading and processing detections ===")
    detections_by_frame = load_detection_files(args.detections_dir)
    
    # Debug: Save detection data
    if args.debug:
        print(f"Loaded detections for {len(detections_by_frame)} frames")
        dump_debug_info(detections_by_frame, "detections_by_frame.json", debug_dir)
    
    print(f"Filtering detections with confidence threshold {args.confidence_threshold}...")
    filtered_detections = filter_detections_by_confidence(
        detections_by_frame, confidence_threshold=args.confidence_threshold
    )
    
    # Debug: Count detections before and after filtering
    if args.debug:
        total_before = sum(len(dets) for dets in detections_by_frame.values())
        total_after = sum(len(dets) for dets in filtered_detections.values())
        print(f"Filtered detections: {total_before} -> {total_after} ({total_after/total_before:.1%} kept)")
        dump_debug_info(filtered_detections, "filtered_detections.json", debug_dir)
    
    print("Converting detections to tracking format...")
    tracking_objects = convert_detections_to_tracking_format(filtered_detections)
    
    if args.debug or True:  # Always show this information
        print(f"Found {len(tracking_objects)} tracking objects:")
        for i, obj in enumerate(tracking_objects):
            frame_nums = [occ['frameNum'] for occ in obj['frameOccurences']]
            print(f"  {i+1}. ID: {obj['objectID']}, Name: {obj['objectName']}, "
                  f"Occurrences: {len(obj['frameOccurences'])}, "
                  f"Frame range: {min(frame_nums)}-{max(frame_nums)}")
        
        dump_debug_info(tracking_objects, "tracking_objects.json", debug_dir)
    
    # Get frame names
    print("\n=== STEP 2: Processing frames ===")
    frame_names = get_frame_names(args.frames_dir)
    print(f"Found {len(frame_names)} frames")
    
    # Initialize SAM2 segmenter
    print("\n=== STEP 3: Initializing SAM2 segmenter ===")
    segmenter = SAM2VideoSegmenter(args.model_cfg, args.sam2_checkpoint)
    
    # Initialize video
    print("Setting up video inference...")
    inference_state = segmenter.initialize_video(args.frames_dir)
    
    # Get unique frame numbers from tracking objects
    frame_nums = get_unique_frame_numbers(tracking_objects)
    print(f"Processing {len(frame_nums)} unique frames with detections")
    print(f"Frame range: {min(frame_nums)}-{max(frame_nums)}")
    
    # Process tracking objects with SAM2
    print("\n=== STEP 4: Processing objects with SAM2 ===")
    segmenter.process_tracking_objects(tracking_objects, inference_state, frame_nums)
    
    # Propagate segmentation through video
    print("\n=== STEP 5: Propagating segmentation ===")
    video_segments = segmenter.propagate_segmentation(inference_state)
    
    # Validate video segments
    if not video_segments:
        print("ERROR: No video segments were generated! Segmentation failed.")
        return
    
    # Debug: Save segment data
    if args.debug:
        # We can't directly save the numpy arrays, so let's save frame and object counts
        segment_counts = {
            frame_idx: len(segments) for frame_idx, segments in video_segments.items()
        }
        dump_debug_info(segment_counts, "segment_counts.json", debug_dir)
        
        # Also log which objects appear in which frames
        object_appearances = {}
        for frame_idx, segments in video_segments.items():
            for obj_id in segments.keys():
                if obj_id not in object_appearances:
                    object_appearances[obj_id] = []
                object_appearances[obj_id].append(frame_idx)
        
        dump_debug_info(object_appearances, "object_appearances.json", debug_dir)
        print(f"Found {len(object_appearances)} objects with segment appearances")
    
    # Match detections to segments
    print("\n=== STEP 6: Matching detections to segments ===")
    object_detections = match_detections_to_segments(
        video_segments, 
        filtered_detections, 
        tracking_objects,
        iou_threshold=args.iou_threshold
    )
    
    # Validate object detections
    if not object_detections:
        print("ERROR: No object detections were found! Matching failed.")
    else:
        print(f"Found {len(object_detections)} objects with detections")
        
        # Debug: Save object detections
        if args.debug:
            dump_debug_info(object_detections, "object_detections.json", debug_dir)
    
    # Create summary of segmentation results
    print("\n=== STEP 7: Creating segmentation summary ===")
    results = create_segmentation_summary(video_segments, tracking_objects, object_detections)
    
    # Validate results
    if not results:
        print("ERROR: No results were generated! Summary creation failed.")
    else:
        print(f"Created summary for {len(results)} objects")
        
        # Debug: Check for missing sam dictated appearances
        if args.debug:
            for i, obj in enumerate(results):
                obj_id = obj['samObjectId']
                n_appearances = len(obj['samDictatedAppearances'])
                n_predictions = len(obj['cdfsodPredictions'])
                
                # Log warning if no appearances but has predictions
                if n_appearances == 0 and n_predictions > 0:
                    print(f"WARNING: Object {obj_id} has {n_predictions} CDFSOD predictions but no SAM2 appearances!")
    
    # Save results to JSON
    print(f"\n=== STEP 8: Saving results to {json_path} ===")
    save_results_to_json(results, json_path)
    
    # Visualize results
    if not args.no_vis:
        print("\n=== STEP 9: Visualizing segmentation results ===")
        figures = visualize_segmentation_results(
            args.frames_dir, 
            frame_names, 
            video_segments, 
            vis_frame_stride=args.vis_stride,
            save_path=vis_dir
        )
        
        print(f"Processed {len(frame_names)} frames with {len(tracking_objects)} object classes")
        print(f"Visualization saved to {vis_dir}")
    
    print(f"\nAll results saved to {results_dir}")
    print("=== VIDEO SEGMENTATION PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main() 