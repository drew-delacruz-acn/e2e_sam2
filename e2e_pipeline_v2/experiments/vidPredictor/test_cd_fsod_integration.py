#!/usr/bin/env python3
"""
Test script for CD-FSOD detector integration with the object tracking pipeline.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from object_tracking_pipeline import ObjectTrackingPipeline

def main():
    parser = argparse.ArgumentParser(description="Test CD-FSOD detector integration with the object tracking pipeline")
    
    # Data paths
    parser.add_argument("--frames-dir", required=True, help="Directory containing video frames")
    parser.add_argument("--cd-fsod-path", required=True, help="Path to CD-FSOD JSON detections directory")
    parser.add_argument("--output-dir", default="./cd_fsod_results", help="Output directory for results")
    
    # Model paths
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", required=True, help="Path to SAM2 config file")
    
    # Detection parameters
    parser.add_argument("--text-queries", default=["all"], nargs="+", help="Text queries for object detection (default: 'all')")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--min-gap-frames", type=int, default=10, help="Minimum gap frames for CD-FSOD reappearances")
    
    # Processing options
    parser.add_argument("--separate-objects", action="store_true", help="Process each object separately to avoid dtype issues")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Testing CD-FSOD integration with the following settings:")
    print(f"  - Frames directory: {args.frames_dir}")
    print(f"  - CD-FSOD detections directory: {args.cd_fsod_path}")
    print(f"  - Confidence threshold: {args.confidence}")
    print(f"  - Minimum gap frames: {args.min_gap_frames}")
    print(f"  - Text queries: {args.text_queries}")
    print(f"  - Using separate objects: {args.separate_objects}")
    
    # Initialize pipeline with CD-FSOD detector
    pipeline = ObjectTrackingPipeline(
        owlv2_checkpoint=None,  # Not used with CD-FSOD detector
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        detector_type="cd_fsod",  # Use CD-FSOD detector
        cd_fsod_path=args.cd_fsod_path,
        min_gap_frames=args.min_gap_frames
    )
    
    # Process video using the appropriate method
    if args.separate_objects:
        print("Processing video with separate object initialization...")
        pipeline.process_video_separate_objects(
            frames_dir=args.frames_dir,
            text_queries=args.text_queries
        )
    else:
        print("Processing video with standard method...")
        pipeline.process_video(
            frames_dir=args.frames_dir,
            text_queries=args.text_queries
        )
    
    print(f"Processing complete. Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 