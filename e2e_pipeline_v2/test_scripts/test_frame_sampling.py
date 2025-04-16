#!/usr/bin/env python3
"""
Test script for the frame_sampling module.
This script tests different frame sampling methods on a sample video.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from e2e_pipeline_v2.modules.frame_sampling import sample_frames_from_video

def test_frame_sampling(video_path, methods=None, output_dir=None):
    """
    Test different frame sampling methods on a video.
    
    Args:
        video_path: Path to the video file
        methods: List of methods to test, or None for all methods
        output_dir: Directory to save the sampled frames
    """
    if methods is None:
        methods = ["uniform", "random", "sequential", "scene"]
    
    if output_dir is None:
        output_dir = "results/frame_samples"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test each method
    for method in methods:
        print(f"\n=== Testing {method} sampling method ===")
        
        # Create a configuration for this method
        if method == "uniform":
            config = {
                "video_path": video_path,
                "frame_sampling": {
                    "method": method,
                    "params": {
                        "interval": 30
                    }
                }
            }
        elif method == "random":
            config = {
                "video_path": video_path,
                "frame_sampling": {
                    "method": method,
                    "params": {
                        "num_frames": 10
                    }
                }
            }
        elif method == "sequential":
            config = {
                "video_path": video_path,
                "frame_sampling": {
                    "method": method,
                    "params": {
                        "start_frame": 0,
                        "end_frame": 300,
                        "step": 30
                    }
                }
            }
        elif method == "scene":
            config = {
                "video_path": video_path,
                "frame_sampling": {
                    "method": method,
                    "params": {
                        "threshold": 30.0
                    }
                }
            }
        else:
            print(f"Unknown method: {method}")
            continue
        
        # Sample frames using this method
        try:
            frame_paths = sample_frames_from_video(config)
            
            # Print the results
            print(f"Sampled {len(frame_paths)} frames using {method} method")
            print("Sample frame paths:")
            for i, path in enumerate(frame_paths[:5]):
                print(f"  {i+1}. {path}")
            if len(frame_paths) > 5:
                print(f"  ... and {len(frame_paths)-5} more")
        except Exception as e:
            print(f"Error sampling frames with {method} method: {e}")
    
    print("\nFrame sampling tests completed")

def main():
    parser = argparse.ArgumentParser(description="Test frame sampling methods")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to the video file to sample")
    parser.add_argument("--methods", type=str, nargs="+", 
                        choices=["uniform", "random", "sequential", "scene"],
                        help="Methods to test (default: all)")
    parser.add_argument("--output_dir", type=str, default="results/frame_samples",
                        help="Directory to save the sampled frames")
    
    args = parser.parse_args()
    
    # Verify the video file exists
    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Run the tests
    test_frame_sampling(args.video_path, args.methods, args.output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 