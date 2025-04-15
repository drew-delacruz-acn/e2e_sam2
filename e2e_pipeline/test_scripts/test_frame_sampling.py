"""
Test script for the frame sampling module.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from e2e_pipeline.modules.frame_sampling import sample_frames_from_video

def main():
    # Check if a video path was provided
    import argparse
    parser = argparse.ArgumentParser(description='Test frame sampling module')
    parser.add_argument('--video_path', type=str, default='data/sample_video.mp4',
                      help='Path to video file')
    parser.add_argument('--method', type=str, default='scene',
                      choices=['uniform', 'random', 'sequential', 'scene'],
                      help='Frame sampling method')
    args = parser.parse_args()
    
    # Configuration for testing
    test_config = {
        "video_path": args.video_path,
        "frame_sampling": {
            "method": args.method,
            "params": {
                "threshold": 30.0,    # for scene detection
                "interval": 30,       # for uniform sampling
                "num_frames": 10,     # for random sampling
                "step": 10            # for sequential sampling
            }
        }
    }
    
    # Print configuration
    print(f"Testing frame sampling with configuration:")
    print(f"  Video path: {test_config['video_path']}")
    print(f"  Method: {test_config['frame_sampling']['method']}")
    print(f"  Parameters: {test_config['frame_sampling']['params']}")
    
    # Run the frame sampling
    try:
        frame_paths = sample_frames_from_video(test_config)
        print(f"\nSuccess! Sampled {len(frame_paths)} frames.")
        if frame_paths:
            print(f"First few frame paths:")
            for i, path in enumerate(frame_paths[:5]):
                print(f"  {i+1}. {path}")
            if len(frame_paths) > 5:
                print(f"  ... and {len(frame_paths)-5} more")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 