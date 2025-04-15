import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add the root directory to path to fix import issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from FrameSampling.frame_sample import VideoFrameSampler

def sample_frames_from_video(video_config):
    """
    Sample frames from a video based on the provided configuration.
    
    Args:
        video_config (dict): Configuration for video processing with keys:
            - video_path: Path to the video file
            - frame_sampling: Dictionary with method and params
    
    Returns:
        list: List of paths to saved frame images
    """
    video_path = video_config["video_path"]
    sampling_config = video_config["frame_sampling"]
    method = sampling_config["method"]
    
    # Create a results directory for frames if it doesn't exist
    frames_dir = Path("results/frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a VideoFrameSampler instance
    sampler = VideoFrameSampler(video_path)
    
    # Sample frames based on the specified method
    frames = []
    if method == "uniform":
        interval = sampling_config.get("params", {}).get("interval", 30)
        frames = sampler.uniform_sampling(interval=interval)
    elif method == "random":
        num_frames = sampling_config.get("params", {}).get("num_frames", 10)
        frames = sampler.random_sampling(num_frames=num_frames)
    elif method == "sequential":
        start_frame = sampling_config.get("params", {}).get("start_frame", 0)
        end_frame = sampling_config.get("params", {}).get("end_frame", None)
        step = sampling_config.get("params", {}).get("step", 1)
        frames = sampler.sequential_sampling(start_frame=start_frame, end_frame=end_frame, step=step)
    elif method == "scene":
        threshold = sampling_config.get("params", {}).get("threshold", 30.0)
        frames = sampler.scene_based_sampling(threshold=threshold)
    else:
        raise ValueError(f"Unsupported frame sampling method: {method}")
    
    # Save the frames to disk and return their paths
    frame_paths = []
    for i, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
    
    # Release the video capture
    sampler.release()
    
    print(f"Sampled {len(frame_paths)} frames using {method} method")
    return frame_paths

# Test function for this module
def test_frame_sampling():
    """Test function for frame sampling module"""
    from pathlib import Path
    import os
    
    # Create test directory
    test_dir = Path("e2e_pipeline/test_scripts")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configuration
    test_config = {
        "video_path": "data/sample_video.mp4",  # Update this to your actual test video path
        "frame_sampling": {
            "method": "scene",
            "params": {
                "threshold": 30.0
            }
        }
    }
    
    # Run the function
    try:
        frame_paths = sample_frames_from_video(test_config)
        print(f"Test successful! Sampled {len(frame_paths)} frames.")
        print(f"Frame paths: {frame_paths[:5]}...")
        return True
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_frame_sampling() 