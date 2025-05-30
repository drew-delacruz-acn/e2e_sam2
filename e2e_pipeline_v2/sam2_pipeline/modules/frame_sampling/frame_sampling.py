import cv2
import numpy as np
import random
import os
from pathlib import Path
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

class VideoFrameSampler:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def uniform_sampling(self, interval=30):
        """
        Sample frames uniformly every 'interval' frames.
        """
        sampled_frames = []
        for idx in range(0, self.frame_count, interval):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
            else:
                break
        return sampled_frames

    def random_sampling(self, num_frames=10):
        """
        Randomly sample 'num_frames' frames from the video.
        """
        sampled_frames = []
        frame_indices = sorted(random.sample(range(self.frame_count), min(num_frames, self.frame_count)))
        for idx in frame_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
        return sampled_frames

    def sequential_sampling(self, start_frame=0, end_frame=None, step=1):
        """
        Sample frames sequentially from 'start_frame' to 'end_frame' with a given 'step'.
        """
        if end_frame is None:
            end_frame = self.frame_count

        sampled_frames = []
        for idx in range(start_frame, min(end_frame, self.frame_count), step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
            else:
                break
        return sampled_frames

    def scene_based_sampling(self, threshold=30.0):
        """
        Use PySceneDetect to detect scene boundaries, then sample one frame (the middle frame) from each scene.
        Adjust 'threshold' for sensitivity.
        """
        # Create a new SceneManager and add the ContentDetector
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        # Use open_video from PySceneDetect
        video_stream = open_video(self.video_path)

        # Perform scene detection
        scene_manager.detect_scenes(video=video_stream)
        scene_list = scene_manager.get_scene_list()

        sampled_frames = []
        for scene in scene_list:
            start_time, end_time = scene
            # Convert timecode to frame index
            start_frame = start_time.get_frames()
            end_frame = end_time.get_frames()

            # Choose the midpoint frame
            mid_frame = (start_frame + end_frame) // 2

            # Read that midpoint frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = self.cap.read()
            if ret:
                sampled_frames.append(frame)
        return sampled_frames

    def release(self):
        self.cap.release()


def save_frames(frames, output_dir, base_filename):
    """
    Save frames as images to the specified directory.
    
    Args:
        frames: List of frames (numpy arrays) to save
        output_dir: Directory to save the frames
        base_filename: Base filename for the saved frames
        
    Returns:
        List of paths to the saved images
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame as an image
    frame_paths = []
    for i, frame in enumerate(frames):
        # Generate a filename
        frame_path = os.path.join(output_dir, f"{base_filename}_frame_{i:04d}.jpg")
        
        # Save the frame
        cv2.imwrite(frame_path, frame)
        
        # Add the path to the list
        frame_paths.append(frame_path)
    
    return frame_paths


def sample_frames_from_video(video_config):
    """
    Sample frames from a video based on the configuration.
    
    Args:
        video_config: Configuration dictionary with video path and sampling parameters
        
    Returns:
        List of paths to the sampled frames
    """
    # Extract parameters from the config
    video_path = video_config["video_path"]
    frame_sampling_config = video_config.get("frame_sampling", {})
    
    # Extract the sampling method and parameters
    method = frame_sampling_config.get("method", "uniform")
    params = frame_sampling_config.get("params", {})
    
    # Create the output directory
    output_dir = Path("results/frame_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base filename for the sampled frames
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Initialize the frame sampler
    print(f"Sampling frames from {video_path} using {method} method...")
    sampler = VideoFrameSampler(video_path)
    
    try:
        # Sample frames based on the selected method
        if method == "uniform":
            interval = params.get("interval", 30)
            frames = sampler.uniform_sampling(interval=interval)
            print(f"Uniformly sampled {len(frames)} frames with interval {interval}")
        
        elif method == "random":
            num_frames = params.get("num_frames", 10)
            frames = sampler.random_sampling(num_frames=num_frames)
            print(f"Randomly sampled {len(frames)} frames")
        
        elif method == "sequential":
            start_frame = params.get("start_frame", 0)
            end_frame = params.get("end_frame")
            step = params.get("step", 1)
            frames = sampler.sequential_sampling(
                start_frame=start_frame,
                end_frame=end_frame,
                step=step
            )
            print(f"Sequentially sampled {len(frames)} frames from {start_frame} to {end_frame} with step {step}")
        
        elif method == "scene":
            threshold = params.get("threshold", 30.0)
            frames = sampler.scene_based_sampling(threshold=threshold)
            print(f"Scene-based sampling detected {len(frames)} scenes with threshold {threshold}")
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Save the frames as images
        frame_paths = save_frames(frames, output_dir, base_filename)
        print(f"Saved {len(frame_paths)} frames to {output_dir}")
        
        return frame_paths
    
    finally:
        # Always release the video capture
        sampler.release()


# Example usage
if __name__ == "__main__":
    # Example configuration
    test_config = {
        "video_path": "data/test.mp4",
        "frame_sampling": {
            "method": "scene",
            "params": {
                "threshold": 30.0
            }
        }
    }
    
    # Sample frames
    frame_paths = sample_frames_from_video(test_config)
    
    # Print the sampled frame paths
    for path in frame_paths:
        print(path)
