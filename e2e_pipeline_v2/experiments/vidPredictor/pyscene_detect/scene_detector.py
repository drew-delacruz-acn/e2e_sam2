#!/usr/bin/env python3
"""scene_detector.py
Detect scene boundaries in a directory of frame images.

This script provides a function to detect scene boundaries in a sequence of frames
and can be used both as a module and as a standalone script.

Example usage as a module:
    from scene_detector import detect_scenes
    scenes = detect_scenes("/path/to/frames", fps=30, detector="adaptive")

Example usage as a script:
    python scene_detector.py
"""

import os
from pathlib import Path
from scenedetect import SceneManager, VideoManager, open_video
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector

def detect_scenes(frames_dir, fps=30, detector="adaptive", threshold=27, 
                 sigma=0.33, min_scene_len=15, ext=".jpg"):
    """
    Detect scene boundaries in a directory of frame images.
    
    Args:
        frames_dir (str or Path): Directory containing frame images
        fps (float): Frames per second (for timecode calculations)
        detector (str): Detection method ("adaptive", "content", or "threshold")
        threshold (float): Detection threshold (for content/threshold detectors)
        sigma (float): Sigma factor for adaptive detector (0.0-1.0)
        min_scene_len (int): Minimum scene length in frames
        ext (str): Image file extension (e.g., ".jpg" or ".png")
    Returns:
        list: List of tuples containing (start_frame, end_frame) for each scene
    Raises:
        ValueError: If frames_dir doesn't exist or is not a directory
        RuntimeError: If no frames are found or scene detection fails
    """
    # Convert to Path object if string
    frames_dir = Path(frames_dir)
    
    # Validate input directory
    if not frames_dir.is_dir():
        raise ValueError(f"Frames directory does not exist: {frames_dir}")
    
    # Check for frame files
    frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
    if not frame_files:
        raise RuntimeError(f"No frame files found in {frames_dir}")
    
    print(f"Found {len(frame_files)} frames in {frames_dir}")
    
    # Build OpenCV-style pattern for image sequence
    ext = ext if ext.startswith('.') else f'.{ext}'
    pattern = str(frames_dir / f"%d{ext}")
    print(f"Using image sequence pattern: {pattern}")
    
    # Initialize detector
    detector = detector.lower()
    if detector == "adaptive":
        detector = AdaptiveDetector(min_scene_len=min_scene_len)
    elif detector == "content":
        detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    elif detector == "threshold":
        detector = ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len)
    else:
        raise ValueError(f"Unknown detector type: {detector}")
    
    # Create scene manager and add detector
    scene_manager = SceneManager()
    scene_manager.add_detector(detector)
    
    # Open the image sequence as a video
    video = open_video(pattern, framerate=fps)
    
    # Detect scenes
    print("Detecting scenes...")
    scene_manager.detect_scenes(video, show_progress=True)
    scenes = scene_manager.get_scene_list()
    
    if not scenes:
        print("No scenes detected!")
        return []
    
    # Create a list of start indices for each scene, including the first (0)
    scene_start_indices = [0] + [scene[0].get_frames() for scene in scenes[1:]]
    
    # Print summary
    print(f"\nScene start indices: {scene_start_indices}")
    
    return scene_start_indices

if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    frames_dir = "/Users/andrewdelacruz/e2e_sam2/gitignore_exception/data/frames/Scenes 061-080__265H-2-_20230815215828529"  # Change this to your frames directory
    fps = 24
    detector = "threshold"
    threshold = 27
    sigma = 0.33
    min_scene_len = 15
    ext = ".jpg"
    
    # Run detection
    scenes = detect_scenes(
        frames_dir=frames_dir,
        fps=fps,
        detector=detector,
        threshold=threshold,
        sigma=sigma,
        min_scene_len=min_scene_len,
        ext=ext
    ) 

    print(scenes)