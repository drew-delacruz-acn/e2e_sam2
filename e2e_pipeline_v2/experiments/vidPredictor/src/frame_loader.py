# frame_loader.py
import os
import numpy as np
from PIL import Image
import glob

def load_frames_from_directory(video_dir, pattern="*.jpg"):
    """Load frames from a directory of images"""
    # Find all matching frames
    frame_paths = sorted(glob.glob(os.path.join(video_dir, pattern)))
    
    # Load frames
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(np.array(img))
    
    print(f"Loaded {len(frames)} frames from {video_dir}")
    return frames, frame_paths