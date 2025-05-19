import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scene_detector import detect_scenes
import re

def natural_sort_key(s):
    """Helper function for natural sorting of strings containing numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def load_frame(frame_path):
    """Load and resize frame for display"""
    img = cv2.imread(str(frame_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_timeline_plot(scene_starts, total_frames):
    """Create a timeline visualization of scene starts"""
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Plot timeline
    ax.plot([0, total_frames], [0, 0], 'k-', alpha=0.3)
    
    # Plot scene starts
    for start in scene_starts:
        ax.axvline(x=start, color='r', alpha=0.5)
        ax.text(start, 0.1, f'Scene {start}', rotation=45)
    
    ax.set_xlim(0, total_frames)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Frame Number')
    ax.set_title('Scene Timeline')
    
    return fig

def display_scene_context(frame_files, scene_start, is_first_scene=False):
    """Display frames around a scene boundary"""
    # Get frame numbers from filenames
    frame_numbers = [int(f.stem) for f in frame_files]
    
    if is_first_scene:
        # For first scene, show first 3 frames
        frames_to_show = frame_files[:3]
        display_numbers = frame_numbers[:3]
    else:
        # For other scenes, show 3 frames before and after
        # Find the index of the scene start frame
        start_idx = frame_numbers.index(scene_start)
        # Get 3 frames before and after
        start_display = max(0, start_idx - 3)
        end_display = min(len(frame_files), start_idx + 3)
        frames_to_show = frame_files[start_display:end_display]
        display_numbers = frame_numbers[start_display:end_display]
    
    # Create columns for the frames
    cols = st.columns(len(frames_to_show))
    
    # Display each frame with its number
    for col, (frame_path, frame_num) in enumerate(zip(frames_to_show, display_numbers)):
        with cols[col]:
            frame = load_frame(frame_path)
            st.image(frame, caption=f"Frame {frame_num}")
            if frame_num == scene_start and not is_first_scene:
                st.markdown("**Scene Boundary**")

def get_scene_directories(base_dir):
    """Get all scene directories from the base directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    # Get all directories that contain image files
    scene_dirs = []
    for d in base_path.iterdir():
        if d.is_dir():
            # Check if directory contains image files
            if list(d.glob("*.jpg")) or list(d.glob("*.png")):
                scene_dirs.append(str(d))
    
    # Sort using natural sort (handles numbers in strings properly)
    return sorted(scene_dirs, key=natural_sort_key)

def main():
    st.set_page_config(page_title="Scene Detection Visualizer", layout="wide")
    st.title("Scene Detection Visualizer")
    
    # Base directory for all scenes
    base_dir = "data/frames/"
    
    # Get available scene directories
    scene_dirs = get_scene_directories(base_dir)
    
    if not scene_dirs:
        st.error("No scene directories found!")
        return
    
    # Scene directory selection
    selected_dir = st.selectbox(
        "Select Scene Directory",
        scene_dirs,
        format_func=lambda x: Path(x).name
    )
    
    # Sidebar controls
    st.sidebar.header("Detection Parameters")
    fps = st.sidebar.slider("FPS", 1, 60, 24)
    detector = st.sidebar.selectbox("Detector", ["adaptive", "content", "threshold"])
    
    # Show relevant parameters based on detector type
    if detector == "adaptive":
        adaptive_threshold = st.sidebar.slider("Adaptive Threshold", 0.01, 2.0, 0.33, 0.01, 
                                help="Controls how quickly the detector adapts to changes (0.0-1.0)")
        min_content_val = st.sidebar.slider("Min Content Value", 1.0, 20.0, 8.0, 0.5,
                                help="Minimum content value to trigger a scene cut")
        window_width = st.sidebar.slider("Window Width", 1, 15, 2, 1,
                                help="Number of frames to average for adaptive detection")
        threshold = 27  # Default value, not used by adaptive detector
    else:
        threshold = st.sidebar.slider("Threshold", 1, 100, 27,
                                    help="Detection sensitivity (lower = more sensitive)")
        adaptive_threshold = 0.33  # Default value, only used by adaptive detector
        min_content_val = 8.0  # Default value
        window_width = 2  # Default value
    
    min_scene_len = st.sidebar.slider("Min Scene Length", 1, 100, 15, 
                                    help="Minimum number of frames a scene must contain")
    
    if st.button("Detect Scenes"):
        if not Path(selected_dir).exists():
            st.error("Selected directory does not exist!")
            return
            
        # Get frame files and sort them naturally
        frame_files = sorted(
            list(Path(selected_dir).glob("*.jpg")) + 
            list(Path(selected_dir).glob("*.png")),
            key=natural_sort_key
        )
        
        if not frame_files:
            st.error("No frame files found!")
            return
            
        # Detect scenes
        scene_starts = detect_scenes(
            frames_dir=selected_dir,
            fps=fps,
            detector=detector,
            threshold=threshold,
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len,
            min_content_val=min_content_val,
            window_width=window_width
        )
        
        # Display results
        st.header("Detection Results")
        
        # Show timeline
        st.pyplot(create_timeline_plot(scene_starts, len(frame_files)))
        
        # Show scene information with context
        st.subheader("Scene Information")
        for i, start in enumerate(scene_starts):
            end = scene_starts[i + 1] if i < len(scene_starts) - 1 else len(frame_files)
            duration = end - start
            
            st.markdown(f"### Scene {i+1}: Frames {start}-{end-1} (Duration: {duration} frames)")
            
            # Display frames around scene boundary
            display_scene_context(frame_files, start, is_first_scene=(i == 0))
            st.markdown("---")  # Add separator between scenes
        
        # Show all scene starts
        st.subheader("Scene Start Indices")
        st.write(scene_starts)

if __name__ == "__main__":
    main() 