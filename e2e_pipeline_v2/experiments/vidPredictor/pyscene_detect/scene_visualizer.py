import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scene_detector import detect_scenes

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
    if is_first_scene:
        # For first scene, show first 3 frames
        frames_to_show = frame_files[:3]
        frame_numbers = list(range(3))
    else:
        # For other scenes, show 3 frames before and after
        start_idx = max(0, scene_start - 3)
        end_idx = min(len(frame_files), scene_start + 3)
        frames_to_show = frame_files[start_idx:end_idx]
        frame_numbers = list(range(start_idx, end_idx))
    
    # Create columns for the frames
    cols = st.columns(len(frames_to_show))
    
    # Display each frame with its number
    for col, (frame_path, frame_num) in enumerate(zip(frames_to_show, frame_numbers)):
        with cols[col]:
            frame = load_frame(frame_path)
            st.image(frame, caption=f"Frame {frame_num}")
            if frame_num == scene_start and not is_first_scene:
                st.markdown("**Scene Boundary**")

def main():
    st.set_page_config(page_title="Scene Detection Visualizer", layout="wide")
    st.title("Scene Detection Visualizer")
    
    # Sidebar controls
    st.sidebar.header("Detection Parameters")
    fps = st.sidebar.slider("FPS", 1, 60, 24)
    detector = st.sidebar.selectbox("Detector", ["adaptive", "content", "threshold"])
    threshold = st.sidebar.slider("Threshold", 1, 100, 27)
    min_scene_len = st.sidebar.slider("Min Scene Length", 1, 100, 15)
    
    # Frame directory input
    frames_dir = st.text_input(
        "Frames Directory",
        "data/frames/Scenes 061-080__265H-2-_20230815215828529"
    )
    
    if st.button("Detect Scenes"):
        if not Path(frames_dir).exists():
            st.error("Frames directory does not exist!")
            return
            
        # Get frame files
        frame_files = sorted(list(Path(frames_dir).glob("*.jpg")) + 
                           list(Path(frames_dir).glob("*.png")))
        
        if not frame_files:
            st.error("No frame files found!")
            return
            
        # Detect scenes
        scene_starts = detect_scenes(
            frames_dir=frames_dir,
            fps=fps,
            detector=detector,
            threshold=threshold,
            min_scene_len=min_scene_len
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