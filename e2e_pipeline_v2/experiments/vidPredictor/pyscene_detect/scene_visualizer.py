import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scene_detector import detect_scenes
import re
import pandas as pd
import os
import shutil
import matplotlib
import logging
import sys
import math
matplotlib.use('Agg')

# Configure logging to write to both console and file
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_visualizer.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force a test log message on import
logger.info("Scene Visualizer starting - log test")
print("Scene Visualizer starting - print test")

def natural_sort_key(s):
    """Helper function for natural sorting of strings containing numbers"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def load_frame(frame_path):
    """Load and resize frame for display"""
    img = cv2.imread(str(frame_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def add_highlight_border(img, is_boundary=False):
    """Add colored border to highlight scene boundary frames"""
    if is_boundary:
        # Add a red border (5 pixels wide) for scene boundary frames
        border_color = [255, 0, 0]  # Red for scene boundary
        border_size = 5
    else:
        # Add a thin gray border for regular frames
        border_color = [200, 200, 200]  # Light gray for regular frames
        border_size = 1
        
    h, w = img.shape[:2]
    bordered = cv2.copyMakeBorder(
        img, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, 
        value=border_color
    )
    return bordered

def create_timeline_plot(scene_starts, total_frames):
    """Create a timeline visualization of scene starts"""
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Plot timeline
    ax.plot([0, total_frames], [0, 0], 'k-', alpha=0.3)
    
    # Create colored segments for alternating scenes
    colors = ['#e6f7ff', '#fff0e6']  # Light blue and light orange
    for i in range(len(scene_starts)):
        start = scene_starts[i]
        end = scene_starts[i+1] if i < len(scene_starts)-1 else total_frames
        ax.axvspan(start, end, alpha=0.2, color=colors[i % len(colors)])
    
    # Plot scene starts as thicker red lines
    for start in scene_starts:
        ax.axvline(x=start, color='r', linewidth=2, alpha=0.7)
        ax.text(start, 0.1, f'Frame {start}', rotation=45, color='darkred', fontweight='bold')
    
    ax.set_xlim(0, total_frames)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel('Frame Number')
    ax.set_title('Scene Timeline')
    
    return fig

def display_scene_context(frame_files, scene_boundaries, scene_idx, is_first_scene=False):
    """Display all frames in a scene with highlighted boundaries
    
    Args:
        frame_files: List of all frame files
        scene_boundaries: List of frame indices where scenes start
        scene_idx: Index of the current scene
        is_first_scene: Whether this is the first scene
    """
    # Determine scene start and end frame indices
    scene_start = scene_boundaries[scene_idx]
    scene_end = scene_boundaries[scene_idx + 1] if scene_idx < len(scene_boundaries) - 1 else len(frame_files)
    
    # Get frame numbers from filenames
    frame_numbers = [int(f.stem) for f in frame_files]
    
    # Find the indices of the scene's frames in the frame_files list
    start_idx = frame_numbers.index(scene_start)
    end_idx = min(len(frame_files), frame_numbers.index(scene_start) + (scene_end - scene_start)) if scene_idx < len(scene_boundaries) - 1 else len(frame_files)
    
    # Get all frames for this scene
    scene_frames = frame_files[start_idx:end_idx]
    scene_frame_numbers = frame_numbers[start_idx:end_idx]
    
    # Calculate how many rows we need (8 frames per row maximum)
    frames_per_row = 8
    num_rows = math.ceil(len(scene_frames) / frames_per_row)
    
    st.write(f"Total frames in this scene: {len(scene_frames)}")
    
    # Display frames in rows with 8 columns max
    for row_idx in range(num_rows):
        start_frame_idx = row_idx * frames_per_row
        end_frame_idx = min((row_idx + 1) * frames_per_row, len(scene_frames))
        row_frames = scene_frames[start_frame_idx:end_frame_idx]
        row_frame_numbers = scene_frame_numbers[start_frame_idx:end_frame_idx]
        
        # Create columns for each frame in this row
        cols = st.columns(len(row_frames))
        
        # Display each frame with its number
        for col_idx, (frame_path, frame_num) in enumerate(zip(row_frames, row_frame_numbers)):
            with cols[col_idx]:
                frame = load_frame(frame_path)
                
                # Check if this is a scene boundary frame
                is_boundary = frame_num in scene_boundaries
                frame_with_border = add_highlight_border(frame, is_boundary)
                
                # Add caption with appropriate styling
                st.image(frame_with_border, caption=f"Frame {frame_num}")
                if is_boundary:
                    st.markdown(
                        f'<div style="text-align: center; color: red; font-weight: bold; margin-top: -15px;">⬆ SCENE BOUNDARY ⬆</div>', 
                        unsafe_allow_html=True
                    )

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

def display_scene_summary(scene_starts, total_frames):
    """Display a summary table showing which frames belong to which scenes"""
    scene_data = []
    
    for i, start in enumerate(scene_starts):
        end = scene_starts[i + 1] - 1 if i < len(scene_starts) - 1 else total_frames - 1
        scene_data.append({
            "Scene Number": i + 1,
            "Start Frame": start,
            "End Frame": end,
            "Duration (frames)": end - start + 1
        })
    
    # Convert to DataFrame for easy display
    df = pd.DataFrame(scene_data)
    
    # Display the summary table
    st.dataframe(df)

def save_visualizations(scene_starts, frame_files, selected_dir, output_dir):
    """Save all visualizations to the specified directory"""
    output_path = Path(output_dir)
    print('------------------------------------------')
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_path}")
    
    # Save the timeline plot
    fig = create_timeline_plot(scene_starts, len(frame_files))
    print(f"Saving timeline plot to {output_path}")
    timeline_path = output_path / "timeline.png"
    fig.savefig(str(timeline_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved timeline plot to: {timeline_path}")
    
    # Save scene summary as CSV
    scene_data = []
    for i, start in enumerate(scene_starts):
        end = scene_starts[i + 1] - 1 if i < len(scene_starts) - 1 else len(frame_files) - 1
        scene_data.append({
            "Scene Number": i + 1,
            "Start Frame": start,
            "End Frame": end,
            "Duration (frames)": end - start + 1
        })
    df = pd.DataFrame(scene_data)
    summary_path = output_path / "scene_summary.csv"
    df.to_csv(str(summary_path), index=False)
    logger.info(f"Saved scene summary to: {summary_path}")
    
    # Save frame visualizations for each scene
    frame_numbers = [int(f.stem) for f in frame_files]
    
    # Create a scenes subdirectory
    scenes_dir = output_path / "scenes"
    scenes_dir.mkdir(exist_ok=True)
    logger.info(f"Created scenes directory: {scenes_dir}")
    
    # Save frames around each scene boundary
    for i, start in enumerate(scene_starts):
        scene_dir = scenes_dir / f"scene_{i+1}"
        scene_dir.mkdir(exist_ok=True)
        logger.info(f"Created directory for scene {i+1}: {scene_dir}")
        
        # Determine frames to save
        if i == 0:
            # First scene, save first few frames
            display_range = range(0, min(3, len(frame_files)))
        else:
            # Find index of scene start
            start_idx = frame_numbers.index(start)
            # Get frames around boundary
            start_display = max(0, start_idx - 3)
            end_display = min(len(frame_files), start_idx + 3)
            display_range = range(start_display, end_display)
        
        # Save individual frames
        saved_frames = []
        for idx in display_range:
            frame_path = frame_files[idx]
            frame_num = frame_numbers[idx]
            
            # Load frame
            frame = load_frame(frame_path)
            
            # Add appropriate border
            is_boundary = frame_num == start and i > 0
            frame_with_border = add_highlight_border(frame, is_boundary)
            
            # Save frame with descriptive filename
            boundary_marker = "_BOUNDARY" if is_boundary else ""
            frame_save_path = scene_dir / f"frame_{frame_num}{boundary_marker}.png"
            cv2.imwrite(str(frame_save_path), cv2.cvtColor(frame_with_border, cv2.COLOR_RGB2BGR))
            saved_frames.append(str(frame_save_path))
        
        logger.info(f"Saved {len(saved_frames)} frames for scene {i+1}")
        if len(saved_frames) <= 6:  # Only log individual paths if there aren't too many
            for frame_path in saved_frames:
                logger.info(f"  - {frame_path}")
    
    logger.info(f"All visualizations successfully saved to: {output_path}")
    return output_path

def display_all_frames(frame_files, scene_boundaries):
    """Display all frames in rows of 8, with scene boundaries highlighted
    
    Args:
        frame_files: List of all frame files
        scene_boundaries: List of frame indices where scenes start
    """
    # Get frame numbers from filenames
    frame_numbers = [int(f.stem) for f in frame_files]
    
    # Calculate how many rows we need (8 frames per row maximum)
    frames_per_row = 8
    num_rows = math.ceil(len(frame_files) / frames_per_row)
    
    st.write(f"Total frames: {len(frame_files)}")
    
    # Display frames in rows with 8 columns max
    for row_idx in range(num_rows):
        start_frame_idx = row_idx * frames_per_row
        end_frame_idx = min((row_idx + 1) * frames_per_row, len(frame_files))
        row_frames = frame_files[start_frame_idx:end_frame_idx]
        row_frame_numbers = frame_numbers[start_frame_idx:end_frame_idx]
        
        # Create columns for each frame in this row
        cols = st.columns(len(row_frames))
        
        # Display each frame with its number
        for col_idx, (frame_path, frame_num) in enumerate(zip(row_frames, row_frame_numbers)):
            with cols[col_idx]:
                frame = load_frame(frame_path)
                
                # Check if this is a scene boundary frame
                is_boundary = frame_num in scene_boundaries
                frame_with_border = add_highlight_border(frame, is_boundary)
                
                # Add caption with appropriate styling
                st.image(frame_with_border, caption=f"Frame {frame_num}")
                if is_boundary:
                    st.markdown(
                        f'<div style="text-align: center; color: red; font-weight: bold; margin-top: -15px;">⬆ SCENE BOUNDARY ⬆</div>', 
                        unsafe_allow_html=True
                    )

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
        
        # Show scene summary table
        st.subheader("Scene Summary")
        display_scene_summary(scene_starts, len(frame_files))
        
        # Show all frames in rows of 8, highlighting scene boundaries
        st.subheader("All Frames")
        display_all_frames(frame_files, scene_starts)
        
        # Show all scene starts
        st.subheader("Scene Start Indices")
        st.write(scene_starts)
        
        # Add save visualizations section
        st.header("Save Visualizations")
        save_dir = st.text_input("Output Directory (absolute path)", 
                                value=os.path.join(os.path.dirname(selected_dir), "scene_visualizations"))
        
        if st.button("Save All Visualizations"):
            with st.spinner("Saving visualizations..."):
                output_path = save_visualizations(scene_starts, frame_files, selected_dir, save_dir)
                st.success(f"Visualizations saved to: {output_path}")
                
                # Display log of saved files
                with st.expander("View Save Log", expanded=True):
                    st.info(f"Timeline plot: {output_path}/timeline.png")
                    st.info(f"Scene summary: {output_path}/scene_summary.csv")
                    
                    for i in range(len(scene_starts)):
                        scene_path = f"{output_path}/scenes/scene_{i+1}"
                        st.info(f"Scene {i+1} frames: {scene_path}/")
                
                st.balloons()

if __name__ == "__main__":
    main() 