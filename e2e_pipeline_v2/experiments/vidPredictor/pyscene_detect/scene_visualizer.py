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
            
            # Highlight scene boundary frames
            is_boundary = frame_num == scene_start and not is_first_scene
            frame_with_border = add_highlight_border(frame, is_boundary)
            
            # Add caption with appropriate styling
            if is_boundary:
                st.image(frame_with_border, caption=f"Frame {frame_num}")
                st.markdown(
                    f'<div style="text-align: center; color: red; font-weight: bold; margin-top: -15px;">⬆ SCENE BOUNDARY ⬆</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.image(frame_with_border, caption=f"Frame {frame_num}")

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
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_path}")
    
    # Save the timeline plot
    fig = create_timeline_plot(scene_starts, len(frame_files))
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

def main():
    st.set_page_config(page_title="Scene Detection Visualizer", layout="wide")
    st.title("Scene Detection Visualizer")
    
    # Base directory for all scenes
    base_dir = "/Users/andrewdelacruz/e2e_sam2/gitignore_exception/data/frames"
    
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
        adaptive_threshold = st.sidebar.slider("Adaptive Threshold", 0.01, 1.0, 0.33, 0.01, 
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