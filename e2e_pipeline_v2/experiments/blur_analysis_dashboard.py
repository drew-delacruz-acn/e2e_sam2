import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image
import glob

def find_scene_folder(base_dir, scene_name):
    """
    Find the folder matching the scene name in the base directory.
    """
    # First try exact match
    exact_path = os.path.join(base_dir, scene_name)
    if os.path.exists(exact_path) and os.path.isdir(exact_path):
        return exact_path
    
    # If exact match doesn't exist, look for partial matches
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and scene_name in item:
            return item_path
    
    # If no match, return None
    return None

def find_image_file(directory, filename):
    """
    Find an image file in a directory, trying different extensions if needed.
    """
    # Try exact filename first
    exact_path = os.path.join(directory, filename)
    if os.path.exists(exact_path):
        return exact_path
    
    # Try adding .jpg extension if needed
    if not filename.endswith(('.jpg', '.jpeg', '.png')):
        jpg_path = os.path.join(directory, f"{filename}.jpg")
        if os.path.exists(jpg_path):
            return jpg_path
    
    # Try finding by number (e.g., if filename is "42.jpg" or just "42")
    base_filename = os.path.splitext(filename)[0]
    pattern = os.path.join(directory, f"{base_filename}.*")
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]
    
    # If still not found, look for any file containing the number
    if base_filename.isdigit():
        for file in os.listdir(directory):
            if base_filename in os.path.splitext(file)[0] and file.endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(directory, file)
    
    return None

def main():
    st.title("Blur Analysis Visualization Tool")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your blur analysis CSV", type=["csv"])
    
    # Base directory for images
    base_dir = st.text_input("Base directory for image files", 
                            value="/Users/andrewdelacruz/e2e_sam2/e2e_pipeline_v2/experiments/blur_results_images/analysis_20250506_045623")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Get unique scenes
        scenes = df['scene'].unique()
        selected_scene = st.selectbox("Select a scene to analyze", scenes)
        
        # Filter data by selected scene
        scene_data = df[df['scene'] == selected_scene]
        
        # Select metric to visualize
        metric = st.selectbox("Select blur metric", ["laplacian", "tenengrad", "fft"])
        
        # Create two columns for layout
        col_dist, col_images = st.columns([1, 1])
        
        with col_dist:
            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(scene_data[metric], bins=30, kde=True, ax=ax)
            
            # Add threshold slider
            threshold = st.slider(f"Set {metric} threshold", 
                                float(scene_data[metric].min()), 
                                float(scene_data[metric].max()),
                                float(scene_data[metric].mean()))
            
            # Draw threshold line
            ax.axvline(threshold, color='red', linestyle='--')
            
            # Calculate percentage of frames that would be classified as blurry
            if metric == "laplacian":
                blurry_frames = scene_data[metric] < threshold
            else:
                blurry_frames = scene_data[metric] < threshold
                
            blurry_percent = blurry_frames.mean() * 100
                
            ax.set_title(f"{metric.capitalize()} Distribution for {selected_scene}")
            st.pyplot(fig)
            
            # Display basic statistics
            st.write(f"With threshold {threshold:.2f}, {blurry_percent:.1f}% of frames would be classified as blurry")
        
        # Find the scene folder
        scene_folder = find_scene_folder(base_dir, selected_scene)
        
        with col_images:
            if scene_folder is not None:
                st.write(f"Found scene folder: {os.path.basename(scene_folder)}")
                
                # Get frames just below and above threshold
                scene_data_sorted = scene_data.sort_values(by=metric)
                
                # Find frames closest to threshold (below and above)
                below_threshold = scene_data_sorted[scene_data_sorted[metric] < threshold].tail(3)
                above_threshold = scene_data_sorted[scene_data_sorted[metric] >= threshold].head(3)
                
                # Display images in sub-columns
                st.write("#### Sample Images")
                subcol1, subcol2 = st.columns(2)
                
                with subcol1:
                    st.write("**Classified as Blurry**")
                    for _, row in below_threshold.iterrows():
                        img_path = find_image_file(scene_folder, row['filename'])
                        if img_path:
                            try:
                                img = Image.open(img_path)
                                st.image(img, caption=f"{row['filename']}, {metric}: {row[metric]:.2f}")
                            except Exception as e:
                                st.write(f"Error loading image: {e}")
                        else:
                            st.write(f"Image not found: {row['filename']}")
                
                with subcol2:
                    st.write("**Classified as Clear**")
                    for _, row in above_threshold.iterrows():
                        img_path = find_image_file(scene_folder, row['filename'])
                        if img_path:
                            try:
                                img = Image.open(img_path)
                                st.image(img, caption=f"{row['filename']}, {metric}: {row[metric]:.2f}")
                            except Exception as e:
                                st.write(f"Error loading image: {e}")
                        else:
                            st.write(f"Image not found: {row['filename']}")
            else:
                st.error(f"Scene folder not found for: {selected_scene}")
                # For debugging: show what folders are actually available
                available_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
                st.write("Available folders:")
                st.write(available_folders[:10])  # Show first 10 folders to avoid overwhelming the UI

if __name__ == "__main__":
    main() 