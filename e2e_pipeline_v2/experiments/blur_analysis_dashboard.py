import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from PIL import Image

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
        
        # Display statistics
        st.write(f"With threshold {threshold:.2f}, {blurry_percent:.1f}% of frames would be classified as blurry")
        
        # Compare with existing classifications
        st.subheader("Comparison with existing methods")
        methods = ["percentile", "stddev", "fixed"]
        for method in methods:
            method_percent = scene_data[f"is_blurry_{method}"].mean() * 100
            st.write(f"Using {method} method: {method_percent:.1f}% of frames classified as blurry")
        
        # Show metric statistics
        st.subheader(f"{metric.capitalize()} Statistics")
        st.write(scene_data[metric].describe())
        
        # Display sample images
        st.subheader("Sample Images")
        
        # Display images near the threshold
        if os.path.exists(base_dir):
            # Get frames just below and above threshold
            scene_data_sorted = scene_data.sort_values(by=metric)
            
            # Find frames closest to threshold (below and above)
            below_threshold = scene_data_sorted[scene_data_sorted[metric] < threshold].tail(3)
            above_threshold = scene_data_sorted[scene_data_sorted[metric] >= threshold].head(3)
            
            # Display images
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Classified as Blurry")
                for _, row in below_threshold.iterrows():
                    img_path = os.path.join(base_dir, row['scene'], row['filename'])
                    try:
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            st.image(img, caption=f"Filename: {row['filename']}, {metric}: {row[metric]:.2f}")
                        else:
                            st.write(f"Image not found: {img_path}")
                    except Exception as e:
                        st.write(f"Error loading image: {e}")
            
            with col2:
                st.write("#### Classified as Clear")
                for _, row in above_threshold.iterrows():
                    img_path = os.path.join(base_dir, row['scene'], row['filename'])
                    try:
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            st.image(img, caption=f"Filename: {row['filename']}, {metric}: {row[metric]:.2f}")
                        else:
                            st.write(f"Image not found: {img_path}")
                    except Exception as e:
                        st.write(f"Error loading image: {e}")
        else:
            st.error(f"Base directory not found: {base_dir}")

if __name__ == "__main__":
    main() 