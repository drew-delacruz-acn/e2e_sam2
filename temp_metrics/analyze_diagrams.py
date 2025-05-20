import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    print("Analyzing diagnostic plots...")
    
    # Directory with the diagnostic plots
    dir_path = "visualization_comparison"
    
    # Dictionary to store metrics for each epoch
    metrics = {}
    
    # Analyze each epoch
    for epoch in [10, 15, 20, 25, 30]:
        diag_file = os.path.join(dir_path, f"e{epoch}_diagnostics.png")
        pca_file = os.path.join(dir_path, f"e{epoch}_pca.png")
        tsne_file = os.path.join(dir_path, f"e{epoch}_tsne.png")
        umap_file = os.path.join(dir_path, f"e{epoch}_umap.png")
        
        print(f"\nEpoch {epoch}:")
        print(f"PCA Visualization: {pca_file}")
        print(f"t-SNE Visualization: {tsne_file}")
        print(f"UMAP Visualization: {umap_file}")
        print(f"Diagnostics: {diag_file}")
        
        # Visual inspection notes - add these manually
        print("Visual observations:")
        print("- Please examine the plots and note cluster separation and quality")
        print("- Note any signs of overfitting such as perfect separation")
        print("- Check distance histograms for overlap between classes")
        
    print("\nPlease examine each plot manually to determine the best epoch.")
    print("Look for a balance between good clustering (high silhouette score) ")
    print("and avoiding overfitting (some overlap in distance histograms)")

if __name__ == "__main__":
    main() 