#!/usr/bin/env python3
"""
Script to display the generated visualization plots.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def display_plots():
    """Display the loss curve and t-SNE plots."""
    results_path = Path("/Users/andrewdelacruz/e2e_sam2/gitignore_exception/first_pass_rep_embeddings")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Load and display loss curve
    loss_img = mpimg.imread(results_path / 'loss_curve.png')
    ax1.imshow(loss_img)
    ax1.set_title('Training Loss Curve', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Load and display t-SNE plot
    tsne_img = mpimg.imread(results_path / 'tsne_plot.png')
    ax2.imshow(tsne_img)
    ax2.set_title('t-SNE Visualization of Embeddings', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Loki/TVA Dataset - Contrastive Learning Results', fontsize=18, fontweight='bold', y=0.98)
    
    # Save combined plot
    output_path = results_path / 'combined_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Combined visualization saved to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    display_plots() 