#!/usr/bin/env python
# visualize_clusters.py - Visualize contrastive learning results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import sys
import logging
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastiveLearning.pipeline import run_pipeline
from contrastiveLearning.example_supcon import create_sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualize_clusters')

def try_import_optional_deps():
    """Try to import optional dependencies and return their availability status."""
    deps = {}
    try:
        import umap
        deps['umap'] = True
    except ImportError:
        deps['umap'] = False
        logger.warning("UMAP not installed. Install with 'pip install umap-learn'")
    
    try:
        from sklearn.manifold import TSNE
        deps['tsne'] = True
    except ImportError:
        deps['tsne'] = False
        logger.warning("t-SNE not installed. It should be available with scikit-learn")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        deps['plotly'] = True
    except ImportError:
        deps['plotly'] = False
        logger.warning("Plotly not installed. Install with 'pip install plotly'")
    
    return deps

def plot_pca(embeddings, labels, prototypes, proto_labels=None):
    """Plot PCA projection of embeddings and prototypes."""
    # PCA to 2D
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    
    # Transform both embeddings and prototypes
    emb_2d = pca.transform(embeddings)
    proto_2d = pca.transform(prototypes)
    
    if proto_labels is None:
        proto_labels = np.unique(labels)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=proto_labels, cmap='tab10', 
               marker='*', s=300, edgecolors='k', linewidths=1.2)
    
    # Add labels and legend
    ax.set_title("PCA: Embeddings (dots) vs. Class Prototypes (stars)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    
    plt.tight_layout()
    return fig

def plot_umap(embeddings, labels, prototypes, proto_labels=None, random_state=42):
    """Plot UMAP projection of embeddings and prototypes."""
    import umap
    
    # UMAP to 2D
    umap_reducer = umap.UMAP(n_components=2, random_state=random_state, init="spectral")
    umap_reducer.fit(embeddings)
    
    # Transform both embeddings and prototypes
    emb_2d = umap_reducer.transform(embeddings)
    proto_2d = umap_reducer.transform(prototypes)
    
    if proto_labels is None:
        proto_labels = np.unique(labels)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=proto_labels, cmap='tab10', 
               marker='*', s=300, edgecolors='k', linewidths=1.2)
    
    # Add labels and legend
    ax.set_title("UMAP: Embeddings (dots) vs. Class Prototypes (stars)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    
    plt.tight_layout()
    return fig

def plot_tsne(embeddings, labels, prototypes, proto_labels=None, random_state=42):
    """Plot t-SNE projection of embeddings and prototypes."""
    from sklearn.manifold import TSNE
    
    # t-SNE to 2D
    tsne = TSNE(n_components=2, random_state=random_state)
    
    # We need to fit and transform in one go for t-SNE
    combined = np.vstack([embeddings, prototypes])
    combined_2d = tsne.fit_transform(combined)
    
    # Split back into embeddings and prototypes
    emb_2d = combined_2d[:len(embeddings)]
    proto_2d = combined_2d[len(embeddings):]
    
    if proto_labels is None:
        proto_labels = np.unique(labels)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab10', s=15, alpha=0.6)
    ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=proto_labels, cmap='tab10', 
               marker='*', s=300, edgecolors='k', linewidths=1.2)
    
    # Add labels and legend
    ax.set_title("t-SNE: Embeddings (dots) vs. Class Prototypes (stars)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    
    plt.tight_layout()
    return fig

def plot_plotly_3d(embeddings, labels, prototypes, proto_labels=None, method='pca'):
    """Create an interactive 3D plot using Plotly."""
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Get 3D projection
    if method.lower() == 'pca':
        pca = PCA(n_components=3)
        pca.fit(embeddings)
        emb_3d = pca.transform(embeddings)
        proto_3d = pca.transform(prototypes)
        title = "PCA 3D Visualization"
    
    elif method.lower() == 'umap':
        import umap
        umap_reducer = umap.UMAP(n_components=3, random_state=42)
        umap_reducer.fit(embeddings)
        emb_3d = umap_reducer.transform(embeddings)
        proto_3d = umap_reducer.transform(prototypes)
        title = "UMAP 3D Visualization"
    
    elif method.lower() == 'tsne':
        from sklearn.manifold import TSNE
        combined = np.vstack([embeddings, prototypes])
        combined_3d = TSNE(n_components=3, random_state=42).fit_transform(combined)
        emb_3d = combined_3d[:len(embeddings)]
        proto_3d = combined_3d[len(embeddings):]
        title = "t-SNE 3D Visualization"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if proto_labels is None:
        proto_labels = np.unique(labels)
    
    # Convert labels to strings for plotly
    str_labels = [f"Class {l}" for l in labels]
    str_proto_labels = [f"Prototype Class {l}" for l in proto_labels]
    
    # Create a dataframe for better plotting with plotly
    df = pd.DataFrame({
        'x': emb_3d[:, 0],
        'y': emb_3d[:, 1],
        'z': emb_3d[:, 2],
        'label': str_labels,
        'numeric_label': labels
    })
    
    # Create the 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='numeric_label', 
        color_continuous_scale=px.colors.qualitative.Set1,
        opacity=0.7,
        title=title
    )
    
    # Add prototypes as stars
    fig.add_trace(go.Scatter3d(
        x=proto_3d[:, 0],
        y=proto_3d[:, 1],
        z=proto_3d[:, 2],
        mode='markers',
        marker=dict(
            symbol='star',
            size=12,
            color=proto_labels,
            colorscale=px.colors.qualitative.Set1,
            line=dict(color='black', width=1)
        ),
        name='Prototypes'
    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        coloraxis_showscale=False
    )
    
    return fig

def plot_diagnostics(embeddings, labels, prototypes):
    """Plot diagnostic metrics for clustering quality."""
    # Calculate silhouette score
    sil_score = silhouette_score(embeddings, labels)
    
    # Calculate Davies-Bouldin index
    db_score = davies_bouldin_score(embeddings, labels)
    
    # Calculate pairwise distances
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Sample if there are too many points (for performance)
    max_samples = 5000
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        sample_emb = embeddings[indices]
        sample_labels = labels[indices]
    else:
        sample_emb = embeddings
        sample_labels = labels
    
    # Compute pairwise distances
    distances = euclidean_distances(sample_emb)
    
    # Separate same-class and different-class distances
    same_class_dists = []
    diff_class_dists = []
    
    for i in range(len(sample_labels)):
        for j in range(i+1, len(sample_labels)):
            if sample_labels[i] == sample_labels[j]:
                same_class_dists.append(distances[i, j])
            else:
                diff_class_dists.append(distances[i, j])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Metric scores
    metrics = ['Silhouette\n(higher is better)', 'Davies-Bouldin\n(lower is better)']
    values = [sil_score, db_score]
    bars = ax1.bar(metrics, values)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    ax1.set_title('Clustering Quality Metrics')
    ax1.set_ylim(0, max(1.0, max(values) * 1.2))
    
    # Plot 2: Distance histograms
    ax2.hist(same_class_dists, bins=50, alpha=0.5, label='Same Class')
    ax2.hist(diff_class_dists, bins=50, alpha=0.5, label='Different Classes')
    ax2.set_title('Pairwise Distance Distribution')
    ax2.set_xlabel('Euclidean Distance')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description='Visualize contrastive learning results')
    parser.add_argument('--input', type=str, help='Path to input DataFrame pickle with embeddings')
    parser.add_argument('--prototypes', type=str, help='Path to prototypes DataFrame pickle')
    parser.add_argument('--output', type=str, default='contrastive_viz', help='Output directory for plots')
    parser.add_argument('--method', type=str, default='mean', choices=['mean', 'medoid'], 
                        help='Method for prototype generation if prototypes not provided')
    parser.add_argument('--sample', action='store_true', help='Use sample data for demonstration')
    parser.add_argument('--num_classes', type=int, default=5, 
                        help='Number of classes for sample data')
    parser.add_argument('--samples_per_class', type=int, default=50, 
                        help='Samples per class for sample data')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check for optional dependencies
    deps = try_import_optional_deps()
    
    # Load or generate data
    if args.sample:
        logger.info("Creating sample data...")
        df = create_sample_data(args.num_classes, args.samples_per_class)
        logger.info(f"Created sample data with {len(df)} embeddings across {args.num_classes} classes")
    elif args.input:
        logger.info(f"Loading embeddings from {args.input}...")
        df = pd.read_pickle(args.input)
        logger.info(f"Loaded {len(df)} embeddings")
    else:
        parser.error("Either --input or --sample must be provided")
    
    # Load or generate prototypes
    if args.prototypes:
        logger.info(f"Loading prototypes from {args.prototypes}...")
        proto_df = pd.read_pickle(args.prototypes)
    else:
        logger.info(f"Generating prototypes using {args.method} method...")
        proto_df = run_pipeline(df, method=args.method)
        
        # Save the prototypes
        proto_path = os.path.join(args.output, 'prototypes.pkl')
        proto_df.to_pickle(proto_path)
        logger.info(f"Saved prototypes to {proto_path}")
    
    logger.info(f"Generated {len(proto_df)} prototypes")
    
    # Prepare data for visualization
    emb_array = np.vstack(df['embedding'].values)
    labels = df['class'].values
    proto_array = np.vstack(proto_df['representative_embedding'].values)
    proto_labels = proto_df['class'].values
    
    # Generate and save plots
    logger.info("Generating PCA plot...")
    pca_fig = plot_pca(emb_array, labels, proto_array, proto_labels)
    pca_path = os.path.join(args.output, 'pca_plot.png')
    pca_fig.savefig(pca_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved PCA plot to {pca_path}")
    
    if deps['umap']:
        logger.info("Generating UMAP plot...")
        umap_fig = plot_umap(emb_array, labels, proto_array, proto_labels)
        umap_path = os.path.join(args.output, 'umap_plot.png')
        umap_fig.savefig(umap_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved UMAP plot to {umap_path}")
    
    if deps['tsne']:
        logger.info("Generating t-SNE plot...")
        tsne_fig = plot_tsne(emb_array, labels, proto_array, proto_labels)
        tsne_path = os.path.join(args.output, 'tsne_plot.png')
        tsne_fig.savefig(tsne_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved t-SNE plot to {tsne_path}")
    
    logger.info("Generating diagnostic plots...")
    diag_fig = plot_diagnostics(emb_array, labels, proto_array)
    diag_path = os.path.join(args.output, 'diagnostics_plot.png')
    diag_fig.savefig(diag_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved diagnostic plots to {diag_path}")
    
    if deps['plotly']:
        logger.info("Generating 3D interactive plots...")
        # PCA 3D
        plotly_pca_fig = plot_plotly_3d(emb_array, labels, proto_array, proto_labels, method='pca')
        plotly_pca_path = os.path.join(args.output, 'pca_3d.html')
        plotly_pca_fig.write_html(plotly_pca_path)
        logger.info(f"Saved 3D PCA plot to {plotly_pca_path}")
        
        # UMAP 3D (if available)
        if deps['umap']:
            plotly_umap_fig = plot_plotly_3d(emb_array, labels, proto_array, proto_labels, method='umap')
            plotly_umap_path = os.path.join(args.output, 'umap_3d.html')
            plotly_umap_fig.write_html(plotly_umap_path)
            logger.info(f"Saved 3D UMAP plot to {plotly_umap_path}")
        
        # t-SNE 3D (if available)
        if deps['tsne']:
            plotly_tsne_fig = plot_plotly_3d(emb_array, labels, proto_array, proto_labels, method='tsne')
            plotly_tsne_path = os.path.join(args.output, 'tsne_3d.html')
            plotly_tsne_fig.write_html(plotly_tsne_path)
            logger.info(f"Saved 3D t-SNE plot to {plotly_tsne_path}")
    
    logger.info("Visualization complete!")
    logger.info(f"All plots saved to {args.output} directory")

if __name__ == "__main__":
    main() 