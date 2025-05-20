#!/usr/bin/env python
# visualize_clusters_both.py - Visualize both original and projected embeddings
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
import torch
from sklearn.preprocessing import LabelEncoder

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contrastiveLearning.pipeline import run_pipeline
from contrastiveLearning.example_supcon import create_sample_data
from contrastiveLearning.dataset import ContrastiveDataset
from contrastiveLearning.training import train_supcon
from contrastiveLearning.model import ProjectionHead

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualize_clusters_both')

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

def plot_pca(embeddings, labels, prototypes, proto_labels=None, title_prefix=""):
    """Plot PCA projection of embeddings and prototypes."""
    # Print shapes for debugging
    logger.info(f"Embeddings shape: {embeddings.shape}, Prototypes shape: {prototypes.shape}")
    
    # PCA to 2D
    try:
        pca = PCA(n_components=2)
        pca.fit(embeddings)
        
        # Transform both embeddings and prototypes
        emb_2d = pca.transform(embeddings)
        proto_2d = pca.transform(prototypes)
        
        if proto_labels is None:
            proto_labels = np.unique(labels)
        
        # Convert labels to numeric if they're strings
        if isinstance(labels[0], str):
            label_encoder = LabelEncoder()
            numeric_labels = label_encoder.fit_transform(labels)
        else:
            numeric_labels = labels
            
        if isinstance(proto_labels[0], str):
            # Try to use the same encoder if the classes match
            try:
                numeric_proto_labels = label_encoder.transform(proto_labels)
            except:
                proto_encoder = LabelEncoder()
                numeric_proto_labels = proto_encoder.fit_transform(proto_labels)
        else:
            numeric_proto_labels = proto_labels
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use a colormap with enough colors
        cmap = plt.cm.get_cmap('tab20', len(np.unique(numeric_labels)))
        
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=numeric_labels, cmap=cmap, s=15, alpha=0.6)
        ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=numeric_proto_labels, cmap=cmap, 
                  marker='*', s=300, edgecolors='k', linewidths=1.2)
        
        # Add labels and legend - use custom legend to ensure all classes appear
        ax.set_title(f"{title_prefix}PCA: Embeddings (dots) vs. Class Prototypes (stars)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        
        # Create a custom legend
        unique_labels = np.unique(numeric_labels)
        legend_elements = []
        for i, label in enumerate(unique_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=cmap(i), markersize=8, 
                                             label=f'Class {labels[np.where(numeric_labels == label)[0][0]]}'))
        
        ax.legend(handles=legend_elements, title="Classes", loc='best')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error in PCA: {e}")
        return None

def plot_umap(embeddings, labels, prototypes, proto_labels=None, title_prefix="", random_state=42):
    """Plot UMAP projection of embeddings and prototypes."""
    import umap
    
    # Convert labels to numeric if they're strings
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
    else:
        numeric_labels = labels
        
    if isinstance(proto_labels[0], str):
        # Try to use the same encoder if the classes match
        try:
            numeric_proto_labels = label_encoder.transform(proto_labels)
        except:
            proto_encoder = LabelEncoder()
            numeric_proto_labels = proto_encoder.fit_transform(proto_labels)
    else:
        numeric_proto_labels = proto_labels
    
    try:
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
        
        # Use a colormap with enough colors
        cmap = plt.cm.get_cmap('tab20', len(np.unique(numeric_labels)))
        
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=numeric_labels, cmap=cmap, s=15, alpha=0.6)
        ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=numeric_proto_labels, cmap=cmap, 
                marker='*', s=300, edgecolors='k', linewidths=1.2)
        
        # Add labels and legend
        ax.set_title(f"{title_prefix}UMAP: Embeddings (dots) vs. Class Prototypes (stars)")
        
        # Create a custom legend
        unique_labels = np.unique(numeric_labels)
        legend_elements = []
        for i, label in enumerate(unique_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=cmap(i), markersize=8, 
                                             label=f'Class {labels[np.where(numeric_labels == label)[0][0]]}'))
        
        ax.legend(handles=legend_elements, title="Classes", loc='best')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error in UMAP: {e}")
        return None

def plot_tsne(embeddings, labels, prototypes, proto_labels=None, title_prefix="", random_state=42):
    """Plot t-SNE projection of embeddings and prototypes."""
    from sklearn.manifold import TSNE
    
    # Convert labels to numeric if they're strings
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
    else:
        numeric_labels = labels
        
    if isinstance(proto_labels[0], str):
        # Try to use the same encoder if the classes match
        try:
            numeric_proto_labels = label_encoder.transform(proto_labels)
        except:
            proto_encoder = LabelEncoder()
            numeric_proto_labels = proto_encoder.fit_transform(proto_labels)
    else:
        numeric_proto_labels = proto_labels
    
    try:
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
        
        # Use a colormap with enough colors
        cmap = plt.cm.get_cmap('tab20', len(np.unique(numeric_labels)))
        
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=numeric_labels, cmap=cmap, s=15, alpha=0.6)
        ax.scatter(proto_2d[:, 0], proto_2d[:, 1], c=numeric_proto_labels, cmap=cmap, 
                marker='*', s=300, edgecolors='k', linewidths=1.2)
        
        # Add labels and legend
        ax.set_title(f"{title_prefix}t-SNE: Embeddings (dots) vs. Class Prototypes (stars)")
        
        # Create a custom legend
        unique_labels = np.unique(numeric_labels)
        legend_elements = []
        for i, label in enumerate(unique_labels):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=cmap(i), markersize=8, 
                                             label=f'Class {labels[np.where(numeric_labels == label)[0][0]]}'))
        
        ax.legend(handles=legend_elements, title="Classes", loc='best')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error in t-SNE: {e}")
        return None

def plot_plotly_3d(embeddings, labels, prototypes, proto_labels=None, title_prefix="", method='pca'):
    """Create an interactive 3D plot using Plotly."""
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Convert labels to numeric if they're strings
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
    else:
        numeric_labels = labels
        
    if isinstance(proto_labels[0], str):
        # Try to use the same encoder if the classes match
        try:
            numeric_proto_labels = label_encoder.transform(proto_labels)
        except:
            proto_encoder = LabelEncoder()
            numeric_proto_labels = proto_encoder.fit_transform(proto_labels)
    else:
        numeric_proto_labels = proto_labels
    
    try:
        # Get 3D projection
        if method.lower() == 'pca':
            pca = PCA(n_components=3)
            pca.fit(embeddings)
            emb_3d = pca.transform(embeddings)
            proto_3d = pca.transform(prototypes)
            title = f"{title_prefix}PCA 3D Visualization"
        
        elif method.lower() == 'umap':
            import umap
            umap_reducer = umap.UMAP(n_components=3, random_state=42)
            umap_reducer.fit(embeddings)
            emb_3d = umap_reducer.transform(embeddings)
            proto_3d = umap_reducer.transform(prototypes)
            title = f"{title_prefix}UMAP 3D Visualization"
        
        elif method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            combined = np.vstack([embeddings, prototypes])
            combined_3d = TSNE(n_components=3, random_state=42).fit_transform(combined)
            emb_3d = combined_3d[:len(embeddings)]
            proto_3d = combined_3d[len(embeddings):]
            title = f"{title_prefix}t-SNE 3D Visualization"
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if proto_labels is None:
            proto_labels = np.unique(labels)
        
        # Create a list of actual class labels for each data point
        if isinstance(labels[0], str):
            str_labels = [f"{l}" for l in labels]
        else:
            str_labels = [f"Class {l}" for l in labels]
            
        # Use a colormap with enough colors
        # Create a custom colorscale with enough colors
        num_classes = len(np.unique(numeric_labels))
        colorscale = px.colors.qualitative.Dark24[:num_classes]
        if num_classes > 24:  # If more than 24 classes, extend with more colors
            colorscale.extend(px.colors.qualitative.Light24[:(num_classes-24)])
        
        # Create a dataframe for better plotting with plotly
        df = pd.DataFrame({
            'x': emb_3d[:, 0],
            'y': emb_3d[:, 1],
            'z': emb_3d[:, 2],
            'label': str_labels,
            'numeric_label': numeric_labels
        })
        
        # Create the 3D scatter plot
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='label',  # Use actual labels instead of numeric
            color_discrete_sequence=colorscale,
            opacity=0.7,
            title=title
        )
        
        # Create a map of prototype labels to colors
        label_to_color = {}
        for i, label in enumerate(np.unique(labels)):
            if i < len(colorscale):
                label_to_color[label] = colorscale[i]
            else:
                label_to_color[label] = colorscale[i % len(colorscale)]
                
        # Get colors for each prototype
        proto_colors = [label_to_color[l] for l in proto_labels]
        
        # Add prototypes as stars
        fig.add_trace(go.Scatter3d(
            x=proto_3d[:, 0],
            y=proto_3d[:, 1],
            z=proto_3d[:, 2],
            mode='markers',
            marker=dict(
                symbol='star',
                size=12,
                color=proto_colors,
                line=dict(color='black', width=1)
            ),
            text=[f"Prototype: {l}" for l in proto_labels],
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
            legend_title_text='Classes'
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error in 3D plotting: {e}")
        return None

def plot_diagnostics(embeddings, labels, prototypes, title_prefix=""):
    """Plot diagnostic metrics for clustering quality."""
    # Convert labels to numeric if they're strings
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        numeric_labels = label_encoder.fit_transform(labels)
    else:
        numeric_labels = labels
    
    try:
        # Calculate silhouette score
        sil_score = silhouette_score(embeddings, numeric_labels)
        
        # Calculate Davies-Bouldin index
        db_score = davies_bouldin_score(embeddings, numeric_labels)
        
        # Calculate pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Sample if there are too many points (for performance)
        max_samples = 5000
        if len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            sample_emb = embeddings[indices]
            sample_labels = numeric_labels[indices]
        else:
            sample_emb = embeddings
            sample_labels = numeric_labels
        
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
        
        ax1.set_title(f'{title_prefix}Clustering Quality Metrics')
        ax1.set_ylim(0, max(1.0, max(values) * 1.2))
        
        # Plot 2: Distance histograms
        ax2.hist(same_class_dists, bins=50, alpha=0.5, label='Same Class')
        ax2.hist(diff_class_dists, bins=50, alpha=0.5, label='Different Classes')
        ax2.set_title(f'{title_prefix}Pairwise Distance Distribution')
        ax2.set_xlabel('Euclidean Distance')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}")
        return None

def project_embeddings(df, vec_dim, epochs=1, temperature=0.1):
    """Project embeddings using the contrastive learning model."""
    # Create the dataset
    ds = ContrastiveDataset(df)
    
    # Train projection head
    model = train_supcon(ds, vec_dim, epochs=epochs, temperature=temperature)
    
    # Project the embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        projected_emb_array = model(ds.vecs.to(device)).cpu().numpy()
    
    return projected_emb_array, model

def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description='Visualize both original and projected embeddings')
    parser.add_argument('--input', type=str, help='Path to input DataFrame pickle with embeddings')
    parser.add_argument('--prototypes', type=str, help='Path to prototypes DataFrame pickle')
    parser.add_argument('--output', type=str, default='contrastive_viz_both', help='Base output directory for plots')
    parser.add_argument('--method', type=str, default='mean', choices=['mean', 'medoid'], 
                        help='Method for prototype generation if prototypes not provided')
    parser.add_argument('--sample', action='store_true', help='Use sample data for demonstration')
    parser.add_argument('--num_classes', type=int, default=5, 
                        help='Number of classes for sample data')
    parser.add_argument('--samples_per_class', type=int, default=50, 
                        help='Samples per class for sample data')
    parser.add_argument('--skip_original', action='store_true', 
                        help='Skip visualization of original embeddings (if they are too high-dimensional)')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of epochs for contrastive learning training')
    parser.add_argument('--temperature', type=float, default=0.1, 
                        help='Temperature parameter for contrastive loss (lower=harder boundaries)')
    parser.add_argument('--no_auto_naming', action='store_true',
                        help='Disable automatic output directory naming based on parameters')
    args = parser.parse_args()
    
    # Create output directory with epochs and temperature in the name
    if args.no_auto_naming:
        output_dir = args.output
    else:
        # Format temperature with appropriate precision
        temp_str = f"{args.temperature:.3f}".rstrip('0').rstrip('.') if args.temperature != int(args.temperature) else str(int(args.temperature))
        output_dir = f"{args.output}_e{args.epochs}_t{temp_str}"
        if args.method != 'mean':  # Add method only if not the default
            output_dir += f"_{args.method}"
    
    logger.info(f"Using output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Get original embeddings and labels
    orig_emb_array = np.vstack(df['embedding'].values)
    orig_labels = df['class'].values
    
    logger.info(f"Original embedding dimension: {orig_emb_array.shape[1]}")
    vec_dim = orig_emb_array.shape[1]  # Embedding dimension
    
    # Project the embeddings
    logger.info(f"Projecting embeddings using contrastive learning (epochs={args.epochs}, temperature={args.temperature})...")
    projected_emb_array, model = project_embeddings(df, vec_dim, epochs=args.epochs, temperature=args.temperature)
    logger.info(f"Projected embedding dimension: {projected_emb_array.shape[1]}")
    
    # Save the projection model
    model_path = os.path.join(output_dir, 'projection_model.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved projection model to {model_path}")
    
    # Load or generate prototypes for the projected embeddings
    if args.prototypes:
        logger.info(f"Loading prototypes from {args.prototypes}...")
        proto_df = pd.read_pickle(args.prototypes)
    else:
        logger.info(f"Generating prototypes using {args.method} method...")
        # Create a temporary DataFrame with the projected embeddings
        proj_df = df.copy()
        proj_df['finetuned_embedding'] = list(projected_emb_array)
        proto_df = run_pipeline(proj_df, method=args.method)
        
        # Save the prototypes
        proto_path = os.path.join(output_dir, 'prototypes.pkl')
        proto_df.to_pickle(proto_path)
        logger.info(f"Saved prototypes to {proto_path}")
    
    logger.info(f"Generated {len(proto_df)} prototypes")
    
    # Get the prototype embeddings
    projected_proto_array = np.vstack(proto_df['representative_embedding'].values)
    projected_proto_labels = proto_df['class'].values
    
    # PART 1: ORIGINAL EMBEDDINGS (if not skipped)
    if not args.skip_original:
        logger.info("Processing original embeddings...")
        
        # Generate original prototypes based on the method
        from sklearn.metrics import pairwise_distances_argmin_min
        
        orig_proto_array = np.zeros((len(np.unique(orig_labels)), orig_emb_array.shape[1]))
        orig_proto_labels = []
        
        for i, cls in enumerate(np.unique(orig_labels)):
            # Get embeddings for this class
            class_mask = orig_labels == cls
            class_embeddings = orig_emb_array[class_mask]
            
            if args.method == 'mean':
                # Use mean as prototype
                orig_proto_array[i] = np.mean(class_embeddings, axis=0)
            else:  # medoid
                # Use medoid as prototype
                centroid = np.mean(class_embeddings, axis=0).reshape(1, -1)
                idx, _ = pairwise_distances_argmin_min(centroid, class_embeddings)
                orig_proto_array[i] = class_embeddings[idx[0]]
            
            orig_proto_labels.append(cls)
        
        orig_proto_labels = np.array(orig_proto_labels)
        
        # Generate and save original embedding plots
        logger.info("Generating plots for original embeddings...")
        
        try:
            # PCA for original embeddings
            orig_pca_fig = plot_pca(orig_emb_array, orig_labels, orig_proto_array, orig_proto_labels, 
                                   title_prefix="Original Embeddings - ")
            if orig_pca_fig:
                orig_pca_path = os.path.join(output_dir, 'original_pca_plot.png')
                orig_pca_fig.savefig(orig_pca_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved original PCA plot to {orig_pca_path}")
            
            # Diagnostics for original embeddings
            orig_diag_fig = plot_diagnostics(orig_emb_array, orig_labels, orig_proto_array, 
                                            title_prefix="Original Embeddings - ")
            if orig_diag_fig:
                orig_diag_path = os.path.join(output_dir, 'original_diagnostics_plot.png')
                orig_diag_fig.savefig(orig_diag_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved original diagnostics to {orig_diag_path}")
                
            # Other plots if dependencies are available
            if deps['umap']:
                logger.info("Generating UMAP plot for original embeddings...")
                orig_umap_fig = plot_umap(orig_emb_array, orig_labels, orig_proto_array, orig_proto_labels,
                                         title_prefix="Original Embeddings - ")
                if orig_umap_fig:
                    orig_umap_path = os.path.join(output_dir, 'original_umap_plot.png')
                    orig_umap_fig.savefig(orig_umap_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved original UMAP plot to {orig_umap_path}")
            
            if deps['tsne']:
                logger.info("Generating t-SNE plot for original embeddings...")
                orig_tsne_fig = plot_tsne(orig_emb_array, orig_labels, orig_proto_array, orig_proto_labels,
                                         title_prefix="Original Embeddings - ")
                if orig_tsne_fig:
                    orig_tsne_path = os.path.join(output_dir, 'original_tsne_plot.png')
                    orig_tsne_fig.savefig(orig_tsne_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved original t-SNE plot to {orig_tsne_path}")
            
            if deps['plotly']:
                logger.info("Generating 3D PCA plot for original embeddings...")
                orig_plotly_fig = plot_plotly_3d(orig_emb_array, orig_labels, orig_proto_array, orig_proto_labels,
                                                title_prefix="Original Embeddings - ", method='pca')
                if orig_plotly_fig:
                    orig_plotly_path = os.path.join(output_dir, 'original_pca_3d.html')
                    orig_plotly_fig.write_html(orig_plotly_path)
                    logger.info(f"Saved original 3D PCA plot to {orig_plotly_path}")
        except Exception as e:
            logger.error(f"Error generating original embedding plots: {e}")
    
    # PART 2: PROJECTED EMBEDDINGS
    logger.info("Processing projected embeddings...")
    
    # Generate and save projected embedding plots
    logger.info("Generating plots for projected embeddings...")
    
    # PCA for projected embeddings
    proj_pca_fig = plot_pca(projected_emb_array, orig_labels, projected_proto_array, projected_proto_labels,
                           title_prefix="Projected Embeddings - ")
    if proj_pca_fig:
        proj_pca_path = os.path.join(output_dir, 'projected_pca_plot.png')
        proj_pca_fig.savefig(proj_pca_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved projected PCA plot to {proj_pca_path}")
    
    # Diagnostics for projected embeddings
    proj_diag_fig = plot_diagnostics(projected_emb_array, orig_labels, projected_proto_array,
                                    title_prefix="Projected Embeddings - ")
    if proj_diag_fig:
        proj_diag_path = os.path.join(output_dir, 'projected_diagnostics_plot.png')
        proj_diag_fig.savefig(proj_diag_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved projected diagnostics to {proj_diag_path}")
    
    if deps['umap']:
        logger.info("Generating UMAP plot for projected embeddings...")
        proj_umap_fig = plot_umap(projected_emb_array, orig_labels, projected_proto_array, projected_proto_labels,
                                 title_prefix="Projected Embeddings - ")
        if proj_umap_fig:
            proj_umap_path = os.path.join(output_dir, 'projected_umap_plot.png')
            proj_umap_fig.savefig(proj_umap_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved projected UMAP plot to {proj_umap_path}")
    
    if deps['tsne']:
        logger.info("Generating t-SNE plot for projected embeddings...")
        proj_tsne_fig = plot_tsne(projected_emb_array, orig_labels, projected_proto_array, projected_proto_labels,
                                 title_prefix="Projected Embeddings - ")
        if proj_tsne_fig:
            proj_tsne_path = os.path.join(output_dir, 'projected_tsne_plot.png')
            proj_tsne_fig.savefig(proj_tsne_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved projected t-SNE plot to {proj_tsne_path}")
    
    if deps['plotly']:
        logger.info("Generating 3D PCA plot for projected embeddings...")
        proj_plotly_fig = plot_plotly_3d(projected_emb_array, orig_labels, projected_proto_array, projected_proto_labels,
                                        title_prefix="Projected Embeddings - ", method='pca')
        if proj_plotly_fig:
            proj_plotly_path = os.path.join(output_dir, 'projected_pca_3d.html')
            proj_plotly_fig.write_html(proj_plotly_path)
            logger.info(f"Saved projected 3D PCA plot to {proj_plotly_path}")
    
    logger.info("Visualization complete!")
    logger.info(f"All plots saved to {output_dir} directory")

if __name__ == "__main__":
    main() 