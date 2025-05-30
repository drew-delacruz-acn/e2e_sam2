"""
Visualization module for contrastive learning experiment.

This module provides functions to:
1. Plot training loss curves
2. Create t-SNE visualizations of embeddings and representatives
3. Generate analysis plots for experiment results
4. Save plots to specified output directories
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def create_plots(history: Dict[str, List[float]], 
                representatives: torch.Tensor,
                embeddings: torch.Tensor, 
                labels: torch.Tensor,
                output_dir: Path,
                class_names: Optional[List[str]] = None) -> None:
    """
    Create and save all visualization plots for the experiment.
    
    Args:
        history: Training history with 'losses' key
        representatives: Learned representatives tensor
        embeddings: Training embeddings tensor
        labels: Training labels tensor
        output_dir: Directory to save plots
        class_names: Optional list of class names for labeling
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Loss curve
    plot_loss_curve(history['losses'], output_dir / 'loss_curve.png')
    
    # 2. t-SNE visualization
    plot_tsne_embeddings(embeddings, labels, representatives, 
                        output_dir / 'tsne_plot.png', class_names)
    
    # 3. Representative analysis (if 2D embeddings)
    if embeddings.shape[1] == 2:
        plot_2d_embeddings(embeddings, labels, representatives,
                          output_dir / 'embeddings_2d.png', class_names)


def plot_loss_curve(losses: List[float], save_path: Path) -> None:
    """
    Plot and save training loss curve.
    
    Args:
        losses: List of loss values per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, color='blue')
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add trend line if enough points
    if len(losses) > 5:
        epochs = np.arange(len(losses))
        z = np.polyfit(epochs, losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), "--", alpha=0.7, color='red', 
                label=f'Trend (slope: {z[0]:.4f})')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tsne_embeddings(embeddings: torch.Tensor, 
                        labels: torch.Tensor,
                        representatives: torch.Tensor,
                        save_path: Path,
                        class_names: Optional[List[str]] = None) -> None:
    """
    Create t-SNE visualization of embeddings and representatives.
    
    Args:
        embeddings: Embeddings tensor
        labels: Labels tensor
        representatives: Representatives tensor
        save_path: Path to save the plot
        class_names: Optional class names for labeling
    """
    # Convert to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    representatives_np = representatives.detach().cpu().numpy()
    
    # Only run t-SNE if embeddings are high-dimensional
    if embeddings_np.shape[1] > 2:
        # Combine embeddings and representatives for t-SNE
        all_data = np.vstack([embeddings_np, representatives_np])
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_data)-1))
        tsne_results = tsne.fit_transform(all_data)
        
        # Split back
        embeddings_2d = tsne_results[:len(embeddings_np)]
        representatives_2d = tsne_results[len(embeddings_np):]
    else:
        # Use original 2D data
        embeddings_2d = embeddings_np
        representatives_2d = representatives_np
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot embeddings by class
    unique_labels = np.unique(labels_np)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        class_name = class_names[label] if class_names else f'Class {label}'
        
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[colors[i]], alpha=0.6, s=50, label=class_name)
    
    # Plot representatives
    for i, label in enumerate(unique_labels):
        class_name = class_names[label] if class_names else f'Class {label}'
        plt.scatter(representatives_2d[i, 0], representatives_2d[i, 1],
                   c=[colors[i]], s=200, marker='*', edgecolors='black',
                   linewidth=2, label=f'{class_name} Rep')
    
    plt.title('t-SNE Visualization: Embeddings and Representatives', 
              fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_2d_embeddings(embeddings: torch.Tensor,
                      labels: torch.Tensor, 
                      representatives: torch.Tensor,
                      save_path: Path,
                      class_names: Optional[List[str]] = None) -> None:
    """
    Plot 2D embeddings directly (for toy data).
    
    Args:
        embeddings: 2D embeddings tensor
        labels: Labels tensor
        representatives: 2D representatives tensor
        save_path: Path to save the plot
        class_names: Optional class names for labeling
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    representatives_np = representatives.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Plot embeddings by class
    unique_labels = np.unique(labels_np)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        class_name = class_names[label] if class_names else f'Class {label}'
        
        plt.scatter(embeddings_np[mask, 0], embeddings_np[mask, 1],
                   c=[colors[i]], alpha=0.7, s=60, label=class_name)
    
    # Plot representatives
    for i, label in enumerate(unique_labels):
        class_name = class_names[label] if class_names else f'Class {label}'
        plt.scatter(representatives_np[i, 0], representatives_np[i, 1],
                   c=[colors[i]], s=300, marker='*', edgecolors='black',
                   linewidth=3, label=f'{class_name} Representative')
    
    plt.title('2D Embeddings and Learned Representatives', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_components(pull_losses: List[float], 
                        push_losses: List[float],
                        save_path: Path) -> None:
    """
    Plot pull and push loss components separately.
    
    Args:
        pull_losses: List of pull loss values
        push_losses: List of push loss values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    epochs = np.arange(len(pull_losses))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, pull_losses, linewidth=2, color='blue', label='Pull Loss')
    plt.title('Pull Loss (Positive Attraction)', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, push_losses, linewidth=2, color='red', label='Push Loss')
    plt.title('Push Loss (Negative Repulsion)', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_similarity_matrix(representatives: torch.Tensor,
                          embeddings: torch.Tensor,
                          labels: torch.Tensor,
                          save_path: Path,
                          class_names: Optional[List[str]] = None) -> None:
    """
    Plot cosine similarity matrix between representatives and embeddings.
    
    Args:
        representatives: Representatives tensor
        embeddings: Embeddings tensor
        labels: Labels tensor
        save_path: Path to save the plot
        class_names: Optional class names for labeling
    """
    from .loss_functions import cosine_similarity_matrix
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity_matrix(representatives, embeddings)
    sim_matrix_np = sim_matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    im = plt.imshow(sim_matrix_np, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Set labels
    if class_names:
        rep_labels = [f'Rep {name}' for name in class_names]
    else:
        rep_labels = [f'Rep {i}' for i in range(len(representatives))]
    
    plt.yticks(range(len(rep_labels)), rep_labels)
    plt.xlabel('Embedding Samples')
    plt.ylabel('Representatives')
    plt.title('Cosine Similarity Matrix: Representatives vs Embeddings', 
              fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 