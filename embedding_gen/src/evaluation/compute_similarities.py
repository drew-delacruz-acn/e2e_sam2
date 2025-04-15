"""
Compute and analyze similarities between image embeddings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from src.models.model_loader import ModelType

def load_embeddings(results_dir: Path, model_type: ModelType) -> Dict[str, np.ndarray]:
    """Load embeddings for a specific model.
    
    Args:
        results_dir: Directory containing embedding files
        model_type: Type of model to load embeddings for
        
    Returns:
        Dictionary mapping image paths to embeddings
    """
    embedding_file = results_dir / f'{model_type}_embeddings.npz'
    return dict(np.load(embedding_file))

def compute_similarity_matrix(embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Compute cosine similarity matrix between embeddings.
    
    Args:
        embeddings: Dictionary mapping image paths to embeddings
        
    Returns:
        Tuple of (similarity matrix, list of image paths)
    """
    # Get sorted list of paths for consistent ordering
    paths = sorted(embeddings.keys())
    
    # Stack embeddings into matrix
    embedding_matrix = np.stack([embeddings[path] for path in paths])
    
    # Compute cosine similarity
    similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)
    
    return similarity_matrix, paths

def plot_similarity_matrix(similarity_matrix: np.ndarray, 
                         paths: List[str],
                         model_type: str,
                         output_path: Path) -> None:
    """Plot and save similarity matrix heatmap.
    
    Args:
        similarity_matrix: Matrix of similarity scores
        paths: List of image paths in order
        model_type: Type of model used
        output_path: Path to save plot to
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        xticklabels=[Path(p).stem for p in paths],
        yticklabels=[Path(p).stem for p in paths],
        cmap='viridis',
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.2f'
    )
    
    # Add title and labels
    plt.title(f'Similarity Matrix - {model_type}')
    plt.xlabel('Images')
    plt.ylabel('Images')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_similarities(similarities: Dict[ModelType, np.ndarray],
                       paths: List[str],
                       output_dir: Path) -> None:
    """Analyze and compare similarities across models.
    
    Args:
        similarities: Dictionary mapping model types to similarity matrices
        paths: List of image paths in order
        output_dir: Directory to save analysis results to
    """
    # Create DataFrame for each model's similarities
    image_names = [Path(p).stem for p in paths]
    dfs = {}
    
    for model_type, sim_matrix in similarities.items():
        df = pd.DataFrame(sim_matrix, columns=image_names, index=image_names)
        dfs[model_type] = df
        
        # Save individual model results
        df.to_csv(output_dir / f'{model_type}_similarities.csv')
        
    # Compare models
    model_comparisons = {}
    model_types = list(similarities.keys())
    
    for i in range(len(model_types)):
        for j in range(i + 1, len(model_types)):
            model1, model2 = model_types[i], model_types[j]
            diff = np.abs(similarities[model1] - similarities[model2])
            
            comparison_df = pd.DataFrame(diff, columns=image_names, index=image_names)
            comparison_df.to_csv(output_dir / f'{model1}_vs_{model2}_diff.csv')
            
            # Plot difference matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                diff,
                xticklabels=image_names,
                yticklabels=image_names,
                cmap='RdYlBu',
                center=0,
                annot=True,
                fmt='.2f'
            )
            plt.title(f'Similarity Difference: {model1} vs {model2}')
            plt.xlabel('Images')
            plt.ylabel('Images')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_dir / f'{model1}_vs_{model2}_diff.png')
            plt.close()

def main():
    """Main function to compute and analyze similarities."""
    # Setup paths
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Model types to analyze
    model_types: List[ModelType] = ['resnet50', 'vit', 'clip']
    
    # Load embeddings and compute similarities for each model
    similarities = {}
    paths = None
    
    for model_type in model_types:
        # Load embeddings
        embeddings = load_embeddings(results_dir, model_type)
        
        # Compute similarity matrix
        similarity_matrix, current_paths = compute_similarity_matrix(embeddings)
        
        # Store results
        similarities[model_type] = similarity_matrix
        if paths is None:
            paths = current_paths
            
        # Plot similarity matrix
        plot_similarity_matrix(
            similarity_matrix,
            paths,
            model_type,
            plots_dir / f'{model_type}_similarities.png'
        )
    
    # Analyze similarities across models
    analyze_similarities(similarities, paths, plots_dir)
    
    print("Analysis complete. Results saved in results/plots/")

if __name__ == '__main__':
    main() 