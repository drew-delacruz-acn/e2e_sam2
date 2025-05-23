#!/usr/bin/env python3
"""
Test script to demonstrate different initialization methods for contrastive learning.

This script shows how different initialization strategies affect the starting
representatives and provides a comparison of their characteristics.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.trainer import ContrastiveTrainer
from src.loss_functions import cosine_similarity_matrix


def create_toy_data(num_classes=3, samples_per_class=20, embedding_dim=128, seed=42):
    """Create toy data for testing initialization methods."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    embeddings = []
    labels = []
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    for class_idx in range(num_classes):
        # Create class-specific center
        center = torch.randn(embedding_dim) * 2
        
        # Generate samples around the center
        class_embeddings = center.unsqueeze(0) + torch.randn(samples_per_class, embedding_dim) * 0.5
        
        embeddings.append(class_embeddings)
        labels.extend([class_idx] * samples_per_class)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)
    
    return embeddings, labels, class_names


def analyze_representatives(representatives, method_name, embeddings, labels):
    """Analyze characteristics of initialized representatives."""
    print(f"\n📊 Analysis for {method_name}:")
    
    # Basic statistics
    rep_mean = representatives.mean().item()
    rep_std = representatives.std().item()
    rep_min = representatives.min().item()
    rep_max = representatives.max().item()
    
    print(f"  Mean: {rep_mean:.4f}")
    print(f"  Std:  {rep_std:.4f}")
    print(f"  Min:  {rep_min:.4f}")
    print(f"  Max:  {rep_max:.4f}")
    
    # Inter-representative similarities
    rep_similarities = cosine_similarity_matrix(representatives, representatives)
    # Remove diagonal (self-similarities)
    mask = ~torch.eye(len(representatives), dtype=bool)
    inter_similarities = rep_similarities[mask]
    
    print(f"  Inter-rep similarity - Mean: {inter_similarities.mean().item():.4f}")
    print(f"  Inter-rep similarity - Max:  {inter_similarities.max().item():.4f}")
    print(f"  Inter-rep similarity - Min:  {inter_similarities.min().item():.4f}")
    
    # Distance to data
    similarities_to_data = cosine_similarity_matrix(representatives, embeddings)
    print(f"  Similarity to data - Mean: {similarities_to_data.mean().item():.4f}")
    print(f"  Similarity to data - Max:  {similarities_to_data.max().item():.4f}")
    
    return {
        'method': method_name,
        'rep_mean': rep_mean,
        'rep_std': rep_std,
        'inter_sim_mean': inter_similarities.mean().item(),
        'inter_sim_max': inter_similarities.max().item(),
        'data_sim_mean': similarities_to_data.mean().item(),
        'data_sim_max': similarities_to_data.max().item()
    }


def test_initialization_methods():
    """Test all initialization methods and compare their characteristics."""
    print("🧪 Testing Initialization Methods for Contrastive Learning")
    print("=" * 60)
    
    # Create toy data
    embeddings, labels, class_names = create_toy_data()
    print(f"📊 Created toy data: {len(embeddings)} samples, {len(class_names)} classes, {embeddings.shape[1]}D")
    
    # Test each initialization method
    methods = ['class_means', 'random', 'bounded_random', 'perturbed_means']
    results = []
    representatives_dict = {}
    
    config = {'lr': 0.01, 'margin': 0.15, 'lambda_push': 0.25}
    
    for method in methods:
        print(f"\n🎯 Testing {method}...")
        
        # Create trainer and initialize
        trainer = ContrastiveTrainer(config)
        trainer.initialize_representatives(embeddings, labels, init_method=method)
        
        # Store representatives
        representatives_dict[method] = trainer.representatives.detach().clone()
        
        # Analyze
        analysis = analyze_representatives(
            trainer.representatives.detach(), 
            method, 
            embeddings, 
            labels
        )
        results.append(analysis)
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("📋 COMPARISON TABLE")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Create visualization
    create_comparison_plot(representatives_dict, embeddings, labels, class_names)
    
    print(f"\n✅ Comparison plot saved to 'initialization_comparison.png'")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    print("🎯 class_means: Best for stable, predictable training")
    print("🎲 random: Use when you suspect class means are poor starting points")
    print("🎯 bounded_random: Good compromise between exploration and data-awareness")
    print("🔄 perturbed_means: Slight exploration around class means")
    print("\n📈 For your dataset, consider:")
    print("  - If classes are well-separated: class_means or perturbed_means")
    print("  - If classes overlap significantly: random or bounded_random")
    print("  - If training gets stuck in local optima: try random initialization")


def create_comparison_plot(representatives_dict, embeddings, labels, class_names):
    """Create visualization comparing different initialization methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    methods = list(representatives_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    for i, method in enumerate(methods):
        ax = axes[i]
        representatives = representatives_dict[method]
        
        # For visualization, use first 2 dimensions
        if embeddings.shape[1] >= 2:
            emb_2d = embeddings[:, :2]
            rep_2d = representatives[:, :2]
        else:
            # If 1D, create dummy second dimension
            emb_2d = torch.cat([embeddings, torch.zeros_like(embeddings)], dim=1)
            rep_2d = torch.cat([representatives, torch.zeros_like(representatives)], dim=1)
        
        # Plot embeddings by class
        for class_idx in range(len(class_names)):
            mask = labels == class_idx
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                      c=[colors[class_idx]], alpha=0.6, s=30, 
                      label=f'{class_names[class_idx]}' if i == 0 else "")
        
        # Plot representatives
        for class_idx in range(len(class_names)):
            ax.scatter(rep_2d[class_idx, 0], rep_2d[class_idx, 1],
                      c=[colors[class_idx]], s=200, marker='*', 
                      edgecolors='black', linewidth=2)
        
        ax.set_title(f'{method.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Representative Initialization Methods Comparison\n(Stars = Representatives, Dots = Data)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('initialization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test initialization methods')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='Number of classes for toy data')
    parser.add_argument('--samples-per-class', type=int, default=20,
                       help='Samples per class')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Override toy data creation with command line args
    global create_toy_data
    original_create_toy_data = create_toy_data
    
    def create_toy_data_with_args():
        return original_create_toy_data(
            num_classes=args.num_classes,
            samples_per_class=args.samples_per_class,
            embedding_dim=args.embedding_dim,
            seed=args.seed
        )
    
    create_toy_data = create_toy_data_with_args
    
    test_initialization_methods()


if __name__ == '__main__':
    main() 