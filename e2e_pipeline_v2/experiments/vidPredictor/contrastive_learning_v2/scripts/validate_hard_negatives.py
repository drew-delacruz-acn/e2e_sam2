#!/usr/bin/env python3
"""
Validation script to test hard negative effectiveness.

This script validates that hard negatives are being properly pushed away
from their corresponding class representatives.
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from loss_functions import cosine_similarity_matrix

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate hard negative effectiveness')
    parser.add_argument('--representatives_before', type=str, required=True,
                       help='Path to representatives before training')
    parser.add_argument('--representatives_after', type=str, required=True,
                       help='Path to representatives after training')
    parser.add_argument('--hard_negatives', type=str, required=True,
                       help='Path to hard negatives pickle file')
    parser.add_argument('--output_path', type=str, default='validation_results.csv',
                       help='Output path for validation results')
    
    return parser.parse_args()

def load_representatives(path: str) -> pd.DataFrame:
    """Load representatives from pickle file."""
    print(f"📥 Loading representatives from: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"   Found {len(data)} representatives")
    return data

def load_hard_negatives(path: str) -> Dict[str, List[Dict]]:
    """Load hard negatives from pickle file."""
    print(f"📥 Loading hard negatives from: {path}")
    with open(path, 'rb') as f:
        hard_negatives = pickle.load(f)
    
    total_negatives = sum(len(negs) for negs in hard_negatives.values())
    print(f"   Found {total_negatives} hard negatives across {len(hard_negatives)} classes")
    return hard_negatives

def compute_similarities(representatives_df: pd.DataFrame, 
                        hard_negatives: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Compute similarities between representatives and their hard negatives.
    
    Args:
        representatives_df: DataFrame with representatives
        hard_negatives: Dictionary of hard negatives per class
        
    Returns:
        List of similarity records
    """
    print("🔍 Computing similarities...")
    
    # Create class to embedding mapping
    class_to_embedding = {}
    for _, row in representatives_df.iterrows():
        class_name = row['class']
        embedding = row['finetuned_embedding']
        if isinstance(embedding, np.ndarray):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        elif isinstance(embedding, list):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        class_to_embedding[class_name] = embedding
    
    similarity_records = []
    
    for class_name, negatives in hard_negatives.items():
        if class_name not in class_to_embedding:
            print(f"⚠️  Warning: Class '{class_name}' not found in representatives")
            continue
        
        representative = class_to_embedding[class_name].unsqueeze(0)  # Add batch dim
        
        for neg_info in negatives:
            neg_embedding = neg_info['embedding'].unsqueeze(0)  # Add batch dim
            
            # Compute cosine similarity
            similarity = cosine_similarity_matrix(representative, neg_embedding).item()
            
            similarity_records.append({
                'class': class_name,
                'video': neg_info['video'],
                'frame': neg_info['frame'],
                'predicted_class': neg_info['predicted_class'],
                'original_confidence': neg_info['confidence'],
                'cosine_similarity': similarity
            })
    
    print(f"   Computed {len(similarity_records)} similarity scores")
    return similarity_records

def compare_similarities(before_similarities: List[Dict], 
                        after_similarities: List[Dict]) -> pd.DataFrame:
    """
    Compare similarities before and after training.
    
    Args:
        before_similarities: Similarities before training
        after_similarities: Similarities after training
        
    Returns:
        DataFrame with comparison results
    """
    print("📊 Comparing similarities before and after training...")
    
    # Create lookup for before similarities
    before_lookup = {}
    for record in before_similarities:
        key = (record['class'], record['video'], record['frame'], record['predicted_class'])
        before_lookup[key] = record['cosine_similarity']
    
    # Compare with after similarities
    comparison_records = []
    for record in after_similarities:
        key = (record['class'], record['video'], record['frame'], record['predicted_class'])
        
        if key in before_lookup:
            before_sim = before_lookup[key]
            after_sim = record['cosine_similarity']
            improvement = before_sim - after_sim  # Positive = similarity decreased (good)
            
            comparison_records.append({
                'class': record['class'],
                'video': record['video'],
                'frame': record['frame'],
                'predicted_class': record['predicted_class'],
                'original_confidence': record['original_confidence'],
                'similarity_before': before_sim,
                'similarity_after': after_sim,
                'improvement': improvement,
                'improved': improvement > 0
            })
    
    comparison_df = pd.DataFrame(comparison_records)
    print(f"   Compared {len(comparison_df)} hard negatives")
    
    return comparison_df

def print_validation_summary(comparison_df: pd.DataFrame):
    """Print summary of validation results."""
    print("\n" + "="*60)
    print("🎯 HARD NEGATIVE VALIDATION SUMMARY")
    print("="*60)
    
    total_negatives = len(comparison_df)
    improved_count = comparison_df['improved'].sum()
    improvement_rate = improved_count / total_negatives * 100
    
    avg_improvement = comparison_df['improvement'].mean()
    avg_sim_before = comparison_df['similarity_before'].mean()
    avg_sim_after = comparison_df['similarity_after'].mean()
    
    print(f"Total hard negatives evaluated: {total_negatives}")
    print(f"Hard negatives improved: {improved_count} ({improvement_rate:.1f}%)")
    print(f"Average improvement: {avg_improvement:.4f}")
    print(f"Average similarity before: {avg_sim_before:.4f}")
    print(f"Average similarity after: {avg_sim_after:.4f}")
    print()
    
    # Per-class breakdown
    print("Per-class improvement rates:")
    class_stats = comparison_df.groupby('class').agg({
        'improved': ['count', 'sum'],
        'improvement': 'mean'
    }).round(4)
    
    for class_name in class_stats.index:
        total = class_stats.loc[class_name, ('improved', 'count')]
        improved = class_stats.loc[class_name, ('improved', 'sum')]
        avg_imp = class_stats.loc[class_name, ('improvement', 'mean')]
        rate = improved / total * 100
        print(f"  {class_name}: {improved}/{total} ({rate:.1f}%) - avg: {avg_imp:.4f}")
    
    # Top improvements
    print("\nTop 5 improvements:")
    top_improvements = comparison_df.nlargest(5, 'improvement')
    for i, (_, row) in enumerate(top_improvements.iterrows()):
        print(f"  {i+1}. {row['class']} -> {row['predicted_class']}: "
              f"{row['similarity_before']:.4f} → {row['similarity_after']:.4f} "
              f"(Δ{row['improvement']:.4f})")
    
    # Worst cases (negative improvements)
    worst_cases = comparison_df[comparison_df['improvement'] < 0].nsmallest(5, 'improvement')
    if len(worst_cases) > 0:
        print("\nWorst 5 cases (similarity increased):")
        for i, (_, row) in enumerate(worst_cases.iterrows()):
            print(f"  {i+1}. {row['class']} -> {row['predicted_class']}: "
                  f"{row['similarity_before']:.4f} → {row['similarity_after']:.4f} "
                  f"(Δ{row['improvement']:.4f})")

def main():
    args = parse_args()
    
    print("🚀 Starting hard negative validation...")
    print(f"   Representatives before: {args.representatives_before}")
    print(f"   Representatives after: {args.representatives_after}")
    print(f"   Hard negatives: {args.hard_negatives}")
    print(f"   Output: {args.output_path}")
    print()
    
    # Load data
    representatives_before = load_representatives(args.representatives_before)
    representatives_after = load_representatives(args.representatives_after)
    hard_negatives = load_hard_negatives(args.hard_negatives)
    
    # Compute similarities before and after training
    similarities_before = compute_similarities(representatives_before, hard_negatives)
    similarities_after = compute_similarities(representatives_after, hard_negatives)
    
    # Compare similarities
    comparison_df = compare_similarities(similarities_before, similarities_after)
    
    # Save results
    print(f"💾 Saving validation results to: {args.output_path}")
    comparison_df.to_csv(args.output_path, index=False)
    
    # Print summary
    print_validation_summary(comparison_df)
    
    print("\n🎉 Validation completed!")

if __name__ == "__main__":
    main() 